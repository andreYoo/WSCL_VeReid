import argparse
import os
import os.path as osp
import sys
from collections import OrderedDict
import pprint
import random
import pdb
import time
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import numpy as np
import time
from tensorboardX import SummaryWriter
import _init_paths
from lib.datasets.dataset import DataSet
from lib import models
from lib.trainer import Trainer
from lib.evaluator import Evaluator
from lib.utils.data import transforms as T
from lib.utils.data.preprocessor import Preprocessor, UnsupervisedPreprocessor
from lib.utils.logging import Logger
from lib.utils.serialization import load_checkpoint, save_checkpoint
from lib.utils.config import config, update_config
from lib.utils.netutils import get_optimizer



import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Unsupervised Vehicle ReID via Self-Supervised Quadruplet Metric Learning')
    parser.add_argument('--experiments', dest='cfg_file',
                        help='optional config file',
                        default='experiments/market.yml', type=str)
    parser.add_argument('--gpus', type=str, help='gpus')
    parser.add_argument('--workers', type=int, help='num of dataloader workers')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--tb', type=str, default=True, help='observing optimisation process via tensorboard')
    parser.add_argument('--mlp', type=str, default='SMLC', help='multi label prediction methods')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix to add at the end of the log file e.g,, 20200514_13:40:20_suffix')
    parser.add_argument('--use_dram', type=bool, default=False, help='Use DRAM for extraordinary large-scale dataset i.e., Veri-wild')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    update_config(args.cfg_file)

    if args.mlp:
        config.MLP.TYPE = args.mlp

    if args.gpus:
        config.GPUS = args.gpus
    else:
        config.CUDA = False
    if args.workers:
        config.WORKERS = args.workers
    print('Using config:')
    pprint.pprint(config)

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if config.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    if config.CUDA:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUS
    device = torch.device('cuda' if config.CUDA else 'cpu')
    device_0 = torch.device('cuda:0')
    device_1 = torch.device('cuda:1')

    if args.tb:
        _time_rightnow = time.strftime('%y%m%d_%H:%M:%S')
        log_dic = './logs/'+_time_rightnow
        log_dic = log_dic+ '_' + args.suffix + '_'+str(config.WSCL.L) + '_'+ str(config.WSCL.T)
        if not os.path.exists(log_dic):
            os.makedirs(log_dic)
            writter = SummaryWriter(log_dic)

    # Redirect print to both console and log file
    sys.stdout = Logger(osp.join(config.OUTPUT_DIR, 'log.txt'))
    sys.stdout = Logger(osp.join(config.OUTPUT_DIR, 'log.txt'))

    # Create data loaders
    dataset = DataSet(config.DATASET.ROOT, config.DATASET.DATASET)
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(*config.MODEL.IMAGE_SIZE),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(EPSILON=config.DATASET.RE),
    ])
    test_transformer = T.Compose([
        T.Resize(config.MODEL.IMAGE_SIZE, interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    #Dataset transformation
    train_loader = DataLoader(
        UnsupervisedPreprocessor(dataset.train,
                     root=osp.join(dataset.images_dir, dataset.train_path), transform=train_transformer),
        batch_size=config.TRAIN.BATCH_SIZE, num_workers=config.WORKERS,
        shuffle=config.TRAIN.SHUFFLE, pin_memory=True)

    query_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=osp.join(dataset.images_dir, dataset.query_path), transform=test_transformer),
        batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.images_dir, dataset.gallery_path), transform=test_transformer),
        batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS,
        shuffle=False, pin_memory=True)


    if config.DATASET.DATASET=='veri-wild':
        small_query_loader = DataLoader(
            Preprocessor(dataset.small_query,
                         root=osp.join(dataset.images_dir, dataset.small_query_path), transform=test_transformer),
            batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS,
            shuffle=False, pin_memory=True)

        small_gallery_loader = DataLoader(
            Preprocessor(dataset.small_gallery,
                         root=osp.join(dataset.images_dir, dataset.small_gallery_path), transform=test_transformer),
            batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS,
            shuffle=False, pin_memory=True)
        
        middle_query_loader = DataLoader(
            Preprocessor(dataset.middle_query,
                         root=osp.join(dataset.images_dir, dataset.middle_query_path), transform=test_transformer),
            batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS,
            shuffle=False, pin_memory=True)

        middle_gallery_loader = DataLoader(
            Preprocessor(dataset.middle_gallery,
                         root=osp.join(dataset.images_dir, dataset.middle_gallery_path), transform=test_transformer),
            batch_size=config.TEST.BATCH_SIZE, num_workers=config.WORKERS,
            shuffle=False, pin_memory=True)




    # Create model
    model = models.create(config.MODEL.NAME,pretrained=config.MODEL.PRETRAINED)

    # Memory Network
    num_tgt = len(dataset.train)
    num_cam = int(config.DATASET.CAM_NUM)
    memory = models.create('memory', config.MODEL.FEATURES, num_tgt,num_cam)

    # Load from checkpoint
    if config.TRAIN.RESUME:
        checkpoint = load_checkpoint(config.TRAIN.CHECKPOINT)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> Start epoch {} "
              .format(checkpoint['epoch']))

    # Set model
    model = nn.DataParallel(model).to(device_0)
    memory = memory.to(device_1)

    # Optimizer
    base_param_ids = set(map(id, model.module.base.parameters()))

    base_params_need_for_grad = filter(lambda p: p.requires_grad, model.module.base.parameters())

    new_params = [p for p in model.parameters() if
                  id(p) not in base_param_ids]
    param_groups = [
        {'params': base_params_need_for_grad, 'lr_mult': 0.1},
        {'params': new_params, 'lr_mult': 1.0}]

    optimizer = get_optimizer(config, param_groups)

    # Trainer
    trainer = Trainer(config, model, memory)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def adjust_lr(epoch):
        step_size = config.TRAIN.LR_STEP
        lr = config.TRAIN.LR * (config.TRAIN.LR_FACTOR ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    best_r1 = 0.0
    # Start training


    MEM_INIT=True
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        # lr_scheduler.step()
        #if epoch!=0 and epoch%5==0 :
        #    print('Process is inturrupted for 120 secs to cool VGA')
        #    time.sleep(150)

        adjust_lr(epoch)

        current_lr = get_lr(optimizer)

        print('learning rate is %f'%(current_lr))

        writter.add_scalar("Learning_rate/train",current_lr, epoch)



        trainer.train(epoch, train_loader, optimizer,writter,gi=MEM_INIT)
        GRAPH_INIT=False
        save_checkpoint({
               'state_dict': model.module.state_dict(),
               'state_dict_memory': memory.state_dict(),
               'epoch': epoch + 1,
        }, fpath=osp.join(config.OUTPUT_DIR, 'checkpoint_%d.pth.tar'%(epoch)))

        if epoch > 5:
            print('Test with latest model:')
            evaluator = Evaluator(model)
            r1 = evaluator.evaluate(query_loader, gallery_loader, dataset.query,dataset.gallery,writter=writter,epoch=epoch,output_feature=config.TEST.OUTPUT_FEATURES)
            
            if config.DATASET.DATASET=='veri-wild':
                evaluator.evaluate(middle_query_loader, middle_gallery_loader, dataset.middle_query,dataset.middle_gallery,writter=writter,epoch=epoch,output_feature=config.TEST.OUTPUT_FEATURES,suffix='middle')
                evaluator.evaluate(small_query_loader, small_gallery_loader, dataset.small_query,dataset.small_gallery,writter=writter,epoch=epoch,output_feature=config.TEST.OUTPUT_FEATURES,suffix='small')


            if r1 > best_r1:
                best_r1 = r1
                save_checkpoint({
                    'state_dict': model.module.state_dict(),
                    'state_dict_memory': memory.state_dict(),
                    'epoch': epoch + 1,
                }, fpath=osp.join(config.OUTPUT_DIR, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d} \n'.
                  format(epoch))
            torch.cuda.empty_cache()
        
    # Final test

    print('Test with best model:')
    evaluator = Evaluator(model)
    checkpoint = load_checkpoint(osp.join(config.OUTPUT_DIR, 'checkpoint.pth.tar'))
    print('best model at epoch: {}'.format(checkpoint['epoch']))
    model.module.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(query_loader, gallery_loader, dataset.query,dataset.gallery,writter=None,epoch=None,output_feature=config.TEST.OUTPUT_FEATURES)
    writter.close()

if __name__ == '__main__':
    main()

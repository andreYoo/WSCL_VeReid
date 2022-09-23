from __future__ import print_function

import torch
import torch.nn as nn
import pdb
import numpy as np
import torch.nn.functional as F

device_0 = torch.device('cuda:0')
device_1 = torch.device('cuda:1')

class WSCL_SDA_Loss(nn.Module):
    """A loss functions for the: camera-tracklet-awareness memory-based Semi-supervised contrastive learning"""
    def __init__(self, temperature=0.07, contrast_mode='all',base_temperature=0.07,gz_ratio = 0.001):
        super(WSCL_SDA_Loss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.global_select = 5
        self.grey_zone_rate = gz_ratio
        self.base_temperature = base_temperature

    def forward(self,memory,logits,camids,hard_pos=None,trackids=None,type='local',thr=0.5):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if type=='WSCL':#Contrastive learning loss
            loss = []
            _hard_pos = []
            for i,feat in enumerate(logits):
                _cam_logit = feat[memory.mem_CID==camids[i]] #basement
                _tid_list = memory.mem_TID[memory.mem_CID==camids[i]]
                _hard_sim_index = torch.argmin(_cam_logit[_tid_list==trackids[i]])
                _th_pos = memory.mem[memory.mem_CID==camids[i]][_tid_list==trackids[i]][_hard_sim_index]
                if len(np.shape(_th_pos))>=2: # if there are more than two features are detected
                    _features = memory.mem[memory.mem_CID==camids[i]]
                    _features = _features[_tid_list==trackids[i]]
                    _centre_feature = torch.mean(_features,0)
                    _dist = torch.cdist(torch.unsqueeze(_centre_feature,0),_th_pos)
                    _dist_index = torch.argmax(_dist)
                    _th_pos  = _th_pos[_dist_index]
                _hard_pos.append(_th_pos)

                anchor_dot_contrast = torch.div(_cam_logit,self.temperature)
                logits_max, _ = torch.max(anchor_dot_contrast, dim=0, keepdim=True)
                anchor_dot_contrast = anchor_dot_contrast - logits_max.detach() #Stabilisation

                _positive = anchor_dot_contrast[_tid_list==trackids[i]] #positive
                _num_positive = len(_positive)

                exp_logits = torch.exp(anchor_dot_contrast)
                # mask-out self-contrast cases
                log_prob = _positive - torch.log(exp_logits.sum(0, keepdim=True))

                # compute mean of log-likelihood over positive
                mean_log_prob_pos = (log_prob).sum(0) / (_num_positive)
                mean_log_prob_pos = - (self.temperature / self.base_temperature) * mean_log_prob_pos
                loss.append(mean_log_prob_pos)

            return torch.mean(torch.stack(loss)),torch.stack(_hard_pos)


        if type == 'SDA':#Subdomain Adaptation
            pc_loss = []
            da_loss = []
            _sim_dist_mtx = []
            _tid_list_batch = []
            _hard_pos = []
            for i, feat in enumerate(logits):

                _tid_list = memory.mem_TID[memory.mem_CID == camids[i]]

                _out_domains_feature = memory.mem[memory.mem_CID!=camids[i]]
                _out_domain_tid = memory.mem_TID[memory.mem_CID!=camids[i]]
                _unique_tid = torch.unique(_out_domain_tid,sorted=True)
                _key_prototype = F.normalize(torch.mean(memory.mem[memory.mem_CID == camids[i]][_tid_list == trackids[i]],0),p=2,dim=0)
                _set_prototype = []
                for _tid in _unique_tid:
                    _set_prototype.append(torch.exp(torch.matmul(_key_prototype,torch.mean(_out_domains_feature[_out_domain_tid==_tid],0)))/self.temperature) #0.07 is temp paranmeter
                _sim_dist = torch.div(torch.stack(_set_prototype),torch.sum(torch.stack(_set_prototype)))
                _sim_dist_mtx.append(_sim_dist)
                _tid_list_batch.append(_out_domain_tid)
                _entropy = -1.*torch.mean(torch.sum(torch.mul(_sim_dist,torch.log(_sim_dist+1e-5))))
                pc_loss.append(_entropy)


                # if _entropy < tau:
                #     #when entropy is lower than the threshold, which means there is obvious matching between key and other tracklet.
                #     #this is for the prototypical-entropy guided contrastive learning

                _out_domain_logit = feat[memory.mem_CID != camids[i]]  # basement
                _out_domain_logit = torch.div(_out_domain_logit, self.temperature),
                _out_domain_logit = _out_domain_logit[0]
                logits_max = torch.max(_out_domain_logit)



                _sorted_idx = torch.argsort(_sim_dist,descending=True)
                _sorted_tid = _unique_tid[_sorted_idx]
                _positive_range = int(self.grey_zone_rate*len(_sorted_tid)) #0.1 is range of positive feature
                if _positive_range ==0:
                    _positive_range = 1
                _skip = _positive_range
                _cross_proto_logit = torch.squeeze(_out_domain_logit - logits_max.detach())
                
                if _positive_range <= torch.sum(_out_domain_tid==_sorted_tid[0]):
                    _positive_logit = _cross_proto_logit[_out_domain_tid==_sorted_tid[0]]
                    for _t,_ndx in enumerate(_sorted_idx[1:1+_positive_range]):
                        if _t==0:
                            _negative_logit = _cross_proto_logit[_out_domain_tid==_sorted_tid[_ndx]]
                        else:
                            _negative_logit = torch.cat((_negative_logit,_cross_proto_logit[_out_domain_tid==_sorted_tid[_ndx]]),0)
                else:
                    _pos_scale  = 0
                    _cnt = 0
                    while _pos_scale < _positive_range:
                        _pos_scale += torch.sum(_out_domain_tid==_sorted_tid[_cnt])
                        _cnt +=1
                    for _t,_pdx in enumerate(_sorted_idx[:_cnt]):
                        if _t==0:
                            _positive_logit = _cross_proto_logit[_out_domain_tid==_sorted_tid[_pdx]]
                        else:
                            _positive_logit = torch.cat((_positive_logit,_cross_proto_logit[_out_domain_tid==_sorted_tid[_pdx]]),0)
                    for _t,_ndx in enumerate(_sorted_idx[_cnt:_cnt+_positive_range]):
                        if _t==0:
                            _negative_logit = _cross_proto_logit[_out_domain_tid==_sorted_tid[_ndx]]
                        else:
                            _negative_logit = torch.cat((_negative_logit,_cross_proto_logit[_out_domain_tid==_sorted_tid[_ndx]]),0)

                _num_out_positive = len(_positive_logit)

                _inter_denominator = torch.cat([_positive_logit, _negative_logit])
                exp_logits = torch.exp(_inter_denominator)
                log_prob = _positive_logit - torch.log(exp_logits.sum(0, keepdim=True))
                mean_log_prob_pos = (log_prob).sum(0) / (_num_out_positive)
                mean_log_prob_pos = - (self.temperature / self.base_temperature) * mean_log_prob_pos
                da_loss.append(mean_log_prob_pos)
            return torch.mean(torch.stack(pc_loss))+torch.mean(torch.stack(da_loss)),_sim_dist_mtx,_tid_list_batch
            #return torch.mean(torch.stack(da_loss)),_sim_dist_mtx,_tid_list_batch


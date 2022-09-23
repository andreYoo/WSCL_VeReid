# WSCL-SDA
Source code for the cvpr paper-5482 titled 'Contrastive leanring and domain adaptation using weak supervision for unsupervised vehicle Re-identification'

#Prepare dataset
1. unzip dataset files into './data' directory
2. change the ditectory names to 'bounding_box_train' (training set), 'bounding_box_test' (test set), 'query' (query set).
* You may need reorganise the datasets.

#Train
run train.sh

#Test
run test.sh

#Reproduct experimental results
Since the limitation of upload file size, we could not include the model files, 
So, to reproduct the experimental results, please follows bellow guide.


1. Download learnt models frrom 'https://www.dropbox.com/s/256s0g5xtm4llk0/saved_models.zip?dl=0'
2. unzip
3. Run test.sh



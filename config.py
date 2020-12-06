import os
import os.path as osp
import sys
import numpy as np


class Config:

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    output_dir = osp.join(cur_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    result_dir = osp.join(output_dir, 'result')

    train_dir = osp.join(cur_dir, 'ImageNet-100', 'train')
    test_dir =  osp.join(cur_dir, 'ImageNet-100', 'test')



    ## Set path for model config

    model_name = 'DenseNet'  ## [AlexNet, ResNet18, ResNet50, DenseNet]
    pretrained = 0 # Set 1 to load the pretrained Model for finetune
    load_model_true = 1 # Loading the model that was stopped during training
    load_model_path = osp.join(model_dir, 'no_aug_ResNet18_cvd.pth')
    test_model_path = osp.join(result_dir, 'fine_tunedDenseNet_cvd.pth')


    ## Data preprocess config
    pixel_mean = (0.485, 0.456, 0.406)
    pixel_std = (0.229, 0.224, 0.225)

    ## training config
    epoch = 50
    lr = 0.01
    lr_step_size = 7
    lr_dec_factor = 0.1
    batch_size = 128
    momentum = 0.9

    ## testing config
    test_batch_size = 16


    ## others
    num_workers = 4
    num_thread = 8
    gpu_ids = '0'
    num_gpus = 1


cfg = Config()
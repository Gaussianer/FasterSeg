# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import os
import sys
import numpy as np
from easydict import EasyDict as edict
from datasetHelper import GetTheNumberOfTrainingExamplesFor

C = edict()
config = C
cfg = C

C.seed = 12345

"""please config ROOT_dir and user when u first using"""
C.repo_name = 'FasterSeg'
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]

"""Data Dir"""
C.dataset_path = os.environ["DATASET_PATH"]
C.img_root_folder = C.dataset_path
C.gt_root_folder = C.dataset_path
C.train_source = osp.join(C.dataset_path, "train_mapping_list.txt") 
C.eval_source = osp.join(C.dataset_path, "val_mapping_list.txt") 

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'tools'))

"""Image Config"""
C.num_classes = 3 # TODO
C.background = -1
C.image_mean = np.array([0.485, 0.456, 0.406])
C.image_std = np.array([0.229, 0.224, 0.225])
C.down_sampling = 2 # first down_sampling then crop ......
C.image_height = 160 # this size is after down_sampling
C.image_width = 160*2
C.gt_down_sampling = 8 # model's default output size without final upsampling
C.num_train_imgs = GetTheNumberOfTrainingExamplesFor("train") 
C.num_eval_imgs = GetTheNumberOfTrainingExamplesFor("val") 

""" Settings for network, this would be different for each kind of model"""
C.bn_eps = 1e-5
C.bn_momentum = 0.1

"""Train Config"""
C.lr = 0.01
C.momentum = 0.9
C.weight_decay = 5e-4
C.num_workers = 4
C.train_scale_array = [0.75, 1, 1.25]

"""Eval Config"""
C.eval_stride_rate = 5 / 6
C.eval_scale_array = [1, ]
C.eval_flip = False
C.eval_height = 1024
C.eval_width = 2048


""" Search Config """
C.grad_clip = 5
C.train_portion = 0.5
C.arch_learning_rate = 3e-4
C.arch_weight_decay = 0
C.layers = 16
C.branch = 2

C.pretrain = True # TODO
# C.pretrain = "search-pretrain-256x512_F12.L16_batch3-20200101-012345" # TODO
########################################
C.prun_modes = ['max', 'arch_ratio',]
C.Fch = 12
C.width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.]
C.stem_head_width = [(1, 1), (8./12, 8./12),]
C.FPS_min = [0, 155.]
C.FPS_max = [0, 175.]
if C.pretrain == True:
    C.batch_size = 3
    C.niters_per_epoch = max(C.num_train_imgs // 2 // C.batch_size, 400) # TODO 
    C.lr = 2e-2
    C.latency_weight = [0, 0]
    C.image_height = 256 # this size is after down_sampling
    C.image_width = 256*2
    C.nepochs = 20 # TODO
    C.save = "pretrain-%dx%d_F%d.L%d_batch%d"%(C.image_height, C.image_width, C.Fch, C.layers, C.batch_size)
else:
    C.batch_size = 2
    C.niters_per_epoch = max(C.num_train_imgs // 2 // C.batch_size, 400) # TODO
    C.latency_weight = [0, 1e-2,]
    C.image_height = 224 # this size is after down_sampling
    C.image_width = 224*2
    C.nepochs = 30 # TODO
    C.save = "%dx%d_F%d.L%d_batch%d"%(C.image_height, C.image_width, C.Fch, C.layers, C.batch_size)
########################################
assert len(C.latency_weight) == len(C.stem_head_width) and len(C.stem_head_width) == len(C.FPS_min) and len(C.FPS_min) == len(C.FPS_max)

C.unrolled = False

#!/usr/bin/env python3
# encoding: utf-8
import os
import time
import cv2
cv2.setNumThreads(0)
import torchvision
from PIL import Image
import argparse
import numpy as np

import torch
import torch.multiprocessing as mp

from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_prediction
from engine.tester import Tester
from engine.logger import get_logger
from seg_opr.metric import hist_info, compute_score
from datasets.cityscapes import Cityscapes

#added imports
import csv
import os
import time
from collections import namedtuple

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

################################# Labels for own approach ############################
#--------------------------------------------------------------------------------
# Converts 'something' to boolean. Raises exception for invalid formats
#    Possible True  values: 1, True, "1", "TRue", "yes", "y", "t"
#    Possible False values: 0, False, None, [], {}, "", "0", "faLse", "no", "n", "f", ...    
#--------------------------------------------------------------------------------
def to_bool(value):
    if str(value).lower() in ("yes", "y", "true",  "t", "1"): 
        return True
    if str(value).lower() in ("no",  "n", "false", "f", "0", "none"):
         return False
    raise Exception('Invalid value for boolean conversion: ' + str(value))

#--------------------------------------------------------------------------------
# Read in the customized labels from the csv file 
#--------------------------------------------------------------------------------
def readInLabels():

    # Where to look for the datasets
    if 'DATASET_PATH' in os.environ:
        datasetPath = os.environ['DATASET_PATH']
    else:
        datasetPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')

    #create empty array
    labels=[]

    #check if labelDefinitions.csv exists
    if os.path.exists(datasetPath+'/data_exchange/labelDefinitions.csv'):
        # open labelDefinitions.csv
        with open(datasetPath+'/data_exchange/labelDefinitions.csv', mode='r') as csv_file:
            
            csv_reader = csv.DictReader(csv_file)
            
            for row in csv_reader:
                labels.append(Label(row["name"], int(row["id"]), int(row["trainId"]), row["category"], int(row["catId"]), to_bool(row["hasInstances"]), to_bool(row["ignoreInEval"]), (int(row["color_r"]), int(row["color_g"]), int(row["color_b"]))))
                
        #close labelDefinitions.csv
        csv_file.close()
    else:
        print("labelDefinitions.csv does not exist")

    #return the read label definitions
    return labels

#read in the environment variable CUSTOMDATA
def readEnvVar():
    custom = True
    if 'CUSTOMDATA' in os.environ:
        custom = os.environ['CUSTOMDATA'] 
    else:
        custom = False
    return to_bool(custom)

#map trainID which is not 255 to the corresponding id  
def map_trainID2id(labels):

    mapping=dict()
    for l in labels:
        if l[2]!=255:
            mapping[l[2]]=l[1]
    return mapping

#create the dictonary with the mapping
def getCityscapes_trainID2id():

    isCustomData=readEnvVar()

    if isCustomData==False: #default dict
        cityscapes_trainID2id = {
            0: 7,
            1: 8,
            2: 11,
            3: 12,
            4: 13,
            5: 17,
            6: 19,
            7: 20,
            8: 21,
            9: 22,
            10: 23,
            11: 24,
            12: 25,
            13: 26,
            14: 27,
            15: 28,
            16: 31,
            17: 32,
            18: 33,
            19: 0
            }
    else: # customized dict
        labels = readInLabels()
        cityscapes_trainID2id=map_trainID2id(labels)

    return cityscapes_trainID2id


logger = get_logger()

cityscapes_trainID2id=getCityscapes_trainID2id()

class SegTester(Tester):
    def func_per_iteration(self, data, device, iter=None):
        if self.config is not None: config = self.config
        img = data['data']
        label = data['label']
        name = data['fn']

        if len(config.eval_scale_array) == 1:
            pred = self.whole_eval(img, None, device)
        else:
            pred = self.sliding_eval(img, config.eval_crop_size, config.eval_stride_rate, device)

        if self.show_prediction:
            colors = self.dataset.get_class_colors()
            image = img
            comp_img = show_prediction(colors, config.background, image, pred)
            cv2.imwrite(os.path.join(os.path.realpath('.'), self.config.save, "test", name+".viz.png"), comp_img[:,:,::-1])

        for x in range(pred.shape[0]):
            for y in range(pred.shape[1]):
                pred[x, y] = cityscapes_trainID2id[pred[x, y]]
        cv2.imwrite(os.path.join(os.path.realpath('.'), self.config.save, "test", name+".png"), pred)

    def compute_metric(self, results):
        hist = np.zeros((self.config.num_classes, self.config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iu, mean_IU, mean_IU_no_back, mean_pixel_acc = compute_score(hist, correct, labeled)
        result_line = print_iou(iu, mean_pixel_acc, self.dataset.get_class_names(), True)
        return result_line, mean_IU

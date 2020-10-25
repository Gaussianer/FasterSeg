import os
import cv2
cv2.setNumThreads(0)
import torch
import numpy as np
from random import shuffle

import torch.utils.data as data

#own imports
import os
import csv
from collections import namedtuple


class BaseDataset(data.Dataset):

    #*************calss mebers*************************
    isCustomData = False
    labels=[]
    trans_labels=[]

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

    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(BaseDataset, self).__init__()
        self._split_name = split_name
        self._img_path = setting['img_root']
        self._gt_path = setting['gt_root']
        self._portion = setting['portion'] if 'portion' in setting else None
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self._test_source = setting['test_source'] if 'test_source' in setting else setting['eval_source']
        self._down_sampling = setting['down_sampling']
        print("using downsampling:", self._down_sampling)
        self._file_names = self._get_file_names(split_name)
        print("Found %d images"%len(self._file_names))
        self._file_length = file_length
        self.preprocess = preprocess

        self.readEnvVar() #Read CUSTOMDATA from environment
        self.Labels=self.readInLabels() #read In customized labels
        
        #read In trans_labels
        if self.isCustomData==True:
            self.trans_labels=self.pickTransLabels()
        else:
            self.trans_labels= [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            names = self._construct_new_file_names(self._file_length)[index]
        else:
            names = self._file_names[index]
        img_path = os.path.join(self._img_path, names[0])
        gt_path = os.path.join(self._gt_path, names[1])
        item_name = names[1].split("/")[-1].split(".")[0]

        img, gt = self._fetch_data(img_path, gt_path)
        img = img[:, :, ::-1]
        if self.preprocess is not None:
            img, gt, extra_dict = self.preprocess(img, gt)

        if self._split_name is 'train':
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            if self.preprocess is not None and extra_dict is not None:
                for k, v in extra_dict.items():
                    extra_dict[k] = torch.from_numpy(np.ascontiguousarray(v))
                    if 'label' in k:
                        extra_dict[k] = extra_dict[k].long()
                    if 'img' in k:
                        extra_dict[k] = extra_dict[k].float()

        output_dict = dict(data=img, label=gt, fn=str(item_name),
                           n=len(self._file_names))
        if self.preprocess is not None and extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _fetch_data(self, img_path, gt_path, dtype=None):
        img = self._open_image(img_path, down_sampling=self._down_sampling)
        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=dtype, down_sampling=self._down_sampling)

        return img, gt

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val', 'test']
        source = self._train_source
        if split_name == "val":
            source = self._eval_source
        elif split_name == 'test':
            source = self._test_source

        file_names = []
        with open(source) as f:
            files = f.readlines()
        if self._portion is not None:
            shuffle(files)
            num_files = len(files)
            if self._portion > 0:
                split = int(np.floor(self._portion * num_files))
                files = files[:split]
            elif self._portion < 0:
                split = int(np.floor((1 + self._portion) * num_files))
                files = files[split:]

        for item in files:
            img_name, gt_name = self._process_item_names(item)
            file_names.append([img_name, gt_name])

        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    @staticmethod
    def _process_item_names(item):
        item = item.strip()
        # item = item.split('\t')
        item = item.split(' ')
        img_name = item[0]
        gt_name = item[1]

        return img_name, gt_name

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None, down_sampling=1):
        # cv2: B G R
        # h w c
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)

        if isinstance(down_sampling, int):
            H, W = img.shape[:2]
            if len(img.shape) == 3:
                img = cv2.resize(img, (W // down_sampling, H // down_sampling), interpolation=cv2.INTER_LINEAR)
            else:
                img = cv2.resize(img, (W // down_sampling, H // down_sampling), interpolation=cv2.INTER_NEAREST)
            assert img.shape[0] == H // down_sampling and img.shape[1] == W // down_sampling
        else:
            assert (isinstance(down_sampling, tuple) or isinstance(down_sampling, list)) and len(down_sampling) == 2
            if len(img.shape) == 3:
                img = cv2.resize(img, (down_sampling[1], down_sampling[0]), interpolation=cv2.INTER_LINEAR)
            else:
                img = cv2.resize(img, (down_sampling[1], down_sampling[0]), interpolation=cv2.INTER_NEAREST)
            assert img.shape[0] == down_sampling[0] and img.shape[1] == down_sampling[1]

        return img

    @classmethod
    def s(*args):
        raise NotImplementedError

    @classmethod
    def get_class_names(*args):
        raise NotImplementedError

    #convert string to bool 
    @classmethod
    def to_bool(cls, value):
        if str(value).lower() in ("yes", "y", "true",  "t", "1"): 
            return True
        if str(value).lower() in ("no",  "n", "false", "f", "0", "none"):
            return False
        raise Exception('Invalid value for boolean conversion: ' + str(value))

    #read in EnvironmentVar and store value in class member 
    @classmethod
    def readEnvVar(cls):
        custom = True
        if 'CUSTOMDATA' in os.environ:
            custom = os.environ['CUSTOMDATA'] 
        else:
            custom = False
        cls.isCustomData = cls.to_bool(custom)
    
    #--------------------------------------------------------------------------------
    # Read in the customized labels from the csv file 
    #--------------------------------------------------------------------------------
    @classmethod
    def readInLabels(cls):

        # Where to look for the datasets
        if 'DATASET_PATH' in os.environ:
            datasetPath = os.environ['DATASET_PATH']
        else:
            datasetPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')

        #create empty array
        cls.labels=[]

        #check if labelDefinitions.csv exists
        if os.path.exists(datasetPath+'/data_exchange/labelDefinitions.csv'):
            # open labelDefinitions.csv
            with open(datasetPath+'/data_exchange/labelDefinitions.csv', mode='r') as csv_file:
                
                csv_reader = csv.DictReader(csv_file)
                
                for row in csv_reader:
                    cls.labels.append(cls.Label(row["name"], int(row["id"]), int(row["trainId"]), row["category"], int(row["catId"]), cls.to_bool(row["hasInstances"]), cls.to_bool(row["ignoreInEval"]), (int(row["color_r"]), int(row["color_g"]), int(row["color_b"]))))
                    
            #close labelDefinitions.csv
            csv_file.close()
        else:
            print("labelDefinitions.csv does not exist")
        
        cls.labels.pop(0) #remove "unlabeled" row
    
    # pick the colors from the label tupel and store it as a list
    @classmethod
    def pickColor(cls):
        
        color=[]
        
        for l in cls.labels:
            sub_color=[]
            for c in range(0, 3):
                sub_color.append(l[7][c])
            color.append(sub_color)

        return color

    #pick the names from the label tupel and return it as a list
    @classmethod
    def pickNames(cls):
        
        names=[]
        
        for l in cls.labels:
            names.append(l[0])
        
        return names

    #pick the Ids of the labels and return it as a list 
    @classmethod
    def pickTransLabels(cls):
        translabels=[]
        for l in cls.labels:
            translabels.append(l[1])
        
        return translabels




if __name__ == "__main__":
    data_setting = {'img_root': '',
                    'gt_root': '',
                    'train_source': '',
                    'eval_source': ''}
    bd = BaseDataset(data_setting, 'train', None)
    print(bd.get_class_names())

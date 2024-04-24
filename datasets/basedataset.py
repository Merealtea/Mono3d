"""
Basedataset class for lidar data pre-processing
"""

import os
import math
from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base dataset for all kinds of fusion. Mainly used to assign correct
    index and add noise.

    Parameters
    __________
    params : dict
        The dictionary contains all parameters for training/testing.

    visualize : false
        If set to true, the dataset is used for visualization.

    Attributes
    ----------
    scenario_database : OrderedDict
        A structured dictionary contains all file information.

    len_record : list
        The list to record each scenario's data length. This is used to
        retrieve the correct index during training.

    """

    def __init__(self, data_root, img_prefix, ann_prefix, test_mode, classes):
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.ann_prefix = ann_prefix
        self.test_mode = test_mode
        self.classes = classes

        self.data_infos = []


    def load_infos(self, info_path):
        with open(info_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                img_path = os.path.join(self.data_root, self.img_prefix, line + '.jpg')
                image_info = {"filename" : img_path}
                
                ann_path = os.path.join(self.data_root, self.ann_prefix, line + '.pkl')
                ann_info = {"filename" : ann_path}
                self.data_infos.append([image_info, ann_info])

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        idx = idx % len(self.data_infos)
        return self.get_single_item(idx)

    def get_single_item(self, idx):
        """
        Get the single item .
        """
        raise NotImplementedError
    
    def collate(self, results):
        """
        Collate the results.
        """
        raise NotImplementedError
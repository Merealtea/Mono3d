# Copyright (c) OpenMMLab. All rights reserved.

import os
import numpy as np
from datasets.basedataset import BaseDataset
from datasets.pipelines import Compose
import torch


class CustomMonoDataset(BaseDataset):
    r"""Monocular 3D detection on Custom Dataset.

    This class serves as the API for experiments on the Custom Dataset.

    Please refer to `Custom Dataset <https://www.Custom.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        data_root (str): Path of dataset root.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Camera' in this class. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        eval_version (str, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool, optional): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
        version (str, optional): Dataset version. Defaults to 'v1.0-trainval'.
    """
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    # https://github.com/nutonomy/Custom-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/Custom/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }

    def __init__(self,
                 data_root,
                 ann_prefix,
                 img_prefix,
                 pipeline,
                 load_interval=1,
                 with_velocity=False,
                 modality=None,
                 box_type_3d='Camera',
                 use_valid_flag=False,
                 classes=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk'),
                 **kwargs):
        super(CustomMonoDataset, self).__init__(data_root, img_prefix, ann_prefix, test_mode, classes)
        self.proposal_file = proposal_file
        self.filter_empty_gt = filter_empty_gt
        self.with_class = self.classes is not None
        self.with_velocity = with_velocity

        # load training/test data info
        if not self.test_mode:
            self.info_path = os.path.join(data_root, f'data_train_info.txt')
        else:
            self.info_path = os.path.join(data_root, f'data_val_info.txt')
        
        self.load_infos(self.info_path)
        self.test_mode = False
        # processing pipeline
        self.pipeline = Compose(pipeline)

    def get_single_item(self, idx):
        img_info, ann_info = self.data_infos[idx]

        results = dict(img_info=img_info, ann_info=ann_info, bbox_fields = [])
        return self.pipeline(results)

    def collate(self, results):
        batch_results = {}

        imgs = []
        gt_bboxes_3d = []
        gt_labels_3d = []
        centers2d = []
        depths = []
        timestamps = []
        gt_bboxes = []
        gt_labels = []

        for result in results:
            imgs.append(result['img'].transpose(2, 0, 1))
            gt_bboxes_3d.append(torch.FloatTensor(result['gt_bboxes_3d']))
            gt_labels_3d.append(torch.LongTensor(result['gt_labels_3d']))
            centers2d.append(torch.FloatTensor(result['centers2d']))
            depths.append(torch.FloatTensor(result['depths']))
            timestamps.append(result['timestamp'])
            gt_bboxes.append(torch.FloatTensor(result['gt_bboxes']))
            gt_labels.append(torch.LongTensor(result['gt_labels']))

        batch_results['img'] = torch.FloatTensor(np.stack(imgs))
        batch_results['gt_bboxes'] = gt_bboxes
        batch_results['gt_labels'] = gt_labels 
        batch_results['gt_bboxes_3d'] = gt_bboxes_3d
        batch_results['gt_labels_3d'] = gt_labels_3d
        batch_results['centers2d'] = centers2d
        batch_results['depths'] = depths
        batch_results['timestamp'] = timestamps 
        
        batch_results["img_metas"] = []
        for res in results:
            img_meta = {}
        
            for key in res:
                if key not in batch_results:
                    img_meta[key] = res[key]
            img_meta["timestamp"] = res["timestamp"]
            img_meta['gt_bboxes3d'] = res['gt_bboxes_3d']
            batch_results["img_metas"].append(img_meta)
        if self.test_mode:
            print("gt bboxes: ", batch_results['gt_bboxes'])
        return batch_results




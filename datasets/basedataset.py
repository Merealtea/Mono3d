"""
Basedataset class for lidar data pre-processing
"""

import os
import math
from collections import OrderedDict
from configs.FisheyeParam import CamModel

import torch
import numpy as np
from torch.utils.data import Dataset
import mmcv
from mmcv import Config
import tempfile
from core import bbox3d2result, box3d_multiclass_nms, xywhr2xyxyr
from core.bbox import (Box3DMode, CameraInstance3DBoxes,  LiDARInstance3DBoxes, points_cam2img, get_box_type)


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

    def __init__(self, data_root, img_prefix, ann_prefix, test_mode, classes, load_mode, num_camera, vehicle):
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.ann_prefix = ann_prefix
        self.test_mode = test_mode
        self.classes = classes
        self.load_mode = load_mode
        self.num_cams = num_camera
        self.data_infos = []
        self.vehicle = vehicle
        
        self.cam_models = dict(zip(['left', 'right', 'front', 'back'], 
                                   [CamModel(dir, vehicle, 'torch', 'cpu') for dir in ['left', 'right', 'front', 'back']]))


    def load_infos(self, info_path):
        with open(info_path, 'r') as f:
            for image_idx, line in enumerate(f.readlines()):
                line = line.strip()
                img_path = os.path.join(self.data_root, self.img_prefix, line + '.jpg')
                image_info = {"filename" : img_path,
                              "image_idx" : image_idx,
                              "image_shape" : (1920, 1080, 3),
                              "direction" : img_path.split('/')[-2],
                              "timestamp" : img_path.split('/')[-1].split('.jpg')[0]}
                
                ann_path = os.path.join(self.data_root, self.ann_prefix, line + '.pkl')
                ann_info = {"filename" : ann_path}
                self.data_infos.append({'image' : image_info, 
                                        'annotation' :ann_info})

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        idx = idx % len(self.data_infos)
        return self.get_single_item(idx)

    def get_single_item(self, idx):
        img_info, ann_info = self.data_infos[idx]['image'],\
                            self.data_infos[idx]['annotation']

        results = dict(img_info=img_info, ann_info=ann_info, bbox_fields = [])
        return self.pipeline(results)

    
    def collate(self, results):
        """
        Collate the results.
        """
        raise NotImplementedError
    
    def bbox2result_kitti2d(self,
                            net_outputs,
                            class_names,
                            pklfile_prefix=None,
                            submission_prefix=None):
        """Convert 2D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str): The prefix of pkl file.
            submission_prefix (str): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries have the kitti format
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        det_annos = []
        print('\nConverting prediction to KITTI format')
        for i, bboxes_per_sample in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            anno = dict(
                name=[],
                truncated=[],
                occluded=[],
                alpha=[],
                bbox=[],
                dimensions=[],
                location=[],
                rotation_y=[],
                score=[])

            sample_idx = self.data_infos[i]['image']['image_idx']
            if 'img_bbox' in bboxes_per_sample:
                bboxes_per_sample = bboxes_per_sample['img_bbox']
            bbox_3d, scores_3d, labels3d = \
                bboxes_per_sample['boxes_3d'].tensor, bboxes_per_sample['scores_3d'], bboxes_per_sample['labels_3d']
            num_example = bbox_3d.shape[0]

            for i in range(bbox_3d.shape[0]):
                label = labels3d[i]
                score = scores_3d[i]
                anno['name'].append(class_names[int(label)])
                anno['truncated'].append(0.0)
                anno['occluded'].append(0)
                anno['alpha'].append(0.0)
                anno['bbox'].append(np.zeros(shape=[4], dtype=np.float32))
                # set dimensions (height, width, length) to zero
                anno['dimensions'].append(bbox_3d[i, 3:6])
                # set the 3D translation to (-1000, -1000, -1000)
                anno['location'].append(bbox_3d[i, :3])
                anno['rotation_y'].append(bbox_3d[i, 6])
                anno['score'].append(score)
                num_example += 1

            if num_example == 0:
                annos.append(
                    dict(
                        name=np.array([]),
                        truncated=np.array([]),
                        occluded=np.array([]),
                        alpha=np.array([]),
                        bbox=np.zeros([0, 4]),
                        dimensions=np.zeros([0, 3]),
                        location=np.zeros([0, 3]),
                        rotation_y=np.array([]),
                        score=np.array([]),
                    ))
            else:
                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * num_example, dtype=np.int64)
            det_annos += annos

        if pklfile_prefix is not None:
            # save file in pkl format
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)

        if submission_prefix is not None:
            # save file in submission format
            mmcv.mkdir_or_exist(submission_prefix)
            print(f'Saving KITTI submission to {submission_prefix}')
            for i, anno in enumerate(det_annos):
                sample_idx = self.data_infos[i]['image']['image_idx']
                cur_det_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(cur_det_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions'][::-1]  # lhw -> hwl
                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} '
                            '{:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f}'.format(
                                anno['name'][idx],
                                anno['alpha'][idx],
                                *bbox[idx],  # 4 float
                                *dims[idx],  # 3 float
                                *loc[idx],  # 3 float
                                anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f,
                        )
            print(f'Result is saved to {submission_prefix}')

        return det_annos

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None,
                       data_format='waymo'):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            data_format (str, optional): Output data format.
                Default: 'waymo'. Another supported choice is 'kitti'.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = os.path.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        assert ('waymo' in data_format or 'kitti' in data_format), \
            f'invalid data_format {data_format}'

        if not isinstance(outputs[0], dict):
            result_files = self.bbox2result_kitti2d(
                outputs, self.CLASSES, submission_prefix=submission_prefix)
        elif 'pts_bbox' in outputs[0] or 'img_bbox' in outputs[0] or \
                'img_bbox2d' in outputs[0]:
            result_files = dict()

            for name in outputs[0]:
                results_ = [out[name] for out in outputs]
                # pklfile_prefix_ = pklfile_prefix + name
                if submission_prefix is not None:
                    submission_prefix_ = submission_prefix + name
                else:
                    submission_prefix_ = None
                # do not input prefix into bbox2result_kitti
                # to avoid generate kitti format result pkl
                # to save disk space
                if '2d' in name:
                    result_files_ = self.bbox2result_kitti2d(
                        results_,
                        self.CLASSES,
                        submission_prefix=submission_prefix_)
                else:
                    result_files_ = self.bbox2result_kitti(
                        results_,
                        self.CLASSES,
                        submission_prefix=submission_prefix_)
                result_files[name] = result_files_
        else:
            result_files = self.bbox2result_kitti(
                outputs, self.CLASSES, submission_prefix=submission_prefix)
        if 'waymo' in data_format:
            from ..core.evaluation.waymo_utils.prediction_kitti_to_waymo import \
                KITTI2Waymo  # noqa
            waymo_root = os.path.join(
                self.data_root.split('kitti_format')[0], 'waymo_format')
            if self.split == 'training':
                waymo_tfrecords_dir = os.path.join(waymo_root, 'validation')
                prefix = '1'
            elif self.split == 'testing':
                waymo_tfrecords_dir = os.path.join(waymo_root, 'testing')
                prefix = '2'
            elif self.split == 'testing_cam_only':
                waymo_tfrecords_dir = os.path.join(waymo_root, 'testing')
                prefix = '3'
            else:
                raise ValueError('Not supported split value.')
            save_tmp_dir = tempfile.TemporaryDirectory()
            waymo_results_save_dir = save_tmp_dir.name
            waymo_results_final_path = f'{pklfile_prefix}.bin'
            if 'pts_bbox' in result_files:
                converter = KITTI2Waymo(
                    result_files['pts_bbox'],
                    waymo_tfrecords_dir,
                    waymo_results_save_dir,
                    waymo_results_final_path,
                    prefix,
                    file_client_args=self.file_client_args)
            elif 'img_bbox' in result_files:
                converter = KITTI2Waymo(
                    result_files['img_bbox'],
                    waymo_tfrecords_dir,
                    waymo_results_save_dir,
                    waymo_results_final_path,
                    prefix,
                    file_client_args=self.file_client_args)
            else:
                converter = KITTI2Waymo(
                    result_files,
                    waymo_tfrecords_dir,
                    waymo_results_save_dir,
                    waymo_results_final_path,
                    prefix,
                    file_client_args=self.file_client_args)
            converter.convert()
            save_tmp_dir.cleanup()

        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 ground_truth,
                 metric='waymo',
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'waymo'. Another supported metric is 'kitti'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str, optional): The prefix of pkl files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str, optional): The prefix of submission data.
                If not specified, the submission data will not be generated.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str: float]: results of each evaluation metric
        """
        assert ('waymo' in metric or 'kitti' in metric), \
            f'invalid metric {metric}'
        if 'kitti' in metric:
            result_files, tmp_dir = self.format_results(
                results,
                pklfile_prefix,
                submission_prefix,
                data_format='kitti')
            from core.evaluation import kitti_eval

            # Note: Here we use raw_data_infos for evaluation
            gt_annos = ground_truth

            if isinstance(result_files, dict):
                ap_dict = dict()
                for name, result_files_ in result_files.items():
                    eval_types = ['bev', '3d']
                    ap_result_str, ap_dict_ = kitti_eval(
                        gt_annos,
                        result_files_,
                        self.CLASSES,
                        eval_types=eval_types)
                    for ap_type, ap in ap_dict_.items():
                        ap_dict[f'{name}/{ap_type}'] = float(
                            '{:.4f}'.format(ap))

                    print(f'Results of {name}:\n' + ap_result_str)

            else:
                if metric == 'img_bbox2d':
                    ap_result_str, ap_dict = kitti_eval(
                        gt_annos,
                        result_files,
                        self.CLASSES,
                        eval_types=['bbox'])
                else:
                    ap_result_str, ap_dict = kitti_eval(
                        gt_annos, result_files, self.CLASSES)
                print('\n' + ap_result_str)
        if 'waymo' in metric:
            waymo_root = os.path.join(
                self.data_root.split('kitti_format')[0], 'waymo_format')
            if pklfile_prefix is None:
                eval_tmp_dir = tempfile.TemporaryDirectory()
                pklfile_prefix = os.path.join(eval_tmp_dir.name, 'results')
            else:
                eval_tmp_dir = None
            result_files, tmp_dir = self.format_results(
                results,
                pklfile_prefix,
                submission_prefix,
                data_format='waymo')
            import subprocess
            eval_script = 'mmdet3d/core/evaluation/waymo_utils/' + \
                f'compute_detection_metrics_main {pklfile_prefix}.bin '
            # parse the text to get ap_dict
            ap_dict = {
                'Vehicle/L1 mAP': 0,
                'Vehicle/L1 mAPH': 0,
                'Vehicle/L2 mAP': 0,
                'Vehicle/L2 mAPH': 0,
                'Pedestrian/L1 mAP': 0,
                'Pedestrian/L1 mAPH': 0,
                'Pedestrian/L2 mAP': 0,
                'Pedestrian/L2 mAPH': 0,
                'Sign/L1 mAP': 0,
                'Sign/L1 mAPH': 0,
                'Sign/L2 mAP': 0,
                'Sign/L2 mAPH': 0,
                'Cyclist/L1 mAP': 0,
                'Cyclist/L1 mAPH': 0,
                'Cyclist/L2 mAP': 0,
                'Cyclist/L2 mAPH': 0,
                'Overall/L1 mAP': 0,
                'Overall/L1 mAPH': 0,
                'Overall/L2 mAP': 0,
                'Overall/L2 mAPH': 0
            }
            if self.load_mode == 'lidar_frame':
                if self.modality['use_lidar']:
                    eval_script += f'{waymo_root}/gt.bin'
                else:
                    eval_script += f'{waymo_root}/cam_gt.bin'
            elif self.load_mode == 'cam_mono':
                eval_script += f'{waymo_root}/fov_gt.bin'
            elif self.load_mode == 'cam_frame':
                eval_script += f'{waymo_root}/cam_gt.bin'
            if self.cam_sync:  # use let metric when using cam_sync
                eval_script = eval_script.replace(
                    'compute_detection_metrics_main',
                    'compute_detection_let_metrics_main')
                ap_dict = {
                    'Vehicle mAPL': 0,
                    'Vehicle mAP': 0,
                    'Vehicle mAPH': 0,
                    'Pedestrian mAPL': 0,
                    'Pedestrian mAP': 0,
                    'Pedestrian mAPH': 0,
                    'Sign mAPL': 0,
                    'Sign mAP': 0,
                    'Sign mAPH': 0,
                    'Cyclist mAPL': 0,
                    'Cyclist mAP': 0,
                    'Cyclist mAPH': 0,
                    'Overall mAPL': 0,
                    'Overall mAP': 0,
                    'Overall mAPH': 0
                }
            ret_bytes = subprocess.check_output(eval_script, shell=True)
            ret_texts = ret_bytes.decode('utf-8')

            if not self.cam_sync:
                mAP_splits = ret_texts.split('mAP ')
                mAPH_splits = ret_texts.split('mAPH ')
                for idx, key in enumerate(ap_dict.keys()):
                    split_idx = int(idx / 2) + 1
                    if idx % 2 == 0:  # mAP
                        ap_dict[key] = float(
                            mAP_splits[split_idx].split(']')[0])
                    else:  # mAPH
                        ap_dict[key] = float(
                            mAPH_splits[split_idx].split(']')[0])
                ap_dict['Overall/L1 mAP'] = \
                    (ap_dict['Vehicle/L1 mAP'] +
                     ap_dict['Pedestrian/L1 mAP'] +
                     ap_dict['Cyclist/L1 mAP']) / 3
                ap_dict['Overall/L1 mAPH'] = \
                    (ap_dict['Vehicle/L1 mAPH'] +
                     ap_dict['Pedestrian/L1 mAPH'] +
                     ap_dict['Cyclist/L1 mAPH']) / 3
                ap_dict['Overall/L2 mAP'] = \
                    (ap_dict['Vehicle/L2 mAP'] +
                     ap_dict['Pedestrian/L2 mAP'] +
                     ap_dict['Cyclist/L2 mAP']) / 3
                ap_dict['Overall/L2 mAPH'] = \
                    (ap_dict['Vehicle/L2 mAPH'] +
                     ap_dict['Pedestrian/L2 mAPH'] +
                     ap_dict['Cyclist/L2 mAPH']) / 3
            else:
                mAPL_splits = ret_texts.split('mAPL ')
                mAP_splits = ret_texts.split('mAP ')
                mAPH_splits = ret_texts.split('mAPH ')
                for idx, key in enumerate(ap_dict.keys()):
                    split_idx = int(idx / 3) + 1
                    if idx % 3 == 0:  # mAPL
                        ap_dict[key] = float(
                            mAPL_splits[split_idx].split(']')[0])
                    elif idx % 3 == 1:  # mAP
                        ap_dict[key] = float(
                            mAP_splits[split_idx].split(']')[0])
                    else:  # mAPH
                        ap_dict[key] = float(
                            mAPH_splits[split_idx].split(']')[0])
                ap_dict['Overall mAPL'] = \
                    (ap_dict['Vehicle mAPL'] + ap_dict['Pedestrian mAPL'] +
                     ap_dict['Cyclist mAPL']) / 3
                ap_dict['Overall mAP'] = \
                    (ap_dict['Vehicle mAP'] + ap_dict['Pedestrian mAP'] +
                     ap_dict['Cyclist mAP']) / 3
                ap_dict['Overall mAPH'] = \
                    (ap_dict['Vehicle mAPH'] + ap_dict['Pedestrian mAPH'] +
                     ap_dict['Cyclist mAPH']) / 3
            if eval_tmp_dir is not None:
                eval_tmp_dir.cleanup()

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return ap_dict


    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert results to kitti format for evaluation and test submission.

        Args:
            net_outputs (List[np.ndarray]): list of array storing the
                bbox and score
            class_nanes (List[String]): A list of class names
            pklfile_prefix (str): The prefix of pkl file.
            submission_prefix (str): The prefix of submission file.

        Returns:
            List[dict]: A list of dict have the kitti 3d format
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')

        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']
            image_shape = info['image']['image_shape'][:2]
            timestamp = info['image']['timestamp']
            direction = info['image']['direction']
            
            if self.load_mode == 'cam_frame':
                if idx % self.num_cams == 0:
                    box_dict_per_frame = []
                    cam0_idx = idx

            box_dict = self.convert_valid_bboxes(pred_dicts, info)

            if self.load_mode == 'cam_frame':
                box_dict_per_frame.append(box_dict)
                if (idx + 1) % self.num_cams != 0:
                    continue
                box_dict = self.merge_multi_view_boxes(
                    box_dict_per_frame, self.data_infos[cam0_idx])

            if len(box_dict['box3d_camera']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']

                anno = {
                    'name': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'bbox': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': []
                }

                for box, box_lidar, bbox, score, label in zip(
                        box_preds, box_preds_lidar, box_2d_preds, scores,
                        label_preds):
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    if self.load_mode == 'lidar_frame':
                        anno['alpha'].append(
                            -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                    elif self.load_mode in ['cam_mono', 'cam_frame']:
                        # alpha is meaningless here for cam_frame
                        # because here we have merged boxes to cam0
                        # we also do not evaluate alpha for waymo
                        anno['alpha'].append(-np.arctan2(box[0], box[2]) + box[6])

                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

                if submission_prefix is not None:
                    curr_file = f'{submission_prefix}/{sample_idx:07d}.txt'
                    with open(curr_file, 'w') as f:
                        bbox = anno['bbox']
                        loc = anno['location']
                        dims = anno['dimensions']  # lhw -> hwl

                        for idx in range(len(bbox)):
                            print(
                                '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                                '{:.4f} {:.4f} {:.4f} '
                                '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.
                                format(anno['name'][idx], anno['alpha'][idx],
                                       bbox[idx][0], bbox[idx][1],
                                       bbox[idx][2], bbox[idx][3],
                                       dims[idx][1], dims[idx][2],
                                       dims[idx][0], loc[idx][0], loc[idx][1],
                                       loc[idx][2], anno['rotation_y'][idx],
                                       anno['score'][idx]),
                                file=f)
            else:
                annos.append({
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                })

            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * len(annos[-1]['score']), dtype=np.int64)
            annos[-1]['timestamp'] = np.array(timestamp)
            annos[-1]['direction'] = np.array(direction)
            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def convert_valid_bboxes(self, box_dict, info):
        """Convert the boxes into valid format.

        Args:
            box_dict (dict): Bounding boxes to be converted.

                - boxes_3d (:obj:``LiDARInstance3DBoxes``): 3D bounding boxes.
                - scores_3d (np.ndarray): Scores of predicted boxes.
                - labels_3d (np.ndarray): Class labels of predicted boxes.
            info (dict): Dataset information dictionary.

        Returns:
            dict: Valid boxes after conversion.

                - bbox (np.ndarray): 2D bounding boxes (in camera 0).
                - box3d_camera (np.ndarray): 3D boxes in camera coordinates.
                - box3d_lidar (np.ndarray): 3D boxes in lidar coordinates.
                - scores (np.ndarray): Scores of predicted boxes.
                - label_preds (np.ndarray): Class labels of predicted boxes.
                - sample_idx (np.ndarray): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['image']['image_idx']
        direction = info['image']['direction']
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0]),
                sample_idx=sample_idx)

        if self.load_mode == 'lidar_frame':
            box_preds_lidar = box_preds
            box_preds_tmp = box_preds.tensor

            box_preds_camera = self.cam_models[direction].world2cam(box_preds_tmp[:, :3].T).T
            box_preds_camera = torch.cat([box_preds_camera, box_preds_tmp[:, 3:]], dim=1)
            box_preds_camera = CameraInstance3DBoxes(box_preds_camera, box_dim = box_preds_camera.shape[-1])
        elif self.load_mode in ['cam_frame', 'cam_mono']:
            box_preds_camera = box_preds
            box_preds_tmp = box_preds.tensor
            box_preds_lidar = self.cam_models[direction].cam2world(box_preds_tmp[:, :3].T).T
            box_preds_lidar = torch.cat([box_preds_lidar, box_preds_tmp[:, 3:]], dim=1)
            box_preds_lidar = LiDARInstance3DBoxes(box_preds_lidar, box_dim = box_preds_lidar.shape[-1])

        box_corners = box_preds_camera.corners
        box_corners_shape = (box_corners.shape[0], box_corners.shape[1], 2)
        box_corners = box_corners.reshape(-1, 3).T
        box_corners_in_image =  self.cam_models[direction].cam2image(box_corners, False).T
        box_corners_in_image = box_corners_in_image.reshape(box_corners_shape)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds
        if self.load_mode == 'lidar_frame':
            limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
            valid_pcd_inds = ((box_preds.center > limit_range[:3]) &
                              (box_preds.center < limit_range[3:]))
            valid_inds = valid_pcd_inds.all(-1)
        elif self.load_mode in ['cam_frame', 'cam_mono']:
            img_shape = info['image']['image_shape']
            # check box_preds_camera
            # if the projected 2d bbox has intersection
            # with the image, we keep it, otherwise, we omit it.
            image_shape = box_preds.tensor.new_tensor(img_shape)
            valid_cam_inds = ((box_2d_preds[:, 0] < image_shape[1]) &
                              (box_2d_preds[:, 1] < image_shape[0]) &
                              (box_2d_preds[:, 2] > 0) &
                              (box_2d_preds[:, 3] > 0))
            valid_inds = valid_cam_inds

        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                box3d_lidar=box_preds_lidar[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx,
            )
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0]),
                sample_idx=sample_idx,
            )

    def merge_multi_view_boxes(self, box_dict_per_frame, cam0_info):
        box_dict = dict()
        # convert list[dict] to dict[list]
        for key in box_dict_per_frame[0].keys():
            box_dict[key] = list()
            for cam_idx in range(self.num_cams):
                box_dict[key].append(box_dict_per_frame[cam_idx][key])
        # merge each elements
        box_dict['sample_idx'] = cam0_info['image']['image_idx']
        for key in ['bbox', 'box3d_lidar', 'scores', 'label_preds']:
            box_dict[key] = np.concatenate(box_dict[key])

        # apply nms to box3d_lidar (box3d_camera are in different systems)
        # TODO: move this global setting into config
        nms_cfg = dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=500,
            nms_thr=0.05,
            score_thr=0.001,
            min_bbox_size=0,
            max_per_frame=100)
        nms_cfg = Config(nms_cfg)
        lidar_boxes3d = LiDARInstance3DBoxes(
            torch.from_numpy(box_dict['box3d_lidar']).cuda())
        scores = torch.from_numpy(box_dict['scores']).cuda()
        labels = torch.from_numpy(box_dict['label_preds']).long().cuda()
        nms_scores = scores.new_zeros(scores.shape[0], len(self.CLASSES) + 1)
        indices = labels.new_tensor(list(range(scores.shape[0])))
        nms_scores[indices, labels] = scores
        lidar_boxes3d_for_nms = xywhr2xyxyr(lidar_boxes3d.bev)
        boxes3d = lidar_boxes3d.tensor
        # generate attr scores from attr labels
        boxes3d, scores, labels = box3d_multiclass_nms(
            boxes3d, lidar_boxes3d_for_nms, nms_scores, nms_cfg.score_thr,
            nms_cfg.max_per_frame, nms_cfg)
        lidar_boxes3d = LiDARInstance3DBoxes(boxes3d)
        det = bbox3d2result(lidar_boxes3d, scores, labels)
        box_preds_lidar = det['boxes_3d']
        scores = det['scores_3d']
        labels = det['labels_3d']
        # box_preds_camera is in the cam0 system
        direction = cam0_info['image']['direction']

        box_preds = box_preds_lidar.tensor.numpy()
        box_preds_camera = self.camera_models[direction].world2cam(box_preds[:, :3])
        box_preds_camera = np.concatenate([box_preds_camera, box_preds[:, 3:]], axis=1)
        box_preds_camera = CameraInstance3DBoxes(box_preds_camera, box_dim=box_preds_camera.shape[-1])

        # box_preds_camera = box_preds_lidar.convert_to(
        #     Box3DMode.CAM, rect @ Trv2c, correct_yaw=True)
        # Note: bbox is meaningless in final evaluation, set to 0
        merged_box_dict = dict(
            bbox=np.zeros([box_preds_lidar.tensor.shape[0], 4]),
            box3d_camera=box_preds_camera.tensor.numpy(),
            box3d_lidar=box_preds_lidar.tensor.numpy(),
            scores=scores.numpy(),
            label_preds=labels.numpy(),
            sample_idx=box_dict['sample_idx'],
        )
        return merged_box_dict

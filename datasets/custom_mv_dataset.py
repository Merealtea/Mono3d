# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import tempfile
from .basedataset import BaseDataset
from datasets.pipelines import Compose
import mmcv
import cv2
import numpy as np
import torch
from mmcv.utils import print_log
from core import bbox3d2result, box3d_multiclass_nms, xywhr2xyxyr
from core.bbox import (Box3DMode, CameraInstance3DBoxes,
                         LiDARInstance3DBoxes, points_cam2img, get_box_type)

class CustomMV3DDataset(BaseDataset):
    """Waymo Dataset.

    This class serves as the API for experiments on the Waymo Dataset.

    Please refer to `<https://waymo.com/open/download/>`_for data downloading.
    It is recommended to symlink the dataset root to $MMDETECTION3D/data and
    organize them as the doc shows.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': box in LiDAR coordinates
            - 'Depth': box in depth coordinates, usually for indoor dataset
            - 'Camera': box in camera coordinates
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list(float), optional): The range of point cloud used
            to filter invalid predicted boxes.
            Default: [-85, -85, -5, 85, 85, 5].
        load_mode: Loading mode for different settings. Supported choices
            include: 'lidar_frame', 'cam_frame', 'cam_mono'. 'lidar_frame'
            supports loading frame-based data for 3D detection based on LiDAR
            coordinate system; 'cam_mono' means only loading image-based
            front-view data; 'cam_frame' means loading multi-view images based
            on perspective views. Defaults to 'lidar_frame'.
    """

    CLASSES = ('Pedestrian', )

    def __init__(self,
                 data_root,
                 img_prefix,
                 ann_prefix,
                 pipeline=None,
                 classes=None,
                 test_mode=False,
                 load_interval=1,
                 cam_sync=False,
                 multiview_indices=[
                     'image_0', 'image_1', 'image_2', 'image_3', 'image_4'
                 ],
                 max_sweeps=0,
                 load_mode='lidar_frame',
                 num_camera = 4,
                 box_type_3d = "Lidar",
                 num_ref_frames=0,
                 vehicle = None,
                 **kwargs):
        super().__init__(data_root, img_prefix, ann_prefix, test_mode, classes, load_mode, num_camera, vehicle)
        self.load_interval = load_interval
        # set loading mode for different task settings
        
        self.cam_sync = cam_sync
        # construct self.cat_ids for vision-only anns parsing
        self.cat_ids = range(len(self.CLASSES))
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.bbox_code_size = 7
        self.multiview_indices = tuple(multiview_indices)
        self.max_sweeps = max_sweeps # The max history frame concerned in the model
        self.num_ref_frames = num_ref_frames
        self.box_type_3d = box_type_3d

        # load training/test data info
        if not self.test_mode:
            self.info_path = os.path.join(data_root, f'mv_data_train_info.txt')
        else:
            self.info_path = os.path.join(data_root, f'mv_data_val_info.txt')
        self.pipeline = Compose(pipeline)
        self.load_infos(self.info_path)

    def load_infos(self, info_path):
        with open(info_path, 'r') as f:
            for image_idx, timestamp in enumerate(f.readlines()):
                timestamp = timestamp.strip()
                img_path = os.path.join(self.data_root, self.img_prefix)
                image_info = {
                    "img_filename" : [os.path.join(img_path, cam_dir, f"{timestamp}.jpg") for cam_dir in self.multiview_indices],
                    "image_idx" : image_idx,
                    "image_shape" : (1920, 1280, 3),
                    "timestamp" : timestamp,
                    "direction" : self.multiview_indices
                } 
                
                ann_path = os.path.join(self.data_root, self.ann_prefix, "lidar", timestamp + '.pkl')
                ann_info = {"filename" : ann_path}
                self.data_infos.append({"image" : image_info, 
                                        "annotation" : ann_info})

    def collate(self, results):
        batch_results = {}

        imgs = []
        gt_bboxes_3d = []
        gt_labels_3d = []
        timestamps = []
        gt_labels = []

        for result in results:
            img = np.stack(result['img'], axis=0)
            img_shape = result["pad_shape"][0]
            imgs.append(img.transpose(0, 3, 1, 2).reshape(len(self.multiview_indices), 3, img_shape[0], img_shape[1]))
            
            assert len(result['gt_bboxes_3d']) == len(result['gt_labels_3d'])
            gt_bboxes_3d.append(torch.FloatTensor(result['gt_bboxes_3d']))
            gt_labels_3d.append(torch.LongTensor(result['gt_labels_3d']))
            gt_labels.append(torch.LongTensor(result['gt_labels']))
            timestamps.append(result['img_info']['timestamp'])

        batch_results['img'] = torch.FloatTensor(np.stack(imgs))
        batch_results['gt_labels'] = gt_labels 
        batch_results['gt_bboxes_3d'] = gt_bboxes_3d
        batch_results['gt_labels_3d'] = gt_labels_3d
        batch_results['timestamp'] = timestamps 
        
        batch_results["img_metas"] = []
        for res in results:
            img_meta = {}
            for key in res:
                if key not in batch_results:
                    img_meta[key] = res[key]
            img_meta['num_ref_frames'] = self.num_ref_frames
            img_meta['direction'] = self.multiview_indices
            img_meta["timestamp"] = res['img_info']["timestamp"]
            img_meta['gt_bboxes3d'] = res['gt_bboxes_3d']
            img_meta['box_type_3d'] = get_box_type(self.box_type_3d)[0]
            batch_results["img_metas"].append(img_meta)
        return batch_results

    def _get_pts_filename(self, idx):
        pts_filename = os.path.join(self.root_split, self.pts_prefix,
                                f'{idx:07d}.bin')
        return pts_filename

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Standard input_dict consists of the
                data information.

                - sample_idx (str): sample index
                - pts_filename (str): filename of point clouds
                - img_prefix (str): prefix of image files
                - img_info (dict): image info
                - lidar2img (list[np.ndarray], optional): transformations from
                    lidar to different cameras
                - ann_info (dict): annotation info
        """
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']
        img_filename = os.path.join(self.data_root,
                                    info['image']['image_path'])

        # TODO: consider use torch.Tensor only
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)
        lidar2img = P0 @ rect @ Trv2c

        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None,
            lidar2img=lidar2img)

        if self.load_mode == 'lidar_frame':
            img_filenames = []
            lidar2imgs = []
            lidar2cams = []
            cam2imgs = []
            ego2globals = []
            for sweep_idx in range(-1, min(self.max_sweeps,
                                           len(info['sweeps']))):
                if sweep_idx >= 0:
                    ego2globals.append(
                        info['sweeps'][sweep_idx]['pose'].astype(np.float32))
                else:
                    ego2globals.append(info['pose'].astype(np.float32))
                for idx, multiview_index in enumerate(self.multiview_indices):
                    if sweep_idx >= 0:  # previous frames
                        view_filename = os.path.join(
                            self.data_root,
                            info['sweeps'][sweep_idx]['image_path'])
                    else:
                        view_filename = img_filename
                    view_filename = view_filename.replace(
                        self.default_view_index, multiview_index)
                    img_filenames.append(view_filename)
                    cam2img = info['calib'][f'P{idx}'].astype(np.float32)
                    cam2imgs.append(cam2img)
                    if idx == 0:
                        lidar2cam = rect @ info['calib'][
                            'Tr_velo_to_cam'].astype(np.float32)
                    else:
                        lidar2cam = rect @ info['calib'][
                            f'Tr_velo_to_cam{idx}'].astype(np.float32)
                    lidar2cams.append(lidar2cam)
                    lidar2imgs.append(cam2img @ lidar2cam)
            # TODO: unify the img_filenames in img_info and input_dict
            input_dict['img_info'] = dict(filename=img_filenames)
            input_dict['img_filename'] = img_filenames
            input_dict['lidar2img'] = lidar2imgs
            input_dict['lidar2cam'] = lidar2cams
            input_dict['cam2img'] = cam2imgs
            input_dict['ego2global'] = ego2globals
            # TODO: merge lidar get_ann_info and cam parse_cam_ann_info
            if not self.test_mode:
                annos = self.get_ann_info(index)
                input_dict['ann_info'] = annos
        elif self.load_mode in ['cam_frame', 'cam_mono']:
            # get img_info for monocular setting
            cam_intrinsic = P0
            input_dict['img_info'] = dict(
                img_filename=img_filename,
                cam_intrinsic=cam_intrinsic,
                width=info['image']['image_shape'][1],
                height=info['image']['image_shape'][0])
            # convert ann_info for monocular setting
            if not self.test_mode:
                parse_cam_ann_info = self._parse_cam_ann_info(
                    input_dict['img_info'], self.cam_ann_infos[index])
                input_dict['ann_info'] = parse_cam_ann_info
        else:
            raise NotImplementedError

        return input_dict

    def _parse_cam_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_bboxes_cam3d = []
        centers2d = []
        depths = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                # Note that the categories are defined as
                # ('Pedestrian', 'Cyclist', 'Car') in the converter
                # TODO: unify the settings
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                # 3D annotations in camera coordinates
                bbox_cam3d = np.array(ann['bbox_cam3d']).reshape(-1, )
                gt_bboxes_cam3d.append(bbox_cam3d)
                # 2.5D annotations in camera coordinates
                center2d = ann['center2d'][:2]
                depth = ann['center2d'][2]
                centers2d.append(center2d)
                depths.append(depth)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_cam3d:
            gt_bboxes_cam3d = np.array(gt_bboxes_cam3d, dtype=np.float32)
            centers2d = np.array(centers2d, dtype=np.float32)
            depths = np.array(depths, dtype=np.float32)
        else:
            gt_bboxes_cam3d = np.zeros((0, self.bbox_code_size),
                                       dtype=np.float32)
            centers2d = np.zeros((0, 2), dtype=np.float32)
            depths = np.zeros((0), dtype=np.float32)

        gt_bboxes_cam3d = CameraInstance3DBoxes(
            gt_bboxes_cam3d,
            box_dim=gt_bboxes_cam3d.shape[-1],
            origin=(0.5, 0.5, 0.5))
        gt_labels_3d = copy.deepcopy(gt_labels)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_bboxes_3d=gt_bboxes_cam3d,
            gt_labels_3d=gt_labels_3d,
            centers2d=centers2d,
            depths=depths,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

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
                 logger=None,
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
            # [info['annos'] for info in self.raw_data_infos]
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

                    print_log(
                        f'Results of {name}:\n' + ap_result_str, logger=logger)

            else:
                if metric == 'img_bbox2d':
                    ap_result_str, ap_dict = kitti_eval(
                        gt_annos,
                        result_files,
                        self.CLASSES,
                        eval_types=['bbox'])
                else:
                    ap_result_str, ap_dict = kitti_eval(
                        gt_annos, result_files, self.CLASSES, eval_types = ['bev', '3d'])
                print_log('\n' + ap_result_str, logger=logger)
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
            print_log(ret_texts)
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

            if self.load_mode == 'cam_frame':
                if idx % self.num_cams == 0:
                    box_dict_per_frame = []
                    cam0_idx = idx
            # box_dict = self.convert_valid_bboxes(pred_dicts, info)

            if self.load_mode == 'cam_frame':
                box_dict_per_frame.append(box_dict)
                if (idx + 1) % self.num_cams != 0:
                    continue
                box_dict = self.merge_multi_view_boxes(
                    box_dict_per_frame, self.data_infos[cam0_idx])


            if len(pred_dicts['boxes_3d']) > 0:
                box_preds = pred_dicts['boxes_3d']
                scores = pred_dicts['scores_3d']
                box_preds_lidar = pred_dicts['boxes_3d']
                label_preds = pred_dicts['labels_3d']

                anno = {
                    'name': [],
                    'bbox' : [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': []
                }

                for box, box_lidar, score, label in zip(
                        box_preds, box_preds_lidar,  scores,
                        label_preds):
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
                        anno['alpha'].append(-np.arctan2(box[0], box[2]) +
                                             box[6])
                    anno['bbox'].append(np.zeros(4))
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

                if submission_prefix is not None:
                    curr_file = f'{submission_prefix}/{sample_idx:07d}.txt'
                    with open(curr_file, 'w') as f:
                        loc = anno['location']
                        dims = anno['dimensions']  # lhw -> hwl

                        for idx in range(len(loc)):
                            print(
                                '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                                '{:.4f} {:.4f} {:.4f} '
                                '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.
                                format(anno['name'][idx], anno['alpha'][idx],
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
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0]),
                sample_idx=sample_idx)

        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)
        P0 = box_preds.tensor.new_tensor(P0)

        if self.load_mode == 'lidar_frame':
            box_preds_camera = box_preds.convert_to(Box3DMode.CAM,
                                                    rect @ Trv2c)
            box_preds_lidar = box_preds
        elif self.load_mode in ['cam_frame', 'cam_mono']:
            box_preds_camera = box_preds
            box_preds_lidar = box_preds.convert_to(
                Box3DMode.LIDAR, np.linalg.inv(rect @ Trv2c), correct_yaw=True)

        box_corners = box_preds_camera.corners
        box_corners_in_image = points_cam2img(box_corners, P0)
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
        from mmcv import Config
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
        rect = cam0_info['calib']['R0_rect'].astype(np.float32)
        Trv2c = cam0_info['calib']['Tr_velo_to_cam'].astype(np.float32)
        box_preds_camera = box_preds_lidar.convert_to(
            Box3DMode.CAM, rect @ Trv2c, correct_yaw=True)
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

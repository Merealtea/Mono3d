# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import cv2
from ..builder import PIPELINES
import pickle

@PIPELINES.register_module()
class LoadImageFromFile:
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_info']['filename']

        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        if self.color_type == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['timestamp'] = filename.split('/')[-1].split('.jpg')[0]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"channel_order='{self.channel_order}')")
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        # results['world2cam'] = results['img_info']['cam_extrinsic']
        results["direction"] = results['img_info']['filename'].split('/')[-2]
        return results


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = [cv2.imread(name) for name in filename]
        if self.color_type == 'rgb':
            img = [cv2.cvtColor(img_i, cv2.COLOR_BGR2RGB) for img_i in img]
        img = np.stack(img, axis=-1)
        
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results["num_views"] = len(filename)
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@PIPELINES.register_module()
class LoadAnnotations3D():
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_bbox=False,
                 with_label=False,
                 with_bbox_depth=False,
                 with_vel=True):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_vel = with_vel

    def _load_bboxes(self, results):
        """Private function to load 2D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2D bounding box annotations.
        """
        ann_info = results['ann_info']
        anntation_file = ann_info['filename']
        with open(anntation_file, 'rb') as f:
            annotations = pickle.load(f)

        results["gt_bboxes"] = []
        for ann in annotations:
            results["gt_bboxes"].append(ann['bbox'])
        results['gt_bboxes'] = np.array(results['gt_bboxes']).reshape(-1, 4)
        results['bbox_fields'].append("gt_bboxes")
        return results

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        ann_info = results['ann_info']
        anntation_file = ann_info['filename']
        with open(anntation_file, 'rb') as f:
            annotations = pickle.load(f)

        # x,y, z, l, w, h, yaw, velx, vely
        results["gt_bboxes_3d"] = []
        for ann in annotations:
            results["gt_bboxes_3d"].append(ann['bbox3d'])

        results['gt_bboxes_3d'] = np.array(results['gt_bboxes_3d']).reshape(-1, 9)
        if not self.with_vel:
            results['gt_bboxes_3d'] = results['gt_bboxes_3d'][:, :-2]
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        ann_info = results['ann_info']
        anntation_file = ann_info['filename']
        with open(anntation_file, 'rb') as f:
            annotations = pickle.load(f)

        results['centers2d'] = []
        results['depths'] = []

        for ann in annotations:
            results['centers2d'].append(ann['center2d'])
            results['depths'].append(ann['depth'])
        results['centers2d'] = np.array(results['centers2d']).reshape(-1, 2)
        results['depths'] = np.array(results['depths']).reshape(-1)
        results['bbox_fields'].append("centers2d")
        return results


    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        ann_info = results['ann_info']
        anntation_file = ann_info['filename']
        with open(anntation_file, 'rb') as f:
            annotations = pickle.load(f)

        results['gt_labels_3d'] = []
        for ann in annotations:
            if 'label_3d' in ann:
                results['gt_labels_3d'].append(ann['label_3d'])
            else:
                results['gt_labels_3d'].append([0] * len(ann['bbox3d']))
        results['gt_labels_3d'] = np.array(results['gt_labels_3d']).reshape(-1)
        results['attr_labels'] = np.zeros_like(results['gt_labels_3d'])
        results['gt_labels'] = np.zeros_like(results['gt_labels_3d'])
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox:
            results = self._load_bboxes(results)
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
        if self.with_label_3d:
            results = self._load_labels_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '

        return repr_str

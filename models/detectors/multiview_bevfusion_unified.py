# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from ..dense_heads import CenterHead, TransFusionHead
from ..builder import DETECTORS, build_head
from torch import nn
from configs.FisheyeParam import CamModel
from core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from ..fusion_layers.point_fusion import (point_sample_fisheye,
                                                       voxel_sample)
from core import xywhr2xyxyr, box3d_multiclass_nms, limit_period, bbox3d2result, build_prior_generator
from ..builder import (
    build_backbone,
    build_head,
    build_neck,
)
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F

@DETECTORS.register_module()
class MultiViewBEVFusionUnified(nn.Module):
    """Monocular 3D Object Detection with Lift Shift Shoot."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head_3d,
                 voxel_size,
                 anchor_generator,
                 neck_3d,
                 num_views,
                 train_cfg=None,
                 test_cfg=None,
                 transform_depth=True,
                 valid_sample=True,
                 temporal_aggregate='mean',
                 num_ref_frames=0,
                 **kwargs):
        super(MultiViewBEVFusionUnified, self).__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        
        self.cam_models = {"Hycan" : dict(zip(["left", "right", "front", "back"], 
                                   [CamModel(cam_dir, "Hycan", "torch", "cpu") for cam_dir in ["left", "right", "front", "back"]]) ),
                            "Rock" : dict(zip(["left", "right", "front", "back"],
                                    [CamModel(cam_dir, "Rock", "torch", "cpu") for cam_dir in ["left", "right", "front", "back"]]) ),
                           }

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if 'train_cfg' not in bbox_head_3d:
            bbox_head_3d.update(train_cfg=train_cfg)
            bbox_head_3d.update(test_cfg=test_cfg)
        self.neck_3d = build_neck(neck_3d)
        self.bbox_head_3d = build_head(bbox_head_3d)
        self.num_views = num_views
        
        self.voxel_size = voxel_size
        self.num_views = num_views
        self.num_ref_frames = num_ref_frames
        self.voxel_range = anchor_generator['ranges'][0]
        self.n_voxels = [
            round((self.voxel_range[3] - self.voxel_range[0]) /
                  self.voxel_size[0]),
            round((self.voxel_range[4] - self.voxel_range[1]) /
                  self.voxel_size[1]),
            round((self.voxel_range[5] - self.voxel_range[2]) /
                  self.voxel_size[2])
        ]
        self.anchor_generator = build_prior_generator(anchor_generator)
        self.valid_sample = valid_sample
        self.temporal_aggregate = temporal_aggregate
        self.transform_depth = transform_depth

        # onnx export parameters
        self.input_shape = None
        self.pad_shape = None
        self.img_shape = None
        self.ratio = None



    def to_device(self, device):
        self.to(device)
        for vehicle in self.cam_models:
            for direction in self.cam_models[vehicle]:
                self.cam_models[vehicle][direction].to(device)


    def set_onnx_parameters(self, input_shape, pad_shape, ratio, img_shape):
        self.input_shape = input_shape
        self.pad_shape = pad_shape
        self.ratio = ratio
        self.img_shape = img_shape


    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if pretrained is not None:
            pass
        
        self.bbox_head_3d.init_weights()
        # # He Initialization for bev_encode
        # for m in self.bev_encode.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def extract_feat(self, img, img_metas):
        """
        Args:
            img (torch.Tensor): [B, Nv, C_in, H, W]
            img_metas (list): each element corresponds to a group of images.
                len(img_metas) == B.

        Returns:
            torch.Tensor: bev feature with shape [B, C_out, N_y, N_x].
        """
        
        # TODO: Nt means the number of frames temporally
        # num_views means the number of views of a frame
        batch_size, _, C_in, H, W = img.shape



        if self.num_ref_frames > 0:
            num_frames = self.num_ref_frames + 1
        else:
            num_frames = 1
        input_shape = img.shape[-2:]
   
        # NOTE: input_shape is the largest pad_shape of the batch of images
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)
        if self.num_ref_frames > 0:
            cur_imgs = img[:, :self.num_views].reshape(-1, C_in, H, W)
            prev_imgs = img[:, self.num_views:].reshape(-1, C_in, H, W)
            cur_feats = self.backbone(cur_imgs)
            cur_feats = self.neck(cur_feats)[0]
            with torch.no_grad():
                prev_feats = self.backbone(prev_imgs)
                prev_feats = self.neck(prev_feats)[0]
            _, C_feat, H_feat, W_feat = cur_feats.shape
            cur_feats = cur_feats.view(batch_size, -1, C_feat, H_feat, W_feat)
            prev_feats = prev_feats.view(batch_size, -1, C_feat, H_feat, W_feat)
            batch_feats = torch.cat([cur_feats, prev_feats], dim=1)
        else:
            batch_imgs = img.view(-1, C_in, H, W)

            batch_feats = self.backbone(batch_imgs)
            # TODO: support SPP module neck
            batch_feats = self.neck(batch_feats)[0]
            _, C_feat, H_feat, W_feat = batch_feats.shape
            batch_feats = batch_feats.view(batch_size, -1, C_feat, H_feat,
                                           W_feat)
        # transform the feature to voxel & stereo space
        
        transform_feats = self.feature_transformation(batch_feats, img_metas,
                                                      self.num_views, num_frames)
    
        
        return transform_feats

    def feature_transformation(self, batch_feats, img_metas, num_views,
                               num_frames):
        print("Vehicle is {}: ".format(img_metas[0]['dataset_type']))
        # import pdb; pdb.set_trace()

        # TODO: support more complicated 2D feature sampling
        points = self.anchor_generator.grid_anchors(
            [self.n_voxels[::-1]], device=batch_feats.device)[0][:, :3]

        volumes = []

        img_crop_offsets = []
        for feature, img_meta in zip(batch_feats, img_metas):

            if torch.onnx.is_in_onnx_export():
                img_scale_factor = self.ratio
            else:
                if 'scale_factor' in img_meta:
                    if isinstance(
                            img_meta['scale_factor'],
                            np.ndarray) and len(img_meta['scale_factor']) >= 2:
                        img_scale_factor = (
                            points.new_tensor(img_meta['scale_factor'][:2])).to(points.device)
                    else:
                        img_scale_factor = (
                            points.new_tensor(img_meta['scale_factor'])).to(points.device)
                else:
                    img_scale_factor = (1)

            img_flip = False #img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = (
                points.new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
   
 
            img_crop_offsets.append(img_crop_offset)
            # TODO: remove feature sampling from back
            # TODO: support different scale_factors/flip/crop_offset for
            # different views
            frame_volume = []
            frame_valid_nums = []
            
            for frame_idx in range(num_frames):
                volume = []
                valid_flags = []
                for view_idx in range(num_views):
                    
                    cam_dir = img_meta['direction'][view_idx]
  
                    sample_idx = frame_idx * num_views + view_idx
                    if torch.onnx.is_in_onnx_export():
                        sample_results = point_sample_fisheye(
                            img_meta,
                            img_features=feature[sample_idx][None, ...],
                            points=points,
                            camera_model=self.cam_models[cam_dir],
                            coord_type='LIDAR',
                            img_scale_factor=img_scale_factor,
                            img_crop_offset=img_crop_offset,
                            img_flip=img_flip,
                            img_pad_shape=self.pad_shape,
                            img_shape=self.input_shape,
                            aligned=False,
                            valid_flag=self.valid_sample)
                    else:
                        sample_results = point_sample_fisheye(
                            img_meta,
                            img_features=feature[sample_idx][None, ...],
                            points=points,
                            camera_model=self.cam_models[img_meta['dataset_type']][cam_dir],
                            coord_type='LIDAR',
                            img_scale_factor=img_scale_factor,
                            img_crop_offset=img_crop_offset,
                            img_flip=img_flip,
                            img_pad_shape=img_meta['input_shape'],
                            img_shape=img_meta['img_shape'][sample_idx][:2],
                            aligned=False,
                            valid_flag=self.valid_sample)
                    if self.valid_sample:
                        volume.append(sample_results[0])
                        valid_flags.append(sample_results[1])
                    else:
                        volume.append(sample_results)
                    # TODO: save valid flags, more reasonable feat fusion
                if self.valid_sample:
                    valid_nums = torch.stack(
                        valid_flags, dim=0).sum(0)  # (N, )
                    
                    volume = torch.stack(volume, dim=0).sum(0)
                    valid_mask = valid_nums > 0

                    # expand valid_mask to the same shape as volume
                    valid_mask = valid_mask[:, None].expand_as(volume)
                    volume[~valid_mask] = 0
                    frame_valid_nums.append(valid_nums)
                else:
                    volume = torch.stack(volume, dim=0).mean(0)
                frame_volume.append(volume)
            if self.valid_sample:
                if self.temporal_aggregate == 'mean':
                    frame_volume = torch.stack(frame_volume, dim=0).sum(0)
                    frame_valid_nums = torch.stack(
                        frame_valid_nums, dim=0).sum(0)
                    frame_valid_mask = frame_valid_nums > 0
                    frame_volume[~frame_valid_mask] = 0
                    frame_volume = frame_volume / torch.clamp(
                        frame_valid_nums[:, None], min=1)
                elif self.temporal_aggregate == 'concat':
                    frame_valid_nums = torch.stack(frame_valid_nums, dim=1)
                    frame_volume = torch.stack(frame_volume, dim=1)
                    frame_valid_mask = frame_valid_nums > 0
                    frame_valid_mask = frame_valid_mask[:, :, None].expand_as(
                        frame_volume)
                    # raise Exception("valid_nums shape is: ", (frame_volume.shape, frame_valid_nums.shape))
                    frame_volume[~frame_valid_mask] = 0
                    frame_volume = (frame_volume / torch.clamp(
                        frame_valid_nums[:, :, None], min=1)).flatten(
                            start_dim=1, end_dim=2)
            else:
                frame_volume = torch.stack(frame_volume, dim=0).mean(0)

            frame_volume = frame_volume.reshape(self.n_voxels[::-1] + [-1]).permute(3, 2, 1, 0)
            if img_meta['dataset_type'] == "Rock":
                frame_volume = torch.flip(frame_volume.transpose(-3, -2), dims=[-3])
            volumes.append(frame_volume)
        
        volume_feat = torch.stack(volumes)  # (B, C, N_x, N_y, N_z)
        volume_feat = self.neck_3d(volume_feat)[0]

        # grid_sample stereo features from the volume feature
 
        # TODO: unify the output format of neck_3d
        transform_feats = volume_feat

        return transform_feats

    
    def forward(self, img, img_metas=None, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img, img_metas, **kwargs)

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)


    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
    
        bev_feat = self.extract_feat(
            img,
            img_metas,
        )
        # depth = depth_img[0][0].cpu().detach().numpy()
        # depth = (depth / np.max(depth) * 255).astype(np.int8)
        # cv2.imwrite("Depth.jpg", depth)

        for veh_id in range(len(img_metas)):
            if img_metas[veh_id]['dataset_type'] == "Rock" and len(gt_bboxes_3d[veh_id]) > 0:
                gt_bboxes_3d[veh_id][:, [1, 0, 3, 4]] = gt_bboxes_3d[veh_id][:, [0, 1, 4, 3]]
                gt_bboxes_3d[veh_id][:, 0] = -gt_bboxes_3d[veh_id][:, 0]
                gt_bboxes_3d[veh_id][:, 2] += 1.8
        
        if isinstance(bev_feat, list):
            bev_feat = bev_feat[0]
      
        if isinstance(self.bbox_head_3d, TransFusionHead):
            gt_bboxes_3d = [LiDARInstance3DBoxes(box, box_dim=box.shape[-1]) for box in gt_bboxes_3d]

            losses = self.bbox_head_3d.loss([bev_feat], gt_bboxes_3d, gt_labels_3d, gt_ids_3d=[np.arange(len(gt_labels_3d[i])) for i in range(len(gt_labels_3d))],
                                            img_metas = img_metas)
        elif isinstance(self.bbox_head_3d, CenterHead):
            outs = self.bbox_head_3d([bev_feat])
            gt_bboxes_3d = [LiDARInstance3DBoxes(box, box_dim=box.shape[-1]) for box in gt_bboxes_3d]
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs, img_metas]
            losses = self.bbox_head_3d.loss(*loss_inputs)
        else:
            outs = self.bbox_head_3d([bev_feat])
            losses = self.bbox_head_3d.loss(*outs, gt_bboxes_3d, gt_labels_3d,
                                            img_metas)

        # depth_loss = self.get_depth_loss(depth_img, depth_output)
        # losses['depth_loss'] = depth_loss

        # TODO: loss_dense_depth, loss_2d, loss_imitation
        return losses

    # From BEVDepth
    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return 3.0 * depth_loss

    def forward_test(self, img, img_metas, **kwargs):
        """Forward of testing.

        Args:
            img (torch.Tensor): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.
0.175
        Returns:
            list[dict]: Predicted 3d boxes.
        """
        # not supporting aug_test for now
        return self.simple_test(img, img_metas)

    def simple_test(self, img, img_metas):
        bev_feat = self.extract_feat(
            img,
            img_metas,
        )
        batch_size = bev_feat.shape[0]

        """
        if self.with_depth_head:
            stereo_feat = feats[1]
            depth_volumes, _, depth_preds = self.depth_head(stereo_feat)
        """
        
        if isinstance(self.bbox_head_3d, CenterHead):
            outs = self.bbox_head_3d([bev_feat])
            bbox_list = self.bbox_head_3d.get_bboxes(
                outs, img_metas, rescale=False)
        elif isinstance(self.bbox_head_3d, TransFusionHead):
            bbox_list = self.bbox_head_3d.predict([bev_feat], None, img_metas)[0]
        else:
            outs = self.bbox_head_3d([bev_feat])
            bbox_list = self.bbox_head_3d.get_bboxes(*outs, img_metas)

        for i in range(len(img_metas)):
            if img_metas[i]['dataset_type'] == "Rock":
                if len(bbox_list[i][0]) > 0:
                    bbox_list[i][0].tensor[:, 0] = -bbox_list[i][0].tensor[:, 0]
                    bbox_list[i][0].tensor[:, [1, 0, 3, 4]] = bbox_list[i][0].tensor[:, [0, 1, 4, 3]]
                    bbox_list[i][0].tensor[:, 2] -= 1.8

        if isinstance(self.bbox_head_3d, TransFusionHead):
            bbox_results = [
                bbox3d2result(det_bboxes, det_scores, det_labels)
                for det_bboxes, det_scores, det_labels, _, _ in bbox_list
            ]
        else:
            bbox_results = [
                bbox3d2result(det_bboxes, det_scores, det_labels)
                for det_bboxes, det_scores, det_labels in bbox_list
            ]

 
        return bbox_results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test with augmentations.

        Args:
            imgs (list[torch.Tensor]): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        raise NotImplementedError
    

    def onnx_export(self, img, img_metas, **kwargs):
        # onnx export not including nms
        bev_feat = self.get_voxels(img, img_metas)
        bev_feat = self.bev_encode(bev_feat)
        outs = self.bbox_head_3d([bev_feat])
        """
        if self.with_depth_head:
            stereo_feat = feats[1]
            depth_volumes, _, depth_preds = self.depth_head(stereo_feat)
        """
        bbox_list = self.bbox_head_3d.get_bboxes_onnx_export(*outs)
        return bbox_list
    
    @staticmethod
    def nms_for_bboxes(prediction_array,
                        box_code_size = 7,
                        max_num=100,
                        score_thr=0.0, 
                        dir_offset=0,
                        dir_limit_offset=0,
                        cfg={}):
        """NMS for bboxes.
            cfg (dict): Config dict.
                score_thr (float): Score threshold to filter bboxes.
                max_num (int): Maximum number of selected bboxes.
                use_rotate_nms (bool): Whether to use rotate nms.
        """
        prediction_array = torch.tensor(prediction_array).reshape(-1, 10)
        mlvl_bboxes, mlvl_scores, mlvl_dir_scores = \
            prediction_array[:, :7],\
                  prediction_array[:, 7:9],\
                      prediction_array[:, 9]
        cfg['use_rotate_nms'] = False
        cfg['nms_thr'] = 0.5
        mlvl_bboxes_for_nms = xywhr2xyxyr(LiDARInstance3DBoxes(
            mlvl_bboxes, box_dim=box_code_size).bev)
        
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_scores, score_thr, max_num,
                                       cfg, mlvl_dir_scores)
        bboxes, scores, labels, dir_scores = results
        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - dir_offset,
                                   dir_limit_offset, np.pi)
            bboxes[..., 6] = (
                dir_rot + dir_offset +
                np.pi * dir_scores.to(bboxes.dtype))
        bboxes = LiDARInstance3DBoxes(bboxes, box_dim=box_code_size)
        bbox_results = [
                bbox3d2result(bboxes, scores, labels)
            ]
        return bbox_results

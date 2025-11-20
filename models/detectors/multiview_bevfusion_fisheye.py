# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from ..dense_heads import CenterHead
from ..builder import DETECTORS, build_head
from torch import nn
from configs.FisheyeParam import CamModel
from core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from core import xywhr2xyxyr, box3d_multiclass_nms, limit_period, bbox3d2result
from ..builder import (
    build_backbone,
    build_head,
    build_neck,
    build_vtransform,
)

@DETECTORS.register_module()
class MultiViewBEVFusionFisheye(nn.Module):
    """Monocular 3D Object Detection with Lift Shift Shoot."""

    def __init__(self,
                 backbone,
                 neck,
                 vtransform,
                 decoder,
                 bbox_head_3d,
                 num_views,
                 train_cfg=None,
                 test_cfg=None,
                 vehicle = None,):
        super(MultiViewBEVFusionFisheye, self).__init__()

        self.bev_encode = nn.ModuleDict(
                {
                    "backbone": build_backbone(backbone),
                    "neck": build_neck(neck),
                    "vtransform": build_vtransform(vtransform),
                }
            )

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        
        self.cam_models = dict(zip(["left", "right", "front", "back"], 
                                   [CamModel(cam_dir, vehicle, "torch", "cuda:0") for cam_dir in ["left", "right", "front", "back"]]) )

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if 'train_cfg' not in bbox_head_3d:
            bbox_head_3d.update(train_cfg=train_cfg)
            bbox_head_3d.update(test_cfg=test_cfg)

        self.bbox_head_3d = build_head(bbox_head_3d)
        self.init_weights()
        self.num_views = num_views

    def to_device(self, device):
        self.to(device)
        for direction in self.cam_models:
            self.cam_models[direction].to(device)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if pretrained is not None:
            pass
        # # He Initialization for bev_encode
        # for m in self.bev_encode.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def extract_camera_features(
            self,
            x,
            img_metas,
            gt_depths=None,
        ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)
        x = self.bev_encode["backbone"](x)
        x = self.bev_encode["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.bev_encode["vtransform"](
            x,
            self.cam_models,
            img_metas,
        )
        return x
    
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
        depth_img=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        auxiliary_losses = {}

        x = self.extract_camera_features(
            img,
            img_metas,
            gt_depths=depth_img,
        )

        x = self.decoder["backbone"](x)

        bev_feat = self.decoder["neck"](x)
        if isinstance(bev_feat, list):
            bev_feat = bev_feat[0]
      
        outs = self.bbox_head_3d([bev_feat])
        if not isinstance(self.bbox_head_3d, CenterHead):
            losses = self.bbox_head_3d.loss(*outs, gt_bboxes_3d, gt_labels_3d, img_metas)
        else:
            gt_bboxes_3d = [LiDARInstance3DBoxes(box, box_dim=box.shape[-1]) for box in gt_bboxes_3d]
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs, img_metas]
            losses = self.bbox_head_3d.loss(*loss_inputs)

        # TODO: loss_dense_depth, loss_2d, loss_imitation
        return losses


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
        x = self.extract_camera_features(
            img,
            img_metas,
        )
        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        bev_feat = self.decoder["neck"](x)

        outs = self.bbox_head_3d([bev_feat])
        """
        if self.with_depth_head:
            stereo_feat = feats[1]
            depth_volumes, _, depth_preds = self.depth_head(stereo_feat)
        """
        
        if not isinstance(self.bbox_head_3d, CenterHead):
            bbox_list = self.bbox_head_3d.get_bboxes(*outs, img_metas)
        else:
            bbox_list = self.bbox_head_3d.get_bboxes(
                outs, img_metas, rescale=False)

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

# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
from core import bbox3d2result, build_prior_generator
from ..dense_heads import CenterHead
from ..fusion_layers.point_fusion import (point_sample_fisheye,
                                                       voxel_sample)
from ..builder import DETECTORS, build_head
from torch import nn
from configs.FisheyeParam import CamModel
from core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from core import xywhr2xyxyr, box3d_multiclass_nms, limit_period
from time import time
from torchvision.models.resnet import resnet18
from .lss_tools import gen_dx_bx, cumsum_trick, QuickCumsum
from copy import deepcopy

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C
        self.trunk = EfficientNet.from_pretrained("efficientnet-b4")

        self.up1 = Up(32+56, 128)
        self.downsample = downsample
        self.depthnet = nn.Conv2d(128, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        downsample = 1
        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
                downsample *= 2
            prev_x = x
            if downsample == self.downsample:
                break
        # Head
        # import pdb; pdb.set_trace()
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_{}'.format(len(endpoints))], endpoints['reduction_{}'.format(len(endpoints)-1)])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x
    
class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        # trunk = resnet18(pretrained=False, zero_init_residual=True)
        # self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = trunk.bn1
        # self.relu = trunk.relu

        # self.layer1 = trunk.layer1
        # self.layer2 = trunk.layer2
        # self.layer3 = trunk.layer3

        # self.up1 = Up(64+256, 256, scale_factor=4)
        # self.up2 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear',
        #                       align_corners=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, outC, kernel_size=1, padding=0),
        # )

        self.conv1 = nn.Sequential(
            nn.Conv2d(inC, 128, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.residual = nn.Sequential( 
            nn.Conv2d(inC + 256, outC, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)

        # x1 = self.layer1(x)
        # x = self.layer2(x1)
        # x = self.layer3(x)

        # x = self.up1(x, x1)
        # x = self.up2(x)
        x1 = self.conv1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.residual(x)
        return x


@DETECTORS.register_module()
class MultiViewLSSFisheye(nn.Module):
    """Monocular 3D Object Detection with Lift Shift Shoot."""

    def __init__(self,
                 backbone,
                 grid_conf,
                 final_dim,
                 bbox_head_3d,
                 num_views,
                 outC,
                 use_quickcumsum,
                 train_cfg=None,
                 test_cfg=None,
                 vehicle = None,
                 normalizer_clamp_value=10,):
        super(MultiViewLSSFisheye, self).__init__()
        self.grid_conf = grid_conf
        self.num_views = num_views
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        # dx is the resolution of the voxel grid in x, y, z
        self.dx = nn.Parameter(dx, requires_grad=False)

        # bx is the offset of the first voxel grid in x, y, z
        self.bx = nn.Parameter(bx, requires_grad=False)

        # nx is the number of voxels in x, y, z
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.use_quickcumsum = use_quickcumsum

        self.downsample = backbone['downsample']
        self.camC = backbone['C']
        self.outC = outC

        self.final_dim = final_dim
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.backbone = CamEncode(self.D, backbone['C'], backbone['downsample'])

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        bbox_head_3d.update(train_cfg=train_cfg)
        bbox_head_3d.update(test_cfg=test_cfg)
        # TODO: remove this hack

        self.bbox_head_3d = build_head(bbox_head_3d)
        self.bev_encode = BevEncode(self.camC, self.outC)
        
        self.cam_models = dict(zip(["left", "right", "front", "back"], 
                                   [CamModel(cam_dir, vehicle, "torch", "cuda:0") for cam_dir in ["left", "right", "front", "back"]]) )
    
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
        # He Initialization for bev_encode
        for m in self.bev_encode.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.backbone(x)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x
    
    def voxel_pooling(self, geom_feats, valid_mask, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        for batch in range(B):
            voxel_features = torch.zeros((self.nx[0], self.nx[1], self.nx[2], C), device=x.device)
            for view in range(N):
                geom_feats_view = geom_feats[batch, view]
                valid_mask_view = valid_mask[batch, view]
                view_feature = x[batch, view]

                geom_idx = ((geom_feats_view - (self.bx - self.dx/2.)) / self.dx).long()

                # filter out points that are outside box
                kept = (geom_idx[:, 0] >= 0) & (geom_idx[:, 0] < self.nx[0])\
                    & (geom_idx[:, 1] >= 0) & (geom_idx[:, 1] < self.nx[1])\
                    & (geom_idx[:, 2] >= 0) & (geom_idx[:, 2] < self.nx[2])\
                    & valid_mask_view
                
                geom_idx = geom_idx[kept]
                view_feature = view_feature[kept]

                
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        valid_mask = valid_mask.reshape(Nprime)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])\
            & valid_mask

        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        # X Y Z B
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_geometry(self, batch_size, img_metas):
        # w, h, d
        points_uv_depth = self.frustum.clone()
        D, H, W, _ = points_uv_depth.shape

        # change uv to original resized uv
        if isinstance(img_metas[0]['scale_factor'], torch.Tensor):
            ratio = img_metas[0]['scale_factor'].to(points_uv_depth.device)[:2].view(1, 2)
        else:
            ratio = torch.FloatTensor(img_metas[0]['scale_factor']).to(points_uv_depth.device)[:2].view(1, 2)
        points_uv_depth[..., :2] = points_uv_depth[..., :2] / ratio
        points_uv_depth = points_uv_depth.reshape(-1, 3).transpose(0, 1)

        # remove the pad voxel
        ori_shape = img_metas[0]['ori_shape'][:2]
        valid_mask = (points_uv_depth[0] < ori_shape[1]) & (points_uv_depth[1] < ori_shape[0])
        
        points_cameras = []
        for cam_dir in img_metas[0]['direction']:
            frustum = deepcopy(points_uv_depth)
            # transform to camera coordinate
            points_camera = self.cam_models[cam_dir].image2cam(frustum[:2], 
                                                               frustum[2])

            # transform to base_link coordinate
            points_camera = self.cam_models[cam_dir].cam2world(points_camera).transpose(0, 1)

            # reshape to points_uv_depth
            points_camera = points_camera.view(D, H, W, 3)
            points_cameras.append(points_camera)
        points_cameras = torch.stack(points_cameras, 0)
        # add batch size                                        B        N   D   H   W  3
        points_cameras = points_cameras.unsqueeze(0).expand(batch_size, -1, -1, -1, -1, -1)
        valid_mask = valid_mask.view(1, 1, D, H, W).expand(batch_size, self.num_views, -1, -1, -1)
        return points_cameras, valid_mask

    def get_voxels(self, x, img_metas):
        geom, valid_mask = self.get_geometry(x.shape[0], img_metas)
        x = self.get_cam_feats(x)
        x = self.voxel_pooling(geom, valid_mask, x) / self.num_views

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

    def forward_train(self, img,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      **kwargs):
        bev_feat = self.get_voxels(img, img_metas)
        bev_feat = self.bev_encode(bev_feat)
        outs = self.bbox_head_3d([bev_feat])
        if not isinstance(self.bbox_head_3d, CenterHead):
            losses = self.bbox_head_3d.loss(*outs, gt_bboxes_3d, gt_labels_3d,
                                            img_metas)
        else:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.bbox_head_3d.loss(*loss_inputs)
        # TODO: loss_dense_depth, loss_2d, loss_imitation
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        """Forward of testing.

        Args:
            img (torch.Tensor): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        # not supporting aug_test for now
        return self.simple_test(img, img_metas)

    def simple_test(self, img, img_metas):
        bev_feat = self.get_voxels(img, img_metas)
        bev_feat = self.bev_encode(bev_feat)
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

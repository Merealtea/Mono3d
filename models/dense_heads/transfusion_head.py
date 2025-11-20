# modify from https://github.com/mit-han-lab/bevfusion
import copy
from typing import List, Tuple
from mmcv.runner import BaseModule, force_fp32
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from core.bbox import (AssignResult, PseudoSampler,
                                       build_assigner, build_bbox_coder,
                                       build_sampler)
from core import multi_apply
from core import circle_nms, draw_heatmap_gaussian,\
                         gaussian_radius, nms_bev, xywhr2xyxyr
from torch import nn
import cv2
from models.dense_heads.centerpoint_head import SeparateHead
from models.builder import MODELS
from models.fusion_modules.delay_compensation import TimeCompensation
from mmcv import ConfigDict
from .train_mixins import AnchorTrainMixin

def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


@MODELS.register_module()
class ConvFuser(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))


@MODELS.register_module()
class TransFusionHead(BaseModule, AnchorTrainMixin):

    def __init__(
        self,
        num_proposals=128,
        auxiliary=True,
        in_channels=128 * 3,
        hidden_channel=128,
        num_classes=4,
        # config for Transformer
        num_decoder_layers=3,
        decoder_layer=dict(),
        num_heads=8,
        nms_kernel_size=1,
        bn_momentum=0.1,
        # config for FFN
        common_heads=dict(),
        num_heatmap_convs=2,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        bias='auto',
        # loss
        loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
        loss_bbox=dict(type='L1Loss', reduction='mean'),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean'),
        # others
        train_cfg=None,
        test_cfg=None,
        bbox_coder=None,
        with_delay=False,
    ):
        super(TransFusionHead, self).__init__()

        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.nms_kernel_size = nms_kernel_size
        self.train_cfg = ConfigDict(train_cfg) if train_cfg is not None else None
        self.test_cfg = ConfigDict(test_cfg) if test_cfg is not None else None

        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_heatmap = MODELS.build(loss_heatmap)
        self.loss_cls = MODELS.build(loss_cls)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.sampling = False

        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        # layers = []
        # layers.append(
        #     ConvModule(
        #         hidden_channel,
        #         hidden_channel,
        #         kernel_size=3,
        #         padding=1,
        #         bias=bias,
        #         conv_cfg=dict(type='Conv2d'),
        #         norm_cfg=dict(type='BN2d'),
        #     ))
        # layers.append(
        #     build_conv_layer(
        #         dict(type='Conv2d'),
        #         hidden_channel,
        #         num_classes,
        #         kernel_size=3,
        #         padding=1,
        #         bias=bias,
        #     ))
        # self.heatmap_head = nn.Sequential(*layers)
        
        self.heatmap_head = nn.Conv2d(hidden_channel, num_classes, 1)

        # self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(MODELS.build(decoder_layer))

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(
                SeparateHead(
                    hidden_channel,
                    heads,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=bias,
                ))
            # import pdb; pdb.set_trace()

        self.init_cfg = dict(
            type='Normal',
            layer='Conv2d',
            std=0.01,
            override=dict(
                type='Normal', name='heatmap_head', std=0.01, bias_prob=0.01))
    
        self.init_weights()
        self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training # noqa: E501
        x_size = self.test_cfg['grid_size'][1] // self.test_cfg[
            'out_size_factor']
        y_size = self.test_cfg['grid_size'][0] // self.test_cfg[
            'out_size_factor']
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None
        
        self.delay_compensation = TimeCompensation(
            embed_dims=hidden_channel,
            RTE_ratio=2,
            num_decoder_layers = 1,
            num_classes = self.num_classes, 
        )
        
        self.with_delay = with_delay



    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        # import pdb; pdb.set_trace()
        return coord_base

    def init_weights(self):
        # initialize transformer
        super().init_weights()
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()


    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    def forward_single(self, inputs, time_delay=None, metas=None):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = inputs.shape[0]

        # img = np.zeros((720 * 4, 1280, 3))
        # for i, img_file in enumerate(metas[0]['img_filename']):
            
        #     tmp = cv2.imread(img_file)
        #     img[i * 720 : (i + 1) * 720] = cv2.resize(tmp, (1280, 720))
        # cv2.imwrite("imgs.png", img)

        # import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()
        fusion_feat = self.shared_conv(inputs)

        #################################
        # image to BEV
        #################################

        fusion_feat_flatten = fusion_feat.reshape(batch_size,
                                               fusion_feat.shape[1],
                                               -1)  # [BS, C, H*W]
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(fusion_feat.device)

        #################################
        # query initialization
        #################################
        with torch.autocast('cuda', enabled=False):
            dense_heatmap = self.heatmap_head(fusion_feat.float())

        heatmap = dense_heatmap.detach().sigmoid()

        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max = F.max_pool2d(
            heatmap, kernel_size=1, stride=1, padding=0)

        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        # top num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(
            dim=-1, descending=True)[..., :self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_feat = fusion_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, fusion_feat_flatten.shape[1], -1),
            dim=-1,
        )
        self.query_labels = top_proposals_class

        # add category embedding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(
                -1, -1, bev_pos.shape[-1]),
            dim=1,
        )
        #################################
        # transformer decoder layer (Fusion feature as K,V)
        #################################
        ret_dicts = []
        for i in range(self.num_decoder_layers):
            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            query_feat = self.decoder[i](
                query_feat,
                key=fusion_feat_flatten,
                query_pos=query_pos,
                key_pos=bev_pos)

            # Prediction
            res_layer = self.prediction_heads[i](query_feat)
            # import pdb; pdb.set_trace()
            res_layer['center'] = res_layer['center'] + query_pos.permute(
                0, 2, 1)
            ret_dicts.append(res_layer)

            # for next level positional embedding
            query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)
            

        ret_dicts[0]['query_heatmap_score'] = heatmap.gather(
            index=top_proposals_index[:,
                                      None, :].expand(-1, self.num_classes,
                                                      -1),
            dim=-1,
        )  # [bs, num_classes, num_proposals]
        ret_dicts[0]['dense_heatmap'] = dense_heatmap
        ret_dicts[0]['query_feat'] = query_feat
        ret_dicts[0]['query_pose'] = query_pos

        if self.with_delay:
            # get decode boxes
            predict_boxes = self.bbox_coder.decode(
                ret_dicts[-1]['heatmap'].clone(),
                ret_dicts[-1]['rot'].clone(),
                ret_dicts[-1]['dim'].clone(),
                ret_dicts[-1]['center'].clone(),
                ret_dicts[-1]['height'].clone(),
                ret_dicts[-1].get('vel', None).clone(),
            )

            predict_boxes_list = []
            for decode_res in predict_boxes:
                predict_boxes_list.append(decode_res['bboxes'])
            predict_boxes = torch.stack(predict_boxes_list, dim=0) # [bs, num_proposals, 9]

            # do feature compensation and predict new query pose
            future_ret_dicts = self.delay_compensation(
                query_feat=ret_dicts[0]['query_feat'],
                dts=time_delay[:,0:1],
                query_pose=ret_dicts[0]['query_pose'],
                predict_boxes=predict_boxes
            )

        if self.auxiliary is False: 
            # only return the results of last decoder layer
            return [ret_dicts[-1]]

        # return all the layer's results for auxiliary superivison
        new_res = {}

        for key in ret_dicts[0].keys():
            if key not in [
                    'dense_heatmap', 'dense_heatmap_old', 'query_heatmap_score'
            ]:
                new_res[key] = torch.cat(
                    [ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]
 
        if not self.with_delay:
            return [new_res]

        return [new_res, future_ret_dicts]

    def forward(self, feats, time_delay=None, metas=None):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.
        Returns:
            tuple(list[dict]): Output results. first index by level, second
            index by layer
        """
        if isinstance(feats, torch.Tensor):
            feats = [feats]

        if isinstance(time_delay, torch.Tensor):
            time_delay = [time_delay]

        if time_delay is None:
            time_delay = [None] * len(feats)
            
        res = multi_apply(self.forward_single, feats, time_delay, [metas])
        if len(res) == 2:
            res = (res[0], res[1][0])  # [dict, dict]
        # assert len(res) == 1, 'only support one level features.'
        return res

    def predict(self, batch_feats, time_delay=None, batch_input_metas=None):
        preds_dicts = self(batch_feats, time_delay, batch_input_metas)
        res = self.predict_by_feat(preds_dicts, batch_input_metas)
        return res

    def predict_by_feat(self,
                        preds_dicts,
                        metas,):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer
            & each batch.
        """
        rets = []

        if len(preds_dicts) > 1:
            # 这个是加上了未来预测
            preds_dicts = preds_dicts[0][:self.num_decoder_layers] + [preds for preds in preds_dicts[1]]
        else:
            preds_dicts = preds_dicts[0]

        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict['heatmap'].shape[0]
            batch_score = preds_dict['heatmap'][
                ..., -self.num_proposals:].sigmoid()
            # if self.loss_iou.loss_weight != 0:
            #    batch_score = torch.sqrt(batch_score * preds_dict[0]['iou'][..., -self.num_proposals:].sigmoid()) # noqa: E501
            one_hot = F.one_hot(
                self.query_labels,
                num_classes=self.num_classes).permute(0, 2, 1)
            
            if layer_id == 0:
                # 只有第一个才有query_heatmap_score
                batch_score = batch_score * preds_dict[
                    'query_heatmap_score'] * one_hot
            else:
                batch_score = batch_score * one_hot

            batch_center = preds_dict['center'][..., -self.num_proposals:]
            batch_height = preds_dict['height'][..., -self.num_proposals:]
            batch_dim = preds_dict['dim'][..., -self.num_proposals:]
            batch_rot = preds_dict['rot'][..., -self.num_proposals:]
            batch_vel = None
            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel'][..., -self.num_proposals:]
            batch_query_feat = preds_dict['query_feat'][..., -self.num_proposals:]
            batch_query_pose = preds_dict['query_pose'][..., -self.num_proposals:]


            temp = self.bbox_coder.decode(
                batch_score,
                batch_rot,
                batch_dim,
                batch_center,
                batch_height,
                batch_vel,
                batch_query_feat,
                batch_query_pose,
                filter=True,
            )

            self.tasks = [
                dict(
                    num_class=1,
                    class_names=['Pedestrian'],
                    indices=[0],
                    radius=0.175),
            ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                query_feat = temp[i]['query']
                query_pose = temp[i]['query_pose']
                # adopt circle nms for different categories
                if self.test_cfg['nms_type'] is not None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task['indices']:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task['radius'] > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                boxes_for_nms = torch.cat(
                                    [
                                        boxes3d[task_mask][:, :2],
                                        scores[:, None][task_mask],
                                    ],
                                    dim=1,
                                )
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius'],
                                    ))
                            else:
                                boxes_for_nms = xywhr2xyxyr(
                                    metas[i]['box_type_3d'](
                                        boxes3d[task_mask][:, :7], 7).bev)
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_bev(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg['pre_maxsize'],
                                    post_max_size=self.
                                    test_cfg['post_maxsize'],
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(
                                task_mask != 0)[0][task_keep_indices]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    ret = dict(
                        bboxes=boxes3d[keep_mask],
                        scores=scores[keep_mask],
                        labels=labels[keep_mask],
                        query_feat=query_feat[keep_mask],
                        query_pose=query_pose[keep_mask],
                    )
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels, query_feat=query_feat, query_pose=query_pose)
                try:
                    ret_layer.append(
                        [metas[0]['box_type_3d'](ret['bboxes'], box_dim=ret['bboxes'].shape[-1]),
                        ret['scores'],
                        ret['labels'].int(),
                        ret['query_feat'],
                        ret['query_pose']], # Transform to vehicle coordinate
                    )
                except:
                    import pdb; pdb.set_trace()

            rets.append(ret_layer)
        # assert len(
        #     rets
        # ) == 1, f'only support one layer now, but get {len(rets)} layers'

        return rets

    def get_targets(self,
                    preds_dict: List[dict], 
                    gt_bboxes_3d,
                    gt_labels_3d, 
                    gt_ids_3d,
                    img_metas,
                    ):
        """Generate training targets.
        Args:
            batch_gt_instances_3d (List[InstanceData]):
            preds_dict (list[dict]): The prediction results. The index of the
                list is the index of layers. The inner dict contains
                predictions of one mini-batch:
                - center: (bs, 2, num_proposals)
                - height: (bs, 1, num_proposals)
                - dim: (bs, 3, num_proposals)
                - rot: (bs, 2, num_proposals)
                - vel: (bs, 2, num_proposals)
                - cls_logit: (bs, num_classes, num_proposals)
                - query_score: (bs, num_classes, num_proposals)
                - heatmap: The original heatmap before fed into transformer
                    decoder, with shape (bs, 10, h, w)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)
                    [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []

        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                preds = []
                for i in range(self.num_decoder_layers):
                    pred_one_layer = preds_dict[i][key][batch_idx:batch_idx +
                                                        1]
                    preds.append(pred_one_layer)
                pred_dict[key] = torch.cat(preds)
            
            list_of_pred_dict.append(pred_dict)
        assert len(gt_bboxes_3d) == len(list_of_pred_dict)
        res_tuple = multi_apply(
            self.get_targets_single,
            list_of_pred_dict,
            gt_bboxes_3d,
            gt_labels_3d,
            gt_ids_3d,
            np.arange(len(gt_bboxes_3d)),
            img_metas,
        )
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        heatmap = torch.cat(res_tuple[7], dim=0)
        assign_ids = torch.cat(res_tuple[8], dim=0)
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
            assign_ids,
        )

    def get_targets_single(self, preds_dict, gt_bboxes_3d, gt_labels_3d, gt_ids_3d, batch_idx, img_meta):
        """Generate training targets for a single sample.
        Args:
            gt_instances_3d (:obj:`InstanceData`): ground truth of instances.
            preds_dict (dict): dict of prediction result for a single sample.
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask) [1,
                    num_proposals] # noqa: E501
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
                - torch.Tensor: heatmap targets.
        """

        # 1. Assignment
        num_proposals = preds_dict['center'].shape[-1]

        # get pred boxes, carefully ! don't change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(
            score, rot, dim, center, height,
            vel)  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]['bboxes']
        
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        # each layer should do label assign separately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1

        assign_result_list = []
        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[self.num_proposals *
                                                idx_layer:self.num_proposals *
                                                (idx_layer + 1), :]
            score_layer = score[..., self.num_proposals *
                                idx_layer:self.num_proposals *
                                (idx_layer + 1), ]
            
            # import pdb; pdb.set_trace()
            if self.train_cfg.assigner.type == 'TransFusionHungarianAssigner3D':
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    gt_labels_3d,
                    score_layer,
                    self.train_cfg,
                )
            elif self.train_cfg.assigner.type == 'HeuristicAssigner':
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    None,
                    gt_labels_3d,
                    self.query_labels[batch_idx],
                )
            else:
                raise NotImplementedError

            assign_result_list.append(assign_result)
        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat(
                [res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]),
        )

        # 2. Sampling. Compatible with the interface of `PseudoSampler` in
        # mmdet.
        sampling_result = self.bbox_sampler.sample(assign_result_ensemble,
                                                   bboxes_tensor,
                                                   gt_bboxes_tensor)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        assert len(pos_inds) + len(neg_inds) == num_proposals

        # 3. Create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size
                                    ]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size
                                    ]).to(center.device)
        ids_targets = torch.zeros((num_proposals, ), device=center.device,
                                dtype=torch.long) - 1

        
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(
            num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression
        # and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[
                    sampling_result.pos_assigned_gt_inds]
                ids_targets[pos_inds] = torch.from_numpy(gt_ids_3d).to(pos_inds.device)[
                    sampling_result.pos_assigned_gt_inds]
                
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # compute dense heatmap targets
        device = labels.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]],
            dim=1).to(device)
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        feature_map_size = (grid_size[:2] // self.train_cfg['out_size_factor']
                            )  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1],
                                         feature_map_size[0])
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.train_cfg['out_size_factor']
            length = length / voxel_size[1] / self.train_cfg['out_size_factor']
            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width),
                    min_overlap=self.train_cfg['gaussian_overlap'])
                radius = max(self.train_cfg['min_radius'], int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = ((x - pc_range[0]) / voxel_size[0] /
                          self.train_cfg['out_size_factor'])
                coor_y = ((y - pc_range[1]) / voxel_size[1] /
                          self.train_cfg['out_size_factor'])

                center = torch.tensor([coor_x, coor_y],
                                      dtype=torch.float32,
                                      device=device)
                center_int = center.to(torch.int32)

                # original
                # draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius) # noqa: E501
                # NOTE: fix
                draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]],
                                      center_int[[0, 1]], radius)

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (
            labels[None],
            label_weights[None],
            bbox_targets[None],
            bbox_weights[None],
            ious[None],
            int(pos_inds.shape[0]),
            float(mean_iou),
            heatmap[None],
            ids_targets[None],  # assign ids for each proposal
        )

    def loss(self, batch_feats, 
                    gt_bboxes_3d, 
                    gt_labels_3d,
                    gt_ids_3d=None,
                    gt_bboxes_3d_future=None,
                    gt_labels_3d_future=None,
                    gt_ids_3d_future=None,
                    time_delay=None,
                    img_metas=None):
        """Loss function for CenterHead.

        Args:
            batch_feats (): Features in a batch.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        preds_dicts = self(batch_feats,
                           time_delay, 
                           img_metas)

        loss = self.loss_by_feat(preds_dicts, 
                                 gt_bboxes_3d, 
                                 gt_labels_3d,
                                 gt_ids_3d,
                                 gt_bboxes_3d_future,
                                 gt_labels_3d_future,
                                 gt_ids_3d_future,
                                 img_metas)

        return loss

    def loss_by_feat(self, preds_dicts: Tuple[List[dict]],
                    gt_bboxes_3d, 
                    gt_labels_3d,
                    gt_ids_3d,
                    gt_bboxes_3d_future,
                    gt_labels_3d_future,
                    gt_ids_3d_future,
                    img_metas,
                     **kwargs):
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
            assign_ids,
        ) = self.get_targets(
                            preds_dicts[0],
                            gt_bboxes_3d,
                            gt_labels_3d, 
                            gt_ids_3d,
                            img_metas,
                            )
        if self.with_delay:
            future_loss = self.delay_compensation.loss(
                preds_dicts[1],
                gt_bboxes_3d_future,
                gt_labels_3d_future,
                gt_ids_3d_future,
                assign_ids,
                labels,
                )
            
        if hasattr(self, 'on_the_image_mask'):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        # For Focal Loss

        # pred_heatmap = preds_dict['dense_heatmap'].float().reshape(-1, 1)
        # heatmap = torch.where(heatmap.eq(1), 0, 1).long().reshape(-1)

        # loss_heatmap = self.loss_heatmap(
        #     pred_heatmap,
        #     heatmap,
        #     avg_factor=max(heatmap.eq(0).float().sum().item(), 1),
        # )
        
        # compute heatmap loss
        pred_heatmap = clip_sigmoid(preds_dict['dense_heatmap']).float()
        heatmap = torch.where(heatmap.eq(1), 1, 0).long()
        loss_heatmap = self.loss_heatmap(
            pred_heatmap,
            heatmap.float(),
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        )

        import numpy as np
        import cv2
        
        pred_heatmap_np = np.concatenate([prob for prob in pred_heatmap[0].cpu().detach().numpy()])
        gt_heatmap_np = np.concatenate([prob for prob in heatmap[0].cpu().detach().numpy()])
        heatmap_np = (np.concatenate([pred_heatmap_np, gt_heatmap_np], axis = 1) * 255).astype(np.uint8) 
        cv2.imwrite('pred_heatmap.png', heatmap_np)
        loss_dict['loss_heatmap'] = loss_heatmap

        # compute loss for each layer
        for idx_layer in range(
                self.num_decoder_layers if self.auxiliary else 1):
            if idx_layer == self.num_decoder_layers - 1 or (
                    idx_layer == 0 and self.auxiliary is False):
                prefix = 'layer_-1'
            else:
                prefix = f'layer_{idx_layer}'

            layer_labels = labels[
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ].reshape(-1)
            layer_label_weights = label_weights[
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ].reshape(-1)

            layer_center = preds_dict['center'][..., idx_layer *
                                                self.num_proposals:(idx_layer +
                                                                    1) *
                                                self.num_proposals, ]
            layer_height = preds_dict['height'][..., idx_layer *
                                                self.num_proposals:(idx_layer +
                                                                    1) *
                                                self.num_proposals, ]
            layer_rot = preds_dict['rot'][..., idx_layer *
                                          self.num_proposals:(idx_layer + 1) *
                                          self.num_proposals, ]
            layer_dim = preds_dict['dim'][..., idx_layer *
                                          self.num_proposals:(idx_layer + 1) *
                                          self.num_proposals, ]
            preds = torch.cat(
                [layer_center, layer_height, layer_dim, layer_rot],
                dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            if 'vel' in preds_dict.keys():
                layer_vel = preds_dict['vel'][..., idx_layer *
                                              self.num_proposals:(idx_layer +
                                                                  1) *
                                              self.num_proposals, ]
                preds = torch.cat([
                    layer_center, layer_height, layer_dim, layer_rot, layer_vel
                ],
                                  dim=1).permute(
                                      0, 2,
                                      1)  # [BS, num_proposals, code_size]
            code_weights = self.train_cfg.get('code_weights', None)
            
            
            layer_bbox_weights = bbox_weights[:, idx_layer *
                                              self.num_proposals:(idx_layer +
                                                                  1) *
                                              self.num_proposals, :, ]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(  # noqa: E501
                code_weights)
            layer_bbox_targets = bbox_targets[:, idx_layer *
                                              self.num_proposals:(idx_layer +
                                                                  1) *
                                              self.num_proposals, :, ]

            layer_score = preds_dict['heatmap'][
                ...,
                idx_layer * self.num_proposals : (idx_layer + 1) * self.num_proposals,
            ]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(-1, self.num_classes)

            layer_loss_cls = self.loss_cls(
                layer_cls_score,
                layer_labels,
                layer_label_weights,
                avg_factor=max(num_pos, 1),
            )

            layer_loss_bbox = self.loss_bbox(
                preds,
                layer_bbox_targets,
                layer_reg_weights,
                avg_factor=max(num_pos, 1))

            loss_dict[f"{prefix}_loss_cls"] = layer_loss_cls
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox

            # loss_dict['vel_diff'] = torch.abs(preds[:, :, -2:] - layer_bbox_targets[:, :, -2:]).mean()
            # loss_dict['scale_diff'] = torch.abs(preds[:, :, 3:6] - layer_bbox_targets[:, :, 3:6]).mean()
            # loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

        loss_dict['matched_ious'] = layer_loss_bbox.new_tensor(matched_ious)
        
        if self.with_delay:
            for key in future_loss.keys():
                loss_dict["future_"+key] = future_loss[key]
       
        return loss_dict

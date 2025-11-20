#-------------------------------------------------------------------------------------------#
# Cyberc3 Lab, Shanghai Jiao Tong University
#-------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import copy
from models import builder
from core import build_bbox_coder, multi_apply
from typing import List, Tuple
from core.bbox import (AssignResult, PseudoSampler, build_bbox_coder)
from models.builder import MODELS
from core.bbox.iou_calculators.builder import IOU_CALCULATORS

from mmcv.cnn.bricks.transformer import build_feedforward_network
from models.dense_heads.centerpoint_head import SeparateHead
import math


class RelTemporalEncoding(nn.Module):
    """
    Implement the Temporal Encoding (Sinusoid) function.
    """

    def __init__(self, n_hid, RTE_ratio, max_len=100, dropout=0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(
            n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(
            n_hid)
        emb.requires_grad = False
        self.RTE_ratio = RTE_ratio
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)
        self.non_lin = nn.Sequential(
            nn.Conv1d(n_hid, n_hid, 1),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid),
            nn.Conv1d(n_hid, n_hid, 1),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid),
            nn.Conv1d(n_hid, n_hid, 1),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid),
        )

    def forward(self, x, t):
        # When t has unit of 50ms, rte_ratio=1.
        # So we can train on 100ms but test on 50ms
        return x + self.lin(self.emb((t * self.RTE_ratio * 2).long())).T

class RTE(nn.Module):
    def __init__(self, dim, RTE_ratio=2):
        super(RTE, self).__init__()
        self.RTE_ratio = RTE_ratio

        self.emb = RelTemporalEncoding(dim, RTE_ratio=self.RTE_ratio)

    def forward(self, x, dts):
        # x: (B,L,H,W,C)
        # dts: (B,L)
        rte_batch = []

        for b in range(x.shape[0]):
            rte_batch.append(
                self.emb(x[b], dts[b])) 
        return self.emb.non_lin(torch.stack(rte_batch, dim=0)) + x


class TimeCompensation(nn.Module):
    def __init__(self, embed_dims, RTE_ratio, num_decoder_layers, num_classes,
                conv_cfg=dict(type='Conv1d'),
                norm_cfg=dict(type='BN1d'),
                common_heads= {
                            "center" : [2, 2],
                            "rot" : [2, 2],
                            "vel": [2, 2],
                            "height": [1, 2],
                        },
                num_heatmap_convs = 2,
                bias = "auto",
                bbox_coder={
                    "type": 'TransFusionBBoxCoder',
                    "pc_range": [-102.4, -40, -5, 102.4, 40, 5],
                    # "post_center_range" : [-110, -50, -10, 110, 50, 10],
                    "score_threshold" : 0.0,
                    "out_size_factor" : 2,
                    "voxel_size" : (0.4, 0.4, 8),
                    "code_size" : 10
                },
                loss_cls = {
                    "type" : "FocalLoss", 
                    "use_sigmoid" : True, 
                    "gamma" : 2.0 ,
                    "alpha" : 0.25 ,
                    "loss_weight" : 2.0},
                loss_bbox = {
                    "type" : "L1Loss", 
                    "loss_weight" : 0.25},
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar')
        ):
        trans_decoder = dict(
                        type= 'TransformerDecoderLayer',
                        self_attn_cfg=dict(
                        embed_dims= embed_dims,
                        num_heads= 8, 
                        dropout= 0.1,),
                        cross_attn_cfg=dict( 
                        embed_dims= embed_dims,
                        num_heads= 8,
                        dropout= 0.1,),
                        ffn_cfg=dict(
                            embed_dims= embed_dims,
                            feedforward_channels= embed_dims,
                            num_fcs= 2,
                            ffn_drop= 0.1,
                            act_cfg=dict(
                            type= 'ReLU', 
                            inplace= True,)),
                        pos_encoding_cfg=dict( 
                        input_channel= 2, 
                        num_pos_feats= embed_dims) 
                    )
        super(TimeCompensation, self).__init__()
        self.RTE = RTE(embed_dims, RTE_ratio)

        self.num_classes = num_classes
        self.num_decoder_layers = num_decoder_layers

        self.get_prior_embedding = nn.Linear(3, embed_dims)

        self.add_proir_info = nn.Sequential(
            nn.Linear(embed_dims + embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
        )

        self.decoder = nn.ModuleList()
        self.prediction_heads = nn.ModuleList()
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.iou_calculator = IOU_CALCULATORS.build(iou_calculator)
        self.bbox_sampler = PseudoSampler()

        self.num_proposals = 200

        for i in range(self.num_decoder_layers):
            self.decoder.append(MODELS.build(trans_decoder))
            heads = deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(
                SeparateHead(
                    embed_dims,
                    heads,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=bias,
                ))
            
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5]

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)

    def forward(self, query_feat, dts, query_pose, predict_boxes):
        predict_boxes = predict_boxes.permute(0, 2, 1) # [bs, num_proposals, 9]
        query_feat = self.RTE(query_feat, dts)

        ret_dicts = []

        for i in range(self.num_decoder_layers):
            
            # query_feat = self.decoder[i](
            #     query_feat,
            #     key=query_feat,
            #     query_pos=query_pose,
            #     key_pos=query_pose,
            #     )

            # compensated query pose
            pos_offset_bias = predict_boxes[:, -2:] * dts.unsqueeze(-1).permute(0, 2, 1) * 0.1 /( torch.FloatTensor(self.bbox_coder.voxel_size[:2]).to(dts.device)[None, : ,None]  * self.bbox_coder.out_size_factor)# [bs, 2, num_proposals]
            yaw_bias = predict_boxes[:, 6:7]

            bias_priors = self.get_prior_embedding(torch.cat([pos_offset_bias, yaw_bias], dim=1).permute(0, 2, 1).float()) # [bs, num_proposals, embed_dims]

            query_feat = (self.add_proir_info(torch.cat([query_feat.permute(0, 2, 1), bias_priors], dim=-1)) + bias_priors).permute(0, 2, 1)

            predict_bias = self.prediction_heads[i](query_feat)  # [bs, num_instance, embed_dims]
            predict_bias['center'] = predict_bias['center'] + query_pose.permute(0, 2, 1)  # [bs, num_instance, 2]
            
            query_pose = predict_bias['center'].detach().clone().permute(0, 2, 1)
  
            predict_bias['dim'] = predict_boxes[:, 3:6].log()
            
            predict_bias['query_feat'] = query_feat
            predict_bias['query_pose'] = query_pose
     
            ret_dicts.append(predict_bias)

        return ret_dicts

    def get_targets(self, 
                    future_gt_bboxes_3d: List[torch.Tensor],
                    future_gt_labels_3d: List[torch.Tensor],
                    future_gt_ids_3d: List[int],
                    preds_dict: List[dict],
                    current_ids : list,
                    current_labels : list = None):
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
        for batch_idx in range(len(future_gt_bboxes_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                preds = []
                for i in range(self.num_decoder_layers):
                    pred_one_layer = preds_dict[i][key][batch_idx:batch_idx +
                                                        1]
                    preds.append(pred_one_layer)
                pred_dict[key] = torch.cat(preds)
            list_of_pred_dict.append(pred_dict)
        assert len(future_gt_bboxes_3d) == len(list_of_pred_dict)
        res_tuple = multi_apply(
            self.get_targets_single,
            future_gt_bboxes_3d,
            future_gt_labels_3d,
            future_gt_ids_3d,
            list_of_pred_dict,
            current_ids,
            current_labels,
            np.arange(len(future_gt_bboxes_3d)),
        )
        labels =  torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])

        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
        )

    def get_targets_single(self,
                            future_gt_bboxes_3d,
                            future_gt_labels_3d,
                            future_gt_ids_3d, 
                            future_preds_dict,
                            current_ids,
                            current_labels, 
                            batch_idx):
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
        # 1. Assignment by ids
        valid_current_ids = current_ids[current_ids != -1]
        min_id = min(min(future_gt_ids_3d), min(valid_current_ids))
        max_id = max(max(future_gt_ids_3d), max(valid_current_ids))
        future_gt_id_2_idx = torch.zeros(max_id - min_id + 1, device=current_ids.device).long()

        future_gt_id_2_idx[future_gt_ids_3d - min_id] = torch.arange(len(future_gt_ids_3d), device=current_ids.device) + 1
        candidate_gt_idx = torch.zeros_like(current_ids, device=current_ids.device)
        candidate_gt_idx[current_ids != -1] = future_gt_id_2_idx[(valid_current_ids - min_id).long()]
        candidate_gt_labels = torch.zeros_like(current_ids, device=current_ids.device) - 1
        candidate_gt_labels[current_ids != -1] = future_gt_labels_3d[(future_gt_id_2_idx[(valid_current_ids - min_id).long()] - 1).long()]

        # 2 decode the prediction dict
        num_proposals = future_preds_dict['center'].shape[-1]

        # get pred boxes, carefully ! don't change the network outputs
        score = copy.deepcopy(future_preds_dict['heatmap'].detach())
        center = copy.deepcopy(future_preds_dict['center'].detach())
        height = copy.deepcopy(future_preds_dict['height'].detach())
        dim = copy.deepcopy(future_preds_dict['dim'].detach())
        rot = copy.deepcopy(future_preds_dict['rot'].detach())

        if 'vel' in future_preds_dict.keys():
            vel = copy.deepcopy(future_preds_dict['vel'].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(
            score, rot, dim, center, height,
            vel)  # decode the prediction to real world metric bbox
        future_gt_bboxes_3d = future_gt_bboxes_3d.tensor.to(score.device)
        # each layer should do label assign separately.

        bbox_targets = []
        bbox_weights = []
        label_weights = []
        ious = []
        pos_indses = []
        labels = []
        for idx_layer in range(self.num_decoder_layers):
            bboxes_tensor = boxes_dict[idx_layer]['bboxes']
            
            max_overlaps = torch.zeros_like(candidate_gt_idx).to(score.device).float()
            matched_indices = torch.where(candidate_gt_idx > 0)
            matched_iou = self.iou_calculator(
                future_gt_bboxes_3d[candidate_gt_idx.long() - 1],
                bboxes_tensor[matched_indices]).diag()
            max_overlaps[matched_indices] = matched_iou
    
            assign_result = AssignResult(
                num_gts=sum(candidate_gt_idx != 0),
                gt_inds=candidate_gt_idx,
                max_overlaps=max_overlaps,
                labels= candidate_gt_labels,
            )
      
            # 2. Sampling. Compatible with the interface of `PseudoSampler` in
            # mmdet.
            sampling_result = self.bbox_sampler.sample(assign_result,
                                                    bboxes_tensor,
                                                    future_gt_bboxes_3d)
            pos_inds = sampling_result.pos_inds
            neg_inds = sampling_result.neg_inds

            assert len(pos_inds) + len(neg_inds) == num_proposals

            # 3. Create target for loss computation
            bbox_target = torch.zeros([num_proposals, self.bbox_coder.code_size
                                        ]).to(center.device)
            bbox_weight = torch.zeros([num_proposals, self.bbox_coder.code_size
                                        ]).to(center.device)
            iou = assign_result.max_overlaps
            iou = torch.clamp(iou, min=0.0, max=1.0)
          
            label_weight = bboxes_tensor.new_zeros(
                num_proposals, dtype=torch.long) + 1

            # both pos and neg have classification loss, only pos has regression
            # and iou loss
            if len(pos_inds) > 0:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_gt_bboxes)

                bbox_target[pos_inds, :] = pos_bbox_targets
                bbox_weight[pos_inds, :] = 1.0

            mean_iou = iou[pos_inds].sum() / max(len(pos_inds), 1)

            # append to the list
            bbox_targets.append(bbox_target)
            bbox_weights.append(bbox_weight)
            labels.append(current_labels)
            label_weights.append(label_weight)
            ious.append(iou)
            pos_indses.append(pos_inds)

        labels = torch.cat(labels, dim=0)  # [1, num_proposals * num_layers]
        label_weights = torch.cat(label_weights, dim=0)  # [1, num_proposals * num_layers]
        bbox_targets = torch.cat(bbox_targets, dim=0)  # [1, num_proposals, num_layers, code_size]
        bbox_weights = torch.cat(bbox_weights, dim=0)  # [1, num_proposals, num_layers, code_size]
        ious = torch.cat(ious, dim=0)  # [1, num_proposals, num_layers]
        pos_indses = torch.cat(pos_indses, dim=0)  # [num_pos, ]

        return (
            labels[None],
            label_weights[None],
            bbox_targets[None],
            bbox_weights[None],
            ious[None],
            int(pos_inds.shape[0]),
            float(mean_iou),
        )

    def loss(self, preds_dicts, future_gt_bboxes_3d, future_gt_labels_3d, future_gt_ids, current_ids, current_labels):
        """Loss function for CenterHead.

        Args:
            batch_feats (): Features in a batch.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        loss = self.loss_by_feat(preds_dicts, future_gt_bboxes_3d, future_gt_labels_3d, 
                                 future_gt_ids, current_ids, current_labels)

        return loss

    def loss_by_feat(self, preds_dicts: Tuple[List[dict]],
                     future_gt_bboxes_3d, 
                     future_gt_labels_3d, 
                     future_gt_ids, 
                     current_ids,
                     current_labels,
                     **kwargs):

        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
        ) = self.get_targets(
                            future_gt_bboxes_3d,
                            future_gt_labels_3d,
                            future_gt_ids,
                            preds_dicts, 
                            current_ids,
                            current_labels
                            )

        preds_dict = {}
        loss_dict = dict()

        for idx_layer in range(self.num_decoder_layers):
            one_layer_preds = preds_dicts[idx_layer]
            for key in one_layer_preds.keys():
                if key not in preds_dict.keys():
                    preds_dict[key] = []
                preds_dict[key].append(one_layer_preds[key])

        # stack the predictions
        for key in preds_dict.keys():
            preds_dict[key] = torch.cat(preds_dict[key], dim=2)

        # compute loss for each layer
        for idx_layer in range(self.num_decoder_layers):
            if idx_layer == self.num_decoder_layers - 1:
                prefix = 'layer_-1'
            else:
                prefix = f'layer_{idx_layer}'

            layer_labels = labels[..., idx_layer *
                                  self.num_proposals:(idx_layer + 1) *
                                  self.num_proposals, ].reshape(-1)
            layer_label_weights = label_weights[
                ..., idx_layer * self.num_proposals:(idx_layer + 1) *
                self.num_proposals, ].reshape(-1)
            
            layer_score = preds_dict['heatmap'][..., idx_layer *
                                                self.num_proposals:(idx_layer +
                                                                    1) *
                                                self.num_proposals, ]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(
                -1, self.num_classes)

            layer_loss_cls = self.loss_cls(
                layer_cls_score.float(),
                layer_labels,
                layer_label_weights,
                avg_factor=max(num_pos, 1),
            )

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
            code_weights = self.code_weights
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
            layer_loss_bbox = self.loss_bbox(
                preds,
                layer_bbox_targets,
                layer_reg_weights,
                avg_factor=max(num_pos, 1))

            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox
            # loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

        loss_dict['matched_ious'] = layer_loss_cls.new_tensor(matched_ious)

        return loss_dict

        


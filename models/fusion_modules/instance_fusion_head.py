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
from core.bbox import (AssignResult, PseudoSampler,
                                       build_assigner, build_bbox_coder,
                                       build_sampler)
from models.builder import MODELS
from models.utils.transformation_utils import transform_boxes_torch, rotation_matrix_to_rpy
from models.builder import HEADS
from models.utils.utilities import compute_iou_rotated_boxes
from models.utils.transformer import PositionEncodingLearned
from models.dense_heads.centerpoint_head import SeparateHead
from models.modules.rope_self_attention import ContinuousRoPEAttention
from mmcv.cnn.bricks.transformer import build_feedforward_network


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def pairwise_assignment(track_array1, track_array2, cost_threshold):
    """
        track_array is [x, y, z, l, w, h, yaw, global_id, score]
    """
    # assign the track
    if min(len(track_array1), len(track_array2)) > 0:
        # calculate the cost matrix
        track1_position = track_array1[:, None, :2]
        track2_position = track_array2[None, :, :2]
        cost_matrix = torch.norm(track1_position - track2_position, p=2, dim=2)

        a = cost_matrix < cost_threshold
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = torch.stack(torch.where(a), dim=1).cpu().detach().numpy()
        else:
            matched_indices = linear_assignment(cost_matrix.cpu().detach().numpy())
    else:
        matched_indices = np.zeros((0,2))

    
    matched_indices = matched_indices.astype(np.int32)

    # assign the unmatched track
    unmatched_track1 = []
    unmatched_track2 = []
    for t, trk in enumerate(track_array1):
        if(t not in matched_indices[:,0]):
            unmatched_track1.append(t)
    
    for t, trk in enumerate(track_array2):
        if(t not in matched_indices[:,1]):
            unmatched_track2.append(t)  

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(cost_matrix[m[0], m[1]] > cost_threshold):
            unmatched_track1.append(m[0])
            unmatched_track2.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_track1), np.array(unmatched_track2)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

@HEADS.register_module()
class InstanceFusionHead(nn.Module):
    def __init__(self, 
                 fusion_distance, 
                 embed_dims=256,
                 common_heads= {
                            "reg" : [2, 2],
                            "height" : [1, 2],
                            "dim" : [3, 2],
                            "rot" : [2, 2],
                            "vel": [2, 2]
                        },
                num_heatmap_convs=2,
                loss_bbox=dict(type='L1Loss', reduction='mean'),
                loss_cls=dict(type='FocalLoss', reduction='mean'),
                num_classes = 3,
                conv_cfg=dict(type='Conv1d'),
                norm_cfg=dict(type='BN1d'),
                num_decoder_layers=1,
                bias='auto',
                # others
                train_cfg=None,
                test_cfg=None,
                bbox_coder=None
                 ):
        super(InstanceFusionHead, self).__init__()
        self.fusion_distance = fusion_distance
        self.embed_dims = embed_dims 
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)

        # reference_points ---> pos_embed
        self.get_prior_embedding = nn.Sequential(
            nn.Linear(2, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),)


        self.add_proir_info = nn.Sequential(
            nn.Linear(embed_dims + embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
        )

        # cross-agent feature alignment
        self.rotation_embed, rotation_embed_dim = get_embedder(10, input_dims=3)


        self.cross_agent_align_pos =  nn.Sequential(
            nn.Linear(embed_dims + rotation_embed_dim, embed_dims),

        )
        # nn.Linear(embed_dims + rotation_embed_dim, embed_dims)

        self.prior_add = nn.Sequential(
            nn.Linear(embed_dims + embed_dims, embed_dims),
            nn.ReLU(inplace=False),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
        )
        
        res_fc_layer = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=False),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
        )
        
        self.res_layers = nn.ModuleList(
            deepcopy(res_fc_layer) for _ in range(1))

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
        
        self.num_classes = num_classes
        self.num_decoder_layers = num_decoder_layers

        # Prediction Head
        self.decoder = nn.ModuleList()
        self.prediction_heads = nn.ModuleList()
       

        for i in range(self.num_decoder_layers):
            self.decoder.append(MODELS.build(trans_decoder))
            heads = deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(
                SeparateHead(
                    self.embed_dims,
                    heads,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=bias,
                ))
            # import pdb; pdb.set_trace()

        self.with_velocity = 'vel' in common_heads.keys()

        self.sampling = False

        self._init_assigner_sampler()
        self.bbox_coder = build_bbox_coder(bbox_coder)

        # parameter initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def set_head_parameters(self, trained_head):
        self.prediction_heads.load_state_dict(trained_head.state_dict())
        self.prediction_heads.train()

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

    def _query_matching(self, 
                        agent_detection_bboxes, 
                        agent_detection_class,
                        agent_localization,):
        """
        agent_detection_bboxes: [num_agent, num_instance, 9] (xyz)
        agent_localization: [num_agent, 3] (xyz)
        """
        prediction_results = {}

        agent_num = len(agent_detection_bboxes)

        # agent distance  mask
        agent_localization = agent_localization.reshape(-1, 2)
        distance = torch.cdist(agent_localization, agent_localization, p=2)

        connection_graph = torch.triu(distance < self.fusion_distance, 1)

        edges = np.where(connection_graph.cpu().detach().numpy())
        agent_ids = np.linspace(0, agent_num-1, agent_num, dtype=int)
        
        for agent_idx in range(agent_num):
            for idx in range(len(agent_detection_bboxes[agent_idx])):
                prediction_results[(agent_idx, idx)] = {(agent_idx, idx)}

        for i, j in zip(edges[0], edges[1]):
            first_agent_idx = agent_ids[i]
            second_agent_idx = agent_ids[j]

            first_agent_bboxes = agent_detection_bboxes[first_agent_idx]
            second_agent_bboxes = agent_detection_bboxes[second_agent_idx]

            first_agent_cls = agent_detection_class[first_agent_idx]
            second_agent_cls = agent_detection_class[second_agent_idx]

            for cls in range(self.num_classes):
                first_agent_bboxes_cls = first_agent_bboxes[first_agent_cls == cls]
                second_agent_bboxes_cls = second_agent_bboxes[second_agent_cls == cls]

                first_agent_bboxes_ind = torch.where(first_agent_cls == cls)[0].cpu().detach().numpy()
                second_agent_bboxes_ind = torch.where(second_agent_cls == cls)[0].cpu().detach().numpy()

                
                matched, _, _ =\
                        pairwise_assignment(first_agent_bboxes_cls.clone(),
                                            second_agent_bboxes_cls.clone(), 
                                            1)
                
                #  将所有相近的检测结果进行融合
                for m in matched:
                    box_1 = (first_agent_idx, first_agent_bboxes_ind[m[0]])
                    box_2 = (second_agent_idx, second_agent_bboxes_ind[m[1]])
                    union_set = prediction_results[box_1] | prediction_results[box_2] 
                    prediction_results[box_1] = union_set
                    prediction_results[box_2] = union_set

        visited = dict(zip(prediction_results.keys(), [False] * len(prediction_results)))
        # 所有的匹配结果
        cluster_idx = 0
        cluster_res = {}
        
        for box_idx in prediction_results:
            if visited[box_idx]:
                continue
            relevant_box_idxes = prediction_results[box_idx]
            cluster_res[cluster_idx] = []
            for agent_idx, box_id in relevant_box_idxes:
                cluster_res[cluster_idx].append((agent_idx, box_id))
                visited[(agent_idx, box_id)] = True
            cluster_idx += 1
        return cluster_res
    
    def forward_single(self, agent_results, agent_localization, agent2center_rt):
        """
        Query-based cross-agent interaction: only update ref_pts and query.
        agent_results: List of Instances from other agents, containing
                        bboxes, labels, scores and query features.
        agent2center_rt: calibration parameters from every agent to center
        """
        # TODO(cxy) : Add Time Delay Compensation

        # Align agent results to the same coordinate system
        aligned_agent_bboxes = []
        aligned_agent_bboxes_cls = []
        aligned_agent_feats = []
        aligned_agent_scores = []
        aligned_agent_pose = []


        agent2center_rt = agent2center_rt.float()

        for agent_idx, agent_result in enumerate(agent_results):

            aligned_agent_scores.append(agent_result['scores_3d'])
            aligned_agent_bboxes.append(
                transform_boxes_torch(agent_result['boxes_3d'],
                                agent2center_rt[agent_idx]))
            
            aligned_agent_bboxes_cls.append(agent_result['labels_3d'])
            
            agent_feat = agent_result['query_3d']
            agent_pose = agent_result['query_pose_3d']
            # Only remain yaw and translation
            roll, pitch, yaw = rotation_matrix_to_rpy(agent2center_rt[agent_idx][:3,:3].cpu().detach().numpy())
            revised_rt = deepcopy(agent2center_rt[agent_idx])
            revised_rt[:3, :3] = torch.tensor([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ]).to(agent2center_rt[agent_idx].device).float()

            print(f"agent {agent_idx} roll: {roll}, pitch: {pitch}, yaw: {yaw}")
            agent_pose = torch.cat(
                    [agent_pose, torch.zeros_like(agent_pose[:, :1]).to(agent_pose.device), torch.ones_like(agent_pose[:, :1]).to(agent_pose.device)], dim=-1)
            
            agent_pose = torch.matmul(revised_rt, agent_pose.T).T[:, :2]
            aligned_agent_pose.append((agent_pose -  agent_pose.new_tensor(self.bbox_coder.pc_range[:2])) / self.bbox_coder.out_size_factor / agent_pose.new_tensor(self.bbox_coder.voxel_size[:2]))

            # get yaw from agent2center_rat 
           
            vehicle_rot = (torch.FloatTensor([roll, pitch, yaw]).reshape(1,3).repeat(agent_feat.shape[0], 1).to(agent_feat.device) % (2*np.pi)) / (2*np.pi)
            
            vehicle_rot = self.rotation_embed(vehicle_rot) # [num_query, embed_dims]
            agent_feat = self.cross_agent_align_pos(torch.cat([agent_feat, vehicle_rot], dim=-1))  # [num_query, embed_dims]
            aligned_agent_feats.append(agent_feat)
        

        # # assign the matched detection results
        match_result = self._query_matching(aligned_agent_bboxes,
                                            aligned_agent_bboxes_cls,
                                            agent_localization)

        all_post_fusion_instance = []
        all_fused_instance_feats = []
        all_instance_anchors = []
        all_instance_cls = []
        all_fused_instance_num = []

        bias_priors = []
        feats = []

        # TODO: Here we will only use one feature instead of all features
        for cluster_idx in match_result:
            instances = match_result[cluster_idx]
            instance_cls = aligned_agent_bboxes_cls[instances[0][0]][instances[0][1]]
            instance_score = torch.stack([aligned_agent_scores[agent_idx][box_idx] 
                                                            for agent_idx, box_idx in instances])

            instance_pose = torch.stack([aligned_agent_pose[agent_idx][box_idx]
                                                            for agent_idx, box_idx in instances])
            cluster_box = torch.stack([aligned_agent_bboxes[agent_idx][box_idx]
                                                for agent_idx, box_idx in instances])

            mean_pose = instance_pose.mean(dim=0, keepdim=True)  # [1, 2]

            box_prior = cluster_box.clone()[:, :2]
            box_prior = (box_prior - box_prior.new_tensor(self.bbox_coder.pc_range[:2])) / self.bbox_coder.out_size_factor / box_prior.new_tensor(self.bbox_coder.voxel_size[:2])

            box_prior -= mean_pose  # center to gravity center
            bias_priors.append(box_prior)  # [num_instance, 7]

            cluster_feat = torch.stack([aligned_agent_feats[agent_idx][box_idx]
                                                for agent_idx, box_idx in instances])

            feats.append(cluster_feat)
            
            all_post_fusion_instance.append(cluster_box.mean(dim=0, keepdim=True))  # [1, 9]
            all_instance_anchors.append(mean_pose)  # [1, 2]
            all_instance_cls.append(instance_cls)
        
        bias_priors = torch.cat(bias_priors, axis=0).to(cluster_feat.device)
        # bias_priors[:, 3:6] = bias_priors[:, 3:6].log()  # l, w, h
        bias_priors = self.get_prior_embedding(bias_priors)
        feats = torch.cat(feats, dim=0)  # [num_cluster, num_instance, embed_dims]
        feats = self.add_proir_info(torch.cat([feats, bias_priors], dim=-1)) # + bias_priors 
        
        all_instance_anchors = torch.cat(all_instance_anchors, dim=0)  # [num_cluster, 2]
        # # DEBUG
        # import cv2
        # boxes = np.concatenate(post_fusion_box, axis =0)
        # # img_meta_vehicles = img_metas[0]
        # # img_files = [ img_meta[len(img_meta) - 1]['img_filename'] for img_meta in img_meta_vehicles ]
        # x_max, y_max = boxes.max(axis = 0)[:2] + 5
        # x_min, y_min = boxes.min(axis = 0)[:2] - 5

        # # print(f"Image files: {img_files}")
        # # print((x_min, x_max), (y_min, y_max))
        # # print(f"vehicle num is {cav_num[0]}")

        # res = 0.1
        # canvas = np.zeros((int((y_max - y_min) / res), int((x_max - x_min) / res)))
        # for box in boxes:
        #     xc, yc,_, l, w,_, yaw = box[:7]
        #     x_idx, y_idx = int((xc - x_min) / res), int((yc - y_min) / res)
        #     l, w = int(l / res), int(w / res)
        #     # draw rectangle with rotation
        #     rect = cv2.boxPoints(((x_idx, y_idx), (l, w), np.degrees(yaw)))
        #     rect = np.int32(rect)
        #     cv2.fillConvexPoly(canvas, rect, 1)
        # cv2.imwrite('canvas_pred.png', (canvas * 255).astype(np.uint8))

        # feats = feats[None]
        # all_instance_anchors = all_instance_anchors[None]
        # for i in range(self.num_attn_layer):
        #     feats = self.multi_head_attn[i]['rope_attn'](feats, all_instance_anchors)
        #     feats = self.multi_head_attn[i]['ffn'](feats)
        #     feats = self.multi_head_attn[i]['ln'](feats)
        # feats = feats[0]

        start_idx = 0
        for cluster_idx in match_result:
            instances = match_result[cluster_idx]

            cluster_feat = feats[start_idx:start_idx + len(instances), :]
            start_idx += len(instances)
            
            cluster_feat = torch.mean(cluster_feat, dim=0, keepdim=True)
            all_fused_instance_feats.append(cluster_feat)
            all_fused_instance_num.append(len(instances))
        

        all_fused_instance_feats = torch.cat(all_fused_instance_feats, dim=0)  # [num_instance, embed_dims]
        all_post_fusion_instance = torch.cat(all_post_fusion_instance, dim=0)

        return [all_fused_instance_feats, all_instance_anchors, all_post_fusion_instance, torch.stack(all_instance_cls), len(all_fused_instance_feats), torch.LongTensor(all_fused_instance_num)]

    
        
    def forward(self, agent_results, agent_localization, agent2center_rt):
        """
        agent_results: List of Instances from other agents, containing
                        bboxes, labels, scores and query features.
                        shape : [batch_size, num_agent, num_instance, embed_dims]
        agent_localization: List of localization of each agent, containing
                            translation and rotation.
                            shape: [batch_size, num_agent, 6]
        agent2center_rt: calibration parameters from every agent to center
                    shape: [batch_size, num_agent, 4, 4]
        threshold: the threshold to filter out low confidence detection results 
        """
        bs = len(agent_results)
        all_fused_instance_feats, all_instance_anchors, all_post_fusion_boxes, all_instance_cls, all_instance_num, all_fused_instance_num\
            = multi_apply(self.forward_single, 
                           agent_results,
                           agent_localization,
                           agent2center_rt)
        
        max_instance_num = max(all_instance_num) +  1
        valid_mask = torch.zeros((bs* 8 , max_instance_num, max_instance_num), dtype=torch.bool).to(all_fused_instance_feats[0].device)
        key_padding_mask = torch.zeros((bs, max_instance_num), dtype=torch.bool).to(all_fused_instance_feats[0].device)
        for batch_idx, (instance_feats, instance_box, instance_anchors, instance_cls, instance_num, fused_instance_num) in\
              enumerate(zip(
                    all_fused_instance_feats,
                    all_post_fusion_boxes,
                    all_instance_anchors,
                    all_instance_cls,
                    all_instance_num,
                    all_fused_instance_num
                )):
            # pad the instance features, anchors and boxes to the same size
            valid_mask[batch_idx * 8 : (batch_idx + 1) * 8 , :instance_num, :instance_num] = True
            key_padding_mask[batch_idx, :instance_num] = True
            instance_feats = torch.cat([instance_feats, torch.zeros((max_instance_num - instance_num, instance_feats.shape[1]), device=instance_feats.device)], dim=0)
            instance_anchors = torch.cat([instance_anchors, torch.zeros((max_instance_num - instance_num, instance_anchors.shape[1]), device=instance_anchors.device)], dim=0)
            instance_box = torch.cat([instance_box, torch.zeros((max_instance_num - instance_num, instance_box.shape[1]), device=instance_box.device)], dim=0)
            instance_cls = torch.cat([instance_cls, torch.zeros((max_instance_num - instance_num,), device=instance_cls.device) - 1], dim=0)
            fused_instance_num = torch.cat([fused_instance_num, torch.zeros((max_instance_num - instance_num,), device=fused_instance_num.device)], dim=0)
            
            all_fused_instance_feats[batch_idx] = instance_feats
            all_instance_anchors[batch_idx] = instance_anchors
            all_post_fusion_boxes[batch_idx] = instance_box
            all_instance_cls[batch_idx] = instance_cls
            all_fused_instance_num[batch_idx] = fused_instance_num

        all_fused_instance_feats = torch.stack(all_fused_instance_feats, dim=0)  # [bs, num_instance, embed_dims]
        # for i in range(len(self.res_layers)):
        #     all_fused_instance_feats = self.res_layers[i](all_fused_instance_feats) + all_fused_instance_feats

        all_instance_anchors = torch.stack(all_instance_anchors, dim=0)  # [bs, num_instance, 2]
        all_post_fusion_boxes = torch.stack(all_post_fusion_boxes, dim=0)  # [bs, num_instance, 9]
        all_instance_cls = torch.stack(all_instance_cls, dim=0)  # [bs, num_instance]
        all_fused_instance_num = torch.stack(all_fused_instance_num).to(all_instance_cls.device)  # [bs]

         # predict new bias based on the fused features
        ret_dicts = []

        all_fused_instance_feats = all_fused_instance_feats.permute(0, 2, 1)  # [bs, embed_dims, num_instance]
        for i in range(self.num_decoder_layers):

            all_fused_instance_feats = self.decoder[i](
                all_fused_instance_feats,
                key=all_fused_instance_feats,
                query_pos=all_instance_anchors,
                key_pos=all_instance_anchors,
                self_attn_mask=valid_mask,
                cross_attn_mask=valid_mask,
                key_padding_mask=key_padding_mask,
                )

            predict_bias = self.prediction_heads[i](all_fused_instance_feats)  # [bs, num_instance, embed_dims]
            predict_bias['center'] = predict_bias['center'] + all_instance_anchors.permute(0, 2, 1)  # [bs, num_instance, 2]
            ret_dicts.append(predict_bias)

        ret_dicts = ret_dicts[0]
        ret_dicts['post_fusion'] = all_post_fusion_boxes  # [bs, num_instance, 9]
        ret_dicts['instance_in_scenario'] = all_instance_num  # [bs]
        ret_dicts['instance_cls'] = all_instance_cls  # [bs, num_instance]
        ret_dicts['anchor'] = all_instance_anchors.permute(0, 2, 1)  # [bs, num_instance, 2]
        ret_dicts['instance_num'] = all_fused_instance_num  # [bs, num_instance]

        batch_ret = []
        for bs_idx in range(bs):
            ret_dict = {}
            for key in ret_dicts:
                ret_dict[key] = ret_dicts[key][bs_idx]
            batch_ret.append(ret_dict)
    
        return batch_ret

    def predict(self, agent_results, agent_localization, agent2center_rt, metas=None):
        rets = []
        preds_dicts = self(agent_results, agent_localization, agent2center_rt)
    
        box_type = metas[0][0][0]['box_type_3d']

        for bs_idx in range(len(preds_dicts)):
            pred_result = {
                "bboxes" : [],
                "scores" : [],
                "labels" : [],
            }
  
            bias = preds_dicts[bs_idx]
            score = bias['heatmap'].sigmoid()
            center = bias['center']
            height = bias['height']
            dim = bias['dim']
            rot = bias['rot']
            vel = bias['vel'] if 'vel' in bias else None
            num_instance = bias['instance_in_scenario']
            post_fusion_box = bias['post_fusion']

            boxes_dict = self.bbox_coder.decode(
                score[None],  rot[None], dim[None], center[None], height[None], vel[None])  # decode the prediction to real world metric bbox
            for key in boxes_dict[0]:
                pred_result[key].append(boxes_dict[0][key][:num_instance])
            
            for key in pred_result:
                pred_result[key] = torch.cat(pred_result[key], dim=0)

            rets.append(
                        [box_type(pred_result['bboxes'], box_dim=pred_result['bboxes'].shape[-1]),
                        pred_result['scores'],
                        pred_result['labels'].int(),
                        post_fusion_box[:num_instance],
                        bias['instance_num'][:num_instance]
                        ]
                    )

        return rets
    
    def get_targets(self, gt_bboxes_3d, gt_labels_3d, 
                    preds_dict: List[dict]):
        """Generate targets.

        How each output is transformed:

            Each nested list is transposed so that all same-index elements in
            each sub-list (1, ..., N) become the new sub-lists.
                [ [a0, a1, a2, ... ], [b0, b1, b2, ... ], ... ]
                ==> [ [a0, b0, ... ], [a1, b1, ... ], [a2, b2, ... ] ]

            The new transposed nested list is converted into a list of N
            tensors generated by concatenating tensors in the new sub-lists.
                [ tensor0, tensor1, tensor2, ... ]

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including
                    the following results in order.

                    - list[torch.Tensor]: Ground truth boxes.
                    - list[torch.Tensor]: Indexes indicating the
                        position of the valid boxes.
                    - list[torch.Tensor]: Masks indicating which
                        boxes are valid.
        """
        res_tuple = multi_apply(
            self.get_targets_single,
            gt_bboxes_3d,
            gt_labels_3d,
            preds_dict,
        )

        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])

        matched_ious = torch.mean(torch.stack(res_tuple[6], dim =0))
        positive_mask = torch.cat(res_tuple[7], dim=0)
        bbox_targets_original = torch.cat(res_tuple[8], dim=0)  # [N, 9]
        # valid_mask = torch.cat(res_tuple[8], dim=0)  # [
        valid_mask = torch.cat(res_tuple[9], dim=0)

        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            positive_mask,
            bbox_targets_original,
            valid_mask
        )

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including
                the following results in order.

                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes
                    are valid.
        """



        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(preds_dict['heatmap'].device)
        
        assign_threshold = {
            0 : 1.0,  # car
            1 : 0.5, # pedestrian
            2 : 1.0, # Truck  
        }
        
        all_input_num = preds_dict['post_fusion'].shape[0]
        all_num_pos_inds = 0
        all_bbox_targets = torch.zeros([all_input_num, self.bbox_coder.code_size
                                        ]).to(gt_bboxes_tensor.device)
        all_bbox_targets_original = torch.zeros([all_input_num, 9
                                        ]).to(gt_bboxes_tensor.device)
        all_bbox_weights = torch.zeros([all_input_num, self.bbox_coder.code_size
                                    ]).to(gt_bboxes_tensor.device)
        all_ious = gt_bboxes_tensor.new_zeros(all_input_num, dtype=torch.float)
        all_labels = gt_bboxes_tensor.new_zeros(all_input_num, dtype=torch.long)
        all_label_weights = gt_bboxes_tensor.new_zeros(
            all_input_num, dtype=torch.long)
        all_positive_mask = gt_bboxes_tensor.new_zeros(all_input_num, dtype=torch.bool)


        for cls in range(self.num_classes):
            class_mask = preds_dict['instance_cls'] == cls

            # class bbox
            cls_gt_bboxes_3d = gt_bboxes_tensor[gt_labels_3d == cls]
            
            # 1. Assignment
            num_proposals = class_mask.sum().item()

            # if num_proposals < len(cls_gt_bboxes_3d):
            #     print(f"Warning: The number of anchors {num_proposals} is less than the number of ground truth boxes {len(cls_gt_bboxes_3d)} in class {cls}. ")

            bboxes_tensor = preds_dict['post_fusion'][class_mask]

            assign_result_list = []
            if self.train_cfg.assigner.type == 'InstanceFusionHungarianAssigner3D':
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor,
                    cls_gt_bboxes_3d,
                    self.train_cfg,
                    assign_threshold[cls],
                )
            else:
                raise NotImplementedError(
                    f"Assign type {self.train_cfg.assigner.type} is not implemented"
                )
            assign_result_list.append(assign_result)
            # 2. Sampling. Compatible with the interface of `PseudoSampler` in
            # mmdet.

            assign_result_ensemble = AssignResult(
                    num_gts=sum([res.num_gts for res in assign_result_list]),
                    gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
                    max_overlaps=torch.cat(
                        [res.max_overlaps for res in assign_result_list])
                )

            
            sampling_result = self.bbox_sampler.sample(assign_result_ensemble,
                                                   bboxes_tensor,
                                                   cls_gt_bboxes_3d)

            pos_inds = sampling_result.pos_inds
            neg_inds = sampling_result.neg_inds

            assert len(pos_inds) + len(neg_inds) == num_proposals

            # 3. Create target for loss computation
            bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size
                                        ]).to(bboxes_tensor.device)
            bbox_targets_original = torch.zeros([num_proposals, 9
                                        ]).to(bboxes_tensor.device)
            bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size
                                        ]).to(bboxes_tensor.device)
            ious = assign_result_ensemble.max_overlaps
            ious = torch.clamp(ious, min=0.0, max=1.0)
            labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
            label_weights = bboxes_tensor.new_zeros(
                num_proposals, dtype=torch.long)
            positive_mask = torch.zeros_like(labels, dtype=torch.bool).to(labels.device)

            # both pos and neg have classification loss, only pos has regression
            # and iou loss
            if len(pos_inds) > 0:
                pos_bbox_targets = self.bbox_coder.encode(
                                sampling_result.pos_gt_bboxes)
    
                bbox_targets[pos_inds, :] = pos_bbox_targets
                bbox_targets_original[pos_inds, :] = sampling_result.pos_gt_bboxes
                bbox_weights[pos_inds, :] = 1.0

                labels[pos_inds] = cls
                if self.train_cfg.pos_weight <= 0:
                    label_weights[pos_inds] = 1.0
                else:
                    label_weights[pos_inds] = self.train_cfg.pos_weight

            if len(neg_inds) > 0:
                label_weights[neg_inds] = 1.0

            positive_mask[pos_inds] = True

            # import pdb; pdb.set_trace()
            all_bbox_targets[class_mask] = bbox_targets
            all_bbox_targets_original[class_mask] = bbox_targets_original
            all_bbox_weights[class_mask] = bbox_weights
            if assign_result_ensemble.num_gts > 0:
                all_ious[class_mask] = ious
            all_labels[class_mask] = labels
            all_label_weights[class_mask] = label_weights
            all_positive_mask[class_mask] = positive_mask
            all_num_pos_inds += int(pos_inds.shape[0])

        all_mean_ious = all_ious[all_positive_mask].sum() / max(all_num_pos_inds, 1)
        
        valid_mask = preds_dict['instance_cls'] != -1
        return (
            all_labels[None],
            all_label_weights[None],
            all_bbox_targets[None],
            all_bbox_weights[None],
            all_ious[None],
            all_num_pos_inds, 
            all_mean_ious[None], 
            all_positive_mask[None],
            all_bbox_targets_original[None],
            valid_mask[None]
        ) 
    
    def loss(self, preds_dicts, gt_bboxes_3d, gt_labels_3d, img_metas):
        """Loss function for CenterHead.

        Args:
            batch_feats (): Features in a batch.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.
        Returns:
            dict[str:torch.Tensor]: Loss of score and bbox of each task.
        """
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            positive_mask,
            bbox_targets_original,
            valid_mask
        ) = self.get_targets(gt_bboxes_3d,
                             gt_labels_3d, 
                             preds_dicts)
        if hasattr(self, 'on_the_image_mask'):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()

        print(f"Assign Targets IOU is {matched_ious:.4f}, ")
        merged_dict = {
            "center": [],
            "height": [],
            "rot": [],
            "dim": [],
            "vel": [],
            "heatmap": [],
            'instance_cls': [],
            "post_fusion": [],
            "anchor" : []
        }
        for bs_idx in range(len(preds_dicts)):
            for pred_head in merged_dict:
                merged_dict[pred_head].append(preds_dicts[bs_idx][pred_head].T)  # [bs, num_proposals, embed_dims]

        for key in merged_dict:
            merged_dict[key] = torch.stack(merged_dict[key])

        loss_dict = dict()
        
        center = merged_dict['center'] 
        height = merged_dict['height']
        rot = merged_dict['rot']
        dim = merged_dict['dim']

        cls_score = merged_dict['heatmap']
   
        loss_cls = self.loss_cls(
            cls_score[valid_mask].reshape(-1, self.num_classes),
            labels[valid_mask].reshape(-1),
            label_weights[valid_mask].reshape(-1),
            avg_factor=max(num_pos, 1),
        )

        if 'vel' in merged_dict.keys():
            vel = merged_dict['vel']
            preds = torch.cat([center, height, dim, rot, vel], dim=-1) # [BS, num_proposals, code_size]
        else:
            preds = torch.cat([center, height, dim, rot], dim=-1)  # [BS, num_proposals, code_size]
        code_weights = self.train_cfg.get('code_weights', None)
        reg_weights = bbox_weights * bbox_weights.new_tensor(code_weights)

        
        loss_bbox = self.loss_bbox(
            preds[valid_mask],
            bbox_targets[valid_mask],
            reg_weights[valid_mask],
            avg_factor=max(num_pos, 1))

        pred_boxes = self.bbox_coder.decode(
            merged_dict['heatmap'].permute(0, 2, 1).sigmoid(),
            merged_dict['rot'].permute(0, 2, 1),
            merged_dict['dim'].permute(0, 2, 1),
            merged_dict['center'].permute(0, 2, 1),
            merged_dict['height'].permute(0, 2, 1),
            merged_dict['vel'].permute(0, 2, 1) if 'vel' in merged_dict else None
        )

        pred_boxes = torch.stack([box['bboxes'] for box in pred_boxes])

        # DEBUG(cxy)
        import cv2
        print("num of all gt:" ,sum([bboxes_3d.tensor.shape[0] for bboxes_3d in gt_bboxes_3d]))
        print("num of gt:", bbox_targets[positive_mask].shape[0])

        boxes = bbox_targets_original[0][positive_mask[0]].cpu().detach().numpy()
        post_fusion_boxes = merged_dict['post_fusion'].permute(0, 2, 1)[0][positive_mask[0]].cpu().detach().numpy()
        
        # img_meta_vehicles = img_metas[0]
        # img_files = [ img_meta[len(img_meta) - 1]['img_filename'] for img_meta in img_meta_vehicles ]
        x_max, y_max = boxes.max(axis = 0)[:2] + 5
        x_min, y_min = boxes.min(axis = 0)[:2] - 5

        # print(f"Image files: {img_files}")
        print((x_min, x_max), (y_min, y_max))

        res = 0.1
        canvas = np.zeros((int((y_max - y_min) / res), int((x_max - x_min) / res), 3))
        for idx, box in enumerate(boxes):
            xc, yc,_, l, w,_, yaw = box[:7]
            x_idx, y_idx = int((xc - x_min) / res), int((yc - y_min) / res)
            l, w = int(l / res), int(w / res)
            # draw rectangle with rotation
            rect = cv2.boxPoints(((x_idx, y_idx), (l, w), np.degrees(yaw)))
            rect = np.int32(rect)
            # draw the hallow rectangle
            cv2.polylines(canvas, [rect], isClosed=True, color=(0, 0, 1), thickness=1)
            # draw the idx to the rectangle
            cv2.putText(canvas, str(idx), (x_idx, y_idx), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (1, 1, 1), 1, cv2.LINE_AA)

        
        boxes = pred_boxes[0][positive_mask[0]].detach().cpu().numpy()
        for idx, box in enumerate(boxes):
            xc, yc, _, l, w, _, yaw = box[:7]
            x_idx, y_idx = int((xc - x_min) / res), int((yc - y_min) / res)
            l, w = int(l / res), int(w / res)
            # draw rectangle with rotation
            rect = cv2.boxPoints(((x_idx, y_idx), (l, w), np.degrees(yaw)))
            rect = np.int32(rect)
            # draw the hallow rectangle
            cv2.polylines(canvas, [rect], isClosed=True, color=(1, 0, 0), thickness=1)
            
            cv2.putText(canvas, str(idx), (x_idx, y_idx), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (1, 1, 1), 1, cv2.LINE_AA)
        cv2.imwrite('canvas_assign.png', (canvas * 255).astype(np.uint8))
        # import pdb; pdb.set_trace()

        # calculate IOU between prediction and target
        pos_target = bbox_targets_original[positive_mask].clone().contiguous()
        pos_pred = pred_boxes[positive_mask].clone().contiguous()

        pos_anchor = merged_dict['anchor'][positive_mask].clone().contiguous()
        pos_anchor[:, :2] = pos_anchor[:, :2] * self.bbox_coder.out_size_factor * pos_anchor.new_tensor(self.bbox_coder.voxel_size[:2])

   
        iou = self.bbox_assigner.iou_calculator(pos_pred, pos_target)[torch.eye(pos_pred.shape[0], dtype=torch.bool, device=pos_pred.device)]

        print(f"anchor largest loss: {torch.abs(pos_target[:, :2] - pos_anchor[:, :2]).max().item()}")
        print(f"center largest loss: {torch.abs(pos_pred[:, :2] - pos_anchor[:, :2]).max().item()}")
        print(f"xy largest loss: {torch.abs((pos_pred[:, :2] - pos_target[:, :2]) ).max().item()}")
        print(f"lwh largest loss: {torch.abs(pos_pred[:, 3:5] - pos_target[:, 3:5]).max().item()}")
        print(f"yaw largest loss: {(torch.abs(torch.atan2(pos_pred[:, 6], pos_pred[:, 7]))  % np.pi - torch.atan2(pos_target[:, 6], pos_target[:, 7])% np.pi).max().item()}")
        loss_dict['fusion_loss_cls'] = loss_cls 
        loss_dict['fusion_loss_bbox'] = loss_bbox
        loss_dict['matched_ious'] = torch.mean(iou)

        return loss_dict


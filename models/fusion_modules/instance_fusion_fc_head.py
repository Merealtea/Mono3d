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
from models.utils.transformation_utils import transform_boxes_torch
from models.builder import HEADS
from scipy.spatial.transform import Rotation as R
from models.modules.multi_head_attention import MultiHeadAttention
from models.modules.rope_self_attention import ContinuousRoPEAttention
from mmcv.cnn.bricks.transformer import build_feedforward_network

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

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


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

def rotation_matrix_to_rpy(R_matrix):
    """
    Convert a 3x3 rotation matrix to roll, pitch, yaw (ZYX Euler angles) using scipy.
    Output angles are in radians and normalized to [-pi, pi].

    Args:
        R_matrix (np.ndarray): shape (3, 3)

    Returns:
        roll, pitch, yaw (float): in radians
    """
    assert R_matrix.shape == (3, 3), "Input must be a 3x3 rotation matrix."

    # 创建 Rotation 对象
    rot = R.from_matrix(R_matrix)

    # 提取 ZYX 欧拉角（yaw, pitch, roll）
    yaw, pitch, roll = rot.as_euler('ZYX', degrees=False)

    return roll, pitch, yaw


@HEADS.register_module()
class InstanceFusionFCHead(nn.Module):
    def __init__(self, 
                 fusion_distance, 
                 embed_dims=256,
                loss_bbox=dict(type='L1Loss', reduction='mean'),
                loss_cls=dict(type='FocalLoss', reduction='mean'),
                num_classes = 3,
                num_attn_layer = 3,
                # others
                train_cfg=None,
                test_cfg=None,
                bbox_coder=None
                 ):
        super(InstanceFusionFCHead, self).__init__()
        self.fusion_distance = fusion_distance
        self.embed_dims = embed_dims // 2
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)

        # reference_points ---> pos_embed
        self.get_prior_embedding = nn.Linear(9, embed_dims)

        self.rot_embed, rot_embed_dim = get_embedder(10)

        # cross-agent feature alignment
        self.cross_agent_align_pos = \
            nn.Sequential(
                nn.Linear(embed_dims + rot_embed_dim, embed_dims),
                nn.Tanh(),
                nn.LayerNorm(embed_dims),
                nn.Linear(embed_dims, embed_dims),
            )

        
        res_fc_layer = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
        )
        
        self.res_layers = nn.ModuleList(
            deepcopy(res_fc_layer) for _ in range(3))
        
        self.add_proir_info = nn.Sequential(
            nn.Linear(embed_dims + embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
        )

        # self.multi_head_attn = MultiHeadAttention(embed_dims, 4) 
        self.num_attn_layer = num_attn_layer
        self.multi_head_attn =  ContinuousRoPEAttention(
                    dim = embed_dims,
                    num_heads=4,
                    )

        self.num_classes = num_classes

        num_classes = [1] * num_classes

        self.sampling = False

        self._init_assigner_sampler()
        self.bbox_coder = build_bbox_coder(bbox_coder)

        # parameter initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        cls_branch = []
        for _ in range(2):
            cls_branch.append(nn.Linear(embed_dims, embed_dims))
            cls_branch.append(nn.LayerNorm(embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(embed_dims, 1))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(2):
            reg_branch.append(nn.Linear(embed_dims, embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(embed_dims, bbox_coder.code_size))
        reg_branch = nn.Sequential(*reg_branch)
        
        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
        self.cls_branches = _get_clones(fc_cls, len(num_classes))
        self.reg_branches = _get_clones(reg_branch, len(num_classes))

    def train(self, mode = True):
        super().train(mode)
        self.training = mode

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
                        pairwise_assignment(deepcopy(first_agent_bboxes_cls),
                                            deepcopy(second_agent_bboxes_cls), 
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

        # Align agent results to the same coordinate system
        aligned_agent_bboxes = []
        aligned_agent_bboxes_cls = []
        aligned_agent_feats = []
        aligned_agent_scores = []

        agent2center_rt = agent2center_rt.float()

        for agent_idx, agent_result in enumerate(agent_results):
            # Only remain yaw and translation
            roll, pitch, yaw = rotation_matrix_to_rpy(agent2center_rt[agent_idx][:3,:3].cpu().detach().numpy())
            # agent2center_rt[agent_idx][:3, :3] = torch.tensor([
            #     [np.cos(yaw), -np.sin(yaw), 0],
            #     [np.sin(yaw), np.cos(yaw), 0],
            #     [0, 0, 1]
            # ]).to(agent2center_rt[agent_idx].device).float()
            
            aligned_agent_scores.append(agent_result['scores_3d'])
            aligned_agent_bboxes.append(
                transform_boxes_torch(agent_result['boxes_3d'],
                                agent2center_rt[agent_idx]))
            
            aligned_agent_bboxes_cls.append(agent_result['labels_3d'])
            
            agent_feat = agent_result['query_3d']
            agent_pose = agent_result['query_pose_3d']

            # agent_feat = F.normalize(agent_feat, dim=-1)
            
            agent_pose = torch.cat(
                    [agent_pose, torch.zeros_like(agent_pose[:, :1]).to(agent_pose.device), torch.ones_like(agent_pose[:, :1]).to(agent_pose.device)], dim=-1)
            
            agent_pose = (agent2center_rt[agent_idx] @ (agent_pose.T)).T[:, :2] 

            # get yaw from agent2center_rat 
            vehicle_rot = torch.FloatTensor([roll, pitch, yaw]).reshape(1,3).repeat(agent_feat.shape[0], 1).to(agent_feat.device)
            

            vehicle_rot = self.rot_embed(vehicle_rot)
            agent_feat = self.cross_agent_align_pos(torch.cat([agent_feat, vehicle_rot], dim=-1))  # [num_query, embed_dims]
            aligned_agent_feats.append(agent_feat)
        

        # assign the matched detection results
        match_result = self._query_matching(aligned_agent_bboxes,
                                            aligned_agent_bboxes_cls,
                                            agent_localization)
        
        # # fuse all the matched detection query features
        all_instance_anchors = dict(zip(list(range(self.num_classes)), [[] for _ in range(self.num_classes)]))
        all_fused_instance_feats = dict(zip(list(range(self.num_classes)), [[] for _ in range(self.num_classes)]))
        all_fused_instance_num = dict(zip(list(range(self.num_classes)), [[] for _ in range(self.num_classes)]))
        all_post_fusion_instance = dict(zip(list(range(self.num_classes)), [[] for _ in range(self.num_classes)]))
        
        bias_priors = []
        feats = []
        anchor_pos = []
        
        for cluster_idx in match_result:
            instances = match_result[cluster_idx]
            instance_cls = aligned_agent_bboxes_cls[instances[0][0]][instances[0][1]].item()
            instance_scores = torch.stack(
                [aligned_agent_scores[agent_idx][box_idx] for agent_idx, box_idx in instances])[:, None]  # [num_instance, 1]
            cluster_box = torch.stack(
                [aligned_agent_bboxes[agent_idx][box_idx] for agent_idx, box_idx in instances])
            

            if self.training:
                cluster_box[:, :2] += (torch.rand_like(cluster_box[:, :2]) * 0.2 - 0.1)
                cluster_box[:, 3:6] *= (torch.rand_like(cluster_box[:, 3:6]) * 0.2 + 0.9)
            mean_box = cluster_box.mean(axis=0, keepdims=True)

            # x,y,yaw
            bias_prior = deepcopy(cluster_box)
            bias_prior[:, :2] -= mean_box[:, :2]
            # bias_prior = bias_prior[:, :7]
            # bias_prior = torch.cat([bias_prior, instance_scores], dim=-1)
            
            bias_priors.append(bias_prior)
            all_post_fusion_instance[instance_cls].append(mean_box)
            
            cluster_feat = torch.stack(
                [aligned_agent_feats[agent_idx][box_idx] for agent_idx, box_idx in instances])
            feats.append(cluster_feat)
            
            instance_center = cluster_box[:, :2].mean(dim=0, keepdim=True)

            anchor_pos.append(cluster_box[:, :2])
            all_instance_anchors[instance_cls].append(instance_center)
            
        bias_priors = torch.cat(bias_priors, axis=0).to(cluster_feat.device)
        bias_priors[:, 3:6] = bias_priors[:, 3:6].log()  # l, w, h
        bias_priors = self.get_prior_embedding(bias_priors)
        feats = torch.cat(feats, dim=0)  # [num_cluster, num_instance, embed_dims]
        feats = self.add_proir_info(torch.cat([feats, bias_priors], dim=-1)) # + bias_priors 
        
        for i in range(len(self.res_layers)):
            feats = self.res_layers[i](feats) + feats
        anchor_pos = torch.cat(anchor_pos, dim=0)  # [num_cluster, 2]
        # attn_mask = (anchor_pos[:, None, :] - anchor_pos[None, :, :]).norm(p=2, dim=-1) < (self.fusion_distance / 2)

        feats = self.multi_head_attn(feats[None], anchor_pos[None])[0]
            # feats = self.multi_head_attn[i]['ln'](feats)

        start_idx = 0
        for cluster_idx in match_result:
            instances = match_result[cluster_idx]
            instance_cls = aligned_agent_bboxes_cls[instances[0][0]][instances[0][1]].item()
            
            cluster_feat = feats[start_idx:start_idx + len(instances), :]
            start_idx += len(instances)
            
            cluster_feat = torch.mean(cluster_feat, dim=0, keepdim=True)
            all_fused_instance_feats[instance_cls].append(cluster_feat)
            all_fused_instance_num[instance_cls].append(len(instances))
        
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
        # import pdb; pdb.set_trace()

        for cls in all_fused_instance_feats:
            all_fused_instance_feats[cls] = torch.cat(all_fused_instance_feats[cls])[None] if len(all_fused_instance_feats[cls]) > 0 else torch.zeros((1, 0, self.embed_dims * 2)).to(feats.device)
            all_instance_anchors[cls] = torch.cat(all_instance_anchors[cls])[None] if len(all_instance_anchors[cls]) > 0 else torch.zeros((1, 0, 2)).to(feats.device)
            all_fused_instance_num[cls] = torch.tensor(all_fused_instance_num[cls], dtype=torch.float32).to(feats.device)[None] if len(all_fused_instance_num[cls]) > 0 else torch.zeros((1, 0)).to(feats.device)
            all_post_fusion_instance[cls] = torch.cat(all_post_fusion_instance[cls], axis=0).to(feats.device) if len(all_post_fusion_instance[cls]) > 0 else torch.zeros((0, 9)).to(feats.device)
        return [all_fused_instance_feats, all_instance_anchors, all_fused_instance_num, all_post_fusion_instance]
    
        
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
        all_fused_instance_feats, all_instance_anchors, all_fused_instance_num, all_post_fusion_boxes\
            = multi_apply(self.forward_single, 
                           agent_results,
                           agent_localization,
                           agent2center_rt)
        
        all_feats = dict(zip(list(range(self.num_classes)), [[] for _ in range(self.num_classes)]))
        all_anchor = dict(zip(list(range(self.num_classes)), [[] for _ in range(self.num_classes)]))


        batch_fused_instance_num = dict(zip(list(range(self.num_classes)), [[] for _ in range(self.num_classes)]))

        all_instance_num = 0
        for fused_instance_feats, instance_anchors in\
              zip(
                    all_fused_instance_feats,
                    all_instance_anchors,
                ):
            for cls in fused_instance_feats:
                all_feats[cls].append(fused_instance_feats[cls])
                all_anchor[cls].append(instance_anchors[cls])
                batch_fused_instance_num[cls].append(fused_instance_feats[cls].shape[1])
                all_instance_num += fused_instance_feats[cls].shape[1]
        print(f"all instance number is {all_instance_num}")

        for cls in all_feats:
            all_feats[cls] = torch.cat(all_feats[cls], dim = 1).contiguous()  if len(all_feats[cls]) > 0 else torch.zeros((1, 0, self.embed_dims * 2)).to(fused_instance_feats.device)
            all_anchor[cls] = torch.cat(all_anchor[cls], dim = 1).contiguous() if len(all_anchor[cls]) > 0 else torch.zeros((1, 0, 2)).to(fused_instance_feats.device)

         # predict new bias based on the fused features
        ret_dicts = {}
        for task_idx, cls in enumerate(all_feats):
            if len(all_feats[cls]):
                predict_bias = self.reg_branches[task_idx](all_feats[cls])
                predict_bias[... , :2] += all_anchor[cls]   

                fusion_score = self.cls_branches[task_idx](all_feats[cls])  # [1, num_instance, 1]
            else:
                predict_bias = torch.zeros((1, 0, self.bbox_coder.code_size)).to(fused_instance_feats.device)
                fusion_score = torch.zeros((1, 0, 1)).to(fused_instance_feats.device)  # [1, num_instance, 1]
            # predict_bias[all_instance_num==1] = 0.0  # if only one instance, no need to predict bias
            ret_dicts[cls] = {
                'bboxes': predict_bias,  # [1, num_instance, 9]
                'scores': fusion_score,  # [1, num_instance, 1]
            }
            
        # separate into batch
        cls_res = dict(zip(list(range(self.num_classes)), [{} for _ in range(self.num_classes)]))
        separated_ret_dicts = [copy.deepcopy(cls_res) for _ in range(bs)]
        for cls in ret_dicts:
            start_idx = 0
            for bs_idx, fused_ins_num in enumerate(batch_fused_instance_num[cls]):
                for head in ret_dicts[cls]:
                    separated_ret_dicts[bs_idx][cls][head] = ret_dicts[cls][head][:, start_idx:start_idx+fused_ins_num]
                separated_ret_dicts[bs_idx][cls]['post_fusion'] = all_post_fusion_boxes[bs_idx][cls]
                separated_ret_dicts[bs_idx][cls]['num_instance'] = all_fused_instance_num[bs_idx][cls]
                start_idx += fused_ins_num

        return separated_ret_dicts

    def predict(self, agent_results, agent_localization, agent2center_rt, metas=None):
        rets = []
        preds_dicts = self(agent_results, agent_localization, agent2center_rt)
    
        box_type = metas[0][0][0]['box_type_3d']

        for bs_idx in range(len(preds_dicts)):
            pred_result = {
                "bboxes" : [],
                "scores" : [],
                "labels" : [],
                "post_fusion" : [],
                'num_instance' : []
            }
            for cls in preds_dicts[bs_idx]:
                bias = preds_dicts[bs_idx][cls]
                score = bias['scores']
                box = bias['bboxes']
                num_instance = bias['num_instance']
                post_fusion_box = bias['post_fusion']
                box_cls = torch.zeros_like(score, dtype=torch.long).to(score.device) + cls
                boxes_dict = self.bbox_coder.decode(
                    box, score, box_cls)  # decode the prediction to real world metric bbox
                for key in boxes_dict[0]:
                    pred_result[key].append(boxes_dict[0][key])
                pred_result['post_fusion'].append(post_fusion_box)
                pred_result['num_instance'].append(num_instance[0])

            for key in pred_result:
                pred_result[key] = torch.cat(pred_result[key], dim=0)

            rets.append(
                        [box_type(pred_result['bboxes'], box_dim=pred_result['bboxes'].shape[-1]),
                        pred_result['scores'],
                        pred_result['labels'].int(),
                        pred_result['post_fusion'],
                        pred_result['num_instance'].int()
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

        labels = torch.cat(res_tuple[0], dim=1)
        label_weights = torch.cat(res_tuple[1], dim=1)
        bbox_targets = torch.cat(res_tuple[2], dim=1)
        bbox_weights = torch.cat(res_tuple[3], dim=1)
        ious = torch.cat(res_tuple[4], dim=1)
        num_pos = np.sum(res_tuple[5])

        matched_ious = torch.cat(res_tuple[6], dim =1)

        reshape_match_ious = matched_ious.reshape(-1, 3)
        for label_id in range(3):
            print(f"Class {label_id} matched iou: {reshape_match_ious[:, label_id].mean().item()}")
        matched_ious = torch.mean(matched_ious)
        positive_mask = torch.cat(res_tuple[7], dim=1)

        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            positive_mask
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
        all_labels = []
        all_label_weights = []
        all_bbox_targets = []
        all_bbox_weights = []
        all_ious = []
        all_num_pos_inds = 0
        all_mean_ious = []
        all_positive_mask = []
   
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(preds_dict[0]['scores'].device)
        
        assign_threshold = {
            0 : 3.0,  # car
            1 : 2.0, # pedestrian
            2 : 5.0, # Truck  
        }

        for cls in preds_dict:
            bias = preds_dict[cls]

            # class bbox
            cls_gt_bboxes_3d = gt_bboxes_tensor[gt_labels_3d == cls]
            
            # 1. Assignment
            num_proposals = bias['scores'].shape[1]


            if num_proposals < len(cls_gt_bboxes_3d):
                print(f"Warning: The number of anchors {num_proposals} is less than the number of ground truth boxes {len(cls_gt_bboxes_3d)} in class {cls}. ")

            bboxes_tensor = bias['post_fusion']
            
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
                bbox_weights[pos_inds, :] = 1.0

                labels[pos_inds] = 1
                if self.train_cfg.pos_weight <= 0:
                    label_weights[pos_inds] = 1.0
                else:
                    label_weights[pos_inds] = self.train_cfg.pos_weight

            if len(neg_inds) > 0:
                label_weights[neg_inds] = 1.0

            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            positive_mask[pos_inds] = True

            all_labels.append(labels)
            all_label_weights.append(label_weights)
            all_bbox_targets.append(bbox_targets)
            all_bbox_weights.append(bbox_weights)
            all_ious.append(ious)
            all_num_pos_inds += int(pos_inds.shape[0])
            all_mean_ious.append(mean_iou)
            all_positive_mask.append(positive_mask)

        return (
            torch.cat(all_labels, dim=0)[None],
            torch.cat(all_label_weights, dim=0)[None],
            torch.cat(all_bbox_targets, dim=0)[None],
            torch.cat(all_bbox_weights, dim=0)[None],
            torch.cat(all_ious, dim=0)[None],
            all_num_pos_inds, 
            torch.stack(all_mean_ious)[None], 
            torch.cat(all_positive_mask, dim=0)[None]
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
        ) = self.get_targets(gt_bboxes_3d,
                             gt_labels_3d, 
                             preds_dicts)
        if hasattr(self, 'on_the_image_mask'):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()
        merged_dict = {
            'bboxes': [],
            "scores": [],
            'num_instance': [],
        }
        for bs_idx in range(len(preds_dicts)):
            for cls in preds_dicts[bs_idx]:
                for pred_head in merged_dict:
                    merged_dict[pred_head].append(preds_dicts[bs_idx][cls][pred_head])

        for key in merged_dict:
            merged_dict[key] = torch.cat(merged_dict[key], dim=1)


        loss_dict = dict()

        preds = merged_dict['bboxes'] 
        cls_score = merged_dict['scores']

        loss_cls = self.loss_cls(
            cls_score.float()[0],
            labels[0],
            label_weights,
            avg_factor=max(num_pos, 1),
        )

        code_weights = self.train_cfg.get('code_weights', None)
        reg_weights = bbox_weights * bbox_weights.new_tensor(code_weights)

        # # DEBUG(cxy)
        # import cv2
        print("num of all gt:" ,sum([bboxes_3d.tensor.shape[0] for bboxes_3d in gt_bboxes_3d]))
        print("num of gt:", bbox_targets[positive_mask].shape[0])
        # boxes = bbox_targets[positive_mask].cpu().detach().numpy()
        # post_fusion_boxes = preds[positive_mask].cpu().detach().numpy()

        # # img_meta_vehicles = img_metas[0]
        # # img_files = [ img_meta[len(img_meta) - 1]['img_filename'] for img_meta in img_meta_vehicles ]
        # x_max, y_max = boxes.max(axis = 0)[:2] + 5
        # x_min, y_min = boxes.min(axis = 0)[:2] - 5

        # # print(f"Image files: {img_files}")
        # print((x_min, x_max), (y_min, y_max))

        # res = 0.1
        # canvas = np.zeros((int((y_max - y_min) / res), int((x_max - x_min) / res), 3))
        # for idx, box in enumerate(boxes[:30]):
        #     xc, yc,_, l, w,_, yaw = box[:7]
        #     x_idx, y_idx = int((xc - x_min) / res), int((yc - y_min) / res)
        #     l, w = int(l / res), int(w / res)
        #     # draw rectangle with rotation
        #     rect = cv2.boxPoints(((x_idx, y_idx), (l, w), np.degrees(yaw)))
        #     rect = np.int32(rect)
        #     # draw the hallow rectangle
        #     cv2.polylines(canvas, [rect], isClosed=True, color=(0, 0, 1), thickness=1)
        #     # draw the idx to the rectangle
        #     cv2.putText(canvas, str(idx), (x_idx, y_idx), cv2.FONT_HERSHEY_SIMPLEX, 
        #                 0.5, (1, 1, 1), 1, cv2.LINE_AA)
        # for idx, box in enumerate(post_fusion_boxes[:30]):
        #     xc, yc, _, l, w, _, yaw = box[:7]
        #     x_idx, y_idx = int((xc - x_min) / res), int((yc - y_min) / res)
        #     l, w = int(l / res), int(w / res)
        #     # draw rectangle with rotation
        #     rect = cv2.boxPoints(((x_idx, y_idx), (l, w), np.degrees(yaw)))
        #     rect = np.int32(rect)
        #     # draw the hallow rectangle
        #     cv2.polylines(canvas, [rect], isClosed=True, color=(1, 0, 0), thickness=1)
            
        #     cv2.putText(canvas, str(idx), (x_idx, y_idx), cv2.FONT_HERSHEY_SIMPLEX, 
        #                 0.5, (1, 1, 1), 1, cv2.LINE_AA)
        # cv2.imwrite('canvas_assign.png', (canvas * 255).astype(np.uint8))

        # import pdb; pdb.set_trace()
        
        loss_bbox = self.loss_bbox(
            preds[positive_mask],
            bbox_targets[positive_mask],
            reg_weights[positive_mask])

        # calculate IOU between prediction and target
        pos_target = bbox_targets[positive_mask].clone().contiguous()
        pos_pred = preds[positive_mask].clone().contiguous()

        pos_target[:, 3:6] = pos_target[:, 3:6].exp()  # l, w, h
        pos_pred[:, 3:6] = pos_pred[:, 3:6].exp()  # l, w, h
        iou = self.bbox_assigner.iou_calculator(pos_pred, pos_target)[torch.eye(pos_pred.shape[0], dtype=torch.bool, device=pos_pred.device)]

        print(f"xy mean loss: {torch.abs(pos_pred[:, :2] - pos_target[:, :2]).mean().item()}")
        print(f"lwh mean loss: {torch.abs(pos_pred[:, 3:5] - pos_target[:, 3:5]).mean().item()}")
        print(f"yaw mean loss: {(torch.abs(torch.atan2(pos_pred[:, 6], pos_pred[:, 7]))  % np.pi - torch.atan2(pos_target[:, 6], pos_target[:, 7])% np.pi).mean().item()}")
        loss_dict['fusion_loss_cls'] = loss_cls 
        loss_dict['fusion_loss_bbox'] = loss_bbox
        loss_dict['matched_ious'] = torch.mean(iou)

        return loss_dict

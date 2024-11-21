import numpy as np
import torch
import cv2
import random
from copy import deepcopy
import os

shift = np.array([0, 0, 0])

def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
        return data
    if isinstance(data, list):
        return [to_device(d, device) for d in data]
    return data

def init_random_seed(seed=None):
    if seed is None:
        seed = np.random.randint(2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def extract_rotation_matrix(affine_matrix):
    """Extract and normalize the rotation matrix from a 4x4 affine matrix."""
    # Extract the 3x3 rotation and scale matrix
    rot_scale_matrix = affine_matrix[:3, :3]
    
    # Normalize each column to get the rotation matrix
    rotation_matrix = rot_scale_matrix / np.linalg.norm(rot_scale_matrix, axis=0)
    
    return rotation_matrix

def calculate_corners(bbox):
    # Extracting components of the bounding box
    bbox_copy = deepcopy(bbox)
    xyz, length, width, height, yaw =\
         deepcopy(bbox_copy[:, 0:3]), bbox_copy[:, 3], bbox_copy[:, 4], bbox_copy[:, 5], bbox_copy[:, 6]
    
    # Define half dimensions for convenience
    half_length = length / 2
    half_width = width / 2
    half_height = height / 2

    # Define the corners of the bounding box in local coordinate system
    corners_local = np.stack([
        np.stack([-half_length, -half_width, -half_height], axis=0),
        np.stack([-half_length, -half_width, half_height], axis=0),
        np.stack([-half_length, half_width, half_height], axis=0),
        np.stack([-half_length, half_width, -half_height], axis=0),
        np.stack([half_length, -half_width, -half_height], axis=0),
        np.stack([half_length, -half_width, half_height], axis=0),
        np.stack([half_length, half_width, half_height], axis=0),
        np.stack([half_length, half_width, -half_height], axis=0)
    ], axis=0).transpose((2, 1, 0))

    # Rotation matrix around the z-axis (yaw)
    rotation_matrix = np.stack([
        np.stack([np.cos(yaw),  -np.sin(yaw), np.zeros_like(yaw)],axis=0),
        np.stack([np.sin(yaw), np.cos(yaw), np.zeros_like(yaw)], axis=0),
        np.stack([np.zeros_like(yaw), np.zeros_like(yaw), np.ones_like(yaw)], axis=0),
    ], axis=0).transpose((2, 0, 1))

    # Apply rotation to each corner
    rotated_corners = np.einsum("nik,nkj->nij", rotation_matrix, corners_local)

    # Translate to the bounding box center
    translated_corners = rotated_corners + xyz.reshape(-1, 3, 1) + shift.reshape(-1, 3, 1)
    
    return translated_corners.transpose(0, 2, 1)

def turn_boxes_to_kitti(boxes, metric ="bbox"):
    """
        metric includes: bbox, bev, 3d
    """
    annos = {}
    num_boxes = len(boxes)
    if metric == "bbox":
        boxes[:, 3:6] = boxes[:, 3:6] * 2
        boxes[:, :2] = boxes[:, :2] - boxes[:, 3:5] / 2
        return boxes
    elif metric == "bev":
        boxes[:, 3:5] = boxes[:, 3:5] * 2
        boxes[:, :2] = boxes[:, :2] - boxes[:, 3:5] / 2
        return boxes
    elif metric == "3d":
        annos['type'] = 'Pedestrian'
        annos['truncated'] = [0.0] * num_boxes
        annos['occluded'] = [0] * num_boxes
        annos['alpha'] = [0] * num_boxes
        annos['bbox'] = boxes[:, -4:]
        annos['dimensions'] = boxes[:, 3:6]
        annos['location'] = boxes[:, :3]
        annos['rotation_y'] = boxes[:, 6]
        return annos
    else:
        raise ValueError(f"metric {metric} is not supported")


def plot_rect3d_on_img(img,
                       num_rects,
                       rect_corners,
                       color=(0, 255, 0),
                       thickness=1,
                       scores = None,
                       vars = None):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))

    if scores is not None and len(scores) != num_rects:
        raise ValueError("The length of scores should be the same as the number of rectangles.")
    
    if vars is not None and len(vars) != num_rects:
        raise ValueError("The length of vars should be the same as the number of rectangles.")
    
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int32)
        for start, end in line_indices:
            try:
                cv2.line(img, (corners[start, 0], corners[start, 1]),
                        (corners[end, 0], corners[end, 1]), color, thickness,
                        cv2.LINE_AA)
                
                if scores is not None:
                    cv2.putText(img, "{:.2f}".format(scores[i]), (corners[0, 0], corners[0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                if vars is not None:
                    cv2.putText(img, "{:.2f}_{:.2f}".format(vars[i][0], vars[i][1]), (corners[0, 0], corners[0, 1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            except:
                import pdb; pdb.set_trace()

    return img.astype(np.uint8)


def detection_visualization(bbox, gt_bbox, filename, cam_model, bbox_res_path, bboxes_coor = "CAM", scores = None, vars = None):
    if bboxes_coor == "CAM":
        bboxes = []
        bbox[:, :3] = cam_model.cam2world(bbox[:, :3])
        gt_bbox[:, :3] = cam_model.cam2world(gt_bbox[:, :3])
        corners = calculate_corners(bbox).reshape(-1, 3)
        gt_corners = calculate_corners(gt_bbox).reshape(-1, 3)          
        corners = cam_model.world2cam(corners.T).T.reshape(-1, 8, 3)
        gt_corners = cam_model.world2cam(gt_corners.T).T.reshape(-1, 8, 3)
     
        for corner in corners:
            pixel_uv = cam_model.cam2image(corner.T).T
            bboxes.append(pixel_uv)

        gt_bboxes = []
        for corner in gt_corners:
            pixel_uv = cam_model.cam2image(corner.T).T
            gt_bboxes.append(pixel_uv)

        img = cv2.imread(filename)

        img = plot_rect3d_on_img(img, len(gt_bboxes), gt_bboxes, color=(0, 255, 0))
        img = plot_rect3d_on_img(img, len(bboxes), bboxes, color=(0, 0, 255), scores=scores, vars=vars)

    elif bboxes_coor == "Lidar":
        bboxes = []
        depth = cam_model.world2cam(bbox[:, :3].T).T[:, 0]
        gt_depth = cam_model.world2cam(gt_bbox[:, :3].T).T[:, 0]
        bbox = bbox[depth > 0.05]
        gt_bbox = gt_bbox[gt_depth > 0.05]

        if scores is not None:
            scores = scores[depth > 0.05]
        if vars is not None:
            vars = vars[depth > 0.05]

        corners = calculate_corners(bbox).reshape(-1, 3)
        gt_corners = calculate_corners(gt_bbox).reshape(-1, 3)
        corners = cam_model.world2cam(corners.T)
        gt_corners = cam_model.world2cam(gt_corners.T)
        corners[0][corners[0] < 0.05] = 0.05
        gt_corners[0][gt_corners[0] < 0.05] = 0.05

        corners = corners.T.reshape(-1, 8, 3)
        gt_corners = gt_corners.T.reshape(-1, 8, 3)

        for corner in corners:
            pixel_uv = cam_model.cam2image(corner.T).T
            if len(pixel_uv) < 8:
                continue
            bboxes.append(pixel_uv)
        gt_bboxes = []
        for corner in gt_corners:
            pixel_uv = cam_model.cam2image(corner.T).T
            if len(pixel_uv) < 8:
                continue
            gt_bboxes.append(pixel_uv)
        img = cv2.imread(filename)

        img = plot_rect3d_on_img(img, len(gt_bboxes), gt_bboxes, color=(0, 255, 0))
        img = plot_rect3d_on_img(img, len(bboxes), bboxes, color=(0, 0, 255), scores=scores, vars=vars)
  
    return img

def turn_gt_to_annos(gts, class_names):
    assert isinstance(gts, list), "gt must be a list"
    if len(gts) == 0:
        return []

    assert isinstance(gts[0], dict), "gt must be a list of dict"

    gt_annos = []
    for image_idx, gt in enumerate(gts):
        annos = []
        gt_labels = gt['gt_labels'][0].cpu()
        if 'gt_bboxes' in gt:
            bboxes = gt['gt_bboxes'][0].cpu()
        else:
            bboxes = torch.zeros([len(gt_labels), 4])
        dimensions = gt['gt_bboxes_3d'][0][:, 3:6].cpu()
        location = gt['gt_bboxes_3d'][0][:, :3].cpu()
        rotation_y = gt['gt_bboxes_3d'][0][:, 6].cpu()
        timestamp = gt['img_metas'][0]['timestamp']
        direction = gt['img_metas'][0]['direction']
        if len(gt_labels) == 0:
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
        else:
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
            for det_idx in range(len(gt_labels)):
                anno['name'].append(class_names[int(gt_labels[det_idx])])
                anno['truncated'].append(0.0)
                anno['occluded'].append(0)
                anno['alpha'].append(0)
                anno['bbox'].append(bboxes[det_idx])
                anno['dimensions'].append(dimensions[det_idx])
                anno['location'].append(location[det_idx])
                anno['rotation_y'].append(rotation_y[det_idx])
                anno['score'].append(np.array([1.0]))
            anno = {k: np.stack(v) for k, v in anno.items()}
            annos.append(anno)

        annos[-1]['sample_idx'] = np.array(
                [image_idx] * len(annos[-1]['score']), dtype=np.int64)
        annos[-1]['timestamp'] = np.array(timestamp)
        annos[-1]['direction'] = np.array(direction)
        gt_annos += annos
    return gt_annos
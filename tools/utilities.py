import numpy as np
import torch
import cv2
from copy import deepcopy
def init_random_seed(seed=None):
    if seed is None:
        seed = np.random.randint(2**32 - 1)
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


def calculate_corners_cam(bbox, world2cam_mat):
    # Extracting components of the bounding box
    bbox_copy = deepcopy(bbox)
    bbox_copy[:, :2] *= 1.2
    xyz, length, width, height, yaw =\
         bbox_copy[:, 0:3], bbox_copy[:, 3], bbox_copy[:, 4], bbox_copy[:, 5], bbox_copy[:, 6]

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
        np.stack([np.cos(yaw), -np.sin(yaw), np.zeros_like(yaw)],axis=0),
        np.stack([np.sin(yaw), np.cos(yaw), np.zeros_like(yaw)], axis=0),
        np.stack([np.zeros_like(yaw), np.zeros_like(yaw), np.ones_like(yaw)], axis=0),
    ], axis=0).transpose((2, 0, 1))
 

    corners_local = np.einsum("nik,nkj->nij", rotation_matrix, corners_local)

    fix_rotation_matrix  = extract_rotation_matrix(world2cam_mat)[None, :, :]
    rotated_corners = np.einsum("nik,nkj->nij", fix_rotation_matrix, corners_local)


    # Translate to the bounding box center
    translated_corners = rotated_corners + xyz.reshape(-1, 3, 1)

    return translated_corners.transpose(0, 2, 1)

def plot_rect3d_on_img(img,
                       num_rects,
                       rect_corners,
                       color=(0, 255, 0),
                       thickness=1):
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
    for i in range(num_rects):
        corners = rect_corners[i]
        for start, end in line_indices:
            try:
                cv2.line(img, (corners[start, 0], corners[start, 1]),
                        (corners[end, 0], corners[end, 1]), color, thickness,
                        cv2.LINE_AA)
            except:
                import pdb; pdb.set_trace()

    return img.astype(np.uint8)
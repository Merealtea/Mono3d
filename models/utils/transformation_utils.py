# -*- coding: utf-8 -*-
# Author: Xingyuan Chen <February25th@sjtu.edu.cn>,
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Transformation utils
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

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


def transform_boxes_torch(boxes: torch.Tensor, transform_matrix: torch.Tensor) -> torch.Tensor:
    """
    对 N 个检测框进行 3D 坐标变换（PyTorch 版本）。

    参数:
        boxes (torch.Tensor): 形状为 (N, 9)，每行格式为 [x, y, z, l, w, h, yaw, vx, vy]
        transform_matrix (torch.Tensor): 4x4 齐次变换矩阵，前3x3为旋转矩阵，第4列为平移向量

    返回:
        torch.Tensor: 变换后的检测框，形状为 (N, 9)
    """
    # 提取变换矩阵的旋转和平移部分
    R = transform_matrix[:3, :3]  # 3x3 旋转矩阵
    T = transform_matrix[:3, 3]  # 3x1 平移向量

    # 提取各部分属性
    positions = boxes[:, :3]       # x, y, z
    dimensions = boxes[:, 3:6]     # l, w, h
    yaws = boxes[:, 6]             # yaw
    velocities = boxes[:, 7:]      # vx, vy

    # 1. 变换位置
    transformed_positions = torch.matmul(positions, R.T) + T

    # 2. 变换 yaw 角度
    cos_yaw = torch.cos(yaws)
    sin_yaw = torch.sin(yaws)
    dir_vectors = torch.stack([cos_yaw, sin_yaw, torch.zeros_like(yaws).to(yaws.device)], dim=-1)  # (N, 3)
    rotated_dir_vectors = torch.matmul(dir_vectors, R.T)
    new_yaws = torch.atan2(rotated_dir_vectors[:, 1], rotated_dir_vectors[:, 0])

    # 3. 变换速度
    vel_3d = torch.cat([velocities, torch.zeros_like(velocities[..., :1]).to(yaws.device)], dim=-1)  # (N, 3)
    rotated_velocities = torch.matmul(vel_3d, R.T)
    new_velocities = rotated_velocities[:, :2]

    # 4. 构建新的 boxes
    new_boxes = torch.zeros_like(boxes).to(yaws.device)
    new_boxes[:, :3] = transformed_positions
    new_boxes[:, 3:6] = dimensions
    new_boxes[:, 6] = new_yaws
    new_boxes[:, 7:] = new_velocities

    return new_boxes

def transform_boxes(boxes, transform_matrix):
    """
    对 N 个检测框进行 3D 坐标变换。

    参数:
        boxes (np.ndarray): 形状为 (N, 9) 的检测框数组，每行格式为 [x, y, z, l, w, h, yaw, vx, vy]
        transform_matrix (np.ndarray): 4x4 齐次变换矩阵，前3x3为旋转矩阵，第4列为平移向量

    返回:
        np.ndarray: 形状为 (N, 9) 的变换后的检测框数组
    """
    # 提取变换矩阵的旋转和平移部分
    R = transform_matrix[:3, :3]
    T = transform_matrix[:3, 3]

    # 提取各部分属性
    positions = boxes[:, :3]       # x, y, z
    dimensions = boxes[:, 3:6]     # l, w, h
    yaws = boxes[:, 6]             # yaw
    velocities = boxes[:, 7:]      # vx, vy

    # 1. 变换位置
    transformed_positions = np.dot(positions, R.T) + T

    # 2. 变换 yaw 角
    cos_yaw = np.cos(yaws)
    sin_yaw = np.sin(yaws)
    dir_vectors = np.column_stack((cos_yaw, sin_yaw, np.zeros_like(yaws)))  # Nx3
    rotated_dir_vectors = np.dot(dir_vectors, R.T)
    new_yaws = np.arctan2(rotated_dir_vectors[:, 1], rotated_dir_vectors[:, 0])

    # 3. 变换速度
    vel_3d = np.column_stack((velocities, np.zeros_like(velocities[:, 0])))  # Nx3
    rotated_velocities = np.dot(vel_3d, R.T)
    new_velocities = rotated_velocities[:, :2]  # 取前两个分量

    # 4. 构建新的 boxes
    new_boxes = np.zeros_like(boxes)
    new_boxes[:, :3] = transformed_positions
    new_boxes[:, 3:6] = dimensions
    new_boxes[:, 6] = new_yaws
    new_boxes[:, 7:] = new_velocities

    return new_boxes

def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def x1_to_x2(x1, x2):
    """
    Transformation matrix from x1 to x2.

    Parameters
    ----------
    x1 : list
        The pose of x1 under world coordinates.
    x2 : list
        The pose of x2 under world coordinates.

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.

    """
    x1_to_world = x_to_world(x1)
    x2_to_world = x_to_world(x2)
    world_to_x2 = np.linalg.inv(x2_to_world)

    transformation_matrix = np.dot(world_to_x2, x1_to_world)
    return transformation_matrix


def dist_to_continuous(p_dist, displacement_dist, res, downsample_rate):
    """
    Convert points discretized format to continuous space for BEV representation.
    Parameters
    ----------
    p_dist : numpy.array
        Points in discretized coorindates.

    displacement_dist : numpy.array
        Discretized coordinates of bottom left origin.

    res : float
        Discretization resolution.

    downsample_rate : int
        Dowmsamping rate.

    Returns
    -------
    p_continuous : numpy.array
        Points in continuous coorindates.

    """
    p_dist = np.copy(p_dist)
    p_dist = p_dist + displacement_dist
    p_continuous = p_dist * res * downsample_rate
    return p_continuous

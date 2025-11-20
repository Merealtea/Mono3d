from tqdm import tqdm
from PIL import Image
import os.path as osp
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
from collections import defaultdict
from utils import design_depth_map
import os
import open3d as o3d
import matplotlib

matplotlib.use('Agg') # 使用非交互式后端

import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression
# from depthanything.metric_depth.depth_anything_v2.dpt import DepthAnythingV2 as Metric_DepthAnythingV2
import cv2



def parse_sf_lut(sf_info):
    """
    解析信息字符串，提取文件夹名、时间戳和 token。

    参数:
    - sf_info: 输入的字符串，格式为 'folder_name__timestamp token'

    返回:
    - folder_name: 提取的文件夹名称
    - timestamp: 提取的时间戳（整数形式）
    - token: 提取的 token
    """
    try:
        # 按空格分割字符串
        folder_name, token = sf_info.split(' ')

        # 提取时间戳：假设文件夹名的格式为 '...__timestamp'
        timestamp_str = folder_name.split('__')[-1]
        timestamp = int(timestamp_str)

        return folder_name, timestamp, token
    except ValueError as e:
        print(f"解析字符串失败: {e}")
        return None, None, None

def find_matching_camera_sweep_in_nusc(nusc, target_timestamp_range, camera_channel):
    """
    找到在指定时间戳范围内指定相机通道的相机数据
    :param nusc: NuScenes 数据集对象
    :param target_timestamp_range: 目标时间戳范围，单位为毫秒。例如：[timestamp_start - 50, timestamp_end + 50]
    :param camera_channel: 相机通道名称，例如 'CAM_FRONT'
    :return: 匹配到的所有相机数据 list，如果没有找到则返回 None
    """
    sample_data_list = []
    # 遍历所有 sample_data 找到指定时间戳范围内的 sample_data
    for sd in nusc.sample_data:
        # 指定传感器模态为 camera，且通道为指定的相机通道
        if sd['sensor_modality'] == 'camera' and sd['channel'] == camera_channel:
            if (((target_timestamp_range[0] - 50 * 1e3) <=
                    sd['timestamp'] <= (target_timestamp_range[1] + 50 * 1e3))):
                    # and sd['is_key_frame'] == False): # 前后各扩展50ms的时间范围
                sample_data_list.append(sd)

    # 如果找到了匹配的相机数据，则返回
    if len(sample_data_list) > 0:
        return sample_data_list
    else:
        return None

def find_matching_camera_sweep_in_lut(camera_lut, target_timestamp_range, camera_channel):
    """
    找到在指定时间戳范围内指定相机通道的相机数据
    :param camera_lut: 相机数据字典
    :param target_timestamp_range: 目标时间戳范围，单位为微秒。例如：[timestamp_start - 50 * 1e3, timestamp_end + 50 * 1e3]
    :param camera_channel: 相机通道名称，例如 'CAM_FRONT'
    :return: 匹配到的所有相机数据 timestamp list，如果没有找到则返回 None
    """
    camera_timestamp_list = []
    # 遍历 camera_lut 找到指定时间戳范围内的 数据
    for cam_data in camera_lut:
        timestamp = cam_data[camera_channel]['timestamp']
        # key = next(iter(cam_data[camera_channel].keys()))
        # timestamp = int(key)
        if ((target_timestamp_range[0] - 50 * 1e3) <= timestamp
                <= (target_timestamp_range[1] + 50 * 1e3)):
            camera_timestamp_list.append(timestamp)

    # 如果找到了匹配的相机数据，则返回
    if len(camera_timestamp_list) > 0:
        return camera_timestamp_list
    else:
        return None

def find_matching_sf_sweep_in_lut(scene_flow_lut, target_timestamp_range):
    """
    找到在指定时间戳范围内的 scene flow 数据
    :param scene_flow_lut: scene flow 数据字典，包含文件夹名和时间戳
    :param target_timestamp_range: 目标时间戳范围，单位为微秒。例如：[timestamp_start, timestamp_end]
    :return: 匹配到的所有 scene flow 数据 list，如果没有找到则返回 None

    Args:
        scene_flow_lut:
    """
    sf_list = []
    # 遍历 scene_flow_lut 找到指定时间戳范围内的 数据
    for sf_data in scene_flow_lut:
        timestamp = sf_data['timestamp']
        if target_timestamp_range[0] <= timestamp <= target_timestamp_range[1]:
            sf_list.append(sf_data)

    if len(sf_list) == 1:
        return sf_list[0]
    elif len(sf_list) > 1:
        print("Warning: Multiple matching scene flow data found.")
        return sf_list[0]
    else:
        return None

def find_matching_filename(camera_data_dict, target_timestamp, delta):
    """查找与目标时间戳匹配的文件名"""
    for timestamp in camera_data_dict:
        if 0 < (target_timestamp - timestamp) <= delta:
            return {timestamp: camera_data_dict[timestamp]}
    return None

def project_lidar_to_surround_view_img(cam_info,
                                       lidar_data,
                                       lidar_indices,
                                       min_dist=1.0):
    '''
    将 LiDAR 点云投影到相机视图上
    Args:
        nusc: nuScenes 数据集对象
        lidar_data: lidar 点云数据
        lidar_indices: lidar 点云序号
        lidar_token: lidar sample_data token
        surround_view_img: lidar 帧对应的环视图像
        min_dist: 用于过滤点云的最小距离

    Returns:
        projected_points: 投影到相机视图上的2D点云
        点云在相机坐标系下的深度值和序号
        点云对应的原始图像

    '''
    projected_points = {}

    for channel in cam_info:
        # 加载相机图像
        im = Image.open(cam_info[channel]['image_path'])

        extrinsic = cam_info[channel]['extrinsic']
        intrinsic = cam_info[channel]['intrinsic']

        # turn lidar data to camera coordination
        pc = np.hstack([lidar_data[:, :3], np.ones((lidar_data.shape[0], 1))])
        pc = np.dot(extrinsic, pc.T).T[:, :3]

        # 获取点的深度信息（z轴）
        depths = pc[:, 2]

        # 将点云投影到相机视图上
        points_2d = view_points(pc.T[:3, :], intrinsic, normalize=True)

        # 过滤点云，将不在相机视野范围内的点剔除
        mask = (depths > min_dist) & \
               (points_2d[0, :] > 1) & (points_2d[0, :] < im.size[0] - 1) & \
               (points_2d[1, :] > 1) & (points_2d[1, :] < im.size[1] - 1)

        # 通过 mask 过滤点云、深度和序号
        points_2d = points_2d[:, mask]
        depths = depths[mask]
        filtered_indices = lidar_indices[mask]

        # 将点云在激光雷达坐标系下的坐标也保存下来
        pc_lidar_coord = lidar_data[mask]

        # 将结果存储在字典中
        projected_points[channel] = {
            'points': points_2d,
            'points_lidar_coord': pc_lidar_coord,
            'depths': depths,
            'indices': filtered_indices,
            'original_img': im
        }

    return projected_points


def get_scale_map(nusc,
                  previous_surround_view_data,
                  current_surround_view_data,
                  previous_lidar_data,
                  current_lidar_data,
                  previous_lidar_token,
                  current_lidar_token):
    '''
    获取连续帧 LiDAR 数据在环视图像上的投影，并计算深度比值
    Args:
        nusc: nuScenes 数据集对象
        previous_surround_view_data: 第一帧 LiDAR 数据对应的环视图像
        current_surround_view_data: 第二帧 LiDAR 数据对应的环视图像
        previous_lidar_data: 第一帧 LiDAR 数据，包含点云和序号（手动添加的 int 类型序号）
        current_lidar_data: 第二帧 LiDAR 数据，包含点云和序号 （手动添加的 int 类型序号）
        previous_lidar_token: 第一帧 LiDAR 数据的 sample_data token
        current_lidar_token: 第二帧 LiDAR 数据的 sample_data token

    Returns:
        scale_map: 包含
        1）第一帧lidar点云投影到环视图像上的2D点云
        2）两帧点云在相机坐标系下的深度信息（PV视角下的深度值）
        3）两帧点云深度比值
        的字典
    '''

    # 从 previous_lidar_data 和 current_lidar_data 中提取点云和序号
    previous_pc = previous_lidar_data[:, :3]
    previous_pc_indices = previous_lidar_data[:, 3]
    current_pc = current_lidar_data[:, :3]
    current_pc_indices = current_lidar_data[:, 3]

    # 调用函数投影 previous_lidar_data 和 current_lidar_data
    previous_projected_points = project_lidar_to_surround_view_img(nusc,
                                                                   previous_pc,
                                                                   previous_pc_indices,
                                                                   previous_lidar_token,
                                                                   previous_surround_view_data)
    current_projected_points = project_lidar_to_surround_view_img(nusc,
                                                                  current_pc,
                                                                  current_pc_indices,
                                                                  current_lidar_token,
                                                                  current_surround_view_data)
    scale_map = {}
    for channel in previous_projected_points:
        # 由于投影时通过 mask 进行了筛选，previous 和 current此时的点云数量可能不一致
        # 因此需要通过 indices 进一步筛选，只保留两者都存在的点云
        common_indices = np.intersect1d(previous_projected_points[channel]['indices'],
                                        current_projected_points[channel]['indices'])

        # 通过 indices 筛选点云和深度信息
        prev_mask = np.isin(previous_projected_points[channel]['indices'], common_indices)
        curr_mask = np.isin(current_projected_points[channel]['indices'], common_indices)

        previous_points = previous_projected_points[channel]['points'][:, prev_mask]
        previous_depths = previous_projected_points[channel]['depths'][prev_mask]
        current_points = current_projected_points[channel]['points'][:, curr_mask]
        current_depths = current_projected_points[channel]['depths'][curr_mask]

        previous_xyz = previous_projected_points[channel]['points_lidar_coord'][prev_mask]
        current_xyz = current_projected_points[channel]['points_lidar_coord'][curr_mask]

        ########################### 去掉重叠的点云 ###########################
        previous_points_round = np.round(previous_points).astype(int)
        previous_points_wo_overlap = np.full((1600, 900, 2), np.inf)
        previous_xyz_wo_overlap = np.full((1600, 900, 3), np.inf)

        for i in range(len(previous_depths)):
            u, v = previous_points_round[0, i], previous_points_round[1, i]
            if previous_points_wo_overlap[u, v, 0] > previous_depths[i]:
                # 更新深度信息，在图像上重叠的点保留近处的点云 - 相邻帧深度信息作比值即为scale
                previous_points_wo_overlap[u, v, 0] = previous_depths[i]
                previous_points_wo_overlap[u, v, 1] = current_depths[i]
                # previous_points_wo_overlap[u, v, 2] = scale[i]

                # 更新3D坐标，保留点云在激光雷达坐标系下的坐标，用于制作range image
                previous_xyz_wo_overlap[u, v, :] = previous_xyz[i]

        previous_depth_wo_overlap = previous_points_wo_overlap[:, :, 0]
        current_depth_wo_overlap = previous_points_wo_overlap[:, :, 1]
        # scale_wo_overlap = previous_points_wo_overlap[:, :, 2]

        u, v = np.meshgrid(np.arange(previous_points_wo_overlap.shape[1]),
                           np.arange(previous_points_wo_overlap.shape[0]))

        # np.inf -> 0
        previous_depth_wo_overlap = np.where(np.isinf(previous_depth_wo_overlap), 0, previous_depth_wo_overlap)
        current_depth_wo_overlap = np.where(np.isinf(current_depth_wo_overlap), 0, current_depth_wo_overlap)

        previous_dense_depth_out, _ = design_depth_map.create_map(previous_projected_points[channel]['original_img'],
                                                      previous_depth_wo_overlap,
                                                      max_depth=50.0,
                                                      dilation_kernel_far=design_depth_map.kernels.diamond_kernel_7(),
                                                      dilation_kernel_med=design_depth_map.kernels.diamond_kernel_7(),
                                                      dilation_kernel_near=design_depth_map.kernels.diamond_kernel_7())

        import matplotlib.pyplot as plt

        depth_map_display = previous_dense_depth_out.T
        # Assuming previous_dense_depth_out is defined and has shape (1600, 900)
        plt.figure(figsize=(10, 8))
        plt.title("Dense Depth Map")
        plt.imshow(depth_map_display, cmap='plasma', vmin=0,
                   vmax=50)  # Adjust vmin and vmax based on depth range
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.colorbar(label="Depth (m)")
        plt.tight_layout()
        plt.show()

        current_dense_depth_out, _ = design_depth_map.create_map(previous_projected_points[channel]['original_img'],
                                                               current_depth_wo_overlap,
                                                               max_depth=50.0,
                                                               dilation_kernel_far=design_depth_map.kernels.diamond_kernel_7(),
                                                               dilation_kernel_med=design_depth_map.kernels.diamond_kernel_5(),
                                                               dilation_kernel_near=design_depth_map.kernels.cross_kernel_3())

        valid_mask = (previous_depth_wo_overlap != 0) & (current_depth_wo_overlap != 0)
        # valid_mask = ~np.isinf(depth_wo_overlap) & ~np.isinf(scale_wo_overlap)
        u = u[valid_mask]
        v = v[valid_mask]

        previous_depth_wo_overlap = previous_dense_depth_out[valid_mask]
        current_depth_wo_overlap = current_dense_depth_out[valid_mask]
        scale_wo_overlap = current_depth_wo_overlap / previous_depth_wo_overlap

        previous_points_wo_overlap_ = np.stack([v, u], axis=0)
        previous_xyz_wo_overlap = previous_xyz_wo_overlap[valid_mask]

        scale_map[channel] = {
            'points': previous_points_wo_overlap_,
            'points_lidar_coord': previous_xyz_wo_overlap,
            'depth_1': previous_depth_wo_overlap,
            'depth_3': current_depth_wo_overlap,
            'scale': scale_wo_overlap,
            'original_img': previous_projected_points[channel]['original_img']
        }

        ########################### 去掉重叠的点云 ###########################

        # scale = current_depths / previous_depths
        #
        # scale_map[channel] = {
        #     'points': previous_points,
        #     'depth_1': previous_depths,
        #     'depth_3': current_depths,
        #     'scale': scale,
        #     'original_img': previous_projected_points[channel]['original_img']
        # }

    return scale_map

def depth_estimator(nusc,
                    previous_surround_view_data,
                    current_surround_view_data,
                    previous_lidar_data,
                    current_lidar_data,
                    previous_lidar_token,
                    current_lidar_token,
                    use_metric_depth_estimator,
                    data_idx):
    '''
    获取连续帧 LiDAR 数据在环视图像上的投影，并计算深度比值
    Args:
        nusc: nuScenes 数据集对象
        previous_surround_view_data: 第一帧 LiDAR 数据对应的环视图像
        current_surround_view_data: 第二帧 LiDAR 数据对应的环视图像
        previous_lidar_data: 第一帧 LiDAR 数据，包含点云和序号（手动添加的 int 类型序号）
        current_lidar_data: 第二帧 LiDAR 数据，包含点云和序号 （手动添加的 int 类型序号）
        previous_lidar_token: 第一帧 LiDAR 数据的 sample_data token
        current_lidar_token: 第二帧 LiDAR 数据的 sample_data token

    Returns:
        scale_map: 包含
        1）第一帧lidar点云投影到环视图像上的2D点云
        2）两帧点云在相机坐标系下的深度信息（PV视角下的深度值）
        3）两帧点云深度比值
        的字典
    '''

    # 从 previous_lidar_data 和 current_lidar_data 中提取点云和序号
    previous_pc = previous_lidar_data[:, :3]
    previous_pc_indices = previous_lidar_data[:, 3]
    current_pc = current_lidar_data[:, :3]
    current_pc_indices = current_lidar_data[:, 3]

    # open3d 激光采集点云可视化
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(previous_pc)
    # o3d.visualization.draw_geometries([pcd])


    # 调用函数投影 previous_lidar_data 和 current_lidar_data
    previous_projected_points = project_lidar_to_surround_view_img(nusc,
                                                                   previous_pc,
                                                                   previous_pc_indices,
                                                                   previous_lidar_token,
                                                                   previous_surround_view_data)
    current_projected_points = project_lidar_to_surround_view_img(nusc,
                                                                  current_pc,
                                                                  current_pc_indices,
                                                                  current_lidar_token,
                                                                  current_surround_view_data)

    previous_lidar_points_fake = generate_pseduo_point_cloud(nusc,
                                                            previous_lidar_token,
                                                            previous_surround_view_data,
                                                            previous_projected_points,
                                                            use_metric_depth_estimator,
                                                            data_idx)
    current_lidar_points_fake = generate_pseduo_point_cloud(nusc,
                                                            current_lidar_token,
                                                            current_surround_view_data,
                                                            current_projected_points,
                                                            use_metric_depth_estimator,
                                                            data_idx)
    return previous_lidar_points_fake, current_lidar_points_fake


def generate_pseduo_point_cloud(cam_info,
                                projected_points,
                                depth_model,
                                use_metric_depth_estimator,
                                data_idx):
    
    fake_depth = {}
    plot_depth_analysis = True
    
    if use_metric_depth_estimator:

        metric_depth_anything = depth_model

        # TODO: 实现深度+图像生成伪激光点云
        for idx, channel in enumerate(projected_points):
            # 读取图像并进行深度推理
            raw_image = np.array(projected_points[channel]['original_img'])
       
            depth_pred = metric_depth_anything.infer_image(raw_image)           
            # 获取激光测量的深度与对应像素坐标
            depth_meas = projected_points[channel]['depths']
            u = projected_points[channel]['points'][0, :].astype(np.int32)
            v = projected_points[channel]['points'][1, :].astype(np.int32)
            # 取出对应像素的预测深度值
            depth_pred_masked = depth_pred[v, u]
            # 准备数据：X 为 dpred, y 为 1/dmeas（这里利用1/dmeas进行线性拟合）
            X = depth_pred_masked.reshape(-1, 1)
            y = depth_meas
            # 使用 RANSAC 拟合线性模型： y = alpha * x + beta
            ransac = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=0.5,
                residual_threshold=1.0,
                max_trials=100
            )
            ransac.fit(X, y)
            alpha = ransac.estimator_.coef_[0]
            beta = ransac.estimator_.intercept_
            depth_meas_sup = alpha * depth_pred + beta
            depth_meas_mask = (depth_meas_sup > 0) & (depth_meas_sup < 80)
            x, y = projected_points[channel]['points'][[1,0]].astype(np.int32)

            new_depth = np.zeros_like(depth_meas_sup)
            new_depth[depth_meas_mask] = depth_meas_sup[depth_meas_mask]
            new_depth[x, y] = projected_points[channel]['depths']
            new_depth[new_depth > 80] = 0

            fake_depth[channel] = new_depth
            # new_depth = (new_depth / np.max(new_depth) * 255).astype(np.uint8)
            # original_depth = np.zeros_like(depth_meas_sup)
            
            # original_depth[x,y] = projected_points[channel]['depths']
            # original_depth[original_depth > 80] = 80

            # original_depth = (original_depth / np.max(original_depth) * 255).astype(np.uint8)
            
            # # repeat depth to 3 channels
            # new_depth = np.repeat(new_depth[:, :, np.newaxis], 3, axis=2)
            # original_depth = np.repeat(original_depth[:, :, np.newaxis], 3, axis=2)

            # images = np.concatenate((raw_image ,new_depth, original_depth), axis=0)
            # cv2.imwrite(f'./metric_depth_{channel}.png', images)

        # lidar_points_fake = np.concatenate([lidar_points_supplement[channel] 
        #                                     for channel in lidar_points_supplement], axis=0)
        # # lidar_points_fake = np.concatenate((lidar_points_supplement['CAM_FRONT'], lidar_points_supplement['CAM_BACK']), axis=0)    
        # # open3d 点云可视化
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(lidar_points_fake)
        # o3d.visualization.draw_geometries([pcd])

        # 绘制深度分析图    
        if plot_depth_analysis:
            save_path_prefix = f'depthanything/metric_depth/depth_analysis'
            if not os.path.exists(save_path_prefix):
                os.makedirs(save_path_prefix)
            # 第一个图：dpred vs dmeas
            fig, axes = plt.subplots(2, 3, figsize=(16, 12))
            for idx, channel in enumerate(projected_points):
                # 取到子图对象
                ax = axes[idx // 3, idx % 3]
                # 读取原图像并推理深度
                raw_image = np.array(projected_points[channel]['original_img'])
                depth_pred = metric_depth_anything.infer_image(raw_image)
                # 获取激光测量的深度与像素坐标
                depth_meas = projected_points[channel]['depths']
                u = projected_points[channel]['points'][0, :].astype(np.int32)
                v = projected_points[channel]['points'][1, :].astype(np.int32)
                # 取出预测深度中对应坐标的深度值
                depth_pred_masked = depth_pred[v, u]
                # 在子图上绘制散点图
                ax.scatter(depth_pred_masked, depth_meas, c='blue', s=1, alpha=0.5)
                ax.set_xlabel('dpred', fontsize=16)
                ax.set_ylabel('dmeas', fontsize=16)
                ax.set_title(f'{channel}', fontsize=18)
                ax.grid(True)
            # 调整子图布局，防止标题和坐标轴重叠
            plt.tight_layout()
            plt.savefig(f"{save_path_prefix}/depth_analysis_{data_idx}_0.png")
            plt.close(fig)  # 关闭图像

            # 第二个图：dpred vs dmeas（RANSAC拟合）
            fig, axes = plt.subplots(2, 3, figsize=(16, 12))
            for idx, channel in enumerate(projected_points):
                ax = axes[idx // 3, idx % 3]
                # 读取图像并进行深度推理
                raw_image = np.array(projected_points[channel]['original_img'])
                depth_pred = metric_depth_anything.infer_image(raw_image)           
                # 获取激光测量的深度与对应像素坐标
                depth_meas = projected_points[channel]['depths']
                u = projected_points[channel]['points'][0, :].astype(np.int32)
                v = projected_points[channel]['points'][1, :].astype(np.int32)
                # 取出对应像素的预测深度值
                depth_pred_masked = depth_pred[v, u]
                # 准备数据：X 为 dpred, y 为 1/dmeas（这里利用1/dmeas进行线性拟合）
                X = depth_pred_masked.reshape(-1, 1)
                y = depth_meas
                # 使用 RANSAC 拟合线性模型： y = alpha * x + beta
                ransac = RANSACRegressor(
                    estimator=LinearRegression(),
                    min_samples=0.5,
                    residual_threshold=1.0,
                    max_trials=100
                )
                ransac.fit(X, y)
                alpha = ransac.estimator_.coef_[0]
                beta = ransac.estimator_.intercept_
                print(f"{channel}: 拟合得到的模型: depth_meas ≈ {alpha:.3f} * depth_pred + {beta:.3f}")
                # 计算拟合直线的预测值
                y_fit = ransac.predict(X)

                ######################区分内点、离群点######################
                # # 获取内点和离群点的掩码
                # inlier_mask = ransac.inlier_mask_
                # outlier_mask = np.logical_not(inlier_mask)
                # # 分别绘制内点和离群点（内点蓝色，离群点红色）
                # ax.scatter(X[inlier_mask], y[inlier_mask], color='blue', s=1, alpha=0.5, label='inliers')
                # ax.scatter(X[outlier_mask], y[outlier_mask], color='red', s=1, alpha=0.5, label='outliers')
                ##########################################################
                
                ######################不区分内点、离群点######################
                # 绘制散点图（内点、外点不分颜色，这里直接显示所有数据点）
                ax.scatter(X, y, color='blue', s=1, alpha=0.5, label='data')  
                ###########################################################

                # 为了绘制平滑的拟合线，对X排序后画出拟合直线
                sorted_indices = np.argsort(X.flatten())
                ax.plot(X.flatten()[sorted_indices], y_fit[sorted_indices],
                        color='green', linewidth=2, label='RANSAC fit')           
                ax.set_xlabel('depth_pred', fontsize=16)
                ax.set_ylabel('depth_meas', fontsize=16)
                ax.set_title(f'{channel}', fontsize=18)
                ax.grid(True)
                ax.legend(fontsize=12)
            # 调整子图布局
            plt.tight_layout()
            # 保存拼接后的图像到指定路径
            plt.savefig(f"{save_path_prefix}/depth_analysis_{data_idx}_2.png")
            plt.close(fig)
            import pdb; pdb.set_trace()
    else:
        depth_anything = depth_model

        for idx, channel in enumerate(projected_points):
            # 读取图像并进行深度推理
            raw_image = np.array(projected_points[channel]['original_img'])
            depth_pred = depth_anything.infer_image(raw_image)           
            # 获取激光测量的深度与对应像素坐标
            depth_meas = projected_points[channel]['depths']
            u = projected_points[channel]['points'][0, :].astype(np.int32)
            v = projected_points[channel]['points'][1, :].astype(np.int32)
            # 取出对应像素的预测深度值
            depth_pred_masked = depth_pred[v, u]
            # 准备数据：X 为 dpred, y 为 1/dmeas（这里利用1/dmeas进行线性拟合）
            X = depth_pred_masked.reshape(-1, 1)
            y = 1 / depth_meas
            # 使用 RANSAC 拟合线性模型： y = alpha * x + beta
            ransac = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=0.5,
                residual_threshold=1.0,
                max_trials=100
            )

            ransac.fit(X, y)
            alpha = ransac.estimator_.coef_[0]
            beta = ransac.estimator_.intercept_
            depth_meas_sup = 1 / (alpha * depth_pred + beta)

            depth_meas_mask = (depth_meas_sup > 0) & (depth_meas_sup < 80)
            x, y = projected_points[channel]['points'][[1,0]].astype(np.int32)

            new_depth = np.zeros_like(depth_meas_sup)
            new_depth[depth_meas_mask] = depth_meas_sup[depth_meas_mask]
            new_depth[x, y] = projected_points[channel]['depths']
            new_depth[new_depth > 80] = 0

            fake_depth[channel] = new_depth
            new_depth = (new_depth / np.max(new_depth) * 255).astype(np.uint8)
            original_depth = np.zeros_like(depth_meas_sup)
            
            original_depth[x,y] = projected_points[channel]['depths']
            original_depth[original_depth > 80] = 80

            original_depth = (original_depth / np.max(original_depth) * 255).astype(np.uint8)
            
            # repeat depth to 3 channels
            new_depth = np.repeat(new_depth[:, :, np.newaxis], 3, axis=2)
            original_depth = np.repeat(original_depth[:, :, np.newaxis], 3, axis=2)

            images = np.concatenate((raw_image ,new_depth, original_depth), axis=0)
            cv2.imwrite(f'./depth_{channel}.png', images)


            # lidar_points_sup = project_surround_view_to_lidar(nusc,
            #                                                   lidar_token,
            #                                                   surround_view_data,
            #                                                   channel,
            #                                                   depth_meas_sup,
            #                                                   depth_meas_mask)
            # lidar_points_supplement[channel] = lidar_points_sup

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(lidar_points_sup)
            # o3d.visualization.draw_geometries([pcd])

        # lidar_points_fake = np.concatenate([lidar_points_supplement[channel] 
        #                                     for channel in lidar_points_supplement], axis=0)
        # # lidar_points_fake = np.concatenate((lidar_points_supplement['CAM_FRONT'], lidar_points_supplement['CAM_BACK']), axis=0)    
        # # open3d 点云可视化
        # # pcd = o3d.geometry.PointCloud()
        # # pcd.points = o3d.utility.Vector3dVector(lidar_points_fake)
        # # o3d.visualization.draw_geometries([pcd])

        # 绘制深度分析图
        if plot_depth_analysis:
            save_path_prefix = f'depthanything/depth_analysis'
            if not os.path.exists(save_path_prefix):
                os.makedirs(save_path_prefix)
            #  第一个图：dpred vs dmeas
            fig, axes = plt.subplots(2, 3, figsize=(16, 12))
            for idx, channel in enumerate(projected_points):
                # 取到子图对象
                ax = axes[idx // 3, idx % 3]
                # 读取原图像并推理深度
                raw_image = np.array(projected_points[channel]['original_img'])
                depth_pred = depth_anything.infer_image(raw_image)
                # 获取激光测量的深度与像素坐标
                depth_meas = projected_points[channel]['depths']
                u = projected_points[channel]['points'][0, :].astype(np.int32)
                v = projected_points[channel]['points'][1, :].astype(np.int32)
                # 取出预测深度中对应坐标的深度值
                depth_pred_masked = depth_pred[v, u]
                # 在子图上绘制散点图
                ax.scatter(depth_pred_masked, depth_meas, c='blue', s=1, alpha=0.5)
                ax.set_xlabel('dpred', fontsize=16)
                ax.set_ylabel('dmeas', fontsize=16)
                ax.set_title(f'{channel}', fontsize=18)
                ax.grid(True)
            # 调整子图布局，防止标题和坐标轴重叠
            plt.tight_layout()
            plt.savefig(f"{save_path_prefix}/depth_analysis_{data_idx}_0.png")
            plt.close(fig)  # 关闭图像
            
            # 第二个图：dpred vs 1/dmeas
            fig, axes = plt.subplots(2, 3, figsize=(16, 12))
            for idx, channel in enumerate(projected_points):
                ax = axes[idx // 3, idx % 3]
                raw_image = np.array(projected_points[channel]['original_img'])
                depth_pred = depth_anything.infer_image(raw_image)
                depth_meas = projected_points[channel]['depths']
                u = projected_points[channel]['points'][0, :].astype(np.int32)
                v = projected_points[channel]['points'][1, :].astype(np.int32)
                depth_pred_masked = depth_pred[v, u]
                ax.scatter(depth_pred_masked, 1/depth_meas, c='blue', s=1, alpha=0.5)
                ax.set_xlabel('dpred', fontsize=16)
                ax.set_ylabel('1/dmeas', fontsize=16)
                ax.set_title(f'{channel}', fontsize=18)
                ax.grid(True)
            plt.tight_layout()
            plt.savefig(f"{save_path_prefix}/depth_analysis_{data_idx}_1.png")
            plt.close(fig)

            # 第三个图：dpred vs 1/dmeas（RANSAC拟合）
            fig, axes = plt.subplots(2, 3, figsize=(16, 12))
            for idx, channel in enumerate(projected_points):
                ax = axes[idx // 3, idx % 3]
                # 读取图像并进行深度推理
                raw_image = np.array(projected_points[channel]['original_img'])
                depth_pred = depth_anything.infer_image(raw_image)           
                # 获取激光测量的深度与对应像素坐标
                depth_meas = projected_points[channel]['depths']
                u = projected_points[channel]['points'][0, :].astype(np.int32)
                v = projected_points[channel]['points'][1, :].astype(np.int32)
                # 取出对应像素的预测深度值
                depth_pred_masked = depth_pred[v, u]
                # 准备数据：X 为 dpred, y 为 1/dmeas（这里利用1/dmeas进行线性拟合）
                X = depth_pred_masked.reshape(-1, 1)
                y = 1 / depth_meas
                # 使用 RANSAC 拟合线性模型： y = alpha * x + beta
                ransac = RANSACRegressor(
                    estimator=LinearRegression(),
                    min_samples=0.5,
                    residual_threshold=1.0,
                    max_trials=100
                )
                ransac.fit(X, y)
                alpha = ransac.estimator_.coef_[0]
                beta = ransac.estimator_.intercept_
                print(f"{channel}: 拟合得到的模型: 1/depth_meas ≈ {alpha:.3f} * depth_pred + {beta:.3f}")
                # 计算拟合直线的预测值

                depth_meas_sup = 1 / (alpha * depth_pred_masked + beta)

                ######################区分内点、离群点######################
                # # 获取内点和离群点的掩码
                # inlier_mask = ransac.inlier_mask_
                # outlier_mask = np.logical_not(inlier_mask)
                # # 分别绘制内点和离群点（内点蓝色，离群点红色）
                # ax.scatter(X[inlier_mask], y[inlier_mask], color='blue', s=1, alpha=0.5, label='inliers')
                # ax.scatter(X[outlier_mask], y[outlier_mask], color='red', s=1, alpha=0.5, label='outliers')
                ##########################################################
                
                ######################不区分内点、离群点######################
                # 绘制散点图（内点、外点不分颜色，这里直接显示所有数据点）
                # ax.scatter(depth_meas, depth_meas_sup, color='blue', s=1, alpha=0.5, label='data')  
                ###########################################################

                near_range_mask = (depth_meas < 20) & (depth_meas_sup > 0) 
                depth_meas = depth_meas[near_range_mask]
                depth_meas_sup = depth_meas_sup[near_range_mask] 
                # 计算量化后的影响
                std = np.std(depth_meas - depth_meas_sup)
                print(f"cam: {channel}, var: {std}")
                
                # 为了绘制平滑的拟合线，对X排序后画出拟合直线
                sorted_indices = np.argsort(depth_meas)
                ax.scatter(depth_meas_sup[sorted_indices], depth_meas[sorted_indices],
                        color='green', linewidth=2, label='RANSAC fit')           
                ax.set_xlabel('depth_pred', fontsize=16)
                ax.set_ylabel('depth_meas', fontsize=16)
                ax.set_title(f'{channel}', fontsize=18)
                ax.grid(True)
                ax.legend(fontsize=12)
            # 调整子图布局
            plt.tight_layout()
            # 保存拼接后的图像到指定路径
            plt.savefig(f"{save_path_prefix}/depth_analysis_{data_idx}_2.png")
            plt.close(fig)
            import pdb; pdb.set_trace()
    # (N,3) 的点云
    return fake_depth

def project_surround_view_to_lidar(nusc,
                                   lidar_token,
                                   surround_view_data,
                                   cam_channel,
                                   depth_map,
                                   mask):
    """
    将环视相机图像及其深度图反投影回 LiDAR 坐标系。

    参数:
        nusc:        nuScenes 数据集对象实例。
        lidar_token:  LiDAR 数据的 token。
        surround_view_data: 包含相机通道数据的字典，
                                     例如 {'CAM_FRONT': {'token': '...'}, ...}。
        image:       numpy.ndarray, 形状 (H, W, 3)，环视相机图像。
        depth_map:   numpy.ndarray, 形状 (H, W)，对应图像的深度图。

    返回:
        numpy.ndarray, 形状 (N, 3)，表示反投影到 LiDAR 坐标系的点云。
    """

    try:
        lidar_sd = nusc.get('sample_data', lidar_token)
    except Exception as e:
        raise ValueError(f"无法根据 token 获取 LiDAR 帧数据: {e}")   
    # 提取 LiDAR 的 ego_pose 和 传感器标定信息
    lidar_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    lidar_calib = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])

    channel = cam_channel
    cam_sd = nusc.get('sample_data', surround_view_data[channel])

    # 提取相机的 ego_pose 和 传感器标定信息（包括内参）
    cam_pose = nusc.get('ego_pose', cam_sd['ego_pose_token'])
    cam_calib = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])

    points_3d = unview_points(depth=depth_map,
                              view=cam_calib['camera_intrinsic'],
                              mask=mask,
                              normalize=True)
    points_cam = points_3d.T
    # 构造旋转矩阵和平移向量（四元数转旋转矩阵）
    # 相机传感器外参: 从相机坐标系 -> 车辆坐标系 (相机时刻)
    cam_R = Quaternion(cam_calib['rotation']).rotation_matrix    # 3x3
    cam_t = np.array(cam_calib['translation'])  # 形状 (3,)
    # 第二步：相机坐标系 -> 相机所在的自车坐标系
    points_ego_cam = points_cam.dot(cam_R.T) + cam_t  # 应用旋转和平移
    # 相机的自车姿态: 从车辆坐标系 (相机时刻) -> 全局坐标系
    ego_cam_R = Quaternion(cam_pose['rotation']).rotation_matrix  # 3x3
    ego_cam_t = np.array(cam_pose['translation'])
    # 第三步：车辆坐标系 (相机时刻) -> 全局坐标系
    points_global = points_ego_cam.dot(ego_cam_R.T) + ego_cam_t
    # LiDAR 的自车姿态: 从车辆坐标系 (LiDAR 时刻) -> 全局坐标系
    ego_lidar_R = Quaternion(lidar_pose['rotation']).rotation_matrix  # 3x3
    ego_lidar_t = np.array(lidar_pose['translation'])
    # 第四步：全局坐标系 -> 车辆坐标系 (LiDAR 时刻) 
    # （注意这是第三步的逆，因此使用旋转矩阵的转置，和平移的相反）
    points_ego_lidar = (points_global - ego_lidar_t).dot(ego_lidar_R)
    # LiDAR 传感器外参: 从 LiDAR 传感器坐标系 -> 车辆坐标系 (LiDAR 时刻)
    lidar_R = Quaternion(lidar_calib['rotation']).rotation_matrix  # 3x3
    lidar_t = np.array(lidar_calib['translation'])
    # 第五步：车辆坐标系 (LiDAR 时刻) -> LiDAR 传感器坐标系
    # （这是第一步的逆，同样使用旋转转置和相反平移）
    points_lidar = (points_ego_lidar - lidar_t).dot(lidar_R.T)
    return points_lidar

def unview_points(depth: np.ndarray, view: np.ndarray, mask: np.ndarray, normalize: bool) -> np.ndarray:
    """
    反投影函数：将深度图中 mask 选中的像素点投影到3D空间。
    
    假设输入的 depth 是 (H, W) 的深度图，与原图像素点一一对应，
    mask 是形状 (H, W) 的布尔数组，True 表示选中该像素进行反投影。
    如果 normalize 为 True，说明原 view_points 函数中对齐次坐标归一化（除以第三行得到 (u, v, 1)），
    此时恢复前需要构造齐次坐标：[u*d, v*d, d]；若 normalize 为 False，则认为 2D 坐标已经是未归一化的 (x, y)，
    与深度 d 结合构成 (x, y, d)。
    
    参数:
      depth: np.ndarray, 形状为 (H, W)，每个像素对应的深度值。
      view: np.ndarray, 投影矩阵，其 shape 小于等于 (4, 4)。
      normalize: bool，与 view_points 中的 normalize 参数对应。
      mask: np.ndarray, 布尔数组，形状为 (H, W)，指示需要反投影的像素。
    
    返回:
      np.ndarray, 形状为 (3, n)，恢复出的3D点坐标，其中 n 为 mask 中 True 的像素数。
    """
    # 确保 view 为 np.array 并构造 4x4 投影矩阵
    view = np.array(view)
    viewpad = np.eye(4)
    viewpad[0, 0] = view[0, 0]   # fx
    viewpad[0, 2] = view[0, 2]   # cx
    viewpad[1, 1] = view[1, 1]   # fy
    viewpad[1, 2] = view[1, 2]   # cy

    # 计算投影矩阵的逆，用于反变换
    inv_viewpad = np.linalg.inv(viewpad)

    # 获取深度图尺寸
    H, W = depth.shape
    # 生成网格坐标
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
    # 仅选取 mask 为 True 的像素点
    u_selected = u_grid[mask]  # shape: (n,)
    v_selected = v_grid[mask]  # shape: (n,)
    # 生成 2D 点，形状 (2, n)
    points_2d = np.stack((u_selected, v_selected), axis=0)

    # 选取对应的深度值，形状 (n,)
    depth_selected = depth[mask]
    # 转换为 (1, n)
    depth_flat = depth_selected.reshape(1, -1)

    nbr_points = points_2d.shape[1]
    # 构造齐次坐标，根据 normalize 参数选择构造方式
    if normalize:
        # 构造 [u*d, v*d, d]
        points_hom = np.concatenate((points_2d * depth_flat, depth_flat), axis=0)  # (3, n)
    else:
        # 构造 (x, y, d)
        points_hom = np.concatenate((points_2d, depth_flat), axis=0)

    # 添加一行全 1，构成齐次坐标 (4, n)
    points_hom = np.concatenate((points_hom, np.ones((1, nbr_points))), axis=0)

    # 应用投影矩阵的逆进行反变换
    points_3d_hom = np.dot(inv_viewpad, points_hom)
    points_3d = points_3d_hom[:3, :]

    return points_3d
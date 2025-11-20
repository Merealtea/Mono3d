import os
import cv2
import pickle
import numpy as np
import os.path as osp
import sys
from PIL import Image
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from utils.nusc_lidar_cam_match import project_lidar_to_surround_view_img, generate_pseduo_point_cloud
from utils.draw import gt_scale_2_rgb
from utils.undistortion import distort_fisheye_image, interpolate_depth
import matplotlib.pyplot as plt
import torch
sys.path.append("/mnt/pool1/cxy/Depth-Anything-V2")
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2 as DepthAnythingV2_metric
from depth_anything_v2.dpt import DepthAnythingV2

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets.basedataset import BaseDataset
import opencood.utils.pcd_utils as pcd_utils
from pyquaternion import Quaternion
from time import time
from copy import deepcopy
import sys
sys.path.append("/mnt/pool1/cxy/Mono3d")
sys.path.append("/mnt/pool1/cxy/Depth-Anything-V2")
from configs.FisheyeParam.cam_model import CamModel
from configs.FisheyeParam.lidar_model import Lidar_transformation

def transform_matrix(translation: list, rotation: Quaternion) -> np.ndarray:
    """创建4x4的变换矩阵"""
    tm = np.eye(4)
    tm[:3, :3] = rotation.rotation_matrix
    tm[:3, 3] = np.array(translation)
    return tm

def load_one_cav_info(cav_path, scene, scene_type, depth_anything_model, use_metric_depth_estimator):
    yaml_files = \
            sorted([os.path.join(cav_path, x)
                    for x in os.listdir(cav_path) if
                    x.endswith('.yaml') and 'additional' not in x])
    timestamps = BaseDataset.extract_timestamps(yaml_files)
    cav_infos = []

    data_root, scene_path = cav_path.split(scene_type + '/')
    depth_result_path = os.path.join(data_root, 'depth_result', scene_type, scene_path)

    if not os.path.exists(depth_result_path):
        os.makedirs(depth_result_path)
    
    for time_idx, timestamp in enumerate(timestamps):
        cam_info = dict()

        yaml_file = os.path.join(cav_path,
                                    timestamp + '.yaml')

        config_params = load_yaml(yaml_file)

        for idx, cam in enumerate(['cam1', 'cam2', 'cam3', 'cam4']):
            cam_info[cam] = {}
            cam_info[cam]['image_path'] = os.path.join(cav_path, timestamp + f'_{cam}.jpeg')
            cam_info[cam]['intrinsic'] = np.array(config_params[cam]['intrinsic'])
            cam_info[cam]['extrinsic'] = np.linalg.inv(np.array(config_params[cam]['extrinsic']))
            cam_info[cam]['shape'] = cv2.imread(cam_info[cam]['image_path']).shape[:2]

        lidar_filename = os.path.join(cav_path, timestamp + '.bin')

        lidar_data = pcd_utils.load_lidar_bin(
                    lidar_filename,
                    zero_intensity=True)

        lidar_data = lidar_data[:, :3]

        lidar_data_with_indices = np.hstack([lidar_data, np.arange(lidar_data.shape[0]).reshape(-1, 1)])


        current_projected_points = project_lidar_to_surround_view_img(cam_info,
                                               lidar_data,
                                               lidar_data_with_indices)
        data_idx = 1
        current_lidar_points_fake = generate_pseduo_point_cloud(cam_info,
                                                        current_projected_points,
                                                        depth_anything_model,
                                                        use_metric_depth_estimator,
                                                        data_idx)
        
        for cam_name in current_lidar_points_fake:
            depth_file_path = os.path.join(depth_result_path, "{}_{}.npy".format(timestamp, cam_name))
            np.save(depth_file_path, current_lidar_points_fake[cam_name])

        import pdb; pdb.set_trace()
    return cav_infos



def generate_fake_depth_estimation_v2x_real(data_root, scene_type, depth_anything_model, use_metric_depth_estimator):
    infos = list()
    scene_path = os.path.join(data_root, scene_type)

    scenes = os.listdir(scene_path)
    for scene in tqdm(scenes):
        cavs = os.listdir(os.path.join(scene_path, scene))
        for cav in cavs:
            if int(cav) < 0:
                continue
            cav_path = os.path.join(scene_path, scene, cav)
            cav_infos = load_one_cav_info(cav_path, scene, scene_type, depth_anything_model, use_metric_depth_estimator)
            infos.extend(cav_infos)
        break
    return infos

def generate_fake_depth_estimation_nuscene(nusc, scenes, depth_anything_model, use_metric_depth_estimator):
    
    depth_result_path = nusc.dataroot.replace('nuscenes', 'depth_result')
    for cur_scene in tqdm(nusc.scene):
        if cur_scene['name'] not in scenes:
            continue
        first_sample_token = cur_scene['first_sample_token']
        cur_sample = nusc.get('sample', first_sample_token)
        while True: 
            st = time()
            cam_names = [
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
            ]

            lidar_sd_rec = nusc.get('sample_data', cur_sample['data']['LIDAR_TOP'])
            lidar_cs_record = nusc.get('calibrated_sensor',
                                lidar_sd_rec['calibrated_sensor_token'])
            lidar_pose_record = nusc.get('ego_pose', lidar_sd_rec['ego_pose_token'])


            # 构造LiDAR到ego的变换矩阵
            lidar_to_lego = transform_matrix(lidar_cs_record['translation'], Quaternion(lidar_cs_record['rotation']))

            # 构造ego到全局的变换矩阵
            lego_to_global = transform_matrix(lidar_pose_record['translation'], Quaternion(lidar_pose_record['rotation']))

            # lidar to global
            lidar_2_global = lego_to_global @ lidar_to_lego 

            cam_info = {}

            for cam_name in cam_names:
                cam_sd = nusc.get('sample_data',
                                    cur_sample['data'][cam_name])
            

                cam_info[cam_name] = {}
                cam_info[cam_name]['image_path'] = os.path.join(
                    nusc.dataroot, cam_sd['filename'])
                # key frame info
                cam_cs_record = nusc.get(
                    'calibrated_sensor', cam_sd['calibrated_sensor_token'])
                cam_pose_record = nusc.get(
                    'ego_pose', cam_sd['ego_pose_token'])
                
                # intrinsic matrix
                cam_info[cam_name]['intrinsic'] = np.eye(4)
                cam_info[cam_name]['intrinsic'][:3, :3] = np.array(cam_cs_record['camera_intrinsic'])

                cam_to_cego = transform_matrix(cam_cs_record['translation'], Quaternion(cam_cs_record['rotation']))
                cego_to_global = transform_matrix(cam_pose_record['translation'], Quaternion(cam_pose_record['rotation']))
            
                # extrinsic matrix
                # lidar_to_cam
                cam_info[cam_name]['extrinsic'] = np.linalg.inv(cam_to_cego) @ np.linalg.inv(cego_to_global) @ lidar_2_global

                if not os.path.exists(os.path.join(depth_result_path, cam_name)):
                    os.makedirs(os.path.join(depth_result_path, cam_name))

            # 从 nuscenes 中获取激光雷达数据
            lidar_filename = os.path.join(nusc.dataroot, lidar_sd_rec['filename'])
            lidar_data = np.fromfile(lidar_filename, dtype=np.float32).reshape(-1, 5)

            lidar_data = lidar_data[:, :3]


            # import pdb; pdb.set_trace()
            # # Turn Lidar into lidar coordination
            # lidar_data = np.hstack([lidar_data, np.ones((lidar_data.shape[0], 1))])
            # lidar_data = np.dot(np.linalg.inv(lidar_2_global), lidar_data.T).T[:, :3]
            lidar_data_with_indices = np.hstack([lidar_data, np.arange(lidar_data.shape[0]).reshape(-1, 1)])


            current_projected_points = project_lidar_to_surround_view_img(cam_info,
                                               lidar_data,
                                               lidar_data_with_indices)
            data_idx = 1
            current_lidar_points_fake = generate_pseduo_point_cloud(cam_info,
                                                            current_projected_points,
                                                            depth_anything_model,
                                                            use_metric_depth_estimator,
                                                            data_idx)
            
            for cam_name in cam_info:
                img_path = cam_info[cam_name]['image_path']
                file_name = os.path.basename(img_path).split('.')[0]
                depth_file_path = os.path.join(depth_result_path, cam_name, file_name + ".npy")
                print(depth_file_path)
                np.save(depth_file_path, current_lidar_points_fake[cam_name])

            if cur_sample['next'] == '':
                break
            else:
                cur_sample = nusc.get('sample', cur_sample['next'])
                
            print("Time: ", time() - st)

def generate_fake_depth_estimation_hycan(hycan_path, depth_anything_model, use_metric_depth_estimator):
    infos = list()
    lidar_path =  os.path.join(hycan_path, 'lidar')
    cam_path = os.path.join(hycan_path, 'images')
    
    directions = ['left', 'right', 'front', 'back']

    cam_models = dict(zip(directions, [CamModel(direction, "Hycan", "numpy") for direction in directions]))
    lidar_model = Lidar_transformation("Hycan")
    for lidar_file in os.listdir(lidar_path):
        timestamp = lidar_file.split('.npy')[0]
        lidar_points = np.load(os.path.join(lidar_path, lidar_file))[:,:3]

        lidar_points = lidar_model.lidar_to_rear(lidar_points.T)
        projected_points = {}
        remap_xy = {}
        for direction in directions:
            image_file = os.path.join(cam_path, direction, timestamp + ".jpg")
            cam_model = cam_models[direction]
            K = np.array([[float(cam_model.fcous_distance[1]), 0, cam_model.img_center[0,0]],
                      [0, float(cam_model.fcous_distance[0]), cam_model.img_center[1,0]],
                      [0, 0, 1]])
            D = cam_model.radial_param

            # 图像去畸变
            
            image = cv2.imread(image_file)
            distorted_image, mapx, mapy = distort_fisheye_image(image, K, D)

            new_image_coordination = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
            new_image_coordination = np.stack(new_image_coordination, axis=-1)

            # 点云反投影到图像上
            cam_points = cam_model.world2cam(lidar_points)
            # 过滤点云，将不在相机视野范围内的点剔除
            depths = cam_points[0, :]
            points_2d = cam_model.cam2image(cam_points, filter_points=False)

            min_dist=0.1

            mask = (depths > min_dist) & \
               (points_2d[0, :] > 1) & (points_2d[0, :] < image.shape[1] - 1) & \
               (points_2d[1, :] > 1) & (points_2d[1, :] < image.shape[0] - 1)
            
            points_2d = points_2d[:, mask]
            depths = depths[mask]


            # Draw the depth image into original image
            # depth_image = deepcopy(image)
            # # 使用 OpenCV 的 applyColorMap 将深度值映射到彩虹色
            # normalized_depth = np.clip(depths / 10, 0, 1) * 255
            # normalized_depth = np.uint8(normalized_depth)

            # color = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
            # # draw different depth points with different colors
            # for point, c in zip(points_2d.T, color):
            #     # import pdb; pdb.set_trace()
            #     cv2.circle(depth_image, (int(point[0]), int(point[1])), 1, tuple(int(x) for x in c[0]), -1)

            # cv2.imwrite("./depth_image.jpg", depth_image)

            # 计算去畸变后的点云坐标
            backward_map = np.zeros((image.shape[0], image.shape[1], 2))
            backward_map[mapy.astype(int), mapx.astype(int)] = new_image_coordination


            new_pixel_points = backward_map[points_2d[1].astype(int), points_2d[0].astype(int)]

            invalid_mask = np.logical_and(new_pixel_points[:, 0] == 0, new_pixel_points[:, 1] == 0)
            new_pixel_points = new_pixel_points[~invalid_mask]
            depths = depths[~invalid_mask]

            
            
            import pdb; pdb.set_trace()

            projected_points[direction] = {
                "points" : new_pixel_points.T,
                "depths" : depths,
                "indices" : np.arange(depths.shape[0]),
                "original_img" : distorted_image
            }

            remap_xy[direction] = {
                "mapx" : mapx,
                "mapy" : mapy
            }

        fake_depth = generate_pseduo_point_cloud("",
                                                projected_points,
                                                depth_anything_model,
                                                use_metric_depth_estimator,
                                                data_idx=1)
        import pdb; pdb.set_trace()
        for direction in fake_depth:
            distort_fisheye_image_depth = fake_depth[direction]
            mapx = remap_xy[direction]['mapx']
            mapy = remap_xy[direction]['mapy']
            final_depth = interpolate_depth(distort_fisheye_image_depth, mapx, mapy, image.shape[:2])
            depth = (final_depth / np.max(final_depth) * 255).astype(np.uint8)
            cv2.imwrite("./depth.jpg", depth)
            import pdb; pdb.set_trace()
            
            

if __name__ == "__main__":
    # NuScenes 数据集路径
    # 相对深度估计模型
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    use_metric_depth_estimator = False
    if use_metric_depth_estimator:
        depth_anything = DepthAnythingV2_metric(**{**model_configs[encoder], 'max_depth': 80})
        depth_anything.load_state_dict(torch.load(f'../../Depth-Anything-V2/checkpoints/depth_anything_v2_metric_vkitti_{encoder}.pth',map_location='cpu'))
        depth_anything.to('cuda').eval()

    else:
        depth_anything = DepthAnythingV2(**model_configs[encoder])
        depth_anything.load_state_dict(torch.load(f'../../Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth',map_location='cpu'))
        depth_anything.to('cuda').eval()

    dataset_type = 'nuscenes' # 'nuscenes' or 'v2x_real' or 'Hycan'
    if dataset_type == 'nuscenes':
        nusc = NuScenes(version='v1.0-mini',
                    dataroot='/mnt/pool1/cxy/nuScene_mini/nuscenes/',
                    verbose=True)
        train_scenes = splits.train
        val_scenes = splits.val

        generate_fake_depth_estimation_nuscene(nusc, train_scenes, depth_anything, use_metric_depth_estimator)
    if dataset_type == 'v2x_real':
        v2x_real_path = "/mnt/pool1/cxy/v2x_real"
        train_scenes = 'train'
        val_scenes = 'val'
        generate_fake_depth_estimation_v2x_real(v2x_real_path, train_scenes, depth_anything, use_metric_depth_estimator)

    if dataset_type == 'Hycan':
        hycan_path = "/mnt/pool1/cxy/Mono3d/data/merged_dataset_Hycan/"
        generate_fake_depth_estimation_hycan(hycan_path, depth_anything, use_metric_depth_estimator)
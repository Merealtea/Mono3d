import pickle
import nuscenes
import os
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
import cv2
# from opencood.hypes_yaml.yaml_utils import load_yaml
# from opencood.utils.box_utils import project_world_objects
# from opencood.data_utils.datasets import GT_RANGE
# from opencood.data_utils.datasets.basedataset import BaseDataset
# from opencood.data_utils import SUPER_CLASS_MAP
import mmcv
from utilities import detection_visualization_pinhole
from scipy.spatial.transform import Rotation as R

def euler_to_rot(yaw, pitch, roll):
    """ 将 yaw-pitch-roll 欧拉角 (in rad) 转换为旋转矩阵 """
    return R.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()

def rot_to_euler(rot_mat):
    """ 将旋转矩阵转换为 yaw-pitch-roll 欧拉角 (in rad) """
    return R.from_matrix(rot_mat).as_euler('ZYX')

def apply_transform(bbox, transform_matrix):
    """
    bbox: list or tuple of [x, y, z, l, w, h, yaw, pitch, roll, vel_x, vel_y]
    transform_matrix: 4x4 transformation matrix (numpy array)
    
    返回: 转换后的 bounding box 参数列表
    """
    x, y, z, l, w, h, yaw, pitch, roll, vel_x, vel_y = bbox
    
    # 1. 转换中心点
    point = np.array([x, y, z, 1.0])
    transformed_point = transform_matrix @ point
    new_x, new_y, new_z = transformed_point[:3]

    # 2. 转换姿态
    rot_bbox = euler_to_rot(yaw, pitch, roll)
    rot_transform = transform_matrix[:3, :3]
    combined_rot = rot_transform @ rot_bbox
    new_yaw, new_pitch, new_roll = rot_to_euler(combined_rot)

    # 3. 转换速度 (只考虑 xy 平面速度)
    vel_local = np.array([vel_x, vel_y, 0.0])
    vel_global = rot_transform @ vel_local
    new_vel_x, new_vel_y = vel_global[0], vel_global[1]

    return [
        new_x, new_y, new_z,
        l, w, h,
        new_yaw, new_pitch, new_roll,
        new_vel_x, new_vel_y
    ]


map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'Pedestrian',
    'human.pedestrian.child': 'Pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'Pedestrian',
    'human.pedestrian.construction_worker': 'Pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}

v2x_map_name_from_general_to_detection = {
    "Truck": "truck",
    "Car": "car",
    "Pedestrian": "Pedestrian",
    "ScooterRider" : "bicycle",
    "ConcreteTruck" : "truck",
    "BicycleRider" : "bicycle",
    "RoadWorker" : "Pedestrian",
    "Van" : "car",
    "Bus" : "bus",
    "TrashCan" : "truck",
    "MotorcyleRider" : "bicycle",
    "Motorcycle" : "bicycle",
    'Scooter' : "bicycle",
    "Child" : "Pedestrian",
    "LongVehicle" : "car",
    "PoliceCar" : "car",
    "FireHydrant" : "ignore"
}

kept_categories = ['Pedestrian', 'car', 'truck', 'bicycle', 'bus'] 

def transform_matrix(translation: list, rotation: Quaternion) -> np.ndarray:
    """创建4x4的变换矩阵"""
    tm = np.eye(4)
    tm[:3, :3] = rotation.rotation_matrix
    tm[:3, 3] = np.array(translation)
    return tm

def generate_nuscenes_info(nusc, scenes, max_cam_sweeps=6):
    infos = list()
    depth_result_path = nusc.dataroot.replace('nuscenes', 'depth_result')
    for cur_scene in tqdm(nusc.scene):
        if cur_scene['name'] not in scenes:
            continue
        first_sample_token = cur_scene['first_sample_token']
        cur_sample = nusc.get('sample', first_sample_token)
        while True:
            info = dict()

            info['timestamp'] = cur_sample['timestamp'] 
            info['num_sweeps'] = max_cam_sweeps
            info['dataset_type'] = 'nuscenes'

            cam_names = [
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
            ]
            cam_info = dict()
            cam_datas = dict()

            sd_rec = nusc.get('sample_data', cur_sample['data']['LIDAR_TOP'])
            cs_record = nusc.get('calibrated_sensor',
                                sd_rec['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])

            # 构造LiDAR到ego的变换矩阵
            lidar_to_lego = transform_matrix(cs_record['translation'], Quaternion(cs_record['rotation']))

            # 构造ego到全局的变换矩阵
            lego_to_global = transform_matrix(pose_record['translation'], Quaternion(pose_record['rotation']))

            # lidar to global
            info['key_frame_ego_pose'] = lego_to_global @ lidar_to_lego 

            for cam_name in cam_names:
                cam_sd = nusc.get('sample_data',
                                    cur_sample['data'][cam_name])
              
                cam_datas[cam_name] = cam_sd

                cam_info[cam_name] = {}
                cam_info[cam_name]['shape'] = (cam_sd['height'], cam_sd['width'])
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

                # cam_para[cam_name] = (cam_to_cego, cego_to_global, 
                #                       cam_pose_record, cam_cs_record)
            

                # extrinsic matrix
                # lidar_to_cam
                cam_info[cam_name]['extrinsic'] = np.linalg.inv(cam_to_cego) @ np.linalg.inv(cego_to_global) @ info['key_frame_ego_pose'] 

                # depth path
                file_name = os.path.basename(cam_sd['filename']).split('.')[0]
                depth_file_path = os.path.join(depth_result_path, cam_name, file_name + ".npy")
                cam_info[cam_name]['depth_path'] = depth_file_path
            info['key_frame_cam_info'] = cam_info

            gt_boxes = list()
            gt_labels = list()

            if 'anns' in cur_sample:
                # new_boxes = []
                for ann in cur_sample['anns']:
                    ann_info = nusc.get('sample_annotation', ann)
               
                    if int(ann_info['visibility_token']) == BoxVisibility.NONE:
                        continue

                    velocity = nusc.box_velocity(ann_info['token'])
                    if np.any(np.isnan(velocity)):
                        velocity = np.zeros(3)
                    ann_info['velocity'] = velocity
                    if (map_name_from_general_to_detection[ann_info['category_name']]
                            not in kept_categories
                            or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <=
                            0):
                        continue
                    box = Box(
                        ann_info['translation'],
                        ann_info['size'],
                        Quaternion(ann_info['rotation']),
                        velocity=ann_info['velocity'],
                    )

                    # test_box = Box(
                    #     ann_info['translation'],
                    #     ann_info['size'],
                    #     Quaternion(ann_info['rotation']),
                    #     velocity=ann_info['velocity'],
                    # )

                    # test_box.translate(-np.array(pose_record['translation']))
                    # test_box.rotate(Quaternion(pose_record['rotation']).inverse)
                    # test_box.translate(-np.array(cs_record['translation']))
                    # test_box.rotate(Quaternion(cs_record['rotation']).inverse)
                    
                    # new_box = apply_transform([box.center[0], box.center[1], box.center[2], box.wlh[1], box.wlh[0], box.wlh[2], box.orientation.yaw_pitch_roll[0], box.orientation.yaw_pitch_roll[1], box.orientation.yaw_pitch_roll[2], box.velocity[0], box.velocity[1]], 
                    #                           np.linalg.inv(info['key_frame_ego_pose']))
                    
                    new_box = apply_transform([box.center[0], box.center[1], box.center[2], box.wlh[1], box.wlh[0], box.wlh[2], box.orientation.yaw_pitch_roll[0], box.orientation.yaw_pitch_roll[1], box.orientation.yaw_pitch_roll[2], box.velocity[0], box.velocity[1]], 
                                                np.linalg.inv(lidar_to_lego) @ np.linalg.inv(lego_to_global))

                    box_xyz = np.array(new_box[:3])
                    box_dxdydz = np.array(new_box[3:6])
                    box_yaw = np.array(new_box[6:7])

                    box_velo = np.array(new_box[9:11])
                    gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
                    gt_boxes.append(gt_box)
                    gt_labels.append(map_name_from_general_to_detection[
                            ann_info['category_name']])
            
            info['gt_boxes'] = gt_boxes
            info['gt_labels'] = gt_labels

            cam_sweeps = [{"cam_info": {}} for _ in range(max_cam_sweeps)]

            # for j in range(max_cam_sweeps):
            #     next_cam_datas = {}
            #     for cam_name, cam_data in cam_datas.items():
            #         sweep_cam_data = cam_data
            #         if sweep_cam_data['prev'] == '':
            #             break
            #         else:
            #             sweep_cam_data = nusc.get('sample_data',
            #                                       sweep_cam_data['prev'])
            #             next_cam_datas[cam_name] = sweep_cam_data
            #             next_cam_datas[cam_name]['ego_pose'] = nusc.get(
            #                 'ego_pose', sweep_cam_data['ego_pose_token'])

            #             sweep_cam_info = dict()
            #             sweep_cam_info['image_path'] = os.path.join(
            #                 nusc.dataroot, sweep_cam_data['filename'])
            #             sweep_cam_info['shape'] = (sweep_cam_data['height'],
            #                                        sweep_cam_data['width'])
                        
            #             calibrated_sensor = nusc.get(
            #                 'calibrated_sensor',
            #                 sweep_cam_data['calibrated_sensor_token'])

            #             # intrinsic matrix
            #             sweep_cam_info['intrinsic'] = np.zeros((4, 4))
            #             sweep_cam_info['intrinsic'][3, 3] = 1
            #             sweep_cam_info['intrinsic'][:3, :3] = np.array(
            #                 calibrated_sensor['camera_intrinsic'])
                        
            #             # extrinsic matrix
            #             global2sensor_rot = Quaternion(calibrated_sensor['rotation']).inverse.rotation_matrix
            #             global2sensor_tran = -np.array(
            #                 calibrated_sensor['translation'])
            #             global2sensor = np.zeros((4, 4))
            #             global2sensor[3, 3] = 1
            #             global2sensor[:3, :3] = global2sensor_rot
            #             global2sensor[:3, -1] = global2sensor_tran
            #             sweep_cam_info['extrinsic'] = global2sensor
            #             cam_sweeps[j]["cam_info"][cam_name] = sweep_cam_info

            #         cam_sweeps[j]["timestamp"] = sweep_cam_data['timestamp']
            #         ego2global_rotation = np.mean(
            #             [next_cam_datas[cam]['ego_pose']['rotation'] for cam in next_cam_datas], 0)
            #         ego2global_translation = np.mean(
            #             [next_cam_datas[cam]['ego_pose']['translation'] for cam in next_cam_datas], 0)

            #         ego2global_mat = np.zeros((4, 4))
            #         ego2global_mat[3, 3] = 1
            #         ego2global_mat[:3, :3] = Quaternion(ego2global_rotation).rotation_matrix
            #         ego2global_mat[:3, -1] = ego2global_translation

            #         cam_sweeps[j]['ego_pose'] = ego2global_mat
            #     cam_datas = next_cam_datas
     
            # Remove empty sweeps.
            for i, sweep in enumerate(cam_sweeps):
                if sweep['cam_info'] == {}:
                    cam_sweeps = cam_sweeps[:i]
                    break

            info['cam_sweeps'] = cam_sweeps
            infos.append(info)

            if cur_sample['next'] == '':
                break
            else:
                cur_sample = nusc.get('sample', cur_sample['next'])
    return infos


def load_one_cav_info(cav_path, scene, scene_type, sweeps):
    yaml_files = \
            sorted([os.path.join(cav_path, x)
                    for x in os.listdir(cav_path) if
                    x.endswith('.yaml') and 'additional' not in x])
    timestamps = BaseDataset.extract_timestamps(yaml_files)

    data_root, scene_path = cav_path.split(scene_type + '/')
    depth_result_path = os.path.join(data_root, 'depth_result', scene_type, scene_path)

    cav_infos = []
    for time_idx, timestamp in enumerate(timestamps):
        info = dict()
        info['timestamp'] = f"{scene}_{timestamp}"
        info['num_sweeps'] = sweeps
        info['dataset_type'] = 'v2x_real'

        cam_info = dict()

        yaml_file = os.path.join(cav_path,
                                    timestamp + '.yaml')

        config_params = load_yaml(yaml_file)
        info['key_frame_ego_pose'] = np.array(config_params['lidar_pose'])

        for idx, cam in enumerate(['cam1', 'cam2', 'cam3', 'cam4']):
            cam_info[cam] = {}
            cam_info[cam]['image_path'] = os.path.join(cav_path, timestamp + f'_{cam}.jpeg')
            cam_info[cam]['intrinsic'] = np.eye(4)
            cam_info[cam]['intrinsic'][:3, :3] = np.array(config_params[cam]['intrinsic']) 
            cam_info[cam]['extrinsic'] = np.linalg.inv(np.array(config_params[cam]['extrinsic']))
            cam_info[cam]['shape'] = cv2.imread(cam_info[cam]['image_path']).shape[:2]

            # add depth image path
        info['key_frame_cam_info'] = cam_info

        cam_sweeps_infos = []
        for j in range(1, sweeps + 1):
            if time_idx - j < 0:
                break
            
            sweep_cam_info = {}

            sweep_timestamp = timestamps[time_idx - j]
            sweep_yaml_file = os.path.join(cav_path, sweep_timestamp + '.yaml')
            config_params = load_yaml(sweep_yaml_file)
            sweep_cam_info['timestamp'] = f"{scene}_{timestamps[time_idx - j]}"
            sweep_cam_info['ego_pose'] = np.array(config_params['lidar_pose'])
            sweep_cam_info['cam_info'] = {}

            for idx, cam in enumerate(['cam1', 'cam2', 'cam3', 'cam4']):
                sweep_cam_info['cam_info'][cam] = {}
                sweep_cam_info['cam_info'][cam]['image_path'] = os.path.join(cav_path, sweep_timestamp + f'_{cam}.jpeg')
                sweep_cam_info['cam_info'][cam]['intrinsic'] = np.array(config_params[cam]['intrinsic'])
                sweep_cam_info['cam_info'][cam]['extrinsic'] = np.array(config_params[cam]['extrinsic'])
                sweep_cam_info['cam_info'][cam]['shape'] = cv2.imread(sweep_cam_info['cam_info'][cam]['image_path']).shape[:2]
                
                
                # depth_file_path = os.path.join(depth_result_path, "{}_{}.npy".format(timestamp, cam))
                # sweep_cam_info['cam_info'][cam]['depth_path'] = depth_file_path
            cam_sweeps_infos.append(sweep_cam_info)

        gt_detection = load_yaml(yaml_file)['vehicles']
        output_dict = {}

        project_world_objects(gt_detection,
                                output_dict,
                                info['key_frame_ego_pose'],
                                GT_RANGE,
                                'lwh')
        gt_boxes = []
        gt_labels = []

        ref_timestamp = timestamps[time_idx + 1] if time_idx == 0 else timestamps[time_idx - 1]
        ref_detection = load_yaml(os.path.join(cav_path, ref_timestamp + '.yaml'))['vehicles']
        ref_output_dict = {}
        project_world_objects(ref_detection,
                                ref_output_dict,
                                info['key_frame_ego_pose'],
                                GT_RANGE,
                                'lwh')

        for det_id in output_dict:

            output_dict[det_id] = output_dict[det_id][0]
            gt_box_category = v2x_map_name_from_general_to_detection[output_dict[det_id][-1]]
            if gt_box_category not in kept_categories:
                continue

            gt_box = output_dict[det_id][:7].astype(np.float32)

            if det_id in ref_output_dict:
                ref_box = ref_output_dict[det_id][0][:7].astype(np.float32)
                if time_idx == 0:
                    bbox_vel = (ref_box[:2] - gt_box[:2]) / 0.1
                else:
                    bbox_vel = (gt_box[:2] - ref_box[:2]) / 0.1
            else:
                bbox_vel = np.zeros(2)

            gt_box = np.concatenate([gt_box, bbox_vel])
            # Turn to ego coordinate
            try:
                gt_boxes.append(gt_box)
                gt_labels.append(gt_box_category)
            except:
                import pdb; pdb.set_trace()

        info['gt_boxes'] = gt_boxes
        info['gt_labels'] = gt_labels
        info['cam_sweeps'] = cam_sweeps_infos
        cav_infos.append(info)
    return cav_infos

def load_one_frame_info(data_root, scene, frame, dataset_type):
    info = dict()
    info['dataset_type'] = dataset_type
    info['num_sweeps'] = 2
    info['timestamp'] = int(float(frame) * 1e6) 
    info['cam_sweeps'] = []

    # 读取GPS位置信息
    info['key_frame_ego_pose'] = np.eye(4)

    # 读取图像信息，包含图像尺寸，路径
    key_frame_cam_info = {}

    for cam in ['left', 'front', 'right', 'back']:
        cam_info = {}
        img_path = os.path.join(data_root, 'images', cam, "{}.jpg".format(frame))
        shape = (1080, 1920) if dataset_type == 'Rock' else (720, 1280)

        cam_info['image_path'] = img_path
        cam_info['shape'] = shape
        # 鱼眼其实可以不用这个
        cam_info['extrinsic'] = np.eye(4)
        cam_info['intrinsic'] = np.eye(4)
        key_frame_cam_info[cam] = cam_info

    info['key_frame_cam_info'] = key_frame_cam_info

    # ground truth boxes and labels

    with open(os.path.join(data_root, 'annotations', "lidar", "{}.pkl".format(frame)), 'rb') as f:
        ann = pickle.load(f)
        info['gt_boxes'] = ann[0]['bbox3d']
        info['gt_labels'] = ['Pedestrian'] * len(ann[0]['bbox3d'])
    return info

def generate_v2x_real_info(data_root, scene_type, max_cam_sweeps=6):
    infos = list()
    scene_path = os.path.join(data_root, scene_type)

    scenes = os.listdir(scene_path)
    for scene in tqdm(scenes):
        cavs = os.listdir(os.path.join(scene_path, scene))
        for cav in cavs:
            if int(cav) < 0:
                continue
            cav_path = os.path.join(scene_path, scene, cav)
            cav_infos = load_one_cav_info(cav_path, scene, scene_type, max_cam_sweeps)
            infos.extend(cav_infos)

    return infos


def generate_hycan_info(data_root, scene_type, max_cam_sweeps=6):
    infos = list()
    with open(os.path.join(data_root, 'mv_data_{}_info.txt'.format(scene_type)), 'r') as f:
        frames = f.readlines()

        for frame in tqdm(frames):
            frame = frame.strip()
            infos.append(load_one_frame_info(data_root, scene_type, frame, dataset_type='Hycan'))

    return infos

def generate_rock_info(data_root, scene_type, max_cam_sweeps=6):
    infos = list()
    with open(os.path.join(data_root, 'mv_data_{}_info.txt'.format(scene_type)), 'r') as f:
        frames = f.readlines()

        for frame in tqdm(frames):
            frame = frame.strip()
            infos.append(load_one_frame_info(data_root, scene_type, frame, dataset_type='Rock'))

    return infos

if __name__ == "__main__":
    num_sweeps = 2

    dataset_type = 'Rock'  # nuscenes / v2x_real / hycan / rock
    if dataset_type == 'nuscenes':
        dataset = nuscenes.NuScenes(version='v1.0-mini',
                             dataroot='/mnt/pool1/cxy/nuScene_mini/nuscenes/',
                             verbose=True)
        train_scenes = splits.train
        val_scenes = splits.val
        train_infos = generate_nuscenes_info(dataset, train_scenes, max_cam_sweeps=num_sweeps)
        val_infos = generate_nuscenes_info(dataset, val_scenes, max_cam_sweeps=num_sweeps)
        mmcv.dump(train_infos, '/mnt/pool1/cxy/Mono3d/data/data_info/nuscenes/nuscenes_infos_train.pkl')
        mmcv.dump(val_infos, '/mnt/pool1/cxy/Mono3d/data/data_info/nuscenes/nuscenes_infos_val.pkl')

    # elif dataset_type == 'v2x_real':
    #     v2x_real_path = "/mnt/pool1/cxy/v2x_real"
    #     train_scenes = 'train'
    #     val_scenes = 'val'
    #     train_infos = generate_v2x_real_info(v2x_real_path, train_scenes, num_sweeps)
    #     # val_infos = generate_v2x_real_info(v2x_real_path, val_scenes, num_sweeps)
    #     if not os.path.exists('/mnt/pool1/cxy/Mono3d/data/data_info/v2x_real'):
    #         os.makedirs('/mnt/pool1/cxy/Mono3d/data/data_info/v2x_real')
    #     mmcv.dump(train_infos, '/mnt/pool1/cxy/Mono3d/data/data_info/v2x_real/v2x_real_infos_train.pkl')
    #     # mmcv.dump(val_infos, '/mnt/pool1/cxy/Mono3d/data/data_info/v2x_real/v2x_real_infos_val.pkl')
    elif dataset_type == 'Hycan':
        hycan_path = "/mnt/pool1/cxy/Mono3d/data/merged_dataset_Hycan"
        train_scenes = 'train'
        val_scenes = 'val'
        train_infos = generate_hycan_info(hycan_path, train_scenes, num_sweeps)
        val_infos = generate_hycan_info(hycan_path, val_scenes, num_sweeps)
        if not os.path.exists('/mnt/pool1/cxy/Mono3d/data/data_info/hycan'):
            os.makedirs('/mnt/pool1/cxy/Mono3d/data/data_info/hycan')
        mmcv.dump(train_infos, '/mnt/pool1/cxy/Mono3d/data/data_info/hycan/hycan_infos_train.pkl')
        mmcv.dump(val_infos, '/mnt/pool1/cxy/Mono3d/data/data_info/hycan/hycan_infos_val.pkl')
    elif dataset_type == 'Rock':
        rock_path = "/mnt/pool1/cxy/Mono3d/data/merged_dataset_Rock"
        train_scenes = 'train'
        val_scenes = 'val'
        train_infos = generate_rock_info(rock_path, train_scenes, num_sweeps)
        val_infos = generate_rock_info(rock_path, val_scenes, num_sweeps)
        if not os.path.exists('/mnt/pool1/cxy/Mono3d/data/data_info/hycan'):
            os.makedirs('/mnt/pool1/cxy/Mono3d/data/data_info/rock')
        mmcv.dump(train_infos, '/mnt/pool1/cxy/Mono3d/data/data_info/rock/rock_infos_train.pkl')
        mmcv.dump(val_infos, '/mnt/pool1/cxy/Mono3d/data/data_info/rock/rock_infos_val.pkl')
    # Final Format
    # {
    #     'dataset_type: ： nuScenes / v2x_real / hycan / rock
    #     'timestamp': 1234567890,
    #     'num_sweeps' : 2,
    #     'cam_info': {
    #         'direction1': {
        #             'image_path': '',
        #             'intrinsic': numpy.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
        #             'extrinsic': numpy.array([[R, t], [0, 0, 0, 1]]),
        #             'shape': (h, w),
        #             'depth_path': '',
    #         },
    #         'direction2': '',
    #         'direction3': '',
    #         'direction4': '',}
    #     'key_frame_ego_pose': numpy.array([[R, t], [0, 0, 0, 1]]),
    #     'cam_sweeps': [
    #             {
    #                 'cam_info': 
    #                     { 
    #                         'direction1':{
    #                             'image_path': '',
        #                         'intrinsic': numpy.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
        #                         'extrinsic': numpy.array([[R, t], [0, 0, 0, 1]]),
        #                         'shape': (h, w),
        #                       },
    #                     },
                    # 'timestamp': 1234567890,
                    # 'ego_pose': numpy.array([[R, t], [0, 0, 0, 1]]),
    #              },
    #      'gt_boxes': [],
    #      'gt_labels': [],
    # }         
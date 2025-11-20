import cv2
import os
import sys
sys.path.append("/mnt/pool1/cxy/Mono3d")
sys.path.append("/mnt/pool1/cxy/Depth-Anything-V2")
from configs.FisheyeParam.cam_model import CamModel
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
import torch
from copy import deepcopy

def get_scaled_camera_params(K, D, image_size, scale=1.0, scale_x=None, scale_y=None):
    """
    根据原始相机矩阵 K 和畸变系数 D，自动计算适合的 scaled_size 和 new_K
    
    参数:
        K: 相机内参矩阵 (3x3)
        D: 鱼眼畸变系数 (4,)
        image_size: 原始图像尺寸 (width, height)
        scale: 统一缩放因子（默认）
        scale_x: 单独设置 x 方向缩放（可选）
        scale_y: 单独设置 y 方向缩放（可选）

    返回:
        scaled_size: 新的图像尺寸 (w_new, h_new)
        new_K: 新的相机矩阵
    """
    w, h = image_size

    balance = 0.6           # 越大保留越多内容（相当于 alpha 的作用）
    fov_scale = 0.6       # 可选调节视野


    # Step 1: 使用 estimateNewCameraMatrixForUndistortRectify 获取基础 new_K
    new_K_base = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, image_size, R=None, balance=balance, fov_scale=fov_scale
    )

    # Step 2: 设置新图像尺寸
    if scale_x is not None and scale_y is not None:
        w_new = int(w * scale_x)
        h_new = int(h * scale_y)
    elif isinstance(scale, (int, float)):
        w_new = int(w * scale)
        h_new = int(h * scale)
    else:
        raise ValueError("scale 必须是数字，或者提供 scale_x 和 scale_y")

    scaled_size = (w_new, h_new)

    # Step 3: 调整新的相机矩阵 new_K
    new_K = new_K_base.copy()
    fx = new_K[0, 0]
    fy = new_K[1, 1]
    cx = w_new / 2
    cy = h_new / 2

    # 保持焦距按图像尺寸同比例缩放（防止拉伸）
    new_K[0, 0] = fx * (w_new / w)
    new_K[1, 1] = fy * (h_new / h)
    new_K[0, 2] = cx
    new_K[1, 2] = cy

    return scaled_size, new_K

def interpolate_depth(depth_image, mapx, mapy, new_image_size):
    depth_image = depth_image.reshape(-1,1)
    mapx, mapy = mapx.reshape(-1, 1), mapy.reshape(-1, 1)
    left_top = (mapx.astype(int), mapy.astype(int)) 
    # left_bottom = mapx.astype(int), np.ceil(mapy).astype(int)
    # right_top = np.ceil(mapx).astype(int), mapy.astype(int)
    # right_bottom = np.ceil(mapx).astype(int), np.ceil(mapy).astype(int)

    # dist_left_top = np.zeros(new_image_size) + 1e6
    # dist_left_bottom = np.zeros(new_image_size) + 1e6
    # dist_right_top = np.zeros(new_image_size) + 1e6
    # dist_right_bottom = np.zeros(new_image_size) + 1e6

    # dist_left_top[left_top[1], left_top[0]] = 1/(np.sqrt((mapx - left_top[0])**2 + (mapy - left_top[1])**2))
    # dist_left_bottom[left_bottom[1], left_bottom[0]] = 1/(np.sqrt((mapx - left_bottom[0])**2 + (mapy - left_bottom[1])**2))
    # dist_right_top[right_top[1], right_top[0]] = 1/(np.sqrt((mapx - right_top[0])**2 + (mapy - right_top[1])**2))
    # dist_right_bottom[right_bottom[1], right_bottom[0]] = 1/(np.sqrt((mapx - right_bottom[0])**2 + (mapy - right_bottom[1])**2))
    # sum_dist = dist_left_top + dist_left_bottom + dist_right_top + dist_right_bottom

    depth_left_top = np.zeros(new_image_size) 
    # depth_left_bottom = np.zeros(new_image_size)
    # depth_right_top = np.zeros(new_image_size)
    # depth_right_bottom = np.zeros(new_image_size)

    depth_left_top[left_top[1], left_top[0]] =  depth_image
    # depth_left_bottom[left_bottom[1], left_bottom[0]] =  depth_image
    # print(np.sum(depth_left_top > 0))
    
    # depth_right_top[right_top[1], right_top[0]] = depth_image
    # depth_right_bottom[right_bottom[1], right_bottom[0]] = depth_image

    final_depth = depth_left_top 
    

    # dialation 
    for i in range(1):
        add_depth = np.zeros(new_image_size)
        empty_space = np.where(final_depth == 0)
        for i in range(len(empty_space[0])):
            x, y = empty_space[0][i], empty_space[1][i]
            if x > 0 and y > 0 and x < new_image_size[0] - 1 and y < new_image_size[1] - 1 and np.sum(final_depth[x-1:x+2, y-1:y+2]) > 0.5 :
                max_depth = np.max(final_depth[x-1:x+2, y-1:y+2])
                valid_mask = final_depth[x-1:x+2, y-1:y+2] > max_depth * 0.5
                if np.sum(valid_mask) > 0:
                    add_depth[x, y] = np.mean(final_depth[x-1:x+2, y-1:y+2][valid_mask]) 

        final_depth = final_depth + add_depth
    print(np.sum(final_depth > 0))
    import pdb; pdb.set_trace()
    return final_depth

def distort_fisheye_image(image, K, D):
    """
    Distort a fisheye image using the given camera matrix and distortion coefficients.

    Args:
        image (numpy.ndarray): The input image to be distorted.
        K (numpy.ndarray): The camera matrix.
        D (numpy.ndarray): The distortion coefficients.

    Returns:
        numpy.ndarray: The distorted image.
    """
    h, w = image.shape[:2]
    image_size = (w, h)  # 原始图像尺寸

    scaled_size = (int(w), int(h))  # 放大 1.5 倍

    # 设置 balance 来代替 alpha
    balance = 0.6           # 越大保留越多内容（相当于 alpha 的作用）
    fov_scale = 0.6       # 可选调节视野

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,
        D,
        (w, h),
        R=None,             # 一般不需要旋转
        balance=balance,    # 替代 alpha 的参数
        fov_scale=fov_scale
    )

    # scaled_size, new_K = get_scaled_camera_params(K, D, image_size, scale=1)

    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(K, D, None, new_K, scaled_size, cv2.CV_32FC1)
    distorted_image = cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR)

    return distorted_image, mapx, mapy


if __name__ == "__main__":
    image_path = "/mnt/pool1/cxy/Mono3d/data/merged_dataset_Hycan/images"
    directions = ["front", "left", "right", "back"]
    result_path = "/mnt/pool1/cxy/Mono3d/data/merged_dataset_Hycan/distort_images"  
    depth_path = "/mnt/pool1/cxy/Mono3d/data/merged_dataset_Hycan/depth_images"  
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'/mnt/pool1/cxy/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if not os.path.exists(depth_path):
        os.makedirs(depth_path)

    for direction in directions:
        image_dir = os.path.join(image_path, direction)
        cam_model = CamModel(direction, "Hycan", "numpy")
        K = np.array([[float(cam_model.fcous_distance[1]), 0, cam_model.img_center[0,0]],
                      [0, float(cam_model.fcous_distance[0]), cam_model.img_center[1,0]],
                      [0, 0, 1]])
        D = cam_model.radial_param

        angles = np.linspace(-np.pi, np.pi, 100)
        points = np.array([np.cos(angles), np.sin(angles)])
        points = np.concatenate((np.zeros((1, points.shape[1])) + 0.05, points), axis=0)
        
        pixel_value = cam_model.cam2image(points,  filter_points = True)

        if not os.path.exists(os.path.join(result_path, direction)):
            os.makedirs(os.path.join(result_path, direction))
        if not os.path.exists(os.path.join(depth_path, direction)):
            os.makedirs(os.path.join(depth_path, direction))
        for image_name in os.listdir(image_dir):
            image_file = os.path.join(image_dir, image_name)
            image = cv2.imread(image_file)

            depth_image_direct = model.infer_image(image)
            max_depth = np.max(depth_image_direct)
            depth_image_direct = (depth_image_direct / max_depth * 255).astype(int)

            cv2.imwrite(os.path.join(depth_path, direction, "direct_depth_" + image_name), depth_image_direct)
            print("save depth image: ", os.path.join(depth_path, direction, "direct_depth_" + image_name))

            # # draw points in image
            # for i in range(points.shape[1]):
            #     x = int(pixel_value[0, i])
            #     y = int(pixel_value[1, i])
            #     cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            # cv2.imwrite(os.path.join(result_path, direction, "circle_" + image_name), image)
            distorted_image, mapx, mapy = distort_fisheye_image(image, K, D)
            cv2.imwrite(os.path.join(result_path, direction, image_name), distorted_image)
            distort_fisheye_image_depth = model.infer_image(distorted_image)
            # remap into original space
            cv2.imwrite(os.path.join(result_path, direction, "distort_depth_remap" + image_name), 
                        distort_fisheye_image_depth / np.max(distort_fisheye_image_depth) * 255)

            final_depth = interpolate_depth(distort_fisheye_image_depth, mapx, mapy, image.shape[:2])
            cv2.imwrite(os.path.join(depth_path, direction, "distort_depth_" + image_name), final_depth / np.max(final_depth) * 255)
            import pdb; pdb.set_trace()
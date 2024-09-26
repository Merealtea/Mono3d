import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import os

class CamModel:
    """ 
        Class for cam model (fisheye model). 
        Hycan fisheye model can only project the point with positive depth in camera coordinate.
        Rock fisheye model can project the point with negative depth in camera coordinate.
    """
    def __init__(self, cam_dir: str, vehicle : str, version = "numpy", device = None):
        assert cam_dir in ["left", "right", 'front', 'back'], "cam_dir must be one of 'left', 'right', 'front', 'back'. "
        assert vehicle in ["Rock", "Hycan"], "vehicle must be one of 'Rock', 'Hycan'. "
        self.version = version
        self.vehicle = vehicle
        self.cam_dir = cam_dir
        if self.version == "torch":
            self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_fisheye_model(cam_dir, vehicle)

    def get_fisheye_model(self, cam_dir, vehicle):
        root_path = os.path.dirname(__file__)
        filename = root_path + "/{}/{}/calib_results_{}.txt".format(vehicle, cam_dir, cam_dir)
        pose_file = root_path + "/{}/{}/results_{}.csv".format(vehicle, cam_dir, cam_dir)

        # extrinsic parameters
        self.world2cam_mat = np.genfromtxt(pose_file, delimiter=',')
        self.cam2world_mat = np.linalg.inv(self.world2cam_mat)

        # intrinsic parameters
        self.height = 1080
        self.width = 1920
        self.img_center = np.array([[self.width / 2], [self.height / 2]])

        """ Reads .txt file containing the camera calibration parameters exported from the Matlab toolbox. """
        with open(filename) as f:
            lines = [l for l in f]
            l = lines[2]
            data = l.split()
            if vehicle == "Rock":
                self.length_pol = int(data[0])           # 多项式长度
                self.pol = np.array([float(d) for d in data[1:]])  # 多项式

                l = lines[6]
                data = l.split()
                self.length_invpol = int(data[0])        # 反投影的逆
                self.invpol = np.array([float(d) for d in data[1:]])

                l = lines[10]
                data = l.split()
                self.img_center[0,0] = float(data[0])                 # 图像中心
                self.img_center[1,0] = float(data[1])

                l = lines[14]
                data = l.split()
                self.c = float(data[0])                  # 仿射参数
                self.d = float(data[1])
                self.e = float(data[2])

                l = lines[18]                            # 图像宽高
                data = l.split()
                self.height = int(data[0])
                self.width = int(data[1])
                self.stretch_mat = np.array([[self.c, self.d], [self.e, 1]])
                self.inv_stretch_mat = np.linalg.inv(self.stretch_mat)

                rotm = R.from_euler('xyz', [np.pi / 2, 0, np.pi / 2]).as_matrix()
                r = R.from_euler('yzx', [np.pi, np.pi / 2, 0]).as_matrix()
            elif vehicle == "Hycan":
                self.length_radial_param = int(data[0])   # 多项式长度
                self.radial_param = np.array([float(d) for d in data[1:]])  #i径向畸变多项式

                l = lines[6]
                data = l.split()
                self.fcous_distance = np.array([float(data[0]), float(data[1])]).reshape(2,1)

                l = lines[10]
                data = l.split()
                self.img_center[0,0] = float(data[0])         # 图像中心
                self.img_center[1,0] = float(data[1])

                l = lines[14]
                data = l.split()
                self.length_inv_distort = int(data[0])        # 逆畸变投影参数
                self.inv_distort = np.array([float(d) for d in data[1:][::-1]])

                l = lines[18]                            # 图像宽高
                data = l.split()
                self.height = int(data[0])
                self.width = int(data[1])
                rotm = R.from_euler('zxy', [np.pi / 2, 0, np.pi / 2]).as_matrix()
                r = R.from_euler('yzx', [0, 0, 0]).as_matrix()
                rot = np.eye(4)
                rot[:3,:3] = rotm
                self.world2cam_mat = np.matmul(rot, self.world2cam_mat)
                self.cam2world_mat = np.linalg.inv(self.world2cam_mat)

            self.r1 = np.dot(r, rotm.transpose())
            self.r2 = rotm

        if self.version == "torch":
            if vehicle == "Rock":
                self.stretch_mat = torch.from_numpy(self.stretch_mat).float()
                self.inv_stretch_mat = torch.from_numpy(self.inv_stretch_mat).float()
                self.pol = torch.from_numpy(self.pol).float()
                self.invpol = torch.from_numpy(self.invpol).float()
                self.img_center = torch.from_numpy(self.img_center).float()
                self.r1 = torch.from_numpy(self.r1).float()
                self.r2 = torch.from_numpy(self.r2).float()
                self.world2cam_mat = torch.from_numpy(self.world2cam_mat).float()
                self.cam2world_mat = torch.from_numpy(self.cam2world_mat).float()
            elif vehicle == "Hycan":
                self.fcous_distance = torch.FloatTensor(self.fcous_distance)
                self.img_center = torch.FloatTensor(self.img_center)
                self.inv_distort = torch.FloatTensor(self.inv_distort)
                self.radial_param = torch.FloatTensor(self.radial_param)
                self.r1 = torch.FloatTensor(self.r1)
                self.r2 = torch.FloatTensor(self.r2)
                self.world2cam_mat = torch.from_numpy(self.world2cam_mat).float()
                self.cam2world_mat = torch.from_numpy(self.cam2world_mat).float()
            self.to(self.device)
    
    def to(self, device):
        if self.version == "torch":
            self.device = device
            if self.vehicle == "Rock":
                self.stretch_mat = self.stretch_mat.to(device)
                self.inv_stretch_mat = self.inv_stretch_mat.to(device)
                self.pol = self.pol.to(device)
                self.invpol = self.invpol.to(device)
                self.img_center = self.img_center.to(device)
                self.r1 = self.r1.to(device)
                self.r2 = self.r2.to(device)
                self.world2cam_mat = self.world2cam_mat.to(device)
                self.cam2world_mat = self.cam2world_mat.to(device)
            else:
                self.fcous_distance = self.fcous_distance.to(device)
                self.img_center = self.img_center.to(device)
                self.inv_distort = self.inv_distort.to(device)
                self.radial_param = self.radial_param.to(device)
                self.r1 = self.r1.to(device)
                self.r2 = self.r2.to(device)
                self.world2cam_mat = self.world2cam_mat.to(device)
                self.cam2world_mat = self.cam2world_mat.to(device)

    def image2cam(self, points2D, points_depth):
        """ Returns the 3D points projected on the sphere from the image pixels. """
        assert points2D.shape[0] == 2, "points2D must be a 2xN matrix. "
        assert len(points_depth) == points2D.shape[1], "depth must have the same length as points2D. "
        if self.version == "numpy":
            if self.vehicle == "Rock":
                return self._image2cam_numpy(points2D, points_depth)
            elif self.vehicle == "Hycan":
                return self._image2cam_hycan(points2D, points_depth)

        elif self.version == "torch":
            if self.vehicle == "Rock":
                return self._image2cam_torch(points2D, points_depth)
            elif self.vehicle == "Hycan":
                return self._image2cam_hycan_torch(points2D, points_depth)
        else:
            raise ValueError("version must be one of 'numpy', 'torch'. ")

    def _image2cam_numpy(self, points2D, points_depth):
        # minus the image center
        points2D = points2D - self.img_center[[1,0],:]
        points2D = np.matmul(self.inv_stretch_mat, points2D)
        
        norm = np.linalg.norm(points2D, axis=0)
        norm_poly = np.array([norm ** i for i in range(0, self.length_pol)])
        zp = np.dot(self.pol, norm_poly)
        
        lamda = points_depth / zp
        xc = lamda * points2D[0]
        yc = lamda * points2D[1]
        zc = points_depth

        points3D = np.vstack((xc, yc, zc))
        points3D = np.matmul(self.r2, points3D)
        return points3D 
    
    def _image2cam_torch(self, points2D, points_depth):
        # minus the image center
        points2D = points2D - self.img_center[[1,0],:]
        points2D = torch.matmul(self.inv_stretch_mat, points2D)

        norm = torch.norm(points2D, dim=0)
        norm_poly = torch.stack([norm ** i for i in range(0, self.length_pol)])
        zp = torch.matmul(self.pol, norm_poly)

        lamda = points_depth / zp
        xc = lamda * points2D[0]
        yc = lamda * points2D[1]
        zc = points_depth

        points3D = torch.stack((xc, yc, zc))
        return torch.matmul(self.r2, points3D)

    def cam2image(self, points3D, filter_points = True):
        """ Projects 3D points on the image and returns the pixel coordinates. """
        assert points3D.shape[0] == 3, "points3D must be a 3xN matrix. "
        if self.version == "numpy":
            if self.vehicle == "Rock":
                return self._cam2image_numpy(points3D, filter_points)
            elif self.vehicle == "Hycan":
                return self._cam2image_hycan(points3D, filter_points)

        elif self.version == "torch":
            if self.vehicle == "Rock":
                return self._cam2image_torch(points3D, filter_points)
            elif self.vehicle == "Hycan":
                return self._cam2image_hycan_torch(points3D, filter_points)
        else:
            raise ValueError("version must be one of 'numpy', 'torch'. ")
        
    def _cam2image_hycan(self, points3D, filter_points):
        points3D = np.dot(self.r1, points3D)
        
        if filter_points:
            valid = points3D[2] > 0
            points3D = points3D[:, valid]

        points3D /= points3D[2]

        # 只考虑了r小于10的点，如果大于10，可能误差比较大
        r = np.sqrt(points3D[0] ** 2 + points3D[1] ** 2)

        # 计算未畸变的角度
        theta = np.arctan(r)

        # 应用畸变模型
        theta_poly = np.array([theta ** i for i in [2, 4, 6, 8]]) 
        theta_d = (np.dot(self.radial_param, theta_poly) + 1) * theta
    
        radial_distortion = theta_d / (r + 1e-8) 

        points3D *= radial_distortion
        u, v = points3D[:2] * self.fcous_distance + self.img_center

        if filter_points:
            u[u < 0], u[u > self.width] = 0, self.width
            v[v < 0], v[v > self.height] = 0, self.height
        return np.array([u, v])
    
    def _cam2image_hycan_torch(self, points3D, filter_points):
        points3D = torch.matmul(self.r1, points3D)
        if filter_points:
            valid = points3D[2] > 0
            points2D = points2D[:, valid]
        points2D = points3D[:2] / points3D[2]

        # 只考虑了r小于10的点，如果大于10，可能误差比较大
        r = torch.sqrt(points2D[0] ** 2 + points2D[1] ** 2)

        # 计算未畸变的角度
        theta = torch.arctan(r)
        
        # 应用畸变模型
        theta_poly = torch.stack([theta ** i for i in [2, 4, 6, 8]]) 
        theta_d = torch.matmul(self.radial_param, theta_poly) * theta + theta

        radial_distortion = theta_d / (r + 1e-8) 
    
        points2D *= radial_distortion
        u, v = points2D * self.fcous_distance + self.img_center
    
        if filter_points:
            u[u < 0], u[u > self.width] = 0, self.width
            v[v < 0], v[v > self.height] = 0, self.height
        return torch.stack([u, v])
    
    def _image2cam_hycan(self, points2D, points_depth):
        # minus the image center
        points2D = (points2D - self.img_center) / self.fcous_distance
        theta_d = np.sqrt(points2D[0] ** 2 + points2D[1] ** 2)

        poly = np.array([theta_d ** i for i in range(0, self.length_inv_distort)])
        theta = np.dot(self.inv_distort, poly)

        # 未畸变的径向距离
        r = np.tan(theta)

        undist = r / (theta_d + 1e-8)        
        points2D = points2D * undist * points_depth

        points3D = np.vstack((points2D[0], points2D[1], points_depth))
        points3D = np.matmul(self.r2, points3D)
        return points3D
    
    def undistort(self, image):
        points2D = np.array(np.meshgrid(np.arange(self.width), np.arange(self.height))).reshape(2, -1)
        points2D = (points2D - self.img_center) / self.fcous_distance
        theta_d = np.sqrt(points2D[0] ** 2 + points2D[1] ** 2)

        poly = np.array([theta_d ** i for i in range(0, self.length_inv_distort)])
        theta = np.dot(self.inv_distort, poly)

        # 未畸变的径向距离
        r = np.tan(theta)
        undist = r / (theta_d + 1e-8)        
        points2D = points2D * undist 
        u, v = points2D * self.fcous_distance + self.img_center + np.array([320, 180]).reshape((2, 1))
        
        u[u < 0], u[u > 1920] = 0, 1920-1
        v[v < 0], v[v > 1080] = 0, 1080-1

        undistort_image = np.zeros((1080, 1920, 3)) 
        # np.zeros_like(image)
        undistort_image[v.astype(int), u.astype(int), :] = image.reshape(-1, 3)
        return undistort_image.reshape(1080, 1920, 3)

    def _image2cam_hycan_torch(self, points2D, points_depth):
        # minus the image center
        points2D = (points2D - self.img_center) / self.fcous_distance
        theta_d = torch.sqrt(points2D[0] ** 2 + points2D[1] ** 2)

        poly = torch.stack([theta_d ** i for i in range(0, self.length_inv_distort)])
        theta = torch.matmul(self.inv_distort, poly)

        # 未畸变的径向距离
        r = torch.tan(theta)
        undist = r / (theta_d + 1e-8)        
        points2D = points2D * undist * points_depth

        points3D = torch.vstack((points2D[0], points2D[1], points_depth))
        points3D = torch.matmul(self.r2, points3D)
        return points3D

    def _cam2image_numpy(self, points3D, filter_points):
        """ Projects 3D points on the image and returns the pixel coordinates. """
        points3D = points3D.T
    
        n = np.linalg.norm(points3D, axis=1)
        x, y, z = points3D.T / n

        # 计算投影坐标
        points3D = np.dot(self.r1, np.vstack((x, y, z)))

        # 只考虑向下的点
        # if filter_points:
        #     valid = points3D[2] < 0
        #     points3D = points3D[:, valid]

        norm = np.linalg.norm(points3D[:2], axis=0)
        theta = np.arctan2(points3D[2], norm)
        invnorm = 1.0 / norm

        theta_poly = np.array([theta ** i for i in range(0, self.length_invpol)])
        rho = np.dot(self.invpol, theta_poly)

        x, y, _ = points3D * invnorm * rho 
        v, u = np.around(np.dot(self.stretch_mat, np.vstack((x, y))) + self.img_center)

        # 使用矩阵索引安全检查边界条件
        if filter_points:
            u[u < 0], u[u > self.width] = 0, self.width
            v[v < 0], v[v > self.height] = 0, self.height
        return np.array([u, v]) 

    def _cam2image_torch(self, points3D, filter_points):
        """ Projects 3D points on the image and returns the pixel coordinates. """
        points3D = points3D.T

        n = torch.norm(points3D, dim=1)
        x, y, z = points3D.T / n

        # 计算投影坐标
        points3D = torch.matmul(self.r1, torch.stack((x, y, z)))

        # 只考虑向下的点
        if filter_points:
            valid = points3D[2] < 0
            points3D = points3D[:, valid]

        norm = torch.norm(points3D[:2], dim=0)
        theta = torch.atan(points3D[2], norm)
        invnorm = 1.0 / norm

        theta_poly = torch.stack([theta ** i for i in range(0, self.length_invpol)])
        rho = torch.matmul(self.invpol, theta_poly)

        x, y, _ = points3D * invnorm * rho 
        v, u = torch.round(torch.matmul(self.stretch_mat, torch.stack((x, y))) + self.img_center)

        # 使用矩阵索引安全检查边界条件
        if filter_points:
            u[u < 0], u[u > self.width] = 0, self.width
            v[v < 0], v[v > self.height] = 0, self.height
        return torch.stack([u, v])
    
    def cam2world(self, points3D):
        """ Projects 3D points on the image and returns the cam coordinates. """
        assert points3D.shape[0] == 3, "points3D must be a 3xN matrix. "
        if self.version == "numpy":
            points = np.vstack([points3D, np.ones(points3D.shape[1])])
            return np.dot(self.cam2world_mat, points)[0:3, :]
        elif self.version == "torch":
            points = torch.vstack([points3D, torch.ones(points3D.shape[1]).to(self.device)])
            return torch.matmul(self.cam2world_mat, points)[0:3, :]
        else:
            raise ValueError("version must be one of 'numpy', 'torch'. ")

    def world2cam(self, points3D):
        """ Projects 3D points on the image and returns the cam coordinates. """
        assert points3D.shape[0] == 3, "points3D must be a 3xN matrix. "
        if self.version == "numpy":
            points = np.vstack([points3D, np.ones(points3D.shape[1])])
            return np.dot(self.world2cam_mat, points)[0:3, :]
        elif self.version == "torch":
            points = torch.vstack([points3D, torch.ones(points3D.shape[1]).to(self.device)])
            return torch.matmul(self.world2cam_mat, points)[0:3, :]
        else:
            raise ValueError("version must be one of 'numpy', 'torch'. ")

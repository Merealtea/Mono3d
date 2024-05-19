import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

class CamModel:
    """ Class for cam model (fisheye model). """
    def __init__(self, cam_dir: str, version = "numpy", device = None):
        assert cam_dir in ["left", "right", 'front', 'back'], "cam_dir must be one of 'left', 'right', 'front', 'back'. "
        self.version = version
        if self.version == "torch":
            self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_fisheye_model(cam_dir)

    def get_fisheye_model(self, cam_dir):
        filename = "configs/FisheyeParam/{}/calib_results_{}.txt".format(cam_dir, cam_dir)
        pose_file = "configs/FisheyeParam/{}/results_{}.csv".format(cam_dir, cam_dir)

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

        # some rotation matrix
        rotm = R.from_euler('xyz', [np.pi / 2, 0, np.pi / 2]).as_matrix()
        r = R.from_euler('yzx', [np.pi, np.pi / 2, 0]).as_matrix()
        self.r1 = np.dot(r, rotm.transpose())
        self.r2 = rotm

        if self.version == "torch":
            self.stretch_mat = torch.from_numpy(self.stretch_mat).float()
            self.inv_stretch_mat = torch.from_numpy(self.inv_stretch_mat).float()
            self.pol = torch.from_numpy(self.pol).float()
            self.invpol = torch.from_numpy(self.invpol).float()
            self.img_center = torch.from_numpy(self.img_center).float()
            self.r1 = torch.from_numpy(self.r1).float()
            self.r2 = torch.from_numpy(self.r2).float()
            self.world2cam_mat = torch.from_numpy(self.world2cam_mat).float()
            self.cam2world_mat = torch.from_numpy(self.cam2world_mat).float()
            self.to(self.device)
    
    def to(self, device):
        if self.version == "torch":
            self.device = device
            self.stretch_mat = self.stretch_mat.to(device)
            self.inv_stretch_mat = self.inv_stretch_mat.to(device)
            self.pol = self.pol.to(device)
            self.invpol = self.invpol.to(device)
            self.img_center = self.img_center.to(device)
            self.r1 = self.r1.to(device)
            self.r2 = self.r2.to(device)
            self.world2cam_mat = self.world2cam_mat.to(device)
            self.cam2world_mat = self.cam2world_mat.to(device)

    def image2cam(self, points2D, points_depth):
        """ Returns the 3D points projected on the sphere from the image pixels. """
        assert points2D.shape[0] == 2, "points2D must be a 2xN matrix. "
        assert len(points_depth) == points2D.shape[1], "depth must have the same length as points2D. "
        if self.version == "numpy":
            return self._image2cam_numpy(points2D, points_depth)
        elif self.version == "torch":
            return self._image2cam_torch(points2D, points_depth)
        else:
            raise ValueError("version must be one of 'numpy', 'torch'. ")

    def _image2cam_numpy(self, points2D, points_depth):
        points3D = []
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
        r = R.from_euler('xyz', [np.pi / 2, 0, np.pi / 2]).as_matrix()

        points3D = np.matmul(self.r2, points3D)
        return points3D #np.matmul(self.r2, points3D)
    
    def _image2cam_torch(self, points2D, points_depth):
        points3D = []
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

    def cam2image(self, points3D):
        """ Projects 3D points on the image and returns the pixel coordinates. """
        assert points3D.shape[0] == 3, "points3D must be a 3xN matrix. "
        if self.version == "numpy":
            return self._cam2image_numpy(points3D)
        elif self.version == "torch":
            return self._cam2image_torch(points3D)
        else:
            raise ValueError("version must be one of 'numpy', 'torch'. ")
        

    def _cam2image_numpy(self, points3D):
        """ Projects 3D points on the image and returns the pixel coordinates. """
        points3D = points3D.T
    
        n = np.linalg.norm(points3D, axis=1)
        x, y, z = points3D.T / n

        # 计算投影坐标
        points3D = np.dot(self.r1, np.vstack((x, y, z)))

        # 只考虑向下的点
        # valid = points3D[2] < 0
        # points3D = points3D[:, valid]
        norm = np.linalg.norm(points3D[:2], axis=0)
        theta = np.arctan2(points3D[2], norm)
        invnorm = 1.0 / norm

        theta_poly = np.array([theta ** i for i in range(0, self.length_invpol)])
        rho = np.dot(self.invpol, theta_poly)

        # import pdb; pdb.set_trace()
        x, y, _ = points3D * invnorm * rho 
        v, u = np.around(np.dot(self.stretch_mat, np.vstack((x, y))) + self.img_center).astype(int)

        # 使用矩阵索引安全检查边界条件
        u[u < 0], u[u > self.width] = 0, self.width
        v[v < 0], v[v > self.height] = 0, self.height
        return np.array([u, v]) 

    def _cam2image_torch(self, points3D):
        """ Projects 3D points on the image and returns the pixel coordinates. """
        points3D = points3D.T

        n = torch.norm(points3D, dim=1)
        x, y, z = points3D.T / n

        # 计算投影坐标
        points3D = torch.matmul(self.r1, torch.stack((x, y, z)))

        # 只考虑向下的点
        # valid = points3D[2] < 0
        # points3D = points3D[:, valid]

        norm = torch.norm(points3D[:2], dim=0)
        theta = torch.atan2(points3D[2], norm)
        invnorm = 1.0 / norm

        theta_poly = torch.stack([theta ** i for i in range(0, self.length_invpol)])
        rho = torch.matmul(self.invpol, theta_poly)

        x, y, _ = points3D * invnorm * rho 
        v, u = torch.round(torch.matmul(self.stretch_mat, torch.stack((x, y))) + self.img_center).int()

        # 使用矩阵索引安全检查边界条件
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

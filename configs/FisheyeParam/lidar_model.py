# The lidar model is to translate data in lidar coordination
# to Vehicle rear axie coordination.
import os
import numpy as np 

class Lidar_transformation:
    def __init__(self, vehicle):
        assert vehicle in ["Hycan", "Rock"]
        self.vehicle = vehicle
        self.get_lidar_to_vehicle()

    def get_lidar_to_vehicle(self):
        root_path = os.path.dirname(__file__)
        lidar_to_rear_file = root_path + "/{}/lidar/lidar_calib.csv".format(self.vehicle)
        rear_to_center_file = root_path + "/{}/lidar/rear_calib.csv".format(self.vehicle)
        # extrinsic parameters
        self.rear2lidar = np.genfromtxt(lidar_to_rear_file, delimiter=',')
        self.lidar2rear = np.linalg.inv(self.rear2lidar)

        self.rear2center = np.genfromtxt(rear_to_center_file, delimiter=',')
        self.center2rear = np.linalg.inv(self.rear2center)

    def lidar_to_rear(self, lidar_points):
        assert lidar_points.shape[0] == 3, "Lidar points should be in shape (3, N)"
        return np.dot(self.lidar2rear, lidar_points)
    
    def rear_to_lidar(self, rear_points):
        assert rear_points.shape[0] == 3, "Rear points should be in shape (3, N)"
        return np.dot(self.rear2lidar, rear_points)
    
    def rear_to_center(self, rear_points):
        assert rear_points.shape[0] == 3, "Rear points should be in shape (3, N)"
        return np.dot(self.rear2center, rear_points)
    
    def center_to_rear(self, center_points):
        assert center_points.shape[0] == 3, "Center points should be in shape (3, N)"
        return np.dot(self.center2rear, center_points)
    

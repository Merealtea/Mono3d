# **长安车鱼眼相机内外参标定**

## 文件介绍

front、back、left、right: 前、后、左、右相机标定结果

images: 激光雷达点云投影至鱼眼相机结果图

calib_results_*.txt: 鱼眼相机内参标定结果

results_*.csv: 鱼眼相机外参标定结果

## 坐标系示意图

![Frame](./frame.jpg)

<!-- <p float="left">
  <img src="/frame.jpg?raw=true" width="59%" />
</p> -->

## 标定工具

#### 内参标定工具
[ImprovedOcamCalib](https://github.com/urbste/ImprovedOcamCalib/tree/master)
#### 外参标定工具
[fisheye_lidar_calibration](https://github.com/ARVCUMH/fisheye_lidar_calibration)

## 标定流程
内参标定  —>  外参标定  —>  交互式手动调节



import argparse
import os
from os import path 
import sys
abs_path = os.path.abspath(__file__)
sys.path.append(abs_path.split('tools')[0])

import torch
import rosbag
import rospy
import yaml

from builder.build_dataset import build_dataset
from builder.build_model import build_detector
import cv2
import numpy as np
from configs.FisheyeParam import CamModel
from utilities import init_random_seed, detection_visualization,\
      to_device, turn_gt_to_annos


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--gt_path', help='gt file path')
    parser.add_argument('--pre_bag_path', help='val data path')
    args = parser.parse_args()
    return args

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
    else:
        print(f"Folder {folder} already exists. Skip creation.")

def main():
    args = parse_args()
    rosbag_topic = "/fusion_result"

    gt_path = args.gt_path
    fusion_result_gt = os.listdir(args.gt_path)
    paired_gt = []
    paired_prediction = []

    with open(os.path.join('/mnt/pool1/cxy/Mono3d/ckpt/ours', 'config', 'dataset_config.yaml')) as f:
        dataset_cfg = yaml.safe_load(f)
    cfg = yaml.safe_load(open(os.path.join('/mnt/pool1/cxy/Mono3d/ckpt/ours', 'config', 'train_config.yaml')))
    dataset_cfg["data_root"] = args.val_data_path
    dataset_cfg["ann_prefix"] = cfg["annotation_prefix"]
    dataset_cfg["img_prefix"] = cfg["image_prefix"]   
    dataset_cfg['test_mode'] = True
    dataset_cfg["vehicle"] = cfg["vehicle"]
    val_dataset = build_dataset(dataset_cfg)

    for scenario_date in fusion_result_gt:
        timestamps = os.listdir(os.path.join(gt_path, scenario_date))
        timestamps = [float(timestamps.split('.txt')[0]) for timestamps in timestamps]
        timestamps = sorted(timestamps)

        prediction_bag_path = os.path.join(args.pre_bag_path, "{}.bag".format(scenario_date))
        bag = rosbag.Bag(prediction_bag_path, 'r')
        prediction_msgs = []
        for topic, msg, t in bag.read_messages(topics=[rosbag_topic]):
            prediction_msgs.append(msg)

        # align the prediction and ground truth
        start_idx = 0
        end_idx = min(50, len(timestamps))
        for msg in prediction_msgs:
            if msg.sender.stamp.to_sec() < timestamps[start_idx]:
                continue
            if msg.sender.stamp.to_sec() > timestamps[end_idx]:
                start_idx = min(start_idx + 50, end_idx)
                end_idx = min(end_idx + 50, len(timestamps))

            timstamp_diff = np.array(timestamps[start_idx:end_idx]) - msg.sender.stamp.to_sec()
            if np.min(np.abs(timstamp_diff)) < 0.1:
                min_idx = np.argmin(np.abs(timstamp_diff))
                timestamp = timestamps[start_idx:end_idx][min_idx]

                one_gt = []
                with open(os.path.join(gt_path, scenario_date, "{}.txt".format(timestamp)), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        one_gt.append(line)
                
                paired_gt.append(one_gt)

                one_prediction = []
                for box_3d in msg.boxes_3d:
                    one_prediction.append([box_3d.x, box_3d.y, box_3d.z, box_3d.length, box_3d.width, box_3d.height, box_3d.heading])
                paired_prediction.append(one_prediction)    

        
        # Evaluate the result with prediction and ground truth  
        ground_truth = turn_gt_to_annos(ground_truth, val_dataset.CLASSES)
        val_dataset.evaluate(detection_res, ground_truth, metric='kitti')

if __name__ == '__main__':
    main()

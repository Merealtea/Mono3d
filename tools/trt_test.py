import argparse
import os
from os import path 
import sys
abs_path = os.path.abspath(__file__)
sys.path.append(abs_path.split('tools')[0])

import torch
import yaml
from tqdm import tqdm
import pycuda.driver as cuda
import pycuda.autoinit
from inference_test import TRTModel
from builder.build_dataset import build_dataset
from configs.FisheyeParam import CamModel
from utilities import  detection_visualization, turn_gt_to_annos


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--last_ckpt', help='train config file path')
    parser.add_argument('--val_data_path', help='val data path')
    parser.add_argument('--vehicle', help='vehicle type', default=None)
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

    ckpt_path = args.last_ckpt
    cfg = yaml.safe_load(open(os.path.join(ckpt_path, 'config', 'train_config.yaml')))
    vehicle = cfg["vehicle"] if args.vehicle is None else args.vehicle

    cam_models = dict(zip(["left", "right", "front", "back"], [CamModel(direction, vehicle) for direction in ["left", "right", "front", "back"]]))
    # load dataset
    with open(os.path.join(ckpt_path, 'config', 'dataset_config.yaml')) as f:
        dataset_cfg = yaml.safe_load(f)

    # load model
    cfx = cuda.Device(0).make_context()
    trt_model = TRTModel(ckpt_path)

    # Create save folder to save the ckpt
    data_datetime = args.val_data_path.split('/')[-1]
    save_path = os.path.join(cfg['save_path'], 'val_{}'.format(data_datetime))
    create_folder(save_path)

    # load dataset
    with open(cfg['dataset_config']) as f:
        dataset_cfg = yaml.load(f, Loader=yaml.FullLoader)

    dataset_cfg["data_root"] = args.val_data_path
    dataset_cfg["ann_prefix"] = cfg["annotation_prefix"]
    dataset_cfg["img_prefix"] = cfg["image_prefix"]   
    dataset_cfg['test_mode'] = True
    dataset_cfg["vehicle"] = cfg["vehicle"]
    val_dataset = build_dataset(dataset_cfg)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=True,
                                collate_fn=val_dataset.collate)


    # eval model
    bbox_res_path = os.path.join(save_path, "trt", "val_bbox_pre")
    create_folder(bbox_res_path)

    for direction in ["left", "right", "front", "back"]:
        create_folder(os.path.join(bbox_res_path, direction))

    detection_res = []
    ground_truth = []

    for i, data in tqdm(enumerate(val_loader)):
        img = data['img'].detach().numpy()
        bbox_res = trt_model(input_tesor=img, return_loss=False, rescale=True)
        detection_res += bbox_res
        del data['img'] 
        ground_truth.append(data)

        # save bbox_res
        for idx ,(bbox, img_meta) in enumerate(zip(bbox_res, data['img_metas'])):
            # extract bbox from bbox_res
            if 'img_bbox' in bbox:
                bbox = bbox['img_bbox']['boxes_3d'].tensor.numpy()[:, :7]
            else:
                bbox = bbox['boxes_3d'].tensor.numpy()[:, :7]
            gt_bbox = data['gt_bboxes_3d'][idx].cpu().numpy()[:, :7]
            if cfg['bbox_coordination'] == "CAM":
                cam_model = cam_models[img_meta["direction"]]
                bbox_res_dir_path = os.path.join(bbox_res_path, img_meta["direction"])
                filename = img_meta['filename']
                detection_visualization(bbox, gt_bbox, filename, cam_model, bbox_res_dir_path, bboxes_coor = "CAM")
            elif cfg['bbox_coordination'] == "Lidar":
                for filename, direction in zip(img_meta['img_filename'], img_meta['direction']):
                    cam_model = cam_models[direction]
                    bbox_res_dir_path = os.path.join(bbox_res_path, direction)
                    detection_visualization(bbox, gt_bbox, filename, cam_model, bbox_res_dir_path, bboxes_coor = "Lidar")

    # Evaluate the result with prediction and ground truth  
    ground_truth = turn_gt_to_annos(ground_truth, val_dataset.CLASSES)
    val_dataset.evaluate(detection_res, ground_truth, metric='kitti')
    
    cfx.pop()

if __name__ == '__main__':
    main()

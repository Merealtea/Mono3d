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
# import pycuda.autoinit
from inference_test import TRTModel
from builder.build_dataset import build_dataset
from configs.FisheyeParam import CamModel
from utilities import  detection_visualization, turn_gt_to_annos
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--last_ckpt', help='train config file path')
    parser.add_argument('--val_data_path', help='val data path')
    parser.add_argument('--vehicle', help='vehicle type', default=None)
    parser.add_argument('--trt_model_path', help='trt model path')
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
    context = cuda.Device(0).make_context()
    ckpt_path = args.last_ckpt
    cfg = yaml.safe_load(open(os.path.join(ckpt_path, 'config', 'train_config.yaml')))
    model_cfg = yaml.safe_load(open(os.path.join(ckpt_path, 'config', 'model_config.yaml')))
    vehicle = cfg["vehicle"] if args.vehicle is None else args.vehicle
    box_code_size = model_cfg["test_cfg"]["box_code_size"]
    nms_thresh = model_cfg["test_cfg"]["nms_thr"]
    pos_thresh = model_cfg["test_cfg"]["score_thr"]
    cuda.init()
    cam_models = dict(zip(["left", "right", "front", "back"], [CamModel(direction, vehicle) for direction in ["left", "right", "front", "back"]]))
    # load dataset
    with open(os.path.join(ckpt_path, 'config', 'dataset_config.yaml')) as f:
        dataset_cfg = yaml.safe_load(f)

    # load model
    trt_model_path = args.trt_model_path
    # cfx = cuda.Device(0).make_context()
    trt_model = TRTModel(trt_model_path, box_code_size=box_code_size, nms_thresh=nms_thresh, pos_thresh=pos_thresh)

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


    detection_res = []
    ground_truth = []

    for i, data in tqdm(enumerate(val_loader)):
        img = data['img'].detach().numpy()
        bbox_res = trt_model(input_tesor=img, return_loss=False, img_metas=data['img_metas'])
        # import pdb; pdb.set_trace()
        detection_res += bbox_res
        del data['img'] 
        ground_truth.append(data)

        # save bbox_res
        for idx ,(result, img_meta) in enumerate(zip(bbox_res, data['img_metas'])):
            # extract res from bbox_res
            if 'img_bbox' in result:
                bbox = result['img_bbox']['boxes_3d'].tensor.numpy()
                scores = result['img_bbox']['scores_3d'].numpy()
            else:
                bbox = result['boxes_3d'].tensor.numpy()
                scores = result['scores_3d'].numpy()
            gt_bbox = data['gt_bboxes_3d'][idx].cpu().numpy()[:, :7]
            if bbox.shape[1] > 7:
                vars = bbox[:, 7:9]
            else:
                vars = None
            
            if cfg['bbox_coordination'] == "CAM":
                cam_model = cam_models[img_meta["direction"]]
                bbox_res_dir_path = os.path.join(bbox_res_path, img_meta["direction"])
                filename = img_meta['filename']
                img = detection_visualization(bbox, gt_bbox, filename, cam_model, bbox_res_dir_path, bboxes_coor = "CAM", scores = scores, vars=vars)
            elif cfg['bbox_coordination'] == "Lidar":
                vis_imgs = []
                for filename, direction in zip(img_meta['img_filename'], img_meta['direction']):
                    cam_model = cam_models[direction]
                    bbox_res_dir_path = os.path.join(bbox_res_path, direction)
                    img = detection_visualization(bbox, gt_bbox, filename, cam_model, bbox_res_dir_path, bboxes_coor = "Lidar", scores=scores, vars=vars)
                    vis_imgs.append(img)
                vis_imgs = np.concatenate(vis_imgs, axis=1)
                cv2.imwrite(os.path.join(bbox_res_path, f"{filename.split('/')[-1].split('.jpg')[0]}.jpg"), vis_imgs)

    context.pop()

    # Evaluate the result with prediction and ground truth  
    ground_truth = turn_gt_to_annos(ground_truth, val_dataset.CLASSES)
    val_dataset.evaluate(detection_res, ground_truth, metric='kitti')
    

if __name__ == '__main__':
    main()

import argparse
import copy
import os
import time
import warnings
from os import path 
import sys
sys.path.append('/data/cxy/Mono3d')

import torch
import numpy as np
import torch.distributed as dist
import yaml
from tqdm import tqdm
import cv2
from torch.nn.utils import clip_grad
from builder.build_dataset import build_dataset
from builder.build_model import build_detector
from datetime import datetime
import pickle
from configs.FisheyeParam import CamModel

import tensorboardX
from utils import init_random_seed, calculate_corners_cam, plot_rect3d_on_img

def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
        return data
    if isinstance(data, list):
        return [to_device(d, device) for d in data]
    return data

def clip_grads(params, grad_clip):
    params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
    # 对requires_grap为True的进行clip
    if len(params) > 0: 
        return clip_grad.clip_grad_norm_(params, **grad_clip) 

def create_lr_scheduler(optimizer,
                        init_lr : float,
                        warmup_iters: int,
                        warmup_ratio=1e-3):
    """
    :param optimizer: 优化器
    :param num_step: 每个epoch迭代多少次，len(data_loader)
    :param epochs: 总共训练多少个epoch
    :param warmup: 是否采用warmup
    :param warmup_epochs: warmup进行多少个epoch
    :param warmup_factor: warmup的一个倍数因子
    :return:
    """

    def f(cur_iters):
        if cur_iters < warmup_iters:
            k = cur_iters / warmup_iters
            warmup_lr = init_lr * (1 - k * (1 - warmup_ratio))
            return warmup_lr
        else:
            return init_lr
        # （1-a/b）^0.9 b是当前这个epoch结束训练总共了多少次了（除去warmup），这个关系是指一个epcoch中
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path', default=None)
    parser.add_argument('--last_ckpt', help='last checkpoint file path', default=None)
    args = parser.parse_args()
    return args

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
    else:
        print(f"Folder {folder} already exists. Skip creation.")

def main():
    cam_models = dict(zip(["left", "right", "front", "back"], [CamModel(direction) for direction in ["left", "right", "front", "back"]]))

    args = parse_args()
    ckpt_file = None
    start_epoch = 0
    if args.last_ckpt is not None:
        save_path = args.last_ckpt
        cfg = yaml.load(open(os.path.join(save_path, 'config', 'train_config.yaml')), Loader=yaml.FullLoader)
        tensorboard_path = os.path.join(save_path, 'tensorboard')

        # load dataset
        with open(os.path.join(save_path, 'config', 'dataset_config.yaml')) as f:
            dataset_cfg = yaml.load(f, Loader=yaml.FullLoader)

        # load model
        with open(os.path.join(save_path, 'config', 'model_config.yaml')) as f:
            model_cfg = yaml.load(f, Loader=yaml.FullLoader)

        for file in os.listdir(save_path):
            if file.endswith(".pth"):
                epoch = int(file.split('_')[-1].split('.')[0])
                if epoch > start_epoch:
                    start_epoch = epoch
                    ckpt_file = os.path.join(save_path, file)
    else:
        with open(args.config) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        # Create save folder to save the ckpt
        save_path = cfg['save_path']
        save_path = os.path.join(save_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        create_folder(save_path)

        tensorboard_path = os.path.join(save_path, 'tensorboard')
        create_folder(tensorboard_path)

        config_save_path = os.path.join(save_path, 'config')
        create_folder(config_save_path)

        # save config file
        with open(os.path.join(config_save_path, 'train_config.yaml'), 'w') as f:
            yaml.dump(cfg, f)

        # load dataset
        with open(cfg['dataset_config']) as f:
            dataset_cfg = yaml.load(f, Loader=yaml.FullLoader)

        # save dataset config
        with open(os.path.join(config_save_path, 'dataset_config.yaml'), 'w') as f:
            yaml.dump(dataset_cfg, f)

        # load model
        with open(cfg['model_config']) as f:
            model_cfg = yaml.load(f, Loader=yaml.FullLoader)

        # save model config 
        with open(os.path.join(config_save_path, 'model_config.yaml'), 'w') as f:
            yaml.dump(model_cfg, f)

    # create tensorboard writer
    writer = tensorboardX.SummaryWriter(tensorboard_path)

    dataset = build_dataset(dataset_cfg, 
            data_root=cfg["data_root"],
            annotation_prefix=cfg["annotation_prefix"],
            image_prefix=cfg["image_prefix"], 
            eval=False       
    )

    val_dataset = build_dataset(dataset_cfg,
            data_root=cfg["data_root"],
            annotation_prefix=cfg["annotation_prefix"],
            image_prefix=cfg["image_prefix"],
            eval=True
    )

    val_loss = 1000
    model = build_detector(model_cfg)
    total_epochs = cfg['epoches']
    batch_size = cfg['batch_size']
    eval_interval = cfg['eval_interval']
    # set_device
    device = torch.device(f"cuda:{cfg['gpu_id']}") if torch.cuda.is_available() else torch.device('cpu')
    
    # create dataloader
    train_loader = torch.utils.data.DataLoader(dataset, 
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=True,
                                collate_fn=dataset.collate)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=True,
                                collate_fn=val_dataset.collate)

    # set random seeds
    seed = init_random_seed(cfg['seed'])

    if ckpt_file is not None:
        model.load_state_dict(torch.load(ckpt_file))
    else:
        model.init_weights()
    model = model.to(device)
    model.train()

    for name, param in model.backbone.named_parameters():
        param.requires_grad = False
    
    # setup optimizer and hyper parameters
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=cfg["optimizer"]['lr'])
    lr_scheduler = create_lr_scheduler(optimizer, cfg["optimizer"]['lr'], 
                                       cfg['lr_config']['warmup_iters'], 
                                       cfg["lr_config"]['warmup_ratio'])
    grad_clip_cfg = cfg['optimizer']['grad_clip'] if 'grad_clip' in cfg['optimizer'] else None

    # train model
    for epoch in range(start_epoch, total_epochs):
        for i, data in tqdm(enumerate(train_loader)):

            optimizer.zero_grad()
            to_device(data, device)
            data['img_metas'][0]['epoch'] = epoch
            loss_res = model(data['img'], data['img_metas'], return_loss = True,
                        gt_bboxes = data['gt_bboxes'],
                        gt_labels = data['gt_labels'],
                        gt_bboxes_3d = data['gt_bboxes_3d'],
                        gt_labels_3d = data['gt_labels_3d'],
                        centers2d = data['centers2d'],
                        depths = data['depths']
                        )

            loss = loss_res['loss_cls'] +\
                    loss_res['loss_depth']  + \
                    loss_res['loss_size'] + \
                    loss_res['loss_rotsin'] + \
                    loss_res['loss_offset'] + \
                    loss_res['loss_centerness'] 
            
            print(f"Epoch: {epoch}, Iter: {i}, Loss: {loss.item()}\n",
                    f"Loss cls: {loss_res['loss_cls'].item()}\n",
                    f"Loss depth: {loss_res['loss_depth'].item()}\n",
                    f"Loss size: {loss_res['loss_size'].item()}\n",
                    f"Loss rotsin: {loss_res['loss_rotsin'].item()}\n",
                    f"Loss offset: {loss_res['loss_offset'].item()}\n",
                    f"Loss centerness: {loss_res['loss_centerness'].item()}\n"
            )
            
            # record loss
            writer.add_scalar('train loss', loss.item(), epoch * len(train_loader) + i)
            # record loss details
            writer.add_scalar('train loss cls', loss_res['loss_cls'].item(), epoch * len(train_loader) + i)
            writer.add_scalar('train loss depth', loss_res['loss_depth'].item(), epoch * len(train_loader) + i)
            writer.add_scalar('train loss size', loss_res['loss_size'].item(), epoch * len(train_loader) + i)
            writer.add_scalar('train loss rotsin', loss_res['loss_rotsin'].item(), epoch * len(train_loader) + i)
            writer.add_scalar('train loss offset', loss_res['loss_offset'].item(), epoch * len(train_loader) + i)
            writer.add_scalar('train loss centerness', loss_res['loss_centerness'].item(), epoch * len(train_loader) + i)

            loss.backward()

            if grad_clip_cfg is not None:
                clip_grads(model.parameters(), grad_clip_cfg)

            optimizer.step()
            lr_scheduler.step()

        if epoch % eval_interval == 0:
            model.eval()
            # eval model
            with torch.no_grad():
                val_epoch = epoch // eval_interval
                for i, data in tqdm(enumerate(val_loader)):
                    to_device(data, device)
                    data['img_metas'][0]['test'] = 1
                    loss_res = model(data['img'], data['img_metas'], return_loss = True,
                            gt_bboxes = data['gt_bboxes'],
                            gt_labels = data['gt_labels'],
                            gt_bboxes_3d = data['gt_bboxes_3d'],
                            gt_labels_3d = data['gt_labels_3d'],
                            centers2d = data['centers2d'],
                            depths = data['depths']
                            )
                    
                    loss = loss_res['loss_cls'] +\
                        loss_res['loss_depth'] + \
                        loss_res['loss_size'] + \
                        loss_res['loss_rotsin'] + \
                        loss_res['loss_offset'] + \
                        loss_res['loss_centerness'] 
                    
                    # record loss
                    writer.add_scalar('val loss', loss.item(), val_epoch * len(val_loader) + i)
                    # record loss details
                    writer.add_scalar('val loss cls', loss_res['loss_cls'].item(), val_epoch * len(val_loader) + i)
                    writer.add_scalar('val loss depth', loss_res['loss_depth'].item(), val_epoch * len(val_loader) + i)
                    writer.add_scalar('val loss size', loss_res['loss_size'].item(), val_epoch * len(val_loader) + i)
                    writer.add_scalar('val loss rotsin', loss_res['loss_rotsin'].item(), val_epoch * len(val_loader) + i)
                    writer.add_scalar('val loss offset', loss_res['loss_offset'].item(), val_epoch * len(val_loader) + i)
                    writer.add_scalar('val loss centerness', loss_res['loss_centerness'].item(), val_epoch * len(val_loader) + i)
                    # if val_loss > loss.item():
                    #     val_loss = loss.item()
                        # save model
                if epoch % 10 == 0:
                    torch.save(model.state_dict(), f"{save_path}/epoch_{epoch}.pth")
                    print(f"Save best model at epoch {epoch}")
            if epoch > total_epochs -11:
                bbox_res_path = os.path.join(save_path, f"{epoch}", "val_bbox_pre")
                bbox_train_res_path = os.path.join(save_path, f"{epoch}", "train_bbox_pre")
                create_folder(bbox_res_path)
                create_folder(bbox_train_res_path)
                            
                with torch.no_grad():
                    for i, data in tqdm(enumerate(train_loader)):
                            to_device(data, device)
                            bbox_res = model([data['img']], [data['img_metas']], return_loss = False)
                            # save bbox_res
                            for idx, (bbox, img_meta) in enumerate(zip(bbox_res, data['img_metas'])):
                                # extract bbox from bbox_res
                                bbox = bbox['img_bbox']['boxes_3d'].tensor.cpu().numpy()[:, :7]
                                if (len(bbox) == 0):
                                    continue
                                gt_bbox = data['gt_bboxes_3d'][idx].cpu().numpy()[:, :7]
                                direction = img_meta["direction"]
                                world2cam_mat = cam_models[direction].world2cam_mat
                                corners = calculate_corners_cam(bbox, world2cam_mat).reshape(-1, 8, 3)
                                gt_corners = calculate_corners_cam(gt_bbox, world2cam_mat).reshape(-1, 8, 3)
                                bboxes = []
                                for corner in corners:
                                    pixel_uv = cam_models[direction].cam2image(corner.T).T
                                    if pixel_uv.shape[0] == 8:
                                        bboxes.append(pixel_uv)
                                
                                img = cv2.imread(img_meta['filename'])
                                img = plot_rect3d_on_img(img, len(bboxes), bboxes, color=(0, 0, 255))

                                gt_bboxes = []
                                for corner in gt_corners:
                                    pixel_uv = cam_models[direction].cam2image(corner.T).T
                                    if pixel_uv.shape[0] == 8:
                                        gt_bboxes.append(pixel_uv)
                            
                                img = plot_rect3d_on_img(img, len(gt_bboxes), gt_bboxes, color=(0, 255, 0))
                                cv2.imwrite(os.path.join(bbox_train_res_path, f"{img_meta['timestamp']}.jpg"), img)
                    
                with torch.no_grad():
                    val_epoch = epoch // eval_interval
                    for i, data in tqdm(enumerate(val_loader)):
                        to_device(data, device)
                        bbox_res = model([data['img']], [data['img_metas']], return_loss = False)
                        # save bbox_res
                        for bbox, img_meta in zip(bbox_res, data['img_metas']):
                            # extract bbox from bbox_res
                            bbox = bbox['img_bbox']['boxes_3d'].tensor.cpu().numpy()[:, :7]
                            bboxes = []
                            if (len(bbox) != 0):
                                direction = img_meta["direction"]
                                world2cam_mat = cam_models[direction].world2cam_mat
                                corners = calculate_corners_cam(bbox, world2cam_mat).reshape(-1, 8, 3)
                                for corner in corners:
                                    pixel_uv = cam_models[direction].cam2image(corner.T).T
                                    if pixel_uv.shape[0] == 8:
                                        bboxes.append(pixel_uv)
                            img = cv2.imread(img_meta['filename'])
                            img = plot_rect3d_on_img(img, len(bboxes), bboxes)
                            cv2.imwrite(os.path.join(bbox_res_path, f"{img_meta['timestamp']}.jpg"), img)   
            model.train()
if __name__ == '__main__':
    main()

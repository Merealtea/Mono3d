import argparse
import copy
import os
import time
import warnings
from os import path 
import sys
sys.path.append('/data/cxy/Mono3d')

import torch
import torch.distributed as dist
import yaml
from tqdm import tqdm

from builder.build_dataset import build_dataset
from builder.build_model import build_detector
from datetime import datetime
import pickle

import tensorboardX
from utils import init_random_seed

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


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
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

    # create tensorboard writer
    writer = tensorboardX.SummaryWriter(tensorboard_path)

    # load dataset
    with open(cfg['dataset_config']) as f:
        dataset_cfg = yaml.load(f, Loader=yaml.FullLoader)
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

    # save dataset config
    with open(os.path.join(config_save_path, 'dataset_config.yaml'), 'w') as f:
        yaml.dump(dataset_cfg, f)
    
    # load model
    with open(cfg['model_config']) as f:
        model_cfg = yaml.load(f, Loader=yaml.FullLoader)
    model = build_detector(model_cfg)

    # save model config 
    with open(os.path.join(config_save_path, 'model_config.yaml'), 'w') as f:
        yaml.dump(model_cfg, f)

    # setup optimizer and hyper parameters
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=cfg["optimizer"]['lr'])
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

    model.init_weights()
    model = model.to(device)

    # train model
    for epoch in range(total_epochs):
        for i, data in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            to_device(data, device)
 
            loss_res = model(data['img'], data['img_metas'], return_loss = True,
                        gt_bboxes = data['gt_bboxes'],
                        gt_labels = data['gt_labels'],
                        gt_bboxes_3d = data['gt_bboxes_3d'],
                        gt_labels_3d = data['gt_labels_3d'],
                        centers2d = data['centers2d'],
                        depths = data['depths']
                        )

            loss = loss_res['loss_cls'] * 0 +\
                    loss_res['loss_depth'] + \
                    loss_res['loss_size'] + \
                    loss_res['loss_rotsin'] + \
                    loss_res['loss_offset'] + \
                    loss_res['loss_centerness'] 
            
            print(f"Epoch: {epoch}, Iter: {i}, Loss: {loss.item()}\n",
                    f"Loss depth: {loss_res['loss_depth'].item()}\n",
                    f"Loss size: {loss_res['loss_size'].item()}\n",
                    f"Loss rotsin: {loss_res['loss_rotsin'].item()}\n",
                    f"Loss offset: {loss_res['loss_offset'].item()}\n",
                    f"Loss centerness: {loss_res['loss_centerness'].item()}\n"
            )
            
            # record loss
            writer.add_scalar('train loss', loss.item(), epoch * len(train_loader) + i)
            # record loss details
            writer.add_scalar('train loss depth', loss_res['loss_depth'].item(), epoch * len(train_loader) + i)
            writer.add_scalar('train loss size', loss_res['loss_size'].item(), epoch * len(train_loader) + i)
            writer.add_scalar('train loss rotsin', loss_res['loss_rotsin'].item(), epoch * len(train_loader) + i)
            writer.add_scalar('train loss offset', loss_res['loss_offset'].item(), epoch * len(train_loader) + i)
            writer.add_scalar('train loss centerness', loss_res['loss_centerness'].item(), epoch * len(train_loader) + i)

            loss.backward()
            optimizer.step()
            

        if epoch % eval_interval == 0:
            model.eval()
            # eval model
            bbox_res_path = os.path.join(save_path, f"{epoch}", "val_bbox_pre")
            create_folder(bbox_res_path)
            with torch.no_grad():
                for i, data in tqdm(enumerate(val_loader)):
                    to_device(data, device)
                    loss_res = model(data['img'], data['img_metas'], return_loss = True,
                            gt_bboxes = data['gt_bboxes'],
                            gt_labels = data['gt_labels'],
                            gt_bboxes_3d = data['gt_bboxes_3d'],
                            gt_labels_3d = data['gt_labels_3d'],
                            centers2d = data['centers2d'],
                            depths = data['depths']
                            )

                    bbox_res = model([data['img']], [data['img_metas']], return_loss = False)
                    # save bbox_res
                    for bbox, img_meta in zip(bbox_res, data['img_metas']):
                        # extract bbox from bbox_res
                        bbox = bbox['img_bbox']['boxes_3d'].tensor.cpu().numpy()[:, :7]
                        with open(os.path.join(bbox_res_path, f"{img_meta['timestamp']}.pkl"), 'wb') as f:
                            pickle.dump(bbox, f)
                    
                    loss = loss_res['loss_cls'] * 0 +\
                            loss_res['loss_depth'] + \
                            loss_res['loss_size'] + \
                            loss_res['loss_rotsin'] + \
                            loss_res['loss_offset'] + \
                            loss_res['loss_centerness'] 
                    
                    # record loss
                    writer.add_scalar('val loss', loss, epoch * len(val_loader) + i)
                    # record loss details
                    writer.add_scalar('val loss depth', loss_res['loss_depth'], epoch * len(val_loader) + i)
                    writer.add_scalar('val loss size', loss_res['loss_size'], epoch * len(val_loader) + i)
                    writer.add_scalar('val loss rotsin', loss_res['loss_rotsin'], epoch * len(val_loader) + i)
                    writer.add_scalar('val loss offset', loss_res['loss_offset'], epoch * len(val_loader) + i)
                    writer.add_scalar('val loss centerness', loss_res['loss_centerness'], epoch * len(val_loader) + i)
            model.train()

        # save model
        if epoch % cfg['save_interval'] == 0:
            torch.save(model.state_dict(), f"{save_path}/epoch_{epoch}.pth")

if __name__ == '__main__':
    main()

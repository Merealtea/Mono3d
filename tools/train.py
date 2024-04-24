import argparse
import copy
import os
import time
import warnings
from os import path 


import torch
import torch.distributed as dist
import yaml

from builder.build_dataset import build_dataset
from builder.build_model import build_detector

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

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Create save folder to save the ckpt
    save_path = cfg['save_path']
    create_folder(save_path)

    # load dataset
    with open(cfg['dataset_config']) as f:
        dataset_cfg = yaml.load(f, Loader=yaml.FullLoader)
    dataset = build_dataset(dataset_cfg, 
            data_root=cfg["data_root"],
            annotation_prefix=cfg["annotation_prefix"],
            image_prefix=cfg["image_prefix"], 
            eval=False       
    )
    
    # load model
    with open(cfg['model_config']) as f:
        model_cfg = yaml.load(f, Loader=yaml.FullLoader)
    model = build_detector(model_cfg)

    # setup optimizer and super parameters
    # optimizer = torch.optim.Adam(model.parameters(), 
                               #  lr=cfg["optimizer"]['lr'])
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


    # set random seeds
    seed = init_random_seed(cfg['seed'])

    model.init_weights()
    model = model.to(device)

    # train model
    for epoch in range(total_epochs):
        for i, data in enumerate(train_loader):
            to_device(data, device)
            res = model(data['img'], data['img_metas'], return_loss = True,
                        gt_bboxes = data['gt_bboxes'],
                        gt_labels = data['gt_labels'],
                        gt_bboxes_3d = data['gt_bboxes_3d'],
                        gt_labels_3d = data['gt_labels_3d'],
                        centers2d = data['centers2d'],
                        depths = data['depths']
                        )
            
            import pdb; pdb.set_trace()

        if epoch % eval_interval == 0:
            pass

        # save model
        # if epoch % cfg['save_interval'] == 0:
        #     torch.save(model.state_dict(), f"{save_path}/epoch_{epoch}.pth")

if __name__ == '__main__':
    main()

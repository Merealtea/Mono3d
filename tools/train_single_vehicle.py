import argparse
import os
from os import path 
abs_path = path.abspath(__file__)
import sys
sys.path.append(abs_path.split("tools")[0])
from utilities import init_random_seed, detection_visualization, to_device
# set random seeds
import random
import torch
import numpy as np
seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print(f"Set random seed to {seed}")
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


def clip_grads(params, grad_clip):
    params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
    # 对requires_grap为True的进行clip
    if len(params) > 0: 
        return clip_grad.clip_grad_norm_(params, **grad_clip) 

def create_lr_scheduler(optimizer,
                        init_lr : float,
                        warmup_iters: int,
                        warmup_ratio=1e-3
                        ):
    """
    :param optimizer: 优化器
    :param num_step: 每个epoch迭代多少次，len(data_loader)
    :param epochs: 总共训练多少个epoch
    :param warmup: 是否采用warmup
    :param warmup_epochs: warmup进行多少个epoch
    :param warmup_factor: warmup的一个倍数因子
    :return:
    """
    # TODO: Add learning rate decay
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
    args = parse_args()
    ckpt_file = None
    start_epoch = 0
    vehicle = None
    best_valid_loss = 1000
    # set random seeds
    seed = init_random_seed(2025)
    print(f"Set random seed to {seed}")
    if args.last_ckpt is not None:
        save_path = args.last_ckpt
        cfg = yaml.safe_load(open(os.path.join(save_path, 'config', 'train_config.yaml')))
        tensorboard_path = os.path.join(save_path, 'tensorboard')

        vehicle = cfg["vehicle"]
        # load dataset
        with open(os.path.join(save_path, 'config', 'dataset_config.yaml')) as f:
            dataset_cfg = yaml.safe_load(f)

        # load model
        with open(os.path.join(save_path, 'config', 'model_config.yaml')) as f:
            model_cfg = yaml.safe_load(f)

        for file in os.listdir(save_path):
            if 'best' in file and file.endswith(".pth"):
                continue
            if file.endswith(".pth"):
                epoch = int(file.split('_')[-1].split('.')[0])
                if epoch > start_epoch:
                    start_epoch = epoch
                    ckpt_file = os.path.join(save_path, file)
    else:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
            vehicle = cfg["vehicle"]

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
            yaml.safe_dump(cfg, f)

        # load dataset
        with open(cfg['dataset_config']) as f:
            dataset_cfg = yaml.safe_load(f)

        # save dataset config
        with open(os.path.join(config_save_path, 'dataset_config.yaml'), 'w') as f:
            yaml.safe_dump(dataset_cfg, f)

        # load model
        with open(cfg['model_config']) as f:
            model_cfg = yaml.safe_load(f)

        # save model config 
        with open(os.path.join(config_save_path, 'model_config.yaml'), 'w') as f:
            yaml.safe_dump(model_cfg, f)

    cam_models = dict(zip(["left", "right", "front", "back"], [CamModel(direction, vehicle) for direction in ["left", "right", "front", "back"]]))
    # create tensorboard writer
    writer = tensorboardX.SummaryWriter(tensorboard_path)

    dataset_cfg["data_root"] = cfg["data_root"]
    dataset_cfg["ann_prefix"] = cfg["annotation_prefix"]
    dataset_cfg["img_prefix"] = cfg["image_prefix"]   
    dataset_cfg['test_mode'] = False
    dataset_cfg["vehicle"] = vehicle
    dataset = build_dataset(dataset_cfg)

    dataset_cfg['test_mode'] = True
    val_dataset = build_dataset(dataset_cfg)

    val_loss = 1000
    model_cfg["vehicle"] = cfg["vehicle"]
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

    if ckpt_file is not None:
        model.load_state_dict(torch.load(ckpt_file))
    else:
        model.init_weights()

    model.to_device(device)
    model.train()
    best_epoch = None

    if cfg["freeze_backbone"]:
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
    
    # setup optimizer and hyper parameters
    if 'type' in cfg['optimizer']:
        optimizer = getattr(torch.optim, cfg['optimizer']['type'])(model.parameters(),
                                    lr=cfg['optimizer']['lr'],
                                    weight_decay=cfg['optimizer']['weight_decay'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=cfg["optimizer"]['lr'])
    lr_scheduler = create_lr_scheduler(optimizer, cfg["optimizer"]['lr'], 
                                       cfg['lr_config']['warmup_iters'], 
                                       cfg["lr_config"]['warmup_ratio'])
    grad_clip_cfg = cfg['optimizer']['grad_clip'] if 'grad_clip' in cfg['optimizer'] else None

    # train model
    for epoch in range(start_epoch, total_epochs):
        for i, data in tqdm(enumerate(train_loader)):
            to_device(data, device)
            data['img_metas'][0]['epoch'] = epoch
            data["return_loss"] = True

            ############################################
            # DEBUG
            # img_data = data['img'][0]
            # if not os.path.exists("./debug_img"):
            #     os.makedirs("./debug_img")

            # for j, cam_dir in enumerate(data['img_metas'][0]['direction']):
            #     img = (img_data[j].detach().cpu().numpy().transpose(1, 2, 0) * 80).astype(np.uint8)
            #     cv2.imwrite(f"./debug_img/{cam_dir}_{j}_{i}.jpg", img)
            ############################################

            loss_res = model(**data)
            loss = sum([loss_res[key][0] if isinstance(loss_res[key], list) else loss_res[key] for key in loss_res])
            print("Epoch: ", epoch, "Iter: ", i, "Loss: ", loss.item())


            loss.backward()

            writer.add_scalar('train loss', loss.item(), epoch * len(train_loader) + i)
            for key in loss_res:
                loss_single = loss_res[key][0] if isinstance(loss_res[key], list) else loss_res[key]
                writer.add_scalar(f'train {key}', loss_single.item(), epoch * len(train_loader) + i)
                print(f"   Loss {key}: {loss_single.item()}")   
            if grad_clip_cfg is not None:
                clip_grads(model.parameters(), grad_clip_cfg)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if (epoch + 1) % eval_interval == 0 or epoch == 0:
            model.eval()
            # eval model
            with torch.no_grad():
                val_epoch = epoch // eval_interval
                val_loss = []
                for i, data in tqdm(enumerate(val_loader)):
                    to_device(data, device)
                    data["return_loss"] = True
                    data['img_metas'][0]['test'] = 1
                
                    loss_res = model(**data)
                    loss = sum([loss_res[key][0] if isinstance(loss_res[key], list) else loss_res[key] for key in loss_res])

                    val_loss.append(loss.item())
                    # record loss
                    writer.add_scalar('val loss', loss.item(), val_epoch * len(val_loader) + i)
                    for key in loss_res:
                        loss_single = loss_res[key][0] if isinstance(loss_res[key], list) else loss_res[key]
                        writer.add_scalar(f'val {key}', loss_single.item(), val_epoch * len(val_loader) + i)

                val_loss = np.mean(val_loss)
                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), f"{save_path}/best.pth")
                    print(f"Save best model at epoch {epoch} in {save_path}")

                    if cfg["save_backbone"]:
                        torch.save(model.backbone.state_dict(), f"{save_path}/best_backbone.pth")
                        print(f"Save backbone at epoch {epoch}")
                          
                if (epoch + 1) % 10 == 0:
                    torch.save(model.state_dict(), f"{save_path}/epoch_{epoch}.pth")
                    print(f"Save best model at epoch {epoch} in {save_path}")

                    if cfg["save_backbone"]:
                        torch.save(model.backbone.state_dict(), f"{save_path}/epoch_{epoch}_backbone.pth")
                        print(f"Save backbone at epoch {epoch}")

            if epoch >= 0: # total_epochs -11:
                bbox_res_path = os.path.join(save_path, f"{epoch}", "val_bbox_pre")
                bbox_train_res_path = os.path.join(save_path, f"{epoch}", "train_bbox_pre")
                create_folder(bbox_res_path)
                create_folder(bbox_train_res_path)
                            
                with torch.no_grad():
                    for i, data in tqdm(enumerate(train_loader)):
                        to_device(data, device)
                        data["return_loss"] = False
                        bbox_res = model(**data)
                        # save bbox_res
                        for idx, (bbox, img_meta) in enumerate(zip(bbox_res, data['img_metas'])):
                            # extract bbox from bbox_res
                            img_meta = img_meta['img_info']
                            if 'img_bbox' in bbox:
                                bbox = bbox['img_bbox']['boxes_3d'].tensor.cpu().numpy()[:, :7]
                            else:
                                bbox = bbox['boxes_3d'].tensor.cpu().numpy()[:, :7]
                            gt_bbox = data['gt_bboxes_3d'][idx].cpu().numpy()[:, :7]
                            
                            if cfg['bbox_coordination'] == "CAM":
                                cam_model = cam_models[img_meta["direction"]]
                                filename = img_meta['filename']
                                detection_visualization(bbox, gt_bbox, filename, cam_model, bbox_train_res_path, bboxes_coor = "CAM")
                            elif cfg['bbox_coordination'] == "Lidar":
                                vis_imgs = []
                                for filename, direction in zip(img_meta['img_filename'], img_meta['direction']):
                                    cam_model = cam_models[direction]
                                    img = detection_visualization(bbox, gt_bbox, filename, cam_model, bbox_train_res_path, bboxes_coor = "Lidar")
                                    vis_imgs.append(img)
                                vis_imgs = np.concatenate(vis_imgs, axis=1)
                                cv2.imwrite(os.path.join(bbox_train_res_path, f"{filename.split('/')[-1].split('.jpg')[0]}.jpg"), vis_imgs)

                with torch.no_grad():
                    val_epoch = epoch // eval_interval
                    for i, data in tqdm(enumerate(val_loader)):
                        to_device(data, device)
                        data["return_loss"] = False
                        bbox_res = model(**data)
                        # save bbox_res
                        for idx ,(bbox, img_meta) in enumerate(zip(bbox_res, data['img_metas'])):
                            # extract bbox from bbox_res
                            img_meta = img_meta['img_info']
                            if 'img_bbox' in bbox:
                                bbox = bbox['img_bbox']['boxes_3d'].tensor.cpu().numpy()[:, :7]
                            else:
                                bbox = bbox['boxes_3d'].tensor.cpu().numpy()[:, :7]
                            gt_bbox = data['gt_bboxes_3d'][idx].cpu().numpy()[:, :7]
                            if cfg['bbox_coordination'] == "CAM":
                                cam_model = cam_models[img_meta["direction"]]
                                filename = img_meta['filename']
                                img = detection_visualization(bbox, gt_bbox, filename, cam_model, bbox_res_path, bboxes_coor = "CAM")
                            elif cfg['bbox_coordination'] == "Lidar":
                                vis_imgs = []
                                for filename, direction in zip(img_meta['img_filename'], img_meta['direction']):
                                    cam_model = cam_models[direction]
                                    img = detection_visualization(bbox, gt_bbox, filename, cam_model, bbox_res_path, bboxes_coor = "Lidar")
                                    vis_imgs.append(img)
                                vis_imgs = np.concatenate(vis_imgs, axis=1)
                                cv2.imwrite(os.path.join(bbox_res_path, f"{filename.split('/')[-1].split('.jpg')[0]}.jpg"), vis_imgs)
            model.train()
    print("Training finished.")
    writer.close()
    print('Best epoch is ', best_epoch) 

if __name__ == '__main__':
    main()
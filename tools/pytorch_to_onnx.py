#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import numpy as np
import yaml
import argparse
import os
# Add the path to the 'src' directory (parent of common and centerserver)
# from models.builder import build_detector
import sys
# get current path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from models.builder import build_detector
import torch
import onnx_graphsurgeon as gs
from time import time
import logging

def arg_parse():
    parser = argparse.ArgumentParser(description='Convert Pytorch model to ONNX model')
    parser.add_argument('--vehicle', type=str, default='hycan', help='vehicle name')
    parser.add_argument('--ckpt_path', type=str, default=None, help='path to save the onnx model')
    return parser.parse_args()

args = arg_parse()
vehicle = args.vehicle
config_path = './configs/models'
config = os.path.join(config_path, 'mv_dfm_{}.yaml'.format(vehicle))
with open(config, 'r') as f:
    config = yaml.safe_load(f)
detector = build_detector(config)
device = "cuda" if torch.cuda.is_available() else "cpu"
# detector.load_state_dict(torch.load(ckpt_path))
height = 360
width = 640

pad_height = 368
pad_width = 640

# start to turn torch into onnx
if args.ckpt_path is not None:
    detector.load_state_dict(torch.load(args.ckpt_path))
    onnx_model_path = args.ckpt_path.replace('.pth', '.onnx')
else:
    ckpt_path = '../ckpt/onnx/'
    os.makedirs(ckpt_path, exist_ok=True)
    onnx_model_path = os.path.join(ckpt_path, '{}.onnx'.format(vehicle))

detector.to(device)
detector.eval()

dummy_input = torch.randn(1, 4, 3, pad_height, pad_width).to(device)
log_level = logging.INFO
img_metas = [
    dict(
        img_shape=[(height, width, 3)] * 4,
        ori_shape=[(720, 1280, 3)] *4,
        pad_shape=[(pad_height, pad_width, 3)] * 4,
        scale_factor=torch.FloatTensor([height / 720, height / 720]).to(device),
        flip=False,
        keep_ratio=True,
        num_views = 4,
        num_ref_frames = 0,
        direction = ['front', 'back', 'left', 'right'],
        pad_size_divisor = 16,
    )
]

st = time()

with torch.no_grad():
    torch.onnx.export(detector, (dummy_input, img_metas, False), 
                        onnx_model_path, verbose=True, opset_version=12,
                        input_names=['input', 'img_metas', 'return_loss'],
                        output_names=['boxes_3d'],
                        do_constant_folding=False)
print("Exporting onnx model costs: ", time() - st)
print('Now run the following command to convert the onnx model to int32:')
print('onnxsim {} {}'.format(onnx_model_path, onnx_model_path.split('.onnx')[0] + '_simplified.onnx'))
print('polygraphy surgeon sanitize {} --fold-constants --output {}'.format(onnx_model_path.split('.onnx')[0] + '_simplified.onnx', onnx_model_path.split('.onnx')[0] + '_folded.onnx'))
print('python tools/onnx_to_tensorrt.py --onnx_model_path {}'.format(onnx_model_path.split('.onnx')[0] + '_folded.onnx'))
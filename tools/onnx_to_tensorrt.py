import time
from mmdeploy.utils import Backend
from mmdeploy.apis.utils import to_backend
import logging
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Convert ONNX model to TensorRT model')
    parser.add_argument('--onnx_model_path', type=str, default='/media/data/ckpt/onnx/hycan_folded.onnx', help='ONNX model path')
    return parser.parse_args() 

st = time.time()

args = parse_args()
device = 'cuda:0'
log_level = logging.INFO

height = 368
width = 640
onnx_model_path = args.onnx_model_path 

backend = Backend.TENSORRT
backend_config = {
    'backend_config': {
        'common_config': {
            'max_workspace_size': 1 << 36,
        },
        'fp16_mode' : True,
        'model_inputs' : [{
            # "save_file": onnx_model_path.split('.onnx')[0] + '.engine',
            "input_shapes": {
                "input": {
                    "min_shape": [1, 4, 3, height, width],
                    "opt_shape": [1, 4, 3, height, width],
                    "max_shape": [1, 4, 3, height, width],
                }
            }
        }]
    }
}

if onnx_model_path[0] != '/':
    workdir = os.path.abspath(os.path.join(os.getcwd(), *onnx_model_path.split('/')[:-1]))
else:
    workdir = '/' + os.path.join(*onnx_model_path.split('/')[:-1])

trt_model_file = to_backend(
    backend,
    [onnx_model_path],
    work_dir=workdir,
    deploy_cfg=backend_config,
    log_level=log_level,
    device=device)

print("TensorRT model is saved at: ", trt_model_file)
print("Time cost: ", time.time() - st)

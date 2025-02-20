import tensorrt as trt
import torch
import numpy as np
import pycuda.driver as cuda
import os
abs_path = os.path.abspath(__file__)
import sys
sys.path.append(abs_path.split('tools')[0])
from models.detectors import MultiViewDfMFisheye

class LayerProfiler(trt.IProfiler):
    def report_layer_time(self, layer_name, ms):
        print(f"Layer {layer_name}: {ms:.3f} ms")

# 加载 TensorRT 引擎
def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

# 创建执行上下文
def create_execution_context(engine):
    return engine.create_execution_context()

# 分配输入输出缓冲区
def allocate_buffers(engine, batch_size=1):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        if engine.binding_is_input(binding):
            input_host_mem = cuda.pagelocked_empty(size, dtype)
            input_device_mem = cuda.mem_alloc(input_host_mem.nbytes)
            inputs.append((input_host_mem, input_device_mem))
        else:
            output_host_mem = cuda.pagelocked_empty(size, dtype)
            output_device_mem = cuda.mem_alloc(output_host_mem.nbytes)
            outputs.append((output_host_mem, output_device_mem))
        bindings.append(int(input_device_mem if engine.binding_is_input(binding) else output_device_mem))

    return inputs, outputs, bindings, stream

# 推理函数
def infer(engine, context, input_dict):
    # 提取输入字典中的 key 和 value
    input = input_dict['input']  # np.ndarray
    # 分配缓冲区
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # 填充输入数据到缓冲区
    np.copyto(inputs[0][0], input.ravel())

    # 将输入数据拷贝到设备
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)

    # 执行推理
    context.execute_async_v2(bindings, stream.handle)

    # 将输出数据拷贝回主机
    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)

    # 同步流
    stream.synchronize()

    # 获取推理结果
    output = outputs[0]
    return output

class TRTModel:
    def __init__(self, engine_file_path, pos_thresh=0.5, nms_thresh=0.5, box_code_size=7):
        self.engine = load_engine(engine_file_path)
        self.context = create_execution_context(self.engine)
        self.pos_thresh = pos_thresh
        self.cfg = {'use_rotate_nms': True, 'nms_thr': nms_thresh, 
                    'box_code_size': box_code_size}

    def __call__(self, input_tesor, img_metas, return_loss):
        input_dict = {'input': input_tesor}
        output = infer(self.engine, self.context, input_dict)[0]
        bbox_res = MultiViewDfMFisheye.nms_for_bboxes(output, score_thr=self.pos_thresh, cfg=self.cfg)
        return bbox_res

# 主函数
if __name__ == "__main__":
    # 加载引擎
    import pycuda.autoinit
    engine = load_engine("/media/data/V2X_AEB/model_ckpt/hycan_mv_fcos3d_folded.engine")

    # 创建上下文
    context = create_execution_context(engine)

    # 实例化 profiler
    profiler = LayerProfiler()

    # 将 profiler 赋给 execution context
    context.profiler = profiler

    # 准备输入数据
    input_tensor = np.random.randn(1, 4, 3, 368, 640)  # 假设的输入大小
    img_metas = {'info': 'example metadata',
                 'scale_factor': np.array([0.5, 0.5, 0.5, 0.5], np.float32)}  # 假设的元数据
    input_dict = {'input': input_tensor, 'img_scale_factor': img_metas['scale_factor'][:2]}

    # 运行推理
    import time
    start = time.time()
    output = infer(engine, context, input_dict)
    bbox_res = MultiViewDfMFisheye.nms_for_bboxes(output[0])
    print("Inference Output: ", bbox_res)
    # print("Inference Output Shape: ", output.shape)
    print("Inference Time: ", time.time() - start)
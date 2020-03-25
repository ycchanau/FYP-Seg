import argparse
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from math import ceil
from glob import glob
from PIL import Image
import dataloaders
import models
from utils.helpers import colorize_mask
import time
import torch.onnx
import onnx
import onnx.optimizer
import onnx.helper
import onnxruntime
import time

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

def to_numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.float32) if tensor.requires_grad else tensor.cpu().numpy().astype(np.float32)

def add_initializers_into_inputs(model: onnx.ModelProto) -> onnx.ModelProto:
    for x in model.graph.initializer:
        input_names = [x.name for x in model.graph.input]
        if x.name not in input_names:
            shape = onnx.TensorShapeProto()
            for dim in x.dims:
                shape.dim.extend([onnx.TensorShapeProto.Dimension(dim_value=dim)])
            model.graph.input.extend(
                [onnx.ValueInfoProto(name=x.name,
                                     type=onnx.TypeProto(tensor_type=onnx.TypeProto.Tensor(elem_type=x.data_type,
                                                                                           shape=shape)))])
    return model

def optimize(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    :param model: The onnx model.
    :return: The optimized onnx model.
    Before simplifying, use this method to generate value_info, which is used in `forward_all`
    After simplifying, use this method to fold constants generated in previous step into initializer,
    and eliminate unused constants.
    """

    # Due to a onnx bug, https://github.com/onnx/onnx/issues/2417, we need to add missing initializers into inputs
    input_num = len(model.graph.input)
    model = add_initializers_into_inputs(model)
    onnx.helper.strip_doc_string(model)
    model = onnx.optimizer.optimize(model, ['eliminate_deadend', 'eliminate_identity', 'eliminate_nop_dropout',
                                            'eliminate_nop_monotone_argmax', 'eliminate_nop_pad',
                                            'extract_constant_to_initializer', 'eliminate_unused_initializer',
                                            'eliminate_nop_transpose'],
                                    fixed_point=True)
    del model.graph.input[input_num:]
    onnx.checker.check_model(model)
    return model

##################################################
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        print(binding)
        print(size)
        print(dtype)
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream
######################################################

def main():
    args = parse_arguments()
    config = json.load(open(args.config))

    # Dataset used for training the model
    dataset_type = config['train_loader']['type']
    loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
    to_tensor = transforms.ToTensor()
    #normalize = transforms.Normalize(loader.MEAN, loader.STD)
    num_classes = loader.dataset.num_classes
    palette = loader.dataset.palette
    base_size = loader.dataset.base_size

    # Model
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    checkpoint = torch.load(args.model)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        print("getting checkpoint")
        checkpoint = checkpoint['state_dict']

    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        print('convert model to DataParallel')
        model = torch.nn.DataParallel(model)

    model.load_state_dict(checkpoint)
    model.to(device)
    '''
    print("saving model")
    torch.save(model.module.state_dict(), 'model.pkl') 
    print(model.module.state_dict())
    '''
    model.eval()
    
    ###########################################################################################################
    
    batch_size=1

    image = Image.open(args.images).convert('RGB')
    original_size=image.size
    image_name = os.path.basename(args.images)
    target=Image.open("/home/ubuntu/TM2/mask/"+image_name)

    if base_size:
        image = image.resize(size=(base_size, base_size), resample=Image.BILINEAR)
        target = target.resize(size=(base_size, base_size), resample=Image.NEAREST)

    #dummy_input = torch.randn(batch_size, 3, base_size, base_size, device="cuda")
    dummy_input = to_tensor(image).unsqueeze(0).to(device)
    print("exporting model")
    # Export the model
    torch.onnx.export(model.module,           # model being run
                        dummy_input,                         # model input (or a tuple for multiple inputs)
                        "model.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      #dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}},    # variable lenght axes
                      verbose=True,
                      opset_version=11,
                      keep_initializers_as_inputs=True)
    
    
    # #####################################################################
    print("checking onnx model")
    onnx_model = onnx.load("model.onnx")
    onnx.checker.check_model(onnx_model)
    onnx.checker.check_graph(onnx_model.graph)
    print("onnx model is checked")

    # #####################################################################
    print("optimizing onnx model")
    optimized_onnx_model = optimize(onnx_model)
    onnx.checker.check_model(optimized_onnx_model)
    onnx.checker.check_graph(optimized_onnx_model.graph)
    print("optimization done")
    onnx.save(optimized_onnx_model, 'optimized_model.onnx')

    #############################################
    image = Image.open(args.images).convert('RGB')
    original_size=image.size
    image_name = os.path.basename(args.images)
    target=Image.open("/home/ubuntu/TM2/mask/"+image_name)

    if base_size:
        image = image.resize(size=(base_size, base_size), resample=Image.BILINEAR)
        target = target.resize(size=(base_size, base_size), resample=Image.NEAREST)

    ####################################################################
    pytorch_input = to_tensor(image).unsqueeze(0)
    pytorch_time=time.time()
    with torch.no_grad():
        pytorch_prediction = model(pytorch_input.to(device))
        pytorch_prediction = to_numpy(pytorch_prediction)
        print("pytorch time used:{}".format(time.time()-pytorch_time))

    #######################################################################

    ort_session = onnxruntime.InferenceSession("optimized_model.onnx")
    ort_input = to_numpy(pytorch_input)
    ort_inputs = {ort_session.get_inputs()[0].name: ort_input}
    ort_time=time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    ort_prediction = ort_outs[0]
    print("ort time used:{}".format(time.time()-ort_time))

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(pytorch_prediction, ort_prediction, rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    ###################################################################

    with open("model.engine", 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        print("Trying Tensorrt")
        shape_of_output = (batch_size, num_classes, 128, 128)
        engine = runtime.deserialize_cuda_engine(f.read())
        inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings
        with engine.create_execution_context() as context:
            inputs[0].host = ort_input.reshape(-1)

            t1 = time.time()
            trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) # numpy data
            t2 = time.time()
            feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)

            print('TensorRT ok')
            print("Inference time with the TensorRT engine: {}".format(t2-t1))
            np.testing.assert_allclose(pytorch_prediction, feat, rtol=1e-03, atol=1e-05)
            print('All completed!')
    print("DLLMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM")   






def parse_arguments():
    parser = argparse.ArgumentParser(description='convertToONNX')
    parser.add_argument('-c', '--config', default='./saved/TM2-HRNetV2_OCR/03-01_07-05/config.json',type=str,
                        help='The config used to train the model')
    parser.add_argument('-m', '--model', default='./saved/TM2-HRNetV2_OCR/03-01_07-05/checkpoint-epoch200.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default='/home/ubuntu/TM2/photo/104.png', type=str,
                        help='Path to the images to be segmented')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
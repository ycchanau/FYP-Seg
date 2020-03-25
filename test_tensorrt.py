import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
#import torch
import os
import time
#from PIL import Image
#import cv2
#import torchvision
import sys
import glob
import math
import logging
import argparse


TRT_LOGGER = trt.Logger()
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

def add_profiles(config, inputs, opt_profiles):
    logger.debug("=== Optimization Profiles ===")
    for i, profile in enumerate(opt_profiles):
        for inp in inputs:
            _min, _opt, _max = profile.get_shape(inp.name)
            logger.debug("{} - OptProfile {} - Min {} Opt {} Max {}".format(inp.name, i, _min, _opt, _max))
        config.add_optimization_profile(profile)

def mark_outputs(network):
    # Mark last layer's outputs if not already marked
    # NOTE: This may not be correct in all cases
    last_layer = network.get_layer(network.num_layers-1)
    if not last_layer.num_outputs:
        logger.error("Last layer contains no outputs.")
        return

    for i in range(last_layer.num_outputs):
        network.mark_output(last_layer.get_output(i))


def check_network(network):
    if not network.num_outputs:
        logger.warning("No output nodes found, marking last layer's outputs as network outputs. Correct this if wrong.")
        mark_outputs(network)
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    max_len = max([len(inp.name) for inp in inputs] + [len(out.name) for out in outputs])

    logger.debug("=== Network Description ===")
    for i, inp in enumerate(inputs):
        logger.debug("Input  {0} | Name: {1:{2}} | Shape: {3}".format(i, inp.name, max_len, inp.shape))
    for i, out in enumerate(outputs):
        logger.debug("Output {0} | Name: {1:{2}} | Shape: {3}".format(i, out.name, max_len, out.shape))


def get_batch_sizes(max_batch_size):
    # Returns powers of 2, up to and including max_batch_size
    max_exponent = math.log2(max_batch_size)
    for i in range(int(max_exponent)+1):
        batch_size = 2**i
        yield batch_size
    
    if max_batch_size != batch_size:
        yield max_batch_size


# TODO: This only covers dynamic shape for batch size, not dynamic shape for other dimensions
def create_optimization_profiles(builder, inputs, batch_sizes=[1,4,8]): 
    # Check if all inputs are fixed explicit batch to create a single profile and avoid duplicates
    if all([inp.shape[0] > -1 for inp in inputs]):
        profile = builder.create_optimization_profile()
        for inp in inputs:
            fbs, shape = inp.shape[0], inp.shape[1:]
            profile.set_shape(inp.name, min=(fbs, *shape), opt=(fbs, *shape), max=(fbs, *shape))
            return [profile]
    
    # Otherwise for mixed fixed+dynamic explicit batch inputs, create several profiles
    profiles = {}
    for bs in batch_sizes:
        if not profiles.get(bs):
            profiles[bs] = builder.create_optimization_profile()

        for inp in inputs: 
            shape = inp.shape[1:]
            # Check if fixed explicit batch
            if inp.shape[0] > -1:
                bs = inp.shape[0]

            profiles[bs].set_shape(inp.name, min=(bs, *shape), opt=(bs, *shape), max=(bs, *shape))

    return list(profiles.values())

def get_engine(args):
    # Network flags
    network_flags = 0
    if args.explicit_batch:
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if args.explicit_precision:
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)

    builder_flag_map = {
            'gpu_fallback': trt.BuilderFlag.GPU_FALLBACK,
            'refittable': trt.BuilderFlag.REFIT,
            'debug': trt.BuilderFlag.DEBUG,
            'strict_types': trt.BuilderFlag.STRICT_TYPES,
            'fp16': trt.BuilderFlag.FP16,
            'int8': trt.BuilderFlag.INT8,
    }

    # Building engine
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(network_flags) as network, \
         builder.create_builder_config() as config, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
            
        config.max_workspace_size = 2**29 # 1GiB

        # Set Builder Config Flags
        for flag in builder_flag_map:
            if getattr(args, flag):
                logger.info("Setting {}".format(builder_flag_map[flag]))
                config.set_flag(builder_flag_map[flag])

        if args.fp16 and not builder.platform_has_fast_fp16:
            logger.warning("FP16 not supported on this platform.")

        if args.int8 and not builder.platform_has_fast_int8:
            logger.warning("INT8 not supported on this platform.")
        '''
        if args.int8:
            config.int8_calibrator = get_int8_calibrator(args.calibration_cache,
                                                         args.calibration_data,
                                                         args.max_calibration_size,
                                                         args.preprocess_func,
                                                         args.calibration_batch_size)
        '''

        # Fill network atrributes with information by parsing model
        with open(args.onnx, "rb") as f:
            if not parser.parse(f.read()):
                print('ERROR: Failed to parse the ONNX file: {}'.format(args.onnx))
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                sys.exit(1)

        # Display network info and check certain properties
        check_network(network)
        #?????????????????????????????????????????????????????
        if args.explicit_batch:
            # Add optimization profiles
            batch_sizes = [1, 4]
            inputs = [network.get_input(i) for i in range(network.num_inputs)]
            opt_profiles = create_optimization_profiles(builder, inputs, batch_sizes)
            add_profiles(config, inputs, opt_profiles)
        # Implicit Batch Network
        else:
            builder.max_batch_size = args.max_batch_size

        logger.info("Building Engine...")
        with builder.build_engine(network, config) as engine, open(args.output, "wb") as f:
            logger.info("Serializing engine to file: {:}".format(args.output))
            f.write(engine.serialize())


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


#python test_tensorrt.py --explicit-batch -v --explicit-precision
def main():
    parser = argparse.ArgumentParser(description="Creates a TensorRT engine from the provided ONNX file.\n")
    parser.add_argument("--onnx", type=str, default="model.onnx", help="The ONNX model file to convert to TensorRT")
    parser.add_argument("-o", "--output", type=str, default="model.engine", help="The path at which to write the engine")
    parser.add_argument("-b", "--max-batch-size", type=int, default=1, help="The max batch size for the TensorRT engine input")
    parser.add_argument("-v", "--verbosity", action="count", help="Verbosity for logging. (None) for ERROR, (-v) for INFO/WARNING/ERROR, (-vv) for VERBOSE.")
    parser.add_argument("--explicit-batch", action='store_true', help="Set trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH.")
    parser.add_argument("--explicit-precision", action='store_true', help="Set trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION.")
    parser.add_argument("--gpu-fallback", action='store_true', help="Set trt.BuilderFlag.GPU_FALLBACK.")
    parser.add_argument("--refittable", action='store_true', help="Set trt.BuilderFlag.REFIT.")
    parser.add_argument("--debug", action='store_true', help="Set trt.BuilderFlag.DEBUG.")
    parser.add_argument("--strict-types", action='store_true', help="Set trt.BuilderFlag.STRICT_TYPES.")
    parser.add_argument("--fp16", action="store_true", help="Attempt to use FP16 kernels when possible.")
    parser.add_argument("--int8", action="store_true", help="Attempt to use INT8 kernels when possible. This should generally be used in addition to the --fp16 flag. \
                                                             ONLY SUPPORTS RESNET-LIKE MODELS SUCH AS RESNET50/VGG16/INCEPTION/etc.")
    parser.add_argument("--calibration-cache", help="(INT8 ONLY) The path to read/write from calibration cache.", default="calibration.cache")
    parser.add_argument("--calibration-data", help="(INT8 ONLY) The directory containing {*.jpg, *.jpeg, *.png} files to use for calibration. (ex: Imagenet Validation Set)", default=None)
    parser.add_argument("--calibration-batch-size", help="(INT8 ONLY) The batch size to use during calibration.", type=int, default=32)
    parser.add_argument("--max-calibration-size", help="(INT8 ONLY) The max number of data to calibrate on from --calibration-data.", type=int, default=512)
    parser.add_argument("-p", "--preprocess_func", type=str, default=None, help="(INT8 ONLY) Function defined in 'processing.py' to use for pre-processing calibration data.")
    args, _ = parser.parse_known_args()

    if args.verbosity is None:
        TRT_LOGGER.min_severity = trt.Logger.Severity.ERROR
    # -v
    elif args.verbosity == 1:
        TRT_LOGGER.min_severity = trt.Logger.Severity.INFO
    # -vv
    else:
        TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE
    logger.info("TRT_LOGGER Verbosity: {:}".format(TRT_LOGGER.min_severity))


    base_size=512
    channel_no=4
    max_batch_size=1
    shape_of_output = (max_batch_size, channel_no, 128, 128)

    image=np.random.randn(max_batch_size,3,base_size,base_size).astype(np.float32)

    # engine = get_engine(args)

    '''
    t1 = time.time()
    output = infer(engine_path=args.output, batch_size=max_batch_size, input_data=image)
    output = postprocess_the_outputs(output, shape_of_output)
    t2 = time.time()
    print('TensorRT ok')
    print("Inference time with the TensorRT engine: {}".format(t2-t1))
    print('All completed!')
    '''

    with open(args.output, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        print(args.output)
        engine = runtime.deserialize_cuda_engine(f.read())
        inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings
        with engine.create_execution_context() as context:
            print(image.shape)
            print(image.dtype)
            inputs[0].host = image.reshape(-1)

            t1 = time.time()
            trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) # numpy data
            t2 = time.time()
            feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)

            print('TensorRT ok')
            print("Inference time with the TensorRT engine: {}".format(t2-t1))
            print('All completed!')

if __name__ == "__main__":
    main()
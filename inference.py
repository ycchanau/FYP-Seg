import argparse
import scipy
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image
import dataloaders
import models
from utils.helpers import colorize_mask
import time

import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
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


def to_numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.float32) if tensor.requires_grad else tensor.cpu().numpy().astype(np.float32)
######################################################

def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img

def sliding_predict(model, image, num_classes, flip=True):
    image_size = image.shape
    tile_size = (int(image_size[2]//2.5), int(image_size[3]//2.5))
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    
    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    total_predictions = np.zeros((num_classes, image_size[2], image_size[3]))
    count_predictions = np.zeros((image_size[2], image_size[3]))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = image[:, :, y_min:y_max, x_min:x_max]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            padded_prediction = model(padded_img)
            if flip:
                fliped_img = padded_img.flip(-1)
                fliped_predictions = model(padded_img.flip(-1))
                padded_prediction = 0.5 * (fliped_predictions.flip(-1) + padded_prediction)
            predictions = padded_prediction[:, :, :img.shape[2], :img.shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.data.cpu().numpy().squeeze(0)

    total_predictions /= count_predictions
    return total_predictions


def multi_scale_predict(model, image, scales, num_classes, device, flip=False):
    input_size = (image.size(2), image.size(3))
    upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    total_predictions = np.zeros((num_classes, image.size(2), image.size(3)))

    image = image.data.data.cpu().numpy()
    for scale in scales:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, float(scale), float(scale)), order=1, prefilter=False)
        scaled_img = torch.from_numpy(scaled_img).to(device)
        scaled_prediction = upsample(model(scaled_img).cpu())

        if flip:
            fliped_img = scaled_img.flip(-1).to(device)
            fliped_predictions = upsample(model(fliped_img).cpu())
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)
    return total_predictions


def save_images(image, GT, mask, output_path, image_file, palette, original_size):
	# Saves the image, the model output and the results after the post processing
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)

    w, h = image.size

    if original_size:
        w, h =original_size

    image_file = os.path.basename(image_file).split('.')[0]
    colorized_mask = colorize_mask(mask, palette)
    GT = GT.convert('P')
    GT.putpalette(palette)


    if image.size != original_size:
        image = image.resize(size=original_size, resample=Image.BILINEAR)
    if colorized_mask.size != original_size:
        colorized_mask = colorized_mask.resize(size=original_size, resample=Image.NEAREST)
    if GT != original_size:
        GT = GT.resize(size=original_size, resample=Image.NEAREST)

    blend = Image.blend(image, colorized_mask.convert('RGB'), 0.5)

    colorized_mask.save(os.path.join(output_path, image_file+'.png'))
    output_im = Image.new('RGB', (w*4, h))
    output_im.paste(image, (0,0))
    output_im.paste(GT, (w,0))
    output_im.paste(colorized_mask, (w*2,0))
    output_im.paste(blend, (w*3,0))
    output_im.save(os.path.join(output_path, image_file+'_colorized.png'))
    # mask_img = Image.fromarray(mask, 'L')
    # mask_img.save(os.path.join(output_path, image_file+'.png'))

def my_pixel_accuracy(output, target):
    output = np.asarray(output)
    target = np.asarray(target)
    pixel_labeled = np.sum(target >= 0)
    pixel_correct = np.sum((output == target) * (target >= 0))
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled
def class_pixel_accuracy(output, target, cls):
    output = np.asarray(output)
    target = np.asarray(target)
    pixel_labeled = np.sum(target == cls)
    pixel_correct = np.sum((output == target) * (target == cls))
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    pix_acc = (pixel_correct/pixel_labeled) if pixel_labeled!=0 else 1
    return pixel_correct, pixel_labeled, pix_acc
def inter_over_union(output, target, num_class):
    output = np.asarray(output) + 1
    target = np.asarray(target) + 1
    output = output * (target > 0)

    intersection = output * (output == target)
    area_inter, _ = np.histogram(intersection, bins=num_class, range=(1, num_class))
    area_pred, _ = np.histogram(output, bins=num_class, range=(1, num_class))
    area_lab, _ = np.histogram(target, bins=num_class, range=(1, num_class))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    iou = np.divide(area_inter,area_union,out=np.ones(area_inter.shape,dtype=float),where=area_union!=0)
    return area_inter, area_union, iou


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

    print(config['arch']['type'])
    # Model
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    checkpoint = torch.load(args.model)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    image_files = sorted(glob(os.path.join(args.images, f'*.{args.extension}')))

    # with open("fp16_model.engine", 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime, torch.no_grad():

    #     engine = runtime.deserialize_cuda_engine(f.read())
    #     inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings
    #     shape_of_output = (1, num_classes, 128, 128)

    #     with engine.create_execution_context() as context:

    #         tbar = tqdm(image_files, ncols=100)
    #         total_image=0
    #         total_pixel_correct=0
    #         total_pixel_labeled=0
    #         cls_total_IOU = np.zeros(num_classes)
    #         cls_total_pix_acc = np.zeros(num_classes)
    #         cls_total_pix_correct=np.zeros(num_classes)
    #         cls_total_pix_labeled=np.zeros(num_classes)
    #         Total_Inference_Time=0

    #         for img_file in tbar:
    #             total_image += 1      
    #             image = Image.open(img_file).convert('RGB')
    #             original_size=image.size

    #             image_name = os.path.basename(img_file)
    #             target=Image.open("/home/ubuntu/TM2/mask/"+image_name)

    #             if base_size:
    #                 image = image.resize(size=(base_size, base_size), resample=Image.BILINEAR)
    #                 target = target.resize(size=(base_size, base_size), resample=Image.NEAREST)
                
    #             ticks = time.time()
    #             input = to_tensor(image).unsqueeze(0)
    #             trt_input_image = to_numpy(input)
    #             inputs[0].host = trt_input_image.reshape(-1)              
    #             trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    #             trt_feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)
    #             trt_prediction = F.interpolate(torch.from_numpy(trt_feat), size=(512,512), mode='bilinear', align_corners=True)                
    #             trt_prediction = trt_prediction.squeeze(0).cpu().numpy()
    #             trt_prediction = F.softmax(torch.from_numpy(trt_prediction), dim=0).argmax(0).cpu().numpy()

    #             Total_Inference_Time += time.time()-ticks

    #             ####################################
    #             _,_,iou = inter_over_union(trt_prediction, target, num_classes)
    #             cls_total_IOU = cls_total_IOU + iou

    #             pixel_correct, pixel_labeled=my_pixel_accuracy(trt_prediction,target)
    #             total_pixel_correct+=pixel_correct
    #             total_pixel_labeled+=pixel_labeled
    #             for i in range(num_classes):
    #                 cls_pix_correct, cls_pix_labeled, acc=class_pixel_accuracy(trt_prediction,target,i)
    #                 cls_total_pix_correct[i]+=cls_pix_correct
    #                 cls_total_pix_labeled[i]+=cls_pix_labeled
    #                 cls_total_pix_acc[i]+=acc

    #             save_images(image, target, trt_prediction, args.output, img_file, palette, original_size)

    #         print("time used: {}".format(Total_Inference_Time))
    #         print("pix acc: {}".format(total_pixel_correct/total_pixel_labeled))
    #         print("class pix acc: {}".format(cls_total_pix_correct/cls_total_pix_labeled))
    #         print("avg class IOU: {}".format(cls_total_IOU/total_image))
    #         print("avg class pix_acc: {}".format(cls_total_pix_acc/total_image))
            
    with torch.no_grad():
        tbar = tqdm(image_files, ncols=100)
        total_image=0
        total_pixel_correct=0
        total_pixel_labeled=0
        cls_total_IOU = np.zeros(num_classes)
        cls_total_pix_acc = np.zeros(num_classes)
        cls_total_pix_correct=np.zeros(num_classes)
        cls_total_pix_labeled=np.zeros(num_classes)
        Total_Inference_Time=0

        for img_file in tbar:
            total_image += 1      
            image = Image.open(img_file).convert('RGB')
            original_size=image.size

            image_name = os.path.basename(img_file)
            target=Image.open("/home/ubuntu/TM2/mask/"+image_name)

            if base_size:
                image = image.resize(size=(base_size, base_size), resample=Image.BILINEAR)
                target = target.resize(size=(base_size, base_size), resample=Image.NEAREST)

            #input = normalize(to_tensor(image)).unsqueeze(0)
            ticks = time.time()
            input = to_tensor(image).unsqueeze(0)

            if args.mode == 'multiscale':
                prediction = multi_scale_predict(model, input, scales, num_classes, device)
            elif args.mode == 'sliding':
                prediction = sliding_predict(model, input, num_classes)
            else:
                prediction = model(input.to(device))
                if config['arch']['type'][:2] == 'IC':
                    prediction = prediction[0]
                elif config['arch']['type'][-3:] == 'OCR':
                    prediction = prediction[0]
                elif 'Nearest' in config['arch']['type']:
                    prediction = prediction[0]
                elif 'Inference' in config['arch']['type']:
                    prediction = F.interpolate(prediction, size=(512,512), mode='bilinear', align_corners=True)                
                elif config['arch']['type'][:3] == 'Enc':
                    prediction = prediction[0]
                elif config['arch']['type'][:5] == 'DANet':
                    prediction = prediction[0]
                prediction = prediction.squeeze(0).cpu().numpy()

            prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()

            Total_Inference_Time += time.time()-ticks

            _,_,iou = inter_over_union(prediction, target, num_classes)
            cls_total_IOU = cls_total_IOU + iou



            pixel_correct, pixel_labeled=my_pixel_accuracy(prediction,target)
            total_pixel_correct+=pixel_correct
            total_pixel_labeled+=pixel_labeled
            for i in range(num_classes):
                cls_pix_correct, cls_pix_labeled, acc=class_pixel_accuracy(prediction,target,i)
                cls_total_pix_correct[i]+=cls_pix_correct
                cls_total_pix_labeled[i]+=cls_pix_labeled
                cls_total_pix_acc[i]+=acc

            save_images(image, target, prediction, args.output, img_file, palette, original_size)

        print("time used: {}".format(Total_Inference_Time))
        print("pix acc: {}".format(total_pixel_correct/total_pixel_labeled))
        print("class pix acc: {}".format(cls_total_pix_correct/cls_total_pix_labeled))
        print("avg class IOU: {}".format(cls_total_IOU/total_image))
        print("avg class pix_acc: {}".format(cls_total_pix_acc/total_image))



def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='VOC',type=str,
                        help='The config used to train the model')
    parser.add_argument('-mo', '--mode', default='normal', type=str,
                        help='Mode used for prediction: either [multiscale, sliding]')
    parser.add_argument('-m', '--model', default='model_weights.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=None, type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='outputs', type=str,  
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='png', type=str,
                        help='The extension of the images to be segmented')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()

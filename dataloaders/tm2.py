from base import BaseDataSet, BaseDataLoader
from PIL import Image
from glob import glob
import numpy as np
import scipy.io as sio
from utils import palette
import torch
import os
import cv2

class TM2Stuff(BaseDataSet):
    def __init__(self, warp_image = True, **kwargs):
        self.warp_image = warp_image
        self.num_classes = 4
        self.palette = palette.COCO_palette
        super(TM2Stuff, self).__init__(**kwargs)

    def _set_files(self):
        self.image_dir = os.path.join(self.root, "photo")
        self.label_dir = os.path.join(self.root, "mask")
        self.index_dir = os.path.join(self.root, "index")
        file_list = os.path.join(self.index_dir, self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]
    
    def _load_data(self, index):
        pth = self.files[index]
        image_id = self.files[index].split("/")[-1]
        image_path = os.path.join(self.image_dir, image_id + '.png')
        label_path = os.path.join(self.label_dir, image_id + '.png')
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        return image, label, image_id

class TM2(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=False, num_workers=1,
                    shuffle=True, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False, val=False):

        self.MEAN = [0.43931922, 0.41310471, 0.37480941]
        self.STD = [0.24272706, 0.23649098, 0.23429529]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': False,
            'base_size': base_size,
            'scale': False,
            'flip': False,
            'blur': False,
            'rotate': False,
            'return_id': return_id,
            'val': val
        }

        self.dataset = TM2Stuff(**kwargs)
        super(TM2, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
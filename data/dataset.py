from os import listdir
from os.path import join
import random

from PIL import Image, ImageFilter
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np

from utils.image_processing import read_split_image
from utils.bytesIO import PickledImageProvider, bytes_to_file


class DatasetFromObj(data.Dataset):
    def __init__(self, obj_path, augment=False, bold=False, rotate=False, blur=False):
        super(DatasetFromObj, self).__init__()
        self.image_provider = PickledImageProvider(obj_path)
        self.transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.augment = augment
        self.bold = bold
        self.rotate = rotate
        self.blur = blur

    def __getitem__(self, index):
        item = self.image_provider.examples[index]
        img_A, img_B = self.process(item[1])
        return item[0], img_A, img_B

    def __len__(self):
        return len(self.image_provider.examples)

    def process(self, img_bytes):
        "process byte stream to training data entry"
        image_file = bytes_to_file(img_bytes)
        img = Image.open(image_file)
        try:
            img_A, img_B = read_split_image(img)
            if self.augment:
                # augment the image by:
                # 1) enlarge the image
                # 2) random crop the image back to its original size
                # NOTE: image A and B needs to be in sync as how much
                # to be shifted
                w, h = img_A.size
                if self.bold:
                    multiplier = random.uniform(1.00, 1.40)
                else:
                    multiplier = random.uniform(1.00, 1.20)
                # add an eps to prevent cropping issue
                nw = int(multiplier * w) + 1
                nh = int(multiplier * h) + 1
                img_A = img_A.resize((nw, nh), Image.BICUBIC)
                img_B = img_B.resize((nw, nh), Image.BICUBIC)

                if self.rotate and random.random > 0.9:
                    angle_list = [0, 90, 180, 270]
                    random_angle = random.choice(angle_list)
                    img_A = img_A.rotate(random_angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255))
                    img_B = img_B.rotate(random_angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255))

                if self.blur and random.random > 0.8:
                    sigma_list = [1, 1.5, 2]
                    sigma = random.choice(sigma_list)
                    img_A = img_A.filter(ImageFilter.GaussianBlur(radius=sigma))
                    img_B = img_B.filter(ImageFilter.GaussianBlur(radius=sigma))

                img_A = transforms.ToTensor()(img_A)
                img_B = transforms.ToTensor()(img_B)

                w_offset = random.randint(0, max(0, nh - h - 1))
                h_offset = random.randint(0, max(0, nh - h - 1))

                img_A = img_A[:, h_offset: h_offset + h, w_offset: w_offset + h]
                img_B = img_B[:, h_offset: h_offset + h, w_offset: w_offset + h]

                img_A = self.transform(img_A)
                img_B = self.transform(img_B)

            return img_A, img_B

        finally:
            image_file.close()

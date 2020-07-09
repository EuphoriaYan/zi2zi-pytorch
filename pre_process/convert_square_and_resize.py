from PIL import Image, UnidentifiedImageError
import numpy as np
import os
from tqdm import tqdm
from torch import nn
from torchvision import transforms
import time

def convert_square_pad(image):
    """
    convert to square with ConstantPad
    """
    width, height = image.size
    # Convert PIL.Image to FloatTensor, scale from 0 to 1, 0 = black, 1 = white
    img = transforms.ToTensor()(image)
    img = img.unsqueeze(0)  # 加轴
    pad_len = int(abs(width - height) / 2)  # 预填充区域的大小
    # 需要填充区域，如果宽大于高则上下填充，否则左右填充
    if width > height:
        fill_area = (0, 0, pad_len, pad_len)
    else:
        fill_area = (pad_len, pad_len, 0, 0)
    # 填充像素常值
    fill_value = 1
    img = nn.ConstantPad2d(fill_area, fill_value)(img)
    # img = nn.ZeroPad2d(m)(img) #直接填0
    img = img.squeeze(0)  # 去轴
    img = transforms.ToPILImage()(img)
    return img


def main(jpg_path, convert_path):

    for jpg_file in tqdm(os.listdir(jpg_path)):
        try:
            image = Image.open(os.path.join(jpg_path, jpg_file))
        except UnidentifiedImageError:
            print(jpg_file)
            continue
        image = image.convert('L')
        image = convert_square_pad(image)
        image = image.resize((128, 128), Image.ANTIALIAS)
        image.save(os.path.join(convert_path, jpg_file))
    return


if __name__ == '__main__':
    jpg_path = '../shufa_pic/shufa'
    square_path = '../shufa_pic/square_img'
    main(jpg_path, square_path)

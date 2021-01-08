
import os
import sys

import argparse
import json
import random
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont

from torchvision import transforms
from torch import nn


def chk_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fonts_path', type=str, required=True)
    parser.add_argument('--canvas_size', type=int, default=256)
    parser.add_argument('--char_size', type=int, default=256)
    parser.add_argument('--generate_cnt', type=int, default=5)
    parser.add_argument('--output_path', type=str, default='./fonts_output')
    args = parser.parse_args()
    return args


def draw_single_char(ch, font, canvas_size):
    img = Image.new("L", (canvas_size * 2, canvas_size * 2), 0)
    draw = ImageDraw.Draw(img)
    try:
        draw.text((10, 10), ch, 255, font=font)
    except OSError:
        return None
    bbox = img.getbbox()
    if bbox is None:
        return None
    l, u, r, d = bbox
    l = max(0, l - 5)
    u = max(0, u - 5)
    r = min(canvas_size * 2 - 1, r + 5)
    d = min(canvas_size * 2 - 1, d + 5)
    if l >= r or u >= d:
        return None
    img = np.array(img)
    img = img[u:d, l:r]
    img = 255 - img
    img = Image.fromarray(img)
    # img.show()
    width, height = img.size
    # Convert PIL.Image to FloatTensor, scale from 0 to 1, 0 = black, 1 = white
    try:
        img = transforms.ToTensor()(img)
    except SystemError:
        return None
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
    img = img.resize((canvas_size, canvas_size), Image.ANTIALIAS)
    return img


if __name__ == '__main__':
    args = parse_args()
    chk_mkdir(args.output_path)
    charset = json.load(open("./charset/cjk.json"))['gb2312_t']
    fonts_list = [f for f in os.listdir(args.fonts_path) if os.path.splitext(f)[-1].lower() in {'.otf', '.ttf', '.ttc'}]
    # fonts_map = {font: idx for idx, font in enumerate(fonts_list)}
    for font in fonts_list:
        font_output_dir = os.path.join(args.output_path, font)
        chk_mkdir(font_output_dir)
        cnt = 0
        font_if = ImageFont.truetype(os.path.join(args.fonts_path, font), size=args.char_size)
        while cnt < args.generate_cnt:
            char = random.choice(charset)
            try:
                char_img = draw_single_char(char, font_if, canvas_size=args.canvas_size)
            except ValueError as e:
                continue
            if char_img is None or np.array(char_img).mean() < 1:
                continue
            img_output_path = os.path.join(font_output_dir, str(cnt) + '.jpg')
            char_img.save(img_output_path)
            cnt += 1

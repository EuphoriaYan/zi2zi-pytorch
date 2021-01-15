# -*- coding: utf-8 -*-
import argparse
import sys
import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import json
import collections
import re
from tqdm import tqdm
import random
from fontTools.ttLib import TTFont
from torchvision import transforms
from torch import nn

from utils.charset_util import processGlyphNames


def draw_single_char(ch, font, canvas_size, x_offset=0, y_offset=0):
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


def draw_font2font_example(ch, src_font, dst_font, canvas_size, x_offset, y_offset):
    dst_img = draw_single_char(ch, dst_font, canvas_size, x_offset, y_offset)
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    if dst_img is None or src_img is None:
        return None
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    # convert to gray img
    example_img = example_img.convert('L')
    return example_img


parser = argparse.ArgumentParser()

parser.add_argument('--src_fonts_dir', type=str, default=None, help='path of the source fonts\' path')
parser.add_argument('--dst_json', type=str, default=None, help='path of the target fonts\' json info.')
parser.add_argument('--shuffle', default=False, action='store_true', help='shuffle a charset before processings')
parser.add_argument('--char_size', type=int, default=250, help='character size')
parser.add_argument('--canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--x_offset', type=int, default=0, help='x offset')
parser.add_argument('--y_offset', type=int, default=0, help='y_offset')
parser.add_argument('--sample_count', type=int, default=2333, help='number of characters to draw')
parser.add_argument('--sample_dir', type=str, default='sample_dir', help='directory to save examples')
parser.add_argument('--start_from', type=int, default=0, help='resume from idx')

args = parser.parse_args()


if __name__ == "__main__":
    if not os.path.isdir(args.sample_dir):
        os.mkdir(args.sample_dir)

    src_fonts_dir = args.src_fonts_dir
    fontPlane00 = TTFont(os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane00_2020051520200519101119.TTF'))
    fontPlane02 = TTFont(os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane02_2020051520200519101142.TTF'))

    charSetPlane00 = processGlyphNames(fontPlane00.getGlyphNames())
    charSetPlane02 = processGlyphNames(fontPlane02.getGlyphNames())

    charSetTotal = charSetPlane00 | charSetPlane02
    charListTotal = list(charSetTotal)

    fontPlane00 = ImageFont.truetype(
        os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane00_2020051520200519101119.TTF'), args.char_size)
    fontPlane02 = ImageFont.truetype(
        os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane02_2020051520200519101142.TTF'), args.char_size)

    dst_json = args.dst_json
    with open(dst_json, 'r', encoding='utf-8') as fp:
        dst_fonts = json.load(fp)

    font_label_map = dict()

    for idx, dst_font in tqdm(enumerate(dst_fonts)):
        font_name = dst_font['font_name']
        print(font_name + ': ' + str(idx))
        font_label_map[font_name] = idx

        if idx < args.start_from:
            continue

        font_path = dst_font['font_pth']
        font_missing = set(dst_font['missing'])
        font_fake = set(dst_font['fake'])
        dst_font = ImageFont.truetype(font_path, args.char_size)
        dst_font_chars = processGlyphNames(TTFont(font_path).getGlyphNames())
        if args.shuffle:
            np.random.shuffle(dst_font_chars)
        cur = 0
        for char in dst_font_chars:
            if cur == args.sample_count:
                break
            if char in font_missing or char in font_fake:
                continue
            else:
                if char in charSetPlane00:
                    img = draw_font2font_example(char, fontPlane00, dst_font, args.canvas_size, args.x_offset, args.y_offset)
                elif char in charSetPlane02:
                    img = draw_font2font_example(char, fontPlane02, dst_font, args.canvas_size, args.x_offset, args.y_offset)
                else:
                    continue
                if img is not None:
                    img.save(os.path.join(args.sample_dir, '%d_%04d.png' % (idx, cur)))
                    cur += 1

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


def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    except OSError:
        return None
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


def processGlyphNames(GlyphNames):
    res = set()
    for char in GlyphNames:
        if char.startswith('uni'):
            char = char.replace('uni', '\\u')
        elif char.startswith('uF'):
            char = char.replace('uF', '\\u')
        else:
            continue
        char_utf8 = char.encode('utf-8')
        try:
            char_escape = char_utf8.decode('unicode_escape')
        except UnicodeDecodeError:
            continue
        res.add(char_escape)
    return res



parser = argparse.ArgumentParser()

parser.add_argument('--src_fonts_dir', type=str, default=None, help='path of the source fonts\' path')
parser.add_argument('--dst_json', type=str, default=None, help='path of the target fonts\' json info.')
parser.add_argument('--shuffle', default=False, action='store_true', help='shuffle a charset before processings')
parser.add_argument('--char_size', type=int, default=250, help='character size')
parser.add_argument('--canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--x_offset', type=int, default=0, help='x offset')
parser.add_argument('--y_offset', type=int, default=0, help='y_offset')
parser.add_argument('--sample_count', type=int, default=5000, help='number of characters to draw')
parser.add_argument('--sample_dir', type=str, default='sample_dir', help='directory to save examples')
parser.add_argument('--start_from', type=int, default=0, help='resume from idx')

args = parser.parse_args()


if __name__ == "__main__":
    if not os.path.isdir(args.sample_dir):
        os.mkdir(args.sample_dir)

    src_fonts_dir = args.src_fonts_dir
    fontPlane00 = TTFont(os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane00_2020051520200519101119.TTF'))
    fontPlane02 = TTFont(os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane02_2020051520200519101142.TTF'))
    fontPlane15 = TTFont(os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane15_2020051520200519101206.TTF'))

    charSetPlane00 = processGlyphNames(fontPlane00.getGlyphNames())
    charSetPlane02 = processGlyphNames(fontPlane02.getGlyphNames())
    charSetPlane15 = processGlyphNames(fontPlane15.getGlyphNames())
    charSetTotal = charSetPlane00 | charSetPlane02 | charSetPlane15
    charListTotal = list(charSetTotal)

    fontPlane00 = ImageFont.truetype(
        os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane00_2020051520200519101119.TTF'), args.char_size)
    fontPlane02 = ImageFont.truetype(
        os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane02_2020051520200519101142.TTF'), args.char_size)
    fontPlane15 = ImageFont.truetype(
        os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane15_2020051520200519101206.TTF'), args.char_size)

    dst_json = args.dst_json
    with open(dst_json, 'r', encoding='utf-8') as fp:
        dst_fonts = json.load(fp)
    with open('charset/font_state.txt', 'w', encoding='utf-8') as fs:
        for i, font in enumerate(dst_fonts):
            fs.write(str(i))
            fs.write('\t')
            fs.write(font['font_name'])
            fs.write('\n')

    font_label_map = dict()

    for idx, dst_font in enumerate(dst_fonts):
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
                elif char in charSetPlane15:
                    img = draw_font2font_example(char, fontPlane15, dst_font, args.canvas_size, args.x_offset, args.y_offset)
                else:
                    continue
                if img is not None:
                    img.save(os.path.join(args.sample_dir, '%d_%05d.png' % (idx, cur)))
                    cur += 1
    with open('font_label_map.json', 'w', encoding='utf-8') as fp:
        json.dump(font_label_map, fp)

# -*- coding: utf-8 -*-
import argparse
import sys
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
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


def draw_fine_single_char(ch, font, canvas_size):
    img = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        draw.text((10, 10), ch, (255, 255, 255), font=font)
    except OSError:
        return None
    l, u, r, d = img.getbbox()
    img = np.ndarray(img)



parser = argparse.ArgumentParser()

parser.add_argument('--create_num', type=int, default=0, help='use which model')
parser.add_argument('--src_fonts_dir', type=str, default='charset/ZhongHuaSong', help='path of the src fonts')
parser.add_argument('--fonts_json', type=str, default=None, help='path of the target fonts\' json info.')
parser.add_argument('--char_size', type=int, default=250, help='character size')
parser.add_argument('--canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--x_offset', type=int, default=0, help='x offset')
parser.add_argument('--y_offset', type=int, default=0, help='y_offset')
parser.add_argument('--sample_count', type=int, default=20000, help='number of characters to draw')
parser.add_argument('--sample_dir', type=str, default='sample_dir', help='directory to save samples')
parser.add_argument('--charset_path', type=str, default='charset/charset_l.txt', help='path of charset file')
parser.add_argument('--bad_fonts', type=str, default='charset/error_font.txt', help='path of bad font list file')

args = parser.parse_args()


if __name__ == "__main__":
    if not os.path.isdir(args.sample_dir):
        os.mkdir(args.sample_dir)
    # Create bad_fonts_list
    with open(args.bad_fonts, 'r', encoding='utf-8') as bd_fs:
        bd_fs_lines = bd_fs.readlines()
    if args.create_num > 4:
        raise ValueError
    bd_fs_list = [int(num) for num in bd_fs_lines[args.create_num].strip().split()]

    dst_json = args.dst_json
    with open(dst_json, 'r', encoding='utf-8') as fp:
        dst_fonts = json.load(fp)
    samples_count = collections.defaultdict(int)

    # Get start_num, font_cnt
    if args.create_num == 0:
        start_num = 0
        font_cnt = 201
    else:
        start_num = 200 * args.create_num + 1
        if args.create_num == 4:
            font_cnt = 165
        else:
            font_cnt = 200

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

    for idx, dst_font in enumerate(dst_fonts):
        font_name = dst_font['font_name']
        print(font_name + ': ' + str(idx))

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

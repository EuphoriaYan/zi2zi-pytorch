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
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    return img


def draw_font2font_example(ch, src_font, dst_font, canvas_size, x_offset, y_offset):
    dst_img = draw_single_char(ch, dst_font, canvas_size, x_offset, y_offset)
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    # convert to gray img
    example_img = example_img.convert('L')
    return example_img


def draw_font2imgs_example(ch, src_font, dst_img, canvas_size, x_offset, y_offset):
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    dst_img = dst_img.resize((canvas_size, canvas_size), Image.ANTIALIAS).convert('RGB')
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    # convert to gray img
    example_img = example_img.convert('L')
    return example_img


def draw_imgs2imgs_example(src_img, dst_img, canvas_size):
    src_img = src_img.resize((canvas_size, canvas_size), Image.ANTIALIAS).convert('RGB')
    dst_img = dst_img.resize((canvas_size, canvas_size), Image.ANTIALIAS).convert('RGB')
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    # convert to gray img
    example_img = example_img.convert('L')
    return example_img


def filter_recurring_hash(charset, font, canvas_size, x_offset, y_offset):
    """ Some characters are missing in a given font, filter them
    by checking the recurring hashes
    """
    _charset = charset.copy()
    np.random.shuffle(_charset)
    sample = _charset[:2000]
    hash_count = collections.defaultdict(int)
    for c in sample:
        img = draw_single_char(c, font, canvas_size, x_offset, y_offset)
        hash_count[hash(img.tobytes())] += 1
    recurring_hashes = filter(lambda d: d[1] > 2, hash_count.items())
    return [rh[0] for rh in recurring_hashes]


def font2font(src, dst, charset, char_size, canvas_size,
             x_offset, y_offset, sample_count, sample_dir, label=0, filter_by_hash=True):
    src_font = ImageFont.truetype(src, size=char_size)
    dst_font = ImageFont.truetype(dst, size=char_size)

    filter_hashes = set()
    if filter_by_hash:
        filter_hashes = set(filter_recurring_hash(charset, dst_font, canvas_size, x_offset, y_offset))
        print("filter hashes -> %s" % (",".join([str(h) for h in filter_hashes])))

    count = 0

    for c in charset:
        if count == sample_count:
            break
        e = draw_font2font_example(c, src_font, dst_font, canvas_size, x_offset, y_offset, filter_hashes)
        if e:
            e.save(os.path.join(sample_dir, "%d_%04d.jpg" % (label, count)))
            count += 1
            if count % 500 == 0:
                print("processed %d chars" % count)


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
parser.add_argument('--char_size', type=int, default=256, help='character size')
parser.add_argument('--canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--x_offset', type=int, default=0, help='x offset')
parser.add_argument('--y_offset', type=int, default=0, help='y_offset')
parser.add_argument('--sample_count', type=int, default=20000, help='number of characters to draw')
parser.add_argument('--sample_dir', type=str, default='sample_dir', help='directory to save examples')

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

    fontPlane00 = ImageFont.truetype(os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane00_2020051520200519101119.TTF'), args.char_size)
    fontPlane02 = ImageFont.truetype(os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane02_2020051520200519101142.TTF'), args.char_size)
    fontPlane15 = ImageFont.truetype(os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane15_2020051520200519101206.TTF'), args.char_size)

    dst_json = args.dst_json
    with open(dst_json, 'r', encoding='utf-8') as fp:
        dst_fonts = json.load(fp)

    font_label_map = dict()

    for idx, dst_font in enumerate(dst_fonts):
        if args.shuffle:
            np.random.shuffle(charListTotal)
        font_name = dst_font['font_name']
        font_label_map[font_name] = idx
        font_path = dst_font['font_pth']
        font_missing = set(dst_font['missing'])
        font_fake = set(dst_font['fake'])
        dst_font = ImageFont.truetype(font_path, args.char_size)
        cur = 0
        for char in charListTotal:
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
                    raise ValueError
                img.save(os.path.join(args.sample_dir, '%d_%05d' % (idx, cur)))
                cur += 1


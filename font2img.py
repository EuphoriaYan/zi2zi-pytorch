# -*- coding: utf-8 -*-

import os
import sys

import argparse
import numpy as np

from PIL import Image, ImageFont, ImageDraw
import json
import collections
import re
from fontTools.ttLib import TTFont
from tqdm import tqdm
import random

from torch import nn
from torchvision import transforms

from utils.charset_util import processGlyphNames

CN_CHARSET = None
CN_T_CHARSET = None
JP_CHARSET = None
KR_CHARSET = None

DEFAULT_CHARSET = "./charset/cjk.json"


def load_global_charset():
    global CN_CHARSET, JP_CHARSET, KR_CHARSET, CN_T_CHARSET
    cjk = json.load(open(DEFAULT_CHARSET))
    CN_CHARSET = cjk["gbk"]
    JP_CHARSET = cjk["jp"]
    KR_CHARSET = cjk["kr"]
    CN_T_CHARSET = cjk["gb2312_t"]


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


def draw_font2font_example(ch, src_font, dst_font, canvas_size, x_offset, y_offset, filter_hashes):
    dst_img = draw_single_char(ch, dst_font, canvas_size, x_offset, y_offset)
    # check the filter example in the hashes or not
    dst_hash = hash(dst_img.tobytes())
    if dst_hash in filter_hashes:
        return None
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


def font2imgs(src, dst, char_size, canvas_size,
              x_offset, y_offset, sample_count, sample_dir):
    src_font = ImageFont.truetype(src, size=char_size)

    # -*- You should fill the target imgs' label_map -*-
    writer_dict = {
        '智永': 0, ' 隸書-趙之謙': 1, '張即之': 2, '張猛龍碑': 3, '柳公權': 4, '標楷體-手寫': 5, '歐陽詢-九成宮': 6,
        '歐陽詢-皇甫誕': 7, '沈尹默': 8, '美工-崩雲體': 9, '美工-瘦顏體': 10, '虞世南': 11, '行書-傅山': 12, '行書-王壯為': 13,
        '行書-王鐸': 14, '行書-米芾': 15, '行書-趙孟頫': 16, '行書-鄭板橋': 17, '行書-集字聖教序': 18, '褚遂良': 19, '趙之謙': 20,
        '趙孟頫三門記體': 21, '隸書-伊秉綬': 22, '隸書-何紹基': 23, '隸書-鄧石如': 24, '隸書-金農': 25,  '顏真卿-顏勤禮碑': 26,
        '顏真卿多寶塔體': 27, '魏碑': 28
    }
    count = 0

    # -*- You should fill the target imgs' regular expressions. -*-
    pattern = re.compile('(.)~(.+)~(\d+)')

    for c in tqdm(os.listdir(dst)):
        if count == sample_count:
            break
        res = re.match(pattern, c)
        ch = res[1]
        writter = res[2]
        label = writer_dict[writter]
        img_path = os.path.join(dst, c)
        dst_img = Image.open(img_path)
        e = draw_font2imgs_example(ch, src_font, dst_img, canvas_size, x_offset, y_offset)
        if e:
            e.save(os.path.join(sample_dir, "%d_%04d.jpg" % (label, count)))
            count += 1


def fonts2imgs(src_fonts_dir, dst, char_size, canvas_size,
               x_offset, y_offset, sample_count, sample_dir):
    fontPlane00 = TTFont(os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane00_2020051520200519101119.TTF'))
    fontPlane02 = TTFont(os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane02_2020051520200519101142.TTF'))

    charSetPlane00 = processGlyphNames(fontPlane00.getGlyphNames())
    charSetPlane02 = processGlyphNames(fontPlane02.getGlyphNames())

    charSetTotal = charSetPlane00 | charSetPlane02
    charListTotal = list(charSetTotal)

    fontPlane00 = ImageFont.truetype(
        os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane00_2020051520200519101119.TTF'), char_size)
    fontPlane02 = ImageFont.truetype(
        os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane02_2020051520200519101142.TTF'), char_size)

    # -*- You should fill the target imgs' label_map -*-
    writer_dict = {
        '智永': 0, ' 隸書-趙之謙': 1, '張即之': 2, '張猛龍碑': 3, '柳公權': 4, '標楷體-手寫': 5, '歐陽詢-九成宮': 6,
        '歐陽詢-皇甫誕': 7, '沈尹默': 8, '美工-崩雲體': 9, '美工-瘦顏體': 10, '虞世南': 11, '行書-傅山': 12, '行書-王壯為': 13,
        '行書-王鐸': 14, '行書-米芾': 15, '行書-趙孟頫': 16, '行書-鄭板橋': 17, '行書-集字聖教序': 18, '褚遂良': 19, '趙之謙': 20,
        '趙孟頫三門記體': 21, '隸書-伊秉綬': 22, '隸書-何紹基': 23, '隸書-鄧石如': 24, '隸書-金農': 25,  '顏真卿-顏勤禮碑': 26,
        '顏真卿多寶塔體': 27, '魏碑': 28
    }
    count = 0

    # -*- You should fill the target imgs' regular expressions. -*-
    pattern = re.compile('(.)~(.+)~(\d+)')

    for c in tqdm(os.listdir(dst)):
        if count == sample_count:
            break
        res = re.match(pattern, c)
        ch = res[1]
        writter = res[2]
        label = writer_dict[writter]
        img_path = os.path.join(dst, c)
        dst_img = Image.open(img_path)
        if ch in charSetPlane00:
            e = draw_font2imgs_example(ch, fontPlane00, dst_img, canvas_size, x_offset, y_offset)
        elif ch in charSetPlane02:
            e = draw_font2imgs_example(ch, fontPlane02, dst_img, canvas_size, x_offset, y_offset)
        else:
            e = None
        if e:
            e.save(os.path.join(sample_dir, "%d_%04d.jpg" % (label, count)))
            count += 1


def imgs2imgs(src, dst, canvas_size, sample_count, sample_dir):

    # -*- You should fill the target imgs' label_map -*-
    label_map = {
        '1号字体': 0,
        '2号字体': 1,
    }
    count = 0
    # For example
    # source images are 南~0号字体1，南~0号字体2，京~0号字体，市~0号字体，长~0号字体，江~0号字体，大~0号字体，桥~0号字体
    # make sure all the source images are same font, or at least, very close fonts.
    # target images are 南~1号字体，京~1号字体，市~1号字体，长~2号字体

    # -*- You should fill the source/target imgs' regular expressions. -*-

    # We only need character in source img.
    source_pattern = re.compile('(.)~0号字体')
    # We need character and label in target img.
    target_pattern = re.compile('(.)~(/d)号字体')

    # Multi-imgs with a same character in src_imgs are allowed.
    # Use default_dict(list) to storage.
    source_ch_list = collections.defaultdict(list)
    for c in tqdm(os.listdir(src)):
        res = re.match(source_pattern, c)
        ch = res[1]
        source_ch_list[ch].append(c)

    def get_source_img(ch):
        res = source_ch_list.get(ch)
        if res is None or len(res) == 0:
            return None
        if len(res) == 1:
            return res[0]
        idx = random.randint(0, len(res))
        return res[idx]

    for c in tqdm(os.listdir(dst)):
        if count == sample_count:
            break
        res = re.match(target_pattern, c)
        ch = res[1]
        label = label_map[res[2]]
        src_img_name = get_source_img(ch)
        if src_img_name is None:
            continue
        img_path = os.path.join(src, src_img_name)
        src_img = Image.open(img_path)
        img_path = os.path.join(dst, c)
        dst_img = Image.open(img_path)
        e = draw_imgs2imgs_example(src_img, dst_img, canvas_size)
        if e:
            e.save(os.path.join(sample_dir, "%d_%04d.jpg" % (label, count)))
            count += 1


load_global_charset()
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['imgs2imgs', 'font2imgs', 'font2font', 'fonts2imgs'], required=True,
                    help='generate mode.\n'
                         'use --src_imgs and --dst_imgs for imgs2imgs mode.\n'
                         'use --src_font and --dst_imgs for font2imgs mode.\n'
                         'use --src_font and --dst_font for font2font mode.\n'
                         'use --src_fonts_dir and --dst_imgs for fonts2imgs mode.\n'
                         'No imgs2font mode.'
                    )
parser.add_argument('--src_font', type=str, default=None, help='path of the source font')
parser.add_argument('--src_fonts_dir', type=str, default=None, help='path of the source fonts')
parser.add_argument('--src_imgs', type=str, default=None, help='path of the source imgs')
parser.add_argument('--dst_font', type=str, default=None, help='path of the target font')
parser.add_argument('--dst_imgs', type=str, default=None, help='path of the target imgs')

parser.add_argument('--filter', default=False, action='store_true', help='filter recurring characters')
parser.add_argument('--charset', type=str, default='CN',
                    help='charset, can be either: CN, JP, KR or a one line file. ONLY VALID IN font2font mode.')
parser.add_argument('--shuffle', default=False, action='store_true', help='shuffle a charset before processings')
parser.add_argument('--char_size', type=int, default=256, help='character size')
parser.add_argument('--canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--x_offset', type=int, default=0, help='x offset')
parser.add_argument('--y_offset', type=int, default=0, help='y_offset')
parser.add_argument('--sample_count', type=int, default=5000, help='number of characters to draw')
parser.add_argument('--sample_dir', type=str, default='sample_dir', help='directory to save examples')
parser.add_argument('--label', type=int, default=0, help='label as the prefix of examples')

args = parser.parse_args()

if __name__ == "__main__":
    if not os.path.isdir(args.sample_dir):
        os.mkdir(args.sample_dir)
    if args.mode == 'font2font':
        if args.src_font is None or args.dst_font is None:
            raise ValueError('src_font and dst_font are required.')
        if args.charset in ['CN', 'JP', 'KR', 'CN_T']:
            charset = locals().get("%s_CHARSET" % args.charset)
        else:
            charset = list(open(args.charset, encoding='utf-8').readline().strip())
        if args.shuffle:
            np.random.shuffle(charset)
        font2font(args.src_font, args.dst_font, charset, args.char_size,
                  args.canvas_size, args.x_offset, args.y_offset,
                  args.sample_count, args.sample_dir, args.label, args.filter)
    elif args.mode == 'font2imgs':
        if args.src_font is None or args.dst_imgs is None:
            raise ValueError('src_font and dst_imgs are required.')
        font2imgs(args.src_font, args.dst_imgs, args.char_size,
                  args.canvas_size, args.x_offset, args.y_offset,
                  args.sample_count, args.sample_dir)
    elif args.mode == 'fonts2imgs':
        if args.src_fonts_dir is None or args.dst_imgs is None:
            raise ValueError('src_font and dst_imgs are required.')
        fonts2imgs(args.src_fonts_dir, args.dst_imgs, args.char_size,
                  args.canvas_size, args.x_offset, args.y_offset,
                  args.sample_count, args.sample_dir)
    elif args.mode == 'imgs2imgs':
        if args.src_imgs is None or args.dst_imgs is None:
            raise ValueError('src_imgs and dst_imgs are required.')
        imgs2imgs(args.src_imgs, args.dst_imgs, args.canvas_size, args.sample_count, args.sample_dir)
    else:
        raise ValueError('mode should be font2font, font2imgs or imgs2imgs')

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms
from torch import nn
import random
from natsort import natsorted
import re
from pprint import pprint
import os
from fontTools.ttLib import TTFont
import json


def get_fonts():
    # dst_json = 'experiment/font_missing.json'
    dst_json = '/disks/sdb/projs/AncientBooks/data/chars/font_missing.json'
    with open(dst_json, 'r', encoding='utf-8') as fp:
        dst_fonts = json.load(fp)
    return dst_fonts


if __name__ == '__main__':
    fonts = get_fonts()
    fonts2idx = {os.path.splitext(font['font_name'])[0]: idx for idx, font in enumerate(fonts)}

    type_fonts = 'type/行草行楷手写类.txt'
    with open(type_fonts, 'r', encoding='utf-8') as fp:
        type_fonts = [font_line.strip() for font_line in fp]
    type_fonts_rev = {v: k for k, v in enumerate(type_fonts)}

    new_jsons = []
    for font_name in type_fonts:
        sp_json = fonts[fonts2idx[font_name]]
        json_str = json.dumps(sp_json)
        new_jsons.append(json_str)

    with open('type/caokai_font_missing.json', 'w', encoding='utf-8') as fp:
        for json_str in new_jsons:
            fp.write(json_str + '\n')

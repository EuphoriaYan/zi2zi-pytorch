from PIL import Image, UnidentifiedImageError
import numpy as np
import os
from tqdm import tqdm
from torch import nn
from torchvision import transforms
import collections
from collections import defaultdict, OrderedDict
import time
import re

import collections


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or  #
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


def get_shufu_list():
    jpg_path = '../shufa_pic/square_img'
    pattern = re.compile('(.)~(.+)~(\d+).jpg')
    examples = []
    label_set = defaultdict(int)
    for jpg_file in tqdm(os.listdir(jpg_path)):
        res = re.match(pattern, jpg_file)
        if res is None:
            raise ValueError(jpg_file)
        label = res[1]
        label_set[label] += 1
        guid = res[3]
    label_list = [(v, k) for k, v in label_set.items()]
    label_list.sort(reverse=True)
    label_set = OrderedDict()
    for cnt, wd in label_list:
        label_set[wd] = None
    return label_set


def get_bert_list():
    vocab_path = '../data/vocab_bert.txt'
    label_set = OrderedDict()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            if len(l) > 1 or len(l) == 0:
                continue
            try:
                l_uni = ord(l)
            except TypeError:
                print(l)
            if is_chinese_char(l_uni):
                label_set[l] = None
    return label_set


if __name__ == '__main__':
    shufa_dict = get_shufu_list()
    bert_dict = get_bert_list()
    shufa_set = set(shufa_dict.keys())

    final_list = list(shufa_dict.keys())
    for k, v in bert_dict.items():
        if k not in shufa_set:
            shufa_set.add(k)
            final_list.append(k)

    with open('../data/vocab.txt', 'w', encoding='utf-8') as vocab:
        for i in final_list:
            vocab.write(i + '\n')
    print(len(final_list))

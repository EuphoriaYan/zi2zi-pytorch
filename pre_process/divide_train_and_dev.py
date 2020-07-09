import re
import os
from tqdm import tqdm
from collections import defaultdict
from math import ceil
import random

if __name__ == '__main__':
    jpg_path = '../shufa_pic/square_img'
    dev_path = '../shufa_pic/square_img_dev'
    pattern = re.compile('(.)~(.+)~(\d+).jpg')
    jpg_dict = defaultdict(list)
    for jpg_file in tqdm(os.listdir(jpg_path)):
        res = re.match(pattern, jpg_file)
        jpg_dict[res[1]].append(res[0])
    random.seed(777)
    for k, v in tqdm(jpg_dict.items()):
        if len(v) < 10:
            continue
        dev_num = ceil(len(v) / 20)
        dev_list = random.sample(v, dev_num)
        for dev_jpg in dev_list:
            src = os.path.join(jpg_path, dev_jpg)
            trg = os.path.join(dev_path, dev_jpg)
            os.rename(src, trg)

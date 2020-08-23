# -*- coding: utf-8 -*-
import argparse
import glob
import os
import pickle
import random
from tqdm import tqdm
import re


def pickle_examples_with_split_ratio(paths, train_path, val_path, train_val_split=0.1):
    """
    Compile a list of examples into pickled format, so during
    the training, all io will happen in memory
    """
    with open(train_path, 'wb') as ft:
        with open(val_path, 'wb') as fv:
            for p in tqdm(paths):
                label = int(os.path.basename(p).split("_")[0])
                with open(p, 'rb') as f:
                    img_bytes = f.read()
                    r = random.random()
                    example = (label, img_bytes)
                    if r < train_val_split:
                        pickle.dump(example, fv)
                    else:
                        pickle.dump(example, ft)


def pickle_examples_with_file_name(paths, obj_path):
    with open(obj_path, 'wb') as fa:
        for p in tqdm(paths):
            label = int(os.path.basename(p).split("_")[0])
            with open(p, 'rb') as f:
                img_bytes = f.read()
                example = (label, img_bytes)
                pickle.dump(example, fa)


parser = argparse.ArgumentParser(description='Compile list of images into a pickled object for training')
parser.add_argument('--dir', required=True, help='path of examples')
parser.add_argument('--save_dir', required=True, help='path to save pickled files')
parser.add_argument('--split_ratio', type=float, default=0.1, dest='split_ratio',
                    help='split ratio between train and val')
parser.add_argument('--start_num', type=int, default=None)
parser.add_argument('--end_num', type=int, default=None)
parser.add_argument('--save_obj_dir', type=str, default=None)
args = parser.parse_args()

if __name__ == "__main__":
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    train_path = os.path.join(args.save_dir, "train.obj")
    val_path = os.path.join(args.save_dir, "val.obj")
    total_file_list = sorted(glob.glob(os.path.join(args.dir, "*.jpg")) + glob.glob(os.path.join(args.dir, "*.png")))
    # '%d_%05d.png'
    cur_file_list = []
    for file_name in total_file_list:
        label = os.path.basename(file_name).split('_')[0]
        if label is None:
            continue
        label = int(label)
        if args.start_num is not None and label < args.start_num:
            continue
        if args.end_num is not None and label > args.end_num:
            continue
        cur_file_list.append(file_name)

    if args.split_ratio == 0 and args.save_obj_dir is not None:
        pickle_examples_with_file_name(cur_file_list, args.save_obj_dir)
    else:
        pickle_examples_with_split_ratio(
            cur_file_list,
            train_path=train_path,
            val_path=val_path,
            train_val_split=args.split_ratio
        )

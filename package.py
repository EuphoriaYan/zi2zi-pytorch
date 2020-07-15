# -*- coding: utf-8 -*-
import argparse
import glob
import os
import pickle
import random
from tqdm import tqdm


def pickle_examples(paths, train_path, val_path, train_val_split=0.1):
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


parser = argparse.ArgumentParser(description='Compile list of images into a pickled object for training')
parser.add_argument('--dir', required=True, help='path of examples')
parser.add_argument('--save_dir', required=True, help='path to save pickled files')
parser.add_argument('--split_ratio', type=float, default=0.1, dest='split_ratio',
                    help='split ratio between train and val')
args = parser.parse_args()

if __name__ == "__main__":
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    train_path = os.path.join(args.save_dir, "train.obj")
    val_path = os.path.join(args.save_dir, "val.obj")
    pickle_examples(sorted(glob.glob(os.path.join(args.dir, "*.jpg"))), train_path=train_path, val_path=val_path,
                    train_val_split=args.split_ratio)

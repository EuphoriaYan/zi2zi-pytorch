
import os
import sys

import argparse
import time
from PIL import Image

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

from model.discriminators import Discriminator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', required=True,
                        help='experiment directory, data, samples,checkpoints,etc')
    parser.add_argument('--image_size', type=int, default=256, help="size of your input images")
    parser.add_argument('--embedding_num', type=int, default=40,
                        help="number for distinct embeddings")
    parser.add_argument('--batch_size', type=int, default=16, help='number of examples in batch')
    parser.add_argument('--resume', type=int, required=True, help='resume from previous training')
    parser.add_argument('--input_nc', type=int, default=1)
    parser.add_argument('--type_file', type=str, default='type/宋黑类字符集.txt')

    parser.add_argument('--input_path', type=str, required=True)

    args = parser.parse_args()
    return args


def load_val_dataloader(args):
    IMG_EXT = {'.jpg', '.png', '.tif', '.tiff'}
    raw_img_list = []
    for root, dirs, files in os.walk(args.input_path):
        for file in files:
            if os.path.splitext(file)[-1].lower() in IMG_EXT:
                raw_img_list.append(os.path.join(root, file))
    img_list = [transforms.Normalize(0.5, 0.5)(
        transforms.ToTensor()(
            Image.open(img).convert('L').resize((args.image_size, args.image_size), Image.BICUBIC)
        )
    ).unsqueeze(dim=0) for img in raw_img_list]
    img_list = torch.cat(img_list, dim=0)
    img_list = torch.repeat_interleave(img_list, repeats=2, dim=1)
    img_dataset = TensorDataset(img_list)
    img_dataloader = DataLoader(img_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    return raw_img_list, img_dataloader


if __name__ == '__main__':
    args = parse_args()
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoint")

    font_names = [s.strip() for s in open(args.type_file, encoding='utf-8').readlines()]
    font_map = {idx: font for idx, font in enumerate(font_names)}

    model = Discriminator(input_nc=2 * args.input_nc, embedding_num=args.embedding_num)
    model_ckpt = torch.load(os.path.join(checkpoint_dir, '{}_net_D.pth'.format(args.resume)))
    model.load_state_dict(model_ckpt)
    print('load model {}'.format(args.resume))

    model.to('cuda')

    raw_image_list, img_dataloader = load_val_dataloader(args)

    t0 = time.time()

    total_category = []

    for batch in img_dataloader:
        img = batch[0]
        img = img.to('cuda')
        _, catagory_logits = model(img)
        catagory_logits = catagory_logits.detach().cpu()
        catagory_idx = torch.argmax(catagory_logits, dim=-1)
        catagory_idx = catagory_idx.numpy().tolist()
        total_category.extend(catagory_idx)

    for img, catagory_idx in zip(raw_image_list, total_category):
        print('{}: {}'.format(img, font_map[catagory_idx]))




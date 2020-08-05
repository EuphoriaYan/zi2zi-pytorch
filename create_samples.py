
from torch.utils.data import DataLoader, TensorDataset
from model import Zi2ZiModel
import os
import sys
import argparse
import torch
import random
import time
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import time
from tqdm import tqdm

writer_dict = {
        '智永': 0, ' 隸書-趙之謙': 1, '張即之': 2, '張猛龍碑': 3, '柳公權': 4, '標楷體-手寫': 5, '歐陽詢-九成宮': 6,
        '歐陽詢-皇甫誕': 7, '沈尹默': 8, '美工-崩雲體': 9, '美工-瘦顏體': 10, '虞世南': 11, '行書-傅山': 12, '行書-王壯為': 13,
        '行書-王鐸': 14, '行書-米芾': 15, '行書-趙孟頫': 16, '行書-鄭板橋': 17, '行書-集字聖教序': 18, '褚遂良': 19, '趙之謙': 20,
        '趙孟頫三門記體': 21, '隸書-伊秉綬': 22, '隸書-何紹基': 23, '隸書-鄧石如': 24, '隸書-金農': 25,  '顏真卿-顏勤禮碑': 26,
        '顏真卿多寶塔體': 27, '魏碑': 28
    }


parser = argparse.ArgumentParser(description='Infer')
parser.add_argument('--experiment_dir', required=True,
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--gpu_ids', default=[], nargs='+', help="GPUs")
parser.add_argument('--image_size', type=int, default=256,
                    help="size of your input and output image")
parser.add_argument('--L1_penalty', type=int, default=100, help='weight for L1 loss')
parser.add_argument('--Lconst_penalty', type=int, default=15, help='weight for const loss')
# parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
parser.add_argument('--Lcategory_penalty', type=float, default=1.0,
                    help='weight for category loss')
parser.add_argument('--embedding_num', type=int, default=40,
                    help="number for distinct embeddings")
parser.add_argument('--embedding_dim', type=int, default=128, help="dimension for embedding")
parser.add_argument('--batch_size', type=int, default=1, help='number of examples in batch')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--resume', type=int, default=140000, help='resume from previous training')
parser.add_argument('--input_nc', type=int, default=1)

parser.add_argument('--charset', type=str, choices=['s', 'm', 'l'], default='l')
parser.add_argument('--canvas_size', type=int, default=256)
parser.add_argument('--char_size', type=int, default=256)
parser.add_argument('--src_font', type=str, default='charset/gbk/方正新楷体_GBK(完整).TTF')
parser.add_argument('--label', type=int, default=0)


def draw_single_char(ch, font, canvas_size):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), ch, (0, 0, 0), font=font)
    img = img.convert('L')
    return img


def get_charset(charset_size):
    if charset_size == 's':
        with open('charset/charset_s.txt', 'r', encoding='utf-8') as charset_s_file:
            charset_txt = charset_s_file.readlines()
            charset = [s.strip() for s in charset_txt]
    else:
        charset_csv = pd.read_csv('charset/all_abooks.unigrams_desc.Clean.rate.csv')
        if charset_size == 'm':
            charset = charset_csv[['char']][charset_csv['acc_rate'] <= 0.999].values.squeeze(axis=-1).tolist()
        elif charset_size == 'l':
            charset = charset_csv[['char']][charset_csv['acc_rate'] <= 0.9999].values.squeeze(axis=-1).tolist()
        else:
            raise ValueError('charset_size should be s, m or l.')
        return ''.join(charset)


def main():
    args = parser.parse_args()
    data_dir = os.path.join(args.experiment_dir, "data")
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoint")
    sample_dir = os.path.join(args.experiment_dir, "sample")
    infer_dir = os.path.join(args.experiment_dir, "infer")

    # train_dataset = DatasetFromObj(os.path.join(data_dir, 'train.obj'), augment=True, bold=True, rotate=True, blur=True)
    # val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'))
    # dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


    model = Zi2ZiModel(
        input_nc=args.input_nc,
        embedding_num=args.embedding_num,
        embedding_dim=args.embedding_dim,
        Lconst_penalty=args.Lconst_penalty,
        Lcategory_penalty=args.Lcategory_penalty,
        save_dir=checkpoint_dir,
        gpu_ids=args.gpu_ids,
        is_training=False
    )
    model.setup()
    model.print_networks(True)
    model.load_networks(args.resume)
    sys.stdout.flush()

    src = get_charset(args.charset)
    font = ImageFont.truetype(args.src_font, size=args.char_size)
    img_list = [transforms.Normalize(0.5, 0.5)(
        transforms.ToTensor()(
            draw_single_char(ch, font, args.canvas_size)
        )
    ).unsqueeze(dim=0) for ch in src]
    label_list = [args.label for _ in src]

    img_list = torch.cat(img_list, dim=0)
    label_list = torch.tensor(label_list)

    dataset = TensorDataset(label_list, img_list, img_list)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    global writer_dict
    writer_dict_inv = {v: k for k, v in writer_dict.items()}

    for label_idx in range(29):
        dir_path = os.path.join(infer_dir, writer_dict_inv[label_idx])
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

    for label_idx in range(29):
        writer = writer_dict_inv[label_idx]
        for i, batch in enumerate(dataloader):
            model.set_input(torch.ones_like(batch[0]) * label_idx, batch[2], batch[1])
            model.forward()
            tensor_to_plot = model.fake_B.detach()
            tensor_to_plot = tensor_to_plot.squeeze(0)
            tensor_to_plot = tensor_to_plot.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).squeeze(-1)
            tensor_to_plot = tensor_to_plot.to('cpu', torch.uint8)
            img_ndarray = tensor_to_plot.numpy()
            im = Image.fromarray(img_ndarray, mode='L')
            im.save(os.path.join(infer_dir, writer, src[i] + '.png'))
        print('writer: ' + writer + ' complete.')


if __name__ == '__main__':
    with torch.no_grad():
        main()


import os
import sys

import argparse
import time
from PIL import Image

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from model.discriminators import Discriminator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['train', 'eval'])

    parser.add_argument('--image_size', type=int, default=256, help="size of your input images")
    parser.add_argument('--embedding_num', type=int, default=40,
                        help="number for distinct embeddings")
    parser.add_argument('--batch_size', type=int, default=16, help='number of examples in batch')
    parser.add_argument('--input_nc', type=int, default=1)
    parser.add_argument('--type_file', type=str, default='type/宋黑类字符集.txt')

    # train
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='output')

    # eval
    parser.add_argument('--ckpt_path', type=str, default='output/category.pth')
    parser.add_argument('--input_path', type=str, required=True)

    args = parser.parse_args()
    return args


def collate_fn(img_path):
    img = Image.open(img_path).convert('L').resize((args.image_size, args.image_size), Image.BICUBIC)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(0.5, 0.5)(img)
    img = img.unsqueeze(dim=0)
    return img


def load_val_dataloader(args):
    IMG_EXT = {'.jpg', '.png', '.tif', '.tiff'}
    raw_img_list = []
    for root, dirs, files in os.walk(args.input_path):
        for file in files:
            if os.path.splitext(file)[-1].lower() in IMG_EXT:
                raw_img_list.append(os.path.join(root, file))
    img_list = [collate_fn(img_path) for img_path in raw_img_list]
    img_list = torch.cat(img_list, dim=0)
    # img_list = torch.repeat_interleave(img_list, repeats=2, dim=1)
    img_dataset = TensorDataset(img_list)
    img_dataloader = DataLoader(img_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    return raw_img_list, img_dataloader


def eval(args):
    font_names = [s.strip() for s in open(args.type_file, encoding='utf-8').readlines()]
    font_map = {idx: font for idx, font in enumerate(font_names)}

    model = Discriminator(input_nc=args.input_nc, embedding_num=args.embedding_num)
    model_ckpt = torch.load(os.path.join(args.ckpt_path))
    model.load_state_dict(model_ckpt)
    print('load model {}'.format(args.resume))

    model.to('cuda')

    raw_image_list, img_dataloader = load_val_dataloader(args)

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



def load_train_dataloader(args, inv_font_map):
    IMG_EXT = {'.jpg', '.png', '.tif', '.tiff'}
    raw_img_list = []
    raw_label_list = []
    for root, dirs, files in os.walk(args.input_path):
        for file in files:
            if os.path.splitext(file)[-1].lower() in IMG_EXT:
                raw_img_list.append(os.path.join(root, file))
                raw_label_list.append(os.path.split(root)[-1])

    img_list = [collate_fn(img_path) for img_path in raw_img_list]
    label_list = [inv_font_map[font_name] for font_name in raw_label_list]

    img_train, img_val, label_train, label_val = train_test_split(img_list, label_list, test_size=0.1, random_state=777)

    img_train = torch.cat(img_train, dim=0)
    label_train = torch.LongTensor(label_train)
    train_dataset = TensorDataset(img_train, label_train)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    img_val = torch.cat(img_val, dim=0)
    label_val = torch.LongTensor(label_val)
    val_dataset = TensorDataset(img_val, label_val)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    return train_dataloader, val_dataloader


def train(args):

    font_names = [s.strip() for s in open(args.type_file, encoding='utf-8').readlines()]
    font_map = {idx: font for idx, font in enumerate(font_names)}
    inv_font_map = {font: idx for idx, font in enumerate(font_names)}

    model = Discriminator(input_nc=args.input_nc, embedding_num=args.embedding_num)

    if args.resume is not None:
        model_ckpt = torch.load(args.resume)
        model_ckpt.pop('model.0.weight')
        try:
            model.load_state_dict(model_ckpt, strict=False)
        except RuntimeError:
            print('Guess resume ckpt and your model have different catagories. Try pop catagory parameters.')
            model_ckpt.pop('catagory.weight')
            model_ckpt.pop('catagory.bias')
            model.load_state_dict(model_ckpt, strict=False)
        print('load model {}'.format(args.resume))

    model.to('cuda')
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    train_dataloader, val_dataloader = load_train_dataloader(args, inv_font_map)

    for epoch_idx in range(args.epoch):
        for batch_idx, batch in enumerate(train_dataloader):
            img, label = batch
            img = img.to('cuda')
            label = label.to('cuda')
            _, catagory_logits = model(img)
            loss = nn.CrossEntropyLoss()(catagory_logits, label)
            loss.backward()
            optimizer.step()
            # print('Epoch: {}, Batch: {}, Loss:{:.4f}'.format(epoch_idx, batch_idx, loss.item()))
        with torch.no_grad():
            pred = []
            gold = []
            for batch_idx, batch in enumerate(val_dataloader):
                img, label = batch
                img = img.to('cuda')
                gold.extend(label.numpy().tolist())
                _, catagory_logits = model(img)
                catagory_idx = torch.argmax(catagory_logits, dim=-1)
                catagory_idx = catagory_idx.detach().cpu().numpy().tolist()
                pred.extend(catagory_idx)
            acc = accuracy_score(gold, pred)
            print('Epoch: {}, ACC: {:.4f}'.format(epoch_idx, acc))
            torch.save(model.state_dict(), os.path.join(args.save_path, 'category_{}.pth'.format(epoch_idx)))


if __name__ == '__main__':
    args = parse_args()
    if args.action == 'eval':
        eval(args)
    if args.action == 'train':
        train(args)


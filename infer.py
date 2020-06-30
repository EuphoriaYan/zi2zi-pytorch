from data import DatasetFromObj
from torch.utils.data import DataLoader
from model import Zi2ZiModel
import os
import argparse
import torch
import random
import time
import math
import logging

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
parser.add_argument('--batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--schedule', type=int, default=10, help='number of epochs to half learning rate')
parser.add_argument('--resume', type=int, default=None, help='resume from previous training')


def main():
    args = parser.parse_args()
    data_dir = os.path.join(args.experiment_dir, "data")
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoint")
    sample_dir = os.path.join(args.experiment_dir, "sample")
    infer_dir = os.path.join(args.experiment_dir, "infer")

    # train_dataset = DatasetFromObj(os.path.join(data_dir, 'train.obj'), augment=True, bold=True, rotate=True, blur=True)
    # val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'))
    # dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = Zi2ZiModel(embedding_num=args.embedding_num, embedding_dim=args.embedding_dim,
                       Lconst_penalty=args.Lconst_penalty, Lcategory_penalty=args.Lcategory_penalty,
                       save_dir=checkpoint_dir, gpu_ids=args.gpu_ids, is_training=False)
    model.setup()
    model.print_networks(True)
    model.load_networks(args.resume)

    global_steps = 0
    val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'))
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    for batch in dataloader:
        model.set_input(batch[0], batch[2], batch[1])
        # model.optimize_parameters()
        model.sample(batch, os.path.join(infer_dir, "infer_{}".format(global_steps)))
        global_steps += 1


if __name__ == '__main__':
    main()

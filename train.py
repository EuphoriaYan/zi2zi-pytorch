from data import DatasetFromObj
from torch.utils.data import DataLoader
from model import Zi2ZiModel
import os
import argparse
import torch

parser = argparse.ArgumentParser(description='Train')
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
parser.add_argument('--embedding_num', type=int, default=41,
                    help="number for distinct embeddings")
parser.add_argument('--embedding_dim', type=int, default=128, help="dimension for embedding")
parser.add_argument('--epoch', type=int, default=100, help='number of epoch')
parser.add_argument('--batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--schedule', dest='schedule', type=int, default=10, help='number of epochs to half learning rate')
parser.add_argument('--resume', type=int, default=None, help='resume from previous training')
parser.add_argument('--freeze_encoder', action='store_true',
                    help="freeze encoder weights during training")
parser.add_argument('--fine_tune', type=str, default=None,
                    help='specific labels id to be fine tuned')
parser.add_argument('--inst_norm', action='store_true',
                    help='use conditional instance normalization in your model')
parser.add_argument('--sample_steps', type=int, default=10,
                    help='number of batches in between two samples are drawn from validation set')
parser.add_argument('--checkpoint_steps', type=int, default=100,
                    help='number of batches in between two checkpoints')
parser.add_argument('--flip_labels', action='store_true',
                    help='whether flip training data labels or not, in fine tuning')


def main():
    args = parser.parse_args()
    data_dir = os.path.join(args.experiment_dir, "data")
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoint")
    sample_dir = os.path.join(args.experiment_dir, "sample")
    log_dir = os.path.join(args.experiment_dir, "logs")

    train_dataset = DatasetFromObj(os.path.join(data_dir, 'train.obj'))
    val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'))
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = Zi2ZiModel(embedding_num=args.embedding_num, embedding_dim=args.embedding_dim,
                       Lconst_penalty=args.Lconst_penalty, Lcategory_penalty=args.Lcategory_penalty,
                       save_dir=checkpoint_dir, gpu_ids=args.gpu_ids)
    model.setup()
    model.print_networks(True)
    if args.resume is not None:
        model.load_networks(args.resume)

    start_epoch = args.resume if args.resume is not None else 0
    global_steps = 0
    for epoch in range(start_epoch, args.epoch):
        for batch in dataloader:
            model.set_input(batch[0], batch[2], batch[1])
            model.optimize_parameters()
            if global_steps % args.checkpoint_steps == 0:
                model.save_networks(epoch)
            if global_steps % args.sample_steps == 0:
                print("Step: %d" % global_steps)
                print("G_loss: %.4f, D_loss: %.4f" % (model.g_loss.item(), model.d_loss.item()))
                model.sample(batch, os.path.join(sample_dir, "sample_{}_{}".format(epoch, global_steps)))
            global_steps += 1
        model.update_lr()


if __name__ == '__main__':
    main()

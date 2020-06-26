import torch
import torch.nn as nn
from .generators import UNetGenerator
from .discriminators import Discriminator
from .losses import CategoryLoss, BinaryLoss
import os
from torch.optim.lr_scheduler import StepLR
from utils.init_net import init_net
import torchvision.utils as vutils


class Zi2ZiModel:
    def __init__(self, input_nc=3, embedding_num=40, embedding_dim=128,
                 ngf=64, ndf=64,
                 Lconst_penalty=15, Lcategory_penalty=1, L1_penalty=100,
                 schedule=10, lr=0.001, gpu_ids=None, save_dir='.', is_training=True):

        if is_training:
            self.use_dropout = True
        else:
            self.use_dropout = False

        self.Lconst_penalty = Lconst_penalty
        self.Lcategory_penalty = Lcategory_penalty
        self.L1_penalty = L1_penalty

        self.schedule = schedule

        self.save_dir = save_dir
        self.gpu_ids = gpu_ids

        self.input_nc = input_nc
        self.embedding_dim = embedding_dim
        self.embedding_num = embedding_num
        self.ngf = ngf
        self.ndf = ndf
        self.lr = lr
        self.is_training = is_training

    def setup(self):

        self.netG = UNetGenerator(input_nc=self.input_nc, embedding_num=self.embedding_num,
                                  embedding_dim=self.embedding_dim,
                                  ngf=self.ngf, use_dropout=self.use_dropout)
        self.netD = Discriminator(input_nc=2 * self.input_nc, embedding_num=self.embedding_num, ndf=self.ndf)

        init_net(self.netG, gpu_ids=self.gpu_ids)
        init_net(self.netD, gpu_ids=self.gpu_ids)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.scheduler_G = StepLR(self.optimizer_G, step_size=self.schedule, gamma=0.5)
        self.scheduler_D = StepLR(self.optimizer_D, step_size=self.schedule, gamma=0.5)

        self.category_loss = CategoryLoss(self.embedding_num)
        self.real_binary_loss = BinaryLoss(True)
        self.fake_binary_loss = BinaryLoss(False)
        self.l1_loss = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()

        if self.gpu_ids:
            self.category_loss.cuda()
            self.real_binary_loss.cuda()
            self.fake_binary_loss.cuda()
            self.l1_loss.cuda()
            self.mse.cuda()
            self.sigmoid.cuda()

        if self.is_training:
            self.netD.train()
            self.netG.train()
        else:
            self.netD.eval()
            self.netG.eval()

    def set_input(self, labels, real_A, real_B):
        if self.gpu_ids:
            self.real_A = real_A.to(self.gpu_ids[0])
            self.real_B = real_B.to(self.gpu_ids[0])
            self.labels = labels.to(self.gpu_ids[0])
        else:
            self.real_A = real_A
            self.real_B = real_B
            self.labels = labels

    def forward(self):
        # generate fake_B

        self.fake_B, self.encoded_real_A = self.netG(self.real_A, self.labels)
        self.encoded_fake_B = self.netG(self.fake_B).view(self.fake_B.shape[0], -1)

    def backward_D(self, no_target_source=False):

        real_AB = torch.cat([self.real_A, self.real_B], 1)
        fake_AB = torch.cat([self.real_A, self.fake_B], 1)

        real_D_logits, real_category_logits = self.netD(real_AB)
        fake_D_logits, fake_category_logits = self.netD(fake_AB.detach())

        real_category_loss = self.category_loss(real_category_logits, self.labels)
        fake_category_loss = self.category_loss(fake_category_logits, self.labels)
        category_loss = (real_category_loss + fake_category_loss) * self.Lcategory_penalty

        d_loss_real = self.real_binary_loss(real_D_logits)
        d_loss_fake = self.fake_binary_loss(fake_D_logits)

        self.d_loss = d_loss_real + d_loss_fake + category_loss / 2.0
        self.d_loss.backward()

    def backward_G(self, no_target_source=False):

        fake_AB = torch.cat([self.real_A, self.fake_B], 1)
        fake_D_logits, fake_category_logits = self.netD(fake_AB)

        const_loss = self.mse(self.encoded_real_A, self.encoded_fake_B)
        l1_loss = self.L1_penalty * self.l1_loss(self.fake_B, self.real_B)
        fake_category_loss = self.category_loss(fake_category_logits, self.labels)

        cheat_loss = self.real_binary_loss(fake_D_logits)

        self.g_loss = cheat_loss + l1_loss + self.Lcategory_penalty * fake_category_loss + const_loss
        self.g_loss.backward()

    def update_lr(self):
        self.scheduler_D.step()
        self.scheduler_G.step()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def print_networks(self, verbose=False):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in ['G', 'D']:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def save_networks(self, epoch):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in ['G', 'D']:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if self.gpu_ids and torch.cuda.is_available():
                    # torch.save(net.cpu().state_dict(), save_path)
                    torch.save(net.state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in ['G', 'D']:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                net.load_state_dict(torch.load(load_path))
                # net.eval()

    def sample(self, batch, basename):
        self.set_input(batch[0], batch[2], batch[1])
        self.forward()
        tensor_to_plot = torch.cat([self.fake_B, self.real_B], 3)
        img = vutils.make_grid(tensor_to_plot)
        vutils.save_image(tensor_to_plot, basename + "_construct.png")
        self.set_input(torch.randn(1, self.embedding_dim).repeat(batch[0].shape[0], 1), batch[2], batch[1])
        self.forward()
        tensor_to_plot = torch.cat([self.fake_B, self.real_A], 3)
        vutils.save_image(tensor_to_plot, basename + "_generate.png")


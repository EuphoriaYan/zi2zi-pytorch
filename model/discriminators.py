import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import math


class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, embedding_num, ndf=64, norm_layer=nn.BatchNorm2d, image_size=256):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the first conv layer
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        # as tf implement, kernel_size = 5, use "SAME" padding, so we should use kw = 5 and padw = 2
        # kw = 4
        # padw = 1
        kw = 5
        padw = 2
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        # in tf implement, there are only 3 conv2d layers with stride=2.
        # for n in range(1, 4):
        for n in range(1, 3):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = 8
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Maybe useful? Experiment need to be done later.
        # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)
        # final_channels = ndf * nf_mult
        final_channels = 1
        # use stride of 2 conv2 layer 3 times, cal the image_size
        image_size = math.ceil(image_size / 2)
        image_size = math.ceil(image_size / 2)
        image_size = math.ceil(image_size / 2)
        # 524288 = 512(num_of_channels) * (w/2/2/2) * (h/2/2/2) = 2^19  (w=h=256)
        # 131072 = 512(num_of_channels) * (w/2/2/2) * (h/2/2/2) = 2^17  (w=h=128)
        final_features = final_channels * image_size * image_size
        self.binary = nn.Linear(final_features, 1)
        self.catagory = nn.Linear(final_features, embedding_num)

    def forward(self, input):
        """Standard forward."""
        # features = self.model(input).view(input.shape[0], -1)
        features = self.model(input)
        features = features.view(input.shape[0], -1)
        binary_logits = self.binary(features)
        catagory_logits = self.catagory(features)
        return binary_logits, catagory_logits

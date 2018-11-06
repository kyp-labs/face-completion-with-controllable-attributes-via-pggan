"""StarGAN model from https://github.com/yunjey/StarGAN."""

import torch
import torch.nn as nn
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        """constructor."""
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        """forward."""
        return x + self.main(x)


class StarGenerator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, use_mask=True):
        """constructor."""
        super(StarGenerator, self).__init__()

        self.use_mask = use_mask
        layers = []
        if use_mask:
            layers.append(nn.Conv2d(3+1+c_dim, conv_dim, kernel_size=7,
                                    stride=1, padding=3, bias=False))
        else:
            layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7,
                                    stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True,
                                        track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for _ in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4,
                                    stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True,
                                            track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for _ in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for _ in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2,
                                             kernel_size=4, stride=2,
                                             padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True,
                                            track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1,
                                padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, mask=None, c=None):
        """forward."""
        # Replicate spatially and concatenate domain information.
        if c is not None:
            c = c.view(c.size(0), c.size(1), 1, 1)
            c = c.repeat(1, 1, x.size(2), x.size(3))
            x = torch.cat([x, c], dim=1)
        if mask is not None and self.use_mask:
            x = torch.cat([x, mask], dim=1)
        return self.main(x)


class StarDiscriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        """constructor."""
        super(StarDiscriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4,
                                stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for _ in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4,
                                    stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size,
                               bias=False)

    def forward(self, x):
        """forward."""
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

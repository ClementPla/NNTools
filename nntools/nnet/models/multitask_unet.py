import torch
import torch.nn as nn

from nntools.nnet import AbstractNet
from nntools.nnet.ops import ResidualBlock, Conv2d


class ExchangeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ExchangeBlock, self).__init__()
        if 'activation' not in kwargs:
            kwargs['activation'] = nn.ReLU()
        self.W11 = nn.Sequential(nn.Linear(in_channels, in_channels // 2), kwargs['activation'])
        self.W12 = nn.Sequential(nn.Linear(in_channels // 2, in_channels), nn.Sigmoid())
        self.conv1 = Conv2d(in_channels, out_channels, **kwargs)

        self.W13 = nn.Sequential(nn.Linear(out_channels, out_channels // 2), kwargs['activation'])
        self.W14 = nn.Sequential(nn.Linear(out_channels // 2, out_channels), nn.Sigmoid())

        self.W21 = nn.Sequential(nn.Linear(in_channels, in_channels // 2), kwargs['activation'])
        self.W22 = nn.Sequential(nn.Linear(in_channels // 2, in_channels), nn.Sigmoid())

        self.conv2 = Conv2d(in_channels, out_channels, **kwargs)

        self.W23 = nn.Sequential(nn.Linear(out_channels, out_channels // 2), kwargs['activation'])
        self.W24 = nn.Sequential(nn.Linear(out_channels // 2, out_channels), nn.Sigmoid())

        self.inner_conv = Conv2d(in_channels * 2, out_channels, **kwargs)

    def forward(self, x1, x2):
        x1_m = x1.mean(axis=(2, 3))
        alpha_1 = self.W12(self.W11(x1_m)).unsqueeze(2).unsqueeze(3)

        x2_m = x2.mean(axis=(2, 3))
        alpha_2 = self.W12(self.W11(x2_m)).unsqueeze(2).unsqueeze(3)

        i = self.inner_conv(torch.cat([x1 * alpha_1, x2 * alpha_2], dim=1))
        i_m = i.mean(axis=(2, 3))

        beta_1 = self.W14(self.W13(i_m)).unsqueeze(2).unsqueeze(3)
        beta_2 = self.W24(self.W23(i_m)).unsqueeze(2).unsqueeze(3)

        out1 = self.conv1(x1) * beta_1 + (1 - beta_1) * i
        out2 = self.conv2(x2) * beta_2 + (1 - beta_2) * i
        return out1, out2


class MultiTaskUnet(AbstractNet):
    def __init__(self, img_chan=3, output_chs=(2, 2), merge_output=True):
        super(MultiTaskUnet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = ResidualBlock(img_chan, 64, activation=nn.SELU())
        self.conv2 = ResidualBlock(64, 128, activation=nn.SELU())
        self.conv3 = ResidualBlock(128, 256, activation=nn.SELU())
        self.conv4 = ResidualBlock(256, 512, activation=nn.SELU())
        self.conv5 = ResidualBlock(512, 512, t=1, activation=nn.SELU())

        self.up = nn.Upsample(scale_factor=2)
        self.branch_1 = nn.ModuleDict({'left': Conv2d(1024, 512, activation=nn.SELU()),
                                       'right': Conv2d(1024, 512, activation=nn.SELU()),
                                       'exchange': ExchangeBlock(512, 256, activation=nn.SELU())})

        self.branch_2 = nn.ModuleDict({'left': Conv2d(512, 256, activation=nn.SELU()),
                                       'right': Conv2d(512, 256, activation=nn.SELU()),
                                       'exchange': ExchangeBlock(256, 128, activation=nn.SELU())})

        self.branch_3 = nn.ModuleDict({'left': Conv2d(256, 128, activation=nn.SELU()),
                                       'right': Conv2d(256, 128, activation=nn.SELU()),
                                       'exchange': ExchangeBlock(128, 64, activation=nn.SELU())})

        self.branch_4 = nn.ModuleDict({'left': nn.Sequential(Conv2d(128, 64, activation=nn.SELU()),
                                                             Conv2d(64, 64, activation=nn.SELU()),
                                                             Conv2d(64, output_chs[0], activation=nn.Identity())),
                                       'right': nn.Sequential(Conv2d(128, 64, activation=nn.SELU()),
                                                              Conv2d(64, 64, activation=nn.SELU()),
                                                              Conv2d(64, output_chs[1], activation=nn.Identity()))
                                       })
        self.merge_output = merge_output

    def forward(self, x):
        x1 = self.conv1(x)  # b x 64 x h x w
        x1_s = self.maxpool(x1)  # b x 64 x h/2x w/2

        x2 = self.conv2(x1_s)  # b x 128 x h/2 x w/2
        x2_s = self.maxpool(x2)  # b x 128 x h/4 x w/4

        x3 = self.conv3(x2_s)  # b x 256 x h/4 x w/4
        x3_s = self.maxpool(x3)  # b x 256 x h/8 x w/8

        x4 = self.conv4(x3_s)  # b x 512 x h/8 x w/8
        x4_s = self.maxpool(x4)  # b x 512 x h/16 x w/16

        x5 = self.conv5(x4_s)  # b x 512 x h/16 x w/16

        y4 = torch.cat([self.up(x5), x4], dim=1)  # b x 1024 x h/8 x w/8
        y14 = self.branch_1['left'](y4)  # b x 512 x h/8 x w/8
        y24 = self.branch_1['right'](y4)  # b x 512 x h/8 x w/8
        y14, y24 = self.branch_1['exchange'](y14, y24)  # b x 256 x h/8 x w/8

        y13 = torch.cat([self.up(y14), x3], dim=1)  # b x 512 x h/4 x w/4
        y23 = torch.cat([self.up(y24), x3], dim=1)  # b x 512 x h/4 x w/4
        y13 = self.branch_2['left'](y13)  # b x 256 x h/4 x w/4
        y23 = self.branch_2['right'](y23)  # b x 256 x h/4 x w/4
        y13, y23 = self.branch_2['exchange'](y13, y23)  # b x 128 x h/4 x w/4

        y12 = torch.cat([self.up(y13), x2], dim=1)  # b x 256 x h/2 x w/2
        y22 = torch.cat([self.up(y23), x2], dim=1)  # b x 256 x h/2 x w/2

        y12 = self.branch_3['left'](y12)  # b x 128 x h/2 x w/2
        y22 = self.branch_3['right'](y22)  # b x 128 x h/2 x w/2
        y12, y22 = self.branch_3['exchange'](y12, y22)  # b x 64 x h/2 x w/2

        y11 = torch.cat([self.up(y12), x1], dim=1)  # b x 64 x h x w
        y21 = torch.cat([self.up(y22), x1], dim=1)  # b x 64 x h x w
        y1 = self.branch_4['left'](y11)  # b x n_out1 x h x w
        y2 = self.branch_4['left'](y21)  # b x n_out2 x h x w
        if not self.merge_output:
            return y1, y2
        else:
            y = torch.cat([y1, y2], dim=1)  # b x n_out1+n_out2 x h x w
            return y

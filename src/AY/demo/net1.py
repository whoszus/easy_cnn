# -*- coding utf-8 -*-
"""
# @Time: 2019/7/14  16:54
# Author:
"""
import torch.nn as nn
from torch.nn import functional as F


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, 1),  # 输入和输出的feature 大小不变
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 1, 1),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        x = x + self.residual(x)
        x = F.relu(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = self.make_layer(28, 128, n_res=3)
        self.layer2 = self.make_layer(128, 256, n_res=5)
        self.layer3 = self.make_layer(256, 128, n_res=3)

        self.out_1 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Flatten(),
            nn.Linear(128 * W * H, HHHH1111),
            nn.LogSoftmax(dim=1)
        )
        self.out_2 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Flatten(),
            nn.Linear(128 * W * H, HHHH2222),
            nn.ReLU(),
            nn.Linear(HHHH2222, hhhhh33333),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out_1 = self.out_1(x)
        out_2 = self.out_2(x)
        return out_1, out_2

    def make_layer(self, in_c, out_c, n_res=3):
        layer_lst = nn.ModuleList([
            nn.Conv2d(in_c, out_c, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        ])
        layer_lst.extend([ResidualBlock(out_c, out_c) for _ in range(n_res)])
        return nn.Sequential(*layer_lst)

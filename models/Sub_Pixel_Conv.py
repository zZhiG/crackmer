import torch
import torch.nn as nn
import torch.nn.functional as F


class oneDConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_sizes, padding=paddings, dilation=dilations),
            nn.BatchNorm1d(out_channels),
            # nn.LayerNorm([out_channels,3199]),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class SubConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, r, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * r, kernel_size=kernel_sizes, padding=paddings, dilation=dilations),
            nn.BatchNorm1d(in_channels * r),
            # nn.LayerNorm([out_channels,3199]),
        )

    def forward(self, x):
        B, C, T = x.size()
        print(B, C, T)
        x = self.conv(x)
        x = torch.reshape(x, (B, C, -1))  ###变形
        return x


# x = torch.randn(5, 1, 32000)
#
# net = SubConv(1, 2, 1, 0, 1)
# outputs = net(x)
# print(outputs.size())

pc = nn.PixelShuffle(2)
x = torch.rand(1, 12, 12, 12)
print(pc(x).shape)
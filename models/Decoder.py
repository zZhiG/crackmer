import torch
import torch.nn as nn

from Model2.Transformer import Block
from Model2.Separable_convolution import S_conv_r

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.t3 = S_conv_r(384, 256)
        self.t2 = S_conv_r(192, 128)
        self.t1 = S_conv_r(96, 64)
        # self.t0 = nn.Conv2d(48, 32, 1)

        # self.block3 = Block(256, window_size=2, alpha=0.2, dropout=0.)
        # self.block2 = Block(128, window_size=2, alpha=0.3, dropout=0.)
        # self.block1 = Block(64, window_size=2, alpha=0.4, dropout=0.)

        self.block3 = Block(256, window_size=2, alpha=0.2)
        self.block2 = Block(128, window_size=2, alpha=0.3)
        self.block1 = Block(64, window_size=2, alpha=0.4)

        self.up = nn.PixelShuffle(2)

        self.final = nn.Sequential(nn.PixelShuffle(4),
                                   S_conv_r(4, 1))


    def forward(self, x, x1, x2, x3):
        temp = self.up(x) #torch.Size([1, 128, 32, 32])
        temp = torch.cat([temp, x3], dim=1)
        temp = self.t3(temp)
        B, C, H, W = temp.shape
        # temp = temp.reshape(B, C, -1).permute(0, 2, 1)
        x3_out = self.block3(temp) #torch.Size([1, 256, 32, 32])

        # x3_out = x3_out.permute(0, 2, 1).reshape(B, C, H, W)
        temp = self.up(x3_out)
        temp = torch.cat([temp, x2], dim=1) #torch.Size([1, 192, 64, 64])
        temp = self.t2(temp)
        B, C, H, W = temp.shape
        # temp = temp.reshape(B, C, -1).permute(0, 2, 1)
        x2_out = self.block2(temp) #torch.Size([1, 128, 64, 64])

        # x2_out = x2_out.permute(0, 2, 1).reshape(B, C, H, W)
        temp = self.up(x2_out)
        temp = torch.cat([temp, x1], dim=1) #torch.Size([1, 96, 128, 128])
        temp = self.t1(temp)
        B, C, H, W = temp.shape
        # temp = temp.reshape(B, C, -1).permute(0, 2, 1)
        x1_out = self.block1(temp) #torch.Size([1, 64, 128, 128])

        # x1_out = x1_out.permute(0, 2, 1).reshape(B, C, H, W)
        # temp = self.up(x1_out) #torch.Size([1, 16, 256, 256])
        # temp = torch.cat([temp, x4], dim=1)
        # # temp = self.t0(temp)
        out = self.final(x1_out)

        return out

if __name__ == '__main__':
    x1 = torch.rand(1, 64, 128, 128).cuda()
    x2 = torch.rand(1, 128, 64, 64).cuda()
    x3 = torch.rand(1, 256, 32, 32).cuda()
    # x4 = torch.rand(1, 32, 256, 256).cuda()
    x = torch.rand(1, 512, 16, 16).cuda()
    model = Decoder().cuda()
    out = model(x, x1, x2, x3)
    print(out.shape)
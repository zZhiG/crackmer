import torch
import torch.nn as nn

from transformer import Block
from dsc import IDSC

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.t3 = IDSC(384, 256)
        self.t2 = IDSC(192, 128)
        self.t1 = IDSC(96, 64)

        self.block3 = Block(256, window_size=2, alpha=0.2)
        self.block2 = Block(128, window_size=2, alpha=0.3)
        self.block1 = Block(64, window_size=2, alpha=0.4)

        self.up = nn.PixelShuffle(2)

        self.final = nn.Sequential(nn.PixelShuffle(4),
                                   IDSC(4, 1))


    def forward(self, x, x1, x2, x3):
        temp = self.up(x)
        temp = torch.cat([temp, x3], dim=1)
        temp = self.t3(temp)
        x3_out = self.block3(temp)

        temp = self.up(x3_out)
        temp = torch.cat([temp, x2], dim=1)
        temp = self.t2(temp)

        x2_out = self.block2(temp)

        temp = self.up(x2_out)
        temp = torch.cat([temp, x1], dim=1)
        temp = self.t1(temp)

        x1_out = self.block1(temp)

        out = self.final(x1_out)

        return out
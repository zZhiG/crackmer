import torch
import torch.nn as nn

from dsc import DSC, IDSC

class FCM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(dim, dim)
        self.down = IDSC(3*dim, dim)

        self.fuse = nn.Sequential(IDSC(3*dim, dim),
                                  nn.BatchNorm2d(dim),
                                  nn.GELU(),
                                  DSC(dim, dim),
                                  nn.BatchNorm2d(dim),
                                  nn.GELU(),
                                  DSC(dim, dim),
                                  nn.BatchNorm2d(dim),
                                  nn.GELU()
                                  )

    def forward(self, x1, y1):
        B1, C1, H1, W1 = x1.shape
        B2, C2, H2, W2 = y1.shape

        x_temp = self.avg(x1)
        y_temp = self.avg(y1)
        x_weight = self.linear(x_temp.reshape(B1, 1, 1, C1))
        y_weight = self.linear(y_temp.reshape(B2, 1, 1, C2))
        x_temp = x1.permute(0, 2, 3, 1)
        y_temp = y1.permute(0, 2, 3, 1)


        x1 = x_temp * x_weight
        y1 = y_temp * y_weight

        out1 = torch.cat([x1, y1], dim=3)

        out2 = x1 * y1

        fuse = torch.cat([out1, out2], dim=3)
        fuse = fuse.permute(0, 3, 1, 2)

        out = self.fuse(fuse)
        out = out + self.down(fuse)

        return out
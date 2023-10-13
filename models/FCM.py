import torch
import torch.nn as nn

from Model2.Separable_convolution import S_conv, S_conv_r

class FCM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(dim, dim)
        self.down = S_conv_r(3*dim, dim)

        self.fuse = nn.Sequential(S_conv_r(3*dim, dim),
                                  nn.BatchNorm2d(dim),
                                  nn.GELU(),
                                  S_conv(dim, dim),
                                  nn.BatchNorm2d(dim),
                                  nn.GELU(),
                                  S_conv(dim, dim),
                                  nn.BatchNorm2d(dim),
                                  nn.GELU()
                                  )

    def forward(self, x1, y1):
        B1, C1, H1, W1 = x1.shape
        B2, C2, H2, W2 = y1.shape
        # 两部分输入首先进行通道注意力，抑制影响力较低的通道
        x_temp = self.avg(x1)
        y_temp = self.avg(y1)
        x_weight = self.linear(x_temp.reshape(B1, 1, 1, C1))
        y_weight = self.linear(y_temp.reshape(B2, 1, 1, C2))
        x_temp = x1.permute(0, 2, 3, 1)
        y_temp = y1.permute(0, 2, 3, 1)
        # print(x_temp.shape)

        x1 = x_temp * x_weight
        y1 = y_temp * y_weight

        out1 = torch.cat([x1, y1], dim=3)

        out2 = x1 * y1

        fuse = torch.cat([out1, out2], dim=3)
        fuse = fuse.permute(0, 3, 1, 2)
        
        out = self.fuse(fuse)
        out = out + self.down(fuse)

        return out

if __name__ == '__main__':
    x = torch.rand(1, 32, 256, 256).cuda()
    y = torch.rand(1, 32, 256, 256).cuda()
    net = FCM(32).cuda()
    out = net(x, y)
    print(out.shape)
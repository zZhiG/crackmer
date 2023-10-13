import torch
import torch.nn as nn

from Model2.CNN import CNN
from Model2.Transformer import Transformer
from Model2.Decoder import Decoder
from Model2.FCM import FCM
from Model2.Separable_convolution import S_conv, S_conv_r
from Model2.ASPP import SASPP
from Model2.vgg import vgg

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = CNN()
        # self.encoder1 = vgg('vgg16', True)
        self.encoder2 = Transformer()

        self.fuse1 = FCM(64)
        self.fuse2 = FCM(128)
        self.fuse3 = FCM(256)

        self.Conv = nn.Sequential(S_conv_r(1024, 512),
                                  nn.BatchNorm2d(512),
                                  nn.GELU())
        self.decoder = Decoder()
        # self.enhance = SASPP(512, ratio=[1, 3, 5, 7])

        #通道注意力
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(512, 512)

    def forward(self, x):
        x1, x2, x3, x4, out1 = self.encoder1(x)
        y1, y2, y3, out2 = self.encoder2(x)

        f1 = self.fuse1(x2, y1)
        f2 = self.fuse2(x3, y2)
        f3 = self.fuse3(x4, y3)

        #通道注意力
        B1, C1, H1, W1 = out1.shape
        B2, C2, H2, W2 = out2.shape
        x_temp = self.avg(out1)
        y_temp = self.avg(out2)
        x_weight = self.linear(x_temp.reshape(B1, 1, 1, C1))
        y_weight = self.linear(y_temp.reshape(B2, 1, 1, C2))
        x_temp = out1.permute(0, 2, 3, 1)
        y_temp = out2.permute(0, 2, 3, 1)
        x1 = x_temp * x_weight
        y1 = y_temp * y_weight
        # out = x1 + y1
        # out = out.permute(0, 3, 1, 2)
        x1 = x1.permute(0, 3, 1, 2)
        y1 = y1.permute(0, 3, 1, 2)

        # out = out1 + out2
        out = torch.cat([x1, y1], dim=1)
        out = self.Conv(out) #torch.Size([1, 256, 16, 16])

        # out = self.enhance(out)
        mask = self.decoder(out, f1, f2, f3)

        return mask

if __name__ == '__main__':
    x = torch.rand(1, 3, 512, 512).cuda()
    net = Net().cuda()
    out = net(x)
    print(out.shape)
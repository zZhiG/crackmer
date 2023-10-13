import torch
import torch.nn as nn

from Model2.Separable_convolution import S_conv

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # layer1
        self.conv1_1 = S_conv(3, 32)
        self.conv1_2 = S_conv(32, 32)
        self.conv1_3 = S_conv(32, 32)
        self.norm1 = nn.BatchNorm2d(32)
        self.act = nn.GELU()

        self.c1_1 = nn.Sequential(self.conv1_1,
                                  self.norm1,
                                  self.act)
        self.c1_2 = nn.Sequential(self.conv1_2,
                                  self.norm1,
                                  self.act)
        self.c1_3 = nn.Sequential(self.conv1_3,
                                  self.norm1,
                                  self.act)
        self.res1 = nn.Conv2d(64, 32, kernel_size=1)

        self.pool1 = S_conv(32, 32, 2, 2, 0)

        #layer2
        self.conv2_1 = S_conv(32, 64)
        self.conv2_2 = S_conv(64, 64)
        self.conv2_3 = S_conv(64, 64)
        self.norm2 = nn.BatchNorm2d(64)

        self.c2_1 = nn.Sequential(self.conv2_1,
                                  self.norm2,
                                  self.act)
        self.c2_2 = nn.Sequential(self.conv2_2,
                                  self.norm2,
                                  self.act)
        self.c2_3 = nn.Sequential(self.conv2_2,
                                  self.norm2,
                                  self.act)
        self.res2 = nn.Conv2d(128, 64, kernel_size=1)

        self.pool2 = S_conv(64, 64, 2, 2, 0)

        #layer3
        self.conv3_1 = S_conv(64, 128)
        self.conv3_2 = S_conv(128, 128)
        self.conv3_3 = S_conv(128, 128)
        self.conv3_4 = S_conv(128, 128)
        self.conv3_5 = S_conv(128, 128)
        self.norm3 = nn.BatchNorm2d(128)
        self.c3_1 = nn.Sequential(self.conv3_1,
                                  self.norm3,
                                  self.act)
        self.c3_2 = nn.Sequential(self.conv3_2,
                                  self.norm3,
                                  self.act)
        self.c3_3 = nn.Sequential(self.conv3_3,
                                  self.norm3,
                                  self.act)
        self.c3_4 = nn.Sequential(self.conv3_4,
                                  self.norm3,
                                  self.act)
        self.c3_5 = nn.Sequential(self.conv3_5,
                                  self.norm3,
                                  self.act)
        self.res3 = nn.Conv2d(256, 128, 1)

        self.pool3 = S_conv(128, 128, 2, 2, 0)

        #layer4
        self.conv4_1 = S_conv(128, 256)
        self.conv4_2 = S_conv(256, 256)
        self.conv4_3 = S_conv(256, 256)
        self.norm4 = nn.BatchNorm2d(256)
        self.c4_1 = nn.Sequential(self.conv4_1,
                                  self.norm4,
                                  self.act)
        self.c4_2 = nn.Sequential(self.conv4_2,
                                  self.norm4,
                                  self.act)
        self.c4_3 = nn.Sequential(self.conv4_3,
                                  self.norm4,
                                  self.act)
        self.res4 = nn.Conv2d(512, 256, 1)

        self.pool4 = S_conv(256, 256, 2, 2, 0)

        # self.conv5_1 = S_conv(256, 512)
        # self.conv5_2 = S_conv(512, 512)
        # self.norm5 = nn.BatchNorm2d(512)
        # self.c5_1 = nn.Sequential(self.conv5_1,
        #                           self.norm5,
        #                           self.act)
        # self.c5_2 = nn.Sequential(self.conv5_2,
        #                           self.norm5,
        #                           self.act)
        # self.res5 = nn.Conv2d(1024, 512, 1)
        self.pool5 = S_conv(256, 512, 2, 2, 0)


    def forward(self, x):
        #layer1
        x1_1 = self.c1_1(x)
        x1_p = self.pool1(x1_1)
        x1_2 = self.c1_2(x1_p)
        x1_3 = self.c1_3(x1_2)
        temp1 = torch.cat([x1_p, x1_3], dim=1)
        x1_out = self.res1(temp1) #x1_out torch.Size([1, 32, 256, 256])

        # print('x1_out', x1_out.shape)

        #layer2
        x2_1 = self.c2_1(x1_out)
        x2_p = self.pool2(x2_1)
        x2_2 = self.c2_2(x2_p)
        x2_3 = self.c2_3(x2_2)
        temp2 = torch.cat([x2_p, x2_3], dim=1)
        x2_out = self.res2(temp2) #x2_out torch.Size([1, 64, 128, 128])

        # print('x2_out', x2_out.shape)

        #layer3
        x3_1 = self.c3_1(x2_out)
        x3_2 = self.c3_2(x3_1)
        x3_p = self.pool3(x3_2)
        x3_3 = self.c3_3(x3_p)
        x3_4 = self.c3_4(x3_3)
        x3_5 = self.c3_5(x3_4)
        temp3 = torch.cat([x3_p, x3_5], dim=1)
        x3_out = self.res3(temp3) #x3_out torch.Size([1, 128, 64, 64])

        # print('x3_out', x3_out.shape)

        #layer4
        x4_1 = self.c4_1(x3_out)
        x4_p = self.pool4(x4_1)
        x4_2 = self.c4_2(x4_p)
        x4_3 = self.c4_3(x4_2)
        temp4 = torch.cat([x4_p, x4_3], dim=1)
        x4_out = self.res4(temp4) #x4_out torch.Size([1, 256, 32, 32])

        # print('x4_out', x4_out.shape)

        #layer5
        out = self.pool5(x4_out) #torch.Size([1, 512, 16, 16])


        return x1_out, x2_out, x3_out, x4_out, out

if __name__ == '__main__':
    x = torch.rand(1, 3, 512, 512).cuda()
    model = CNN().cuda()
    # x1_out, x2_out, x3_out, x4_out, out = model(x)
    out = model(x)
    for i in out:
        print(i.shape)
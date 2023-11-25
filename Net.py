import torch
import torch.nn as nn


class S_conv(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(S_conv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_in, c_in, k_size, stride, padding, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out

class S_conv_r(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(S_conv_r, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_out, c_out, k_size, stride, padding, groups=c_out)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.pw(x)
        out = self.dw(out)
        return out

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

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.t3 = S_conv_r(384, 256)
        self.t2 = S_conv_r(192, 128)
        self.t1 = S_conv_r(96, 64)

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

        x3_out = self.block3(temp) #torch.Size([1, 256, 32, 32])

        temp = self.up(x3_out)
        temp = torch.cat([temp, x2], dim=1) #torch.Size([1, 192, 64, 64])
        temp = self.t2(temp)

        x2_out = self.block2(temp) #torch.Size([1, 128, 64, 64])

        temp = self.up(x2_out)
        temp = torch.cat([temp, x1], dim=1) #torch.Size([1, 96, 128, 128])
        temp = self.t1(temp)

        x1_out = self.block1(temp) #torch.Size([1, 64, 128, 128])

        out = self.final(x1_out)

        return out


class FCM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(dim, dim)
        self.down = S_conv_r(3 * dim, dim)

        self.fuse = nn.Sequential(S_conv_r(3 * dim, dim),
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

class PatchEmbed(nn.Module):
    def __init__(self, dim, p_size):
        super().__init__()
        self.embed = S_conv(3, dim, p_size, p_size, 0)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = self.norm(self.embed(x))
        return x

class PatchMerge(nn.Module):
    def __init__(self, inc, outc, kernel_size = 2):
        super().__init__()
        self.merge = S_conv(inc, outc, k_size=kernel_size, stride=kernel_size, padding=0)
        self.norm = nn.BatchNorm2d(outc)

    def forward(self, x):
        return self.norm(self.merge(x))

class Attention(nn.Module):
    def __init__(self, dim, window_size=2, num_head=8, qk_scale=None, qkv_bias=None, alpha=0.5):
        super().__init__()
        head_dim = int(dim / num_head)
        self.dim = dim

        self.l_head = int(num_head * alpha)
        self.l_dim = self.l_head * head_dim

        self.h_head = num_head - self.l_head
        self.h_dim = self.h_head * head_dim

        self.ws = window_size
        if self.ws == 1:
            self.h_head = 0
            self.h_dim = 0
            self.l_head = num_head
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5

        if self.l_head > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = S_conv(self.dim, self.l_dim)
            self.l_kv = S_conv(self.dim, self.l_dim * 2)
            self.l_proj = S_conv(self.l_dim, self.l_dim)

        if self.h_head > 0:
            self.h_qkv = S_conv(self.dim, self.h_dim * 3)
            self.h_proj = S_conv(self.h_dim, self.h_dim)

    def hifi(self, x):
        B, C, H, W = x.shape
        h_group, w_group = H // self.ws, W // self.ws
        total_groups = h_group * w_group

        qkv = self.h_qkv(x).reshape(B, 3, self.h_head, self.h_dim // self.h_head, total_groups, self.ws * self.ws)\
            .permute(1, 0, 4, 2, 5, 3)

        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim).permute(0, 3, 1, 2)
        x = self.h_proj(x)
        return x

    def lofi(self, x):
        B, C, H, W = x.shape
        q = self.l_q(x).reshape(B, self.l_head, self.l_dim // self.l_head, H*W).permute(0, 1, 3, 2)

        if self.ws > 1:
            x_ = self.sr(x)
            kv = self.l_kv(x_).reshape(B, 2, self.l_head, self.l_dim // self.l_head, -1).permute(1, 0, 2, 4, 3)
        else:
            kv = self.l_kv(x).reshape(B, 2, self.l_head, self.l_dim // self.l_head, -1).permute(1, 0, 2, 4, 3)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim).permute(0, 3, 1, 2)
        x = self.l_proj(x)
        return x

    def forward(self, x):
        if self.h_head > 0 and self.l_head>0:
            x_h = self.hifi(x)
            x_l = self.lofi(x)
            x = torch.cat([x_h, x_l], dim=1)
            return x

        elif self.l_head > 0 and self.h_head == 0:
            x_l = self.lofi(x)
            return x_l

        else:
            x_h = self.hifi(x)
            return x_h

class Mlp(nn.Module):
    def __init__(self, inc, outc = None, dropout = 0.2):
        super().__init__()
        # outc = outc or inc * 2
        outc = outc or inc
        self.fc1 = nn.Conv2d(inc, outc, 1)
        self.fc2 = S_conv(outc, outc)
        self.fc3 = nn.Conv2d(outc, inc, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)

        return x

class Block(nn.Module):
    def __init__(self, inc, window_size=2, num_head=8, alpha = 0.5, dropout = 0.):
        super().__init__()
        self.norm = nn.BatchNorm2d(inc)

        self.HiLo = Attention(inc, window_size=window_size, num_head=num_head, alpha=alpha)
        self.mlp = Mlp(inc, dropout=dropout)

    def forward(self, x):
        x = x + self.norm(self.HiLo(x))
        x = x + self.norm(self.mlp(x))

        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        #layer1
        self.embed = PatchEmbed(64, 4)
        self.block1 = Block(64, window_size=2, num_head=8, alpha = 0.4, dropout=0.)

        #layer2
        self.merge2 = PatchMerge(64, 128)
        self.block2 = Block(128, window_size=2, num_head=8, alpha=0.3, dropout=0.)

        #layer3
        self.merge3 = PatchMerge(128, 256)
        self.block3 = Block(256, window_size=2, num_head=8, alpha=0.2, dropout=0.)

        #layer4
        self.merge4 = PatchMerge(256, 512)
        self.block4 = Block(512, window_size=2, num_head=8, alpha=0.1, dropout=0.)

    def forward(self, x):
        #layer1
        x1 = self.embed(x)
        x1 = self.block1(x1) #torch.Size([1, 64, 128, 128])

        #layer2
        x2 = self.merge2(x1)
        x2 = self.block2(x2) #torch.Size([1, 128, 64, 64])

        #layer3
        x3 = self.merge3(x2)
        x3 = self.block3(x3) #torch.Size([1, 256, 32, 32])

        #layer4
        x4 = self.merge4(x3) #torch.Size([1, 512, 16, 16])
        out = self.block4(x4) #torch.Size([1, 512, 16, 16])
        return x1, x2, x3, out



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = CNN()
        self.encoder2 = Transformer()

        self.fuse1 = FCM(64)
        self.fuse2 = FCM(128)
        self.fuse3 = FCM(256)

        self.Conv = nn.Sequential(S_conv_r(1024, 512),
                                  nn.BatchNorm2d(512),
                                  nn.GELU())
        self.decoder = Decoder()

        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(512, 512)

    def forward(self, x):
        x1, x2, x3, x4, out1 = self.encoder1(x)
        y1, y2, y3, out2 = self.encoder2(x)

        f1 = self.fuse1(x2, y1)
        f2 = self.fuse2(x3, y2)
        f3 = self.fuse3(x4, y3)

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

        x1 = x1.permute(0, 3, 1, 2)
        y1 = y1.permute(0, 3, 1, 2)

        out = torch.cat([x1, y1], dim=1)
        out = self.Conv(out) #torch.Size([1, 256, 16, 16])

        mask = self.decoder(out, f1, f2, f3)

        return mask

if __name__ == '__main__':
    x = torch.rand(1, 3, 512, 512).cuda()
    net = Net().cuda()
    out = net(x)
    print(out.shape)


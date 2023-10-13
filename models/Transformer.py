import torch
import torch.nn as nn

from Model2.Separable_convolution import S_conv

class PatchEmbed(nn.Module):
    def __init__(self, dim, p_size):
        super().__init__()
        self.embed = S_conv(3, dim, p_size, p_size, 0)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = self.norm(self.embed(x)) #torch.Size([1, 64, 128, 128])
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
        head_dim = int(dim / num_head) #每个头可以获得的维度
        self.dim = dim

        self.l_head = int(num_head * alpha) #低频获得的头
        self.l_dim = self.l_head * head_dim #低频获得的维度

        self.h_head = num_head - self.l_head #高频获得的头
        self.h_dim = self.h_head * head_dim #高频获得的维度

        self.ws = window_size #低频压缩的窗口大小
        if self.ws == 1:
            # ws == 1 则表明低频无压缩
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
        self.block2 = Block(128, window_size=2, num_head=8, alpha=0.4, dropout=0.)

        #layer3
        self.merge3 = PatchMerge(128, 256)
        self.block3 = Block(256, window_size=2, num_head=8, alpha=0.4, dropout=0.)

        #layer4
        self.merge4 = PatchMerge(256, 512)
        self.block4 = Block(512, window_size=2, num_head=8, alpha=0.4, dropout=0.)

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

if __name__ == '__main__':
    x = torch.rand(1, 3, 512, 512).cuda()
    model = Transformer().cuda()
    x1, x2, x3, out = model(x)
    print(out.shape)
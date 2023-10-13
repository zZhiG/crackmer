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

#pw+dw,降维可用
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

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

def get_resnet18(is_bool):
    model = resnet18(pretrained=is_bool)
    return model

def get_resnet34(is_bool):
    model = resnet34(pretrained=is_bool)
    return model

def get_resnet50(is_bool):
    model = resnet50(pretrained=is_bool)
    return model

def get_resnet101(is_bool):
    model = resnet101(pretrained=is_bool)
    return model

def get_resnet152(is_bool):
    model = resnet152(pretrained=is_bool)
    return model

def get_resnet(model='resnet50', is_bool=True):
    if model == 'resnet18':
        return get_resnet18(is_bool)

    elif model == 'resnet50':
        return get_resnet50(is_bool)

    elif model == 'resnet34':
        return get_resnet34(is_bool)

    elif model == 'resnet101':
        return get_resnet101(is_bool)

    elif model == 'resnet152':
        return get_resnet152(is_bool)

    else:
        print('Not exist model!!!!')
        return ""

class Res(nn.Module):
    def __init__(self, mode='resnet34', is_bool=True):
        super().__init__()
        cnn = get_resnet(mode, is_bool)
        self.init = nn.Sequential(cnn.conv1,
                                      cnn.bn1,
                                      cnn.relu)
        self.maxp = cnn.maxpool
        self.layer1 = cnn.layer1
        self.layer2 = cnn.layer2
        self.layer3 = cnn.layer3
        self.layer4 = cnn.layer4

        self.c1 = nn.Conv2d(64, 32, 1)
        # self.c2 = nn.Conv2d(64, 64, 1)
        # self.c3 = nn.Conv2d(64, 128, 1)
        # self.c4 = nn.Conv2d(64, 256, 1)
        # self.c5 = nn.Conv2d(64, 512, 1)

    def forward(self, x):
        x1 = self.init(x)
        x2 = self.maxp(x1)
        x1 = self.c1(x1)
        x2 = self.layer1(x2)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)


        return x1, x2, x3, x4, x5

if __name__ == '__main__':
    x = torch.rand(2, 3, 512, 512).cuda()
    # model = get_resnet('resnet34', True)
    # print(model)
    m = Res('resnet18', False).cuda()
    o = m(x)
    for i in o:
        print(i.shape)
'''
torch.Size([1, 32, 256, 256])
torch.Size([1, 64, 128, 128])
torch.Size([1, 128, 64, 64])
torch.Size([1, 256, 32, 32])
torch.Size([1, 512, 16, 16])
'''

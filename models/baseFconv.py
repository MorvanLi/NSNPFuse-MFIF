# -*- coding: utf-8 -*-
# @Time    : 2024/1/4 21:52
# @Author  : MorvanLi
# @Email   : morvanli1995@gmail.com
# @File    : baseFconv.py
# @Software: PyCharm

import torch.nn as nn
import torch
import F_conv as fn
class ConvSNP(nn.Module):
    def __init__(self, inChannels, growRate, kSize=5, tranNum=8, ifIni=0, Smooth=False, use=False, dilation=1, islast=False):
        super(ConvSNP, self).__init__()
        Cin = inChannels
        G = growRate
        ifIni = ifIni
        if islast:
            self.convSNP = nn.Sequential(
                nn.Conv2d(inChannels, growRate, kernel_size=kSize, padding=(kSize - 1) // 2 * dilation,
                          dilation=dilation),
                nn.Sigmoid(),  # 非线性激活函数
            )
        else:
            if use:
                self.convSNP = nn.Sequential(
                    fn.Fconv_PCA(kSize, Cin, G, tranNum=tranNum, inP=kSize, padding=(kSize - 1) // 2, ifIni=ifIni,
                                 Smooth=Smooth),
                    nn.ReLU(),  # 非线性激活函数
                )
            else:
                self.convSNP = nn.Sequential(
                    nn.Conv2d(inChannels, growRate, kernel_size=kSize, padding=(kSize - 1) // 2 * dilation,
                              dilation=dilation),
                    nn.ReLU(),  # 非线性激活函数
                )


    def forward(self, x):
        out = self.convSNP(x)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Shallow features
        self.conv1 = ConvSNP(inChannels=3, growRate=8, kSize=5, tranNum=8, ifIni=1, use=True)

        # Deep features
        self.conv2 = ConvSNP(inChannels=8, growRate=8, kSize=5, tranNum=8, use=True)
        self.conv3 = ConvSNP(inChannels=8, growRate=8, kSize=5, tranNum=8, use=True)
        self.conv4 = ConvSNP(inChannels=8, growRate=8, kSize=5, tranNum=8, use=True)
        self.conv5 = ConvSNP(inChannels=8, growRate=8, kSize=5, tranNum=8, use=True)

    def forward(self, x1):
        F1 = self.conv1(x1)
        F2 = self.conv2(F1)
        F3 = self.conv3(F2)
        F4 = self.conv4(F3)
        F5 = self.conv5(F4)
        return F5


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.drp = fn.Fconv_1X1(8 * 2, 8, 8, 0, Smooth=False)
        # self.drp = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)

    def forward(self, x1, x2):
        f = torch.cat([x1, x2], dim=1)
        f = self.drp(f)
        return f


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.r_conv1 = ConvSNP(inChannels=64, growRate=64, kSize=3, use=False)
        self.r_conv2 = ConvSNP(inChannels=64, growRate=64, kSize=3, use=False)
        self.r_conv3 = ConvSNP(inChannels=64, growRate=64, kSize=3, use=False)
        self.r_conv4 = ConvSNP(inChannels=64, growRate=64, kSize=3, use=False)
        self.r_conv5 = ConvSNP(inChannels=64, growRate=3,  kSize=3, use=False, islast=True)

    def forward(self, x):

        x1 = self.r_conv1(x)
        x2 = self.r_conv2(x1)
        x3 = self.r_conv3(x2)
        x4 = self.r_conv4(x3)
        x5 = self.r_conv5(x4)
        return x5

class BaseFconvNet(nn.Module):
    def __init__(self):
        super(BaseFconvNet, self).__init__()
        self.encoder = Encoder()
        self.fusion = Fusion()
        self.decoder = Decoder()

    def forward(self, x1, x2):
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        f = self.fusion(f1, f2)
        out = self.decoder(f)
        return out

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    input1 = torch.rand((1, 3, 256, 256))
    input2 = torch.rand((1, 3, 256, 256))
    train_net = BaseFconvNet()

    print(train_net)
    out = train_net(input1, input2)
    print(out.shape)
    print("DenseFuse have {} paramerters in total".format(sum(x.numel() for x in train_net.parameters())  / 1000 / 1000))
    ##### DenseFuse have 572419 paramerters in total
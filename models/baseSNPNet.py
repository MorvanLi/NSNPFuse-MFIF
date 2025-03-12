# -*- coding: utf-8 -*-
# @Time    : 2024/1/5 15:42
# @Author  : MorvanLi
# @Email   : morvanli1995@gmail.com
# @File    : baseSNPNet.py
# @Software: PyCharm
import torch.nn as nn
import torch
import F_conv as fn
class ConvSNP(nn.Module):
    def __init__(self, inChannels, growRate, kSize=5, tranNum=8, ifIni=0, Smooth=False, use=False, dilation=1, bias=True):
        super(ConvSNP, self).__init__()
        Cin = inChannels
        G = growRate
        ifIni = ifIni
        if use:
            self.convSNP = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(inChannels, growRate, kernel_size=kSize, padding=(kSize - 1) // 2 * dilation, dilation=dilation),
            )
        else:
            self.convSNP = nn.Sequential(
                nn.ReLU(),
                fn.Fconv_PCA(kSize, Cin, G, tranNum=tranNum, inP=kSize, padding=(kSize - 1) // 2, ifIni=ifIni,
                             Smooth=Smooth),
            )
        self.lamda_ = nn.Parameter(torch.tensor(0.001), requires_grad=True)

    def forward(self, x):
        out = self.convSNP(x)
        out = out / (1 - self.lamda_)
        return out

    def clip_lambda(self):
        self.lambda_param.data = torch.clamp(self.lambda_param.data, 0, 1)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = ConvSNP(2, 1, kernel_size, bias=False, use=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return x


class SpatialAttentionBlock(nn.Module):
    def __init__(self):
        super(SpatialAttentionBlock, self).__init__()
        self.att1 = SpatialAttention()
        self.att2 = SpatialAttention()
    def forward(self, x1, x2):
        EPSILON = 1e-10
        att1 = self.att1(x1)
        att2 = self.att2(x2)
        # mask1 = self.sigmoid(att1)
        # mask2 = 1 - mask1
        mask1 = torch.exp(att1) / (torch.exp(att1) + torch.exp(att2) + EPSILON)
        mask2 = torch.exp(att2) / (torch.exp(att1) + torch.exp(att2) + EPSILON)
        x1_a = mask1 * x1
        x2_a = mask2 * x2
        return x1_a, x2_a


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Shallow features
        self.conv1 = ConvSNP(inChannels=3, growRate=64, kSize=3, use=True)

        # Deep features
        self.conv2 = ConvSNP(inChannels=64, growRate=64, kSize=3, use=True)
        self.conv3 = ConvSNP(inChannels=64, growRate=64, kSize=3, use=True)
        self.conv4 = ConvSNP(inChannels=64, growRate=64, kSize=3, use=True)
        self.conv5 = ConvSNP(inChannels=64, growRate=64, kSize=3, use=True)

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
        # self.drp = fn.Fconv_1X1(8 * 2, 8, 8, 0, Smooth=False)
        self.att = SpatialAttentionBlock()
        # self.drp = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.drp = ConvSNP(inChannels=128, growRate=64, kSize=1, use=True)
    def forward(self, x1, x2):
        x1, x2 = self.att(x1, x2)
        f = torch.cat([x1, x2], dim=1)
        f = self.drp(f)
        return f


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.r_conv1 = ConvSNP(inChannels=64, growRate=64, kSize=3, use=True)
        self.r_conv2 = ConvSNP(inChannels=64, growRate=64, kSize=3, use=True)
        self.r_conv3 = ConvSNP(inChannels=64, growRate=64, kSize=3, use=True)
        self.r_conv4 = ConvSNP(inChannels=64, growRate=64, kSize=3, use=True)
        self.r_conv5 = ConvSNP(inChannels=64, growRate=3,  kSize=3, use=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):

        x1 = self.r_conv1(x)
        x2 = self.r_conv2(x1)
        x3 = self.r_conv3(x2)
        x4 = self.r_conv4(x3)
        x5 = self.r_conv5(x4)
        output = self.sig(x5)
        return output

class BaseSNPNet(nn.Module):
    def __init__(self):
        super(BaseSNPNet, self).__init__()
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
    import numpy as np
    input1 = torch.rand((1, 3, 256, 256))
    input2 = torch.rand((1, 3, 256, 256))
    train_net = BaseSNPNet()

    print(train_net)
    out = train_net(input1, input2)
    print(out.shape)
    print(np.sum([p.numel() for p in train_net.parameters()]).item() / (1000 ** 2))
    print("DenseFuse have {} paramerters in total".format(sum(x.numel() for x in train_net.parameters())  / 1000 / 1000))
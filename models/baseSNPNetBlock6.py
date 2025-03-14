# -*- coding: utf-8 -*-
# @Time    : 2024/1/5 15:42
# @Author  : MorvanLi
# @Email   : morvanli1995@gmail.com
# @File    : baseSNPNet.py
# @Software: PyCharm

import torch.nn as nn
import torch
import F_conv as fn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import MyLibForSteerCNN as ML
# import scipy.io as sio
import math
from PIL import Image
from spikingjelly.clock_driven import base, surrogate

class Fconv_PCA(nn.Module):

    def __init__(self, sizeP, inNum, outNum, tranNum=8, inP=None, padding=None, ifIni=0, bias=True,
                 Smooth=True, iniScale=1.0, Bscale=1.0):

        super(Fconv_PCA, self).__init__()
        if inP == None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        Basis, Rank, weight = GetBasis_PCA(sizeP, tranNum, inP, Smooth=Smooth)
        self.register_buffer("Basis", Basis * Bscale)  # .cuda())
        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
        iniw = Getini_reg(Basis.size(3), inNum, outNum, self.expand, weight) * iniScale
        self.weights = nn.Parameter(iniw, requires_grad=True)
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        self.c = nn.Parameter(torch.zeros(1, outNum, 1, 1), requires_grad=bias)

    def forward(self, input):

        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)

            Num = tranNum // expand
            tempWList = [torch.cat(
                [tempW[:, i * Num:(i + 1) * Num, :, -i:, :, :], tempW[:, i * Num:(i + 1) * Num, :, :-i, :, :]], dim=3)
                for i in range(expand)]
            tempW = torch.cat(tempWList, dim=1)

            _filter = tempW.reshape([outNum * tranNum, inNum * self.expand, self.sizeP, self.sizeP])
            _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
        else:
            _filter = self.filter
            _bias = self.bias
        output = F.conv2d(input, _filter,
                          padding=self.padding,
                          dilation=1,
                          groups=1)
        return output + _bias

    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
                del self.bias
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)
            Num = tranNum // expand
            tempWList = [torch.cat(
                [tempW[:, i * Num:(i + 1) * Num, :, -i:, :, :], tempW[:, i * Num:(i + 1) * Num, :, :-i, :, :]], dim=3)
                for i in range(expand)]
            tempW = torch.cat(tempWList, dim=1)
            _filter = tempW.reshape([outNum * tranNum, inNum * self.expand, self.sizeP, self.sizeP])
            _bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])
            self.register_buffer("filter", _filter)
            self.register_buffer("bias", _bias)

        return super(Fconv_PCA, self).train(mode)


class Fconv_PCA_out(nn.Module):

    def __init__(self, sizeP, inNum, outNum, tranNum=8, inP=None, padding=None, ifIni=0, bias=True, Smooth=True,
                 iniScale=1.0):

        super(Fconv_PCA_out, self).__init__()
        if inP == None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        Basis, Rank, weight = GetBasis_PCA(sizeP, tranNum, inP, Smooth=Smooth)
        self.register_buffer("Basis", Basis)  # .cuda())

        iniw = Getini_reg(Basis.size(3), inNum, outNum, 1, weight) * iniScale
        self.weights = nn.Parameter(iniw, requires_grad=True)
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        self.c = nn.Parameter(torch.zeros(1, outNum, 1, 1), requires_grad=bias)

    def forward(self, input):

        if self.training:
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            tempW = torch.einsum('ijok,mnak->manoij', self.Basis, self.weights)
            _filter = tempW.reshape([outNum, inNum * tranNum, self.sizeP, self.sizeP])
        else:
            _filter = self.filter
        _bias = self.c
        output = F.conv2d(input, _filter,
                          padding=self.padding,
                          dilation=1,
                          groups=1)
        return output + _bias

    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            tempW = torch.einsum('ijok,mnak->manoij', self.Basis, self.weights)

            _filter = tempW.reshape([outNum, inNum * tranNum, self.sizeP, self.sizeP])
            self.register_buffer("filter", _filter)
        return super(Fconv_PCA_out, self).train(mode)


class Fconv_1X1(nn.Module):

    def __init__(self, inNum, outNum, tranNum=8, ifIni=0, bias=True, Smooth=True, iniScale=1.0):

        super(Fconv_1X1, self).__init__()

        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum

        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
        iniw = Getini_reg(1, inNum, outNum, self.expand) * iniScale
        self.weights = nn.Parameter(iniw, requires_grad=True)

        self.padding = 0
        self.bias = bias

        if bias:
            self.c = nn.Parameter(torch.zeros(1, outNum, 1, 1), requires_grad=True)
        else:
            self.c = torch.zeros(1, outNum, 1, 1)

    def forward(self, input):
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand
        tempW = self.weights.unsqueeze(4).unsqueeze(1).repeat([1, tranNum, 1, 1, 1, 1])

        Num = tranNum // expand
        tempWList = [
            torch.cat([tempW[:, i * Num:(i + 1) * Num, :, -i:, ...], tempW[:, i * Num:(i + 1) * Num, :, :-i, ...]],
                      dim=3) for i in range(expand)]
        tempW = torch.cat(tempWList, dim=1)

        _filter = tempW.reshape([outNum * tranNum, inNum * self.expand, 1, 1])

        bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])  # .cuda()

        output = F.conv2d(input, _filter,
                          padding=self.padding,
                          dilation=1,
                          groups=1)
        return output + bias


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size, tranNum=8, inP=None,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1, Smooth=True, iniScale=1.0):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(
                conv(kernel_size, n_feats, n_feats, tranNum=tranNum, inP=inP, padding=(kernel_size - 1) // 2, bias=bias,
                     Smooth=Smooth, iniScale=iniScale))
            if bn:
                m.append(F_BN(n_feats, tranNum))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


def Getini_reg(nNum, inNum, outNum, expand, weight=1):
    A = (np.random.rand(outNum, inNum, expand, nNum) - 0.5) * 2 * 2.4495 / np.sqrt((inNum) * nNum) * np.expand_dims(
        np.expand_dims(np.expand_dims(weight, axis=0), axis=0), axis=0)
    return torch.FloatTensor(A)


def GetBasis_PCA(sizeP, tranNum=8, inP=None, Smooth=False):
    if inP == None:
        inP = sizeP
    inX, inY, Mask = MaskC(sizeP)
    X0 = np.expand_dims(inX, 2)
    Y0 = np.expand_dims(inY, 2)
    Mask = np.expand_dims(Mask, 2)
    theta = np.arange(tranNum) / tranNum * 2 * np.pi
    theta = np.expand_dims(np.expand_dims(theta, axis=0), axis=0)
    #    theta = torch.FloatTensor(theta)
    X = np.cos(theta) * X0 - np.sin(theta) * Y0
    Y = np.cos(theta) * Y0 + np.sin(theta) * X0
    #    X = X.unsqueeze(3).unsqueeze(4)
    X = np.expand_dims(np.expand_dims(X, 3), 4)
    Y = np.expand_dims(np.expand_dims(Y, 3), 4)
    v = np.pi / inP * (inP - 1)
    p = inP / 2

    k = np.reshape(np.arange(inP), [1, 1, 1, inP, 1])
    l = np.reshape(np.arange(inP), [1, 1, 1, 1, inP])

    BasisC = np.cos((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y)
    BasisS = np.sin((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y)

    BasisC = np.reshape(BasisC, [sizeP, sizeP, tranNum, inP * inP]) * np.expand_dims(Mask, 3)
    BasisS = np.reshape(BasisS, [sizeP, sizeP, tranNum, inP * inP]) * np.expand_dims(Mask, 3)

    BasisC = np.reshape(BasisC, [sizeP * sizeP * tranNum, inP * inP])
    BasisS = np.reshape(BasisS, [sizeP * sizeP * tranNum, inP * inP])

    BasisR = np.concatenate((BasisC, BasisS), axis=1)

    U, S, VT = np.linalg.svd(np.matmul(BasisR.T, BasisR))

    Rank = np.sum(S > 0.0001)
    BasisR = np.matmul(np.matmul(BasisR, U[:, :Rank]), np.diag(1 / np.sqrt(S[:Rank] + 0.0000000001)))
    BasisR = np.reshape(BasisR, [sizeP, sizeP, tranNum, Rank])

    temp = np.reshape(BasisR, [sizeP * sizeP, tranNum, Rank])
    var = (np.std(np.sum(temp, axis=0) ** 2, axis=0) + np.std(np.sum(temp ** 2 * sizeP * sizeP, axis=0),
                                                              axis=0)) / np.mean(
        np.sum(temp, axis=0) ** 2 + np.sum(temp ** 2 * sizeP * sizeP, axis=0), axis=0)
    Trod = 1
    Ind = var < Trod
    Rank = np.sum(Ind)
    Weight = 1 / np.maximum(var, 0.04) / 25
    if Smooth:
        BasisR = np.expand_dims(np.expand_dims(np.expand_dims(Weight, 0), 0), 0) * BasisR

    return torch.FloatTensor(BasisR), Rank, Weight


def MaskC(SizeP):
    p = (SizeP - 1) / 2
    x = np.arange(-p, p + 1) / p
    X, Y = np.meshgrid(x, x)
    C = X ** 2 + Y ** 2

    Mask = np.ones([SizeP, SizeP])
    #        Mask[C>(1+1/(4*p))**2]=0
    Mask = np.exp(-np.maximum(C - 1, 0) / 0.2)

    return X, Y, Mask


class PointwiseAvgPoolAntialiased(nn.Module):

    def __init__(self, sizeF, stride, padding=None):
        super(PointwiseAvgPoolAntialiased, self).__init__()
        sigma = (sizeF - 1) / 2 / 3
        self.kernel_size = (sizeF, sizeF)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride

        if padding is None:
            padding = int((sizeF - 1) // 2)

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        # Build the Gaussian smoothing filter
        grid_x = torch.arange(sizeF).repeat(sizeF).view(sizeF, sizeF)
        grid_y = grid_x.t()
        grid = torch.stack([grid_x, grid_y], dim=-1)
        mean = (sizeF - 1) / 2.
        variance = sigma ** 2.
        r = -torch.sum((grid - mean) ** 2., dim=-1, dtype=torch.get_default_dtype())
        _filter = torch.exp(r / (2 * variance))
        _filter /= torch.sum(_filter)
        _filter = _filter.view(1, 1, sizeF, sizeF)
        self.filter = nn.Parameter(_filter, requires_grad=False)
        # self.register_buffer("filter", _filter)

    def forward(self, input):
        _filter = self.filter.repeat((input.shape[1], 1, 1, 1))
        output = F.conv2d(input, _filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return output


class F_BN(nn.Module):
    def __init__(self, channels, tranNum=8):
        super(F_BN, self).__init__()
        self.BN = nn.BatchNorm2d(channels)
        self.tranNum = tranNum

    def forward(self, X):
        X = self.BN(X.reshape([X.size(0), int(X.size(1) / self.tranNum), self.tranNum * X.size(2), X.size(3)]))
        return X.reshape([X.size(0), self.tranNum * X.size(1), int(X.size(2) / self.tranNum), X.size(3)])


class F_Dropout(nn.Module):
    def __init__(self, zero_prob=0.5, tranNum=8):
        # nn.Dropout2d
        self.tranNum = tranNum
        super(F_Dropout, self).__init__()
        self.Dropout = nn.Dropout2d(zero_prob)

    def forward(self, X):
        X = self.Dropout(X.reshape([X.size(0), int(X.size(1) / self.tranNum), self.tranNum * X.size(2), X.size(3)]))
        return X.reshape([X.size(0), self.tranNum * X.size(1), int(X.size(2) / self.tranNum), X.size(3)])


def build_mask(s, margin=2, dtype=torch.float32):
    mask = torch.zeros(1, 1, s, s, dtype=dtype)
    c = (s - 1) / 2
    t = (c - margin / 100. * c) ** 2
    sig = 2.
    for x in range(s):
        for y in range(s):
            r = (x - c) ** 2 + (y - c) ** 2
            if r > t:
                mask[..., x, y] = math.exp((t - r) / sig ** 2)
            else:
                mask[..., x, y] = 1.
    return mask


class MaskModule(nn.Module):

    def __init__(self, S: int, margin: float = 0.):
        super(MaskModule, self).__init__()

        self.margin = margin
        self.mask = torch.nn.Parameter(build_mask(S, margin=margin), requires_grad=False)

    def forward(self, input):
        assert input.shape[2:] == self.mask.shape[2:]

        out = input * self.mask
        return out


class GroupPooling(nn.Module):
    def __init__(self, tranNum):
        super(GroupPooling, self).__init__()
        self.tranNum = tranNum

    def forward(self, input):
        output = input.reshape([input.size(0), -1, self.tranNum, input.size(2), input.size(3)])
        output = torch.max(output, 2).values
        return output


class GroupMeanPooling(nn.Module):
    def __init__(self, tranNum):
        super(GroupMeanPooling, self).__init__()
        self.tranNum = tranNum

    def forward(self, input):
        output = input.reshape([input.size(0), -1, self.tranNum, input.size(2), input.size(3)])
        output = torch.mean(output, 2)
        return output


class Fconv(nn.Module):

    def __init__(self, sizeP, inNum, outNum, tranNum=8, inP=None, padding=None, ifIni=0, bias=True):

        super(Fconv, self).__init__()
        if inP == None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        BasisC, BasisS = GetBasis(sizeP, tranNum, inP)
        self.register_buffer("Basis", torch.cat([BasisC, BasisS], 3))  # .cuda())

        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
        #        iniw = torch.randn(outNum, inNum, self.expand, self.Basis.size(3))*0.03
        iniw = Getini(inP, inNum, outNum, self.expand)
        self.weights = nn.Parameter(iniw, requires_grad=True)
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding

        if bias:
            self.c = nn.Parameter(torch.zeros(1, outNum, 1, 1), requires_grad=True)
        else:
            self.c = torch.zeros(1, outNum, 1, 1)
        self.register_buffer("filter", torch.zeros(outNum * tranNum, inNum * self.expand, sizeP, sizeP))

    def forward(self, input):
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand
        tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)

        for i in range(expand):
            ind = np.hstack((np.arange(expand - i, expand), np.arange(expand - i)))
            tempW[:, i, :, :, ...] = tempW[:, i, :, ind, ...]
        _filter = tempW.reshape([outNum * tranNum, inNum * self.expand, self.sizeP, self.sizeP])

        #        sio.savemat('Filter2.mat', {'filter': _filter.cpu().detach().numpy()})
        bias = self.c.repeat([1, 1, tranNum, 1]).reshape([1, outNum * tranNum, 1, 1])

        output = F.conv2d(input, _filter,
                          padding=self.padding,
                          dilation=1,
                          groups=1)
        return output + bias


class FCNN_reg(nn.Module):

    def __init__(self, sizeP, inNum, outNum, Basisin, tranNum=8, inP=None, padding=None, ifIni=0, bias=True):

        super(FCNN_reg, self).__init__()
        #        self.sampled_basis = Basisin.cuda()
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        self.register_buffer("Basis", Basisin.cuda())

        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
        iniw = torch.randn(outNum, inNum, self.expand, self.Basis.size(3)) * 0.03
        self.weights = nn.Parameter(iniw, requires_grad=True)
        self.padding = padding

    def forward(self, input):

        #        _filter = self._filter
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand
        tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)

        for i in range(expand):
            ind = np.hstack((np.arange(expand - i, expand), np.arange(expand - i)))
            tempW[:, i, :, :, ...] = tempW[:, i, :, ind, ...]
        _filter = tempW.reshape([outNum * tranNum, inNum * self.expand, self.sizeP, self.sizeP])

        output = F.conv2d(input, _filter,
                          padding=self.padding,
                          dilation=1,
                          groups=1)

        return output


def Getini(sizeP, inNum, outNum, expand):
    inX, inY, Mask = MaskC(sizeP)
    X0 = np.expand_dims(inX, 2)
    Y0 = np.expand_dims(inY, 2)
    X0 = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(X0, 0), 0), 4), 0)
    y = Y0[:, 1]
    y = np.expand_dims(np.expand_dims(np.expand_dims(np.expand_dims(y, 0), 0), 3), 0)

    orlW = np.zeros([outNum, inNum, expand, sizeP, sizeP, 1, 1])
    for i in range(outNum):
        for j in range(inNum):
            for k in range(expand):
                temp = np.array(
                    Image.fromarray(((np.random.randn(3, 3)) * 2.4495 / np.sqrt((inNum) * sizeP * sizeP))).resize(
                        (sizeP, sizeP)))
                orlW[i, j, k, :, :, 0, 0] = temp

    v = np.pi / sizeP * (sizeP - 1)
    k = np.reshape((np.arange(sizeP)), [1, 1, 1, 1, 1, sizeP, 1])
    l = np.reshape((np.arange(sizeP)), [1, 1, 1, 1, 1, sizeP])

    tempA = np.sum(np.cos(k * v * X0) * orlW, 4) / sizeP
    tempB = -np.sum(np.sin(k * v * X0) * orlW, 4) / sizeP
    A = np.sum(np.cos(l * v * y) * tempA + np.sin(l * v * y) * tempB, 3) / sizeP
    B = np.sum(np.cos(l * v * y) * tempB - np.sin(l * v * y) * tempA, 3) / sizeP
    A = np.reshape(A, [outNum, inNum, expand, sizeP * sizeP])
    B = np.reshape(B, [outNum, inNum, expand, sizeP * sizeP])
    iniW = np.concatenate((A, B), axis=3)
    return torch.FloatTensor(iniW)


def GetBasis(sizeP, tranNum=8, inP=None):
    if inP == None:
        inP = sizeP
    inX, inY, Mask = MaskC(sizeP)
    X0 = np.expand_dims(inX, 2)
    Y0 = np.expand_dims(inY, 2)
    Mask = np.expand_dims(Mask, 2)
    theta = np.arange(tranNum) / tranNum * 2 * np.pi
    theta = np.expand_dims(np.expand_dims(theta, axis=0), axis=0)
    #    theta = torch.FloatTensor(theta)
    X = np.cos(theta) * X0 - np.sin(theta) * Y0
    Y = np.cos(theta) * Y0 + np.sin(theta) * X0
    #    X = X.unsqueeze(3).unsqueeze(4)
    X = np.expand_dims(np.expand_dims(X, 3), 4)
    Y = np.expand_dims(np.expand_dims(Y, 3), 4)
    v = np.pi / inP * (inP - 1)
    p = inP / 2

    k = np.reshape(np.arange(inP), [1, 1, 1, inP, 1])
    l = np.reshape(np.arange(inP), [1, 1, 1, 1, inP])

    BasisC = np.cos((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y)
    BasisS = np.sin((k - inP * (k > p)) * v * X + (l - inP * (l > p)) * v * Y)

    BasisC = np.reshape(BasisC, [sizeP, sizeP, tranNum, inP * inP]) * np.expand_dims(Mask, 3)
    BasisS = np.reshape(BasisS, [sizeP, sizeP, tranNum, inP * inP]) * np.expand_dims(Mask, 3)
    return torch.FloatTensor(BasisC), torch.FloatTensor(BasisS)



###
class SNPNeuron(base.MemoryModule):
    def __init__(self, tau_u: float, tau_e: float,
                 in_channels, out_channels: int, kernel_size: Tuple[int, ...], stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 batch_size: int, shape: Tuple[int, ...],
                 local: bool = True, shared: bool = True, self_feedback: bool = True, multi_channel: bool = False,
                 learnable: bool = True,
                 linking: bool = True, encoding: bool = True):

        super().__init__()

        if local and shared and self_feedback and not multi_channel and learnable:
            self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)  # 参数应该从外面传进来，暂时这样写
        elif not local and not shared and self_feedback and not multi_channel and learnable:
            self.op = nn.Linear(*shape, *shape, bias=False)

        self.linking = linking
        self.encoding = encoding

        self.tau_u = nn.Parameter(1 / torch.tensor([tau_u], dtype=torch.float))

        self.tau_e = nn.Parameter(1 / torch.tensor([tau_e], dtype=torch.float))

        self.surrogate_function = SG.apply
        # self.surrogate_function = surrogate.ATan()

        self.register_memory('u', torch.zeros(batch_size, *shape))
        self.register_memory('tau', torch.ones(batch_size, *shape))
        self.register_memory('p', torch.zeros(batch_size, *shape))

    def forward(self, x: torch.Tensor):

        self.u = self.u * (1.0 - self.tau_u) + x + self.op(self.p)

        self.tau = self.tau * (1.0 - self.tau_e) + self.p

        self.p = self.surrogate_function(self.u - self.tau, 1.0)
        return self.p


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
        self.lamda_.data = torch.clamp(self.lamda_.data, 0, 1)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = ConvSNP(2, 1, kernel_size, bias=False, use=True)
        # self.sigmoid = nn.Sigmoid()

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
        self.conv6 = ConvSNP(inChannels=64, growRate=64, kSize=3, use=True)
        # self.conv7 = ConvSNP(inChannels=64, growRate=64, kSize=3, use=True)

    def forward(self, x1):
        F1 = self.conv1(x1)
        F2 = self.conv2(F1)
        F3 = self.conv3(F2)
        F4 = self.conv4(F3)
        F5 = self.conv5(F4)
        F6 = self.conv6(F5)
        # F7 = self.conv6(F6)
        return F6


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
        self.r_conv5 = ConvSNP(inChannels=64, growRate=64,  kSize=3, use=True)
        self.r_conv6 = ConvSNP(inChannels=64, growRate=3, kSize=3, use=True)
        # self.r_conv7 = ConvSNP(inChannels=64, growRate=3, kSize=3, use=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):

        x1 = self.r_conv1(x)
        x2 = self.r_conv2(x1)
        x3 = self.r_conv3(x2)
        x4 = self.r_conv4(x3)
        x5 = self.r_conv5(x4)
        x6 = self.r_conv6(x5)
        # x7 = self.r_conv7(x6)
        output = self.sig(x6)
        return output

class BaseSNPNet6(nn.Module):
    def __init__(self):
        super(BaseSNPNet6, self).__init__()
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
    # import numpy as np
    # from thop import profile
    # device = "cpu"
    # input1 = torch.rand((1, 3, 520, 520)).to(device)
    # input2 = torch.rand((1, 3, 520, 520)).to(device)
    # train_net = BaseSNPNet6().to(device)
    # # print(train_net)
    # out = train_net(input1, input2)
    # print(out.shape)
    # print(np.sum([p.numel() for p in train_net.parameters()]).item() / (1000 ** 2))
    # print("DenseFuse have {} paramerters in total".format(sum(x.numel() for x in train_net.parameters())  / 1000 / 1000))
    # flops, params = profile(train_net, inputs=(input1,input2))
    # print(flops / 1e9, params / 1e6)  # flops单位G，para单位M
    import kornia
    input1 = torch.rand(1, 3, 5, 5)
    input2 = torch.rand(1, 3, 5, 5)
    criterion = kornia.losses.MS_SSIMLoss()
    loss = criterion(input1, input2)
    print(loss)
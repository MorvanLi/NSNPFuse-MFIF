# -*- coding: utf-8 -*-
# @Time    : 2024/10/28 19:48
# @Author  : MorvanLi
# @Email   : morvanli1995@gmail.com
# @File    : mainBaseNetfusion.py
# @Software: PyCharm

import argparse
from baseNet import BaseNet
from baseFconv import BaseFconvNet
from baseSNPNet import BaseSNPNet
from torch.utils.data import DataLoader
from torchvision import transforms
from config2 import  Config2Net
from baseSNPNetBlock6 import BaseSNPNet6
import time
import torch
print(torch.__version__)
# import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from dataloader import DataTest


def img_save(image,imagename,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    imsave(os.path.join(savepath, "{}.png".format(imagename)),image)

def fusion(test_dataloader, args):
    device = args.device
    net = BaseSNPNet6().to(device)
    net.load_state_dict(torch.load(args.saveModel_dir, map_location="cuda:0"))
    net.eval()
    # print(net)
    begin_time = time.time()
    with torch.no_grad():
        for i_test, (image1, image2) in enumerate(test_dataloader):
            print(i_test)
            image1 = torch.FloatTensor(image1).to(device).permute(0, 3, 1, 2)
            image2 = torch.FloatTensor(image2).to(device).permute(0, 3, 1, 2)
            data_Fuse = net(image1, image2)

            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
            fi = np.squeeze(((data_Fuse).clamp(0, 1) * 255).cpu().numpy().astype(np.uint8))
            fi = np.transpose(fi, (1, 2, 0))  # 转换维度顺序
            img_save(fi, str(i_test+1), "./results/Lytro/")


    proc_time = time.time() - begin_time
    print("Model have {} paramerters in total".format(sum(x.numel() for x in net.parameters()) / 1000 / 1000))
    print(f'Avg processing time of {proc_time/20} ')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_ch", type=int, default=3)
    parser.add_argument("--out_ch", type=int, default=64)
    parser.add_argument("--n_resblocks", type=int, default=3)
    parser.add_argument("--n_convs", type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--testData_dir', type=str, default="./lytro")
    parser.add_argument('--saveModel_dir', type=str, default='./weights/best-475.pth')
    parser.add_argument('--result', type=str, default='result')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    transforms_ = transforms.ToTensor()
    test_set = DataTest(testData_dir=args.testData_dir, transforms_=transforms_)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    fusion(test_dataloader, args)
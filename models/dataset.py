# -*- coding: utf-8 -*-
# @Time    : 2024/5/6 15:10
# @Author  : MorvanLi
# @Email   : morvanli1995@gmail.com
# @File    : dataset.py
# @Software: PyCharm

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random


# ########################### Data Augmentation ###########################
# class Normalize(object):
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std
#
#     def __call__(self, image1, image2, label, body=None, detail=None):
#         # image1 = (image1 - self.mean) / self.std
#         # image2 = (image2 - self.mean) / self.std
#         return image1/255, image2/255, label/255

# class RandomCrop(object):
#     def __call__(self, near, far, gt, detail=None):
#         H,W,_   = near.shape
#         randw   = np.random.randint(W/8)
#         randh   = np.random.randint(H/8)
#         offseth = 0 if randh == 0 else np.random.randint(randh)
#         offsetw = 0 if randw == 0 else np.random.randint(randw)
#         p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
#
#         return near[p0:p1,p2:p3, :], far[p0:p1,p2:p3], gt[p0:p1,p2:p3]
#
# class RandomFlip(object):
#     def __call__(self, image, mask=None, body=None, detail=None):
#         if np.random.randint(2)==0:
#             if mask is None:
#                 return image[:,::-1,:].copy()
#             return image[:,::-1,:].copy(), mask[:, ::-1].copy(), body[:, ::-1].copy()
#         else:
#             if mask is None:
#                 return image
#             return image, mask, body
#
# class Resize(object):
#     def __init__(self, H, W):
#         self.H = H
#         self.W = W
#
#     def __call__(self, image, mask, body, detail=None):
#         image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
#         mask  = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
#         body  = cv2.resize(body, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
#         # detail= cv2.resize( detail, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
#         return image, mask, body
#
# class ToTensor(object):
#     def __call__(self, image1, image2, label, body=None, detail=None):
#         image1 = torch.from_numpy(image1)
#         image1 = image1.permute(2, 0, 1)
#
#         image2 = torch.from_numpy(image2)
#         image2 = image2.permute(2, 0, 1)
#
#         label = torch.from_numpy(label)
#         label = label.permute(2, 0, 1)
#         return image1, image2, label
#
#
# ########################### Dataset Class ###########################
# class Data(Dataset):
#     def __init__(self,dataset_dir):
#         # self.normalize = Normalize(mean=np.array([[[124.55, 118.90, 102.94]]]), std=np.array([[[ 56.77,  55.97,  57.50]]]))
#         self.randomcrop = RandomCrop()
#         self.randomflip = RandomFlip()
#         self.resize = Resize(352, 352)
#         self.totensor = ToTensor()
#
#         self.far_list,self.next_list, self.gt = self.get_MFF()
#
#     def __getitem__(self, idx):
#
#         near = cv2.imread(self.next_list[idx]).astype(np.float32)/255 # '.jpg')[:,:,::-1].astype(np.float32)
#         far = cv2.imread(self.far_list[idx]).astype(np.float32)/255
#         gt = cv2.imread(self.gt[idx]).astype(np.float32)/255
#
#
#         near, far, gt = self.randomcrop(near, far, gt)
#         near, far, gt = self.randomflip(near, far, gt)
#         near, far, gt = self.resize(near, far, gt)
#         near, far, gt = self.totensor(near, far, gt)
#         # image1 = image1.astype(np.float32)
#         # image2 = image2.astype(np.float32)
#
#
#         return near, far, gt
#
#     def __len__(self):
#         return len(self.far_list)
#
#     def get_MFF(self):
#         far_list = []
#         next_list = []
#         gt_list = []
#         dataset_dir = "./RealMFF/"
#         far_dir = os.path.join(dataset_dir, "imageB")
#         next_dir = os.path.join(dataset_dir, "imageA")
#         gt_dir = os.path.join(dataset_dir, "Fusion")
#         for path in os.listdir(next_dir):
#             # check if current path is a file
#             if os.path.isfile(os.path.join(next_dir, path)):
#                 next_list.append(os.path.join(next_dir, path))
#                 temp0 = path.split("_")[0]
#                 temp = temp0 + "_B.png"
#                 fusion = temp0 + "_F.png"
#                 far_list.append(os.path.join(far_dir, temp))
#                 gt_list.append(os.path.join(gt_dir, fusion))
#
#         return next_list, far_list, gt_list
#
#     def collate(self, batch):
#         image1, image2, label = [list(item) for item in zip(*batch)]
#         image1 = torch.from_numpy(np.stack(image1, axis=0))
#         image2 = torch.from_numpy(np.stack(image2, axis=0))
#         label = torch.from_numpy(np.stack(label, axis=0))
#         return image1, image2, label
#
# if __name__ == '__main__':
#     from torch.utils.data import DataLoader
#     data = Data(dataset_dir="./DataSet/TrainSet")
#     loader = DataLoader(data, collate_fn=data.collate, batch_size=2, shuffle=True, pin_memory=True,
#                         num_workers=6)
#     for i in loader:
#         print(i[0].shape, "-----", i[1].shape)


























########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image1, image2, gt, body=None, detail=None):
        image1 = (image1 - self.mean) / self.std
        image2 = (image2 - self.mean) / self.std
        gt = gt
        return image1, image2, gt

class RandomCrop(object):
    def __call__(self, near, far, gt, detail=None):
        H,W,_   = near.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw

        return near[p0:p1,p2:p3, :], far[p0:p1,p2:p3], gt[p0:p1,p2:p3]

class RandomFlip(object):
    def __call__(self, image, mask=None, body=None, detail=None):
        if np.random.randint(2)==0:
            if mask is None:
                return image[:,::-1,:].copy()
            return image[:,::-1,:].copy(), mask[:, ::-1].copy(), body[:, ::-1].copy()
        else:
            if mask is None:
                return image
            return image, mask, body

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, body, detail=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        body  = cv2.resize(body, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        # detail= cv2.resize( detail, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, body

class ToTensor(object):
    def __call__(self, image1, image2, label, body=None, detail=None):
        image1 = torch.from_numpy(image1)
        image1 = image1.permute(2, 0, 1)

        image2 = torch.from_numpy(image2)
        image2 = image2.permute(2, 0, 1)

        label = torch.from_numpy(label)
        label = label.permute(2, 0, 1)
        return image1, image2, label

# class ToTensor(object):
#     def __call__(self, image1, image2, label, body=None, detail=None):
#         image1 = torch.from_numpy(image1)
#         image1 = image1.permute(2, 0, 1)
#
#         image2 = torch.from_numpy(image2)
#         image2 = image2.permute(2, 0, 1)
#
#         label = torch.from_numpy(label)
#         return image1, image2, label


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self,dataset_dir):
        self.normalize = Normalize(mean=np.array([[[124.55, 118.90, 102.94]]]), std=np.array([[[ 56.77,  55.97,  57.50]]]))
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize = Resize(256, 256)
        self.totensor = ToTensor()
        self.dataset_dir = dataset_dir
        self.file_list = os.listdir(self.dataset_dir)

    def __getitem__(self, idx):
        temp_dir = os.listdir(os.path.join(self.dataset_dir, self.file_list[idx]))
        temp_idx = random.randint(0, 3)
        chird_dir = temp_dir[temp_idx]

        image1 = cv2.imread(self.dataset_dir + '/' + self.file_list[idx] + "/" + chird_dir + "/" + self.file_list[idx] + "_1.jpg").astype(np.float32)/255.
        image2 = cv2.imread(self.dataset_dir + '/' + self.file_list[idx] + "/" + chird_dir + "/" + self.file_list[idx] + "_2.jpg").astype(np.float32)/255.
        label = cv2.imread(self.dataset_dir + '/' + self.file_list[idx] + "/" + chird_dir + "/" + self.file_list[idx] + "_ground.jpg").astype(np.float32)/255.

        # near, far, gt = self.normalize(image1, image2, label)
        # near = near.astype(np.float32)
        # far = far.astype(np.float32)
        near, far, gt = self.randomcrop(image1, image2, label)
        near, far, gt = self.randomflip(near, far, gt)
        near, far, gt = self.resize(near, far, gt)
        near, far, gt = self.totensor(near, far, gt)


        return near, far, gt

    def __len__(self):
        return len(self.file_list)

    def collate(self, batch):
        image1, image2, label = [list(item) for item in zip(*batch)]
        image1 = torch.from_numpy(np.stack(image1, axis=0))
        image2 = torch.from_numpy(np.stack(image2, axis=0))
        label = torch.from_numpy(np.stack(label, axis=0))
        return image1, image2, label

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data = Data(dataset_dir="./DataSet/TrainSet")
    loader = DataLoader(data, collate_fn=data.collate, batch_size=1, shuffle=True, pin_memory=True,
                        num_workers=6)
    for i in loader:
        print(i)
# -*- coding: utf-8 -*-
# @Time    : 2023/10/19 15:26
# @Author  : MorvanLi
# @Email   : morvanli1995@gmail.com
# @File    : dataloader.py
# @Software: PyCharm


from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import random

class CustomDataset(Dataset):
    def __init__(self, dataset_dir, transforms_,  rgb=True):
        self.dataset_dir = dataset_dir
        self.file_list = os.listdir(self.dataset_dir)
        self.transform = transforms.Compose(transforms_)
        self.rgb = rgb

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        temp_dir = os.listdir(os.path.join(self.dataset_dir, self.file_list[item]))
        temp_idx = random.randint(0, 3)
        chird_dir = temp_dir[temp_idx]
        if self.rgb == True:
            img1 = Image.open(
                self.dataset_dir + '/' + self.file_list[item] + "/" + chird_dir + "/" + self.file_list[item] + "_1.jpg")
            img2 = Image.open(
                self.dataset_dir + '/' + self.file_list[item] + "/" + chird_dir + "/" + self.file_list[item] + "_2.jpg")
            label = Image.open(self.dataset_dir + '/' + self.file_list[item] + "/" + chird_dir + "/" + self.file_list[
                item] + "_ground.jpg")
        else:
            img1 = Image.open(self.dataset_dir + '/' + self.file_list[item] + "/" + chird_dir + "/" + self.file_list[
                item] + "_1.jpg").convert('L')
            img2 = Image.open(self.dataset_dir + '/' + self.file_list[item] + "/" + chird_dir + "/" + self.file_list[
                item] + "_2.jpg").convert('L')
            label = Image.open(self.dataset_dir + '/' + self.file_list[item] + "/" + chird_dir + "/" + self.file_list[
                item] + "_ground.jpg").convert('L')
        # 裁剪为256*256
        img1 = img1.resize((256, 256), Image.BICUBIC)
        img2 = img2.resize((256, 256), Image.BICUBIC)
        label = label.resize((256, 256), Image.BICUBIC)
        # 水平翻转
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        label = self.transform(label)
        return img1, img2, label

# class DataTest(Dataset):
#     def __init__(self, testData_dir, transforms_):
#         self.testData_dir = testData_dir
#         self.file_list = os.listdir(os.path.join(testData_dir, "far"))
#         self.file_list = sorted(self.file_list)
#         self.transform = transforms_
#
#     def __getitem__(self, idx):
#
#         path1 = self.testData_dir + "/" + "far/" + f"{idx + 1}.jpg"
#         path2 = self.testData_dir + "/" + "near/" + f"{idx + 1}.jpg"
#
#         if os.path.isfile(path1):
#             image1 = Image.open(self.testData_dir + "/" + "far/" + f"{idx + 1}.jpg")
#             image2 = Image.open(self.testData_dir + "/" + "near/" + f"{idx + 1}.jpg")
#         else:
#             image1 = Image.open(self.testData_dir + "/" + "far/" + f"{idx + 1}.png")
#             image2 = Image.open(self.testData_dir + "/" + "near/" + f"{idx + 1}.png")
#
#         if image1.mode != "RGB":
#             img1 = image1.convert("RGB")
#             img2 = image2.convert("RGB")
#         else:
#             img1 = image1
#             img2 = image2
#         img1 = self.transform(img1)
#         img2 = self.transform(img2)
#         return img1, img2
#
#     def __len__(self):
#         return len(self.file_list)

import cv2
import numpy as np
def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img
class DataTest(Dataset):
    def __init__(self, testData_dir, transforms_):
        self.testData_dir = testData_dir
        self.file_list = os.listdir(os.path.join(testData_dir, "far"))
        self.file_list = sorted(self.file_list)
        self.transform = transforms_

    def __getitem__(self, idx):

        path1 = self.testData_dir + "/" + "far/" + f"{idx + 1}.jpg"
        path2 = self.testData_dir + "/" + "near/" + f"{idx + 1}.jpg"

        if os.path.isfile(path1):
            image1 = image_read_cv2(self.testData_dir + "/" + "far/" + f"{idx + 1}.jpg", mode="RGB")/255.0
            image2 = image_read_cv2(self.testData_dir + "/" + "near/" + f"{idx + 1}.jpg", mode="RGB")/255.0
        else:
            image1 = image_read_cv2(self.testData_dir + "/" + "far/" + f"{idx + 1}.png", mode="RGB")/255.0
            image2 = image_read_cv2(self.testData_dir + "/" + "near/" + f"{idx + 1}.png", mode="RGB")/255.0

        # if image1.mode != "RGB":
        #     img1 = image1.convert("RGB")
        #     img2 = image2.convert("RGB")
        # else:
        #     img1 = image1
        #     img2 = image2
        # img1 = self.transform(img1)
        # img2 = self.transform(img2)
        return image1, image2

    def __len__(self):
        return len(self.file_list)



IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif'
]
class IVIF(Dataset):
    def __init__(self, impath1, impath2, mode='RGB', transform=None):
        self.impath1 = impath1
        self.impath2 = impath2
        self.mode = mode
        self.transform = transform

    def loader(self, path):

        return Image.open(path).convert(self.mode)

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def get_pair(self):
        if self.is_image_file(self.impath1):
            img1 = self.loader(self.impath1)
        if self.is_image_file(self.impath2):
            img2 = self.loader(self.impath2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2

    def get_source(self):
        if self.is_image_file(self.impath1):
            img1 = self.loader(self.impath1)
        if self.is_image_file(self.impath2):
            img2 = self.loader(self.impath2)
        return img1, img2


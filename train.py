import os.path
import datetime
from tqdm import tqdm
import torchvision.transforms as transforms
import torch
from torch.optim import lr_scheduler, Adam
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim
import numpy as np
import random
# from network import SNPFuseNet, initialize_weights, ConvSNP
from dataloader import CustomDataset
from PIL import Image
import kornia
import argparse
from loss import LpLssimLoss
from tensorboardX import SummaryWriter
import scipy.io
# from baseNet import BaseNet   #### Baseline
# from baseFconv import BaseFconvNet
from baseSNPNet import BaseSNPNet
from baseSNPNetBlock6 import BaseSNPNet6, initialize_weights, ConvSNP
from ssimL1Loss import MS_SSIM_L1_LOSS


import dataset

from config2 import Config2Net
def training_setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

list_content = []
def main(train_loder, args):
    device = args.device
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    model_dir = args.saveModel_dir
    writer1 = SummaryWriter(log_dir="log/loss")
    # model
    net = BaseSNPNet6().to(device)
    initialize_weights(net)

    ssim_loss = LpLssimLoss().to(device)
    l2_loss = torch.nn.MSELoss().to(device)



    # optimize
    optimizer = optim.Adam(net.parameters(), 0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)

    best_loss = 100000.0
    # train process
    running_loss = 0.
    for epoch in range(0, n_epochs):
        net.train()
        train_bar = tqdm(train_loder)
        for i, (image1, image2, label) in enumerate(train_bar):
            image1 = image1.to(device)
            image2 = image2.to(device)
            label = label.to(device)
            out = net(image1, image2)
            loss = 0.2 * l2_loss(label, out) + 0.8 * ssim_loss(label, out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ## train SNP need this code
            for layer in net.modules():
                if isinstance(layer, ConvSNP):
                    # print("yes")
                    layer.clip_lambda()

            # for layer in net.children():
            #     if isinstance(layer, ConvSNP):
            #         print("sss")
            #         layer.clip_lambda()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     n_epochs,
                                                                     loss)
            # train_bar.set_description("[epoch: %3d/%3d, batch: %5d/%5d] train loss: %8f " % (
            #     epoch + 1, n_epochs, (i + 1) * batch_size, train_num, loss.item()))

        writer1.add_scalar('train loss', running_loss, epoch)
        if (running_loss <
                best_loss):
            best_loss = running_loss
            torch.save(net.state_dict(), model_dir + "best.pth" )
            print(f"The best pth is: {best_loss}, at: {epoch}")

        list_content.append(running_loss)
        running_loss = 0.0
        scheduler.step()

    print('-------------Congratulations! Training Done!!!-------------')
    scipy.io.savemat('data.mat', {'data': list_content})

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_ch", type=int, default=3, help='rgb is 3,gray is 1')
    parser.add_argument("--out_ch", type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=10, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--dataset_dir', type=str, default="./TrainSet")
    parser.add_argument('--saveModel_dir', type=str, default='./weights/')
    return parser.parse_args()


if __name__ == '__main__':
    training_setup_seed(42)
    args = parse_args()
    transforms_ = [transforms.Resize((256, 256), Image.BICUBIC),
                   # transforms.RandomHorizontalFlip(p=0.6),
                   transforms.ToTensor()]
    train_set = dataset.Data(dataset_dir=args.dataset_dir)
    # train_set = CustomDataset(dataset_dir=args.dataset_dir, transforms_=transforms_, rgb=True)
    custom_dataloader = DataLoader(train_set, collate_fn=train_set.collate,batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=6)
    main(custom_dataloader, args)


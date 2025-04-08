import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
import datetime

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import joblib
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from dataset.dataset import Dataset2D

from utils.metrics import dice_coef, mean_iou, iou_score,sensitivity
import utils.losses as losses
from utils.utils import str2bool, count_params
import pandas as pd
from net import UNet2D

arch_names = list(UNet2D.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNet',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: NestedUNet)')
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--dataset', default="LITS",
                        help='dataset name')
    parser.add_argument('--input-channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='npy',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='npy',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=100, type=int,
                        metavar='N', help='early stopping (default: 20)')
    
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()
    m_ious = AverageMeter()
    dices_1s = AverageMeter()
    sen_s = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)

                print(output[0,0,:,:].shape)
                # torchvision.utils.save_image(input,'/data/yike/G/Dataset/input.jpg')
                # torchvision.utils.save_image(output,'/data/yike/G/Dataset/output.jpg')
                # torchvision.utils.save_image(target,'/data/yike/G/Dataset/target.jpg')

                loss = criterion(output, target)
                iou = iou_score(output, target)
                m_iou = mean_iou(output,target)
                sen = sensitivity(output,target)
                dice_1 = dice_coef(output, target)[0]
                # dice_2 = dice_coef(output, target)[1]
                print(iou,dice_1)
            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            m_ious.update(m_iou, input.size(0))
            sen_s.update(sen, input.size(0))
            dices_1s.update(torch.tensor(dice_1), input.size(0))
 
            # dices_2s.update(torch.tensor(dice_2), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice_1', dices_1s.avg),
        ('m_iou', m_ious.avg),
        ('sen', sen_s.avg)
    ])

    return log

def load_model_and_test(args, model_path, val_loader, criterion):
    """
    加载训练好的模型权重并进行测试。

    参数:
        args: 命令行参数或配置对象。
        model_path (str): 训练好的模型权重文件路径。
        val_loader (DataLoader): 验证集的数据加载器。
        criterion: 损失函数。
    """
    # 创建模型实例
    model = UNet2D.My_UNet2d(in_channels=3, n_classes=1, n_channels=64)  # 替换为你的模型类
    model = torch.nn.DataParallel(model).cuda()  # 使用多 GPU（如果可用）

    # 加载模型权重
    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint)  # 加载模型权重
    print(f"成功加载模型权重: {model_path}")

    # 切换到评估模式
    model.eval()

    # 进行测试
    log = validate(args, val_loader, model, criterion)

    # 打印测试结果
    print("测试结果:")
    for key, value in log.items():
        print(f"{key}: {value:.4f}")

    return log


# 示例用法
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args = parse_args()
    # 假设你已经定义了 args、val_loader 和 criterion

    val_img_paths = sorted(glob('/data/yike/G/Dataset/ISIC2018_Task1-2_Validation_Input/*.jpg'))
    val_mask_paths = sorted(glob('/data/yike/G/Dataset/ISIC2018_Task1_Validation_GroundTruth/*.png'))

    val_dataset = Dataset2D(args, val_img_paths, val_mask_paths,val=True)  # 验证集数据集

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)  # 验证集数据加载器

    criterion = losses.BCEDiceLoss().cuda()  # 损失函数

    # 模型权重路径
    model_path = "/data/yike/G/LITS2017-main2-master/models/LITS_UNet_lym/0216UNet(me_ISIC18)/epoch74-0.8651_model.pth"

    # 加载模型并测试
    test_log = load_model_and_test(args, model_path, val_loader, criterion)
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


import os
import numpy as np
import torch
import cv2
from PIL import Image
from dataset.dataset import Dataset2D

from utils.metrics import dice_coef, mean_iou, iou_score,sensitivity
import utils.losses as losses
from utils.utils import str2bool, count_params
import pandas as pd
from net import UNet2D

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
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
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=100, type=int,
                        metavar='N', help='early stopping (default: 20)')
    
    parser.add_argument('-b', '--batch-size', default=4, type=int,
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

def load_model_and_test(args, model_path, val_loader):
    """
    参数:
        args: 命令行参数或配置对象。
        model_path (str): 训练好的模型权重文件路径。
        val_loader (DataLoader): 验证集的数据加载器。
        criterion: 损失函数。
    """
    model = UNet2D.My_UNet2d(in_channels=3, n_classes=1, n_channels=64)  # 替换为你的模型类
    model = torch.nn.DataParallel(model).cuda()  # 使用多 GPU（如果可用）

    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint)  # 加载模型权重
    print(f"成功加载模型权重: {model_path}")


    return model


# 示例用法
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()

    val_img_paths = sorted(glob('/data/yike/G/Dataset/ISIC2018_Task1-2_Validation_Input/*.jpg'))
    val_mask_paths = sorted(glob('/data/yike/G/Dataset/ISIC2018_Task1_Validation_GroundTruth/*.png'))

    val_dataset = Dataset2D(args, val_img_paths, val_mask_paths,val=True)  # 验证集数据集
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)  # 验证集数据加载器

    model_path = "/data/yike/G/LITS2017-main2-master/models/LITS_UNet_lym/0216UNet(me_ISIC18)/epoch61-0.8529_model.pth"
    model = load_model_and_test(args, model_path, val_loader)
    output_dir = "./ISIC_visualization_results/"  # 保存结果的文件夹
    os.makedirs(output_dir, exist_ok=True)

    for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
        input = input.cuda()  # [B, 3, H, W]
        target = target.cuda()
        output, att, correction = model(input)  # correction: [B, 3, H, W]

        vis_input = input[0].detach().cpu().numpy()  # [3, H, W]
        vis_correction = correction[0].detach().cpu().numpy()  # [3, H, W]


        correction_heatmap = np.mean(vis_correction, axis=0)  # [H, W]


        vis_input = np.transpose(vis_input, (1, 2, 0))  # [H, W, 3]
        vis_input = (vis_input - vis_input.min()) / (vis_input.max() - vis_input.min())  # 归一化到 [0, 1]
        vis_input = (vis_input * 255).astype(np.uint8)  # 转为 [0, 255]

        correction_heatmap = (correction_heatmap - correction_heatmap.min()) / (correction_heatmap.max() - correction_heatmap.min())
        correction_heatmap = (correction_heatmap * 255).astype(np.uint8)  # 转为 [0, 255]


        correction_heatmap_color = cv2.applyColorMap(correction_heatmap, cv2.COLORMAP_JET)  # 伪彩色

        overlay = cv2.addWeighted(vis_input, 0.6, correction_heatmap_color, 0.4, 0)

        Image.fromarray(vis_input).save(os.path.join(output_dir, f'original_{i}.png'))  # 保存原始输入图像
        Image.fromarray(correction_heatmap_color).save(os.path.join(output_dir, f'heatmap_{i}.png'))  # 保存热力图
        Image.fromarray(overlay).save(os.path.join(output_dir, f'overlay_{i}.png'))  # 保存叠加图

        print(f"Saved results for sample {i}")
        # break  # 只处理第一个 Batch 的第一个样本
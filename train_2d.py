# -*- coding: utf-8 -*-

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

from utils.metrics import dice_coef, mean_iou, iou_score
import utils.losses as losses
from utils.utils import str2bool, count_params
import pandas as pd
from net import UNet2D,UNets



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
    
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
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

def get_mseloss(y_pr,y_gt,attentions, patch_size=16,att_depth=1):

    unfold = nn.Unfold(kernel_size=(patch_size, patch_size),
                    stride=(patch_size, patch_size))

    conv_feamap_size = nn.Conv2d(1, 1, kernel_size=(2 ** att_depth, 2 ** att_depth),
                            stride=(2 ** att_depth, 2 ** att_depth), groups=1, bias=False)
    conv_feamap_size.weight = nn.Parameter(torch.ones((1, 1, 2 ** att_depth, 2 ** att_depth)))
    conv_feamap_size.to(y_pr.device)
    for param in conv_feamap_size.parameters():
        param.requires_grad = False
    y_gt_conv=conv_feamap_size(y_gt)/(2 ** att_depth*2 ** att_depth)

    attentions_gt=[]

    for i in range(y_gt_conv.size()[1]):
        unfold_y_gt = unfold(y_gt[:, i:i + 1, :, :]).transpose(-1, -2)
        unfold_y_gt_conv = unfold(y_gt_conv[:, i:i + 1, :, :])
        att=torch.matmul(unfold_y_gt,unfold_y_gt_conv)/(patch_size*patch_size)
        att=torch.unsqueeze(att,dim=1)
        attentions_gt.append(att)

    attentions_gt=torch.cat(attentions_gt,dim=1)

    loss_mse = F.mse_loss(attentions, attentions_gt)

   

    return loss_mse

def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    ious = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()
    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):

        #print(input.shape)
        #print(target.shape)
        #v = input()
        input = input.cuda()
        target = target.cuda()

        lambda_ce = max(0.3 - 0.04*(epoch/args.epochs), 0.2)  # 从0.7线性衰减到0.3

        # compute output
        if args.deepsupervision:
            outputs = model(input,target)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output,att,g_att = model(input)
            BCE_loss = criterion(output,target)
            iou = iou_score(output, target)
            dice_1 = dice_coef(output, target)[0]
            # dice_2 = dice_coef(output, target)[1]
            mse_loss = get_mseloss(output,target,att)
            loss = mse_loss * lambda_ce + BCE_loss * (1-lambda_ce)

        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))
        dices_1s.update(torch.tensor(dice_1), input.size(0))
        dices_2s.update(torch.tensor(mse_loss), input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice_1', dices_1s.avg),
        ('dice_2', dices_2s.avg)
    ])

    return log




def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()

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
                output,_,_= model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)
                dice_1 = dice_coef(output, target)[0]
                # dice_2 = dice_coef(output, target)[1]

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices_1s.update(torch.tensor(dice_1), input.size(0))
            # dices_2s.update(torch.tensor(dice_2), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice_1', dices_1s.avg)
        # ('dice_2', dices_2s.avg)
    ])

    return log

def main():
    args = parse_args()
    #args.dataset = "datasets"

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_lym' %(args.dataset, args.arch)
        else:
            args.name = '%s_%s_lym' %(args.dataset, args.arch)
    timestamp  = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists('models/{}/{}'.format(args.name,timestamp)):
        os.makedirs('models/{}/{}'.format(args.name,timestamp))

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/{}/{}/args.txt'.format(args.name,timestamp), 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/{}/{}/args.pkl'.format(args.name,timestamp))

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.BCEDiceLoss().cuda()

    cudnn.benchmark = True

    # Data loading code
    img_paths = sorted(glob('/data/yike/G/LITS2017-main2-master/LITS/images/*.png'))
    mask_paths = sorted(glob('/data/yike/G/LITS2017-main2-master/LITS/labels/*.png'))



    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.1, random_state=39)

    # train_img_paths = sorted(glob('/data/yike/G/Dataset/ISIC-2018_Data_train/*.jpg'))
    # train_mask_paths = sorted(glob('/data/yike/G/Dataset/ISIC-2018_label_train/*.png'))

    # val_img_paths = sorted(glob('/data/yike/G/Dataset/ISIC-2018_Data_test/*.jpg'))
    # val_mask_paths = sorted(glob('/data/yike/G/Dataset/ISIC-2018_label_test/*.png'))


    print("train_num:%s"%str(len(train_img_paths)))
    print("val_num:%s"%str(len(val_img_paths)))

 
    # create model
    print("=> creating model %s" %args.arch)
    model = UNet2D.My_UNet2d(in_channels=1, n_classes=1, n_channels=64)
    # model = UNets.AttU_Net(img_ch=3,output_ch=1)
    model = torch.nn.DataParallel(model).cuda()
    #model._initialize_weights()
    #model.load_state_dict(torch.load('model.pth'))

    print("model params:",count_params(model))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    train_dataset = Dataset2D(args, train_img_paths, train_mask_paths, args.aug)
    val_dataset = Dataset2D(args, val_img_paths, val_mask_paths)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou','dice_1', 'mse loss ','val_loss', 'val_iou',
    ])

    best_loss = 100
    # best_iou = 0
    trigger = 0
    first_time = time.time()
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(args, val_loader, model, criterion)

        print('train_loss %.4f - iou %.4f - dice_1 %.4f - mse_loss %.4f- val_loss %.4f - val_iou %.4f - val_dice_1 %.4f'
                  %(train_log['loss'], train_log['iou'], train_log['dice_1'],train_log['dice_2'], val_log['loss'], val_log['iou'], val_log['dice_1']))
        
        end_time = time.time()
        print("time:", (end_time - first_time) / 60)

        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['iou'],
            train_log['dice_1'],

            val_log['loss'],
            val_log['iou'],
            val_log['dice_1'],

        ], index=['epoch', 'lr', 'loss', 'iou', 'dice_1' ,'val_loss', 'val_iou', 'val_dice_1' ])

        log = pd.concat([log, tmp], ignore_index=True)
        log.to_csv('models/{}/{}/log.csv'.format(args.name,timestamp), index=False)

        trigger += 1

        val_loss = val_log['loss']
        if val_loss < best_loss:
            torch.save(model.state_dict(), 'models/{}/{}/epoch{}-{:.4f}_model.pth'.format(args.name,timestamp,epoch,val_log['dice_1']))
            best_loss = val_loss
            print("=> saved best model")
            trigger = 0

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()

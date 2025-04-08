import numpy as np
import nibabel as nib
import cv2 #https://www.jianshu.com/p/f2e88197e81d
import random

from skimage.io import imread
from skimage import color
from PIL import Image

import torch
import torch.utils.data
from torchvision import datasets, models, transforms
import albumentations as albu

train_transform = [
            albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

            albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
            albu.RandomCrop(height=512, width=512, always_apply=True),

            albu.GaussNoise(p=0.2),
            albu.Perspective(p=0.5),

            albu.OneOf(
                [
                    #albu.CLAHE(p=1),
                    #albu.RandomBrightness(p=1),
                    albu.RandomBrightnessContrast(p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    albu.Sharpen(p=1),
                    albu.Blur(blur_limit=3, p=1),
                    albu.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),

            albu.OneOf(
                [
                    #albu.RandomContrast(p=1),
                    albu.RandomBrightnessContrast(p=1),
                    albu.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
        ]


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, aug=False):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = list(map(lambda x: x.replace('volume', 'segmentation').replace('image','label'), self.img_paths))
        # self.mask_paths = mask_paths
        self.aug = aug

        # print(self.img_paths,self.mask_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        npimage = nib.load(img_path).get_fdata()
        npmask = nib.load(mask_path).get_fdata()

        npimage = npimage[:, :, :, np.newaxis]
        npimage = npimage.transpose((3, 0, 1, 2))

        liver_label = npmask.copy()
        liver_label[npmask == 2] = 1
        liver_label[npmask == 1] = 1

        tumor_label = npmask.copy()
        tumor_label[npmask == 1] = 0
        tumor_label[npmask == 2] = 1

        nplabel = np.stack([tumor_label], axis=0).astype('float32')
        assert npimage.shape[1:] == nplabel.shape[1:], "Image and label shapes do not match"

        nplabel = nplabel.astype("float32")
        npimage = npimage.astype("float32")
        #print(npimage.shape)

        return npimage,nplabel

class Dataset2D(Dataset):
    def __init__(self, args, img_paths, mask_paths,val=False, aug=False):
        self.args = args
        self.img_paths = img_paths
        # 如果 mask_paths 未提供，则根据 img_paths 生成
        if mask_paths is None:
            self.mask_paths = list(map(lambda x: x.replace('volume', 'segmentation').replace('image', 'label'), self.img_paths))
        else:
            self.mask_paths = mask_paths
        self.aug = aug
        self.val = val
        self.resize = (512, 512)  # 调整为目标尺寸
        self.transform = albu.Compose(train_transform)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        # print(img_path,mask_path)
        # 加载 2D 图像
        npimage = np.array(Image.open(img_path))  # 使用 PIL 加载图像
        npmask = np.array(Image.open(mask_path))  # 使用 PIL 加载掩码


        # 使用 OpenCV 的 resize 方法调整大小
        if self.val:
            npimage = cv2.resize(npimage, self.resize, interpolation=cv2.INTER_LINEAR)
            npmask = cv2.resize(npmask, self.resize, interpolation=cv2.INTER_NEAREST)
        else:
            augmented = self.transform(image=npimage,mask=npmask)
            npimage,npmask = augmented['image'],augmented['mask']

        
        # 如果图像是灰度图，增加通道维度
        if len(npimage.shape) == 2:
            npimage = npimage[:, :, np.newaxis]  # 形状变为 (height, width, 1)
        npimage = npimage.transpose((2, 0, 1))  # 转换为 (channels, height, width)




        npmask[npmask == 255] = 1
        npmask = np.expand_dims(npmask, axis=0)
        # print(npimage.shape,npmask.shape)

        # 堆叠掩码
        # nplabel = np.stack([liver_label, tumor_label], axis=0).astype('float32')

        # 检查图像和掩码的形状是否匹配
        assert npimage.shape[1:] == npmask.shape[1:], "Image and label shapes do not match"

        # 转换为 float32 类型
        npmask = npmask.astype("float32")
        npimage = npimage.astype("float32")

        return npimage, npmask
    

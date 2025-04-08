## Requirements
* albumentations==1.3.0
* inplace_abn==1.1.0
* mmcv==2.2.0
* numpy==1.24.3
* torch==2.0.1
* torchvision==0.15.2

## Data
* [LiTS](https://www.kaggle.com/andrewmvd/liver-tumor-segmentation) 130 CT scans for segmentation of the liver as well as tumor lesions.


# model
The My_UNet2d model consists of the following components:

Encoder (Downsampling):
Uses a series of convolutional layers (DoubleConv) and max-pooling operations (Down) to extract hierarchical features.
Includes two branches:
Branch 1: Captures high-resolution global information through fewer downsampling steps.
Branch 2: Extracts patch-level features using a convolutional layer with a large stride.
Decoder (Upsampling):
Uses upsampling layers (Up) and skip connections to reconstruct the segmentation mask.
Combines features from both branches for better localization and detail preservation.
Attention Mechanism:
Implements the DPAA module to compute attention weights and refine the segmentation output.
Output Layer:
Produces the final segmentation mask using a 1x1 convolutional layer (Out).

## Run the codes
train: python train.py
validate: python val.py
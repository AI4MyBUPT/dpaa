## Requirements
* albumentations==1.3.0
* inplace_abn==1.1.0
* mmcv==2.2.0
* numpy==1.24.3
* torch==2.0.1
* torchvision==0.15.2

## Data
* [LiTS](https://www.kaggle.com/andrewmvd/liver-tumor-segmentation) 130 CT scans for segmentation of the liver as well as tumor lesions.

## Installation
```bash
git clone https://github.com/AI4MyBUPT/dpaa.git
cd dpaa 
pip install -r requirements.txt
```

# Model 

## Model structure

Uses a series of convolutional layers (DoubleConv) and max-pooling operations (Down) to extract hierarchical features 

Branch 1: Captures high-resolution global information through fewer downsampling steps.  

Branch 2: Extracts patch-level features using a convolutional layer with a large stride.

<p align="center"><img width="100%" src="figures/net.png" /></p>

## DPAA Attention Mechanism

The DPAA (Dynamic Patch-Aware Attention) module is designed to compute attention weights by dynamically assessing the similarity between image patches and shallow layer features. 

This process allows for the explicit modeling of the importance weights of different image regions, thereby highlighting the most relevant areas for the task at hand.

By emphasizing these critical regions, DPAA enhances the model's decision-making capabilities, leading to more accurate and refined segmentation outputs. The mechanism effectively integrates local details with global context, ensuring 

that the model can make more informed predictions while preserving important details and boundaries.

<p align="center"><img width="100%" src="figures/dpaa.png" /></p>

## Run the codes
```bash
train: python train.py 
validate: python val.py
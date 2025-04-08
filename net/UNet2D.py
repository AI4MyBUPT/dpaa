import torch
import torch.nn as nn
import torch.nn.functional as F

from net import DPAA


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(stride) if stride > 1 else nn.Identity(),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 输入是CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Out, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class My_UNet2d(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        # 主干网络
        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 16 * n_channels)
        self.enc5 = Down(16 * n_channels, 16 * n_channels)

        self.dec0 = Up(32 * n_channels,8 * n_channels)
        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

        
        self.dpaa_builder = DPAA.DPAABuilder2D(in_channels,n_classes,patch_size=16,n_channel=self.n_channels,att_depth=1)
        self.dpaa_user = DPAA.DPAAUser2D(n_classes,patch_size=16,att_depth=1)

        self.branch1_conv = DoubleConv(in_channels, n_channels)
        self.branch1_enc1 = Down(n_channels, 2 * n_channels, stride=2)
        self.branch1_enc2 = Down(2 * n_channels, 8 * n_channels, stride=4)

        self.patch_extractor = nn.Conv2d(in_channels, 8 * n_channels, 
                                       kernel_size=3, stride=8, padding=1)

    def forward(self, x):
        x1 = self.conv(x)        # [B, C, H, W]
        x2 = self.enc1(x1)      

        branch1_x1 = self.branch1_conv(x)
        branch1_x2 = self.branch1_enc1(branch1_x1)

        x2 = x2 + branch1_x2 
        att = self.dpaa_builder(x,x2)


        x3 = self.enc2(x2)      
        x4 = self.enc3(x3)       
        branch1_x3 = self.branch1_enc2(branch1_x2)


        x4 = x4 + branch1_x3 

        x5 = self.enc4(x4)       
        x6 = self.enc5(x5)

        patch_features = self.patch_extractor(x)
        x4 = x4 + patch_features  


        # 解码器部分
        mask = self.dec0(x6, x5)
        mask = self.dec1(mask, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)


        #应用注意力
        mask,correction = self.dpaa_user(mask,att)


        return mask,att,correction

# 测试用例
if __name__ == "__main__":
    model = My_UNet2d(in_channels=3, n_classes=1, n_channels=16)
    x = torch.randn(2, 3, 512, 512)  
    output = model(x)
    print("Output shape:", output.shape)  
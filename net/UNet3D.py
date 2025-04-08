from torch import nn
from torch import cat
import torch.nn.functional as F
import torch

from net import DPAA

class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
          )

    def forward(self,x):
        return self.double_conv(x)

    
class Down(nn.Module):

    def __init__(self, in_channels, out_channels,stride=2,num_groups=8):
        super().__init__()
        if stride == 2:
            self.encoder = nn.Sequential(
                nn.MaxPool3d(2, 2),
                DoubleConv(in_channels, out_channels)
            )
        else :
            self.encoder =  nn.Sequential(
                nn.MaxPool3d(2, 2),
                nn.Conv3d(in_channels,out_channels,stride=stride,kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
                nn.ReLU(inplace=True),

                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                #nn.BatchNorm3d(out_channels),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.encoder(x)

    
class Up(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    
class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)


class UNet3d(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)


        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask
    

class My_UNet3d(nn.Module):
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
        self.enc5 = Down(16 * n_channels,16 * n_channels)

        self.dpaa_builder = DPAA.DPAABuilder3D(in_channels,n_classes,patch_size=8,n_channel=self.n_channels,att_depth=1)
        self.dpaa_user = DPAA.DPAAUser3D(n_classes,patch_size=8,att_depth=1)

        self.dec0 = Up(32 * n_channels, 8 * n_channels)
        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

        # 第一分支：两个下采样阶段
        self.branch1_conv = DoubleConv(in_channels, n_channels)
        self.branch1_enc1 = Down(n_channels, 2 * n_channels,stride=2)
        self.branch1_enc2 = Down(2 * n_channels, 8 * n_channels,stride=4)




    def forward(self, x):
        # 主干网络前向传播
        x1 = self.conv(x)
        x2 = self.enc1(x1)

        att = self.dpaa_builder(x,x2)

        branch1_x1 = self.branch1_conv(x)
        branch1_x2 = self.branch1_enc1(branch1_x1)
        branch1_x3 = self.branch1_enc2(branch1_x2)

        branch1_x2 = self.branch1_enc1(branch1_x1)
        x2= self.dual_cross_attn1(x2, branch1_x2) 

        x3 = self.enc2(x2)
        x4 = self.enc3(x3)

        branch1_x3 = self.branch1_enc2(branch1_x2)
        x4 = self.dual_cross_attn2(x4, branch1_x3)

        patch_features = self.patch_extractor(x)  

        x5 = self.enc4(x4)
        x6 = self.enc5(x5)

        # 解码器部分
        mask = self.dec0(x6,x5)
        mask = self.dec1(mask, x4)

        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask,x1)
        mask = self.out(mask)

        mask,correction = self.dpaa_user(mask,att)

        return mask
    

if __name__ == '__main__':
    net = My_UNet3d(1,1,8)
    input_tensor = torch.randn(1, 1, 256, 256, 96)
    net = My_UNet3d(1, 1, 8)
    output = net(input_tensor)
    # print(net)
from torch import nn as nn
import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F

class Unfold3D:
    def __init__(self, kernel_size, stride, padding=0):
        """
        kernel_size: 3D卷积核大小 (tuple or int)
        stride: 滑动窗口的步长大小 (tuple or int)
        padding: 填充大小 (tuple or int)
        """
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)

    def __call__(self, x):
        """
        x: [B, C, D, H, W]
        return: [B, C * kernel_size^3, Num_Patches]
        """
        B, C, D, H, W = x.size()

        # 添加填充
        x = F.pad(x, 
                  pad=(self.padding[2], self.padding[2], self.padding[1], self.padding[1], self.padding[0], self.padding[0]), 
                  mode='constant', value=0)

        # 使用 unfold3D 类似操作
        patches = x.unfold(2, self.kernel_size[0], self.stride[0])  # D 维展开
        patches = patches.unfold(3, self.kernel_size[1], self.stride[1])  # H 维展开
        patches = patches.unfold(4, self.kernel_size[2], self.stride[2])  # W 维展开

        # 输出大小：[B, C, Num_Depth, Num_Height, Num_Width, kernel_D, kernel_H, kernel_W]
        patches = patches.contiguous()

        # 将卷积核维度合并回 C
        patches = patches.view(B, C, -1, *self.kernel_size)
        patches = patches.reshape(B, C * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2], -1)
        return patches

class Fold3D:
    def __init__(self, output_size, kernel_size, stride, padding=0):
        """
        output_size: 3D特征图的大小 (D_out, H_out, W_out)
        kernel_size: 卷积核大小 (tuple or int)
        stride: 滑动步长大小 (tuple or int)
        padding: 填充大小 (tuple or int)
        """
        self.output_size = output_size  # (D, H, W)
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)

    def __call__(self, patches):
        """
        patches: [B, C * kernel_size^3, Num_Patches]
        input_tensor_shape: [B, C, D, H, W] (用于正确特征复原)
        return: [B, C, D_out, H_out, W_out] (还原后的3D特征图)
        """
        B, C_k, num_patches = patches.size()
        C = C_k // (self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2])
        D_out, H_out, W_out = self.output_size


        output = torch.zeros((B, C, D_out + 2 * self.padding[0], H_out + 2 * self.padding[1], W_out + 2 * self.padding[2]), 
                             device=patches.device)


        weight = torch.zeros_like(output)

        # 获取特征图展开后的维度序号
        D_eff = (D_out + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        H_eff = (H_out + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        W_eff = (W_out + 2 * self.padding[2] - self.kernel_size[2]) // self.stride[2] + 1

        # 还原 patch 到目标空间
        patch_idx = 0
        for d in range(D_eff):
            for h in range(H_eff):
                for w in range(W_eff):
                    d_start = d * self.stride[0]
                    h_start = h * self.stride[1]
                    w_start = w * self.stride[2]

                    d_end = d_start + self.kernel_size[0]
                    h_end = h_start + self.kernel_size[1]
                    w_end = w_start + self.kernel_size[2]

                    
                    patch = patches[:, :, patch_idx].view(B, C, *self.kernel_size)

                    
                    output[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += patch
                    weight[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += 1
                    patch_idx += 1


        output = output / weight

        return output[:, :, 
                      self.padding[0]:D_out + self.padding[0], 
                      self.padding[1]:H_out + self.padding[1], 
                      self.padding[2]:W_out + self.padding[2]]

                      
class DPAABuilder2D(nn.Module):
    def __init__(self, in_channels, classes_num=1,n_channel=16,patch_size=16, att_depth=1):
        super().__init__()
        self.patch_size = patch_size
        self.att_depth = att_depth
        
        self.conv_img = nn.Sequential(
            nn.Conv2d(in_channels, (n_channel*(2**att_depth)), kernel_size=7, padding=3),
            nn.Conv2d((n_channel*(2**att_depth)), classes_num, kernel_size=3, padding=1)
        )
        feature_channels = n_channel * (2 ** att_depth)

        self.conv_feamap = nn.Conv2d(
            in_channels=feature_channels, out_channels=classes_num,
            kernel_size=1, stride=1
        )
        

        self.unfold = nn.Unfold(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size)
        )
        
        self.resolution_trans = nn.Sequential(
            nn.Linear(patch_size**2, 2*patch_size**2),
            nn.Linear(2*patch_size**2, patch_size**2),
            nn.ReLU()
        )

    def forward(self, x, features):
        B, C, H, W = x.shape
        attentions = []

        ini_img = self.conv_img(x)  

        feamap = self.conv_feamap(features) / (2 ** self.att_depth * 2 ** self.att_depth)  

        


        for i in range(feamap.size()[1]):

            unfold_img = self.unfold(ini_img[:, i:i + 1, :, :]).transpose(-1, -2)
            unfold_img = self.resolution_trans(unfold_img)

            unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])
            unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

            att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)

            att=torch.unsqueeze(att,1)
 
            attentions.append(att)
        attentions = torch.cat((attentions), dim=1)



        return att

class DPAAUser2D(nn.Module):
    def __init__(self, out_channels, patch_size=16, att_depth=3):
        super().__init__()
        self.patch_size = patch_size
        self.att_depth = att_depth
        

        self.fold = nn.Fold(
            output_size=(512,512),
            kernel_size=(patch_size, patch_size),
            stride=patch_size
        )
        
        # 特征图尺寸调整（平均池化替代）
        self.conv_adjust = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=2**(att_depth),
            stride=2**(att_depth),
            groups=out_channels,
            bias=False
        )
        nn.init.ones_(self.conv_adjust.weight)
        self.conv_adjust.requires_grad_(False)

        # 与builder保持一致的unfold操作
        self.unfold = nn.Unfold(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size)
        )

    def forward(self, x, attentions):

        B, C, H, W = x.shape

        x_argmax = torch.argmax(x, dim=1)
        x_softmax = torch.zeros_like(x).scatter_(1, x_argmax.unsqueeze(1), 1.0)
        
        adjusted_feamap = self.conv_adjust(x_softmax) / (2**self.att_depth)**2

        unfold_feamap = self.unfold(adjusted_feamap)  # [B, C*patch_size^2, L2]
        unfold_feamap = unfold_feamap.view(
            B, C, self.patch_size**2, 
            -1  # L2 = (H_adj/patch_size) * (W_adj/patch_size)
        ).permute(0, 1, 3, 2)  # [B, C, L2, patch_size^2]

        correction = []
        for i in range(C):
            att = attentions[:, i, :, :]  

            modulated = torch.bmm(att, unfold_feamap[:, i, :, :])

            folded = self.fold(modulated.permute(0, 2, 1))
            correction.append(folded)

        correction = torch.cat(correction, dim=1)
        # correction = F.normalize(correction,p=2,dim=3)
        x = x + correction * x

        # 残差连接
        return x + correction * x,correction


class DPAABuilder3D(nn.Module):
    def __init__(self, in_channels, classes_num=1, n_channel=16, patch_size=16, att_depth=1):
        super().__init__()
        self.patch_size = patch_size
        self.att_depth = att_depth


        self.conv_img = nn.Sequential(
            nn.Conv3d(in_channels, n_channel * (2 ** att_depth), kernel_size=7, padding=3),
            nn.Conv3d(n_channel * (2 ** att_depth), classes_num, kernel_size=3, padding=1)
        )

        feature_channels = n_channel * (2 ** att_depth)
        self.conv_feamap = nn.Conv3d(
            in_channels=feature_channels, out_channels=classes_num,
            kernel_size=1, stride=1
        )

        self.unfold = Unfold3D(
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size)
        )

        self.resolution_trans = nn.Sequential(
            nn.Linear(patch_size ** 3, 2 * patch_size ** 3),
            nn.Linear(2 * patch_size ** 3, patch_size ** 3),
            nn.ReLU()
        )

    def forward(self, x, features):
        B, C, D, H, W = x.shape
        attentions = []


        ini_img = self.conv_img(x)  # [B, Nc, D, H, W]
        feamap = self.conv_feamap(features) / (2 ** self.att_depth)  # [B, Nc, D//s, H//s, W//s]

        for i in range(feamap.size()[1]):
            unfold_img = self.unfold(ini_img[:, i:i + 1, :, :, :]).transpose(-1, -2)
            unfold_img = self.resolution_trans(unfold_img)

            unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :, :])
            unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)


            att = torch.einsum('bip,bpj->bij', unfold_img, unfold_feamap) / (self.patch_size ** 3)

            att = torch.unsqueeze(att, 1)  # [B, 1, L1, L2]

            attentions.append(att)

        attentions = torch.cat(attentions, dim=1)  # [B, Nc, L1, L2]
        return attentions

class DPAAUser3D(nn.Module):
    def __init__(self, out_channels, patch_size=16, att_depth=3):
        super().__init__()
        self.patch_size = patch_size
        self.att_depth = att_depth


        self.fold = Fold3D(
            output_size=(128, 128, 128),  
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size)
        )

        self.conv_adjust = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=2 ** (att_depth),
            stride=2 ** (att_depth),
            groups=out_channels,
            bias=False
        )
        nn.init.ones_(self.conv_adjust.weight)
        self.conv_adjust.requires_grad_(False)

        self.unfold = Unfold3D(
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size)
        )

    def forward(self, x, attentions):
        B, C, D, H, W = x.shape

        x_argmax = torch.argmax(x, dim=1)
        x_softmax = torch.zeros_like(x).scatter_(1, x_argmax.unsqueeze(1), 1.0)

        adjusted_feamap = self.conv_adjust(x_softmax) / (2 ** self.att_depth) ** 3
        

        unfold_feamap = self.unfold(adjusted_feamap)
        unfold_feamap = unfold_feamap.view(
            B, C, self.patch_size ** 3, 
            -1  # L2 = (D_adj/patch_size)*(H_adj/patch_size)*(W_adj/patch_size)
        ).permute(0, 1, 3, 2)

        correction = []
        for i in range(C):
            att = attentions[:, i, :, :]  # 注意力矩阵 [B, L1, L2]
            modulated = torch.einsum('bip,bpj->bij',(att, unfold_feamap[:, i, :, :]))  # [B, L1, patch_size^3]

            folded = self.fold(modulated.permute(0, 2, 1))
            correction.append(folded)

        correction = torch.cat(correction, dim=1)  # [B, C, D, H, W]


        x = x + correction * x
        return x + correction * x, correction
    
if __name__ == "__main__":
    B, C, H, W = 2, 1, 512, 512
    patch_size = 8
    att_depth = 1


    builder3d = DPAABuilder3D(in_channels=1, classes_num=2, n_channel=32, patch_size=patch_size, att_depth=1)
    user3d = DPAAUser3D(out_channels=2, patch_size=patch_size, att_depth=1)


    input_3d = torch.randn((2, 2, 128, 128, 128))  # [Batch, Channels, Depth, Height, Width]
    features_3d = torch.randn((2, 64, 64, 64, 64))  # 经过主干特征提取后的 

    decoder_out = torch.randn(2,2,128,128,128)

    attentions = builder3d(input_3d, features_3d)

    output, correction = user3d(decoder_out, attentions)

    print(correction)
    print(f"Output shape: {output.shape}, Correction shape: {correction.shape}")
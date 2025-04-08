import torch
import torch.nn as nn


class DualCrossAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()

        self.channels_compressed = max(channels // reduction_ratio, 4)
        
        # 分支A到分支B的注意力路径
        self.A2B_query = nn.Conv2d(channels, self.channels_compressed, kernel_size=1)
        self.A2B_key = nn.Conv2d(channels, self.channels_compressed, kernel_size=1)
        self.A2B_value = nn.Conv2d(channels, channels, kernel_size=1)
        
        # 分支B到分支A的注意力路径
        self.B2A_query = nn.Conv2d(channels, self.channels_compressed, kernel_size=1)
        self.B2A_key = nn.Conv2d(channels, self.channels_compressed, kernel_size=1)
        self.B2A_value = nn.Conv2d(channels, channels, kernel_size=1)

        self.gamma_A = nn.Parameter(torch.zeros(1))
        self.gamma_B = nn.Parameter(torch.zeros(1))
        
        # 特征融合后的归一化
        self.norm = nn.GroupNorm(4, channels)

    def forward(self, feat_A, feat_B):


        query_A = self.A2B_query(feat_A).view(feat_A.size(0), -1, feat_A.size(2)*feat_A.size(3))  # [B, C', H*W]
        key_B = self.A2B_key(feat_B).view(feat_B.size(0), -1, feat_B.size(2)*feat_B.size(3))      # [B, C', H*W]
        energy_A2B = torch.bmm(query_A.permute(0,2,1), key_B)  # [B, H*W_A, H*W_B]
        attn_A2B = F.softmax(energy_A2B, dim=-1)
        

        value_B = self.A2B_value(feat_B).view(feat_B.size(0), -1, feat_B.size(2)*feat_B.size(3))  # [B, C, H*W]
        enhanced_B = torch.bmm(value_B, attn_A2B.permute(0,2,1))  # [B, C, H*W_A]
        enhanced_B = enhanced_B.view(feat_A.shape)  # 恢复空间维度

        query_B = self.B2A_query(feat_B).view(feat_B.size(0), -1, feat_B.size(2)*feat_B.size(3))
        key_A = self.B2A_key(feat_A).view(feat_A.size(0), -1, feat_A.size(2)*feat_A.size(3))
        energy_B2A = torch.bmm(query_B.permute(0,2,1), key_A)  # [B, H*W_B, H*W_A]
        attn_B2A = F.softmax(energy_B2A, dim=-1)
        
        value_A = self.B2A_value(feat_A).view(feat_A.size(0), -1, feat_A.size(2)*feat_A.size(3))
        enhanced_A = torch.bmm(value_A, attn_B2A.permute(0,2,1)).view(feat_B.shape)

        enhanced_A = self.norm(self.gamma_A * enhanced_A + feat_A)
        enhanced_B = self.norm(self.gamma_B * enhanced_B + feat_B)
        
        return enhanced_A
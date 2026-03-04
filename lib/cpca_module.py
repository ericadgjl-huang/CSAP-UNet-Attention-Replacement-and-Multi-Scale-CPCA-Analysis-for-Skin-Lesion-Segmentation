import torch
import torch.nn as nn
import torch.nn.functional as F


class CPCALayer(nn.Module):
    """
    Channel Prior Convolutional Attention (CPCA)
    簡化版實作，用在 2D feature map：
    - channel 方向：類似 SE，用全域平均池化 + MLP 產生 per-channel 權重
    - spatial 方向：用多尺度 depthwise conv 抽 spatial 關係
    - 最後把兩個注意力相乘後，作用在輸入特徵上

    參數:
      channels: 輸入/輸出通道數
      kernels:  多尺度卷積 kernel size 列表，例如 [3,5,7] / [7,11,21] / [11,21,41]
      reduction: channel 壓縮倍率，類似 SE 的 r
    """
    def __init__(self, channels: int,
                 kernels=(3, 5, 7),
                 reduction: int = 4):
        super().__init__()
        self.channels = channels
        self.kernels = list(kernels)

        # ---- Channel prior (SE-like) ----
        hidden = max(channels // reduction, 1)
        self.channel_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        # (B, C, H, W) -> (B, C, 1, 1)
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )

        # ---- Multi-scale spatial branch (depthwise conv) ----
        depthwise_convs = []
        for k in self.kernels:
            pad = k // 2
            depthwise_convs.append(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=k,
                    padding=pad,
                    groups=channels,   # depthwise
                    bias=False,
                )
            )
        self.depthwise_convs = nn.ModuleList(depthwise_convs)

        self.spatial_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # ---- channel prior ----
        ch_att = self.channel_mlp(x)        # (B, C, 1, 1)
        ch_att = self.sigmoid(ch_att)
        x_ch = x * ch_att                   # broadcast

        # ---- spatial multi-scale ----
        spatial = 0
        for conv in self.depthwise_convs:
            spatial = spatial + conv(x)     # 用原始 x 當輸入，跟論文概念一致
        spatial = self.spatial_proj(spatial)
        spatial = self.sigmoid(spatial)     # (B, C, H, W)

        out = x_ch * spatial
        return out

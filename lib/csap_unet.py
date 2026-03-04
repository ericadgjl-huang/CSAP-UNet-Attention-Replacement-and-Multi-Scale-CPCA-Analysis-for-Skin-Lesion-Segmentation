# lib/csap_unet.py

from .cpca_module import CPCALayer

ATTENTION_MODE = 'hybrid_B'   # 'cbam' / 'cpca' / 'hybrid_A' / 'hybrid_B'
CPCA_KERNELS = (7,11,21)  # 你想要的 kernel 組合


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet34, resnet50
from .DeiT import deit_small_patch16_224 as deit_small
from .DeiT import deit_base_patch16_224 as deit_base

# ----------------- 基本積木，很多沿用你原本 CSAP-UNet.py 內的寫法 -----------------


class CBAMLayer(nn.Module):
    """
    論文 AFM 裡的基礎注意力模組：先 Channel attention 再 Spatial attention。:contentReference[oaicite:2]{index=2}
    """
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super().__init__()
        # channel attention
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class ResidualBlock(nn.Module):
    """
    簡化版 residual block：Conv-BN-ReLU x2 + shortcut
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.need_skip = (in_ch != out_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        if self.need_skip:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x) if self.need_skip else x
        out = self.conv(x)
        out = out + identity
        return self.relu(out)


class AttentionGate(nn.Module):
    """
    Attention U-Net 風格的 AG，用在 decoder skip connection 上。:contentReference[oaicite:3]{index=3}
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UpBlock(nn.Module):
    """
    上採樣 + Conv block
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ResidualBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


# ----------------- BEM：Boundary Enhancement Module -----------------


class BEM(nn.Module):
    """
    論文 3.3 的 Boundary Enhancement Module，放在第一層。:contentReference[oaicite:4]{index=4}
    將 CNN branch (l0) 和 Transformer branch (g0) 的特徵，各自經過兩層 conv 後相加。
    """
    def __init__(self, in_ch_g, in_ch_l, mid_g=512, mid_l=128, out_ch=256):
        super().__init__()
        self.conv1_g = nn.Sequential(
            nn.Conv2d(in_ch_g, mid_g, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_g),
        )
        self.conv2_g = nn.Sequential(
            nn.Conv2d(mid_g, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
        )

        self.conv1_l = nn.Sequential(
            nn.Conv2d(in_ch_l, mid_l, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_l),
        )
        self.conv2_l = nn.Sequential(
            nn.Conv2d(mid_l, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, g0, l0):
        g = self.conv2_g(self.conv1_g(g0))
        l = self.conv2_l(self.conv1_l(l0))
        return g + l   # E in the paper


# ----------------- AFM：Attention Fusion Module -----------------


class AFM(nn.Module):
    """
    論文 3.2 的 Attention Fusion Module。
    - 對 CNN/Transformer 特徵各做注意力（CBAM 或 CPCA）
    - concat 後做 global avg pooling + MLP + softmax 得到兩個 branch 的權重
    - 加權後相加，丟進 residual block
    Attention Fusion Module
    - CNN branch: l
    - Transformer branch: g
    """
    def __init__(self, ch):
        super().__init__()

        # === 注意力模組選擇：CBAM / CPCA / Hybrid-A / Hybrid-B ===
        if ATTENTION_MODE == 'cbam':
            # 兩個分支都用 CBAM
            self.att_cnn = CBAMLayer(ch)
            self.att_tr  = CBAMLayer(ch)

        elif ATTENTION_MODE == 'cpca':
            # 兩個分支都用 CPCA(3,5,7)
            self.att_cnn = CPCALayer(ch, kernels=CPCA_KERNELS)
            self.att_tr  = CPCALayer(ch, kernels=CPCA_KERNELS)

        elif ATTENTION_MODE == 'hybrid_A':
            # ✅ Hybrid-A：CNN = CBAM, Transformer = CPCA
            self.att_cnn = CBAMLayer(ch)
            self.att_tr  = CPCALayer(ch, kernels=CPCA_KERNELS)

        elif ATTENTION_MODE == 'hybrid_B':
            # ✅ Hybrid-B：CNN = CPCA, Transformer = CBAM
            self.att_cnn = CPCALayer(ch, kernels=CPCA_KERNELS)
            self.att_tr  = CBAMLayer(ch)

        else:
            raise ValueError(f"Unknown ATTENTION_MODE: {ATTENTION_MODE}")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(ch * 2, ch // 2),
            nn.ReLU(inplace=True),
            nn.Linear(ch // 2, 2),
        )
        self.softmax = nn.Softmax(dim=1)
        self.residual = ResidualBlock(ch, ch)

    def forward(self, g, l):
        # g: Transformer branch, l: CNN branch
        g_att = self.att_tr(g)      # ← 這行改掉
        l_att = self.att_cnn(l)     # ← 這行改掉

        x_sum = g_att + l_att
        w = self.avg_pool(torch.cat([g_att, l_att], dim=1))  # [B,2C,1,1]
        w = w.view(w.size(0), -1)                           # [B,2C]
        w = self.mlp(w)                                     # [B,2]
        w = self.softmax(w)                                 # 每個樣本兩個權重
        w_g = w[:, 0].view(-1, 1, 1, 1)
        w_l = w[:, 1].view(-1, 1, 1, 1)

        out = w_g * g_att + w_l * l_att
        out = self.residual(out)
        return out


# ----------------- 主網路：CSAP-UNet -----------------


class CSAP_UNet(nn.Module):
    """
    依照論文，實作 CSAP-UNet 的平行 Encoder + AFM + BEM + Decoder。:contentReference[oaicite:6]{index=6}
    version: 'S' 使用 ResNet34 + DeiT-small
             'L' 使用 ResNet50 + DeiT-base
    """
    def __init__(self, version='S', img_size=(192, 256), num_classes=1, pretrained=True):
        super().__init__()
        assert version in ['S', 'L']
        self.version = version
        self.num_classes = num_classes
        H, W = img_size
        self.H16, self.W16 = H // 16, W // 16
        self.H8,  self.W8  = H // 8,  W // 8
        self.H4,  self.W4  = H // 4,  W // 4

        # ----- CNN branch -----
        if version == 'S':
            resnet = resnet34(weights=None if not pretrained else None)
            # 如果你想用 torchvision 預訓練，可以改成 weights="IMAGENET1K_V1"
        else:
            resnet = resnet50(weights=None if not pretrained else None)

        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )  # H/4

        self.layer1 = resnet.layer1   # H/4
        self.layer2 = resnet.layer2   # H/8
        self.layer3 = resnet.layer3   # H/16
        # layer4 拿掉，論文也說只用前四個 stage :contentReference[oaicite:7]{index=7}

        if version == 'S':
            ch_l2, ch_l1, ch_l0 = 64, 128, 256
            ch_g0_raw = 384
        else:
            ch_l2, ch_l1, ch_l0 = 256, 512, 1024
            ch_g0_raw = 768  # DeiT-Base hidden dim

        # ----- Transformer branch (DeiT) -----
        if version == 'S':
            self.transformer = deit_small(pretrained=pretrained)
            self.tr_dim = 384
        else:
            self.transformer = deit_base(pretrained=pretrained)
            self.tr_dim = 768

        # 將 Transformer feature 轉成對應 CNN channel
        self.conv_g0 = nn.Conv2d(self.tr_dim, ch_l0, 1)
        self.conv_g1 = nn.Conv2d(self.tr_dim, ch_l1, 1)
        self.conv_g2 = nn.Conv2d(self.tr_dim, ch_l2, 1)

        # ----- BEM & AFM -----
        # scale 0: H/16
        self.bem = BEM(in_ch_g=ch_l0, in_ch_l=ch_l0,
                       mid_g=512 if version == 'S' else 512,
                       mid_l=128 if version == 'S' else 256,
                       out_ch=ch_l0)

        self.afm0 = AFM(ch_l0)   # H/16
        self.afm1 = AFM(ch_l1)   # H/8
        self.afm2 = AFM(ch_l2)   # H/4

        # ----- Decoder -----
        # feature 維度照 Table 1 的 scale 概念，這裡簡化為 ch_l0/1/2
        self.up1 = UpBlock(ch_l0, ch_l1)   # H/16 -> H/8
        self.up2 = UpBlock(ch_l1, ch_l2)   # H/8  -> H/4

        self.ag1 = AttentionGate(F_g=ch_l1, F_l=ch_l1, F_int=ch_l1 // 2)
        self.ag2 = AttentionGate(F_g=ch_l2, F_l=ch_l2, F_int=ch_l2 // 2)

        self.conv_dec1 = ResidualBlock(ch_l1 * 2, ch_l1)
        self.conv_dec2 = ResidualBlock(ch_l2 * 2, ch_l2)

        # 預測頭：最終輸出 + 兩個 deep supervision（對應論文 (10) 的三個 loss）:contentReference[oaicite:8]{index=8}
        self.head_main = nn.Conv2d(ch_l2, num_classes, 1)
        self.head_tr   = nn.Conv2d(ch_l2, num_classes, 1)   # 來自 g2
        self.head_afm0 = nn.Conv2d(ch_l0, num_classes, 1)   # 來自 f0

    # ----------------- forward -----------------

    def forward(self, x):
        B, C, H, W = x.shape

        # ----- CNN branch -----
        x0 = self.conv1(x)          # H/4
        l2 = self.layer1(x0)        # H/4
        l1 = self.layer2(l2)        # H/8
        l0 = self.layer3(l1)        # H/16

        # ----- Transformer branch -----
        # DeiT 這個實作的 forward 回傳 [B, N, D]，TransFuse 也是這樣用 :contentReference[oaicite:9]{index=9}
        t = self.transformer(x)        # [B, N, D]
        t = t.transpose(1, 2)         # [B, D, N]
        t = t.view(B, self.tr_dim, self.H16, self.W16)  # H/16

        g0_raw = t
        g1_raw = F.interpolate(t, scale_factor=2, mode='bilinear', align_corners=True)  # H/8
        g2_raw = F.interpolate(g1_raw, scale_factor=2, mode='bilinear', align_corners=True)  # H/4

        g0 = self.conv_g0(g0_raw)
        g1 = self.conv_g1(g1_raw)
        g2 = self.conv_g2(g2_raw)

        # ----- AFM + BEM -----
        # 第一層：AFM0 + BEM（論文 Fig.1）:contentReference[oaicite:10]{index=10}
        f0_afm = self.afm0(g0, l0)        # H/16, ch_l0
        E = self.bem(g0, l0)              # boundary feature
        f0 = f0_afm + E                   # 先簡單相加

        # 第二、三層 AFM
        f1 = self.afm1(g1, l1)           # H/8
        f2 = self.afm2(g2, l2)           # H/4

        # ----- Decoder -----
        # stage1: f0 -> up H/8，與 f1 做 AG + concat
        d1_up = self.up1(f0)            # H/8, ch_l1
        f1_ag = self.ag1(f1, d1_up)
        d1 = torch.cat([d1_up, f1_ag], dim=1)
        d1 = self.conv_dec1(d1)         # H/8, ch_l1

        # stage2: d1 -> up H/4，與 f2 做 AG + concat
        d2_up = self.up2(d1)            # H/4, ch_l2
        f2_ag = self.ag2(f2, d2_up)
        d2 = torch.cat([d2_up, f2_ag], dim=1)
        d2 = self.conv_dec2(d2)         # H/4, ch_l2

        # 預測頭
        out_main = self.head_main(d2)                   # H/4
        out_main = F.interpolate(out_main, size=(H, W), mode='bilinear', align_corners=True)

        # deep supervision 1：Transformer branch g2
        out_tr = self.head_tr(g2)
        out_tr = F.interpolate(out_tr, size=(H, W), mode='bilinear', align_corners=True)

        # deep supervision 2：AFM0 feature f0
        out_afm0 = self.head_afm0(f0)
        out_afm0 = F.interpolate(out_afm0, size=(H, W), mode='bilinear', align_corners=True)

        return out_main, out_tr, out_afm0

# scripts/measure_complexity.py

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from thop import profile
from lib.csap_unet import CSAP_UNet
import numpy as np

# 讓 cuDNN 自動挑最快 kernel
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(version="S"):
    model = CSAP_UNet(
        version=version,
        img_size=(192, 256),
        num_classes=1,
        pretrained=False,
    )
    return model


def measure(version="S", tag="", batch_size=8, iters=200, rounds=5):
    """
    多輪測量：
      - FLOPs、Params 只算一次（CPU 上）
      - Inference time 在 GPU 上跑 N_rounds，每輪 N_iters 次，最後取 mean / std
    """
    # ---------------------------
    # 1) 建立模型 (GPU / CPU 各一份)
    # ---------------------------
    model_gpu = build_model(version).to(device)
    model_gpu.eval()

    model_cpu = build_model(version).cpu()
    model_cpu.eval()

    # dummy input
    x_gpu = torch.randn(batch_size, 3, 192, 256).to(device)
    x_cpu = torch.randn(batch_size, 3, 192, 256)

    # ---------------------------
    # 2) Params
    # ---------------------------
    n_params = sum(p.numel() for p in model_gpu.parameters() if p.requires_grad)

    # ---------------------------
    # 3) FLOPs (CPU 上量，較穩定)
    # ---------------------------
    macs, _ = profile(model_cpu, inputs=(x_cpu,), verbose=False)
    flops = macs * 2  # MACs -> FLOPs

    # ---------------------------
    # 4) Inference time (GPU，多輪測量)
    # ---------------------------
    times = []

    # 先做一點 warmup
    with torch.no_grad():
        for _ in range(30):
            _ = model_gpu(x_gpu)
    if device.type == "cuda":
        torch.cuda.synchronize()

    for r in range(rounds):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(iters):
                _ = model_gpu(x_gpu)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()

        # 每輪的「單張影像平均時間」
        avg_time_per_img = (end - start) / (iters * batch_size) * 1000.0
        times.append(avg_time_per_img)
        print(f"[Round {r+1}/{rounds}] avg time = {avg_time_per_img:.4f} ms / image")

    times = np.array(times)
    mean_t = float(times.mean())
    std_t = float(times.std())

    # ---------------------------
    # 5) 印結果
    # ---------------------------
    print(f"\n=== {tag} ({version}) ===")
    print(f"Params: {n_params/1e6:.3f} M")
    print(f"FLOPs:  {flops/1e9:.3f} GFLOPs")
    print(f"Time  :  {mean_t:.4f} ± {std_t:.4f} ms / image")

    return n_params, flops, mean_t, std_t


if __name__ == "__main__":
    # 你可以直接這樣跑
    measure("S", tag="S + CURRENT_SETTING")

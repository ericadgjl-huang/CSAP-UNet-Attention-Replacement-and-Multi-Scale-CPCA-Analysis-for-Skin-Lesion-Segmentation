# scripts/make_robust_sets.py

import os
import sys

# Ensure working directory is project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from tqdm import tqdm

# -------------------------------------------------
# 讀取原本的 ISIC2017 測試資料 (data/test_data.npy)
# -------------------------------------------------

DATA_ROOT = "data"   # 跟 process_isic_to_npy_v3.py 一樣
IMG_PATH = os.path.join(DATA_ROOT, "test_data.npy")
MASK_PATH = os.path.join(DATA_ROOT, "test_mask.npy")

print(f"Loading base test set from: {IMG_PATH}")
imgs = np.load(IMG_PATH)   # shape: (N, 192, 256, 3)
masks = np.load(MASK_PATH) # shape: (N, 192, 256)

print(f"imgs shape: {imgs.shape}, masks shape: {masks.shape}")

# -------------------------------------------------
# 一些簡單的變形函式
# -------------------------------------------------

def add_gaussian_noise(imgs, sigma=0.1):
    """
    imgs: uint8 [0,255]
    sigma: noise 標準差比例 (0.1 → 0.1*255)
    """
    imgs_f = imgs.astype(np.float32)
    noise = np.random.normal(0, sigma * 255, imgs_f.shape)
    noisy = imgs_f + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def add_blur(imgs, ksize=(5, 5)):
    out = []
    for im in imgs:
        out.append(cv2.GaussianBlur(im, ksize, 0))
    return np.stack(out, axis=0).astype(np.uint8)

def change_brightness(imgs, factor=1.2):
    """
    factor > 1 變亮 / factor < 1 變暗
    """
    imgs_f = imgs.astype(np.float32) * factor
    imgs_f = np.clip(imgs_f, 0, 255)
    return imgs_f.astype(np.uint8)

def degrade_resolution(imgs, down_h=128, down_w=128):
    """
    先縮小到 (down_h, down_w)，再放回原本 (192,256)
    只是模擬低解析度，不改變模型輸入大小
    """
    N, H, W, C = imgs.shape
    out = []
    for im in tqdm(imgs, desc="Degrade to low-res"):
        low = cv2.resize(im, (down_w, down_h))           # (w, h)
        back = cv2.resize(low, (W, H))                   # 回到原尺寸
        out.append(back)
    return np.stack(out, axis=0).astype(np.uint8)

# -------------------------------------------------
# 輸出 helper
# -------------------------------------------------

def save_variant(var_imgs, var_name):
    """
    var_name: e.g. 'gauss', 'blur', 'bright', 'dark', 'res128'
    會建立資料夾 data_robust_<var_name>/test_data.npy & test_mask.npy
    """
    out_root = f"data_robust_{var_name}"
    os.makedirs(out_root, exist_ok=True)

    np.save(os.path.join(out_root, "test_data.npy"), var_imgs)
    np.save(os.path.join(out_root, "test_mask.npy"), masks)

    print(f"Saved variant '{var_name}' to: {out_root}")
    print(f"  test_data.npy shape: {var_imgs.shape}")
    print(f"  test_mask.npy shape: {masks.shape}")


if __name__ == "__main__":

    # 原圖其實就是 data/ 裡的 test_data.npy，不用再存一份
    # 下面開始做各種 robust 版本

    print("\n=== Generating Gaussian noise version ===")
    imgs_gauss = add_gaussian_noise(imgs, sigma=0.10)
    save_variant(imgs_gauss, "gauss")

    print("\n=== Generating blur version ===")
    imgs_blur = add_blur(imgs, ksize=(5, 5))
    save_variant(imgs_blur, "blur")

    print("\n=== Generating brighter version ===")
    imgs_bright = change_brightness(imgs, factor=1.2)
    save_variant(imgs_bright, "bright")

    print("\n=== Generating darker version ===")
    imgs_dark = change_brightness(imgs, factor=0.7)
    save_variant(imgs_dark, "dark")

    print("\n=== Generating low-resolution (128x128) version ===")
    imgs_res128 = degrade_resolution(imgs, down_h=128, down_w=128)
    save_variant(imgs_res128, "res128")

    print("\n=== All robust variants generated ===")

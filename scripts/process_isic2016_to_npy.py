# scripts/process_isic2016_to_npy.py

import os
import sys

# Ensure working directory is project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# =====================================================
# normalize function（跟 ISIC2017 保持一致）
# =====================================================

def normalize_img(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = img * 255
    return img.astype(np.uint8)

# =====================================================
# load one image + mask
# =====================================================

def load_pair(img_path, mask_path, size=(192, 256)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size[1], size[0]))
    img = normalize_img(img)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8)

    return img, mask


# =====================================================
# process one split
# =====================================================

def process_split(split_name, img_dir, mask_dir, out_prefix):
    # ISIC2016: 影像是 JPG、mask 是 PNG
    img_files = sorted(glob(os.path.join(img_dir, "*.jpg")))
    mask_files = sorted(glob(os.path.join(mask_dir, "*.png")))

    print(f"\nProcessing {split_name}...")
    print(f"Images: {len(img_files)}, Masks: {len(mask_files)}")

    data_list, mask_list = [], []

    for img_path, mask_path in tqdm(zip(img_files, mask_files),
                                    total=len(img_files)):
        img, mask = load_pair(img_path, mask_path)
        data_list.append(img)
        mask_list.append(mask)

    np.save(f"{out_prefix}_data.npy", np.array(data_list, dtype=np.uint8))
    np.save(f"{out_prefix}_mask.npy", np.array(mask_list, dtype=np.uint8))

    print(f"Saved {out_prefix}_data.npy and {out_prefix}_mask.npy")


# =====================================================
# main
# =====================================================

if __name__ == "__main__":
    # 這裡就是你剛剛貼出來的 data2016 路徑
    root = "data2016"

    # 訓練集（可有可無，主要是之後 robustness 會用 test）
    process_split(
        "train2016",
        f"{root}/ISBI2016_ISIC_Part1_Training_Data/ISBI2016_ISIC_Part1_Training_Data",
        f"{root}/ISBI2016_ISIC_Part1_Training_GroundTruth/ISBI2016_ISIC_Part1_Training_GroundTruth",
        f"{root}/train2016"
    )

    # 測試集：我們會拿這一組來做 cross-year robustness
    process_split(
        "test2016",
        f"{root}/ISBI2016_ISIC_Part1_Test_Data/ISBI2016_ISIC_Part1_Test_Data",
        f"{root}/ISBI2016_ISIC_Part1_Test_GroundTruth/ISBI2016_ISIC_Part1_Test_GroundTruth",
        f"{root}/test"   # ★ 這裡用 test，之後 test_csap_isic_v3.py 會讀 test_data.npy / test_mask.npy
    )

    print("\n=== ISIC2016 to npy done ===")

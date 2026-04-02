# scripts/process_isic_to_npy.py

import os
import sys

# Ensure working directory is project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# =====================================================
# normalize function（保持一致色調，可預防顏色偏斜）
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
    root = "data"

    process_split(
        "train",
        f"{root}/ISIC-2017_Training_Data/ISIC-2017_Training_Data",
        f"{root}/ISIC-2017_Training_Part1_GroundTruth/ISIC-2017_Training_Part1_GroundTruth",
        f"{root}/train"
    )

    process_split(
        "val",
        f"{root}/ISIC-2017_Validation_Data/ISIC-2017_Validation_Data",
        f"{root}/ISIC-2017_Validation_Part1_GroundTruth/ISIC-2017_Validation_Part1_GroundTruth",
        f"{root}/val"
    )

    process_split(
        "test",
        f"{root}/ISIC-2017_Test_v2_Data/ISIC-2017_Test_v2_Data",
        f"{root}/ISIC-2017_Test_v2_Part1_GroundTruth/ISIC-2017_Test_v2_Part1_GroundTruth",
        f"{root}/test"
    )

    print("\n=== All splits done ===")

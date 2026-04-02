# dataset_isic_v3.py
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class ISICNpyDatasetV3(Dataset):
    """
    使用 process_isic_to_npy_v3.py 產生的
    train_data.npy / train_mask.npy / val_data.npy / ... 等檔案。
    影像 shape: (H, W, 3)，mask shape: (H, W)
    """

    def __init__(self, root="data", split="train", augment=True):
        assert split in ["train", "val", "test"]
        self.root = root
        self.split = split
        self.augment = augment and (split == "train")

        data_path = os.path.join(root, f"{split}_data.npy")
        mask_path = os.path.join(root, f"{split}_mask.npy")

        self.images = np.load(data_path)   # (N, H, W, 3), uint8
        self.masks = np.load(mask_path)    # (N, H, W),   uint8 / 0-255

        assert len(self.images) == len(self.masks), "image/mask 數量不一致"

    def __len__(self):
        return len(self.images)

    def _augment(self, img, mask):
        # random horizontal flip
        if random.random() < 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        # random vertical flip
        if random.random() < 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)
        return img, mask

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0  # -> [0,1]
        mask = self.masks[idx].astype(np.float32)
        mask = (mask > 0.5).astype(np.float32)             # -> {0,1}

        if self.augment:
            img, mask = self._augment(img, mask)

        # HWC -> CHW
        # Safe copy: 防止 negative-stride 問題
        img = torch.from_numpy(img.copy().transpose(2, 0, 1))
        mask = torch.from_numpy(mask.copy()).unsqueeze(0)  # [1, H, W]

        return img, mask

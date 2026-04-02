# scripts/test.py

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import imageio

from dataset.isic import ISICNpyDatasetV3
from lib.csap_unet import CSAP_UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dice_iou_np(gt, pred, eps=1e-6):
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    inter = (gt * pred).sum()
    union = gt.sum() + pred.sum()

    dice = (2 * inter + eps) / (union + eps)

    union_iou = gt.sum() + pred.sum() - inter
    iou = (inter + eps) / (union_iou + eps)
    return float(dice), float(iou)


def compute_confusion(gt, pred):
    gt = gt.astype(np.bool_)
    pred = pred.astype(np.bool_)

    tp = np.logical_and(gt, pred).sum()
    tn = np.logical_and(~gt, ~pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    return tp, tn, fp, fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--version", type=str, default="S", choices=["S", "L"])
    args = parser.parse_args()

    img_path = os.path.join(args.data_root, "test_data.npy")
    mask_path = os.path.join(args.data_root, "test_mask.npy")

    imgs = np.load(img_path)  # (N, H, W, 3)
    gts = np.load(mask_path)  # (N, H, W)

    N = imgs.shape[0]
    print(f"[INFO] Loaded {N} test images")

    model = CSAP_UNet(
        version=args.version,
        img_size=(192, 256),
        num_classes=1,
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.eval()

    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)

    dice_list, iou_list = [], []
    tp_total = tn_total = fp_total = fn_total = 0

    for i in range(N):
        img = imgs[i].astype(np.float32) / 255.0
        gt = gts[i].astype(np.float32)
        gt_bin = (gt > 0.5).astype(np.uint8)

        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            out_main, _, _ = model(img_tensor)

        pred = out_main.sigmoid().cpu().numpy().squeeze()
        pred_bin = (pred > 0.5).astype(np.uint8)

        if args.save_path is not None:
            imageio.imwrite(
                os.path.join(args.save_path, f"{i}_img.jpg"),
                (img * 255).astype(np.uint8),
            )
            imageio.imwrite(
                os.path.join(args.save_path, f"{i}_pred.jpg"),
                (pred_bin * 255).astype(np.uint8),
            )
            imageio.imwrite(
                os.path.join(args.save_path, f"{i}_gt.jpg"),
                (gt_bin * 255).astype(np.uint8),
            )

        d, j = dice_iou_np(gt_bin, pred_bin)
        dice_list.append(d)
        iou_list.append(j)

        tp, tn, fp, fn = compute_confusion(gt_bin, pred_bin)
        tp_total += tp
        tn_total += tn
        fp_total += fp
        fn_total += fn

    # 聚合
    eps = 1e-6
    dice_mean = float(np.mean(dice_list))
    iou_mean = float(np.mean(iou_list))

    fg_acc = tp_total / (tp_total + fn_total + eps)
    bg_acc = tn_total / (tn_total + fp_total + eps)
    bal_acc = (fg_acc + bg_acc) / 2.0

    precision = tp_total / (tp_total + fp_total + eps)
    recall = fg_acc
    f1 = 2 * precision * recall / (precision + recall + eps)

    overall_acc = (tp_total + tn_total) / (
        tp_total + tn_total + fp_total + fn_total + eps
    )

    print("\n====== Test Results (v3) ======")
    print(f"Dice: {dice_mean:.4f}")
    print(f"IoU : {iou_mean:.4f}")
    print(f"FG Acc: {fg_acc:.4f}")
    print(f"BG Acc: {bg_acc:.4f}")
    print(f"Balanced Acc: {bal_acc:.4f}")
    print(f"Overall Acc : {overall_acc:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1-score    : {f1:.4f}")

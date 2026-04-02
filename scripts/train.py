# scripts/train.py

import argparse
import os
import sys
from datetime import datetime
import csv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.isic import ISICNpyDatasetV3
from lib.csap_unet import CSAP_UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------- 小工具：logger ----------------- #

class CsvLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new_file = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.writer(self.f)
        if new_file:
            self.w.writerow(["epoch", "split", "loss", "dice", "iou"])

    def log(self, epoch, split, loss, dice, iou):
        self.w.writerow([epoch, split, loss, dice, iou])
        self.f.flush()

    def close(self):
        self.f.close()


# ----------------- Loss & Metrics ----------------- #

def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    targets = targets.float()

    dims = (2, 3)
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def bce_dice_loss(logits, targets, smooth=1e-6):
    """
    logits: [B, 1, H, W]
    targets: [B, H, W] 或 [B, 1, H, W]
    """
    # 若是 [B, H, W]，補成 [B, 1, H, W]
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    # BCE with logits
    bce = F.binary_cross_entropy_with_logits(logits, targets)

    # Dice loss
    probs = torch.sigmoid(logits)
    # [B, 1, H, W] → 在 H,W 維度上做
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice_score = (2 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice_score.mean()

    return bce + dice_loss



def calc_metrics(logits, targets):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    dims = (2, 3)
    intersection = (preds * targets).sum(dim=dims)
    union = (preds + targets).clamp(0, 1).sum(dim=dims)

    dice = (2 * intersection + 1e-6) / (preds.sum(dim=dims) + targets.sum(dim=dims) + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)

    return dice.mean().item(), iou.mean().item()


# ----------------- train / val loop ----------------- #

def train_one_epoch(model, loader, optimizer, grad_norm, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    for i, (imgs, gts) in enumerate(loader, start=1):
        imgs = imgs.to(device)
        gts = gts.to(device)

        out_main, out_tr, out_afm0 = model(imgs)

        loss_main = bce_dice_loss(out_main, gts)
        loss_tr = bce_dice_loss(out_tr, gts)
        loss_afm0 = bce_dice_loss(out_afm0, gts)

        loss = loss_main + 0.4 * loss_tr + 0.4 * loss_afm0

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()

        dice, iou = calc_metrics(out_main, gts)

        running_loss += loss.item()
        running_dice += dice
        running_iou += iou

        if i % 20 == 0 or i == len(loader):
            print(
                f"{datetime.now()} Epoch [{epoch}/{total_epochs}] "
                f"Step [{i}/{len(loader)}] "
                f"Loss: {running_loss / i:.4f} "
                f"Dice: {running_dice / i:.4f} "
                f"IoU: {running_iou / i:.4f}"
            )

    n = len(loader)
    return running_loss / n, running_dice / n, running_iou / n


@torch.no_grad()
def validate(model, loader):
    model.eval()
    losses, dices, ious = [], [], []

    for imgs, gts in loader:
        imgs = imgs.to(device)
        gts = gts.to(device)

        out_main, _, _ = model(imgs)
        loss = bce_dice_loss(out_main, gts)
        dice, iou = calc_metrics(out_main, gts)

        losses.append(loss.item())
        dices.append(dice)
        ious.append(iou)

    return float(np.mean(losses)), float(np.mean(dices)), float(np.mean(ious))


# ----------------- main ----------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="S", choices=["S", "L"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_norm", type=float, default=2.0)
    parser.add_argument("--train_save", type=str, default="CSAP_v3_S")
    parser.add_argument("--data_root", type=str, default="data")
    opt = parser.parse_args()

    print("==== Config ====")
    print(opt)

    # dataset & dataloader
    train_set = ISICNpyDatasetV3(root=opt.data_root, split="train", augment=True)
    val_set = ISICNpyDatasetV3(root=opt.data_root, split="val", augment=False)

    train_loader = DataLoader(
        train_set,
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=opt.batchsize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # model（這裡特別：pretrained=True）
    model = CSAP_UNet(
        version=opt.version,
        img_size=(192, 256),
        num_classes=1,
        pretrained=True,
    )
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
    )

    # logger & snapshots
    os.makedirs("logs", exist_ok=True)
    save_dir = os.path.join("snapshots", opt.train_save + f"_{opt.version}")
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join("logs", f"{opt.train_save}_{opt.version}.csv")
    logger = CsvLogger(log_path)
    print(f"[Logger] saving to {log_path}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params / 1e6:.2f} M")

    best_val_dice = 0.0

    for epoch in range(1, opt.epochs + 1):
        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, optimizer, opt.grad_norm, epoch, opt.epochs
        )

        val_loss, val_dice, val_iou = validate(model, val_loader)
        print(
            f"==== Val Epoch {epoch} ===="
            f" Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}"
        )

        # log
        logger.log(epoch, "train", train_loss, train_dice, train_iou)
        logger.log(epoch, "val", val_loss, val_dice, val_iou)

        # save best
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            ckpt_path = os.path.join(
                save_dir, f"best_epoch{epoch}_dice{val_dice:.4f}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Save best model] {ckpt_path}")

    logger.close()


if __name__ == "__main__":
    main()

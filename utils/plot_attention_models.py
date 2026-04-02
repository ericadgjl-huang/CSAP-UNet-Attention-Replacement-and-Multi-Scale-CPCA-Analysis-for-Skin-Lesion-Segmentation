# utils/plot_attention_models.py

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8")

# 1. log 檔路徑
log_files = {
    "CBAM":           "logs/CSAP_S_CBAM_v3_S.csv",
    "CPCA_3_5_7":     "logs/CSAP_S_CPCA_3_5_7_v3_S.csv",
    "CPCA_7_11_21":   "logs/CSAP_S_CPCA_7_11_21_v3_S.csv",
    "CPCA_11_21_41":  "logs/CSAP_S_CPCA_11_21_41_v3_S.csv",
}

# 讀取 log，只取 validation
hist = {}
for name, path in log_files.items():
    df = pd.read_csv(path)
    df_val = df[df["split"] == "val"].copy().sort_values("epoch")
    hist[name] = df_val

# -------------------------------------------------------
#  畫圖：Dice / IoU / Loss （不包含 Time）
# -------------------------------------------------------
plt.figure(figsize=(14, 12))

# ========= 1. Dice =========
ax1 = plt.subplot(3, 1, 1)
for name, df in hist.items():
    ax1.plot(df["epoch"], df["dice"], label=name, linewidth=1.8)
ax1.set_ylabel("Validation Dice")
ax1.set_title("Validation Dice Curves (v3)")
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.legend()

# ========= 2. IoU =========
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
for name, df in hist.items():
    ax2.plot(df["epoch"], df["iou"], label=name, linewidth=1.8)
ax2.set_ylabel("Validation IoU")
ax2.set_title("Validation IoU Curves (v3)")
ax2.grid(True, linestyle="--", alpha=0.4)

# ========= 3. Loss =========
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
for name, df in hist.items():
    ax3.plot(df["epoch"], df["loss"], label=name, linewidth=1.8)
ax3.set_ylabel("Validation Loss")
ax3.set_title("Validation Loss Curves (v3)")
ax3.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("fig_v3_attention_models_without_time.png", dpi=300)
print("Saved fig_v3_attention_models_without_time.png")

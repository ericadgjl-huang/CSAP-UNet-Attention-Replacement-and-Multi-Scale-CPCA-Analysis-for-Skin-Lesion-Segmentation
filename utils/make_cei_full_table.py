# utils/make_cei_full_table.py

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import math

models = [
    {
        "Model": "S_CBAM_v3",
        "Attention": "CBAM",
        "Kernels": "-",
        "Dice": 0.8415,
        "IoU": 0.7552,
        "Params_M": 28.731,
        "FLOPs_GF": 138.150,
        "Time_ms": 1.6204,
    },
    {
        "Model": "S_CPCA_1_3_5_v3",
        "Attention": "CPCA",
        "Kernels": "1,3,5",
        "Dice": 0.8430,
        "IoU": 0.7582,
        "Params_M": 28.998,
        "FLOPs_GF": 139.732,
        "Time_ms": 1.5725,
    },
    {
        "Model": "S_CPCA_3_5_7_v3",
        "Attention": "CPCA",
        "Kernels": "3,5,7",
        "Dice": 0.8445,
        "IoU": 0.7592,
        "Params_M": 29.041,
        "FLOPs_GF": 140.260,
        "Time_ms": 1.5904,
    },
    {
        "Model": "S_CPCA_7_11_21_v3",
        "Attention": "CPCA",
        "Kernels": "7,11,21",
        "Dice": 0.8467,
        "IoU": 0.7635,
        "Params_M": 29.514,
        "FLOPs_GF": 146.073,
        "Time_ms": 2.8877,
    },
    {
        "Model": "S_CPCA_11_21_41_v3",
        "Attention": "CPCA",
        "Kernels": "11,21,41",
        "Dice": 0.8453,
        "IoU": 0.7625,
        "Params_M": 30.977,
        "FLOPs_GF": 164.042,
        "Time_ms": 5.0220,
    },
    {
        "Model": "S_HYBA_135_v3",
        "Attention": "Hybrid-A",
        "Kernels": "CNN: CBAM / TR: CPCA(1,3,5)",
        "Dice": 0.8522,
        "IoU": 0.7677,
        "Params_M": 28.865,
        "FLOPs_GF": 138.941,
        "Time_ms": 1.5719,
    },
    {
        "Model": "S_HYBB_135_v3",
        "Attention": "Hybrid-B",
        "Kernels": "CNN: CPCA(1,3,5) / TR: CBAM",
        "Dice": 0.8539,
        "IoU": 0.7707,
        "Params_M": 28.865,
        "FLOPs_GF": 138.941,
        "Time_ms": 1.5719,
    },
    {
        "Model": "S_HYBA_357_v3",
        "Attention": "Hybrid-A",
        "Kernels": "CNN: CBAM / TR: CPCA(3,5,7)",
        "Dice": 0.8405,
        "IoU": 0.7563,
        "Params_M": 28.886,
        "FLOPs_GF": 139.205,
        "Time_ms": 1.6398,
    },
    {
        "Model": "S_HYBB_357_v3",
        "Attention": "Hybrid-B",
        "Kernels": "CNN: CPCA(3,5,7) / TR: CBAM",
        "Dice": 0.8381,
        "IoU": 0.7530,
        "Params_M": 28.886,
        "FLOPs_GF": 139.205,
        "Time_ms": 1.6091,
    },
    {
        "Model": "S_HYBA_71121_v3",
        "Attention": "Hybrid-A",
        "Kernels": "CNN: CBAM / TR: CPCA(7,11,21)",
        "Dice": 0.8461,
        "IoU": 0.7603,
        "Params_M": 29.123,
        "FLOPs_GF": 142.112,
        "Time_ms": 2.7103,
    },
    {
        "Model": "S_HYBB_71121_v3",
        "Attention": "Hybrid-B",
        "Kernels": "CNN: CPCA(7,11,21) / TR: CBAM",
        "Dice": 0.8397,
        "IoU": 0.7543,
        "Params_M": 29.123,
        "FLOPs_GF": 142.112,
        "Time_ms": 1.7687,
    },
]

df = pd.DataFrame(models)
df.to_csv("ablation_cei_full_v3.csv", index=False)
print("Saved ablation_cei_full_v3.csv")
print(df)

df = pd.DataFrame(models)

# 用公式計算 CEI = Dice / sqrt( Params * Time )
df["CEI"] = df.apply(
    lambda row: row["Dice"] / math.sqrt(row["Params_M"] * row["Time_ms"]),
    axis=1,
)

# 四捨五入到小數點四位
df["CEI"] = df["CEI"].round(4)

# 存成 CSV
out_path = "ablation_cei_full_v3.csv"
df.to_csv(out_path, index=False)
print(f"Saved {out_path}")
print(df)
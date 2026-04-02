# utils/make_complexity_table.py

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

rows = [
    {
        "Model": "S_CBAM_v3",
        "Attention": "CBAM",
        "Kernels": "-",
        "Params_M": 28.731,
        "FLOPs_GF": 138.150,
        "Time_ms_mean": 1.6204,
        "Time_ms_std": 0.0140,
    },
    {
        "Model": "S_CPCA_1_3_5_v3",
        "Attention": "CPCA",
        "Kernels": "1,3,5",
        "Params_M": 28.998,
        "FLOPs_GF": 139.732,
        "Time_ms_mean": 1.5725,
        "Time_ms_std": 0.0316,
    },
    {
        "Model": "S_CPCA_3_5_7_v3",
        "Attention": "CPCA",
        "Kernels": "3,5,7",
        "Params_M": 29.041,
        "FLOPs_GF": 140.260,
        "Time_ms_mean": 1.5904,
        "Time_ms_std": 0.0219,
    },
    {
        "Model": "S_CPCA_7_11_21_v3",
        "Attention": "CPCA",
        "Kernels": "7,11,21",
        "Params_M": 29.514,
        "FLOPs_GF": 146.073,
        "Time_ms_mean": 2.8877,
        "Time_ms_std": 0.0176,
    },
    {
        "Model": "S_CPCA_11_21_41_v3",
        "Attention": "CPCA",
        "Kernels": "11,21,41",
        "Params_M": 30.977,
        "FLOPs_GF": 164.042,
        "Time_ms_mean": 5.0220,
        "Time_ms_std": 0.0196,
    },
    {
        "Model": "S_HYBA_135_v3",
        "Attention": "Hybrid-A",
        "Kernels": "CNN: CBAM / TR: CPCA(1,3,5)",
        "Params_M": 28.865,
        "FLOPs_GF": 138.941,
        "Time_ms_mean": 1.5719,
        "Time_ms_std": 0.0152,
    },
    {
        "Model": "S_HYBB_135_v3",
        "Attention": "Hybrid-B",
        "Kernels": "CNN: CPCA(1,3,5) / TR: CBAM",
        "Params_M": 28.865,
        "FLOPs_GF": 138.941,
        "Time_ms_mean": 1.5719,
        "Time_ms_std": 0.0053,
    },    
    {
        "Model": "S_HYBA_357_v3",
        "Attention": "Hybrid-A",
        "Kernels": "CNN: CBAM / TR: CPCA(3,5,7)",
        "Params_M": 28.886,
        "FLOPs_GF": 139.205,
        "Time_ms_mean": 1.6398,
        "Time_ms_std": 0.0322,
    },
    {
        "Model": "S_HYBB_357_v3",
        "Attention": "Hybrid-B",
        "Kernels": "CNN: CPCA(3,5,7) / TR: CBAM",
        "Params_M": 28.886,
        "FLOPs_GF": 139.205,
        "Time_ms_mean": 1.6091,
        "Time_ms_std": 0.0276,
    },
    {
        "Model": "S_HYBA_71121_v3",
        "Attention": "Hybrid-A",
        "Kernels": "CNN: CBAM / TR: CPCA(7,11,21)",
        "Params_M": 29.123,
        "FLOPs_GF": 142.112,
        "Time_ms_mean": 2.7103,
        "Time_ms_std": 0.0023,
    },
    {
        "Model": "S_HYBB_71121_v3",
        "Attention": "Hybrid-B",
        "Kernels": "CNN: CPCA(7,11,21) / TR: CBAM",
        "Params_M": 29.123,
        "FLOPs_GF": 142.112,
        "Time_ms_mean": 1.7687,
        "Time_ms_std": 0.0024,
    },
]

df = pd.DataFrame(rows)
df.to_csv("ablation_complexity_v3.csv", index=False)
print("Saved ablation_complexity_v3.csv")
print(df)

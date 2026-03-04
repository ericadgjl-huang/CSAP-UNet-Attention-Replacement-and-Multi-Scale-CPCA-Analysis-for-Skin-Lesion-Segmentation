# CSAP-UNet 注意力模組替換與多尺度 CPCA 設計之效能分析：以 ISIC 皮膚病灶分割為例
**CSAP-UNet Attention Replacement and Multi-Scale CPCA Analysis for ISIC 2017 Skin Lesion Segmentation**

這是一個基於 CSAP-UNet 模型並針對其注意力機制（Attention Mechanism）進行改良的皮膚病灶分割（Skin Lesion Segmentation）研究專案。本研究以通道主導之 **CPCA（Channel Prior Convolutional Attention）** 取代原始模型中的 **CBAM（Convolutional Block Attention Module）**，並進一步探討多尺度卷積與 Hybrid 注意力架構之影響。

---

## 📖 摘要 (Abstract)
本專案針對皮膚病灶分割任務中，病灶大小變異大、邊界模糊等挑戰，提出以多尺度卷積構成的 **CPCA** 模組取代 CSAP-UNet 中的 CBAM 模組。
* **多尺度卷積探討**：分析 (1,3,5)、(3,5,7)、(7,11,21)、(11,21,41) 對受感野與分割效果之影響。
* **Hybrid 注意力架構**：設計 Hybrid-A 與 Hybrid-B，將 CNN 與 Transformer 分支搭配不同注意力模組，探討互補性。
* **強韌性測試 (Robustness)**：除 ISIC 2017 測試外，進行 ISIC 2016（跨年份測試）、高斯雜訊 (Gaussian noise) 與 低解析度劣化 (Low-resolution) 測試。

實驗證明，**Hybrid-B（CNN 採用 CPCA，Transformer 採用 CBAM）** 在不增加計算量的情況下，達到最佳分割效能（Dice: 0.8539），且在影像劣化情境下展現極高的穩定性。

---

## 🎯 研究動機與目的
1. **提升注意力敏感度**：原 CBAM 僅依賴全域池化，難以捕捉皮膚斑塊邊界細節；CPCA 能透過多尺度卷積建立跨尺度的特徵關聯。
2. **分析多尺度效果**：系統性比較不同卷積尺度對分割結果與參數量、計算成本 (FLOPs) 的權衡。
3. **優化 CNN-Transformer 雙分支**：探討並行架構中，Transformer 提供全域資訊與 CNN 提供局部細節的最適注意力搭配。
4. **驗證臨床強韌性**：透過跨資料集 (Domain Shift) 與雜訊/解析度劣化測試，評估模型於真實場景的泛化能力。

---

## 🏗️ 方法架構設計
本專案以 CSAP-UNet 為基底，重點改良其 **Attention Fusion Module (AFM)**。

### 1. CPCA 模組替換
將 AFM 內部兩條路徑的 CBAM 替換為 CPCA。CPCA 使用多組並行的卷積核（如小、中、大尺度）來直接從特徵圖學習局部高頻與低頻形狀。
* 最佳單一 CPCA 配置：`CPCA(7,11,21)`。

### 2. Hybrid 注意力架構
* **Hybrid-A**：CNN 使用 CBAM、Transformer 使用 CPCA。
* **Hybrid-B (最佳)**：CNN 使用 CPCA、Transformer 使用 CBAM。此配置讓 CNN 專注於 CPCA 帶來的高頻邊界增強，而 Transformer 保留原始 CBAM 的全域穩健性，形成完美互補。

---

## 📊 實驗結果與效能

### 1. 主要指標比較 (ISIC 2017 Test Set)
| 模型 (Model) | 卷積尺度 | Params (M) | FLOPs (G) | Inference Time (ms) | Dice |
|-------------|---------|------------|-----------|---------------------|------|
| **Baseline (CBAM)** | - | 28.73 | 138.15 | 1.62 | 0.8415 |
| **CPCA_7_11_21** | 7, 11, 21 | 29.51 | 146.07 | 2.89 | 0.8467 |
| **Hybrid-B_135** | 1, 3, 5 (CNN) | 28.86 | 138.94 | 1.57 | **0.8539** |

*Hybrid-B 在參數量與推論時間與 Baseline 幾乎一致的情況下，顯著提升了分割精確度。*

### 2. 強韌性測試 (Robustness)
* **Cross-year (ISIC 2016)**：所有組合皆達 0.96~0.97 Dice，展現優秀的泛化能力。
* **Gaussian Noise**：Hybrid-A 展現出最高的抗噪能力，因 CBAM 不會過度放大局部噪聲。
* **Low-Resolution**：Hybrid-B 對於降採樣（128x128）劣化最具抗性，能藉由 CPCA 重建模糊的邊界特徵，Dice 甚至微幅上升至 0.8542。

---

## 📂 專案檔案結構 (Project Structure)
以下為過濾掉暫存檔與資料集後的核心程式碼結構：

```
CSAP-UNet/
├── lib/                        # 模型核心架構與網路定義模組
├── dataset_isic_v3.py          # ISIC 資料集 Dataset Loader 實作
├── train_csap_isic_v3.py       # 模型訓練主程式
├── test_csap_isic_v3.py        # 模型推論與指標計算測試程式
├── process_isic_to_npy_v3.py   # 資料前處理與 .npy 轉換腳本
├── process_isic2016_to_npy.py  # ISIC 2016 測試資料轉換腳本
├── make_robust_sets.py         # 產生高斯雜訊與低解析度測試資料之腳本
├── measure_complexity_v3.py    # 計算參數量與 FLOPs 之測試腳本
├── plot_*.py / make_*.py       # 繪圖與結果表格生成工具腳本
└── .gitignore                  # Git 忽略設定擋
```

---

## 📝 結論
本專案成功證明了將 CSAP-UNet 的 CBAM 替換為 CPCA（多尺度卷積）能有效增進醫療影像分割的效能。其中 **Hybrid-B 架構** 在運算成本不變的情況下取得最優越的精度，而模型在各種退化測試中亦展現強大的臨床應用潛力。

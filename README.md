# 🏭 Die Casting Defect Prediction

<div align="center">

**Real-time defect prediction and root cause analysis for die casting manufacturing processes**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-F1--Score%2079-success?style=flat-square)](https://lightgbm.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-orange?style=flat-square)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

*Team 일오나!! — Data Analysis Bootcamp 10th Cohort, Advanced Project Team 15*

</div>

---

## 📌 Overview

Die casting factories currently rely on **manual visual inspection** after each casting shot — a slow, inconsistent, and reactive process. This project replaces that with a **machine learning pipeline** that:

- Predicts defects **in real time** using process and sensor data
- Identifies the **root cause variables** behind each defect via SHAP
- Outputs a structured **4-step decision report** for operators

> **"불량은 한 번에 잡는다"** — Catch defects before they leave the line.

---

## 🎯 Problem & Solution

| | Conventional Inspection | ML-Based Inspection |
|---|---|---|
| **Method** | Manual visual check after casting | Real-time sensor + process data analysis |
| **Timing** | Post-production (reactive) | During production (proactive) |
| **Consistency** | Varies with operator fatigue | Consistent 24/7 |
| **Defect coverage** | Visible surface defects only | All defect types including internal |
| **Speed** | ~1 shot/min manual review | 100× faster |

---

## 📊 Dataset

| Field | Value |
|---|---|
| **Source** | 주조 품질보증 AI 데이터셋 — KAMP (Korea AI Manufacturing Platform, 2022) |
| **File** | `DieCasting_Quality_Raw_Data.csv` |
| **Size** | 7,535 rows × 57 columns |
| **Structure** | MultiIndex: `Process` / `Sensor` / `Defects` |

### Feature Groups

**Process Variables (Independent)**
| Category | Features |
|---|---|
| Velocity | `Velocity_1`, `Velocity_2`, `Velocity_3`, `High_Velocity` |
| Time | `Spray_Time`, `Rapid_Rise_Time`, `Pressure_Rise_Time`, `Cycle_Time` |
| Pressure | `Cylinder_Pressure`, `Casting_Pressure` |
| Force | `Clamping_Force` |
| Other | `Biscuit_Thickness` |

**Sensor Variables (Independent)**
| Category | Features |
|---|---|
| Temperature | `Factory_Temp`, `Coolant_Temp`, `Melting_Furnace_Temp` |
| Humidity | `Factory_Humidity` |
| Pressure | `Coolant_Pressure`, `Air_Pressure` |

**Defect Groups (Target / Dependent)**
| Group | Defect Types |
|---|---|
| 충전불량 (Fill Defect) | Short Shot |
| 기포/내부 (Void/Internal) | Bubble, Blow Hole |
| 표면손상 (Surface Damage) | Dent, Scratch, Stain, Burning Mark, Exfoliation |
| 기타 (Other) | Crack, Deformation, Impurity, Contamination, Inclusions |

---

## 🔬 Methodology

```
Raw Data (7,535 rows × 57 cols)
        │
        ▼
┌─────────────────────────────┐
│         PREPROCESSING       │
│  • Sensor null → median fill│
│  • Defects → binary (0/1)   │
│  • Group defect types (×4)  │
│  • Drop id / Shot / Type    │
│  • SMOTE (기타 group 1:1)    │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│             EDA             │
│  • Histogram per variable   │
│  • Correlation heatmaps ×3  │
│  • Z-score outlier analysis │
│  • Independent 2-sample     │
│    t-test per defect type   │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│          MODELING           │
│  Random Forest              │
│  XGBoost                    │
│  LightGBM ✓ (selected)      │
│                             │
│  Selection basis:           │
│  PR-AUC + ROC-AUC curves    │
│  Threshold sweep (0.6~0.8)  │
│  → threshold = 0.68         │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│       INTERPRETABILITY      │
│  SHAP TreeExplainer         │
│  • Global Bar / Beeswarm    │
│  • Local Waterfall plot     │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│       DECISION LOGIC        │
│  ① Diagnose (defect prob %) │
│  ② Track   (SHAP Top-N)     │
│  ③ Compare (vs normal med.) │
│  ④ Judge   (risk / safe)    │
└─────────────────────────────┘
```

---

## 📈 Results

### Model Comparison (Threshold = 0.68)

|  | RandomForest | XGBoost | **LightGBM** |
|---|---|---|---|
| 충전불량 F1 | 0.588 | 0.664 | **0.789** |
| 기포/내부 F1 | 0.545 | 0.695 | **0.800** |
| 표면손상 F1 | 0.591 | 0.719 | **0.782** |
| 기타 F1 | 0.200 | 0.269 | **0.264** |
| **Average F1** | ~0.48 | ~0.59 | **~0.66** |

### LightGBM Final Performance

| Defect Group | Accuracy | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|---|
| 충전불량 | 0.9688 | 0.9888 | 0.6567 | 0.7892 | 1 | 46 |
| 기포/내부 | 0.9794 | 0.9254 | 0.7045 | 0.8000 | 5 | 26 |
| 표면손상 | 0.9748 | 0.9189 | 0.6800 | 0.7816 | 6 | 32 |
| 기타 | 0.9741 | 0.4667 | 0.1842 | 0.2642 | 8 | 31 |

> **Why LightGBM?** Higher PR-AUC across all groups under data imbalance. Lower FP count vs XGBoost → fewer unnecessary re-inspections → lower operational cost.

### Key SHAP Variables (충전불량 기준)

| Rank | Variable | Direction |
|---|---|---|
| 1 | `Process\|High_Velocity` | ↑ high → defect risk |
| 2 | `Process\|Clamping_Force` | ↑ high → defect risk |
| 3 | `Sensor\|Melting_Furnace_Temp` | ↑ high → defect risk |
| 4 | `Process\|Spray_2_Time` | ↑ high → defect risk |
| 5 | `Process\|Casting_Pressure` | varies |

---

## 💰 Business Impact

| Metric | Value |
|---|---|
| FN loss (missed defect) | ₩3,000 / unit |
| FP loss (false alarm) | ₩20,000 / unit |
| Estimated savings (10 days) | **₩3,503,430** |
| Inspection speed improvement | **100×** |
| Defect catch rate improvement | **+~15%p** vs manual |
| Task time reduction | **50%** |

---

## 🗂️ Project Structure

```
die-casting-defect-prediction/
│
├── 다이캐스팅_불량예측_v2.ipynb   # Main notebook
├── DieCasting_Quality_Raw_Data.csv # Dataset
├── README.md
│
└── sections/
    ├── 0_setup.py        # Libraries & font config
    ├── 1_load.py         # Data loading
    ├── 2_eda.py          # EDA (histogram / heatmap / t-test)
    ├── 3_preprocess.py   # Preprocessing & SMOTE
    ├── 4_modeling.py     # RF / XGB / LGBM training
    ├── 5_shap.py         # SHAP interpretation
    ├── 6_decision.py     # Decision Logic report
    └── 7_business.py     # Business Impact calculation
```

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/your-username/die-casting-defect-prediction.git
cd die-casting-defect-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
xgboost>=1.7.0
lightgbm>=3.3.0
shap>=0.41.0
scipy>=1.10.0
jupyter>=1.0.0
```

---

## 🚀 Quick Start

```python
# 1. Open the notebook
jupyter notebook 다이캐스팅_불량예측_v2.ipynb

# 2. Run all cells top to bottom (Kernel → Restart & Run All)

# 3. Use the Decision Logic function directly
decision_logic("충전불량", sample_idx=0, top_n=5)
```

**Sample output:**
```
==========================================================
  [① 진단]  그룹: 충전불량  |  샘플 #0
  불량 발생 확률: 72.4%  |  판정: 불량  |  실제: 불량

  [② 추적 → ③ 비교 → ④ 판정]  상위 원인 변수 Top 5
  순위  변수                                기준값    현재값      편차   판정
  ---------------------------------------------------------------------------
  1    High Velocity                       0.191     0.215    +12.6% ⚠ 위험
  2    Clamping Force                    379.000   410.000     +8.2% ✔ 안전
  3    Melting Furnace Temp              650.000   698.000     +7.4% ✔ 안전
  4    Spray 2 Time                       12.200    15.800    +29.5% ⚠ 위험
  5    Casting Pressure                  596.000   641.000     +7.6% ✔ 안전

  [기준] 정상군 중앙값 | [편차] (현재-기준)/기준×100 | |편차|>10% → 위험
```

---

## 🔍 Notebook Walkthrough

| Section | Description |
|---|---|
| **0. Setup** | Libraries, OS-aware Korean font detection, global constants |
| **1. Data Load** | MultiIndex CSV loading, null check, column grouping |
| **2. EDA** | Per-variable histograms, 3 correlation heatmaps, Z-score outlier analysis, t-test by defect type |
| **3. Preprocessing** | Sensor median imputation, defect binarization, 4-group mapping, SMOTE for imbalanced classes |
| **4. Modeling** | PR/ROC curve comparison, threshold sweep, RF/XGB/LGBM training, confusion matrix visualization, 5-fold cross-validation |
| **5. SHAP** | Global bar + beeswarm plots, local waterfall for single sample, feature importance table |
| **6. Decision Logic** | 4-step diagnosis report: defect probability → top-N cause variables → deviation from normal → risk judgment |
| **7. Business Impact** | FN/FP cost calculation, cumulative cost trend, defect catch rate comparison vs manual inspection |

---

## 🔮 Future Work

| Item | Detail |
|---|---|
| **ROI quantification** | Collect baseline (육안검사) data to enable accurate ROI calculation |
| **Multicollinearity check** | Apply VIF + Correlation-based feature pruning for better model stability |
| **Hyperparameter tuning** | Optuna-based Bayesian optimization instead of default params |
| **Feature Engineering** | Interaction terms (e.g., Velocity × Pressure), rolling statistics |
| **Pipeline automation** | MLflow tracking + model registry for production deployment |
| **Real-time dashboard** | Streamlit / Grafana dashboard for live shop-floor monitoring |

---

## 📚 References

- Ministry of SMEs and Startups & KAIST. (2022). *Casting quality assurance AI dataset*. [KAMP](https://www.kamp-ai.kr/)
- Uyan, T.Ç., et al. *Industry 4.0 Foundry Data Management and Supervised Machine Learning in Low-Pressure Die Casting Quality Improvement*. InterMetalcast 17, 414–429 (2023).
- Pechmann, A. & Kanli, S. *Towards Sustainable Manufacturing: Deployable Deep Learning for Automated Defect Detection in Aluminum Die-Cast X-Ray Inspection*. Appl. Sci. 2025, 15, 13134.
- Okuniewska, A., Perzyk, M., & Kozłowski, J. *Machine Learning Methods for diagnosing the causes of die-casting defects*. Computer Methods in Materials Science, 23(2), 45–56. (2023).

---

## 👥 Team

| Name | Role |
|---|---|
| 김동욱 | EDA, Preprocessing |
| 김동현 | Model Training, Evaluation |
| 배진욱 | SHAP Analysis, Decision Logic |
| 유호빈 | Business Impact, Dashboard |
| 최민정 | EDA, Preprocessing, Presentation |

---

<div align="center">

**Team 일오나!! — 불량은 한 번에 잡는다 🔧**

</div>

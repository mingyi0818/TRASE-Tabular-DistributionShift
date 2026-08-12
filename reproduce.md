# Reproduction Guide for TRASE Experiments

This guide provides step-by-step instructions to reproduce all experimental results reported in the paper.

## 1. Environment Setup

### 1.1 System Requirements
- OS: Windows 11 Professional (or compatible)
- Python: 3.13+
- CPU: Multi-core processor (tested on Xeon W7-2595X 24-core)
- RAM: 16GB+ (tested with 48GB DDR5)
- GPU: Not required (CPU-only experiments)

### 1.2 Install Dependencies

```bash
cd code
pip install -r requirements.txt
```

Key dependencies:
- numpy >= 1.24
- scipy >= 1.10
- scikit-learn >= 1.3
- xgboost >= 2.0
- lightgbm >= 4.0
- matplotlib >= 3.7

### 1.3 Data Preparation

Datasets are automatically downloaded or loaded from sklearn/UCI. The data loader (`data_loader.py`) handles:
- Telco Churn: Loaded from local CSV (Kaggle format)
- Bank Marketing: Loaded from UCI repository
- Heart Disease: Loaded from UCI repository
- Mushroom: Loaded from UCI repository
- Adult Income: Loaded from UCI repository
- Wine Quality: Loaded from UCI repository
- Dry Bean: Loaded from UCI repository
- sklearn Wine: Loaded from sklearn.datasets

## 2. Running Experiments

### 2.1 Main Experiment (Table 1, Table 2, Table 3)

**Command:**
```bash
cd code
python run_trase_main.py
```

**Description:** Runs all 7 methods (LR, RF, XGBoost, LightGBM, ExtraTrees, SimpleEns, TRASE) on 8 datasets × 3 split types × 5 seeds = 840 task combinations (630 successful runs, some dataset-split combinations not supported).

**Output:** `results/trase_main_results.json`

**Expected runtime:** ~30 minutes

**To resume from checkpoint:**
```bash
python run_trase_main.py --resume
```

### 2.2 Ablation Experiment (Table 4, Table 5)

**Command:**
```bash
python run_trase_ablation.py
```

**Description:** Tests 6 ablation variants (SingleXGB, DE, DE+VWF, DE+SC+FE, NoST, Full) on 5 datasets × 2 split types (temporal, group) × 5 seeds = 300 runs.

**Output:** `results/trase_ablation_results.json`

**Expected runtime:** ~15 minutes

### 2.3 Sensitivity Analysis (Table 7)

**Command:**
```bash
python run_trase_sensitivity.py
```

**Description:** Tests 4 hyperparameters (shift_threshold, vwf_temperature, st_confidence_thresholds, st_max_pseudo_ratio) with multiple values on 4 datasets × 3 seeds = 228 runs.

**Output:** `results/trase_sensitivity_results.json`

**Expected runtime:** ~45 minutes

**To resume from checkpoint:**
```bash
python run_trase_sensitivity.py --resume
```

### 2.4 Robustness Experiment (Table 8, Figure 5)

**Command:**
```bash
python run_trase_robustness.py
```

**Description:** Tests TRASE robustness to feature noise (6 levels: 0-30%) and missing values (6 levels: 0-30%) on 4 datasets using IID splits = 48 runs.

**Output:** `results/trase_robustness_results.json`

**Expected runtime:** ~6 minutes

### 2.5 Efficiency Experiment (Table 9, Figure 6)

**Command:**
```bash
python run_trase_efficiency.py
```

**Description:** Measures training time, inference time, and peak memory for 7 methods on 4 datasets using temporal splits = 28 runs.

**Output:** `results/trase_efficiency_results.json`

**Expected runtime:** ~5 minutes

## 3. Computing Statistics

### 3.1 Main Statistics

**Command:**
```bash
python compute_trase_stats.py
```

**Output:** Console output (also saved to `results/trase_stats_output.txt`)

**Produces:**
- Overall accuracy comparison (Table 1)
- Paired t-tests (Table 2)
- Per-split-type analysis (Table 3)
- Per-dataset win rate analysis
- F1 and AUC metrics
- Shift analysis (Table 6)
- Self-training analysis
- Ablation results (Tables 4, 5)

### 3.2 Additional Statistics

**Command:**
```bash
python compute_trase_all_stats.py
```

**Output:** `results/trase_all_stats.json`

**Produces:**
- Elasticity coefficients (Table 7)
- Robustness summary (Table 8)
- Efficiency summary (Table 9)

## 4. Generating Figures

**Command:**
```bash
python generate_trase_figures.py
```

**Output:** 6 PNG figures in `plots/` directory (300 DPI):
- `trase_fig1_architecture.png` - Framework architecture diagram
- `trase_fig2_performance.png` - Performance comparison bar charts
- `trase_fig3_ablation.png` - Ablation results bar chart
- `trase_fig4_sensitivity.png` - Parameter sensitivity line charts
- `trase_fig5_robustness.png` - Robustness analysis line charts
- `trase_fig6_efficiency.png` - Efficiency comparison bar charts

## 5. Data Traceability

Every number in the paper can be traced to specific result files:

| Paper Table | Source File | Computation |
|-------------|------------|-------------|
| Table 1 (Overall) | trase_main_results.json | Mean/std/CI of accuracy by method |
| Table 2 (t-tests) | trase_main_results.json | Paired t-test on per-scenario accuracies |
| Table 3 (Per-split) | trase_main_results.json | Filter by split_type, compute mean/std |
| Table 4 (Ablation) | trase_ablation_results.json | Mean/std/CI by variant |
| Table 5 (Ablation t-tests) | trase_ablation_results.json | Paired t-test on per-scenario accuracies |
| Table 6 (Shift analysis) | trase_main_results.json | Aggregate TRASE shift/ST metadata |
| Table 7 (Sensitivity) | trase_sensitivity_results.json + trase_all_stats.json | Elasticity from param sweep |
| Table 8 (Robustness) | trase_robustness_results.json + trase_all_stats.json | Degradation from corruption levels |
| Table 9 (Efficiency) | trase_efficiency_results.json + trase_all_stats.json | Mean time/memory across datasets |

## 6. Verification

To verify that all paper numbers match the experimental data:

```bash
python compute_trase_stats.py > results/trase_stats_output.txt
python compute_trase_all_stats.py
```

Cross-check the output against the tables in `TRASE_Paper.html`.

## 7. Reproducing Specific Results

### Heart/Temporal +16.96% improvement
```bash
python -c "
import json
with open('../results/trase_main_results.json') as f:
    data = json.load(f)
trase = [r['accuracy'] for r in data if r['dataset']=='heart' and r['split_type']=='temporal' and r['method']=='TRASE']
xgb = [r['accuracy'] for r in data if r['dataset']=='heart' and r['split_type']=='temporal' and r['method']=='XGBoost']
print(f'TRASE: {sum(trase)/len(trase):.4f}')
print(f'XGBoost: {sum(xgb)/len(xgb):.4f}')
print(f'Diff: {(sum(trase)/len(trase) - sum(xgb)/len(xgb)):+.4f}')
"
```

### Full Pipeline (All Experiments)
```bash
# Run everything in sequence
python run_trase_main.py
python run_trase_ablation.py
python run_trase_sensitivity.py
python run_trase_robustness.py
python run_trase_efficiency.py
python compute_trase_stats.py
python compute_trase_all_stats.py
python generate_trase_figures.py
```

Total expected runtime: ~100 minutes

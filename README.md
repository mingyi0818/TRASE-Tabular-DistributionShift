# TRASE: Test-time Refinement with Adaptive Self-training Ensemble for Robust Tabular Learning under Distribution Shift

This repository contains the source code and experimental data for the TRASE framework, as described in the paper:

> Jingyuan Zeng, Ming Zeng, Jianghong Guo, Chuanxian Jiang, Yafen Feng. "TRASE: Test-time Refinement with Adaptive Self-training Ensemble for Robust Tabular Learning under Distribution Shift." Submitted to International Journal of Machine Learning and Cybernetics (IJMLC), Springer.

## Overview

TRASE is a two-phase framework for robust tabular classification under distribution shift:

- **Phase 1 - Diverse Ensemble with Shift-Aware Features**: Trains 8 diverse models (4 algorithms × 2 hyperparameter sets) with shift-aware meta-features and validation-weighted fusion (VWF)
- **Phase 2 - Iterative Self-Training Refinement**: Uses curriculum-based pseudo-labeling to adapt the ensemble to the target domain, with global safety checks

Key features:
- Shift detection via per-feature Kolmogorov-Smirnov tests
- Enhancement safety gate to prevent degradation
- Rank-based softmax weighting with shift-adaptive temperature
- Target-similarity weighted validation
- Curriculum confidence thresholds (0.9 → 0.85 → 0.8)
- Global safety check ensuring improvement over best single model

## Repository Structure

```
07_Tabular_FewShot/
├── code/                          # Source code
│   ├── trase_core.py              # TRASE framework implementation
│   ├── run_trase_main.py          # Main experiment runner
│   ├── run_trase_ablation.py      # Ablation experiment runner
│   ├── run_trase_sensitivity.py   # Sensitivity analysis runner
│   ├── run_trase_robustness.py    # Robustness experiment runner
│   ├── run_trase_efficiency.py    # Efficiency experiment runner
│   ├── compute_trase_stats.py     # Statistics computation
│   ├── compute_trase_all_stats.py # Additional statistics
│   ├── generate_trase_figures.py  # Figure generation
│   ├── config.py                  # Configuration
│   ├── data_loader.py             # Data loading and preprocessing
│   ├── splits.py                  # Split strategies (IID, temporal, group)
│   ├── models_new.py              # Base model implementations
│   └── requirements.txt           # Python dependencies
├── results/                       # Experimental results (JSON/CSV)
│   ├── trase_main_results.json    # Main experiment results (630 runs)
│   ├── trase_ablation_results.json# Ablation results (300 runs)
│   ├── trase_sensitivity_results.json # Sensitivity results (228 runs)
│   ├── trase_robustness_results.json  # Robustness results (48 runs)
│   ├── trase_efficiency_results.json  # Efficiency results (28 runs)
│   ├── trase_all_stats.json       # Computed statistics
│   └── trase_stats_output.txt     # Statistics output log
├── plots/                         # Generated figures (PNG, 300 DPI)
│   ├── trase_fig1_architecture.png
│   ├── trase_fig2_performance.png
│   ├── trase_fig3_ablation.png
│   ├── trase_fig4_sensitivity.png
│   ├── trase_fig5_robustness.png
│   └── trase_fig6_efficiency.png
├── TRASE_Paper.html               # Paper manuscript
├── README.md                      # This file
└── reproduce.md                   # Reproduction guide
```

## Results

Running the commands above regenerates all metrics, the ablation study, and the manuscript figures locally under `results/` (which is **not** stored in this repository). Numerical results are intentionally **not** pre-published here to avoid disclosing unpublished findings; reviewers reproduce them by running the code.

## Environment Requirements

- Windows 11 Professional
- Python 3.13
- See `code/requirements.txt` for full dependencies

## Quick Start

```bash
# Install dependencies
pip install -r code/requirements.txt

# Run main experiment (630 runs, ~30 minutes)
cd code
python run_trase_main.py

# Run ablation experiment (300 runs, ~15 minutes)
python run_trase_ablation.py

# Compute statistics
python compute_trase_stats.py
python compute_trase_all_stats.py

# Generate figures
python generate_trase_figures.py
```

See [reproduce.md](reproduce.md) for detailed reproduction instructions.

## Datasets

| Dataset | Samples | Features | Classes | Source |
|---------|---------|----------|---------|--------|
| Telco Churn | 7,043 | 19 | 2 | Kaggle |
| Bank Marketing | 11,162 | 16 | 2 | UCI |
| Heart Disease | 303 | 13 | 2 | UCI |
| Mushroom | 8,124 | 22 | 2 | UCI |
| Adult Income | 32,561 | 14 | 2 | UCI |
| Wine Quality | 1,599 | 11 | 6 | UCI |
| Dry Bean | 13,611 | 16 | 7 | UCI |
| sklearn Wine | 178 | 13 | 3 | sklearn |

## License

This project is for academic research purposes only.

## Acknowledgments

This work was supported by the Guangdong Provincial Higher Education Teaching Reform Project (Grant No. 粤教高函〔2024〕9-989).

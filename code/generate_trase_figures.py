"""
TRASE Figure Generation.

Generates all figures for the TRASE paper:
  Figure 1: Algorithm architecture diagram
  Figure 2: Performance comparison (bar chart by split type)
  Figure 3: Ablation results (bar chart)
  Figure 4: Parameter sensitivity analysis (line charts)
  Figure 5: Robustness analysis (line charts)

All figures saved as PNG with >300 DPI to results/ or plots/ directory.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR
PLOT_DIR = os.path.join(os.path.dirname(RESULT_DIR), 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# Font setup for Chinese compatibility
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def load_json(filename):
    path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(path):
        print(f"WARNING: {filename} not found")
        return []
    with open(path) as f:
        return json.load(f)


# ============================================================
# Figure 1: Architecture Diagram
# ============================================================
def generate_architecture_figure():
    """Generate TRASE architecture diagram using matplotlib."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('TRASE: Test-time Refinement with Adaptive Self-training Ensemble',
                 fontsize=14, fontweight='bold', pad=20)

    # Color scheme
    c_input = '#E8F5E9'
    c_phase1 = '#E3F2FD'
    c_phase2 = '#FFF3E0'
    c_safety = '#FFEBEE'
    c_output = '#F3E5F5'
    c_border = '#455A64'

    def draw_box(x, y, w, h, text, color, fontsize=9, bold=False):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor=c_border, linewidth=1.5)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, wrap=True)

    def draw_arrow(x1, y1, x2, y2, style='->', color='#455A64'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.5))

    # Input data
    draw_box(0.3, 7.5, 2.2, 1.5, 'Source Domain\n(Train + Val)', c_input, fontsize=9, bold=True)
    draw_box(0.3, 5.0, 2.2, 1.5, 'Target Domain\n(Test)', c_input, fontsize=9, bold=True)

    # Phase 1 header
    draw_box(3.5, 8.5, 6.5, 1.0, 'Phase 1: Diverse Ensemble with Shift-Aware Features',
             c_phase1, fontsize=11, bold=True)

    # Phase 1 components
    draw_box(3.5, 6.8, 2.0, 1.2, 'Shift\nCharacterization\n(KS Test)', c_phase1, fontsize=8)
    draw_box(6.0, 6.8, 2.0, 1.2, 'Feature\nEnhancement\n(3 Meta-features)', c_phase1, fontsize=8)
    draw_box(8.5, 6.8, 1.5, 1.2, 'Enhancement\nSafety Gate', c_safety, fontsize=8)

    # Diverse Ensemble
    draw_box(3.5, 5.0, 6.5, 1.2, 'Diverse Ensemble (8 models: XGBoost, LightGBM, RF, ExtraTrees x2 configs)\n'
             'Views: Original | Enhanced | Non-shifted', c_phase1, fontsize=8, bold=True)

    # VWF
    draw_box(3.5, 3.5, 3.0, 1.0, 'Validation-Weighted\nFusion (VWF)', c_phase1, fontsize=9)
    draw_box(7.0, 3.5, 3.0, 1.0, 'Target-Weighted\nValidation Check', c_safety, fontsize=9)

    # Phase 2 header
    draw_box(3.5, 2.3, 6.5, 0.8, 'Phase 2: Iterative Self-Training Refinement',
             c_phase2, fontsize=11, bold=True)

    # Phase 2 components
    draw_box(3.5, 0.8, 1.5, 1.0, 'Pseudo-Label\nGeneration', c_phase2, fontsize=8)
    draw_box(5.3, 0.8, 1.5, 1.0, 'Curriculum\nSelection\n(0.9->0.8)', c_phase2, fontsize=8)
    draw_box(7.1, 0.8, 1.5, 1.0, 'Model\nRetraining', c_phase2, fontsize=8)
    draw_box(8.9, 0.8, 1.5, 1.0, 'Global\nSafety Check', c_safety, fontsize=8)

    # Output
    draw_box(11.0, 3.7, 2.5, 1.2, 'Final\nPrediction', c_output, fontsize=11, bold=True)

    # Arrows - Input to Phase 1
    draw_arrow(2.5, 8.0, 3.5, 7.4)  # Source to SC
    draw_arrow(2.5, 5.5, 3.5, 7.0)  # Target to SC

    # Phase 1 internal
    draw_arrow(5.5, 7.4, 6.0, 7.4)  # SC to FE
    draw_arrow(8.0, 7.4, 8.5, 7.4)  # FE to Safety Gate
    draw_arrow(4.5, 6.8, 4.5, 6.2)  # SC to DE
    draw_arrow(7.0, 6.8, 7.0, 6.2)  # FE to DE
    draw_arrow(9.2, 6.8, 9.2, 6.2)  # Safety Gate to DE

    # DE to VWF
    draw_arrow(6.0, 5.0, 5.0, 4.5)  # DE to VWF
    draw_arrow(7.5, 5.0, 8.5, 4.5)  # DE to Target-weighted check

    # VWF to Phase 2
    draw_arrow(5.0, 3.5, 5.0, 3.1)  # VWF to Phase 2
    draw_arrow(8.5, 3.5, 8.5, 3.1)  # Check to Phase 2

    # Phase 2 internal
    draw_arrow(5.0, 1.3, 5.3, 1.3)  # PL to Selection
    draw_arrow(6.8, 1.3, 7.1, 1.3)  # Selection to Retrain
    draw_arrow(8.6, 1.3, 8.9, 1.3)  # Retrain to Safety

    # Phase 2 feedback loop
    ax.annotate('', xy=(4.25, 0.8), xytext=(9.65, 0.4),
                arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5,
                                connectionstyle='arc3,rad=-0.3', linestyle='dashed'))
    ax.text(7.0, 0.15, 'Iterate (max 3)', ha='center', fontsize=7, color='#E65100', style='italic')

    # Phase 2 to output
    draw_arrow(10.4, 1.3, 11.0, 3.7)  # Safety to Output
    draw_arrow(6.5, 4.0, 11.0, 4.0)   # VWF to Output (Phase 1 fallback)

    # Phase labels on left
    ax.text(0.15, 8.0, 'Input', ha='center', va='center', fontsize=9,
            fontweight='bold', rotation=90, color='#2E7D32')
    ax.text(0.15, 5.5, 'Input', ha='center', va='center', fontsize=9,
            fontweight='bold', rotation=90, color='#2E7D32')
    ax.text(2.9, 6.0, 'P1', ha='center', va='center', fontsize=10,
            fontweight='bold', color='#1565C0')
    ax.text(2.9, 1.3, 'P2', ha='center', va='center', fontsize=10,
            fontweight='bold', color='#E65100')

    # Legend
    legend_y = 9.3
    ax.text(10.5, legend_y, 'Legend:', fontsize=8, fontweight='bold')
    for i, (label, color) in enumerate([('Data', c_input), ('Phase 1', c_phase1),
                                         ('Phase 2', c_phase2), ('Safety', c_safety),
                                         ('Output', c_output)]):
        rect = plt.Rectangle((10.5 + i*0.7, legend_y - 0.3), 0.3, 0.3,
                              facecolor=color, edgecolor=c_border, linewidth=0.5)
        ax.add_patch(rect)
        ax.text(10.5 + i*0.7 + 0.35, legend_y - 0.15, label, fontsize=6, va='center')

    plt.tight_layout()
    output_path = os.path.join(PLOT_DIR, 'trase_fig1_architecture.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# Figure 2: Performance Comparison
# ============================================================
def generate_performance_comparison():
    """Generate bar chart comparing all methods across split types."""
    main = load_json('trase_main_results.json')
    success = [r for r in main if 'accuracy' in r and r.get('status') == 'success']

    methods = ['LR', 'RF', 'XGBoost', 'LightGBM', 'ExtraTrees', 'SimpleEns', 'TRASE']
    splits = ['iid', 'temporal', 'group']
    split_labels = ['IID', 'Temporal', 'Group']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    colors = ['#90CAF9', '#66BB6A', '#FFA726', '#AB47BC', '#26C6DA', '#78909C', '#E53935']

    for ax_idx, (split, split_label) in enumerate(zip(splits, split_labels)):
        ax = axes[ax_idx]
        means = []
        stds = []
        for m in methods:
            accs = [r['accuracy'] for r in success if r['method'] == m and r['split_type'] == split]
            means.append(np.mean(accs) if accs else 0)
            stds.append(np.std(accs) if accs else 0)

        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                       edgecolor='black', linewidth=0.5, width=0.7)

        # Highlight TRASE bar
        bars[-1].set_edgecolor('#B71C1C')
        bars[-1].set_linewidth(2)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        ax.set_title(split_label, fontsize=12, fontweight='bold')
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis='y', alpha=0.3)

        if ax_idx == 0:
            ax.set_ylabel('Accuracy', fontsize=11)

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle('Performance Comparison Across Split Types', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(PLOT_DIR, 'trase_fig2_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# Figure 3: Ablation Results
# ============================================================
def generate_ablation_figure():
    """Generate ablation results bar chart."""
    ablation = load_json('trase_ablation_results.json')
    success = [r for r in ablation if 'accuracy' in r and r.get('status') == 'success']

    variants = ['SingleXGB', 'DE', 'DE+VWF', 'DE+SC+FE', 'NoST', 'Full']
    variant_labels = ['Single\nXGBoost', 'DE', 'DE+VWF', 'DE+SC\n+FE', 'NoST\n(Phase 1)', 'Full\nTRASE']
    colors = ['#BDBDBD', '#90CAF9', '#66BB6A', '#FFA726', '#AB47BC', '#E53935']

    fig, ax = plt.subplots(figsize=(10, 6))

    means = []
    stds = []
    cis_low = []
    cis_high = []
    for v in variants:
        accs = [r['accuracy'] for r in success if r['variant'] == v]
        means.append(np.mean(accs) if accs else 0)
        stds.append(np.std(accs) if accs else 0)
        if accs:
            from scipy import stats as sp_stats
            ci = sp_stats.t.interval(0.95, len(accs)-1, loc=np.mean(accs), scale=sp_stats.sem(accs))
            cis_low.append(means[-1] - ci[0])
            cis_high.append(ci[1] - means[-1])
        else:
            cis_low.append(0)
            cis_high.append(0)

    x = np.arange(len(variants))
    bars = ax.bar(x, means, yerr=[cis_low, cis_high], capsize=5, color=colors,
                   edgecolor='black', linewidth=0.5, width=0.6)

    # Highlight Full TRASE
    bars[-1].set_edgecolor('#B71C1C')
    bars[-1].set_linewidth(2)

    ax.set_xticks(x)
    ax.set_xticklabels(variant_labels, fontsize=9)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Ablation Study: Component Contributions (Temporal + Group Splits)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0.75, 0.85)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Add arrows showing improvements
    for i in range(1, len(variants)):
        if means[i] > means[i-1]:
            ax.annotate('', xy=(i, means[i] - 0.002), xytext=(i-1, means[i-1] - 0.002),
                        arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
        elif means[i] < means[i-1]:
            ax.annotate('', xy=(i, means[i] - 0.002), xytext=(i-1, means[i-1] - 0.002),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    plt.tight_layout()
    output_path = os.path.join(PLOT_DIR, 'trase_fig3_ablation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# Figure 4: Parameter Sensitivity
# ============================================================
def generate_sensitivity_figure():
    """Generate parameter sensitivity analysis figure."""
    sens = load_json('trase_sensitivity_results.json')
    success = [r for r in sens if r.get('status') == 'success']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    params_info = [
        ('shift_threshold', 'Shift Threshold', [0.05, 0.08, 0.1, 0.15, 0.2],
         'Default: 0.1'),
        ('vwf_temperature', 'VWF Temperature', [0.005, 0.01, 0.02, 0.05, 0.1],
         'Default: 0.1'),
        ('st_confidence_thresholds', 'ST Confidence Thresholds',
         ['[0.95,0.90,0.85]', '[0.90,0.85,0.80]', '[0.85,0.80,0.75]', '[0.80,0.75,0.70]'],
         'Default: [0.90,0.85,0.80]'),
        ('st_max_pseudo_ratio', 'ST Max Pseudo Ratio', [0.2, 0.3, 0.4, 0.5, 0.6],
         'Default: 0.5'),
    ]

    for ax_idx, (param_name, param_label, param_values, default_note) in enumerate(params_info):
        ax = axes[ax_idx]

        param_results = [r for r in success if r.get('param_name') == param_name]
        if not param_results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        # Group by param_value_idx
        means = []
        stds = []
        for idx in range(len(param_values)):
            accs = [r['accuracy'] for r in param_results if r.get('param_value_idx') == idx]
            means.append(np.mean(accs) if accs else 0)
            stds.append(np.std(accs) if accs else 0)

        x = np.arange(len(param_values))

        # For st_confidence_thresholds, use string labels
        if isinstance(param_values[0], str):
            ax.errorbar(x, means, yerr=stds, marker='o', capsize=4, linewidth=2,
                       markersize=8, color='#1565C0')
            ax.set_xticks(x)
            ax.set_xticklabels(param_values, fontsize=7, rotation=15)
        else:
            ax.errorbar(x, means, yerr=stds, marker='o', capsize=4, linewidth=2,
                       markersize=8, color='#1565C0')
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in param_values])

        # Mark default value
        defaults = {'shift_threshold': 0.1, 'vwf_temperature': 0.1,
                    'st_max_pseudo_ratio': 0.5}
        if param_name in defaults:
            default_idx = param_values.index(defaults[param_name])
            ax.axvline(x=default_idx, color='red', linestyle='--', alpha=0.5, label='Default')
            ax.plot(default_idx, means[default_idx], 'r*', markersize=15, zorder=5)

        ax.set_xlabel(param_label, fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_title(f'{param_label}\n({default_note})', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)

        # Set y-axis to show variation
        y_min = min(means) - max(stds) - 0.01
        y_max = max(means) + max(stds) + 0.01
        ax.set_ylim(y_min, y_max)

    fig.suptitle('Parameter Sensitivity Analysis (Temporal Split, 4 Datasets x 3 Seeds)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(PLOT_DIR, 'trase_fig4_sensitivity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# Figure 5: Robustness Analysis
# ============================================================
def generate_robustness_figure():
    """Generate robustness analysis figure."""
    rob = load_json('trase_robustness_results.json')
    success = [r for r in rob if r.get('status') == 'success']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    datasets = ['telco', 'bank', 'heart', 'mushroom']
    colors = ['#1565C0', '#2E7D32', '#E65100', '#6A1B9A']
    markers = ['o', 's', '^', 'D']

    # Noise robustness
    ax = axes[0]
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    for ds_idx, ds in enumerate(datasets):
        means = []
        stds = []
        for level in noise_levels:
            accs = [r['accuracy'] for r in success
                    if r.get('dataset') == ds
                    and r.get('corruption_type') == 'noise'
                    and r.get('corruption_level') == level]
            means.append(np.mean(accs) if accs else 0)
            stds.append(np.std(accs) if accs else 0)

        ax.errorbar(noise_levels, means, yerr=stds, marker=markers[ds_idx],
                    capsize=3, linewidth=2, markersize=7, color=colors[ds_idx], label=ds)

    ax.set_xlabel('Noise Level (std)', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Robustness to Feature Noise', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 1.05)

    # Missing values robustness
    ax = axes[1]
    missing_ratios = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    for ds_idx, ds in enumerate(datasets):
        means = []
        stds = []
        for level in missing_ratios:
            accs = [r['accuracy'] for r in success
                    if r.get('dataset') == ds
                    and r.get('corruption_type') == 'missing'
                    and r.get('corruption_level') == level]
            means.append(np.mean(accs) if accs else 0)
            stds.append(np.std(accs) if accs else 0)

        ax.errorbar(missing_ratios, means, yerr=stds, marker=markers[ds_idx],
                    capsize=3, linewidth=2, markersize=7, color=colors[ds_idx], label=ds)

    ax.set_xlabel('Missing Value Ratio', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Robustness to Missing Values', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 1.05)

    fig.suptitle('Robustness Analysis under Feature Corruption (IID Split)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(PLOT_DIR, 'trase_fig5_robustness.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# Figure 6: Efficiency Analysis
# ============================================================
def generate_efficiency_figure():
    """Generate efficiency analysis figure."""
    eff = load_json('trase_efficiency_results.json')
    success = [r for r in eff if r.get('status') == 'success']

    methods = ['LR', 'RF', 'XGBoost', 'LightGBM', 'ExtraTrees', 'SimpleEns', 'TRASE']
    colors = ['#90CAF9', '#66BB6A', '#FFA726', '#AB47BC', '#26C6DA', '#78909C', '#E53935']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Training time
    ax = axes[0]
    train_means = []
    train_stds = []
    for m in methods:
        vals = [r['train_time'] for r in success if r['method'] == m]
        train_means.append(np.mean(vals) if vals else 0)
        train_stds.append(np.std(vals) if vals else 0)

    x = np.arange(len(methods))
    bars = ax.bar(x, train_means, yerr=train_stds, capsize=3, color=colors,
                   edgecolor='black', linewidth=0.5)
    bars[-1].set_edgecolor('#B71C1C')
    bars[-1].set_linewidth(2)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Training Time (s)', fontsize=11)
    ax.set_title('Training Time', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, mean in zip(bars, train_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=7)

    # Inference time
    ax = axes[1]
    infer_means = []
    infer_stds = []
    for m in methods:
        vals = [r['infer_time'] for r in success if r['method'] == m]
        infer_means.append(np.mean(vals) if vals else 0)
        infer_stds.append(np.std(vals) if vals else 0)

    bars = ax.bar(x, infer_means, yerr=infer_stds, capsize=3, color=colors,
                   edgecolor='black', linewidth=0.5)
    bars[-1].set_edgecolor('#B71C1C')
    bars[-1].set_linewidth(2)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Inference Time (s)', fontsize=11)
    ax.set_title('Inference Time', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, mean in zip(bars, infer_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=7)

    # Memory usage
    ax = axes[2]
    mem_means = []
    mem_stds = []
    for m in methods:
        vals = [r['peak_memory_mb'] for r in success if r['method'] == m]
        mem_means.append(np.mean(vals) if vals else 0)
        mem_stds.append(np.std(vals) if vals else 0)

    bars = ax.bar(x, mem_means, yerr=mem_stds, capsize=3, color=colors,
                   edgecolor='black', linewidth=0.5)
    bars[-1].set_edgecolor('#B71C1C')
    bars[-1].set_linewidth(2)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Peak Memory (MB)', fontsize=11)
    ax.set_title('Peak Memory Usage', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, mean in zip(bars, mem_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle('Computational Efficiency Comparison (Temporal Split)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(PLOT_DIR, 'trase_fig6_efficiency.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    print("Generating TRASE figures...")
    print()

    print("Figure 1: Architecture Diagram")
    generate_architecture_figure()

    print("\nFigure 2: Performance Comparison")
    generate_performance_comparison()

    print("\nFigure 3: Ablation Results")
    generate_ablation_figure()

    print("\nFigure 4: Parameter Sensitivity")
    generate_sensitivity_figure()

    print("\nFigure 5: Robustness Analysis")
    generate_robustness_figure()

    print("\nFigure 6: Efficiency Analysis")
    generate_efficiency_figure()

    print("\nAll figures generated successfully!")


if __name__ == '__main__':
    main()

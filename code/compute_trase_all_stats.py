"""Compute all TRASE statistics and save to JSON."""
import json, os, numpy as np
from collections import defaultdict
from scipy import stats

results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

def load(name):
    path = os.path.join(results_dir, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []

# ============= Sensitivity/Elasticity =============
sens = load('trase_sensitivity_results.json')
success_sens = [r for r in sens if r.get('status') == 'success']

param_grid = {
    'shift_threshold': [0.05, 0.08, 0.1, 0.15, 0.2],
    'vwf_temperature': [0.005, 0.01, 0.02, 0.05, 0.1],
    'st_confidence_thresholds': [[0.95, 0.90, 0.85], [0.90, 0.85, 0.80], [0.85, 0.80, 0.75], [0.80, 0.75, 0.70]],
    'st_max_pseudo_ratio': [0.2, 0.3, 0.4, 0.5, 0.6],
}

elasticity_results = {}
for param_name, param_values in param_grid.items():
    param_accs = defaultdict(list)
    for r in success_sens:
        if r.get('param_name') == param_name:
            param_accs[r['param_value_idx']].append(r['accuracy'])

    mean_accs = []
    for idx in range(len(param_values)):
        accs = param_accs.get(idx, [])
        mean_accs.append(float(np.mean(accs)) if accs else 0.0)

    elasticities = []
    for i in range(1, len(param_values)):
        pv_prev = param_values[i-1]
        pv_curr = param_values[i]
        if isinstance(pv_prev, list):
            pv_prev = sum(pv_prev) / len(pv_prev)
        if isinstance(pv_curr, list):
            pv_curr = sum(pv_curr) / len(pv_curr)
        delta_param = pv_curr - pv_prev
        delta_perf = mean_accs[i] - mean_accs[i-1]
        if delta_param == 0 or mean_accs[i-1] == 0:
            elasticities.append(0.0)
            continue
        elasticity = abs((delta_perf / mean_accs[i-1]) / (delta_param / pv_prev))
        elasticities.append(elasticity)

    mean_elasticity = float(np.mean(elasticities)) if elasticities else 0.0
    if mean_elasticity > 0.5:
        sensitivity_level = 'High'
    elif mean_elasticity > 0.2:
        sensitivity_level = 'Medium'
    else:
        sensitivity_level = 'Low'

    best_idx = int(np.argmax(mean_accs))
    best_value = param_values[best_idx]

    elasticity_results[param_name] = {
        'param_range': f'{param_values[0]} to {param_values[-1]}',
        'best_value': str(best_value),
        'best_accuracy': mean_accs[best_idx],
        'mean_elasticity': mean_elasticity,
        'sensitivity_level': sensitivity_level,
        'mean_accuracies': mean_accs,
    }
    print(f'{param_name}: elasticity={mean_elasticity:.4f} -> {sensitivity_level}')
    print(f'  Best: {best_value} (acc={mean_accs[best_idx]:.4f})')
    print(f'  All accs: {[round(a, 4) for a in mean_accs]}')

# ============= Robustness Summary =============
rob = load('trase_robustness_results.json')
success_rob = [r for r in rob if r.get('status') == 'success']

noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
missing_ratios = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
datasets_rob = ['telco', 'bank', 'heart', 'mushroom']

robustness_summary = {}
for ds in datasets_rob:
    ds_results = {}
    for ctype, levels in [('noise', noise_levels), ('missing', missing_ratios)]:
        accs_by_level = {}
        for level in levels:
            accs = [r['accuracy'] for r in success_rob
                    if r.get('dataset') == ds
                    and r.get('corruption_type') == ctype
                    and r.get('corruption_level') == level]
            if accs:
                accs_by_level[str(level)] = {
                    'mean': float(np.mean(accs)),
                    'std': float(np.std(accs)),
                }
        ds_results[ctype] = accs_by_level

    noise_accs = [ds_results['noise'][str(l)]['mean'] for l in noise_levels]
    missing_accs = [ds_results['missing'][str(l)]['mean'] for l in missing_ratios]
    ds_results['noise_degradation'] = float(noise_accs[0] - noise_accs[-1])
    ds_results['missing_degradation'] = float(missing_accs[0] - missing_accs[-1])
    ds_results['noise_degradation_rate'] = float((noise_accs[0] - noise_accs[-1]) / noise_levels[-1])
    ds_results['missing_degradation_rate'] = float((missing_accs[0] - missing_accs[-1]) / missing_ratios[-1])
    robustness_summary[ds] = ds_results

print(f'\nRobustness Summary:')
for ds in datasets_rob:
    s = robustness_summary[ds]
    print(f'  {ds}: noise_deg={s["noise_degradation"]:.4f}, missing_deg={s["missing_degradation"]:.4f}')

# ============= Efficiency Summary =============
eff = load('trase_efficiency_results.json')
success_eff = [r for r in eff if r.get('status') == 'success']

methods = ['LR', 'RF', 'XGBoost', 'LightGBM', 'ExtraTrees', 'SimpleEns', 'TRASE']
efficiency_summary = {}
for m in methods:
    results = [r for r in success_eff if r['method'] == m]
    if results:
        efficiency_summary[m] = {
            'avg_accuracy': float(np.mean([r['accuracy'] for r in results])),
            'avg_train_time': float(np.mean([r['train_time'] for r in results])),
            'avg_infer_time': float(np.mean([r['infer_time'] for r in results])),
            'avg_peak_memory_mb': float(np.mean([r['peak_memory_mb'] for r in results])),
            'n_models': results[0].get('n_models', 1),
        }

print(f'\nEfficiency Summary:')
for m in methods:
    if m in efficiency_summary:
        e = efficiency_summary[m]
        print(f'  {m}: acc={e["avg_accuracy"]:.4f}, train={e["avg_train_time"]:.2f}s, '
              f'infer={e["avg_infer_time"]:.4f}s, mem={e["avg_peak_memory_mb"]:.1f}MB')

# ============= Save all =============
all_stats = {
    'elasticity': elasticity_results,
    'robustness': robustness_summary,
    'efficiency': efficiency_summary,
}

output_path = os.path.join(results_dir, 'trase_all_stats.json')
with open(output_path, 'w') as f:
    json.dump(all_stats, f, indent=2)
print(f'\nAll stats saved to: {output_path}')

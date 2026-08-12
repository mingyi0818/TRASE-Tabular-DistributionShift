"""
TRASE Statistics Computation.

Computes all statistics needed for the TRASE paper:
- Main results (overall, per-split, per-dataset)
- Paired t-tests
- F1 and AUC
- Shift analysis
- Self-training analysis
- Ablation results
"""
import os
import sys
import json
import numpy as np
from collections import defaultdict
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULT_DIR


def load(filename):
    path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(path):
        print(f"WARNING: {filename} not found")
        return []
    with open(path) as f:
        return json.load(f)


def main():
    print("=" * 80)
    print("TRASE Statistics")
    print("=" * 80)

    # === Main Results ===
    main = load('trase_main_results.json')
    success = [r for r in main if 'accuracy' in r and r.get('status') == 'success']

    print(f"\nTotal successful results: {len(success)}")
    methods = sorted(set(r['method'] for r in success))
    datasets = sorted(set(r['dataset'] for r in success))
    splits = sorted(set(r['split_type'] for r in success))
    print(f"Methods: {methods}")
    print(f"Datasets: {datasets}")
    print(f"Splits: {splits}")

    # Group by (dataset, split_type) -> method -> [accuracies]
    scenarios = defaultdict(lambda: defaultdict(list))
    for r in success:
        key = (r['dataset'], r['split_type'])
        scenarios[key][r['method']].append(r['accuracy'])

    # Overall
    method_overall = defaultdict(list)
    for r in success:
        method_overall[r['method']].append(r['accuracy'])

    print("\n=== Overall (all scenarios) ===")
    display_order = ['LR', 'RF', 'XGBoost', 'LightGBM', 'ExtraTrees', 'SimpleEns', 'TRASE']
    for m in display_order:
        accs = method_overall.get(m, [])
        if accs:
            mean = np.mean(accs)
            std = np.std(accs)
            ci = stats.t.interval(0.95, len(accs)-1, loc=mean, scale=stats.sem(accs))
            print("  %-12s: %.4f +/- %.4f  [95%% CI: %.4f, %.4f]  N=%d" %
                  (m, mean, std, ci[0], ci[1], len(accs)))

    # Paired t-tests
    print("\n=== Paired t-tests: TRASE vs baselines ===")
    trase_all = []
    for key in sorted(scenarios.keys()):
        trase_vals = scenarios[key].get('TRASE', [])
        trase_all.extend(trase_vals)

    for baseline in ['LR', 'RF', 'XGBoost', 'LightGBM', 'ExtraTrees', 'SimpleEns']:
        base_all = []
        for key in sorted(scenarios.keys()):
            base_vals = scenarios[key].get(baseline, [])
            base_all.extend(base_vals)

        if len(trase_all) == len(base_all) and len(trase_all) > 0:
            t_stat, p_val = stats.ttest_rel(trase_all, base_all)
            diff = np.mean(trase_all) - np.mean(base_all)
            cohen_d = diff / (np.std(np.array(trase_all) - np.array(base_all)) + 1e-8)
            sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
            print('  vs %-12s: diff=%+.4f t=%.4f p=%.2e d=%.4f %s' %
                  (baseline, diff, t_stat, p_val, cohen_d, sig))

    # Per Split Type
    print("\n=== Per Split Type ===")
    for split_type in ['iid', 'temporal', 'group']:
        print(f"\n  {split_type.upper()}:")
        split_results = defaultdict(list)
        for r in success:
            if r['split_type'] == split_type:
                split_results[r['method']].append(r['accuracy'])

        for m in display_order:
            accs = split_results.get(m, [])
            if accs:
                print("    %-12s: %.4f +/- %.4f" % (m, np.mean(accs), np.std(accs)))

        # TRASE vs XGBoost and SimpleEns
        trase_vals = split_results.get('TRASE', [])
        xgb_vals = split_results.get('XGBoost', [])
        ens_vals = split_results.get('SimpleEns', [])

        if len(trase_vals) == len(xgb_vals) and len(trase_vals) > 0:
            t_stat, p_val = stats.ttest_rel(trase_vals, xgb_vals)
            diff = np.mean(trase_vals) - np.mean(xgb_vals)
            sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
            print("    TRASE vs XGB: diff=%+.4f t=%.4f p=%.4f %s" % (diff, t_stat, p_val, sig))

        if len(trase_vals) == len(ens_vals) and len(trase_vals) > 0:
            t_stat, p_val = stats.ttest_rel(trase_vals, ens_vals)
            diff = np.mean(trase_vals) - np.mean(ens_vals)
            sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
            print("    TRASE vs Ens: diff=%+.4f t=%.4f p=%.4f %s" % (diff, t_stat, p_val, sig))

    # Per-Dataset
    print("\n=== Per-Dataset: TRASE vs XGBoost ===")
    win_count = 0
    total_count = 0
    for key in sorted(scenarios.keys()):
        ds, sp = key
        trase_vals = scenarios[key].get('TRASE', [])
        xgb_vals = scenarios[key].get('XGBoost', [])
        ens_vals = scenarios[key].get('SimpleEns', [])

        if trase_vals and xgb_vals:
            t_mean = np.mean(trase_vals)
            x_mean = np.mean(xgb_vals)
            e_mean = np.mean(ens_vals) if ens_vals else 0
            diff_x = t_mean - x_mean
            diff_e = t_mean - e_mean

            winner = 'TRASE' if diff_x > 0.001 else ('XGB' if diff_x < -0.001 else 'TIE')
            if diff_x > 0:
                win_count += 1
            total_count += 1

            print("  %-14s %-10s: XGB=%.4f Ens=%.4f TRASE=%.4f  diff_x=%+.4f diff_e=%+.4f %s" %
                  (ds, sp, x_mean, e_mean, t_mean, diff_x, diff_e, winner))

    print(f"\n  Win rate: {win_count}/{total_count} ({win_count/total_count*100:.1f}%)")

    # F1 and AUC
    print("\n=== F1 and AUC ===")
    for m in display_order:
        f1s = [r['f1_macro'] for r in success if r['method'] == m and 'f1_macro' in r]
        aucs = [r['auc'] for r in success if r['method'] == m and 'auc' in r]
        if f1s:
            print("  %-12s: F1=%.4f+/-%.4f  AUC=%.4f+/-%.4f" %
                  (m, np.mean(f1s), np.std(f1s), np.mean(aucs), np.std(aucs)))

    # Shift Analysis
    print("\n=== Shift Analysis ===")
    trase_results = [r for r in success if r['method'] == 'TRASE']
    for split_type in ['iid', 'temporal', 'group']:
        shifted = [r.get('n_shifted', 0) for r in trase_results if r['split_type'] == split_type]
        enhanced = [r.get('use_enhanced', False) for r in trase_results if r['split_type'] == split_type]
        st_iters = [r.get('st_iterations', 0) for r in trase_results if r['split_type'] == split_type]
        st_improved = [r.get('st_improved', False) for r in trase_results if r['split_type'] == split_type]
        if shifted:
            print(f"  {split_type}: shifted={np.mean(shifted):.1f}+/-{np.std(shifted):.1f}, "
                  f"enhanced={sum(enhanced)}/{len(enhanced)}, "
                  f"st_iters={np.mean(st_iters):.1f}, "
                  f"st_improved={sum(st_improved)}/{len(st_improved)}")

    # Self-Training Analysis
    print("\n=== Self-Training Analysis ===")
    st_runs = [r for r in trase_results if r.get('st_iterations', 0) > 0]
    st_improved_runs = [r for r in st_runs if r.get('st_improved', False)]
    if st_runs:
        print(f"  Total ST runs: {len(st_runs)}")
        print(f"  ST improved: {len(st_improved_runs)}/{len(st_runs)} ({len(st_improved_runs)/len(st_runs)*100:.1f}%)")
        all_iters = [r.get('st_iterations', 0) for r in st_runs]
        print(f"  Avg ST iterations: {np.mean(all_iters):.2f}")
        all_pseudo = []
        for r in st_runs:
            all_pseudo.extend(r.get('st_pseudo_counts', []))
        if all_pseudo:
            print(f"  Avg pseudo-labels per iteration: {np.mean(all_pseudo):.1f}")

    # === Ablation Results ===
    ablation = load('trase_ablation_results.json')
    if ablation:
        print("\n" + "=" * 80)
        print("=== Ablation Results ===")
        print("=" * 80)

        ablation_success = [r for r in ablation if 'accuracy' in r and r.get('status') == 'success']

        # Group by variant
        variant_results = defaultdict(list)
        for r in ablation_success:
            variant_results[r['variant']].append(r['accuracy'])

        print("\nVariant comparison (temporal + group splits):")
        variant_order = ['SingleXGB', 'DE', 'DE+VWF', 'DE+SC+FE', 'NoST', 'Full']
        for v in variant_order:
            accs = variant_results.get(v, [])
            if accs:
                mean = np.mean(accs)
                std = np.std(accs)
                ci = stats.t.interval(0.95, len(accs)-1, loc=mean, scale=stats.sem(accs))
                print("  %-15s: %.4f +/- %.4f  [95%% CI: %.4f, %.4f]  N=%d" %
                      (v, mean, std, ci[0], ci[1], len(accs)))

        # Ablation t-tests
        print("\nAblation paired t-tests (sequential):")
        seq_pairs = [
            ('SingleXGB', 'DE', 'Ensemble'),
            ('DE', 'DE+VWF', 'VWF'),
            ('DE', 'DE+SC+FE', 'SC+FE'),
            ('DE+SC+FE', 'NoST', 'VWF on enhanced'),
            ('NoST', 'Full', 'Self-Training'),
        ]

        # Group by (dataset, split_type, seed)
        ablation_scenarios = defaultdict(lambda: defaultdict(list))
        for r in ablation_success:
            key = (r['dataset'], r['split_type'], r['seed'])
            ablation_scenarios[key][r['variant']].append(r['accuracy'])

        for v1, v2, label in seq_pairs:
            v1_all = []
            v2_all = []
            for key in sorted(ablation_scenarios.keys()):
                v1_vals = ablation_scenarios[key].get(v1, [])
                v2_vals = ablation_scenarios[key].get(v2, [])
                if v1_vals and v2_vals:
                    v1_all.extend(v1_vals)
                    v2_all.extend(v2_vals)

            if len(v1_all) == len(v2_all) and len(v1_all) > 0:
                t_stat, p_val = stats.ttest_rel(v2_all, v1_all)
                diff = np.mean(v2_all) - np.mean(v1_all)
                cohen_d = diff / (np.std(np.array(v2_all) - np.array(v1_all)) + 1e-8)
                sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
                print("  %s (%s -> %s): diff=%+.4f t=%.4f p=%.4f d=%.4f %s" %
                      (label, v1, v2, diff, t_stat, p_val, cohen_d, sig))

    # Paper Summary
    print("\n" + "=" * 80)
    print("=== Paper Summary ===")
    print("=" * 80)
    trase_accs = method_overall.get('TRASE', [])
    xgb_accs = method_overall.get('XGBoost', [])
    ens_accs = method_overall.get('SimpleEns', [])

    if trase_accs and xgb_accs:
        print(f"Total runs: {len(success)} ({len(trase_accs)} TRASE)")
        print(f"TRASE avg: {np.mean(trase_accs):.4f}")
        print(f"XGBoost avg: {np.mean(xgb_accs):.4f}")
        print(f"SimpleEns avg: {np.mean(ens_accs):.4f}")
        print(f"Improvement (TRASE - XGB): {np.mean(trase_accs) - np.mean(xgb_accs):+.4f} "
              f"({(np.mean(trase_accs) - np.mean(xgb_accs))/np.mean(xgb_accs)*100:+.2f}%)")
        print(f"Improvement (TRASE - Ens): {np.mean(trase_accs) - np.mean(ens_accs):+.4f} "
              f"({(np.mean(trase_accs) - np.mean(ens_accs))/np.mean(ens_accs)*100:+.2f}%)")
        print(f"Win rate: {win_count}/{total_count}")


if __name__ == '__main__':
    main()

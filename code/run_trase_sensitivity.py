"""
TRASE Sensitivity Analysis.

Tests sensitivity to key hyperparameters:
  1. shift_threshold: KS statistic threshold for shift detection
  2. vwf_temperature: Temperature for softmax weighting
  3. st_confidence_thresholds: Initial confidence threshold for pseudo-labeling
  4. st_max_pseudo_ratio: Maximum ratio of pseudo-labeled samples

Saves results to results/trase_sensitivity_results.json
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULT_DIR
from splits import prepare_split, get_supported_splits
from trase_core import TRASEPipeline, TRASE_CONFIG

SEEDS = [42, 456, 1024]
SENSITIVITY_DATASETS = ['telco', 'bank', 'heart', 'mushroom']
SPLIT_TYPE = 'temporal'

OUTPUT_FILE = os.path.join(RESULT_DIR, 'trase_sensitivity_results.json')
CHECKPOINT_FILE = os.path.join(RESULT_DIR, 'trase_sensitivity_checkpoint.json')


def to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


def calculate_elasticity(param_values, performance_values):
    """Calculate elasticity coefficient."""
    elasticities = []
    for i in range(1, len(param_values)):
        delta_param = param_values[i] - param_values[i-1]
        delta_perf = performance_values[i] - performance_values[i-1]

        if delta_param == 0 or performance_values[i-1] == 0:
            elasticities.append(0.0)
            continue

        elasticity = (delta_perf / performance_values[i-1]) / (delta_param / param_values[i-1])
        elasticities.append(elasticity)

    return float(np.mean(elasticities)) if elasticities else 0.0


def classify_sensitivity(elasticity):
    if abs(elasticity) > 0.5:
        return "High"
    elif abs(elasticity) > 0.2:
        return "Medium"
    else:
        return "Low"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    completed = set()
    if args.resume and os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            checkpoint = json.load(f)
            completed = set(checkpoint.get('completed', []))

    all_results = []
    if args.resume and os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            all_results = json.load(f)

    # Parameter grid
    param_grid = {
        'shift_threshold': [0.05, 0.08, 0.1, 0.15, 0.2],
        'vwf_temperature': [0.005, 0.01, 0.02, 0.05, 0.1],
        'st_confidence_thresholds': [
            [0.95, 0.90, 0.85],
            [0.90, 0.85, 0.80],
            [0.85, 0.80, 0.75],
            [0.80, 0.75, 0.70],
        ],
        'st_max_pseudo_ratio': [0.2, 0.3, 0.4, 0.5, 0.6],
    }

    total_tasks = 0
    for param_name, param_values in param_grid.items():
        for val_idx in range(len(param_values)):
            for ds_name in SENSITIVITY_DATASETS:
                if SPLIT_TYPE not in get_supported_splits(ds_name):
                    continue
                for seed in SEEDS:
                    total_tasks += 1

    print(f"Total tasks: {total_tasks}")
    print(f"Parameters: {list(param_grid.keys())}")
    print()

    start_time = time.time()
    task_count = 0

    for param_name, param_values in param_grid.items():
        for val_idx, param_value in enumerate(param_values):
            for ds_name in SENSITIVITY_DATASETS:
                if SPLIT_TYPE not in get_supported_splits(ds_name):
                    continue

                for seed in SEEDS:
                    task_key = f"{param_name}_{val_idx}_{ds_name}_{seed}"

                    if task_key in completed:
                        task_count += 1
                        continue

                    task_count += 1
                    elapsed = time.time() - start_time
                    if task_count % 10 == 0:
                        print(f"[{task_count}/{total_tasks}] {param_name}={param_value} "
                              f"{ds_name}/seed{seed} (elapsed: {elapsed:.0f}s)", flush=True)

                    try:
                        split_data = prepare_split(ds_name, SPLIT_TYPE, seed)
                        if split_data is None:
                            continue

                        X_train = split_data['X_train']
                        y_train = split_data['y_train']
                        X_val = split_data['X_val']
                        y_val = split_data['y_val']
                        X_test = split_data['X_test']
                        y_test = split_data['y_test']

                        config = TRASE_CONFIG.copy()
                        config['random_state'] = seed
                        config['verbose'] = False

                        if param_name == 'shift_threshold':
                            config['shift_threshold'] = param_value
                        elif param_name == 'vwf_temperature':
                            config['vwf_temperature'] = param_value
                        elif param_name == 'st_confidence_thresholds':
                            config['st_confidence_thresholds'] = param_value
                        elif param_name == 'st_max_pseudo_ratio':
                            config['st_max_pseudo_ratio'] = param_value

                        trase = TRASEPipeline(config)
                        trase.fit(X_train, y_train, X_val, y_val, X_test=X_test,
                                  dataset_name=f"{ds_name}/{SPLIT_TYPE}")
                        result = trase.evaluate(X_test, y_test)
                        metrics = result['metrics']
                        metrics['param_name'] = param_name
                        metrics['param_value_idx'] = val_idx
                        metrics['param_value'] = (list(param_value) if isinstance(param_value, list)
                                                   else param_value)
                        metrics['dataset'] = ds_name
                        metrics['split_type'] = SPLIT_TYPE
                        metrics['seed'] = seed
                        metrics['status'] = 'success'
                        all_results.append(to_serializable(metrics))

                    except Exception as e:
                        all_results.append({
                            'param_name': param_name,
                            'param_value_idx': val_idx,
                            'dataset': ds_name,
                            'split_type': SPLIT_TYPE,
                            'seed': seed,
                            'status': 'failed',
                            'error': str(e)[:200]
                        })

                    completed.add(task_key)
                    if task_count % 5 == 0:
                        with open(CHECKPOINT_FILE, 'w') as f:
                            json.dump({'completed': list(completed)}, f)
                        with open(OUTPUT_FILE, 'w') as f:
                            json.dump(to_serializable(all_results), f, indent=2)

    # Compute elasticity
    print("\n=== Elasticity Analysis ===")
    for param_name, param_values in param_grid.items():
        param_accs = defaultdict(list)
        for r in all_results:
            if r.get('param_name') == param_name and r.get('status') == 'success':
                param_accs[r['param_value_idx']].append(r['accuracy'])

        print(f"\n  {param_name}:")
        mean_accs = []
        for idx in range(len(param_values)):
            accs = param_accs.get(idx, [])
            if accs:
                mean_acc = np.mean(accs)
                mean_accs.append(mean_acc)
                pv = param_values[idx]
                if isinstance(pv, list):
                    pv_str = str(pv)
                else:
                    pv_str = str(pv)
                print(f"    val={pv_str}: acc={mean_acc:.4f} +/- {np.std(accs):.4f}  N={len(accs)}")

        if len(mean_accs) >= 2:
            numeric_values = []
            for v in param_values:
                if isinstance(v, list):
                    numeric_values.append(v[0])
                else:
                    numeric_values.append(v)

            elasticity = calculate_elasticity(numeric_values, mean_accs)
            sensitivity = classify_sensitivity(elasticity)
            print(f"    Elasticity: {elasticity:.4f} -> {sensitivity} sensitivity")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(to_serializable(all_results), f, indent=2)

    print(f"\nDone! {len(all_results)} results saved")


if __name__ == '__main__':
    main()

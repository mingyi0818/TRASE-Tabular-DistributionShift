"""
TRASE Robustness Analysis.

Tests robustness to:
  1. Feature noise (Gaussian noise added to features)
  2. Missing values (randomly set features to 0)

Saves results to results/trase_robustness_results.json
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULT_DIR
from splits import prepare_split, get_supported_splits
from trase_core import TRASEPipeline, TRASE_CONFIG

SEED = 42
ROBUSTNESS_DATASETS = ['telco', 'bank', 'heart', 'mushroom']
SPLIT_TYPE = 'iid'  # Use IID to isolate noise effect from shift effect

NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
MISSING_RATIOS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

OUTPUT_FILE = os.path.join(RESULT_DIR, 'trase_robustness_results.json')


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


def add_noise(X, noise_level, rng):
    """Add Gaussian noise to features."""
    if noise_level == 0:
        return X
    noise = rng.normal(0, noise_level * X.std(axis=0), X.shape)
    return X + noise


def add_missing(X, missing_ratio, rng):
    """Set random features to 0 (missing)."""
    if missing_ratio == 0:
        return X
    mask = rng.random(X.shape) < missing_ratio
    X_missing = X.copy()
    X_missing[mask] = 0.0
    return X_missing


def main():
    total_tasks = len(ROBUSTNESS_DATASETS) * (len(NOISE_LEVELS) + len(MISSING_RATIOS))
    print(f"Total tasks: {total_tasks}")
    print()

    all_results = []
    start_time = time.time()
    task_count = 0

    for ds_name in ROBUSTNESS_DATASETS:
        if SPLIT_TYPE not in get_supported_splits(ds_name):
            continue

        split_data = prepare_split(ds_name, SPLIT_TYPE, SEED)
        if split_data is None:
            continue

        X_train = split_data['X_train']
        y_train = split_data['y_train']
        X_val = split_data['X_val']
        y_val = split_data['y_val']
        X_test = split_data['X_test']
        y_test = split_data['y_test']

        # Noise robustness
        for noise_level in NOISE_LEVELS:
            task_count += 1
            elapsed = time.time() - start_time
            print(f"[{task_count}/{total_tasks}] {ds_name} noise={noise_level:.2f} "
                  f"(elapsed: {elapsed:.0f}s)", flush=True)

            rng = np.random.RandomState(SEED)
            X_test_noisy = add_noise(X_test, noise_level, rng)

            config = TRASE_CONFIG.copy()
            config['random_state'] = SEED
            config['verbose'] = False

            try:
                trase = TRASEPipeline(config)
                trase.fit(X_train, y_train, X_val, y_val, X_test=X_test_noisy,
                          dataset_name=f"{ds_name}/noise={noise_level}")
                result = trase.evaluate(X_test_noisy, y_test)
                metrics = result['metrics']
                metrics['dataset'] = ds_name
                metrics['corruption_type'] = 'noise'
                metrics['corruption_level'] = noise_level
                metrics['seed'] = SEED
                metrics['status'] = 'success'
                all_results.append(to_serializable(metrics))
            except Exception as e:
                all_results.append({
                    'dataset': ds_name, 'corruption_type': 'noise',
                    'corruption_level': noise_level, 'seed': SEED,
                    'status': 'failed', 'error': str(e)[:200]
                })

        # Missing values robustness
        for missing_ratio in MISSING_RATIOS:
            task_count += 1
            elapsed = time.time() - start_time
            print(f"[{task_count}/{total_tasks}] {ds_name} missing={missing_ratio:.2f} "
                  f"(elapsed: {elapsed:.0f}s)", flush=True)

            rng = np.random.RandomState(SEED)
            X_test_missing = add_missing(X_test, missing_ratio, rng)

            config = TRASE_CONFIG.copy()
            config['random_state'] = SEED
            config['verbose'] = False

            try:
                trase = TRASEPipeline(config)
                trase.fit(X_train, y_train, X_val, y_val, X_test=X_test_missing,
                          dataset_name=f"{ds_name}/missing={missing_ratio}")
                result = trase.evaluate(X_test_missing, y_test)
                metrics = result['metrics']
                metrics['dataset'] = ds_name
                metrics['corruption_type'] = 'missing'
                metrics['corruption_level'] = missing_ratio
                metrics['seed'] = SEED
                metrics['status'] = 'success'
                all_results.append(to_serializable(metrics))
            except Exception as e:
                all_results.append({
                    'dataset': ds_name, 'corruption_type': 'missing',
                    'corruption_level': missing_ratio, 'seed': SEED,
                    'status': 'failed', 'error': str(e)[:200]
                })

        # Save checkpoint
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(to_serializable(all_results), f, indent=2)

    # Summary
    print("\n=== Robustness Summary ===")
    for ds_name in ROBUSTNESS_DATASETS:
        print(f"\n  {ds_name}:")
        for ctype in ['noise', 'missing']:
            print(f"    {ctype}:")
            for level in (NOISE_LEVELS if ctype == 'noise' else MISSING_RATIOS):
                accs = [r['accuracy'] for r in all_results
                        if r.get('dataset') == ds_name
                        and r.get('corruption_type') == ctype
                        and r.get('corruption_level') == level
                        and r.get('status') == 'success']
                if accs:
                    print(f"      level={level:.2f}: acc={np.mean(accs):.4f}")

    print(f"\nDone! {len(all_results)} results saved")


if __name__ == '__main__':
    main()

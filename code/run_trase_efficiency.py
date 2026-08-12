"""
TRASE Efficiency Analysis.

Measures training time, inference time, and memory usage for TRASE and baselines.

Saves results to results/trase_efficiency_results.json
"""
import os
import sys
import json
import time
import tracemalloc
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULT_DIR
from splits import prepare_split, get_supported_splits
from models_new import create_method
from trase_core import TRASEPipeline, TRASE_CONFIG

SEED = 42
EFFICIENCY_DATASETS = ['telco', 'bank', 'heart', 'mushroom']
SPLIT_TYPE = 'temporal'

OUTPUT_FILE = os.path.join(RESULT_DIR, 'trase_efficiency_results.json')


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


def measure_efficiency(method_name, X_train, y_train, X_val, y_val, X_test, y_test):
    """Measure training time, inference time, and peak memory."""
    result = {'method': method_name}

    try:
        tracemalloc.start()

        if method_name == 'TRASE':
            config = TRASE_CONFIG.copy()
            config['random_state'] = SEED
            config['verbose'] = False

            start = time.time()
            trase = TRASEPipeline(config)
            trase.fit(X_train, y_train, X_val, y_val, X_test=X_test,
                      dataset_name=f"efficiency/{method_name}")
            train_time = time.time() - start

            start = time.time()
            y_pred = trase.predict(X_test)
            infer_time = time.time() - start

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            acc = float(accuracy_score(y_test, y_pred))
            result.update({
                'train_time': train_time,
                'infer_time': infer_time,
                'peak_memory_mb': peak / 1024 / 1024,
                'accuracy': acc,
                'n_models': len(trase.models),
                'status': 'success'
            })

        elif method_name == 'SimpleEns':
            probas = []
            start = time.time()
            for name in ['XGBoost', 'LightGBM', 'RF']:
                m = create_method(name, random_state=SEED)
                m.fit(X_train, y_train)
                p = m.predict_proba(X_test)
                probas.append(p)
            train_time = time.time() - start

            start = time.time()
            ens_proba = np.mean(probas, axis=0)
            y_pred = np.argmax(ens_proba, axis=1)
            infer_time = time.time() - start

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            acc = float(accuracy_score(y_test, y_pred))
            result.update({
                'train_time': train_time,
                'infer_time': infer_time,
                'peak_memory_mb': peak / 1024 / 1024,
                'accuracy': acc,
                'n_models': 3,
                'status': 'success'
            })

        else:
            if method_name == 'ExtraTrees':
                model = ExtraTreesClassifier(n_estimators=100, max_depth=10,
                                             random_state=SEED, n_jobs=-1)
            elif method_name == 'LR':
                model = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
            elif method_name == 'RF':
                model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                               random_state=SEED, n_jobs=-1)
            else:
                model = create_method(method_name, random_state=SEED)

            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            start = time.time()
            y_pred = model.predict(X_test)
            infer_time = time.time() - start

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            acc = float(accuracy_score(y_test, y_pred))
            result.update({
                'train_time': train_time,
                'infer_time': infer_time,
                'peak_memory_mb': peak / 1024 / 1024,
                'accuracy': acc,
                'n_models': 1,
                'status': 'success'
            })

    except Exception as e:
        tracemalloc.stop()
        result['status'] = 'failed'
        result['error'] = str(e)[:200]

    return result


def main():
    methods = ['LR', 'RF', 'XGBoost', 'LightGBM', 'ExtraTrees', 'SimpleEns', 'TRASE']
    total_tasks = len(EFFICIENCY_DATASETS) * len(methods)

    print(f"Total tasks: {total_tasks}")
    print(f"Methods: {methods}")
    print()

    all_results = []
    task_count = 0

    for ds_name in EFFICIENCY_DATASETS:
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

        print(f"\nDataset: {ds_name} (train={len(X_train)}, test={len(X_test)})")

        for method in methods:
            task_count += 1
            print(f"  [{task_count}/{total_tasks}] {method}...", end=' ', flush=True)

            result = measure_efficiency(method, X_train, y_train, X_val, y_val, X_test, y_test)
            result['dataset'] = ds_name
            result['split_type'] = SPLIT_TYPE
            result['seed'] = SEED
            result['n_train'] = len(X_train)
            result['n_test'] = len(X_test)
            result['n_features'] = X_train.shape[1]

            if result.get('status') == 'success':
                print(f"acc={result['accuracy']:.4f} "
                      f"train={result['train_time']:.2f}s "
                      f"infer={result['infer_time']:.4f}s "
                      f"mem={result['peak_memory_mb']:.1f}MB")
            else:
                print(f"FAILED: {result.get('error', 'unknown')}")

            all_results.append(to_serializable(result))

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(to_serializable(all_results), f, indent=2)

    # Summary
    print("\n=== Efficiency Summary (averaged across datasets) ===")
    print(f"{'Method':<15} {'Acc':<8} {'Train(s)':<10} {'Infer(s)':<10} {'Mem(MB)':<10}")
    for method in methods:
        results = [r for r in all_results if r.get('method') == method and r.get('status') == 'success']
        if results:
            accs = [r['accuracy'] for r in results]
            trains = [r['train_time'] for r in results]
            infers = [r['infer_time'] for r in results]
            mems = [r['peak_memory_mb'] for r in results]
            print(f"{method:<15} {np.mean(accs):<8.4f} {np.mean(trains):<10.2f} "
                  f"{np.mean(infers):<10.4f} {np.mean(mems):<10.1f}")

    print(f"\nDone! {len(all_results)} results saved")


if __name__ == '__main__':
    main()

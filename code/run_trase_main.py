"""
TRASE Full Main Experiments.

Runs TRASE and all baselines on all datasets x splits x seeds.
Saves results to results/trase_main_results.json

Usage:
    python run_trase_main.py [--resume]
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATASETS, RESULT_DIR
from splits import prepare_split, get_supported_splits
from models_new import create_method, evaluate_predictions
from trase_core import TRASEPipeline, TRASE_CONFIG

SEEDS = [42, 123, 456, 789, 1024]
BASELINE_METHODS = ['LR', 'RF', 'XGBoost', 'LightGBM', 'ExtraTrees']

OUTPUT_FILE = os.path.join(RESULT_DIR, 'trase_main_results.json')
CHECKPOINT_FILE = os.path.join(RESULT_DIR, 'trase_main_checkpoint.json')


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


def run_baseline(method_name, X_train, y_train, X_test, y_test, seed):
    """Run a single baseline method."""
    try:
        if method_name == 'ExtraTrees':
            model = ExtraTreesClassifier(n_estimators=100, max_depth=10,
                                         random_state=seed, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
        elif method_name == 'LR':
            model = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
        elif method_name == 'RF':
            model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                           random_state=seed, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
        else:
            model = create_method(method_name, random_state=seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

        n_classes = len(np.unique(y_test))
        metrics = evaluate_predictions(y_test, y_pred, y_proba, n_classes)
        metrics['method'] = method_name
        metrics['status'] = 'success'
        return metrics
    except Exception as e:
        return {'method': method_name, 'status': 'failed', 'error': str(e)[:200]}


def run_simple_ensemble(X_train, y_train, X_test, y_test, seed):
    """Run simple equal-weight ensemble of XGBoost + LightGBM + RF."""
    try:
        n_classes = len(np.unique(y_test))
        probas = []

        for name in ['XGBoost', 'LightGBM', 'RF']:
            try:
                m = create_method(name, random_state=seed)
                m.fit(X_train, y_train)
                p = m.predict_proba(X_test)
                if p.shape[1] < n_classes:
                    p = np.hstack([p, np.zeros((p.shape[0], n_classes - p.shape[1]))])
                probas.append(p)
            except Exception:
                pass

        if not probas:
            return {'method': 'SimpleEns', 'status': 'failed', 'error': 'No models trained'}

        ens_proba = np.mean(probas, axis=0)
        y_pred = np.argmax(ens_proba, axis=1)
        metrics = evaluate_predictions(y_test, y_pred, ens_proba, n_classes)
        metrics['method'] = 'SimpleEns'
        metrics['status'] = 'success'
        return metrics
    except Exception as e:
        return {'method': 'SimpleEns', 'status': 'failed', 'error': str(e)[:200]}


def run_trase(X_train, y_train, X_val, y_val, X_test, y_test, feature_names, seed, dataset_name):
    """Run TRASE method."""
    try:
        config = TRASE_CONFIG.copy()
        config['random_state'] = seed
        config['verbose'] = False
        trase = TRASEPipeline(config)
        trase.fit(X_train, y_train, X_val, y_val, X_test=X_test,
                  feature_names=feature_names, dataset_name=dataset_name)
        result = trase.evaluate(X_test, y_test)
        metrics = result['metrics']
        metrics['method'] = 'TRASE'
        metrics['status'] = 'success'
        return metrics
    except Exception as e:
        import traceback
        return {'method': 'TRASE', 'status': 'failed', 'error': str(e)[:200],
                'traceback': traceback.format_exc()[:500]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--quick', action='store_true', help='Quick test on 2 datasets')
    args = parser.parse_args()

    completed = set()
    if args.resume and os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            checkpoint = json.load(f)
            completed = set(checkpoint.get('completed', []))
        print(f"Resuming: {len(completed)} tasks already completed")

    all_results = []
    if args.resume and os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results")

    if args.quick:
        datasets_to_run = ['telco', 'heart']
    else:
        datasets_to_run = list(DATASETS.keys())

    total_tasks = 0
    for ds_name in datasets_to_run:
        splits = get_supported_splits(ds_name)
        for split_type in splits:
            for seed in SEEDS:
                total_tasks += 1

    print(f"Total tasks: {total_tasks}")
    print(f"Datasets: {datasets_to_run}")
    print(f"Seeds: {SEEDS}")
    print()

    start_time = time.time()
    task_count = 0

    for ds_name in datasets_to_run:
        splits = get_supported_splits(ds_name)

        for split_type in splits:
            for seed in SEEDS:
                task_key = f"{ds_name}_{split_type}_{seed}"

                if task_key in completed:
                    task_count += 1
                    continue

                task_count += 1
                elapsed = time.time() - start_time
                print(f"[{task_count}/{total_tasks}] {ds_name}/{split_type}/seed{seed} "
                      f"(elapsed: {elapsed:.0f}s)", flush=True)

                try:
                    split_data = prepare_split(ds_name, split_type, seed)
                    if split_data is None:
                        print(f"  Skip: split not supported")
                        continue

                    X_train = split_data['X_train']
                    y_train = split_data['y_train']
                    X_val = split_data['X_val']
                    y_val = split_data['y_val']
                    X_test = split_data['X_test']
                    y_test = split_data['y_test']
                    feature_names = split_data.get('feature_names', None)
                    n_features = X_train.shape[1]
                    n_classes = len(np.unique(y_train))

                    # Run all baselines
                    for method_name in BASELINE_METHODS:
                        m = run_baseline(method_name, X_train, y_train, X_test, y_test, seed)
                        m['dataset'] = ds_name
                        m['split_type'] = split_type
                        m['seed'] = seed
                        m['n_train'] = len(X_train)
                        m['n_val'] = len(X_val)
                        m['n_test'] = len(X_test)
                        m['n_features'] = n_features
                        m['n_classes'] = n_classes
                        all_results.append(m)

                    # Run Simple Ensemble
                    m = run_simple_ensemble(X_train, y_train, X_test, y_test, seed)
                    m['dataset'] = ds_name
                    m['split_type'] = split_type
                    m['seed'] = seed
                    m['n_train'] = len(X_train)
                    m['n_val'] = len(X_val)
                    m['n_test'] = len(X_test)
                    m['n_features'] = n_features
                    m['n_classes'] = n_classes
                    all_results.append(m)

                    # Run TRASE
                    m = run_trase(X_train, y_train, X_val, y_val, X_test, y_test,
                                  feature_names, seed, f"{ds_name}/{split_type}")
                    m['dataset'] = ds_name
                    m['split_type'] = split_type
                    m['seed'] = seed
                    m['n_train'] = len(X_train)
                    m['n_val'] = len(X_val)
                    m['n_test'] = len(X_test)
                    m['n_features'] = n_features
                    m['n_classes'] = n_classes
                    all_results.append(m)

                    if m.get('status') == 'success':
                        print(f"  TRASE: acc={m['accuracy']:.4f} "
                              f"(models={m.get('n_models',0)}, "
                              f"enh={m.get('use_enhanced',False)}, "
                              f"st_iters={m.get('st_iterations',0)})", flush=True)

                except Exception as e:
                    print(f"  ERROR: {e}")
                    all_results.append({
                        'dataset': ds_name, 'split_type': split_type, 'seed': seed,
                        'status': 'failed', 'error': str(e)[:200]
                    })

                # Checkpoint
                completed.add(task_key)
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump({'completed': list(completed)}, f)
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(to_serializable(all_results), f, indent=2)

    # Final save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(to_serializable(all_results), f, indent=2)

    print(f"\nDone! {len(all_results)} results saved to {OUTPUT_FILE}")
    print(f"Total time: {time.time() - start_time:.0f}s")


if __name__ == '__main__':
    main()

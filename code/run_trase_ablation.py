"""
TRASE Ablation Experiments.

Tests the contribution of each TRASE component:
  1. TRASE-SingleXGB: Single XGBoost (baseline)
  2. TRASE-DE: Diverse Ensemble only (8 models, equal weight, no shift, no ST)
  3. TRASE-DE+VWF: DE + VWF (no shift features, no ST)
  4. TRASE-DE+SC+FE: DE + Shift Characterization + Feature Enhancement (no VWF, no ST)
  5. TRASE-NoST: Full Phase 1 (SC + FE + VWF) but no self-training
  6. TRASE-Full: Everything (SC + FE + VWF + ST)

Saves results to results/trase_ablation_results.json
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATASETS, RESULT_DIR
from splits import prepare_split, get_supported_splits
from trase_core import TRASEPipeline, TRASE_CONFIG, create_model, _pad_proba

SEEDS = [42, 123, 456, 789, 1024]
OUTPUT_FILE = os.path.join(RESULT_DIR, 'trase_ablation_results.json')
CHECKPOINT_FILE = os.path.join(RESULT_DIR, 'trase_ablation_checkpoint.json')

# Datasets and splits for ablation (focus on temporal where shift is most significant)
ABLATION_DATASETS = ['telco', 'bank', 'heart', 'mushroom', 'adult']
ABLATION_SPLITS = ['temporal', 'group']


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


def run_variant(variant_name, X_train, y_train, X_val, y_val, X_test, y_test, seed, dataset_name):
    """Run a specific ablation variant."""
    try:
        cfg = TRASE_CONFIG.copy()
        cfg['random_state'] = seed
        cfg['verbose'] = False

        if variant_name == 'SingleXGB':
            # Just one XGBoost on original features
            model = create_model('XGBoost', cfg, random_state=seed, config_idx=0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            n_classes = len(np.unique(y_train))
            y_proba = _pad_proba(y_proba, n_classes)
            acc = float(accuracy_score(y_test, y_pred))
            return {
                'variant': variant_name,
                'accuracy': acc,
                'status': 'success',
                'n_models': 1,
                'n_shifted': 0,
                'use_enhanced': False,
                'use_vwf': False,
                'st_iterations': 0,
                'st_improved': False,
            }

        # All other variants use TRASEPipeline with different configs
        if variant_name == 'DE':
            # Diverse Ensemble only (no shift features, no VWF, no ST)
            cfg['use_shift_features'] = False
            cfg['use_original_view'] = True
            cfg['use_enhanced_view'] = False
            cfg['use_nonshifted_view'] = False
            cfg['use_vwf'] = False
            cfg['use_self_training'] = False
            cfg['enhancement_safety_check'] = False

        elif variant_name == 'DE+VWF':
            # DE + VWF (no shift features, no ST)
            cfg['use_shift_features'] = False
            cfg['use_original_view'] = True
            cfg['use_enhanced_view'] = False
            cfg['use_nonshifted_view'] = False
            cfg['use_vwf'] = True
            cfg['use_self_training'] = False
            cfg['enhancement_safety_check'] = False

        elif variant_name == 'DE+SC+FE':
            # DE + SC + FE (equal weight, no VWF, no ST)
            cfg['use_shift_features'] = True
            cfg['use_original_view'] = True
            cfg['use_enhanced_view'] = True
            cfg['use_nonshifted_view'] = True
            cfg['use_vwf'] = False
            cfg['use_self_training'] = False
            cfg['enhancement_safety_check'] = True

        elif variant_name == 'NoST':
            # Full Phase 1 but no self-training
            cfg['use_shift_features'] = True
            cfg['use_original_view'] = True
            cfg['use_enhanced_view'] = True
            cfg['use_nonshifted_view'] = True
            cfg['use_vwf'] = True
            cfg['use_self_training'] = False
            cfg['enhancement_safety_check'] = True

        elif variant_name == 'Full':
            # Everything
            cfg['use_shift_features'] = True
            cfg['use_original_view'] = True
            cfg['use_enhanced_view'] = True
            cfg['use_nonshifted_view'] = True
            cfg['use_vwf'] = True
            cfg['use_self_training'] = True
            cfg['enhancement_safety_check'] = True

        else:
            raise ValueError(f"Unknown variant: {variant_name}")

        trase = TRASEPipeline(cfg)
        trase.fit(X_train, y_train, X_val, y_val, X_test=X_test,
                  dataset_name=dataset_name)
        result = trase.evaluate(X_test, y_test)
        metrics = result['metrics']
        metrics['variant'] = variant_name
        metrics['status'] = 'success'
        return metrics

    except Exception as e:
        import traceback
        return {
            'variant': variant_name,
            'status': 'failed',
            'error': str(e)[:200],
            'traceback': traceback.format_exc()[:500]
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
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

    variants = ['SingleXGB', 'DE', 'DE+VWF', 'DE+SC+FE', 'NoST', 'Full']

    total_tasks = 0
    for ds_name in ABLATION_DATASETS:
        for split_type in ABLATION_SPLITS:
            if split_type not in get_supported_splits(ds_name):
                continue
            for seed in SEEDS:
                total_tasks += 1

    print(f"Total tasks: {total_tasks} (datasets={len(ABLATION_DATASETS)}, "
          f"splits={len(ABLATION_SPLITS)}, seeds={len(SEEDS)})")
    print(f"Variants: {variants}")
    print()

    start_time = time.time()
    task_count = 0

    for ds_name in ABLATION_DATASETS:
        for split_type in ABLATION_SPLITS:
            if split_type not in get_supported_splits(ds_name):
                continue

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
                        continue

                    X_train = split_data['X_train']
                    y_train = split_data['y_train']
                    X_val = split_data['X_val']
                    y_val = split_data['y_val']
                    X_test = split_data['X_test']
                    y_test = split_data['y_test']

                    for variant in variants:
                        m = run_variant(variant, X_train, y_train, X_val, y_val,
                                        X_test, y_test, seed, f"{ds_name}/{split_type}")
                        m['dataset'] = ds_name
                        m['split_type'] = split_type
                        m['seed'] = seed
                        all_results.append(m)

                        if m.get('status') == 'success':
                            print(f"  {variant:15s}: acc={m['accuracy']:.4f}", flush=True)

                except Exception as e:
                    print(f"  ERROR: {e}")

                completed.add(task_key)
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump({'completed': list(completed)}, f)
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(to_serializable(all_results), f, indent=2)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(to_serializable(all_results), f, indent=2)

    print(f"\nDone! {len(all_results)} results saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()

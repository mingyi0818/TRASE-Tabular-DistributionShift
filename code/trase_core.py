"""
TRASE: Test-time Refinement with Adaptive Self-training Ensemble
for Robust Tabular Learning under Distribution Shift.

Core Innovation:
  Previous methods (DARE, IWSE, DEQA, SAFE) only used target domain FEATURE
  information. TRASE goes further by using target domain LABEL information
  through high-confidence pseudo-labeling, enabling direct model adaptation
  to the target distribution.

Two-Phase Architecture:
  Phase 1 - Diverse Ensemble with Shift-Aware Features:
    - Train 8+ diverse models (4 algorithms x 2 hyperparameter sets)
    - Add shift-aware meta-features when shift is detected
    - Validation-Weighted Fusion for model weighting
    - Shift gate: disable enhancement when no shift

  Phase 2 - Iterative Self-Training Refinement:
    - Use ensemble predictions on test data as pseudo-labels
    - Select only high-confidence predictions (curriculum: 0.9 -> 0.8 -> 0.7)
    - Retrain models on source + pseudo-labeled target data
    - Target-weighted validation safety check at each iteration
    - Fall back to Phase 1 if refinement hurts validation

Key Safety Mechanisms:
  1. Curriculum confidence thresholds (start conservative, gradually relax)
  2. Target-similarity weighted validation check (focus on target-like samples)
  3. Maximum 3 iterations (prevent overfitting to pseudo-labels)
  4. Revert to best model if any iteration degrades performance
  5. Enhancement safety gate (only use shift features if they help validation)

Theoretical Basis:
  Under distribution shift, P_s(Y|X) != P_t(Y|X). Standard models learn
  P_s(Y|X) which is suboptimal for target. By using high-confidence
  pseudo-labels from the ensemble, we can estimate P_t(Y|X) and adapt
  the decision boundary toward the target distribution.

  The ensemble provides more accurate pseudo-labels than any single model
  (by N -> 0 as N grows under weak learners assumption). The curriculum
  approach ensures that only the most reliable pseudo-labels are used
  initially, gradually including more as the model improves.

  The target-weighted validation check ensures that we only accept
  refinements that improve performance on target-like samples, preventing
  degradation on datasets where pseudo-labeling might hurt.
"""

import os
import sys
import time
import numpy as np
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# ============================================================
# Configuration
# ============================================================

TRASE_CONFIG = {
    # Shift Characterization
    'shift_threshold': 0.1,
    'shift_pvalue': 0.05,

    # Feature Enhancement
    'use_shift_features': True,

    # Diverse Ensemble
    'use_original_view': True,
    'use_enhanced_view': True,
    'use_nonshifted_view': True,

    # Model hyperparameter sets (2 per algorithm for diversity)
    'model_configs': {
        'XGBoost': [
            {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
             'use_label_encoder': False, 'eval_metric': 'logloss', 'n_jobs': -1},
            {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05,
             'use_label_encoder': False, 'eval_metric': 'logloss', 'n_jobs': -1},
        ],
        'LightGBM': [
            {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
             'verbose': -1, 'n_jobs': -1},
            {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05,
             'verbose': -1, 'n_jobs': -1},
        ],
        'RF': [
            {'n_estimators': 100, 'max_depth': 10, 'n_jobs': -1},
            {'n_estimators': 200, 'max_depth': 8, 'n_jobs': -1},
        ],
        'ExtraTrees': [
            {'n_estimators': 100, 'max_depth': 10, 'n_jobs': -1},
            {'n_estimators': 200, 'max_depth': 8, 'n_jobs': -1},
        ],
    },

    # Validation-Weighted Fusion
    'use_vwf': True,
    'vwf_temperature': 0.1,
    'vwf_min_weight': 0.05,
    'vwf_safety_check': True,

    # Self-Training Refinement
    'use_self_training': True,
    'st_confidence_thresholds': [0.9, 0.85, 0.8],  # Curriculum
    'st_min_pseudo_ratio': 0.05,   # Min fraction of test data to use as pseudo
    'st_max_pseudo_ratio': 0.5,    # Max fraction of test data to use as pseudo
    'st_use_sample_weight': True,  # Weight pseudo-labels by confidence
    'st_safety_check': True,       # Revert if validation degrades

    # Target-weighted validation
    'use_target_weighting': True,
    'tw_kernel': 'gaussian',       # 'gaussian' or 'uniform'
    'tw_bandwidth': 1.0,           # Bandwidth for Gaussian kernel
    'tw_clip': [0.1, 10.0],        # Clip weights to prevent extreme values

    # General
    'random_state': 42,
    'verbose': True,
}


def log(msg, verbose=True):
    if verbose:
        print(msg, flush=True)


def _pad_proba(proba, n_classes):
    """Pad probability array to have n_classes columns."""
    if proba.shape[1] == n_classes:
        return proba
    elif proba.shape[1] < n_classes:
        padding = np.zeros((proba.shape[0], n_classes - proba.shape[1]))
        return np.hstack([proba, padding])
    else:
        return proba[:, :n_classes]


# ============================================================
# Shift Characterization
# ============================================================

class ShiftCharacterizer:
    """
    Characterize distribution shift between source and target domains.

    1. Per-feature KS test to identify shifted features
    2. Per-sample shift scores based on z-score statistics
    3. Target-similarity weights for validation samples
    """

    def __init__(self, threshold=0.1, pvalue=0.05,
                 tw_kernel='gaussian', tw_bandwidth=1.0, tw_clip=(0.1, 10.0)):
        self.threshold = threshold
        self.pvalue = pvalue
        self.shifted_features = []
        self.nonshifted_features = []
        self.ks_stats = {}
        self.ks_pvalues = {}
        self.source_mean = None
        self.source_std = None
        self.target_mean = None
        self.target_std = None
        self.tw_kernel = tw_kernel
        self.tw_bandwidth = tw_bandwidth
        self.tw_clip = tw_clip

    def fit(self, X_source, X_target):
        """Detect shifted features and compute source/target statistics."""
        n_features = X_source.shape[1]

        self.source_mean = np.mean(X_source, axis=0)
        self.source_std = np.std(X_source, axis=0) + 1e-8
        self.target_mean = np.mean(X_target, axis=0)
        self.target_std = np.std(X_target, axis=0) + 1e-8

        self.shifted_features = []
        self.nonshifted_features = []
        self.ks_stats = {}
        self.ks_pvalues = {}

        for j in range(n_features):
            ks_stat, p_val = ks_2samp(X_source[:, j], X_target[:, j])
            self.ks_stats[j] = float(ks_stat)
            self.ks_pvalues[j] = float(p_val)

            if ks_stat > self.threshold and p_val < self.pvalue:
                self.shifted_features.append(j)
            else:
                self.nonshifted_features.append(j)

        return self

    def compute_shift_features(self, X):
        """
        Compute 3 shift-aware meta-features for each sample.

        1. global_shift: Mean absolute z-score relative to source
        2. shifted_feature_score: Mean |z-score| for shifted features
        3. nonshifted_feature_score: Mean |z-score| for non-shifted features
        """
        z_scores = np.abs((X - self.source_mean) / self.source_std)

        global_shift = np.mean(z_scores, axis=1, keepdims=True)

        if len(self.shifted_features) > 0:
            shifted_score = np.mean(z_scores[:, self.shifted_features],
                                    axis=1, keepdims=True)
        else:
            shifted_score = np.zeros((X.shape[0], 1))

        if len(self.nonshifted_features) > 0:
            nonshifted_score = np.mean(z_scores[:, self.nonshifted_features],
                                       axis=1, keepdims=True)
        else:
            nonshifted_score = np.zeros((X.shape[0], 1))

        return np.hstack([global_shift, shifted_score, nonshifted_score])

    def compute_target_similarity_weights(self, X_val):
        """
        Compute target-similarity weights for validation samples.

        Samples that are more similar to the target distribution get higher
        weights, so the safety check focuses on the samples that matter.

        Uses Gaussian kernel on standardized distance to target mean.
        """
        if self.target_mean is None:
            return np.ones(len(X_val))

        # Standardized distance to target mean
        z = (X_val - self.target_mean) / (self.target_std + 1e-8)
        dist_sq = np.sum(z ** 2, axis=1)

        if self.tw_kernel == 'gaussian':
            n_features = X_val.shape[1]
            # Scale bandwidth by dimensionality
            bw = self.tw_bandwidth * np.sqrt(n_features)
            weights = np.exp(-dist_sq / (2 * bw ** 2))
        else:
            weights = np.ones(len(X_val))

        # Normalize to mean=1
        weights = weights / (np.mean(weights) + 1e-8)

        # Clip to prevent extreme values
        weights = np.clip(weights, self.tw_clip[0], self.tw_clip[1])

        return weights

    def get_shift_report(self):
        return {
            'n_features_checked': len(self.ks_stats),
            'n_shifted': len(self.shifted_features),
            'n_nonshifted': len(self.nonshifted_features),
            'shifted_features': list(self.shifted_features),
            'ks_stats': self.ks_stats,
            'mean_ks': float(np.mean(list(self.ks_stats.values()))) if self.ks_stats else 0.0,
            'max_ks': float(np.max(list(self.ks_stats.values()))) if self.ks_stats else 0.0,
        }


# ============================================================
# Model Wrappers
# ============================================================

class XGBoostWrapper:
    def __init__(self, params, random_state=42):
        self.params = params.copy()
        self.params['random_state'] = random_state
        self.model = None

    def fit(self, X, y, sample_weight=None):
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class LightGBMWrapper:
    def __init__(self, params, random_state=42):
        self.params = params.copy()
        self.params['random_state'] = random_state
        self.model = None

    def fit(self, X, y, sample_weight=None):
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class RFWrapper:
    def __init__(self, params, random_state=42):
        self.params = params.copy()
        self.params['random_state'] = random_state
        self.model = None

    def fit(self, X, y, sample_weight=None):
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class ExtraTreesWrapper:
    def __init__(self, params, random_state=42):
        self.params = params.copy()
        self.params['random_state'] = random_state
        self.model = None

    def fit(self, X, y, sample_weight=None):
        self.model = ExtraTreesClassifier(**self.params)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def create_model(name, config, random_state=42, config_idx=0):
    """Create a base model by name and config index."""
    model_configs = config.get('model_configs', {})
    if name not in model_configs:
        raise ValueError(f"Unknown model: {name}")

    params = model_configs[name][config_idx]

    if name == 'XGBoost':
        return XGBoostWrapper(params, random_state)
    elif name == 'LightGBM':
        return LightGBMWrapper(params, random_state)
    elif name == 'RF':
        return RFWrapper(params, random_state)
    elif name == 'ExtraTrees':
        return ExtraTreesWrapper(params, random_state)
    else:
        raise ValueError(f"Unknown model: {name}")


# ============================================================
# TRASE Pipeline
# ============================================================

class TRASEPipeline:
    """
    Test-time Refinement with Adaptive Self-training Ensemble.

    Phase 1: Diverse Ensemble with Shift-Aware Features
    Phase 2: Iterative Self-Training Refinement
    """

    def __init__(self, config=None):
        self.config = config or TRASE_CONFIG.copy()
        self.characterizer = None
        self.models = {}            # name -> (model, feature_view, config_idx)
        self.model_weights = None
        self.n_classes = 0
        self.n_features = 0
        self.train_time = 0
        self.predict_time = 0
        self.shift_report = None
        self.use_vwf = True
        self._use_enhanced = False
        self.val_accuracies = {}

        # Self-training state
        self.st_iterations = 0
        self.st_pseudo_counts = []
        self.st_val_accs = []
        self.st_improved = False

        # Feature views
        self._X_train_orig = None
        self._X_train_enh = None
        self._X_train_nonsh = None
        self._X_test_enh = None
        self._X_test_nonsh = None
        self._model_names = []
        self._shifted = []
        self._nonshifted = []

        # Target weights for validation
        self._val_weights = None

    def _build_model_specs(self, use_enhanced):
        """Build list of (model_name, view, config_idx) to train."""
        specs = []

        # View A: Original features
        if self.config.get('use_original_view', True):
            for model_name in ['XGBoost', 'LightGBM', 'RF', 'ExtraTrees']:
                for idx in range(2):
                    specs.append((model_name, 'original', idx))

        # View B: Enhanced features (only when shift detected)
        if self.config.get('use_enhanced_view', True) and use_enhanced:
            for model_name in ['XGBoost', 'LightGBM', 'RF', 'ExtraTrees']:
                specs.append((model_name, 'enhanced', 0))

        # View C: Non-shifted features (only when shift detected)
        if self.config.get('use_nonshifted_view', True) and use_enhanced:
            specs.append(('XGBoost', 'nonshifted', 0))

        return specs

    def _train_models(self, specs, X_train, y_train, X_train_enh,
                      X_train_nonsh, sample_weight=None):
        """Train all models according to specs."""
        cfg = self.config
        rs = cfg['random_state']
        models = {}
        model_names = []

        for model_name, view, idx in specs:
            key = f"{model_name}_{view}_{idx}"
            model = create_model(model_name, cfg, random_state=rs, config_idx=idx)

            if view == 'original':
                if sample_weight is not None:
                    model.fit(X_train, y_train, sample_weight=sample_weight)
                else:
                    model.fit(X_train, y_train)
            elif view == 'enhanced':
                if sample_weight is not None:
                    model.fit(X_train_enh, y_train, sample_weight=sample_weight)
                else:
                    model.fit(X_train_enh, y_train)
            elif view == 'nonshifted':
                if sample_weight is not None:
                    model.fit(X_train_nonsh, y_train, sample_weight=sample_weight)
                else:
                    model.fit(X_train_nonsh, y_train)

            models[key] = (model, view)
            model_names.append(key)

        return models, model_names

    def _compute_val_accuracies(self, models, X_val, X_val_enh, X_val_nonsh, y_val,
                                use_target_weighted=False):
        """Compute validation accuracy for each model.

        When use_target_weighted=True, uses target-similarity weighted accuracy
        to better reflect target domain performance under distribution shift.
        """
        val_accs = {}
        weights = self._val_weights if use_target_weighted else None

        for key, (model, view) in models.items():
            if view == 'original':
                X_v = X_val
            elif view == 'enhanced':
                X_v = X_val_enh
            elif view == 'nonshifted':
                X_v = X_val_nonsh
            else:
                X_v = X_val

            pred = model.predict(X_v)
            if weights is not None:
                correct = (y_val == pred).astype(float)
                acc = float(np.average(correct, weights=weights))
            else:
                acc = float(accuracy_score(y_val, pred))
            val_accs[key] = acc

        return val_accs

    def _weighted_predict_proba(self, models, model_names, weights,
                                X, X_enh, X_nonsh):
        """Get weighted ensemble probability predictions."""
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_classes))

        for i, key in enumerate(model_names):
            model, view = models[key]
            if view == 'original':
                X_p = X
            elif view == 'enhanced':
                X_p = X_enh
            elif view == 'nonshifted':
                X_p = X_nonsh
            else:
                X_p = X

            p = model.predict_proba(X_p)
            p = _pad_proba(p, self.n_classes)
            proba += weights[i] * p

        return proba

    def _compute_vwf_weights(self, val_accs, model_names):
        """Compute Validation-Weighted Fusion weights.

        Uses rank-based softmax weighting for robustness.
        Under high distribution shift, temperature INCREASES (more conservative,
        closer to equal weights) because validation accuracy is less reliable.
        """
        cfg = self.config
        accs = np.array([val_accs[k] for k in model_names])

        # Shift-adaptive temperature: INCREASE under high shift (more conservative)
        shift_severity = self.shift_report['n_shifted'] / max(self.n_features, 1)
        base_temp = cfg.get('vwf_temperature', 0.1)
        adaptive_temp = base_temp * (1.0 + 2.0 * shift_severity)

        # Rank-based weighting for robustness (reduces sensitivity to small acc differences)
        ranks = np.argsort(np.argsort(accs))  # 0=worst, N-1=best
        rank_scores = ranks.astype(float) + 1.0  # 1 to N

        # Softmax on rank scores (much more stable than on raw accuracies)
        weights = np.exp(rank_scores / adaptive_temp)
        weights = weights / weights.sum()

        # Apply minimum weight constraint
        min_w = cfg.get('vwf_min_weight', 0.05)
        weights = np.maximum(weights, min_w)
        weights = weights / weights.sum()

        return weights

    def _target_weighted_accuracy(self, y_true, y_pred, weights):
        """Compute target-weighted accuracy."""
        if weights is None:
            return float(accuracy_score(y_true, y_pred))
        correct = (y_true == y_pred).astype(float)
        return float(np.average(correct, weights=weights))

    def fit(self, X_train, y_train, X_val, y_val, X_test=None,
            feature_names=None, dataset_name=''):
        """
        Full training pipeline with optional self-training refinement.

        Args:
            X_train: Training features (source domain, already scaled)
            y_train: Training labels
            X_val: Validation features (source domain, already scaled)
            y_val: Validation labels
            X_test: Test features (target domain, for shift characterization)
            feature_names: List of feature names
            dataset_name: Dataset name for logging
        """
        start = time.time()
        cfg = self.config
        verbose = cfg.get('verbose', True)
        rs = cfg['random_state']

        self.n_classes = len(np.unique(y_train))
        self.n_features = X_train.shape[1]
        self.use_vwf = cfg.get('use_vwf', True)

        log(f"\n  TRASE fitting on {dataset_name} (n_train={len(X_train)}, "
            f"n_val={len(X_val)}, n_test={len(X_test) if X_test is not None else 'N/A'}, "
            f"n_features={self.n_features}, n_classes={self.n_classes})", verbose)

        # ========== Phase 1: Diverse Ensemble with Shift-Aware Features ==========

        # Phase 1.1: Shift Characterization
        use_shift = cfg.get('use_shift_features', True) and X_test is not None
        if use_shift:
            self.characterizer = ShiftCharacterizer(
                threshold=cfg['shift_threshold'],
                pvalue=cfg['shift_pvalue'],
                tw_kernel=cfg.get('tw_kernel', 'gaussian'),
                tw_bandwidth=cfg.get('tw_bandwidth', 1.0),
                tw_clip=cfg.get('tw_clip', (0.1, 10.0)),
            )
            self.characterizer.fit(X_train, X_test)
            self.shift_report = self.characterizer.get_shift_report()
            log(f"  Phase 1 (SC): {self.shift_report['n_shifted']}/{self.n_features} "
                f"shifted features (mean KS={self.shift_report['mean_ks']:.4f}, "
                f"max KS={self.shift_report['max_ks']:.4f})", verbose)
        else:
            self.characterizer = None
            self.shift_report = {
                'n_shifted': 0, 'n_nonshifted': self.n_features,
                'mean_ks': 0.0, 'max_ks': 0.0,
            }
            log(f"  Phase 1 (SC): Disabled", verbose)

        # Phase 1.2: Feature Enhancement (with shift gate + safety gate)
        use_enhanced = use_shift and self.shift_report['n_shifted'] > 0

        if use_enhanced:
            shift_train = self.characterizer.compute_shift_features(X_train)
            shift_val = self.characterizer.compute_shift_features(X_val)
            shift_test = self.characterizer.compute_shift_features(X_test) if X_test is not None else None

            X_train_enh = np.hstack([X_train, shift_train])
            X_val_enh = np.hstack([X_val, shift_val])
            X_test_enh = np.hstack([X_test, shift_test]) if X_test is not None else None

            # Enhancement safety gate: check if enhancement helps on REGULAR validation
            if cfg.get('enhancement_safety_check', True):
                # Train a quick XGBoost with and without enhancement
                test_model_orig = create_model('XGBoost', cfg, random_state=rs, config_idx=0)
                test_model_orig.fit(X_train, y_train)
                orig_val_acc = float(accuracy_score(y_val, test_model_orig.predict(X_val)))

                test_model_enh = create_model('XGBoost', cfg, random_state=rs, config_idx=0)
                test_model_enh.fit(X_train_enh, y_train)
                enh_val_acc = float(accuracy_score(y_val, test_model_enh.predict(X_val_enh)))

                if enh_val_acc < orig_val_acc:
                    use_enhanced = False
                    X_train_enh = X_train
                    X_val_enh = X_val
                    X_test_enh = X_test
                    log(f"  Phase 1 (FE): Safety gate REJECTED enhancement "
                        f"(orig={orig_val_acc:.4f} > enh={enh_val_acc:.4f})", verbose)
                else:
                    log(f"  Phase 1 (FE): Safety gate APPROVED enhancement "
                        f"(enh={enh_val_acc:.4f} >= orig={orig_val_acc:.4f}), "
                        f"features {X_train.shape[1]} -> {X_train_enh.shape[1]}", verbose)
            else:
                log(f"  Phase 1 (FE): Enhanced features ({X_train.shape[1]} -> {X_train_enh.shape[1]})", verbose)
        else:
            X_train_enh = X_train
            X_val_enh = X_val
            X_test_enh = X_test
            if use_shift:
                log(f"  Phase 1 (FE): Shift gate closed (n_shifted=0), skipping enhancement", verbose)
            else:
                log(f"  Phase 1 (FE): Disabled", verbose)

        self._use_enhanced = use_enhanced

        # Non-shifted features
        shifted = self.characterizer.shifted_features if (self.characterizer and use_enhanced) else []
        nonshifted = self.characterizer.nonshifted_features if (self.characterizer and use_enhanced) else list(range(self.n_features))

        if use_enhanced and len(nonshifted) > 0 and len(nonshifted) < self.n_features:
            X_train_nonsh = X_train[:, nonshifted]
            X_val_nonsh = X_val[:, nonshifted]
            X_test_nonsh = X_test[:, nonshifted] if X_test is not None else None
        else:
            X_train_nonsh = None
            X_val_nonsh = None
            X_test_nonsh = None

        # Store for prediction
        self._X_train_orig = X_train
        self._X_train_enh = X_train_enh
        self._X_train_nonsh = X_train_nonsh
        self._X_test_enh = X_test_enh
        self._X_test_nonsh = X_test_nonsh
        self._shifted = shifted
        self._nonshifted = nonshifted

        # Phase 1.3: Train Diverse Ensemble
        specs = self._build_model_specs(use_enhanced)
        log(f"  Phase 1 (DE): Training {len(specs)} diverse models", verbose)

        self.models, self._model_names = self._train_models(
            specs, X_train, y_train, X_train_enh, X_train_nonsh
        )

        log(f"  Phase 1 (DE): Trained {len(self.models)} models", verbose)

        # Phase 1.4: Validation-Weighted Fusion
        # Compute target-similarity weights for validation FIRST
        if cfg.get('use_target_weighting', True) and self.characterizer is not None:
            self._val_weights = self.characterizer.compute_target_similarity_weights(X_val)
        else:
            self._val_weights = None

        # Use target-weighted validation accuracy when available
        self.val_accuracies = self._compute_val_accuracies(
            self.models, X_val, X_val_enh, X_val_nonsh, y_val,
            use_target_weighted=self._val_weights is not None
        )

        if self.use_vwf and len(self.models) >= 2:
            weights = self._compute_vwf_weights(self.val_accuracies, self._model_names)

            # Safety check: VWF vs equal weight on target-weighted validation
            if cfg.get('vwf_safety_check', True):
                vwf_pred = np.argmax(self._weighted_predict_proba(
                    self.models, self._model_names, weights,
                    X_val, X_val_enh, X_val_nonsh), axis=1)
                vwf_acc = self._target_weighted_accuracy(y_val, vwf_pred, self._val_weights)

                eq_weights = np.ones(len(self.models)) / len(self.models)
                eq_pred = np.argmax(self._weighted_predict_proba(
                    self.models, self._model_names, eq_weights,
                    X_val, X_val_enh, X_val_nonsh), axis=1)
                eq_acc = self._target_weighted_accuracy(y_val, eq_pred, self._val_weights)

                if vwf_acc < eq_acc:
                    log(f"  Phase 1 (VWF): VWF ({vwf_acc:.4f}) < Equal ({eq_acc:.4f}), "
                        f"using equal weights", verbose)
                    self.model_weights = eq_weights
                else:
                    log(f"  Phase 1 (VWF): VWF ({vwf_acc:.4f}) >= Equal ({eq_acc:.4f}), "
                        f"using VWF weights", verbose)
                    self.model_weights = weights
            else:
                self.model_weights = weights
                log(f"  Phase 1 (VWF): Using VWF weights", verbose)
        else:
            n_models = len(self.models)
            self.model_weights = np.ones(n_models) / max(n_models, 1)
            log(f"  Phase 1 (VWF): Disabled, using equal weights", verbose)

        # Phase 1 validation accuracy (target-weighted for safety checks)
        phase1_pred = np.argmax(self._weighted_predict_proba(
            self.models, self._model_names, self.model_weights,
            X_val, X_val_enh, X_val_nonsh), axis=1)
        phase1_acc = self._target_weighted_accuracy(y_val, phase1_pred, self._val_weights)
        log(f"  Phase 1: Val acc={phase1_acc:.4f}", verbose)

        # Compute best single model accuracy on validation (for global safety)
        best_single_val_acc = 0.0
        best_single_key = None
        for key, (model, view) in self.models.items():
            if view == 'original':
                if key in self.val_accuracies and self.val_accuracies[key] > best_single_val_acc:
                    best_single_val_acc = self.val_accuracies[key]
                    best_single_key = key
        log(f"  Phase 1: Best single model val acc={best_single_val_acc:.4f} ({best_single_key})", verbose)

        # ========== Phase 2: Iterative Self-Training Refinement ==========
        # Run self-training when shift is detected (even if enhancement was rejected)
        st_condition = (cfg.get('use_self_training', True) and X_test is not None
                        and use_shift and self.shift_report['n_shifted'] > 0)
        if st_condition:
            self._run_self_training(
                X_train, y_train, X_val, y_val, X_test,
                X_train_enh, X_val_enh, X_test_enh,
                X_train_nonsh, X_val_nonsh, X_test_nonsh,
                phase1_acc, best_single_val_acc, verbose
            )
        else:
            if not use_shift or self.shift_report['n_shifted'] == 0:
                log(f"  Phase 2 (ST): Skipped (no shift detected)", verbose)
            elif X_test is None:
                log(f"  Phase 2 (ST): Skipped (no test data)", verbose)
            else:
                log(f"  Phase 2 (ST): Disabled", verbose)

        self.train_time = time.time() - start
        log(f"  Training time: {self.train_time:.2f}s", verbose)

        return self

    def _run_self_training(self, X_train, y_train, X_val, y_val, X_test,
                           X_train_enh, X_val_enh, X_test_enh,
                           X_train_nonsh, X_val_nonsh, X_test_nonsh,
                           best_val_acc, best_single_val_acc, verbose):
        """Run iterative self-training refinement using target-weighted validation accuracy."""
        cfg = self.config
        thresholds = cfg.get('st_confidence_thresholds', [0.9, 0.85, 0.8])

        # Store best models (Phase 1 baseline)
        best_models = {k: v for k, v in self.models.items()}
        best_weights = self.model_weights.copy()
        best_val_accuracies = self.val_accuracies.copy()
        best_model_names = self._model_names.copy()

        log(f"  Phase 2 (ST): Starting self-training (best val acc={best_val_acc:.4f}, "
            f"best single={best_single_val_acc:.4f})", verbose)

        for iteration, threshold in enumerate(thresholds):
            log(f"  Phase 2 (ST) iter {iteration+1}/{len(thresholds)}: "
                f"threshold={threshold}", verbose)

            # Step 1: Predict test data with current ensemble
            test_proba = self._weighted_predict_proba(
                self.models, self._model_names, self.model_weights,
                X_test, X_test_enh, X_test_nonsh
            )
            test_pred = np.argmax(test_proba, axis=1)
            test_confidence = np.max(test_proba, axis=1)

            # Step 2: Select high-confidence pseudo-labels
            mask = test_confidence >= threshold
            n_pseudo = mask.sum()
            n_test = len(X_test)
            pseudo_ratio = n_pseudo / n_test

            log(f"    Selected {n_pseudo}/{n_test} pseudo-labels "
                f"({pseudo_ratio:.1%})", verbose)

            # Check minimum pseudo-label ratio
            min_ratio = cfg.get('st_min_pseudo_ratio', 0.05)
            if pseudo_ratio < min_ratio:
                log(f"    Below min ratio ({min_ratio:.0%}), stopping", verbose)
                break

            # Cap pseudo-label ratio
            max_ratio = cfg.get('st_max_pseudo_ratio', 0.5)
            if pseudo_ratio > max_ratio:
                n_keep = int(max_ratio * n_test)
                conf_order = np.argsort(-test_confidence)
                keep_idx = conf_order[:n_keep]
                mask = np.zeros(n_test, dtype=bool)
                mask[keep_idx] = True
                n_pseudo = mask.sum()
                log(f"    Capped to {n_pseudo} pseudo-labels ({max_ratio:.0%})", verbose)

            X_pseudo = X_test[mask]
            y_pseudo = test_pred[mask]
            pseudo_confidence = test_confidence[mask]

            # Step 3: Combine source + pseudo-labeled target
            X_combined = np.vstack([X_train, X_pseudo])
            y_combined = np.hstack([y_train, y_pseudo])

            # Sample weights: source=1.0, pseudo=confidence
            if cfg.get('st_use_sample_weight', True):
                source_weights = np.ones(len(y_train))
                pseudo_weights = pseudo_confidence
                sample_weight = np.hstack([source_weights, pseudo_weights])
            else:
                sample_weight = None

            # Prepare enhanced and nonshifted combined features
            if self._use_enhanced:
                shift_pseudo = self.characterizer.compute_shift_features(X_pseudo)
                X_combined_enh = np.vstack([X_train_enh,
                                            np.hstack([X_pseudo, shift_pseudo])])
            else:
                X_combined_enh = X_combined

            if X_train_nonsh is not None and len(self._nonshifted) > 0:
                X_combined_nonsh = np.vstack([X_train_nonsh, X_pseudo[:, self._nonshifted]])
            else:
                X_combined_nonsh = None

            # Step 4: Retrain models on combined data
            specs = self._build_model_specs(self._use_enhanced)
            new_models, new_model_names = self._train_models(
                specs, X_combined, y_combined, X_combined_enh,
                X_combined_nonsh, sample_weight=sample_weight
            )

            # Step 5: Compute new VWF weights (target-weighted)
            new_val_accuracies = self._compute_val_accuracies(
                new_models, X_val, X_val_enh, X_val_nonsh, y_val,
                use_target_weighted=self._val_weights is not None
            )

            if self.use_vwf and len(new_models) >= 2:
                new_weights = self._compute_vwf_weights(new_val_accuracies, new_model_names)

                # Safety check: VWF vs equal weight on target-weighted validation
                if cfg.get('vwf_safety_check', True):
                    vwf_pred = np.argmax(self._weighted_predict_proba(
                        new_models, new_model_names, new_weights,
                        X_val, X_val_enh, X_val_nonsh), axis=1)
                    vwf_acc = self._target_weighted_accuracy(y_val, vwf_pred, self._val_weights)

                    eq_weights = np.ones(len(new_models)) / len(new_models)
                    eq_pred = np.argmax(self._weighted_predict_proba(
                        new_models, new_model_names, eq_weights,
                        X_val, X_val_enh, X_val_nonsh), axis=1)
                    eq_acc = self._target_weighted_accuracy(y_val, eq_pred, self._val_weights)

                    if vwf_acc < eq_acc:
                        new_weights = eq_weights
            else:
                n_new = len(new_models)
                new_weights = np.ones(n_new) / max(n_new, 1)

            # Step 6: Safety check on target-weighted validation accuracy
            new_pred = np.argmax(self._weighted_predict_proba(
                new_models, new_model_names, new_weights,
                X_val, X_val_enh, X_val_nonsh), axis=1)
            new_acc = self._target_weighted_accuracy(y_val, new_pred, self._val_weights)

            self.st_iterations += 1
            self.st_pseudo_counts.append(int(n_pseudo))
            self.st_val_accs.append(new_acc)

            log(f"    New val acc={new_acc:.4f} (best={best_val_acc:.4f}, "
                f"single={best_single_val_acc:.4f})", verbose)

            if cfg.get('st_safety_check', True):
                # Accept only if:
                # 1. New accuracy >= best so far
                # 2. New accuracy >= best single model (global safety)
                if new_acc >= best_val_acc and new_acc >= best_single_val_acc:
                    best_val_acc = new_acc
                    best_models = new_models
                    best_weights = new_weights
                    best_val_accuracies = new_val_accuracies
                    best_model_names = new_model_names
                    self.st_improved = True
                    log(f"    ACCEPTED: val acc improved and >= single model", verbose)
                else:
                    log(f"    REJECTED: val acc degraded or below single model, reverting", verbose)
                    break
            else:
                best_val_acc = new_acc
                best_models = new_models
                best_weights = new_weights
                best_val_accuracies = new_val_accuracies
                best_model_names = new_model_names
                self.st_improved = True

        # Use best models
        self.models = best_models
        self.model_weights = best_weights
        self.val_accuracies = best_val_accuracies
        self._model_names = best_model_names

        if self.st_improved:
            log(f"  Phase 2 (ST): Completed {self.st_iterations} iterations, "
                f"IMPROVED (best val acc={best_val_acc:.4f})", verbose)
        else:
            log(f"  Phase 2 (ST): No improvement after {self.st_iterations} iterations, "
                f"using Phase 1 models", verbose)

    def predict(self, X_test):
        """Predict labels for test data."""
        start = time.time()

        # Prepare enhanced and nonshifted test features
        if self._use_enhanced and self.characterizer is not None:
            shift_test = self.characterizer.compute_shift_features(X_test)
            X_test_enh = np.hstack([X_test, shift_test])
        else:
            X_test_enh = X_test

        if self._use_enhanced and len(self._nonshifted) > 0 and len(self._nonshifted) < self.n_features:
            X_test_nonsh = X_test[:, self._nonshifted]
        else:
            X_test_nonsh = None

        proba = self._weighted_predict_proba(
            self.models, self._model_names, self.model_weights,
            X_test, X_test_enh, X_test_nonsh
        )
        y_pred = np.argmax(proba, axis=1)

        self.predict_time = time.time() - start
        return y_pred

    def predict_proba(self, X_test):
        """Predict class probabilities for test data."""
        # Prepare enhanced and nonshifted test features
        if self._use_enhanced and self.characterizer is not None:
            shift_test = self.characterizer.compute_shift_features(X_test)
            X_test_enh = np.hstack([X_test, shift_test])
        else:
            X_test_enh = X_test

        if self._use_enhanced and len(self._nonshifted) > 0 and len(self._nonshifted) < self.n_features:
            X_test_nonsh = X_test[:, self._nonshifted]
        else:
            X_test_nonsh = None

        return self._weighted_predict_proba(
            self.models, self._model_names, self.model_weights,
            X_test, X_test_enh, X_test_nonsh
        )

    def evaluate(self, X_test, y_test):
        """Evaluate on test data and return metrics."""
        from sklearn.metrics import f1_score, roc_auc_score
        from sklearn.preprocessing import label_binarize

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        n_classes = self.n_classes
        acc = float(accuracy_score(y_test, y_pred))
        f1_macro = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
        f1_micro = float(f1_score(y_test, y_pred, average='micro', zero_division=0))

        # AUC
        try:
            if n_classes == 2:
                auc = float(roc_auc_score(y_test, y_proba[:, 1]))
            else:
                y_bin = label_binarize(y_test, classes=list(range(n_classes)))
                auc = float(roc_auc_score(y_bin, y_proba, multi_class='ovr',
                                          average='macro'))
        except Exception:
            auc = 0.0

        return {
            'metrics': {
                'accuracy': acc,
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'auc': auc,
                'n_models': len(self.models),
                'n_shifted': self.shift_report['n_shifted'] if self.shift_report else 0,
                'use_enhanced': self._use_enhanced,
                'use_vwf': self.use_vwf,
                'mean_ks': self.shift_report['mean_ks'] if self.shift_report else 0.0,
                'max_ks': self.shift_report['max_ks'] if self.shift_report else 0.0,
                'train_time': self.train_time,
                'predict_time': self.predict_time,
                'model_weights': {k: float(v) for k, v in zip(self._model_names, self.model_weights)},
                'val_accuracies': {k: float(v) for k, v in self.val_accuracies.items()},
                'st_iterations': self.st_iterations,
                'st_pseudo_counts': self.st_pseudo_counts,
                'st_improved': self.st_improved,
            }
        }

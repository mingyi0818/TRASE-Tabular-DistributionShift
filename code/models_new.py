"""
Models for Causal Invariant Few-Shot Learning.

Baselines:
  - Logistic Regression (LR)
  - Random Forest (RF)
  - XGBoost
  - LightGBM
  - TabPFN (cloud client)
  - ProtoNet (prototype network)

Our methods (CIA = Causal Invariant Adapter):
  - CIA-LR: causal features + LR
  - CIA-XGBoost: causal features + XGBoost
  - CIA-TabPFN: causal features + TabPFN wrapper

Each method follows the same interface:
  fit(X_train, y_train) -> self
  predict(X_test) -> y_pred
  predict_proba(X_test) -> P(y|X)
"""

import os
import sys
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import warnings

warnings.filterwarnings('ignore')

# Optional imports with graceful fallback
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

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================
# Base Classifier Wrapper
# ============================================================

class BaseClassifier:
    """Base class for all classifiers."""

    def __init__(self, name):
        self.name = name
        self.model = None
        self.fit_time = 0
        self.predict_time = 0

    def fit(self, X, y):
        start = time.time()
        self._fit(X, y)
        self.fit_time = time.time() - start
        return self

    def predict(self, X):
        start = time.time()
        pred = self._predict(X)
        self.predict_time = time.time() - start
        return pred

    def predict_proba(self, X):
        return self._predict_proba(X)

    def _fit(self, X, y):
        raise NotImplementedError

    def _predict(self, X):
        raise NotImplementedError

    def _predict_proba(self, X):
        raise NotImplementedError


class LRClassifier(BaseClassifier):
    """Logistic Regression classifier."""

    def __init__(self, **kwargs):
        super().__init__('LR')
        self.model = LogisticRegression(
            max_iter=1000, random_state=kwargs.get('random_state', 42),
            **{k: v for k, v in kwargs.items() if k != 'random_state'}
        )

    def _fit(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)

    def _predict_proba(self, X):
        return self.model.predict_proba(X)


class RFClassifier(BaseClassifier):
    """Random Forest classifier."""

    def __init__(self, **kwargs):
        super().__init__('RF')
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=kwargs.get('random_state', 42),
            **{k: v for k, v in kwargs.items() if k != 'random_state'}
        )

    def _fit(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)

    def _predict_proba(self, X):
        return self.model.predict_proba(X)


class XGBoostClassifier(BaseClassifier):
    """XGBoost classifier."""

    def __init__(self, **kwargs):
        super().__init__('XGBoost')
        if not HAS_XGB:
            raise ImportError("xgboost not installed")
        self.n_classes = None
        self.model = None
        self.params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': kwargs.get('random_state', 42),
            'use_label_encoder': False,
            'eval_metric': 'logloss',
        }

    def _fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)

    def _predict_proba(self, X):
        return self.model.predict_proba(X)


class LightGBMClassifier(BaseClassifier):
    """LightGBM classifier."""

    def __init__(self, **kwargs):
        super().__init__('LightGBM')
        if not HAS_LGB:
            raise ImportError("lightgbm not installed")
        self.model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=kwargs.get('random_state', 42), verbose=-1,
            **{k: v for k, v in kwargs.items() if k != 'random_state'}
        )

    def _fit(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)

    def _predict_proba(self, X):
        return self.model.predict_proba(X)


# ============================================================
# TabPFN Classifier (cloud client)
# ============================================================

class TabPFNClassifier(BaseClassifier):
    """TabPFN classifier using cloud client."""

    _initialized = False

    @classmethod
    def ensure_initialized(cls):
        if not cls._initialized:
            try:
                import tabpfn_client
                tabpfn_client.init()
                cls._initialized = True
            except Exception as e:
                raise ImportError(f"Failed to init tabpfn_client: {e}")

    def __init__(self, **kwargs):
        super().__init__('TabPFN')
        self.ensure_initialized()
        from tabpfn_client import TabPFNClassifier as _TabPFN
        self.model = _TabPFN()
        self.n_classes = None

    def _fit(self, X, y):
        self.n_classes = len(np.unique(y))
        # TabPFN limit: <= 10000 training samples, <= 100 features
        if X.shape[0] > 10000:
            # Subsample
            idx = np.random.RandomState(42).choice(
                X.shape[0], 10000, replace=False
            )
            X = X[idx]
            y = y[idx]
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)

    def _predict_proba(self, X):
        return self.model.predict_proba(X)


# ============================================================
# ProtoNet (Prototype Network)
# ============================================================

if HAS_TORCH:
    class _ProtoNetModel(nn.Module):
        """Simple prototype network for tabular data."""

        def __init__(self, input_dim, hidden_dim=128, output_dim=64):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            return self.encoder(x)

        def compute_prototypes(self, x_support, y_support, n_way):
            """Compute class prototypes as mean of support embeddings."""
            embeddings = self.forward(x_support)
            prototypes = []
            for c in range(n_way):
                mask = (y_support == c)
                if mask.sum() > 0:
                    proto = embeddings[mask].mean(dim=0)
                else:
                    proto = torch.zeros(embeddings.shape[1])
                prototypes.append(proto)
            return torch.stack(prototypes)

        def classify(self, x_query, prototypes):
            """Classify query points by nearest prototype (euclidean)."""
            embeddings = self.forward(x_query)
            # Euclidean distance
            dists = torch.cdist(embeddings, prototypes)
            return -dists  # negative distance as logits


class ProtoNetClassifier(BaseClassifier):
    """Prototype Network classifier for few-shot tabular learning."""

    def __init__(self, input_dim=None, n_classes=None, **kwargs):
        super().__init__('ProtoNet')
        if not HAS_TORCH:
            raise ImportError("torch not installed")
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.model = None
        self.lr = kwargs.get('lr', 1e-3)
        self.epochs = kwargs.get('epochs', 100)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.support_X = None
        self.support_y = None

    def _fit(self, X, y):
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        if self.n_classes is None:
            self.n_classes = len(np.unique(y))

        self.model = _ProtoNetModel(self.input_dim).to(self.device)

        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Meta-training: episodic training with episodes from training data
        n_way = self.n_classes
        n_episodes = min(self.epochs, 100)

        for ep in range(n_episodes):
            # Sample support and query from training data
            support_idx = []
            query_idx = []
            for c in range(n_way):
                c_idx = np.where(y == c)[0]
                if len(c_idx) < 2:
                    support_idx.extend(c_idx)
                    continue
                k = min(5, len(c_idx) // 2)
                s_idx = np.random.choice(c_idx, k, replace=False)
                q_idx = np.setdiff1d(c_idx, s_idx)
                if len(q_idx) > 10:
                    q_idx = np.random.choice(q_idx, 10, replace=False)
                support_idx.extend(s_idx)
                query_idx.extend(q_idx)

            if len(query_idx) == 0:
                continue

            s_X = X_t[support_idx]
            s_y = y_t[support_idx]
            q_X = X_t[query_idx]
            q_y = y_t[query_idx]

            # Compute prototypes and classify
            prototypes = self.model.compute_prototypes(s_X, s_y, n_way)
            logits = self.model.classify(q_X, prototypes)
            loss = F.cross_entropy(logits, q_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Store support set for prediction
        self.support_X = X_t
        self.support_y = y_t

    def _predict(self, X):
        proba = self._predict_proba(X)
        return np.argmax(proba, axis=1)

    def _predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            # Use all training data as support
            prototypes = self.model.compute_prototypes(
                self.support_X, self.support_y, self.n_classes
            )
            logits = self.model.classify(X_t, prototypes)
            proba = F.softmax(logits, dim=1).cpu().numpy()
        return proba


# ============================================================
# CIA: Causal Invariant Adapter
# ============================================================

class CausalInvariantAdapter(BaseClassifier):
    """
    Causal Invariant Adapter (CIA).

    Wraps any base classifier, using only causally-discovered features.
    """

    def __init__(self, base_classifier_name, causal_features, **kwargs):
        """
        Args:
            base_classifier_name: 'LR', 'XGBoost', or 'TabPFN'
            causal_features: list of feature indices to use
            **kwargs: passed to base classifier
        """
        name = f'CIA-{base_classifier_name}'
        super().__init__(name)
        self.base_classifier_name = base_classifier_name
        self.causal_features = causal_features
        self.kwargs = kwargs
        self.base_model = None
        self.n_classes = None

    def _get_base_model(self):
        if self.base_classifier_name == 'LR':
            return LRClassifier(**self.kwargs)
        elif self.base_classifier_name == 'XGBoost':
            return XGBoostClassifier(**self.kwargs)
        elif self.base_classifier_name == 'RF':
            return RFClassifier(**self.kwargs)
        elif self.base_classifier_name == 'LightGBM':
            return LightGBMClassifier(**self.kwargs)
        elif self.base_classifier_name == 'TabPFN':
            return TabPFNClassifier(**self.kwargs)
        else:
            raise ValueError(f"Unknown base classifier: {self.base_classifier_name}")

    def _select_features(self, X):
        """Select only causal features."""
        if len(self.causal_features) == 0:
            return X
        return X[:, self.causal_features]

    def _fit(self, X, y):
        self.n_classes = len(np.unique(y))
        X_selected = self._select_features(X)
        self.base_model = self._get_base_model()
        self.base_model._fit(X_selected, y)

    def _predict(self, X):
        X_selected = self._select_features(X)
        return self.base_model._predict(X_selected)

    def _predict_proba(self, X):
        X_selected = self._select_features(X)
        return self.base_model._predict_proba(X_selected)


# ============================================================
# SACIS: Stability-Aware Causal Invariant Selection
# ============================================================

class SACISAdapter(BaseClassifier):
    """
    SACIS Adapter: wraps a base classifier using SACIS-selected features.

    Unlike CIA which uses a single PC run, SACIS uses:
    1. Bootstrap stability-aware causal discovery (B=200 runs)
    2. Three-tier stability-stratified selection (Hard/Soft/Exclude)
    3. Synthetic environment + gradient variance verification

    The SACIS-selected features have:
    - Theoretical false positive control (Meinshausen-Buhlmann bound)
    - Empirical invariance verification (gradient variance < threshold)
    """

    def __init__(self, base_classifier_name, sacis_features, sacis_info=None, **kwargs):
        """
        Args:
            base_classifier_name: 'LR', 'XGBoost', 'RF', 'LightGBM', 'TabPFN'
            sacis_features: list of SACIS-selected feature indices
            sacis_info: dict with SACIS metadata (stability scores, tiers, etc.)
            **kwargs: passed to base classifier
        """
        name = f'SACIS-{base_classifier_name}'
        super().__init__(name)
        self.base_classifier_name = base_classifier_name
        self.sacis_features = sacis_features
        self.sacis_info = sacis_info or {}
        self.kwargs = kwargs
        self.base_model = None
        self.n_classes = None

    def _get_base_model(self):
        if self.base_classifier_name == 'LR':
            return LRClassifier(**self.kwargs)
        elif self.base_classifier_name == 'XGBoost':
            return XGBoostClassifier(**self.kwargs)
        elif self.base_classifier_name == 'RF':
            return RFClassifier(**self.kwargs)
        elif self.base_classifier_name == 'LightGBM':
            return LightGBMClassifier(**self.kwargs)
        elif self.base_classifier_name == 'TabPFN':
            return TabPFNClassifier(**self.kwargs)
        else:
            raise ValueError(f"Unknown base classifier: {self.base_classifier_name}")

    def _select_features(self, X):
        """Select only SACIS-verified features."""
        if len(self.sacis_features) == 0:
            return X
        return X[:, self.sacis_features]

    def _fit(self, X, y):
        self.n_classes = len(np.unique(y))
        X_selected = self._select_features(X)
        self.base_model = self._get_base_model()
        self.base_model._fit(X_selected, y)

    def _predict(self, X):
        X_selected = self._select_features(X)
        return self.base_model._predict(X_selected)

    def _predict_proba(self, X):
        X_selected = self._select_features(X)
        return self.base_model._predict_proba(X_selected)


# ============================================================
# Evaluation utilities
# ============================================================

def evaluate_predictions(y_true, y_pred, y_proba=None, n_classes=None):
    """Compute classification metrics."""
    if n_classes is None:
        n_classes = len(np.unique(y_true))

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_micro': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
    }

    # AUC
    if y_proba is not None and n_classes > 1:
        try:
            if n_classes == 2:
                if y_proba.shape[1] == 2:
                    metrics['auc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    metrics['auc'] = float(roc_auc_score(y_true, y_proba))
            else:
                y_bin = label_binarize(y_true, classes=list(range(n_classes)))
                if y_bin.shape[1] == y_proba.shape[1]:
                    metrics['auc'] = float(
                        roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')
                    )
        except Exception:
            metrics['auc'] = 0.5
    else:
        metrics['auc'] = 0.5

    return metrics


def create_method(method_name, causal_features=None, input_dim=None,
                   n_classes=None, random_state=42):
    """
    Factory function to create a method by name.

    Args:
        method_name: one of 'LR', 'RF', 'XGBoost', 'LightGBM', 'TabPFN',
                     'ProtoNet', 'CIA-LR', 'CIA-XGBoost', 'CIA-TabPFN'
        causal_features: list of feature indices (required for CIA methods)
        input_dim: input dimension (required for ProtoNet)
        n_classes: number of classes
        random_state: random seed
    """
    kwargs = {'random_state': random_state}

    if method_name == 'LR':
        return LRClassifier(**kwargs)
    elif method_name == 'RF':
        return RFClassifier(**kwargs)
    elif method_name == 'XGBoost':
        return XGBoostClassifier(**kwargs)
    elif method_name == 'LightGBM':
        return LightGBMClassifier(**kwargs)
    elif method_name == 'TabPFN':
        return TabPFNClassifier(**kwargs)
    elif method_name == 'ProtoNet':
        return ProtoNetClassifier(input_dim=input_dim, n_classes=n_classes, **kwargs)
    elif method_name.startswith('CIA-'):
        base_name = method_name[4:]  # Extract base classifier name
        if causal_features is None:
            raise ValueError("causal_features required for CIA methods")
        return CausalInvariantAdapter(base_name, causal_features, **kwargs)
    elif method_name.startswith('SACIS-'):
        base_name = method_name[6:]  # Extract base classifier name
        if causal_features is None:
            raise ValueError("sacis_features required for SACIS methods")
        return SACISAdapter(base_name, causal_features, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method_name}")


# ============================================================
# Available methods list
# ============================================================

ALL_METHODS = ['LR', 'RF', 'XGBoost', 'LightGBM', 'TabPFN', 'ProtoNet',
               'CIA-LR', 'CIA-XGBoost', 'CIA-TabPFN',
               'SACIS-LR', 'SACIS-XGBoost']

BASELINE_METHODS = ['LR', 'RF', 'XGBoost', 'LightGBM', 'TabPFN', 'ProtoNet']

CIA_METHODS = ['CIA-LR', 'CIA-XGBoost', 'CIA-TabPFN']

SACIS_METHODS = ['SACIS-LR', 'SACIS-XGBoost']


if __name__ == '__main__':
    # Quick test
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)

    for method_name in ['LR', 'RF', 'XGBoost', 'LightGBM']:
        try:
            model = create_method(method_name, random_state=42)
            model.fit(X[:150], y[:150])
            pred = model.predict(X[150:])
            metrics = evaluate_predictions(y[150:], pred)
            print(f"{method_name}: acc={metrics['accuracy']:.4f}, "
                  f"f1={metrics['f1_macro']:.4f}, fit={model.fit_time:.3f}s")
        except Exception as e:
            print(f"{method_name}: FAILED - {e}")

    # Test CIA
    causal_features = [0, 1, 2]
    model = create_method('CIA-LR', causal_features=causal_features, random_state=42)
    model.fit(X[:150], y[:150])
    pred = model.predict(X[150:])
    metrics = evaluate_predictions(y[150:], pred)
    print(f"CIA-LR: acc={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}")

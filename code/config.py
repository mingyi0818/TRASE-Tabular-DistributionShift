import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATASETS = {
    'telco': {
        'name': 'Telco-Customer-Churn',
        'path': os.path.join(RAW_DATA_DIR, 'telco', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'),
        'target': 'Churn',
        'n_way': 2
    },
    'bank': {
        'name': 'Bank-Marketing',
        'path': os.path.join(RAW_DATA_DIR, 'bank', 'bank.csv'),
        'target': 'deposit',
        'n_way': 2
    },
    'adult': {
        'name': 'Adult-Income',
        'path': os.path.join(RAW_DATA_DIR, 'adult', 'adult.csv'),
        'target': 'income',
        'n_way': 2
    },
    'drybean': {
        'name': 'Dry-Bean',
        'path': os.path.join(RAW_DATA_DIR, 'drybean', 'dry_bean.csv'),
        'target': 'Class',
        'n_way': 5
    },
    'heart': {
        'name': 'Heart-Disease',
        'path': os.path.join(RAW_DATA_DIR, 'heart', 'heart.csv'),
        'target': 'target',
        'n_way': 2
    },
    'wine': {
        'name': 'Wine-Quality-Red',
        'path': os.path.join(RAW_DATA_DIR, 'wine', 'wine.csv'),
        'target': 'target',
        'n_way': 2
    },
    'mushroom': {
        'name': 'Secondary-Mushroom',
        'path': os.path.join(RAW_DATA_DIR, 'mushroom', 'mushroom_sample.csv'),
        'target': 'class',
        'n_way': 2
    },
    'sklearn_wine': {
        'name': 'Wine-3Class',
        'path': os.path.join(RAW_DATA_DIR, 'sklearn_wine', 'sklearn_wine.csv'),
        'target': 'target',
        'n_way': 3
    },
}

CONFIG = {
    'seed': 42,
    'n_way': 2,
    'k_shot': [1, 5, 10, 20],
    'k_query': 50,
    'train_params': {
        'batch_size': 32,
        'epochs': 50,
        'lr': 1e-3,
        'patience': 10
    },
    'evaluation': {
        'metrics': ['accuracy', 'f1', 'auc'],
        'num_episodes': 50
    }
}

BASELINE_MODELS = ['logistic', 'xgboost', 'protonet']
INCREMENTAL_MODELS = ['ce_pmn', 'meta_learner']

# === SACIS Configuration ===
SACIS_CONFIG = {
    # Module 1: Bootstrap causal discovery
    'B': 200,                          # bootstrap iterations
    'bootstrap_ratio': 0.8,            # resample ratio per bootstrap
    'alpha': 0.05,                     # PC significance level
    'max_condition_set_size': None,    # None = min(n_features, 3)
    'indep_test': 'auto',              # 'fisherz', 'chi2', 'auto'
    'n_jobs': -1,                      # parallel cores (-1 = all 24)

    # Module 2: Three-tier selection
    'threshold_hard': 0.8,             # Hard tier lower bound
    'threshold_soft': 0.5,             # Soft tier lower bound

    # Module 3: Synthetic environment verification
    'n_synthetic_envs': 5,             # number of synthetic environments
    'intervention_type': 'shift',      # 'shift', 'scale', 'shuffle', 'noise'
    'intervention_strength': 1.0,      # do-intervention strength
    'lambda_gv': 1.0,                  # gradient variance regularization weight
    'gv_hidden_dim': 64,               # GV verification MLP hidden dim
    'gv_epochs': 50,                   # GV verification training epochs
    'gv_lr': 1e-3,                     # GV verification learning rate
    'gv_threshold': 0.1,               # invariance threshold
    'use_verification': True,          # enable Module 3

    # General
    'device': 'cuda',                  # RTX Pro 2000
    'random_state': 42,
    'verbose': True,
}

SACIS_METHODS = ['SACIS-LR', 'SACIS-XGBoost']

ALL_METHODS_WITH_SACIS = [
    'LR', 'RF', 'XGBoost', 'LightGBM', 'TabPFN', 'ProtoNet',
    'CIA-LR', 'CIA-XGBoost', 'CIA-TabPFN',
    'SACIS-LR', 'SACIS-XGBoost',
]

if __name__ == '__main__':
    print("Tabular Few-Shot Incremental Learning Config Loaded")
    print(f"Datasets: {list(DATASETS.keys())}")
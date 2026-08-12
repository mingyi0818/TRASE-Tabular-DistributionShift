import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import DATASETS, CONFIG

def load_telco_data():
    df = pd.read_csv(DATASETS['telco']['path'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    df = df.drop('customerID', axis=1)
    return df

def load_bank_data():
    df = pd.read_csv(DATASETS['bank']['path'])
    return df

def load_adult_data():
    df = pd.read_csv(DATASETS['adult']['path'])
    df = df.dropna()
    return df

def load_drybean_data():
    df = pd.read_csv(DATASETS['drybean']['path'])
    return df

def load_heart_data():
    df = pd.read_csv(DATASETS['heart']['path'])
    df = df.dropna()
    return df

def load_wine_data():
    df = pd.read_csv(DATASETS['wine']['path'])
    df = df.dropna()
    return df

def load_mushroom_data():
    df = pd.read_csv(DATASETS['mushroom']['path'])
    # Secondary Mushroom dataset has intentional missing values; fill rather than drop
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna('missing')
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    return df

def load_sklearn_wine_data():
    df = pd.read_csv(DATASETS['sklearn_wine']['path'])
    df = df.dropna()
    return df

def preprocess_data(df, target_col):
    cat_cols = df.select_dtypes(include=['object']).columns
    
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=CONFIG['seed'])
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=CONFIG['seed'])
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def create_few_shot_episode(X, y, n_way, k_shot, k_query):
    unique_labels = np.unique(y)  # sorted ascending

    # Deterministically select the first n_way classes (already sorted).
    # This ensures the SAME class subset is used across all episodes,
    # which is essential for EMA prototype memory consistency: the
    # prototype stored at index i always corresponds to the same original
    # class.  When n_way == len(unique_labels) (e.g. 2-way binary tasks)
    # all classes are selected and the behaviour is unchanged.  When
    # n_way < len(unique_labels) (e.g. Dry Bean, 5-of-7) the same
    # n_way classes are used every episode, preventing prototype
    # contamination across different class subsets.
    selected_labels = unique_labels[:n_way]

    # Remap selected labels to [0, n_way-1] so logits and targets align.
    label_map = {orig: idx for idx, orig in enumerate(selected_labels)}
    
    support_indices = []
    query_indices = []
    
    for label in selected_labels:
        label_indices = np.where(y == label)[0]
        np.random.shuffle(label_indices)
        
        support_indices.extend(label_indices[:k_shot])
        query_indices.extend(label_indices[k_shot:k_shot+k_query])
    
    np.random.shuffle(support_indices)
    np.random.shuffle(query_indices)
    
    support_y = np.array([label_map[v] for v in y[support_indices]])
    query_y = np.array([label_map[v] for v in y[query_indices]])
    
    return X[support_indices], support_y, X[query_indices], query_y

def load_all_datasets():
    datasets_data = {}
    
    loaders = {
        'telco': load_telco_data,
        'bank': load_bank_data,
        'adult': load_adult_data,
        'drybean': load_drybean_data,
        'heart': load_heart_data,
        'wine': load_wine_data,
        'mushroom': load_mushroom_data,
        'sklearn_wine': load_sklearn_wine_data,
    }
    
    for name, loader in loaders.items():
        try:
            df = loader()
            target_col = DATASETS[name]['target']
            X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(df, target_col)
            
            datasets_data[name] = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'scaler': scaler,
                'df': df,
                'input_dim': X_train.shape[1],
                'num_classes': len(np.unique(y_train)),
                'n_way': DATASETS[name].get('n_way', CONFIG['n_way'])
            }
            
            print(f"Loaded {DATASETS[name]['name']}: {len(X_train)} training samples, {X_train.shape[1]} features, {datasets_data[name]['num_classes']} classes, n_way={datasets_data[name]['n_way']}")
        except Exception as e:
            print(f"Failed to load {name}: {str(e)}")
    
    return datasets_data

if __name__ == '__main__':
    datasets = load_all_datasets()
    print(f"Loaded {len(datasets)} datasets")
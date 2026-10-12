"""Approach 1: Classical ML — Random Forest + XGBoost with feature selection."""
import os
import sys
import time
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.common.pipeline import load_processed_splits
from src.eval.harness import ModelWrapper, evaluate_model, plot_confusion_matrix, RESULTS_DIR

class RFWrapper(ModelWrapper):
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)
        self.selected_features = None
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    def save(self, path):
        joblib.dump({'model': self.model, 'features': self.selected_features}, path)
    def load(self, path):
        data = joblib.load(path)
        self.model = data['model']
        self.selected_features = data.get('features')
    def param_count(self):
        return sum(t.tree_.node_count for t in self.model.estimators_)

class XGBWrapper(ModelWrapper):
    def __init__(self, n_classes=2, **kwargs):
        self.n_classes = n_classes
        if n_classes > 2:
            kwargs['objective'] = 'multi:softprob'
            kwargs['num_class'] = n_classes
        else:
            kwargs['objective'] = 'binary:logistic'
        kwargs['eval_metric'] = 'mlogloss' if n_classes > 2 else 'logloss'
        kwargs['tree_method'] = 'hist'
        self.params = kwargs
        self.model = None
        self.selected_features = None
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, 'train')]
        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'eval'))
        n_rounds = self.params.pop('n_estimators', 300)
        self.model = xgb.train(
            self.params, dtrain, num_boost_round=n_rounds,
            evals=evals, early_stopping_rounds=10, verbose_eval=False
        )
    def predict(self, X):
        dmat = xgb.DMatrix(X)
        preds = self.model.predict(dmat)
        if self.n_classes > 2:
            return np.argmax(preds, axis=1)
        return (preds > 0.5).astype(int)
    def predict_proba(self, X):
        dmat = xgb.DMatrix(X)
        preds = self.model.predict(dmat)
        if self.n_classes > 2:
            return preds
        return np.column_stack([1 - preds, preds])
    def save(self, path):
        self.model.save_model(str(path))
    def load(self, path):
        self.model = xgb.Booster()
        self.model.load_model(str(path))
    def param_count(self):
        if self.model is None:
            return 0
        dump = self.model.get_dump()
        return sum(d.count('leaf=') for d in dump)

def remove_correlated(X, feature_names, threshold=0.95):
    """Remove highly correlated features."""
    corr = np.abs(np.corrcoef(X.T))
    upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
    to_drop = set()
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[1]):
            if corr[i, j] > threshold:
                to_drop.add(j)
    keep_mask = [i not in to_drop for i in range(len(feature_names))]
    kept_names = [f for f, k in zip(feature_names, keep_mask) if k]
    print(f"Removed {len(to_drop)} correlated features, kept {len(kept_names)}")
    return X[:, keep_mask], kept_names, keep_mask

def select_top_k(X_train, y_train, feature_names, k=30, method='rf'):
    """Select top-k features by importance."""
    if k == -1 or k >= len(feature_names):
        return list(range(len(feature_names))), feature_names
    if method == 'rf':
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_
    else:
        importances = mutual_info_classif(X_train, y_train, random_state=42)
    top_idx = np.argsort(importances)[-k:][::-1]
    selected_names = [feature_names[i] for i in top_idx]
    return top_idx.tolist(), selected_names

def apply_smote(X_train, y_train):
    """Apply SMOTE to training data."""
    unique, counts = np.unique(y_train, return_counts=True)
    min_count = counts.min()
    if min_count < 6:
        k = max(1, min_count - 1)
    else:
        k = 5
    sm = SMOTE(random_state=42, k_neighbors=k)
    return sm.fit_resample(X_train, y_train)

def train_and_evaluate(task='binary', seeds=None):
    """Full training pipeline for Approach 1."""
    with open(PROJECT_ROOT / "configs" / "approach1.yaml") as f:
        a1_config = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)

    splits, family_map, int_to_family = load_processed_splits()
    if seeds is None:
        seeds = config.get('seeds', [42])

    X_train = splits['X_train']
    X_val = splits['X_val']
    X_test = splits['X_test']
    feature_names = splits['feature_cols']

    if task == 'binary':
        y_train = splits['y_binary_train']
        y_val = splits['y_binary_val']
        y_test = splits['y_binary_test']
        class_names = ['Benign', 'Attack']
    else:
        y_train = splits['y_family_train']
        y_val = splits['y_family_val']
        y_test = splits['y_family_test']
        class_names = [int_to_family[i] for i in range(len(int_to_family))]

    # Feature selection
    X_train_fs, kept_names, keep_mask = remove_correlated(
        X_train, feature_names, a1_config['correlation_threshold']
    )
    X_val_fs = X_val[:, keep_mask]
    X_test_fs = X_test[:, keep_mask]

    best_k = 30
    best_f1 = 0
    for k in a1_config['feature_k_values']:
        top_idx, sel_names = select_top_k(X_train_fs, y_train, kept_names, k)
        rf_temp = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
        if k == -1 or k >= len(kept_names):
            rf_temp.fit(X_train_fs, y_train)
            val_f1 = f1_score(y_val, rf_temp.predict(X_val_fs), average='macro', zero_division=0)
        else:
            rf_temp.fit(X_train_fs[:, top_idx], y_train)
            val_f1 = f1_score(y_val, rf_temp.predict(X_val_fs[:, top_idx]), average='macro', zero_division=0)
        print(f"  k={k}: val macro-F1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_k = k

    top_idx, selected_names = select_top_k(X_train_fs, y_train, kept_names, best_k)
    if best_k == -1 or best_k >= len(kept_names):
        X_tr = X_train_fs
        X_v = X_val_fs
        X_te = X_test_fs
        selected_names = kept_names
    else:
        X_tr = X_train_fs[:, top_idx]
        X_v = X_val_fs[:, top_idx]
        X_te = X_test_fs[:, top_idx]

    # Imbalance handling
    best_method = 'balanced'
    best_imb_f1 = 0
    for method in a1_config['imbalance_methods']:
        if method == 'smote':
            try:
                X_tr_imb, y_tr_imb = apply_smote(X_tr, y_train)
            except Exception as e:
                print(f"  SMOTE failed: {e}")
                continue
            rf_temp = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
        elif method == 'balanced':
            X_tr_imb, y_tr_imb = X_tr, y_train
            rf_temp = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1,
                                              random_state=42, class_weight='balanced')
        else:
            X_tr_imb, y_tr_imb = X_tr, y_train
            rf_temp = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
        rf_temp.fit(X_tr_imb, y_tr_imb)
        val_f1 = f1_score(y_val, rf_temp.predict(X_v), average='macro', zero_division=0)
        print(f"  imbalance={method}: val macro-F1={val_f1:.4f}")
        if val_f1 > best_imb_f1:
            best_imb_f1 = val_f1
            best_method = method

    print(f"\nBest: k={best_k}, imbalance={best_method}")

    all_rf_metrics = []
    all_xgb_metrics = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        # Random Forest
        rf_kwargs = {
            'n_estimators': 300, 'max_depth': 20, 'min_samples_leaf': 2,
            'max_features': 'sqrt', 'n_jobs': -1, 'random_state': seed,
        }
        if best_method == 'balanced':
            rf_kwargs['class_weight'] = 'balanced'

        if best_method == 'smote':
            try:
                X_tr_final, y_tr_final = apply_smote(X_tr, y_train)
            except:
                X_tr_final, y_tr_final = X_tr, y_train
        else:
            X_tr_final, y_tr_final = X_tr, y_train

        rf_model = RFWrapper(**rf_kwargs)
        start = time.time()
        rf_model.fit(X_tr_final, y_tr_final)
        rf_train_time = time.time() - start

        rf_path = RESULTS_DIR / f"rf_{task}_seed{seed}.joblib"
        rf_model.save(rf_path)

        rf_metrics, rf_cm, _ = evaluate_model(
            rf_model, X_te, y_test, class_names,
            task=task, approach_name='RF', seed=seed,
            model_path=rf_path, train_time=rf_train_time
        )
        all_rf_metrics.append(rf_metrics)
        plot_confusion_matrix(rf_cm, class_names,
                              f'Random Forest — {task} (seed {seed})',
                              RESULTS_DIR / f"Fig_CM_RF_{task}_seed{seed}.png")
        print(f"  RF: acc={rf_metrics['accuracy']:.4f}, macro-F1={rf_metrics['f1_macro']:.4f}")

        # XGBoost
        n_classes = len(class_names)
        xgb_model = XGBWrapper(
            n_classes=n_classes,
            max_depth=8, learning_rate=0.1,
            n_estimators=300, subsample=0.8,
            colsample_bytree=0.8,
        )
        start = time.time()
        xgb_model.fit(X_tr_final, y_tr_final, X_v, y_val if task != 'binary' else splits['y_binary_val'])
        xgb_train_time = time.time() - start

        xgb_path = RESULTS_DIR / f"xgb_{task}_seed{seed}.model"
        xgb_model.save(xgb_path)

        xgb_metrics, xgb_cm, _ = evaluate_model(
            xgb_model, X_te, y_test, class_names,
            task=task, approach_name='XGBoost', seed=seed,
            model_path=xgb_path, train_time=xgb_train_time
        )
        all_xgb_metrics.append(xgb_metrics)
        plot_confusion_matrix(xgb_cm, class_names,
                              f'XGBoost — {task} (seed {seed})',
                              RESULTS_DIR / f"Fig_CM_XGB_{task}_seed{seed}.png")
        print(f"  XGB: acc={xgb_metrics['accuracy']:.4f}, macro-F1={xgb_metrics['f1_macro']:.4f}")

    # Feature importance plot
    plot_feature_importance(rf_model, selected_names, task)

    # Save best config
    best_config = {
        'best_k': best_k, 'best_imbalance': best_method,
        'selected_features': selected_names,
        'rf_params': rf_kwargs, 'task': task,
    }
    with open(PROJECT_ROOT / "configs" / f"approach1_best_{task}.yaml", 'w') as f:
        yaml.dump(best_config, f)

    # Save tuning log
    tuning_df = pd.DataFrame([{
        'k': best_k, 'imbalance': best_method,
        'val_f1': best_imb_f1, 'task': task,
    }])
    tuning_df.to_csv(RESULTS_DIR / f"approach1_tuning_{task}.csv", index=False)

    return all_rf_metrics, all_xgb_metrics, rf_model, selected_names

def plot_feature_importance(model, feature_names, task):
    """Plot global feature importance."""
    importances = model.model.feature_importances_
    idx = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(idx)), importances[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel('Feature Importance (Gini)')
    ax.set_title(f'Top 20 Features — Random Forest ({task})')
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"Fig_feature_importance_RF_{task}.png", dpi=300)
    plt.close()

def run_shap_analysis(model, X_test, feature_names, class_names, family_names_test, task):
    """SHAP analysis for explainability."""
    try:
        import shap
    except ImportError:
        print("SHAP not available, skipping explainability analysis")
        return
    sample_size = min(5000, len(X_test))
    idx = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test[idx]
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_sample)
    fig = plt.figure(figsize=(12, 8))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                          class_names=class_names, show=False, max_display=20)
    else:
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                          show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"Fig_SHAP_summary_{task}.png", dpi=300, bbox_inches='tight')
    plt.close('all')

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("APPROACH 1: Classical ML — Random Forest + XGBoost")
    print("=" * 60)
    print("\n--- Binary Classification ---")
    rf_bin, xgb_bin, rf_model_bin, features_bin = train_and_evaluate('binary')
    print("\n--- Multi-class Classification ---")
    rf_mc, xgb_mc, rf_model_mc, features_mc = train_and_evaluate('multiclass')
    print("\n--- SHAP Analysis ---")
    splits, _, int_to_family = load_processed_splits()
    try:
        run_shap_analysis(rf_model_mc, splits['X_test'], features_mc,
                          [int_to_family[i] for i in range(len(int_to_family))],
                          splits['family_names_test'], 'multiclass')
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
    print("\nApproach 1 training complete.")

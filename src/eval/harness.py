"""Shared evaluation harness for both approaches."""
import time
import json
import tracemalloc
import numpy as np
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)
import psutil
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

class ModelWrapper(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val):
        pass
    @abstractmethod
    def predict(self, X):
        pass
    @abstractmethod
    def predict_proba(self, X):
        pass
    @abstractmethod
    def save(self, path):
        pass
    @abstractmethod
    def load(self, path):
        pass
    @abstractmethod
    def param_count(self):
        pass

def measure_inference_latency(model, X, batch_sizes=[1, 1024], n_repeats=50):
    """Measure inference latency at different batch sizes."""
    results = {}
    for bs in batch_sizes:
        latencies = []
        for _ in range(n_repeats):
            batch = X[:bs]
            start = time.perf_counter()
            model.predict(batch)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed / len(batch))
        latencies = np.array(latencies)
        results[f'latency_batch{bs}_p50_ms'] = float(np.percentile(latencies, 50))
        results[f'latency_batch{bs}_p95_ms'] = float(np.percentile(latencies, 95))
    return results

def measure_throughput(model, X, batch_size=1024, n_repeats=10):
    """Measure throughput in flows/second."""
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        model.predict(X[:batch_size])
        times.append(time.perf_counter() - start)
    avg_time = np.mean(times)
    return float(batch_size / avg_time)

def measure_memory(model, X):
    """Measure peak memory during inference."""
    tracemalloc.start()
    model.predict(X[:1024])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)

def model_size_mb(path):
    """Get model size on disk in MB."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / (1024 * 1024)
    return 0

def evaluate_model(model, X_test, y_test, class_names, task='binary',
                    approach_name='approach', seed=42, model_path=None,
                    X_train_time=None, train_time=None):
    """Full evaluation on test split."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    n_classes = len(class_names)
    is_binary = (n_classes == 2 or task == 'binary')
    metrics = {
        'approach': approach_name,
        'task': task,
        'seed': seed,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision_macro': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
    }
    try:
        if is_binary and y_proba.shape[1] == 2:
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba[:, 1]))
            metrics['pr_auc'] = float(average_precision_score(y_test, y_proba[:, 1]))
        else:
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'))
    except Exception:
        metrics['roc_auc'] = None
        metrics['pr_auc'] = None
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    report = classification_report(y_test, y_pred, target_names=class_names,
                                    output_dict=True, zero_division=0)
    per_class = {}
    for cls_name in class_names:
        if cls_name in report:
            per_class[cls_name] = {
                'precision': report[cls_name]['precision'],
                'recall': report[cls_name]['recall'],
                'f1': report[cls_name]['f1-score'],
                'support': report[cls_name]['support'],
            }
    metrics['per_class'] = per_class
    total_attacks = sum(1 for y in y_test if y > 0) if is_binary else len(y_test)
    if is_binary:
        tn, fp, fn, tp = cm.ravel()
        metrics['fpr'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0
        metrics['fnr'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0
    latency = measure_inference_latency(model, X_test)
    metrics.update(latency)
    metrics['throughput_flows_per_sec'] = measure_throughput(model, X_test)
    metrics['peak_ram_mb'] = measure_memory(model, X_test)
    if model_path:
        metrics['model_size_mb'] = model_size_mb(model_path)
    metrics['param_count'] = model.param_count()
    if train_time is not None:
        metrics['train_time_sec'] = train_time
    result_file = RESULTS_DIR / f"{approach_name}_{task}_seed{seed}.json"
    with open(result_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    return metrics, cm, cm_normalized

def plot_confusion_matrix(cm, class_names, title, save_path, normalized=True):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    data = cm.astype(float)
    if normalized:
        data = data / data.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()

def plot_per_class_recall(metrics_list, class_names, save_path):
    """Bar chart of per-class recall for multiple approaches."""
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(class_names))
    width = 0.35
    for i, m in enumerate(metrics_list):
        recalls = [m['per_class'].get(c, {}).get('recall', 0) for c in class_names]
        ax.bar(x + i * width, recalls, width, label=m['approach'])
    ax.set_xlabel('Attack Family')
    ax.set_ylabel('Recall')
    ax.set_title('Per-Class Recall Comparison')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curves(y_test, proba_dict, class_names, save_path, task='binary'):
    """Plot ROC curves."""
    fig, ax = plt.subplots(figsize=(8, 8))
    for approach_name, y_proba in proba_dict.items():
        if task == 'binary' and y_proba.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            auc_val = roc_auc_score(y_test, y_proba[:, 1])
            ax.plot(fpr, tpr, label=f'{approach_name} (AUC={auc_val:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()

def aggregate_seeds(approach_name, task, seeds, results_dir=None):
    """Aggregate results across seeds."""
    if results_dir is None:
        results_dir = RESULTS_DIR
    all_metrics = []
    for s in seeds:
        fpath = results_dir / f"{approach_name}_{task}_seed{s}.json"
        if fpath.exists():
            with open(fpath) as f:
                all_metrics.append(json.load(f))
    if not all_metrics:
        return {}
    numeric_keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float)) and all_metrics[0][k] is not None]
    agg = {}
    for k in numeric_keys:
        vals = [m[k] for m in all_metrics if m.get(k) is not None]
        if vals:
            agg[f'{k}_mean'] = float(np.mean(vals))
            agg[f'{k}_std'] = float(np.std(vals))
    agg['approach'] = approach_name
    agg['task'] = task
    agg['n_seeds'] = len(all_metrics)
    return agg

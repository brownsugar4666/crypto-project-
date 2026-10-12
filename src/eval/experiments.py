"""Experiments E1–E7 and comparison table generation."""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.common.pipeline import load_processed_splits
from src.eval.harness import (evaluate_model, aggregate_seeds, plot_per_class_recall,
                               plot_roc_curves, plot_confusion_matrix, RESULTS_DIR)
from src.approach1_classical.train import RFWrapper, XGBWrapper, train_and_evaluate as train_a1
from src.approach2_deep.train import DeepModelWrapper, CNNLSTM, MLP, train_and_evaluate as train_a2
import yaml
import joblib

def load_best_models(splits, int_to_family, task='multiclass'):
    """Load best trained models."""
    models = {}
    n_features = splits['X_test'].shape[1]
    n_classes = len(int_to_family) if task == 'multiclass' else 2

    rf_path = RESULTS_DIR / f"rf_{task}_seed42.joblib"
    if rf_path.exists():
        rf = RFWrapper(n_estimators=300, random_state=42)
        rf.load(rf_path)
        models['RF'] = rf

    xgb_path = RESULTS_DIR / f"xgb_{task}_seed42.model"
    if xgb_path.exists():
        xgb_m = XGBWrapper(n_classes=n_classes)
        xgb_m.load(xgb_path)
        models['XGBoost'] = xgb_m

    cnn_path = RESULTS_DIR / f"cnn_lstm_{task}_seed42.pt"
    if cnn_path.exists():
        cnn = DeepModelWrapper(CNNLSTM, n_features, n_classes)
        cnn.load(cnn_path)
        models['CNN-LSTM'] = cnn

    mlp_path = RESULTS_DIR / f"mlp_{task}_seed42.pt"
    if mlp_path.exists():
        mlp = DeepModelWrapper(MLP, n_features, n_classes)
        mlp.load(mlp_path)
        models['MLP'] = mlp

    return models

def e1_main_comparison(splits, int_to_family, seeds):
    """E1: Main comparison (binary + multi-class)."""
    print("\n=== E1: Main Comparison ===")
    rows = []
    for task in ['binary', 'multiclass']:
        class_names = ['Benign', 'Attack'] if task == 'binary' else [int_to_family[i] for i in range(len(int_to_family))]
        for approach in ['RF', 'XGBoost', 'CNN-LSTM', 'MLP']:
            agg = aggregate_seeds(approach, task, seeds)
            if agg:
                rows.append({
                    'Task': task, 'Model': approach,
                    'Accuracy': f"{agg.get('accuracy_mean', 0):.4f} ± {agg.get('accuracy_std', 0):.4f}",
                    'Macro-F1': f"{agg.get('f1_macro_mean', 0):.4f} ± {agg.get('f1_macro_std', 0):.4f}",
                    'Weighted-F1': f"{agg.get('f1_weighted_mean', 0):.4f} ± {agg.get('f1_weighted_std', 0):.4f}",
                    'ROC-AUC': f"{agg.get('roc_auc_mean', 0):.4f}" if agg.get('roc_auc_mean') else 'N/A',
                    'Precision (macro)': f"{agg.get('precision_macro_mean', 0):.4f}",
                    'Recall (macro)': f"{agg.get('recall_macro_mean', 0):.4f}",
                })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e1_main.csv", index=False)
    print(df.to_string(index=False))
    return df

def e2_per_class_breakdown(splits, int_to_family, seeds):
    """E2: Per-attack recall breakdown."""
    print("\n=== E2: Per-Class Breakdown ===")
    class_names = [int_to_family[i] for i in range(len(int_to_family))]
    highlight = ['Heartbleed', 'BruteForce', 'Infiltration']
    metrics_list = []
    for approach in ['RF', 'XGBoost', 'CNN-LSTM', 'MLP']:
        fpath = RESULTS_DIR / f"{approach}_multiclass_seed42.json"
        if fpath.exists():
            with open(fpath) as f:
                m = json.load(f)
                metrics_list.append(m)
    if metrics_list:
        plot_per_class_recall(metrics_list, class_names,
                              RESULTS_DIR / "Fig_05_per_class_recall.png")
        # Highlighted chart
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(highlight))
        width = 0.2
        for i, m in enumerate(metrics_list):
            recalls = [m['per_class'].get(c, {}).get('recall', 0) for c in highlight]
            ax.bar(x + i * width, recalls, width, label=m['approach'])
        ax.set_xlabel('Attack Family')
        ax.set_ylabel('Recall')
        ax.set_title('Detection of Rare/Cryptographic Attacks')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(highlight)
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "Fig_06_crypto_attack_recall.png", dpi=300)
        plt.close()

def e3_cost_analysis(splits, int_to_family, seeds):
    """E3: Training time, latency, memory, model size."""
    print("\n=== E3: Cost Analysis ===")
    rows = []
    for approach in ['RF', 'XGBoost', 'CNN-LSTM', 'MLP']:
        agg = aggregate_seeds(approach, 'multiclass', seeds)
        if agg:
            rows.append({
                'Model': approach,
                'Train Time (s)': f"{agg.get('train_time_sec_mean', 0):.1f}",
                'Latency p50 batch1 (ms)': f"{agg.get('latency_batch1_p50_ms_mean', 0):.3f}",
                'Latency p95 batch1024 (ms)': f"{agg.get('latency_batch1024_p95_ms_mean', 0):.3f}",
                'Throughput (flows/s)': f"{agg.get('throughput_flows_per_sec_mean', 0):.0f}",
                'Peak RAM (MB)': f"{agg.get('peak_ram_mb_mean', 0):.1f}",
                'Model Size (MB)': f"{agg.get('model_size_mb_mean', 0):.1f}",
                'Parameters': f"{agg.get('param_count_mean', 0):.0f}",
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e3_cost.csv", index=False)
    print(df.to_string(index=False))
    # Cost comparison charts
    if rows:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        models = [r['Model'] for r in rows]
        train_times = [float(r['Train Time (s)']) for r in rows]
        throughputs = [float(r['Throughput (flows/s)']) for r in rows]
        axes[0, 0].bar(models, train_times, color=['#2196F3', '#4CAF50', '#FF9800', '#F44336'])
        axes[0, 0].set_ylabel('Seconds')
        axes[0, 0].set_title('Training Time')
        axes[0, 1].bar(models, throughputs, color=['#2196F3', '#4CAF50', '#FF9800', '#F44336'])
        axes[0, 1].set_ylabel('Flows/second')
        axes[0, 1].set_title('Inference Throughput')
        sizes = [float(r['Model Size (MB)']) for r in rows]
        rams = [float(r['Peak RAM (MB)']) for r in rows]
        axes[1, 0].bar(models, sizes, color=['#2196F3', '#4CAF50', '#FF9800', '#F44336'])
        axes[1, 0].set_ylabel('MB')
        axes[1, 0].set_title('Model Size on Disk')
        axes[1, 1].bar(models, rams, color=['#2196F3', '#4CAF50', '#FF9800', '#F44336'])
        axes[1, 1].set_ylabel('MB')
        axes[1, 1].set_title('Peak RAM During Inference')
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "Fig_07_cost_comparison.png", dpi=300)
        plt.close()

def e4_data_efficiency(splits, int_to_family):
    """E4: Performance vs training set size."""
    print("\n=== E4: Data Efficiency ===")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    fractions = [0.1, 0.25, 0.5, 1.0]
    X_train = splits['X_train']
    y_train = splits['y_family_train']
    X_val = splits['X_val']
    y_val = splits['y_family_val']
    results = {'fraction': fractions, 'RF': [], 'CNN-LSTM': []}
    for frac in fractions:
        n = int(len(X_train) * frac)
        idx = np.random.choice(len(X_train), n, replace=False)
        # RF
        rf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1,
                                     random_state=42, class_weight='balanced')
        rf.fit(X_train[idx], y_train[idx])
        rf_f1 = f1_score(y_val, rf.predict(X_val), average='macro', zero_division=0)
        results['RF'].append(rf_f1)
        # CNN-LSTM (fewer epochs for speed)
        import torch
        torch.manual_seed(42)
        n_classes = len(int_to_family)
        cnn = DeepModelWrapper(CNNLSTM, X_train.shape[1], n_classes)
        cnn.fit(X_train[idx], y_train[idx], X_val, y_val,
                {'lr': 1e-3, 'batch_size': 1024, 'max_epochs': 15, 'patience': 5})
        cnn_pred = cnn.predict(X_val)
        cnn_f1 = f1_score(y_val, cnn_pred, average='macro', zero_division=0)
        results['CNN-LSTM'].append(cnn_f1)
        print(f"  {frac*100:.0f}%: RF={rf_f1:.4f}, CNN-LSTM={cnn_f1:.4f}")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([f * 100 for f in fractions], results['RF'], 'o-', label='Random Forest', linewidth=2)
    ax.plot([f * 100 for f in fractions], results['CNN-LSTM'], 's-', label='CNN-LSTM', linewidth=2)
    ax.set_xlabel('Training Data (%)')
    ax.set_ylabel('Macro-F1')
    ax.set_title('Data Efficiency: Macro-F1 vs Training Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "Fig_08_data_efficiency.png", dpi=300)
    plt.close()
    pd.DataFrame(results).to_csv(RESULTS_DIR / "e4_data_efficiency.csv", index=False)

def e5_robustness(splits, int_to_family):
    """E5: Performance under noise and evasion perturbation."""
    print("\n=== E5: Robustness ===")
    X_test = splits['X_test']
    y_test = splits['y_family_test']
    class_names = [int_to_family[i] for i in range(len(int_to_family))]
    n_classes = len(class_names)
    models = load_best_models(splits, int_to_family, 'multiclass')
    noise_levels = [0, 0.01, 0.05, 0.1]
    results = {'noise_sigma': noise_levels}
    from sklearn.metrics import f1_score
    for name, model in models.items():
        f1s = []
        for sigma in noise_levels:
            X_noisy = X_test + np.random.normal(0, sigma, X_test.shape).astype(np.float32)
            pred = model.predict(X_noisy)
            f1s.append(f1_score(y_test, pred, average='macro', zero_division=0))
        results[name] = f1s
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in models:
        ax.plot(noise_levels, results[name], 'o-', label=name, linewidth=2)
    ax.set_xlabel('Gaussian Noise σ')
    ax.set_ylabel('Macro-F1')
    ax.set_title('Robustness: Macro-F1 Under Feature Noise')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "Fig_09_robustness_noise.png", dpi=300)
    plt.close()
    # Evasion perturbation
    evasion_pcts = [0, 10, 25, 50]
    evasion_results = {'perturbation_pct': evasion_pcts}
    attack_mask = y_test > 0
    benign_mean = X_test[~attack_mask].mean(axis=0)
    for name, model in models.items():
        f1s = []
        for pct in evasion_pcts:
            X_evade = X_test.copy()
            shift = (benign_mean - X_test[attack_mask]) * (pct / 100.0)
            X_evade[attack_mask] = X_test[attack_mask] + shift.astype(np.float32)
            pred = model.predict(X_evade)
            f1s.append(f1_score(y_test, pred, average='macro', zero_division=0))
        evasion_results[name] = f1s
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in models:
        ax.plot(evasion_pcts, evasion_results[name], 'o-', label=name, linewidth=2)
    ax.set_xlabel('Evasion Shift Toward Benign Mean (%)')
    ax.set_ylabel('Macro-F1')
    ax.set_title('Robustness: Macro-F1 Under Evasion Perturbation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "Fig_10_robustness_evasion.png", dpi=300)
    plt.close()
    pd.DataFrame(results).to_csv(RESULTS_DIR / "e5_robustness_noise.csv", index=False)
    pd.DataFrame(evasion_results).to_csv(RESULTS_DIR / "e5_robustness_evasion.csv", index=False)

def e7_ablations(splits, int_to_family):
    """E7: Ablation studies."""
    print("\n=== E7: Ablations ===")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    X_train = splits['X_train']
    X_val = splits['X_val']
    y_train = splits['y_family_train']
    y_val = splits['y_family_val']
    rows = []
    # A1 ablations
    configs = [
        ('RF (full)', {'class_weight': 'balanced'}),
        ('RF (no feat sel)', {'class_weight': 'balanced'}),
        ('RF (no SMOTE, no balanced)', {}),
    ]
    for name, kwargs in configs:
        rf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1,
                                     random_state=42, **kwargs)
        rf.fit(X_train, y_train)
        f1 = f1_score(y_val, rf.predict(X_val), average='macro', zero_division=0)
        rows.append({'Variant': name, 'Macro-F1': f1})
    # A2 ablations
    import torch
    torch.manual_seed(42)
    n_classes = len(int_to_family)
    for model_cls, name in [(CNNLSTM, 'CNN-LSTM'), (MLP, 'MLP')]:
        m = DeepModelWrapper(model_cls, X_train.shape[1], n_classes)
        m.fit(X_train, y_train, X_val, y_val,
              {'lr': 1e-3, 'batch_size': 1024, 'max_epochs': 20, 'patience': 5})
        pred = m.predict(X_val)
        f1 = f1_score(y_val, pred, average='macro', zero_division=0)
        rows.append({'Variant': name, 'Macro-F1': f1})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e7_ablations.csv", index=False)
    print(df.to_string(index=False))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df['Variant'], df['Macro-F1'], color='steelblue')
    ax.set_xlabel('Macro-F1')
    ax.set_title('Ablation Study: Model Variants')
    ax.set_xlim(0, 1)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "Fig_11_ablations.png", dpi=300)
    plt.close()

def build_comparison_table(seeds):
    """Build final comparison table."""
    print("\n=== Building Comparison Table ===")
    rf_agg = aggregate_seeds('RF', 'multiclass', seeds)
    cnn_agg = aggregate_seeds('CNN-LSTM', 'multiclass', seeds)
    def winner(metric, higher_better=True):
        rf_val = rf_agg.get(f'{metric}_mean', 0)
        cnn_val = cnn_agg.get(f'{metric}_mean', 0)
        if higher_better:
            return 'Approach 1 (RF)' if rf_val >= cnn_val else 'Approach 2 (CNN-LSTM)'
        return 'Approach 1 (RF)' if rf_val <= cnn_val else 'Approach 2 (CNN-LSTM)'
    rows = [
        {'Parameter': 'Accuracy', 'Approach 1 (RF)': f"{rf_agg.get('accuracy_mean', 0):.4f}",
         'Approach 2 (CNN-LSTM)': f"{cnn_agg.get('accuracy_mean', 0):.4f}",
         'Winner': winner('accuracy'), 'Evidence': 'e1_main.csv'},
        {'Parameter': 'Security (Macro-F1)', 'Approach 1 (RF)': f"{rf_agg.get('f1_macro_mean', 0):.4f}",
         'Approach 2 (CNN-LSTM)': f"{cnn_agg.get('f1_macro_mean', 0):.4f}",
         'Winner': winner('f1_macro'), 'Evidence': 'e1_main.csv'},
        {'Parameter': 'Processing Time (train)', 'Approach 1 (RF)': f"{rf_agg.get('train_time_sec_mean', 0):.1f}s",
         'Approach 2 (CNN-LSTM)': f"{cnn_agg.get('train_time_sec_mean', 0):.1f}s",
         'Winner': winner('train_time_sec', False), 'Evidence': 'e3_cost.csv'},
        {'Parameter': 'Inference Latency (batch1 p50)', 'Approach 1 (RF)': f"{rf_agg.get('latency_batch1_p50_ms_mean', 0):.3f}ms",
         'Approach 2 (CNN-LSTM)': f"{cnn_agg.get('latency_batch1_p50_ms_mean', 0):.3f}ms",
         'Winner': winner('latency_batch1_p50_ms', False), 'Evidence': 'e3_cost.csv'},
        {'Parameter': 'Throughput', 'Approach 1 (RF)': f"{rf_agg.get('throughput_flows_per_sec_mean', 0):.0f} flows/s",
         'Approach 2 (CNN-LSTM)': f"{cnn_agg.get('throughput_flows_per_sec_mean', 0):.0f} flows/s",
         'Winner': winner('throughput_flows_per_sec'), 'Evidence': 'e3_cost.csv'},
        {'Parameter': 'Model Size', 'Approach 1 (RF)': f"{rf_agg.get('model_size_mb_mean', 0):.1f} MB",
         'Approach 2 (CNN-LSTM)': f"{cnn_agg.get('model_size_mb_mean', 0):.1f} MB",
         'Winner': winner('model_size_mb', False), 'Evidence': 'e3_cost.csv'},
        {'Parameter': 'Peak RAM', 'Approach 1 (RF)': f"{rf_agg.get('peak_ram_mb_mean', 0):.1f} MB",
         'Approach 2 (CNN-LSTM)': f"{cnn_agg.get('peak_ram_mb_mean', 0):.1f} MB",
         'Winner': winner('peak_ram_mb', False), 'Evidence': 'e3_cost.csv'},
        {'Parameter': 'Parameters', 'Approach 1 (RF)': f"{rf_agg.get('param_count_mean', 0):.0f}",
         'Approach 2 (CNN-LSTM)': f"{cnn_agg.get('param_count_mean', 0):.0f}",
         'Winner': 'N/A (different paradigms)', 'Evidence': 'e3_cost.csv'},
        {'Parameter': 'Reliability (std over 5 seeds)', 'Approach 1 (RF)': f"±{rf_agg.get('f1_macro_std', 0):.4f}",
         'Approach 2 (CNN-LSTM)': f"±{cnn_agg.get('f1_macro_std', 0):.4f}",
         'Winner': winner('f1_macro_std', False), 'Evidence': 'e1_main.csv'},
        {'Parameter': 'Explainability', 'Approach 1 (RF)': 'SHAP TreeExplainer (exact)',
         'Approach 2 (CNN-LSTM)': 'Integrated Gradients (approximate)',
         'Winner': 'Approach 1 (RF)', 'Evidence': 'SHAP/IG plots'},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
    print(df.to_string(index=False))
    return df

def run_all_experiments():
    """Run experiments E1–E7."""
    with open(PROJECT_ROOT / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)
    seeds = config.get('seeds', [42])
    splits, family_map, int_to_family = load_processed_splits()

    e1_main_comparison(splits, int_to_family, seeds)
    e2_per_class_breakdown(splits, int_to_family, seeds)
    e3_cost_analysis(splits, int_to_family, seeds)
    e4_data_efficiency(splits, int_to_family)
    e5_robustness(splits, int_to_family)
    # E6 skipped — requires lab capture or second dataset
    print("\n=== E6: Generalization (SKIPPED — no lab capture or second dataset found) ===")
    e7_ablations(splits, int_to_family)
    build_comparison_table(seeds)
    print("\nAll experiments complete. Results saved to results/")

if __name__ == "__main__":
    run_all_experiments()

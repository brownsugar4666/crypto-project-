"""Approach 2: Deep Learning — 1D-CNN + LSTM with PyTorch."""
import os
import sys
import time
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.common.pipeline import load_processed_splits
from src.eval.harness import ModelWrapper, evaluate_model, plot_confusion_matrix, RESULTS_DIR

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNNLSTM(nn.Module):
    def __init__(self, n_features, n_classes, conv_filters=(64, 128), kernel_size=3,
                 lstm_hidden=64, dense_hidden=64, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, conv_filters[0], kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(conv_filters[0])
        self.conv2 = nn.Conv1d(conv_filters[0], conv_filters[1], kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(conv_filters[1])
        self.pool = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm = nn.LSTM(conv_filters[1], lstm_hidden, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden, dense_hidden)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_hidden, n_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class MLP(nn.Module):
    def __init__(self, n_features, n_classes, hidden_layers=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_layers:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.net(x)

class DeepModelWrapper(ModelWrapper):
    def __init__(self, model_class, n_features, n_classes, device=DEVICE, **kwargs):
        self.model = model_class(n_features, n_classes, **kwargs).to(device)
        self.device = device
        self.n_features = n_features
        self.n_classes = n_classes
        self.model_class = model_class
        self.kwargs = kwargs

    def fit(self, X_train, y_train, X_val=None, y_val=None, config=None):
        if config is None:
            config = {'lr': 1e-3, 'batch_size': 1024, 'max_epochs': 50, 'patience': 5}
        class_counts = np.bincount(y_train.astype(int))
        weights = 1.0 / (class_counts + 1)
        weights = weights / weights.sum() * len(weights)
        class_weights = torch.FloatTensor(weights).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
        X_t = torch.FloatTensor(X_train).unsqueeze(1)
        y_t = torch.LongTensor(y_train)
        train_ds = TensorDataset(X_t, y_t)
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, drop_last=True)
        if X_val is not None:
            X_v = torch.FloatTensor(X_val).unsqueeze(1).to(self.device)
            y_v = torch.LongTensor(y_val).to(self.device)

        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        best_f1 = 0
        patience_counter = 0
        best_state = None

        for epoch in range(config['max_epochs']):
            self.model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            history['train_loss'].append(avg_loss)

            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(X_v)
                    val_loss = criterion(val_out, y_v).item()
                    val_pred = val_out.argmax(dim=1).cpu().numpy()
                    from sklearn.metrics import f1_score
                    val_f1 = f1_score(y_val, val_pred, average='macro', zero_division=0)
                history['val_loss'].append(val_loss)
                history['val_f1'].append(val_f1)
                scheduler.step(val_f1)
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                if patience_counter >= config['patience']:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, val_f1={val_f1:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        self.history = history
        return history

    def predict(self, X):
        self.model.eval()
        X_t = torch.FloatTensor(X).unsqueeze(1).to(self.device)
        with torch.no_grad():
            out = self.model(X_t)
        return out.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.FloatTensor(X).unsqueeze(1).to(self.device)
        with torch.no_grad():
            out = self.model(X_t)
        return torch.softmax(out, dim=1).cpu().numpy()

    def save(self, path):
        torch.save({
            'model_state': self.model.state_dict(),
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'kwargs': self.kwargs,
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.model = self.model_class(data['n_features'], data['n_classes'], **data['kwargs']).to(self.device)
        self.model.load_state_dict(data['model_state'])

    def param_count(self):
        return sum(p.numel() for p in self.model.parameters())

def plot_learning_curves(history, save_path, title=''):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['train_loss'], label='Train Loss')
    if history['val_loss']:
        ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss Curves {title}')
    ax1.legend()
    if history['val_f1']:
        ax2.plot(history['val_f1'], label='Val Macro-F1', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Macro-F1')
        ax2.set_title(f'Validation Macro-F1 {title}')
        ax2.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()

def train_and_evaluate(task='binary', seeds=None):
    """Full training pipeline for Approach 2."""
    with open(PROJECT_ROOT / "configs" / "approach2.yaml") as f:
        a2_config = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)

    splits, family_map, int_to_family = load_processed_splits()
    if seeds is None:
        seeds = config.get('seeds', [42])

    X_train = splits['X_train']
    X_val = splits['X_val']
    X_test = splits['X_test']
    n_features = X_train.shape[1]

    if task == 'binary':
        y_train = splits['y_binary_train']
        y_val = splits['y_binary_val']
        y_test = splits['y_binary_test']
        n_classes = 2
        class_names = ['Benign', 'Attack']
    else:
        y_train = splits['y_family_train']
        y_val = splits['y_family_val']
        y_test = splits['y_family_test']
        n_classes = len(int_to_family)
        class_names = [int_to_family[i] for i in range(n_classes)]

    train_config = a2_config['training']
    all_cnn_metrics = []
    all_mlp_metrics = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # CNN-LSTM
        cnn_model = DeepModelWrapper(
            CNNLSTM, n_features, n_classes,
            conv_filters=tuple(a2_config['cnn_lstm']['conv_filters']),
            kernel_size=a2_config['cnn_lstm']['kernel_size'],
            lstm_hidden=a2_config['cnn_lstm']['lstm_hidden'],
            dense_hidden=a2_config['cnn_lstm']['dense_hidden'],
            dropout=a2_config['cnn_lstm']['dropout'],
        )
        start = time.time()
        history = cnn_model.fit(X_train, y_train, X_val, y_val, train_config)
        cnn_train_time = time.time() - start

        cnn_path = RESULTS_DIR / f"cnn_lstm_{task}_seed{seed}.pt"
        cnn_model.save(cnn_path)
        plot_learning_curves(history, RESULTS_DIR / f"Fig_learning_curve_CNN_{task}_seed{seed}.png",
                             f'CNN-LSTM {task}')

        cnn_metrics, cnn_cm, _ = evaluate_model(
            cnn_model, X_test, y_test, class_names,
            task=task, approach_name='CNN-LSTM', seed=seed,
            model_path=cnn_path, train_time=cnn_train_time
        )
        all_cnn_metrics.append(cnn_metrics)
        plot_confusion_matrix(cnn_cm, class_names,
                              f'CNN-LSTM — {task} (seed {seed})',
                              RESULTS_DIR / f"Fig_CM_CNN_{task}_seed{seed}.png")
        print(f"  CNN-LSTM: acc={cnn_metrics['accuracy']:.4f}, macro-F1={cnn_metrics['f1_macro']:.4f}")

        # MLP baseline
        mlp_model = DeepModelWrapper(
            MLP, n_features, n_classes,
            hidden_layers=tuple(a2_config['mlp']['hidden_layers']),
            dropout=a2_config['mlp']['dropout'],
        )
        start = time.time()
        mlp_history = mlp_model.fit(X_train, y_train, X_val, y_val, train_config)
        mlp_train_time = time.time() - start

        mlp_path = RESULTS_DIR / f"mlp_{task}_seed{seed}.pt"
        mlp_model.save(mlp_path)
        plot_learning_curves(mlp_history, RESULTS_DIR / f"Fig_learning_curve_MLP_{task}_seed{seed}.png",
                             f'MLP {task}')

        mlp_metrics, mlp_cm, _ = evaluate_model(
            mlp_model, X_test, y_test, class_names,
            task=task, approach_name='MLP', seed=seed,
            model_path=mlp_path, train_time=mlp_train_time
        )
        all_mlp_metrics.append(mlp_metrics)
        plot_confusion_matrix(mlp_cm, class_names,
                              f'MLP — {task} (seed {seed})',
                              RESULTS_DIR / f"Fig_CM_MLP_{task}_seed{seed}.png")
        print(f"  MLP: acc={mlp_metrics['accuracy']:.4f}, macro-F1={mlp_metrics['f1_macro']:.4f}")

    # Save tuning log
    tuning_data = [{
        'model': 'CNN-LSTM', 'task': task,
        'conv_filters': str(a2_config['cnn_lstm']['conv_filters']),
        'lstm_hidden': a2_config['cnn_lstm']['lstm_hidden'],
        'dropout': a2_config['cnn_lstm']['dropout'],
        'lr': train_config['lr'],
        'batch_size': train_config['batch_size'],
    }]
    pd.DataFrame(tuning_data).to_csv(RESULTS_DIR / f"approach2_tuning_{task}.csv", index=False)
    return all_cnn_metrics, all_mlp_metrics

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("APPROACH 2: Deep Learning — CNN-LSTM + MLP")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    print("\n--- Binary Classification ---")
    cnn_bin, mlp_bin = train_and_evaluate('binary')
    print("\n--- Multi-class Classification ---")
    cnn_mc, mlp_mc = train_and_evaluate('multiclass')
    print("\nApproach 2 training complete.")

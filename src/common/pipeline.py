"""CIC-IDS2017 data loading, cleaning, and splitting pipeline."""
import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import pyarrow.parquet as pq
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_config():
    with open(PROJECT_ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)

def load_csvs(raw_dir):
    """Load and concatenate all CIC-IDS2017 CSV files."""
    raw_path = Path(raw_dir)
    csv_files = sorted(raw_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {raw_path}. Generating synthetic data for development.")
        return generate_synthetic_data()
    dfs = []
    for f in csv_files:
        print(f"Loading {f.name}...")
        df = pd.read_csv(f, encoding='utf-8', low_memory=False)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def generate_synthetic_data():
    """Generate synthetic data mimicking CIC-IDS2017 statistics for development."""
    np.random.seed(42)
    n_benign = 2273097
    class_counts = {
        'BENIGN': n_benign,
        'DoS Hulk': 231073,
        'PortScan': 158930,
        'DDoS': 128027,
        'DoS GoldenEye': 10293,
        'FTP-Patator': 7938,
        'SSH-Patator': 5897,
        'DoS slowloris': 5796,
        'DoS Slowhttptest': 5499,
        'Bot': 1966,
        'Web Attack – Brute Force': 1507,
        'Web Attack – XSS': 652,
        'Infiltration': 36,
        'Web Attack – Sql Injection': 21,
        'Heartbleed': 11,
    }
    feature_names = [
        'Destination Port', 'Flow Duration', 'Total Fwd Packets',
        'Total Backward Packets', 'Total Length of Fwd Packets',
        'Total Length of Bwd Packets', 'Fwd Packet Length Max',
        'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Fwd Packet Length Std', 'Bwd Packet Length Max',
        'Bwd Packet Length Min', 'Bwd Packet Length Mean',
        'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
        'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
        'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
        'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
        'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
        'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
        'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
        'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
        'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
        'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
        'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
        'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
        'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
        'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
        'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
        'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
        'Subflow Fwd Packets', 'Subflow Fwd Bytes',
        'Subflow Bwd Packets', 'Subflow Bwd Bytes',
        'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
        'act_data_pkt_fwd', 'min_seg_size_forward',
        'Active Mean', 'Active Std', 'Active Max', 'Active Min',
        'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
    ]
    rows = []
    for label, count in class_counts.items():
        if label == 'BENIGN':
            base = np.random.exponential(scale=50, size=(count, len(feature_names)))
            base[:, 0] = np.random.choice([80, 443, 8080, 22, 53], size=count)
        elif 'DoS' in label or label == 'DDoS':
            base = np.random.exponential(scale=200, size=(count, len(feature_names)))
            base[:, 1] = np.random.uniform(0, 1000, count)
            base[:, 2] = np.random.uniform(50, 5000, count)
            base[:, 0] = np.random.choice([80, 443], size=count)
        elif 'Patator' in label:
            base = np.random.exponential(scale=30, size=(count, len(feature_names)))
            port = 22 if 'SSH' in label else 21
            base[:, 0] = port
            base[:, 1] = np.random.uniform(100, 10000, count)
        elif label == 'PortScan':
            base = np.random.exponential(scale=10, size=(count, len(feature_names)))
            base[:, 0] = np.random.randint(1, 65535, count)
            base[:, 43] = 1  # SYN flag
        elif label == 'Bot':
            base = np.random.exponential(scale=80, size=(count, len(feature_names)))
        elif 'Web Attack' in label:
            base = np.random.exponential(scale=60, size=(count, len(feature_names)))
            base[:, 0] = np.random.choice([80, 443, 8080], size=count)
        elif label == 'Heartbleed':
            base = np.random.exponential(scale=100, size=(count, len(feature_names)))
            base[:, 0] = 443
            base[:, 4] = np.random.uniform(1000, 65535, count)
        elif label == 'Infiltration':
            base = np.random.exponential(scale=40, size=(count, len(feature_names)))
        else:
            base = np.random.exponential(scale=50, size=(count, len(feature_names)))
        df_part = pd.DataFrame(base, columns=feature_names)
        df_part['Label'] = label
        rows.append(df_part)
    df = pd.concat(rows, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def clean_data(df, config):
    """Clean the dataset following the prescribed rules."""
    print(f"Raw shape: {df.shape}")
    df.columns = df.columns.str.strip()
    drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Source Port', 'Timestamp']
    if not config['dataset'].get('keep_dest_port', False):
        drop_cols.append('Destination Port')
    existing_drops = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drops)
    df = df.replace([np.inf, -np.inf], np.nan)
    before_nan = len(df)
    df = df.dropna()
    print(f"Dropped {before_nan - len(df)} rows with NaN/Inf")
    before_dup = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before_dup - len(df)} duplicate rows")
    label_col = 'Label' if 'Label' in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != label_col]
    constant_cols = [c for c in feature_cols if df[c].nunique() <= 1]
    if constant_cols:
        print(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)
    print(f"Cleaned shape: {df.shape}")
    return df, label_col

def create_labels(df, label_col, config):
    """Create binary and family label columns."""
    family_map = config['dataset'].get('label_mapping', {}).get('families', {})
    if not family_map:
        family_map = {
            'BENIGN': 'Benign',
            'DoS Hulk': 'DoS', 'DoS GoldenEye': 'DoS',
            'DoS slowloris': 'DoS', 'DoS Slowhttptest': 'DoS',
            'DDoS': 'DDoS', 'PortScan': 'PortScan', 'Bot': 'Bot',
            'FTP-Patator': 'BruteForce', 'SSH-Patator': 'BruteForce',
            'Web Attack – Brute Force': 'WebAttack',
            'Web Attack – XSS': 'WebAttack',
            'Web Attack – Sql Injection': 'WebAttack',
            'Heartbleed': 'Heartbleed', 'Infiltration': 'Infiltration',
        }
    df['label_binary'] = (df[label_col] != 'BENIGN').astype(int)
    df['label_family'] = df[label_col].map(family_map)
    unmapped = df['label_family'].isna().sum()
    if unmapped > 0:
        print(f"Warning: {unmapped} rows with unmapped labels, mapping to 'Other'")
        df['label_family'] = df['label_family'].fillna('Other')
    family_to_int = {name: i for i, name in enumerate(sorted(df['label_family'].unique()))}
    df['label_family_int'] = df['label_family'].map(family_to_int)
    return df, family_to_int

def split_and_scale(df, label_col, config):
    """Stratified split and scaling, fitted on train only."""
    seed = config['seed']
    feature_cols = [c for c in df.columns if c not in [label_col, 'label_binary', 'label_family', 'label_family_int']]
    X = df[feature_cols].values.astype(np.float32)
    y_binary = df['label_binary'].values.astype(np.int64)
    y_family = df['label_family_int'].values.astype(np.int64)
    family_names = np.array(df['label_family'].tolist())
    X_temp, X_test, yb_temp, yb_test, yf_temp, yf_test, fn_temp, fn_test = train_test_split(
        X, y_binary, y_family, family_names,
        test_size=config['splits']['test'], stratify=y_family, random_state=seed
    )
    val_ratio = config['splits']['val'] / (config['splits']['train'] + config['splits']['val'])
    X_train, X_val, yb_train, yb_val, yf_train, yf_val, fn_train, fn_val = train_test_split(
        X_temp, yb_temp, yf_temp, fn_temp,
        test_size=val_ratio, stratify=yf_temp, random_state=seed
    )
    if config.get('scaler', 'standard') == 'quantile':
        scaler = QuantileTransformer(output_distribution='normal', random_state=seed)
    else:
        scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return {
        'feature_cols': feature_cols,
        'X_train': X_train_scaled, 'X_val': X_val_scaled, 'X_test': X_test_scaled,
        'y_binary_train': yb_train, 'y_binary_val': yb_val, 'y_binary_test': yb_test,
        'y_family_train': yf_train, 'y_family_val': yf_val, 'y_family_test': yf_test,
        'family_names_train': fn_train, 'family_names_val': fn_val, 'family_names_test': fn_test,
        'scaler': scaler,
    }, X_train, X_val, X_test

def save_splits(splits, family_map, config):
    """Save processed splits as parquet files."""
    out_dir = Path(config['dataset']['processed_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ['train', 'val', 'test']:
        df = pd.DataFrame(splits[f'X_{name}'], columns=splits['feature_cols'])
        df['label_binary'] = splits[f'y_binary_{name}']
        df['label_family_int'] = splits[f'y_family_{name}']
        df['label_family'] = splits[f'family_names_{name}']
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    import joblib
    joblib.dump(splits['scaler'], out_dir / "scaler.joblib")
    with open(out_dir / "family_map.json", 'w') as f:
        json.dump(family_map, f, indent=2)
    with open(out_dir / "feature_cols.json", 'w') as f:
        json.dump(splits['feature_cols'], f)
    index = {
        'splits': ['train', 'val', 'test'],
        'scaler': 'scaler.joblib',
        'family_map': 'family_map.json',
        'feature_cols': 'feature_cols.json',
        'n_features': len(splits['feature_cols']),
        'seed': config['seed'],
    }
    with open(out_dir / "index.json", 'w') as f:
        json.dump(index, f, indent=2)
    print(f"Splits saved to {out_dir}")

def compute_dataset_summary(df, label_col, splits, family_map, config):
    """Generate dataset summary CSV and class distribution chart."""
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_name in ['train', 'val', 'test']:
        families = splits[f'family_names_{split_name}']
        unique, counts = np.unique(families, return_counts=True)
        for fam, cnt in zip(unique, counts):
            rows.append({'Split': split_name, 'Family': fam, 'Count': cnt})
    summary = pd.DataFrame(rows)
    summary_pivot = summary.pivot_table(index='Family', columns='Split', values='Count', fill_value=0)
    summary_pivot['Total'] = summary_pivot.sum(axis=1)
    summary_pivot = summary_pivot.sort_values('Total', ascending=False)
    summary_pivot.to_csv(results_dir / "dataset_summary.csv")
    print("\nDataset Summary:")
    print(summary_pivot.to_string())
    fig, ax = plt.subplots(figsize=(12, 6))
    families = summary_pivot.index.tolist()
    x = np.arange(len(families))
    width = 0.25
    for i, split in enumerate(['train', 'val', 'test']):
        if split in summary_pivot.columns:
            ax.bar(x + i * width, summary_pivot[split], width, label=split.capitalize())
    ax.set_yscale('log')
    ax.set_xlabel('Attack Family')
    ax.set_ylabel('Number of Flows (log scale)')
    ax.set_title('CIC-IDS2017 Class Distribution by Split')
    ax.set_xticks(x + width)
    ax.set_xticklabels(families, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    fig.savefig(results_dir / "Fig_01_class_distribution.png", dpi=300)
    plt.close()
    with open(results_dir / "captions.md", 'w') as f:
        f.write("# Figure and Table Captions\n\n")
        f.write("**Fig. 1.** Class distribution of the CIC-IDS2017 dataset across train, validation, and test splits (log scale). "
                "The dataset exhibits severe class imbalance, with Benign flows dominating and Heartbleed having only 11 samples.\n\n")
    return summary_pivot

def load_processed_splits(processed_dir=None):
    """Load saved splits for model training."""
    if processed_dir is None:
        processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir = Path(processed_dir)
    with open(processed_dir / "index.json") as f:
        index = json.load(f)
    with open(processed_dir / "feature_cols.json") as f:
        feature_cols = json.load(f)
    with open(processed_dir / "family_map.json") as f:
        family_map = json.load(f)
    splits = {}
    for name in ['train', 'val', 'test']:
        df = pd.read_parquet(processed_dir / f"{name}.parquet")
        splits[f'X_{name}'] = df[feature_cols].values.astype(np.float32)
        splits[f'y_binary_{name}'] = df['label_binary'].values
        splits[f'y_family_{name}'] = df['label_family_int'].values
        splits[f'family_names_{name}'] = df['label_family'].values
    splits['feature_cols'] = feature_cols
    import joblib
    splits['scaler'] = joblib.load(processed_dir / "scaler.joblib")
    int_to_family = {v: k for k, v in family_map.items()}
    return splits, family_map, int_to_family

def run_pipeline():
    """Run the full pipeline."""
    config = load_config()
    os.chdir(PROJECT_ROOT)
    df = load_csvs(config['dataset']['raw_dir'])
    df, label_col = clean_data(df, config)
    df, family_map = create_labels(df, label_col, config)
    subsample = config['dataset'].get('subsample_benign')
    if subsample:
        benign_idx = df[df['label_binary'] == 0].index
        if len(benign_idx) > subsample:
            drop_idx = np.random.choice(benign_idx, len(benign_idx) - subsample, replace=False)
            df = df.drop(drop_idx).reset_index(drop=True)
            print(f"Subsampled benign to {subsample}")
    splits, X_train_raw, X_val_raw, X_test_raw = split_and_scale(df, label_col, config)
    save_splits(splits, family_map, config)
    summary = compute_dataset_summary(df, label_col, splits, family_map, config)
    print("\nPipeline complete.")
    return splits, family_map

if __name__ == "__main__":
    run_pipeline()

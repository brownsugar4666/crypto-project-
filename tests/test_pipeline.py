"""Tests for the data pipeline."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

def test_no_split_overlap():
    """Verify no sample overlap between train/val/test."""
    from src.common.pipeline import load_processed_splits
    try:
        splits, _, _ = load_processed_splits()
    except FileNotFoundError:
        pytest.skip("Processed data not available")
    n_train = len(splits['X_train'])
    n_val = len(splits['X_val'])
    n_test = len(splits['X_test'])
    total = n_train + n_val + n_test
    assert abs(n_train / total - 0.70) < 0.02
    assert abs(n_val / total - 0.15) < 0.02
    assert abs(n_test / total - 0.15) < 0.02

def test_labels_mapped():
    """Every original label should be mapped to a family."""
    from src.common.pipeline import load_processed_splits
    try:
        splits, family_map, int_to_family = load_processed_splits()
    except FileNotFoundError:
        pytest.skip("Processed data not available")
    for split in ['train', 'val', 'test']:
        families = splits[f'family_names_{split}']
        assert all(f in int_to_family.values() for f in np.unique(families))

def test_scaler_train_only():
    """Scaler should have been fitted on training data dimensions."""
    from src.common.pipeline import load_processed_splits
    try:
        splits, _, _ = load_processed_splits()
    except FileNotFoundError:
        pytest.skip("Processed data not available")
    scaler = splits['scaler']
    assert scaler.n_features_in_ == splits['X_train'].shape[1]

def test_binary_labels():
    """Binary labels should be 0 and 1 only."""
    from src.common.pipeline import load_processed_splits
    try:
        splits, _, _ = load_processed_splits()
    except FileNotFoundError:
        pytest.skip("Processed data not available")
    for split in ['train', 'val', 'test']:
        labels = splits[f'y_binary_{split}']
        assert set(np.unique(labels)).issubset({0, 1})

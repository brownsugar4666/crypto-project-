"""Streamlit demo for AI-Based Attack Detection."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import json
import time

st.set_page_config(page_title="AI Attack Detection Demo", layout="wide")
st.title("AI-Based Cryptographic & Network Attack Detection")
st.markdown("**Comparing Classical ML (Random Forest) vs Deep Learning (CNN-LSTM)**")

@st.cache_resource
def load_models():
    models = {}
    from src.common.pipeline import load_processed_splits
    splits, family_map, int_to_family = load_processed_splits()
    results_dir = PROJECT_ROOT / "results"
    rf_path = results_dir / "rf_multiclass_seed42.joblib"
    if rf_path.exists():
        from src.approach1_classical.train import RFWrapper
        rf = RFWrapper(n_estimators=300, random_state=42)
        rf.load(rf_path)
        models['Random Forest'] = rf
    cnn_path = results_dir / "cnn_lstm_multiclass_seed42.pt"
    if cnn_path.exists():
        from src.approach2_deep.train import DeepModelWrapper, CNNLSTM
        n_features = splits['X_test'].shape[1]
        n_classes = len(int_to_family)
        cnn = DeepModelWrapper(CNNLSTM, n_features, n_classes)
        cnn.load(cnn_path)
        models['CNN-LSTM'] = cnn
    return models, splits, family_map, int_to_family

try:
    models, splits, family_map, int_to_family = load_models()
except Exception as e:
    st.error(f"Models not found. Run training first: {e}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Single Flow", "Batch Analysis", "Upload CSV"])

with tab1:
    st.subheader("Classify a Single Flow")
    idx = st.slider("Select test flow index", 0, len(splits['X_test']) - 1, 42)
    flow = splits['X_test'][idx:idx+1]
    true_label = int_to_family[splits['y_family_test'][idx]]
    st.write(f"**True label:** {true_label}")
    cols = st.columns(len(models))
    for i, (name, model) in enumerate(models.items()):
        with cols[i]:
            start = time.perf_counter()
            pred = model.predict(flow)
            proba = model.predict_proba(flow)
            latency = (time.perf_counter() - start) * 1000
            pred_label = int_to_family[pred[0]]
            confidence = proba[0].max() * 100
            correct = pred_label == true_label
            st.metric(name, pred_label, f"{confidence:.1f}% confidence")
            st.write(f"{'✅' if correct else '❌'} Latency: {latency:.2f} ms")
            proba_df = pd.DataFrame({
                'Class': [int_to_family[j] for j in range(len(proba[0]))],
                'Probability': proba[0]
            }).sort_values('Probability', ascending=False)
            st.bar_chart(proba_df.set_index('Class')['Probability'])

with tab2:
    st.subheader("Batch Analysis")
    n_flows = st.slider("Number of test flows", 100, 5000, 1000)
    if st.button("Run Batch"):
        batch = splits['X_test'][:n_flows]
        y_true = splits['y_family_test'][:n_flows]
        for name, model in models.items():
            start = time.perf_counter()
            preds = model.predict(batch)
            elapsed = (time.perf_counter() - start) * 1000
            from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
            acc = accuracy_score(y_true, preds)
            f1 = f1_score(y_true, preds, average='macro', zero_division=0)
            st.write(f"**{name}:** Accuracy={acc:.4f}, Macro-F1={f1:.4f}, Time={elapsed:.1f}ms ({n_flows/elapsed*1000:.0f} flows/s)")

with tab3:
    st.subheader("Upload CICFlowMeter CSV")
    uploaded = st.file_uploader("Upload a CSV with CICFlowMeter features", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        df.columns = df.columns.str.strip()
        st.write(f"Loaded {len(df)} flows with {len(df.columns)} columns")
        feature_cols = splits['feature_cols']
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            st.warning(f"Missing features: {missing[:5]}...")
        else:
            import joblib
            scaler = joblib.load(PROJECT_ROOT / "data" / "processed" / "scaler.joblib")
            X = scaler.transform(df[feature_cols].values.astype(np.float32))
            for name, model in models.items():
                preds = model.predict(X)
                pred_labels = [int_to_family[p] for p in preds]
                st.write(f"**{name}** predictions:")
                st.write(pd.Series(pred_labels).value_counts())

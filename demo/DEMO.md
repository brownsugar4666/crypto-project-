# Demo — AI-Based Attack Detection

## Prerequisites
- Python 3.11+ with all requirements installed (`pip install -r requirements.txt`)
- Trained models (run `./run_all.sh` or train individually)

## Running the Demo
```bash
cd "PROJECT CRYPTOOO"
streamlit run demo/app.py
```
Open http://localhost:8501 in your browser.

## Features
1. **Single Flow Classification** — Select any test flow and see predictions from both RF and CNN-LSTM side by side with confidence scores and latency
2. **Batch Analysis** — Run both models on N test flows, comparing accuracy, macro-F1, and throughput
3. **Upload CSV** — Upload your own CICFlowMeter output CSV for classification

## Demo Walkthrough for Viva
1. Show the class distribution chart (Fig. 1 in results/)
2. Classify a BruteForce flow — point out that RF gives higher confidence
3. Classify a Heartbleed flow — demonstrate the rare-class detection challenge
4. Run batch analysis with 1000 flows — compare speed and accuracy
5. Show the SHAP explanations for why RF flagged a particular flow

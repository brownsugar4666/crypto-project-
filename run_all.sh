#!/bin/bash
set -e

echo "============================================"
echo "AI-Based Attack Detection — Full Pipeline"
echo "============================================"

export PYTHONPATH="$(pwd)"

echo ""
echo "[1/5] Data Pipeline..."
python3 -c "from src.common.pipeline import run_pipeline; run_pipeline()"

echo ""
echo "[2/5] Approach 1: Classical ML..."
python3 -c "
from src.approach1_classical.train import train_and_evaluate
train_and_evaluate('binary', seeds=[42, 123, 456, 789, 1024])
train_and_evaluate('multiclass', seeds=[42, 123, 456, 789, 1024])
"

echo ""
echo "[3/5] Approach 2: Deep Learning..."
python3 -c "
from src.approach2_deep.train import train_and_evaluate
train_and_evaluate('binary', seeds=[42, 123, 456, 789, 1024])
train_and_evaluate('multiclass', seeds=[42, 123, 456, 789, 1024])
"

echo ""
echo "[4/5] Experiments E1-E7..."
python3 -c "from src.eval.experiments import run_all_experiments; run_all_experiments()"

echo ""
echo "[5/5] Done! Results saved to results/"
echo "Run 'streamlit run demo/app.py' for the demo."

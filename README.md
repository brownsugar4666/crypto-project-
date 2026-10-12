# AI-Based Cryptographic and Network Attack Detection
## Classical ML vs Deep Learning on CIC-IDS2017

**Topic 100** — Cryptography & Network Security Course Project

### Team
| Sub-group | Members | Approach |
|-----------|---------|----------|
| A (Main) | Aahil Shaik, Ram Sai, Rehan Mohammed | Classical ML (Random Forest + XGBoost) |
| B (Sub) | Ayyappa Reddy, Ramidi Hemanth Reddy | Deep Learning (1D-CNN + LSTM) |

### Quick Start
```bash
pip install -r requirements.txt
# Place CIC-IDS2017 CSVs in data/raw/
python -m src.common.pipeline          # Clean + split
python -m src.approach1_classical.train # Train Approach 1
python -m src.approach2_deep.train      # Train Approach 2
python -m src.eval.experiments          # Run E1-E7
streamlit run demo/app.py               # Launch demo
```

### Structure
```
data/raw/            # CIC-IDS2017 CSVs (gitignored)
data/processed/      # Cleaned splits (parquet)
src/common/          # Shared pipeline, metrics, utilities
src/approach1_classical/  # RF + XGBoost
src/approach2_deep/       # CNN-LSTM + MLP
src/eval/            # Evaluation harness + experiments
results/             # All CSVs, figures, metrics
demo/                # Streamlit demo app
report/              # Academic report (21 sections)
paper/               # IEEE research paper
slides/              # Presentation + viva prep
configs/             # YAML configurations
tests/               # pytest tests
```

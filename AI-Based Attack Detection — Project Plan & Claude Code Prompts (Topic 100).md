# AI-Based Attack Detection — Project Plan & Claude Code Prompts (Topic 100)

Oct 12, 2026 · @Someone

## 1. Project overview

Topic 100 builds one AI-based detector for network and cryptographic attacks and solves it two ways: **classical machine learning** (Approach 1: Random Forest / XGBoost on engineered flow features) and **deep learning** (Approach 2: a 1D-CNN + LSTM network on the same flows), compared on identical data and measured numbers.

**Problem statement.** Most modern traffic is encrypted, so payload inspection is no longer possible, and attackers increasingly target the cryptographic layer itself (TLS implementation bugs such as Heartbleed, brute-forcing of SSH/FTP credentials) alongside classic network attacks (DoS/DDoS, port scans, botnets, web attacks). The project must classify network flows as benign or as a specific attack type using only flow-level metadata (packet sizes, timings, flag counts, durations), without decrypting traffic.

**Attack classes covered** (from the CIC-IDS2017 dataset): cryptographic-layer attacks (Heartbleed, SSH-Patator, FTP-Patator brute force), DoS (Hulk, GoldenEye, Slowloris, SlowHTTPTest), DDoS, PortScan, Bot, Web attacks (brute force, XSS, SQL injection), Infiltration.

**Objectives.**

1. Build a clean, reproducible preprocessing pipeline over a public intrusion dataset.
2. Implement Approach 1: classical ML with feature selection and class-imbalance handling.
3. Implement Approach 2: a deep learning model that learns its own feature representation.
4. Evaluate both on the same test split for accuracy, per-attack detection, cost, robustness and generalization.
5. Decide which approach is better, for which attack classes and deployment conditions, with technical justification.

**Why these qualify as two approaches (Guideline 5).** They differ in learning methodology: Approach 1 relies on hand-engineered features and decision-tree ensembles; Approach 2 learns hierarchical representations with neural networks trained by gradient descent. The suggested split in the topic list is exactly classical ML vs deep learning.

**Dates.** The guideline's viva window (28 Sep to 1 Oct 2026) has already passed. Confirm your new date with the faculty; Section 10 gives a compressed schedule.

## 2. Team structure and deliverables

The whole team builds the shared data pipeline and evaluation harness in the first sessions. After that, each sub-group owns one model.

| Who | Members | Owns | Mandatory deliverables |
| --- | --- | --- | --- |
| Whole team (5) | All five members | Dataset download, cleaning, splits, shared evaluation harness, comparative analysis, final verdict | One shared academic report (21 sections), presentation |
| Main team: Sub-group A (3 members) | Aahil Shaik, Ram Sai, Rehan Mohammed | Approach 1: feature selection, imbalance handling, Random Forest + XGBoost, SHAP explanations | Working implementation, report sections 8/12/14, viva, **draft research paper (mandatory: 3-member sub-group)** |
| Sub team: Sub-group B (2 members) | Ayyappa Reddy, Ramidi Hemanth Reddy | Approach 2: 1D-CNN + LSTM in PyTorch, training loop, hyperparameter tuning, GPU/CPU timing | Working implementation, report sections 9/12/14, viva |

The main team has 3 members, so it must produce the **draft research paper**, covering both approaches and the comparison. The sub team should still contribute its Approach 2 results and figures to the paper. Each sub-group demos its code to the other before results are frozen, because viva questions can cover either approach. Record individual contributions in an appendix table.

## 3. Dataset and shared pipeline

Use CIC-IDS2017 as the main dataset because it is public, labelled per flow, and includes cryptographic-layer attacks (Heartbleed, SSH/FTP brute force) alongside network attacks. Both approaches then train and test on exactly the same splits.

&#91;embedded content: Pipeline · one cleaned split feeding both approaches\]

The dataset and any lab capture go through one cleaning and split step. Both models consume the identical splits, and one harness scores them.

**Datasets.**

- **Main:** CIC-IDS2017, the "MachineLearningCSV" files (about 2.8 million flows, roughly 80 CICFlowMeter features), from the Canadian Institute for Cybersecurity at UNB. Download it manually if the Claude Code environment cannot reach the site. If you can find the corrected re-labelled version published by Engelen et al. (2021), use that instead and say so in the report. It fixes known labelling errors.
- **Generalization test (optional):** UNSW-NB15 or a CSE-CIC-IDS2018 subset. These have different feature sets, so restrict the test to features shared by both datasets.
- **Own lab capture (optional, strong for viva):** in your own isolated VMs only, capture an nmap scan and a hydra SSH brute force with tcpdump, convert the capture to flows with the Python `cicflowmeter` package, and run both models on it. Never scan or attack machines you do not own.

**Cleaning and split rules (state all of them in the Methodology section).**

1. Strip whitespace from column names. Drop rows with NaN or infinite values. Drop duplicate rows and constant columns.
2. Drop identifier columns (Flow ID, Source/Destination IP, timestamp). Otherwise the models memorise hosts instead of learning behavior.
3. Merge tiny classes into families for the multi-class task, e.g. Web attacks, DoS, Brute force (SSH/FTP). Keep Heartbleed and Infiltration as named rare classes, and report honestly that they have very few samples.
4. Use a stratified 70/15/15 train/validation/test split with a fixed seed. Fit the scaler on train only. Use the same split files for both approaches.
5. Handle class imbalance on the training set only, with class weights or SMOTE. Never resample validation or test data.
6. Run two tasks: binary (benign vs attack) and multi-class (attack family).

## 4. Approach 1 — Classical ML (Sub-group A)

Approach 1 trains tree ensembles on the engineered CICFlowMeter features: Random Forest is the primary model and XGBoost is a variant within the same approach. Feature importance and SHAP values make every prediction explainable.

**Pipeline.**

1. **Feature selection:** remove highly correlated features (|r| > 0.95), then rank by Random Forest importance or mutual information and keep the top k. Tune k on validation (e.g. 20, 30, 40, all). Report which features matter for each attack family. For example, flag counts for scans and flow duration and packet-rate features for slow DoS.
2. **Imbalance:** use class\_weight='balanced', or SMOTE on train only. Compare both on validation.
3. **Models:** Random Forest (n\_estimators, max\_depth, min\_samples\_leaf tuned with randomized search on validation) and XGBoost (learning rate, depth, n\_estimators, with early stopping on validation).
4. **Explainability:** compute global feature importance and SHAP summary plots, plus one worked SHAP example per attack family for the viva.

**Strengths to verify experimentally:** strong accuracy on tabular flow features, fast training on CPU, small model, low inference latency, and explainable predictions.

**Weaknesses to verify:** depends on manual feature engineering, may generalize poorly to unseen datasets, and has limited ability to model temporal patterns across flows. Measure these instead of assuming them.

## 5. Approach 2 — Deep learning (Sub-group B)

Approach 2 feeds the scaled flow features into a neural network that learns its own representation: a 1D-CNN extracts local feature patterns and an LSTM layer models dependencies across them. A plain MLP serves as a baseline within the same approach.

**Architecture (starting point, tune on validation).**

1. Input: all cleaned features after scaling, with no manual selection. Learning representations is the point of this approach.
2. Conv1D(64, kernel 3) → BatchNorm → ReLU → Conv1D(128, kernel 3) → MaxPool → Dropout(0.3).
3. LSTM(64) → Dense(64) → Dropout(0.3) → output layer (sigmoid for binary, softmax for multi-class).
4. Loss: weighted cross-entropy or focal loss for imbalance. Optimizer: Adam (lr 1e-3) with a ReduceLROnPlateau schedule and early stopping on validation macro-F1.
5. Hyperparameters to sweep: filters, LSTM units, dropout, batch size, learning rate. Record every run in a CSV.

**Optional extension:** sequence mode, which groups consecutive flows from the same source into windows of N flows and classifies the window. The LSTM can then learn cross-flow behavior such as slow brute force. Report it separately if you build it.

**Explainability:** integrated gradients or SHAP DeepExplainer on a sample, so the comparison with Approach 1 is fair.

**Strengths to verify:** no manual feature engineering, may capture complex non-linear patterns, and may generalize better.

**Weaknesses to verify:** longer training, needs a GPU for speed (Google Colab is fine), larger model, higher inference latency on CPU, sensitive to hyperparameters, and harder to explain.

## 6. Comparison parameters and experiments

Both approaches run through one evaluation harness on the identical test split. The verdict comes from the measured numbers. Accuracy alone is misleading here because about 80% of flows are benign, so lead with macro-F1 and per-class recall.

| Guideline parameter | Metric to report | How it is measured |
| --- | --- | --- |
| Accuracy | Accuracy, precision, recall, macro-F1, weighted-F1, ROC-AUC, PR-AUC | Test split, binary and multi-class |
| Security | Per-attack-family recall (especially Heartbleed, brute force, Infiltration), false-negative rate, false-positive rate | Confusion matrix, per-class table |
| Processing time | Training time; inference latency per flow (p50/p95) at batch 1 and batch 1024 | Timers, CPU and GPU reported separately |
| Resource utilization | Model size on disk, peak RAM during training and inference | File size, tracemalloc / psutil |
| Scalability | Throughput (flows/s) at increasing batch sizes; training time vs training-set size (10%, 25%, 50%, 100%) | Scaling runs |
| Reliability | Mean ± std over 5 seeds | Repeated runs |
| Computational complexity | Big-O for training and inference; parameter count | Analytical + counted |
| Robustness | Macro-F1 under Gaussian feature noise and under a simple evasion perturbation of attack features | Perturbed test copies |
| Generalization | Train on CIC-IDS2017, test on a second dataset or your own lab capture | Cross-dataset run |
| Usability / implementation complexity | Explainability quality, lines of code, tuning effort, hardware needed | Described + counted |

**Experiments.**

1. **E1 Main comparison:** binary and multi-class, both models, tuned on validation.
2. **E2 Per-attack breakdown:** confusion matrices and per-class recall, focusing on cryptographic-layer attacks.
3. **E3 Cost:** training time, inference latency, memory, model size.
4. **E4 Data efficiency:** train on 10–100% of the training set and plot macro-F1 against training size.
5. **E5 Robustness:** performance under noise and feature perturbation.
6. **E6 Generalization:** cross-dataset or lab capture.
7. **E7 Ablations:** Approach 1 with vs without feature selection and SMOTE; Approach 2 CNN-LSTM vs MLP.

**Deciding the best approach.** Report a winner for each parameter, then give an overall verdict tied to a deployment scenario, such as an edge gateway with a small CPU vs a central SOC with a GPU. Published work on CIC-IDS2017 often finds tree ensembles competitive with deep models on per-flow features, so test that honestly. If your numbers show it, a well-justified "classical ML is better here, because…" is a perfectly valid conclusion.

## 7. Report format mapped to this project

The report keeps the prescribed 21 sections in order. That order also follows the mandatory workflow: Problem → Literature → Approach 1 → Results → Approach 2 → Results → Comparison → Best approach → Justification → Conclusion.

| # | Section | Content for this project |
| --- | --- | --- |
| 1 | Title Page | Title, team, sub-groups, roll numbers, class, guide, date |
| 2 | Abstract (150–250 words) | Problem, both approaches, dataset, key measured results, verdict; count the words |
| 3 | Introduction | Encrypted traffic, rise of AI-based IDS, why crypto-layer attacks matter |
| 4 | Problem Statement | Section 1 of this plan + threat model (attack classes covered) |
| 5 | Objectives | The five objectives in Section 1 |
| 6 | Literature Review | Datasets, classical ML IDS, deep learning IDS, known pitfalls; ends with the research gap |
| 7 | Existing System | Signature IDS (Snort/Suricata) and their limits on encrypted and novel attacks |
| 8 | Proposed Approach 1 | Section 4: feature selection, RF/XGBoost, imbalance handling, flowchart |
| 9 | Proposed Approach 2 | Section 5: CNN-LSTM architecture diagram, training setup |
| 10 | Methodology | Workflow, cleaning rules, split, tuning protocol, metrics |
| 11 | System / Network Architecture | Pipeline diagram: capture → flow extraction → preprocessing → two models → evaluation / alerts |
| 12 | Implementation | Stack, module structure, key code excerpts, screenshots of demo |
| 13 | Experimental Setup | Hardware (CPU/GPU), OS, library versions, dataset sizes per class, seeds, hyperparameters |
| 14 | Results | Approach 1 results, then Approach 2 results; numbered tables and figures |
| 15 | Comparative Analysis | Side-by-side table on every parameter in Section 6, per-class chart, ROC/PR curves, cost charts |
| 16 | Security Analysis | Detection of crypto-layer attacks, evasion/adversarial risk, data poisoning, false-negative impact, privacy (no payloads used) |
| 17 | Discussion | Why results came out this way; dataset limitations (label errors, rare classes, lab traffic) |
| 18 | Conclusion | Best approach + justification |
| 19 | Future Scope | Hybrid/ensemble, online learning, adversarial training, transformers on flow sequences, real deployment |
| 20 | References | IEEE style, verified only |
| 21 | Appendix | Hyperparameter tables, full confusion matrices, contribution table |

**Throughout (Guideline 9):** number, caption and discuss every figure and table; separate what sources report from your own findings; no fabricated references; every number must trace back to a results file.

## 8. Literature review starter list

I listed these well-known sources from memory and have not looked them up today. Before citing any of them, confirm the authors, year and venue on Google Scholar or IEEE Xplore, and drop any you cannot find.

**Datasets and evaluation pitfalls**

- I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani, "Toward generating a new intrusion detection dataset and intrusion traffic characterization," ICISSP, 2018 (CIC-IDS2017)
- G. Engelen, V. Rimmer, W. Joosen, "Troubleshooting an intrusion detection dataset: the CICIDS2017 case study," IEEE Security and Privacy Workshops, 2021
- N. Moustafa, J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion detection systems," MilCIS, 2015
- M. Ring et al., "A survey of network-based intrusion detection data sets," Computers & Security, 2019
- M. Tavallaee et al., "A detailed analysis of the KDD CUP 99 data set," IEEE CISDA, 2009
- D. Arp et al., "Dos and don'ts of machine learning in computer security," USENIX Security, 2022
- R. Sommer, V. Paxson, "Outside the closed world: On using machine learning for network intrusion detection," IEEE S&P, 2010

**Classical ML**

- L. Breiman, "Random forests," Machine Learning, 2001
- T. Chen, C. Guestrin, "XGBoost: A scalable tree boosting system," ACM KDD, 2016
- S. M. Lundberg, S.-I. Lee, "A unified approach to interpreting model predictions," NeurIPS, 2017 (SHAP)

**Deep learning**

- R. Vinayakumar et al., "Deep learning approach for intelligent intrusion detection system," IEEE Access, 2019
- A. Javaid et al., "A deep learning approach for network intrusion detection system," BICT (EAI), 2016
- Y. Mirsky et al., "Kitsune: An ensemble of autoencoders for online network intrusion detection," NDSS, 2018

**Cryptographic attack context**

- Z. Durumeric et al., "The matter of Heartbleed," ACM IMC, 2014

**Tools:** F. Pedregosa et al., "Scikit-learn," JMLR, 2011; A. Paszke et al., "PyTorch," NeurIPS, 2019.

Aim for 15–25 verified references, with most of them from recent years. A defensible research gap: many papers report near-perfect accuracy on CIC-IDS2017 using random splits and accuracy alone. Fewer compare classical ML and DL under the same split while also reporting per-class recall on rare cryptographic-layer attacks, cost, robustness and cross-dataset generalization. Check that this claim matches what you actually read.

## 9. Claude Code prompts, in order

Run these one at a time in a fresh folder. Check that each step works, and commit to git after each one. Prompts 3 and 4 can run in parallel, one per sub-group. Read everything Claude Code writes, because the viva will test it.

**Prompt 0 — Project memory and skeleton**

```text
Create CLAUDE.md, README.md and the folder skeleton for: "AI-Based Cryptographic and Network Attack Detection — Classical ML vs Deep Learning" (university Cryptography & Network Security course, topic 100). Team: Main team / Sub-group A (Approach 1, classical ML; also writes the draft research paper) = Aahil Shaik, Ram Sai, Rehan Mohammed. Sub team / Sub-group B (Approach 2, deep learning) = Ayyappa Reddy, Ramidi Hemanth Reddy. Record these in CLAUDE.md and README.md. One shared data pipeline, two approaches compared on identical splits.

Rules for all future work in this repo:
- Python 3.11; pandas, numpy, scikit-learn, xgboost, imbalanced-learn, torch, shap, matplotlib, seaborn, psutil, pyarrow, pyyaml, pytest. Pin versions in requirements.txt. Code must run on CPU; use GPU automatically if available.
- Folders: data/raw (gitignored), data/processed, src/common (loading, cleaning, splits, metrics, timing), src/approach1_classical, src/approach2_deep, src/eval, results/ (CSVs + PNG figures), demo/, report/, paper/, slides/, tests/, configs/.
- Approach 1 = Random Forest (primary) + XGBoost (variant) with feature selection. Approach 2 = 1D-CNN+LSTM (primary) + MLP (variant) on all scaled features. Both MUST use the identical split files in data/processed.
- Every random process takes a seed from configs/. Scalers, feature selectors and resamplers are fitted on TRAIN only. Never resample validation/test.
- Never invent numbers, results or references. Every number in report/paper/slides must be read from results/ files.
- Save every figure as results/Fig_XX_<name>.png at 300 dpi and add its caption to results/captions.md.
```

**Prompt 1 — Data loading, cleaning and splits (whole team)**

```text
In src/common, write a pipeline for CIC-IDS2017 MachineLearningCSV files placed in data/raw/ (I will download them manually from the UNB CIC website if you cannot). Steps: load and concatenate all day CSVs; strip column-name whitespace; replace inf with NaN and drop NaN rows; drop duplicate rows and constant columns; drop identifier columns if present (Flow ID, Source IP, Destination IP, Source Port, Timestamp); keep Destination Port only if configs say so.
Create two label columns: binary (BENIGN=0, attack=1) and family (Benign, DoS [Hulk, GoldenEye, slowloris, Slowhttptest], DDoS, PortScan, Bot, BruteForce [FTP-Patator, SSH-Patator], WebAttack [all three web attacks], Heartbleed, Infiltration). Make the mapping a YAML config.
Stratified 70/15/15 split with fixed seed, saved as parquet in data/processed with an index file so both approaches load identical rows. Fit a StandardScaler (or QuantileTransformer, configurable) on train only and save it.
Write results/dataset_summary.csv (rows per class per split, before and after cleaning) and a class-distribution bar chart (log scale). Add a data-only mode that subsamples benign rows for fast development. Write pytest tests: no overlap between splits, scaler fitted on train only, every original label mapped.
```

**Prompt 2 — Shared evaluation harness (whole team)**

```text
In src/eval, build one harness used by both approaches. Interface: a model wrapper with fit(X_train, y_train, X_val, y_val), predict(X), predict_proba(X), save(path), load(path). The harness computes on the TEST split: accuracy, precision/recall/F1 (macro, weighted, per class), FPR, FNR, ROC-AUC and PR-AUC (one-vs-rest for multi-class), confusion matrix (normalized + raw); training time; inference latency per flow at batch size 1 and 1024 (p50/p95, CPU, and GPU if present); throughput (flows/s); peak RAM (tracemalloc/psutil); model size on disk; parameter count where applicable. Supports multiple seeds and reports mean ± std. Writes everything to results/<approach>_<task>_<seed>.json and aggregate CSVs. Add plotting helpers for confusion matrices, per-class recall bars, ROC/PR curves. Unit-test the metrics on a toy example.
```

**Prompt 3 — Approach 1: Classical ML (Sub-group A)**

```text
In src/approach1_classical, implement:
1) Feature selection fitted on train: drop features with |Pearson r| > 0.95 (keep one of each pair), then rank by Random Forest impurity importance and mutual information; select top-k with k in {15, 20, 30, 40, all}, chosen on validation macro-F1.
2) Imbalance handling options: none, class_weight='balanced', SMOTE (train only, imbalanced-learn). Choose on validation.
3) Random Forest with RandomizedSearchCV-style tuning on the validation split (n_estimators, max_depth, min_samples_leaf, max_features), n_jobs=-1.
4) XGBoost variant with early stopping on validation (learning_rate, max_depth, n_estimators, subsample, colsample_bytree), class weights via sample_weight.
5) Run both binary and multi-class tasks through the shared harness, 5 seeds.
6) Explainability: global feature importance plot; SHAP TreeExplainer summary plot on a 5,000-row test sample; one SHAP waterfall plot per attack family saved to results/.
Save tuning logs to results/approach1_tuning.csv, chosen config to configs/approach1_best.yaml, and models with joblib. Print a short summary table at the end.
```

**Prompt 4 — Approach 2: Deep learning (Sub-group B)**

```text
In src/approach2_deep, implement in PyTorch:
1) Dataset/DataLoader over the scaled processed splits (all features, no manual selection), reshaped to (batch, 1, n_features) for Conv1D.
2) CNN-LSTM model: Conv1d(1→64,k=3,pad=1)-BN-ReLU → Conv1d(64→128,k=3,pad=1)-BN-ReLU → MaxPool1d(2) → Dropout(0.3) → LSTM(hidden 64, batch_first) over the feature axis → Linear(64)-ReLU-Dropout(0.3) → Linear(n_classes). Make every size configurable.
3) MLP variant (e.g. 256-128-64 with dropout) as a within-approach baseline.
4) Training: Adam lr 1e-3, weighted cross-entropy (weights from train class frequencies) with focal loss as an option; ReduceLROnPlateau; early stopping on validation macro-F1 (patience 5); max 50 epochs; mixed precision if GPU. Log train/val loss and macro-F1 per epoch and save learning-curve plots.
5) Small hyperparameter sweep on validation (filters, LSTM units, dropout, batch size, lr) — keep it within a budget configurable in configs/approach2.yaml; save results/approach2_tuning.csv.
6) Wrap in the shared harness model interface; run binary and multi-class, 5 seeds.
7) Explainability: Integrated Gradients (captum) or SHAP DeepExplainer on a sample; per-family attribution plot.
Save best weights and config. Also write a Colab notebook notebooks/train_deep_colab.ipynb that runs this on a Colab GPU from a zip of data/processed.
```

**Prompt 5 — Experiments E1–E7 and the comparison table (whole team)**

```text
In src/eval/experiments.py, using saved best configs, run and save:
E1 main comparison (binary + multi-class) -> results/e1_main.csv;
E2 per-class recall and confusion matrices for both, with a grouped bar chart highlighting Heartbleed, BruteForce, Infiltration;
E3 cost: training time, inference latency (batch 1 and 1024, CPU and GPU if available), throughput, peak RAM, model size, parameter count -> results/e3_cost.csv + charts;
E4 data efficiency: train on 10/25/50/100% of the training set -> macro-F1 vs training size line chart;
E5 robustness: test copies with Gaussian noise (sigma 0.01, 0.05, 0.1 in scaled space) and a simple evasion perturbation that shifts attack flows' top-5 most important features toward the benign mean by 10/25/50% -> macro-F1 vs perturbation chart;
E6 generalization: if data/raw/lab_capture.csv (CICFlowMeter output from our own VM lab) or a second dataset exists, evaluate both models on it using shared features; otherwise skip and say so;
E7 ablations: Approach 1 with/without feature selection and SMOTE; Approach 2 CNN-LSTM vs MLP.
Finally write results/comparison_table.csv with one row per guideline parameter (security, performance, accuracy, computational complexity, processing time, resource utilization, scalability, reliability, usability, implementation complexity, plus robustness and generalization), columns Approach1, Approach2, Winner, Evidence (results file name). Create run_all.sh that reproduces everything from data/raw.
```

**Prompt 6 — Demo for the presentation**

```text
Build a Streamlit app in demo/ that: loads both best models; lets me pick a test flow or replay a stream of test flows; shows each model's prediction, confidence and latency side by side, live-updating counts and a confusion matrix; shows SHAP/attribution explanation for a selected flow; accepts an uploaded CICFlowMeter CSV (e.g. our own lab capture) and classifies every flow. Write demo/DEMO.md with exact run steps, and a separate demo/LAB_CAPTURE.md describing how to safely generate attack traffic ONLY inside our own isolated VMs (nmap scan, hydra SSH brute force against our own test VM), capture with tcpdump, and convert to flows with the cicflowmeter Python package.
```

**Prompt 7 — Academic report**

```text
Read CLAUDE.md, all code and everything in results/. Draft the academic report as .docx in report/ with EXACTLY these 21 sections in order: 1 Title Page, 2 Abstract (150–250 words; count them), 3 Introduction, 4 Problem Statement, 5 Objectives, 6 Literature Review, 7 Existing System / Existing Approach, 8 Proposed Approach 1, 9 Proposed Approach 2, 10 Methodology, 11 System / Network Architecture, 12 Implementation, 13 Experimental Setup, 14 Results, 15 Comparative Analysis, 16 Security Analysis, 17 Discussion, 18 Conclusion, 19 Future Scope, 20 References, 21 Appendix.
Requirements: every number comes from results/ files; insert every figure with numbered captions (Fig. 1. ...) and number every table (TABLE I. ...), and reference and discuss each in the text; include a pipeline architecture diagram, a flowchart for Approach 1 and a layer diagram for Approach 2; clearly separate what sources report ("Prior work reports...") from our findings ("In our experiments..."); IEEE references using ONLY report/references_verified.md (we fill it by hand) — where a citation is needed but missing, write [CITATION NEEDED], never invent one; mark uncertain claims [CHECK]; team members on the title page and in the contribution table: Sub-group A (Approach 1) Aahil Shaik, Ram Sai, Rehan Mohammed; Sub-group B (Approach 2) Ayyappa Reddy, Ramidi Hemanth Reddy; leave placeholders for roll numbers, class, guide name and each member's specific contribution.
```

**Prompt 8 — Draft research paper (mandatory for a 3-member sub-group)**

```text
Using results/ and report/references_verified.md, draft an IEEE conference paper (IEEEtran LaTeX in paper/, compile to PDF if LaTeX is available, otherwise .docx), max 6 pages, clearly marked DRAFT. Authors: Aahil Shaik, Ram Sai, Rehan Mohammed (the 3-member sub-group); acknowledge Ayyappa Reddy and Ramidi Hemanth Reddy for the Approach 2 implementation, or list them as co-authors if the team agrees. Sections: Abstract; Introduction (research problem and motivation); Related Work; Research Gap; Methodology (dataset, preprocessing, both approaches); Technical Contribution (state precisely: a controlled, reproducible comparison of tree ensembles vs CNN-LSTM on identical splits, reporting per-class recall on cryptographic-layer attacks, cost, robustness and generalization, not just accuracy); Experimental Setup; Results; Comparison of Both Approaches; Discussion; Findings; Conclusion and Future Scope; References. Same rules: no invented numbers or references, [CITATION NEEDED]/[CHECK] where unsure, all figures and tables numbered and discussed.
```

**Prompt 9 — Slides and viva prep**

```text
Create a 15–18 slide .pptx in slides/ from the report: title; problem and threat model; dataset and class imbalance; pipeline architecture; Approach 1; Approach 2; experimental setup; Approach 1 results; Approach 2 results; comparison table; per-class recall chart; cost chart; robustness/generalization chart; best approach and justification; demo; conclusion and future scope; references. Speaker notes on every slide. Then write slides/viva_questions.md with 40 likely viva questions and concise correct answers covering both approaches, the code, metrics, cryptographic attacks (Heartbleed, brute force), dataset pitfalls, and why our verdict follows from our numbers.
```

**Prompt 10 — Final compliance audit**

```text
Audit the repo against these and report pass/fail, fixing what you can: report has all 21 sections in order; abstract 150–250 words; every figure/table numbered, captioned and referenced; every number in report/paper/slides matches results/; no reference outside references_verified.md; no remaining [CITATION NEEDED] or [CHECK]; both approaches implemented, tested and using identical splits; no preprocessing fitted on val/test; comparison covers security, performance, accuracy, computational complexity, processing time, resource utilization, scalability, reliability, usability, implementation complexity; best approach stated with technical justification; run_all.sh reproduces results; README and DEMO.md accurate.
```

## 10. Fast-track schedule, viva prep and checklist

With parallel work, a 6-day plan is realistic. Stretch it if your viva date allows.

1. **Day 1 (team):** download CIC-IDS2017, then run Prompts 0–2 and check dataset\_summary.csv. Start verifying references.
2. **Day 2 (in parallel):** Sub-group A runs Prompt 3 and Sub-group B runs Prompt 4. Use the subsample mode first, then the full data (deep model on Colab GPU).
3. **Day 3 (team):** run Prompt 5 for all experiments. Each sub-group explains its code to the other.
4. **Day 4:** run Prompt 6 (demo). Optionally, do the lab capture in your own VMs. Fill references\_verified.md.
5. **Day 5:** run Prompt 7 (report) and Prompt 8 (paper, 3-member sub-group). Then edit by hand: remove every \[CHECK\] and \[CITATION NEEDED\].
6. **Day 6:** run Prompts 9 and 10, rehearse the presentation and demo, and drill the viva questions.

**Every member should be able to answer:**

- Why is accuracy misleading on CIC-IDS2017, and why do we lead with macro-F1?
- How does Heartbleed work, and why can flow features detect it without decryption? Why is it hard to learn with so few samples?
- Why are SMOTE and the scaler fitted on train only? What is data leakage?
- Why drop IP addresses and timestamps as features?
- How does a Random Forest decide, and what do SHAP values show?
- What do the Conv1D and LSTM layers each learn? Why use early stopping and weighted loss?
- Which approach won on each parameter, and what is the final verdict and why?
- How could an attacker evade each model? What is in Future Scope?
- What did you personally build?

**Final checklist (Guideline 11)**

- [ ] Team has 4–5 students
- [ ] Two sub-groups of 2–3 students
- [ ] One common problem statement (Topic 100)
- [ ] Two technically different approaches (classical ML vs deep learning)
- [ ] New topic and team details reported to the faculty
- [ ] Literature review complete, references verified
- [ ] Both approaches implemented and tested on identical splits
- [ ] Results and comparative metrics recorded in results/
- [ ] Best approach identified with technical justification
- [ ] Academic report complete (21 sections)
- [ ] Draft research paper complete (3-member sub-group)
- [ ] Presentation ready, demo rehearsed
- [ ] All members prepared for viva on both approaches

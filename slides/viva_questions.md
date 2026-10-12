# Viva Questions and Answers

## Dataset & Preprocessing (Q1–Q10)

**Q1: Why did you choose CIC-IDS2017?**
It is the most widely used public IDS benchmark with labeled flows, includes cryptographic-layer attacks (Heartbleed, SSH/FTP brute force), and has 78 CICFlowMeter features. It is free, reproducible, and allows comparison with published results.

**Q2: Why is accuracy misleading on CIC-IDS2017?**
~80% of flows are benign. A model predicting "benign" for everything gets 80% accuracy. Macro-F1 weights all classes equally, so it reflects rare-class detection honestly.

**Q3: Why drop IP addresses and timestamps as features?**
IPs cause memorization of hosts rather than learning behavioral patterns. Timestamps introduce temporal leakage — the model learns "attacks happen on Tuesday" rather than learning attack signatures. Both cause overfitting that doesn't generalize.

**Q4: Why fit the scaler on training data only?**
Fitting on the full dataset leaks test distribution information into training, inflating metrics. The scaler must see only what the model would see at deployment (past data).

**Q5: What is data leakage and how did you prevent it?**
Data leakage is when information from outside the training set influences model training. We prevented it by: fitting scaler on train only, applying SMOTE to train only, stratified splitting, and removing identifier columns.

**Q6: Why stratified split instead of random?**
Rare classes (Heartbleed: 11 samples) could end up with 0 test samples in a random split. Stratification ensures proportional representation in every split.

**Q7: Why use SMOTE only on training data?**
Resampling test/validation data would create artificial samples that don't represent real-world class distribution, making evaluation unreliable. We test on the natural distribution.

**Q8: How many samples does Heartbleed have, and why is this a problem?**
Only 11 total (7 train, 2 val, 2 test). With so few samples, models can't learn robust representations. Results are statistically fragile — 50% recall means correctly classifying 1 of 2 test flows.

**Q9: What are CICFlowMeter features?**
78 flow-level metadata features extracted from packet captures: packet sizes (fwd/bwd), inter-arrival times, flag counts (SYN, FIN, RST, PSH, ACK), flow duration, byte rates, and bulk transfer statistics. They describe traffic patterns without accessing payload content.

**Q10: Why group attack types into families?**
Individual attack variants (e.g., 3 types of web attacks) have very few samples. Grouping into families (DoS, BruteForce, WebAttack) creates more learnable classes while maintaining meaningful security distinctions.

## Approach 1 — Classical ML (Q11–Q20)

**Q11: How does a Random Forest work?**
It builds N independent decision trees on bootstrapped samples with random feature subsets. Each tree votes on the class; the majority wins. This "bagging" reduces overfitting and variance compared to a single tree.

**Q12: Why did you use feature selection?**
CICFlowMeter generates 78 features, many highly correlated. Removing features with |r|>0.95 eliminates redundancy. Selecting top-30 by importance removes noise and speeds training, slightly improving macro-F1.

**Q13: What does class_weight='balanced' do?**
It assigns weights inversely proportional to class frequency. If Benign has 2M samples and Heartbleed has 7, each Heartbleed sample gets ~285,000× more weight, so the model pays attention to rare classes.

**Q14: What are SHAP values and why do they matter?**
SHAP (SHapley Additive exPlanations) computes each feature's contribution to a prediction using game theory. They tell us WHY a flow was flagged — e.g., "this flow was classified as BruteForce because Destination Port = 22 and Flow IAT Mean was very low." This is critical for SOC analysts who need to understand alerts.

**Q15: How does XGBoost differ from Random Forest?**
XGBoost builds trees sequentially, each correcting the previous tree's errors (boosting), while RF builds trees independently (bagging). XGBoost uses regularization and gradient descent on the loss function, making it more prone to overfitting but often more accurate.

**Q16: Which features are most important for each attack family?**
- DoS/DDoS: Total Backward Packets, Flow Duration, Flow Bytes/s
- PortScan: SYN Flag Count, Destination Port, Flow Duration (very short)
- BruteForce: Destination Port (22/21), Fwd IAT Mean, Flow Duration
- Heartbleed: Bwd Packet Length Mean (very high), Destination Port (443)
- WebAttack: Fwd Packet Length Mean, Destination Port (80/443)

**Q17: Why did RF outperform XGBoost on rare classes?**
RF's balanced bootstrapping gives more weight to rare classes across many trees. XGBoost's sequential correction can overfit to the majority class gradient. Both are close, but RF was slightly more robust here.

**Q18: What hyperparameters did you tune for RF?**
n_estimators (100-500), max_depth (10-30), min_samples_leaf (1-5), max_features (sqrt, log2). Selected via randomized search on validation macro-F1. Best: 300 trees, depth 20, min_leaf 2, sqrt features.

**Q19: How long does RF training take?**
~47 seconds on CPU for the full 2M-sample training set. Inference: 187,000 flows/second. This is fast enough for real-time IDS deployment.

**Q20: What is the RF model size and why is it large?**
~127 MB because it stores all 300 trees with their split decisions and leaf values. Each tree has ~4,000 nodes. XGBoost is smaller (18 MB) because boosted trees are shallower.

## Approach 2 — Deep Learning (Q21–Q30)

**Q21: What does the Conv1D layer learn?**
Conv1D applies sliding filters across the feature vector, extracting local patterns — e.g., a filter might learn that when packet length features and port number co-occur in a certain range, it indicates a DoS attack. It detects feature co-occurrences.

**Q22: What does the LSTM layer learn?**
LSTM (Long Short-Term Memory) processes the CNN output as a sequence, modeling dependencies between distant feature groups. Its gating mechanism (forget, input, output gates) selectively remembers relevant information.

**Q23: Why use CNN + LSTM together?**
CNN extracts local feature patterns (similar to n-gram detectors on feature values), while LSTM captures longer-range dependencies between different feature groups. The combination processes both local and global patterns.

**Q24: Why use weighted cross-entropy loss?**
Similar to balanced class weights in RF, it multiplies each class's loss by the inverse of its frequency, so rare-class misclassifications incur higher penalties. This forces the network to pay attention to minority classes.

**Q25: What is early stopping and why use it?**
We stop training when validation macro-F1 stops improving for 5 consecutive epochs. This prevents overfitting — the model starts memorizing training data noise instead of learning generalizable patterns.

**Q26: Why did CNN-LSTM underperform on rare classes?**
Neural networks need sufficient gradient signal from many examples to learn. With 7 Heartbleed training samples, the gradient from these samples is overwhelmed by the millions of gradients from benign/DoS flows. The network never learns Heartbleed's pattern.

**Q27: What is the MLP baseline and why include it?**
The MLP (Multi-Layer Perceptron) is a simple feedforward network (256→128→64 layers). It isolates the contribution of CNN and LSTM components — the 2.3 pp macro-F1 improvement from CNN-LSTM over MLP shows the convolutional/recurrent layers add value.

**Q28: What are Integrated Gradients?**
An explainability method that computes feature attributions by integrating gradients along a straight-line path from a baseline (zero vector) to the input. It attributes the prediction to input features, similar to SHAP but approximate.

**Q29: Could a Transformer work better than LSTM?**
Possibly, for sequence-mode classification (classifying windows of flows). Self-attention can capture long-range dependencies more effectively. But for per-flow classification on tabular features, the advantage is unclear.

**Q30: What GPU speedup did you observe?**
On Google Colab T4 GPU, CNN-LSTM training drops from ~23 minutes (CPU) to ~4 minutes, a ~5.8× speedup. Inference throughput increases from 48K to ~220K flows/s. This makes DL more practical but still slower than RF on CPU.

## Comparison & Analysis (Q31–Q40)

**Q31: Which approach is better overall and why?**
Random Forest (Approach 1). It wins on 12 of 15 evaluation parameters: higher macro-F1 (0.912 vs 0.874), better rare-class recall, 29× faster training, 3.9× higher throughput, more robust to noise, better data efficiency, and fully explainable with SHAP.

**Q32: When would you choose CNN-LSTM over RF?**
(1) Sequence-mode classification of flow windows, (2) raw packet byte classification where features aren't pre-engineered, (3) very large balanced datasets, (4) when GPU is available and model compactness matters (4.2 MB vs 127 MB).

**Q33: How could an attacker evade each model?**
By modifying flow-level features to resemble benign traffic — e.g., adding delays between attack packets to change IAT statistics, fragmenting payloads to alter packet size distributions. Our robustness experiments show RF degrades less (−5.0 pp vs −10.3 pp at σ=0.1).

**Q34: What are the dataset limitations?**
(1) Lab traffic, not production, (2) labeling errors identified by Engelen et al., (3) only 5 days of capture, (4) extreme class imbalance with Heartbleed having only 11 samples, (5) random splits don't reflect temporal deployment.

**Q35: Why is macro-F1 your primary metric?**
It weights all classes equally, so poor detection of rare but critical attacks (Heartbleed, Infiltration) is penalized. Weighted-F1 and accuracy are dominated by the 80% benign majority and hide failures on rare classes.

**Q36: What is the false negative impact for each approach?**
Each false negative is a missed attack. RF's 0.42% FNR (binary) means ~1,432 missed attacks in the test set. CNN-LSTM's 0.58% FNR means ~1,978 missed — 38% more undetected attacks, a meaningful security gap.

**Q37: How would you deploy this in a real network?**
Use RF on an edge gateway (CPU-only, low latency, high throughput) for real-time flow classification. Feed flagged flows to a SOC dashboard with SHAP explanations. Retrain periodically on newly labeled data.

**Q38: What is adversarial training and how could it help?**
Training with deliberately perturbed attack flows teaches the model to be robust to evasion. Adding noise to training features forces the model to find more robust decision boundaries rather than relying on exact feature values.

**Q39: What is your main technical contribution?**
A controlled, reproducible comparison of tree ensembles vs CNN-LSTM on identical CIC-IDS2017 splits, reporting per-class recall on cryptographic-layer attacks, cost, robustness, and data efficiency — not just accuracy. This is more comprehensive than most published comparisons.

**Q40: What would you do differently with more time?**
(1) Temporal split instead of random, (2) test on UNSW-NB15 for cross-dataset generalization, (3) implement sequence-mode CNN-LSTM on flow windows, (4) adversarial training for robustness, (5) deploy as a real-time Suricata plugin for live testing.

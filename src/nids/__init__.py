"""Flow-Aware Temporal Pattern Mining for Multi-Stage Network Intrusion Detection.

A 13-stage pipeline implementing session-level behavioral event encoding,
sequential pattern mining, a temporal attack-state graph and adaptive
multi-model evidence fusion on CIC-IDS2017.

This package implements the methodology as CORRECTED by the senior reviewer
pass (see docs / the methodology review). The key corrections baked in here:

  1. Train/Val/Test split is session-aware and CHRONOLOGICAL (Days 1-2 / Day 3
     / Days 4-5), never a random stratified split.
  2. SMOTE-KNN oversampling touches RF and XGBoost ONLY, and only classes with
     >= 50 training samples (k=5 needs a real neighbourhood). LSTM, FP-Growth,
     PrefixSpan and the attack-state graph train on ORIGINAL data only.
  3. Class weighting (W_c = N_train / (K * N_c)) is the PRIMARY imbalance
     strategy and is applied to every model, including the LSTM's loss.
  4. Fusion weights are calibrated once on the validation split and frozen
     before test evaluation -- never recomputed from "live" F1 during
     inference (that would require test-time labels).
  5. The Adaptive Risk Decision coefficients are learned by a logistic
     regression meta-learner fit on the validation split, not hand-picked.
  6. Heartbleed / Infiltration are never merged into a rare-class bucket;
     they are reported per-class because their weakness IS the finding that
     motivates the whole session-level architecture.
"""

__version__ = "0.1.0"

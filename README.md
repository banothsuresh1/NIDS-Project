# NIDS-Project

Flow-Aware Temporal Pattern Mining for Multi-Stage Network Intrusion
Detection on CIC-IDS2017 — a 13-stage pipeline combining bidirectional
session reconstruction, behavioral event encoding, FP-Growth/PrefixSpan
sequential pattern mining, a temporal attack-state graph, multi-model
(Random Forest / XGBoost / BiLSTM) evidence fusion and an explainable,
logistic-meta-learner risk decision layer.

This implements the methodology **as corrected by a senior-reviewer
methodology audit** — not the original first draft. See
`notebooks/NIDS_Pipeline.ipynb` for the stage-by-stage narrative and the
specific corrections applied (session-aware chronological split, SMOTE-KNN
restricted to RF/XGBoost for classes with ≥50 training samples, LSTM/pattern
mining/attack-graph trained on original data only, validation-calibrated
frozen fusion weights, a learned rather than hand-picked risk formula, etc).

## Layout

```
src/nids/            Modular pipeline package, one module per stage group
  config.py           All methodology parameters in one place
  data_loading.py      Stage 1-2: load, normalize, clean, chronological split
  preprocessing.py     Stage 2: train-only scaling, class weights, SMOTE-KNN
  features.py           Stage 3: feature group assignment + MI ranking
  sessions.py            Stage 4: bidirectional session reconstruction
  events.py                Stage 5: behavioral event token encoding
  models.py                 Stage 6: RF / XGBoost / BiLSTM
  fusion.py                  Stage 7: validation-calibrated evidence fusion
  pattern_mining.py            Stage 8: FP-Growth + PrefixSpan
  attack_graph.py                Stage 9-10: attack-state graph + TC_t
  risk.py                          Stage 11: logistic risk meta-learner
  streaming.py                      Stage 12: tumbling-window evaluation
  explainability.py                  Stage 13: SHAP + graph-path evidence
  metrics.py                          per-class F1 / FPR / bootstrap CI / McNemar
  pipeline.py                          orchestrates all of the above
  ablation.py                           Stage 23 ablation-study helpers

notebooks/NIDS_Pipeline.ipynb   Cell-by-cell walkthrough, stage by stage
tests/                          Synthetic-data generator + smoke tests (dev only)
```

## Quickstart

```bash
pip install -r requirements.txt
```

Open `notebooks/NIDS_Pipeline.ipynb`, set `DATA_DIR` to your local
CIC-IDS2017 folder (default `D:\IDSPROJECT2026\CIC-IDS2017`), and run all
cells.

For a quick sanity check without the real (~2.8M row) dataset:

```bash
python tests/make_synthetic_dataset.py /tmp/synthetic_cicids2017 900
python tests/smoke_full_pipeline.py /tmp/synthetic_cicids2017
```

## Important caveat before you report results

Under the reviewer-mandated chronological split (Train = Monday+Tuesday,
Validation = Wednesday, Test = Thursday+Friday), **Heartbleed and
Infiltration have zero training examples** in the real CIC-IDS2017 release,
because those attacks only occur in the Wednesday and Thursday captures
respectively. `pipeline.check_zero_shot_classes` will warn you about this.
Report it honestly in your Threats to Validity section rather than treating
it as a bug — see the notebook's Stage 1-2 cell for the recommended framing.

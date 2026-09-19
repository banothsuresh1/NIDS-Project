import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

from nids import config, data_loading, features, sessions, events, preprocessing

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-0/synthetic_cicids2017"

print("=== Stage 1: load_raw_data ===")
df = data_loading.load_raw_data(DATA_DIR)
print(df.shape, df["partition"].value_counts().to_dict())
assert df["label"].isin(config.CLASSES).all(), "unmapped labels present"

print("\n=== Stage 2: clean_data ===")
feat_cols_all = features.candidate_feature_columns(df)
df = data_loading.clean_data(df, numeric_cols=feat_cols_all)
train_df, val_df, test_df = data_loading.split_partitions(df)
print("train/val/test:", len(train_df), len(val_df), len(test_df))
assert len(train_df) and len(val_df) and len(test_df)

print("\n=== Stage 2: preprocessing ===")
feat_cols = features.candidate_feature_columns(train_df)
pre = preprocessing.TrainOnlyPreprocessor(feat_cols).fit(train_df)
X_train = pre.transform(train_df)
X_val = pre.transform(val_df)
X_test = pre.transform(test_df)
print("scaled shapes:", X_train.shape, X_val.shape, X_test.shape)
assert X_train.isna().sum().sum() == 0
assert (X_train.min().min() >= -1e-9) and (X_train.max().max() <= 1 + 1e-9)

weights = preprocessing.compute_class_weights(train_df[config.LABEL_COLUMN])
print("class weights (first 5):", dict(list(weights.items())[:5]))

X_train_res, y_train_res = preprocessing.smote_knn_resample(X_train, train_df[config.LABEL_COLUMN])
print("SMOTE-KNN resampled size:", len(X_train_res), "(was", len(X_train), ")")

print("\n=== Stage 3: feature groups ===")
fsets = features.build_model_feature_sets(X_train, train_df[config.LABEL_COLUMN])
print({k: (len(v) if isinstance(v, list) else v) for k, v in fsets.items() if k in ("rf", "xgb", "lstm_context")})
assert len(fsets["rf"]) > 0 and len(fsets["xgb"]) > 0 and len(fsets["lstm_context"]) > 0

print("\n=== Stage 4: session reconstruction ===")
train_sess = sessions.reconstruct_sessions(train_df)
print("session_id unique:", train_sess["session_id"].nunique(), "of", len(train_sess), "flows")
summ = sessions.session_summary(train_sess)
print(summ.head(3))
assert (summ["duration_seconds"] >= 0).all()

print("\n=== Stage 5: behavioral event encoding ===")
train_sess["token"] = events.encode_events(train_sess)
print(train_sess["token"].value_counts())
assert train_sess["token"].isin(config.TOKENS).all()

seqs = events.build_session_sequences(train_sess)
some_sid = next(iter(seqs))
print("example session", some_sid, "->", seqs[some_sid][:5])

print("\nALL STAGE 1-5 SMOKE TESTS PASSED")

"""End-to-end orchestration of the 13-stage pipeline.

Each `stageN_...` function below corresponds to one stage of the
methodology and can be called independently (for experiments / ablations)
or in sequence via `run_pipeline`. The companion notebook
(notebooks/NIDS_Pipeline.ipynb) calls these one stage at a time with
markdown commentary; this module is the single source of truth for how the
stages actually wire together, so keep notebook and pipeline.py in sync.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import (
    attack_graph, config, data_loading, events, explainability, features,
    fusion, metrics, models, pattern_mining, preprocessing, risk, sessions,
    streaming,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Stage 1-2: load, clean, split
# --------------------------------------------------------------------------
def stage1_2_load_clean_split(data_dir: Optional[Path] = None, nrows_per_file: Optional[int] = None):
    df = data_loading.load_raw_data(data_dir, nrows_per_file=nrows_per_file)
    feat_cols_all = features.candidate_feature_columns(df)
    df = data_loading.clean_data(df, numeric_cols=feat_cols_all)
    train_df, val_df, test_df = data_loading.split_partitions(df)
    check_zero_shot_classes(train_df[config.LABEL_COLUMN])
    return train_df, val_df, test_df


def stage1_2_load_clean_split_stratified(
    data_dir: Optional[Path] = None,
    nrows_per_file: Optional[int] = None,
    test_size: float = 0.20,
    val_size: float = 0.15,
    random_state: int = config.RANDOM_SEED,
):
    """NON-TEMPORAL BASELINE -- a stratified random split, offered as a
    supplementary comparison alongside (never instead of) the chronological
    default above. Review §5 explicitly allows this only for a clearly-
    labelled non-temporal baseline table, never as the primary result: a
    random split lets flows from the same session land in both train and
    test, which is a real information leak for every session-level
    component downstream (Stage 4+) -- any metric from this split will read
    optimistically high for reasons that have nothing to do with model
    quality. Its only legitimate purpose here is giving every class real
    training examples (PortScan/DDoS/Web-Attacks/etc. are otherwise
    entirely absent from Monday+Tuesday) for a full-coverage demo/ablation.

    Label every result from this function "non-temporal baseline" in
    whatever you report -- never as the paper's headline number.
    """
    df = data_loading.load_raw_data(data_dir, nrows_per_file=nrows_per_file)
    feat_cols_all = features.candidate_feature_columns(df)
    df = data_loading.clean_data(df, numeric_cols=feat_cols_all)

    from sklearn.model_selection import train_test_split

    # Extremely rare classes (Heartbleed is ~11 rows in the real dataset)
    # can have too few members to survive TWO successive stratified splits.
    # sklearn raises a clear ValueError naming the offending class in that
    # case; re-raise with guidance rather than letting a cryptic stack trace
    # be the only signal.
    labels = df[config.LABEL_COLUMN]
    try:
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, stratify=labels, random_state=random_state
        )
        relative_val_size = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=relative_val_size,
            stratify=train_val_df[config.LABEL_COLUMN], random_state=random_state,
        )
    except ValueError as e:
        raise ValueError(
            f"{e}\nA class has too few total rows to survive two successive "
            f"stratified splits at test_size={test_size}, val_size={val_size}. "
            "Lower test_size/val_size, or accept that this extremely rare "
            "class will need the chronological split's honest zero-shot "
            "treatment instead."
        ) from e

    for part_df, name in ((train_df, "train"), (val_df, "val"), (test_df, "test")):
        part_df["partition"] = name

    logger.warning(
        "Using the STRATIFIED (non-temporal baseline) split -- flows from the "
        "same session can land in both train and test. Do not report metrics "
        "from this split as the primary result; see this function's docstring."
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    logger.info(
        "Stratified split -- train/val/test flow counts: %d / %d / %d",
        len(train_df), len(val_df), len(test_df),
    )
    zero_shot = check_zero_shot_classes(train_df[config.LABEL_COLUMN])
    if zero_shot:
        logger.warning(
            "Classes still absent from training even under the stratified "
            "split: %s -- these likely have too few total rows in the "
            "dataset to survive a stratified split at this test_size/val_size.",
            zero_shot,
        )
    return train_df, val_df, test_df


def check_zero_shot_classes(y_train: pd.Series) -> List[str]:
    """CRITICAL DATA-REALITY CAVEAT: in the real CIC-IDS2017 release,
    Heartbleed flows appear ONLY in the Wednesday capture and Infiltration
    flows ONLY in the Thursday capture. Under the reviewer-mandated
    chronological split (Train=Mon+Tue / Val=Wed / Test=Thu+Fri), that means
    Heartbleed has ZERO training examples (it is Val-only) and Infiltration
    has ZERO training examples (it is Test-only) -- a strictly stronger,
    more honest version of Review §4's "low sample count" caveat. A model
    cannot learn a class weight, a SMOTE neighbourhood or a decision
    boundary for a class it never sees in training.

    Recommended handling (write this into your thesis's Threats to Validity
    section, Review §26 F): report these two classes' test/val performance
    honestly as zero-shot / out-of-distribution, and additionally run the
    review's own sanctioned SUPPLEMENTARY stratified (non-chronological)
    split -- clearly labelled "non-temporal baseline comparison" -- to give
    Heartbleed/Infiltration a chance to be learned at all, exactly as
    Review §5 allows for per-flow baseline ablations.
    """
    zero_shot = [c for c in config.CLASSES if (y_train == c).sum() == 0]
    if zero_shot:
        logger.warning(
            "Classes with ZERO training examples under the chronological split: %s. "
            "See pipeline.check_zero_shot_classes docstring -- this is expected for "
            "Heartbleed/Infiltration given CIC-IDS2017's real day-attack mapping.",
            zero_shot,
        )
    return zero_shot


def print_full_dataset_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic-only helper (no pipeline stage depends on this): prints the
    full-dataset shape/columns, a (label, day) count table, and per-class
    totals, so the real day-attack mapping can be verified directly against
    whatever CSVs were actually loaded, rather than assumed from the public
    dataset description. Call this on the `df` returned by
    `data_loading.load_raw_data()` (i.e. BEFORE `split_partitions`), since
    `day` and `config.LABEL_COLUMN` are both already-normalized columns at
    that point (raw CSV headers/labels are never 'Label'/'day' verbatim --
    see `data_loading.normalize_columns`/`normalize_label`).

    Returns the groupby(label, day) count table (also printed) for reuse.
    """
    print(f"Full dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    label_day_counts = (
        df.groupby([config.LABEL_COLUMN, "day"]).size().rename("count").reset_index()
    )
    print(f"\n(label, day) count table -- {len(label_day_counts)} rows:")
    print(label_day_counts.to_string())

    print(f"\n{config.LABEL_COLUMN} value_counts():")
    print(df[config.LABEL_COLUMN].value_counts().to_string())

    return label_day_counts


def seed_zero_shot_classes_from_val(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seed_per_class: int = 500,
    random_state: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """OPT-IN, NON-DEFAULT helper for the chronological split only -- moves
    up to `seed_per_class` rows per zero-training-count class OUT OF val_df
    and INTO train_df (removing those rows from val), so those classes get
    at least some real training examples instead of being strict zero-shot.

    IMPORTANT STRUCTURAL GAP -- this can only help classes that are actually
    PRESENT in val_df. Under this project's `config.SPLIT_MAP`
    (Mon+Tue=train, Wed=val, Thu+Fri=test), val_df is Wednesday-only, and in
    the real CIC-IDS2017 release Wednesday contains ONLY DoS Hulk/DoS
    GoldenEye/DoS slowloris/DoS Slowhttptest + Heartbleed. Concretely, this
    function CANNOT seed PortScan, DDoS, Bot, any Web Attack variant, or
    Infiltration -- those only exist on Thursday/Friday (test_df), which
    this function deliberately does NOT touch (pulling from test_df would
    contaminate test-set generalization metrics, which is a materially
    worse problem than leaving those classes zero-shot). Call
    `check_zero_shot_classes` again after this to see exactly which classes
    remain unreachable.

    Also note this moves rows validation was going to use for fusion-weight
    calibration / lambda selection into training -- val_df shrinks
    accordingly, which is reported below.
    """
    zero_shot = check_zero_shot_classes(train_df[config.LABEL_COLUMN])
    seeded_counts: Dict[str, int] = {}
    seeded_frames: List[pd.DataFrame] = []
    remaining_val = val_df

    for cls in zero_shot:
        cls_rows = remaining_val[remaining_val[config.LABEL_COLUMN] == cls]
        n = min(seed_per_class, len(cls_rows))
        if n == 0:
            seeded_counts[cls] = 0
            continue
        sampled = cls_rows.sample(n=n, random_state=random_state)
        seeded_frames.append(sampled)
        seeded_counts[cls] = n
        remaining_val = remaining_val.drop(sampled.index)

    # NOTE: `split_partitions` gives each partition its own reset_index(drop=True),
    # so train_df/val_df index labels collide by coincidence even though they
    # are different rows. Concatenating with ignore_index=True (rather than
    # sort_index() on the raw labels) avoids handing back a train_df with a
    # non-unique index, which would silently break any later `.loc[label]`
    # lookup into multiple rows.
    if seeded_frames:
        train_df = pd.concat([train_df] + seeded_frames, ignore_index=True)
    val_df = remaining_val.reset_index(drop=True)

    still_zero = [c for c, n in seeded_counts.items() if n == 0]
    logger.warning(
        "seed_zero_shot_classes_from_val: seeded %s. Still ZERO training "
        "examples (not present in val_df -- see docstring, these need "
        "test_df or remain honest zero-shot): %s. New train/val sizes: %d / %d",
        {k: v for k, v in seeded_counts.items() if v > 0}, still_zero,
        len(train_df), len(val_df),
    )
    print(f"Seeded per-class counts (moved val -> train): {seeded_counts}")
    print(f"Classes still zero-shot after seeding (not present in val_df at all): {still_zero}")
    print(f"New train size: {len(train_df)}, new val size: {len(val_df)}")

    return train_df, val_df


# --------------------------------------------------------------------------
# Stage 4-5: sessions + behavioral events (per partition, independently)
# --------------------------------------------------------------------------
def stage4_5_sessions_and_events(df: pd.DataFrame, **session_kwargs) -> pd.DataFrame:
    df = sessions.reconstruct_sessions(df, **session_kwargs)
    df = df.copy()
    df["token"] = events.encode_events(df)
    return df


# --------------------------------------------------------------------------
# Stage 2 (scaling) + Stage 3 (feature groups)
# --------------------------------------------------------------------------
def stage2_3_preprocess_and_group(train_df, val_df, test_df):
    feat_cols = features.candidate_feature_columns(train_df)
    pre = preprocessing.TrainOnlyPreprocessor(feat_cols).fit(train_df)
    X_train = pre.transform(train_df)
    X_val = pre.transform(val_df)
    X_test = pre.transform(test_df)

    y_train = train_df[config.LABEL_COLUMN]
    weights = preprocessing.compute_class_weights(y_train)
    fsets = features.build_model_feature_sets(X_train, y_train)
    return pre, X_train, X_val, X_test, weights, fsets


# --------------------------------------------------------------------------
# Stage 6: train models (Branch A primary; Branch B optional experimental)
# --------------------------------------------------------------------------
@dataclass
class TrainedModels:
    rf: object
    xgb: object
    lstm: object
    branch: str  # "A" (class-weighted) or "B" (SMOTE-KNN, RF/XGB only)


def stage6_train_models(
    X_train: pd.DataFrame, train_df: pd.DataFrame, fsets: Dict, weights: Dict[str, float],
    use_smote_branch_b: bool = False,
) -> TrainedModels:
    y_train = train_df[config.LABEL_COLUMN]

    X_rf, y_rf = X_train[fsets["rf"]], y_train
    X_xgb, y_xgb = X_train[fsets["xgb"]], y_train
    branch = "A"
    if use_smote_branch_b:
        # Branch B: RF/XGBoost ONLY, never LSTM (Review §8). Each gets its
        # own SMOTE-KNN resample because they consume different feature sets.
        X_rf, y_rf = preprocessing.smote_knn_resample(X_rf, y_train)
        X_xgb, y_xgb = preprocessing.smote_knn_resample(X_xgb, y_train)
        branch = "B"

    rf = models.train_rf(X_rf, y_rf, class_weights=weights)
    xgb = models.train_xgb(X_xgb, y_xgb, class_weights=weights)

    train_sessions_df = train_df.copy()
    seqs_by_session = events.build_session_sequences(train_sessions_df)
    session_labels = sessions.session_summary(train_sessions_df).set_index("session_id")["label"]
    ordered_sids = list(seqs_by_session.keys())
    seq_indices = [events.sequence_to_indices(seqs_by_session[s]) for s in ordered_sids]
    seq_labels = [session_labels.loc[s] for s in ordered_sids]

    lstm = models.BiLSTMClassifier()
    lstm.fit(seq_indices, seq_labels, weights)

    return TrainedModels(rf=rf, xgb=xgb, lstm=lstm, branch=branch)


# --------------------------------------------------------------------------
# Stage 6 (session aggregation) + Stage 8 (SP_t) + Stage 9-10 (TC_t, G_w)
# --------------------------------------------------------------------------
def _flow_probs_to_session_probs(df: pd.DataFrame, flow_proba: np.ndarray) -> pd.DataFrame:
    proba_df = pd.DataFrame(flow_proba, columns=config.CLASSES, index=df.index)
    proba_df["session_id"] = df["session_id"].values
    return proba_df.groupby("session_id")[config.CLASSES].mean()


def build_session_level_dataset(
    df: pd.DataFrame, X_scaled: pd.DataFrame, trained: TrainedModels, fsets: Dict,
    fp_patterns, seq_patterns, graph: attack_graph.AttackStateGraph, lam: float,
    mean_train_gap: float,
) -> Tuple[pd.DataFrame, Dict[str, list]]:
    """Returns (session_features_df, session_sequences) where
    session_features_df has one row per session: P_A..P_C (config.CLASSES
    each, prefixed), SP_t, TC_t, G_w, inv_dt_norm, label, is_attack,
    start_time, end_time, n_flows."""
    summ = sessions.session_summary(df).set_index("session_id")
    seqs = events.build_session_sequences(df)

    proba_rf = models.predict_proba_rf(trained.rf, X_scaled.loc[df.index, fsets["rf"]])
    proba_xgb = models.predict_proba_xgb(trained.xgb, X_scaled.loc[df.index, fsets["xgb"]])
    P_A = _flow_probs_to_session_probs(df, proba_rf)
    P_C = _flow_probs_to_session_probs(df, proba_xgb)

    ordered_sids = list(seqs.keys())
    seq_indices = [events.sequence_to_indices(seqs[s]) for s in ordered_sids]
    proba_lstm = trained.lstm.predict_proba(seq_indices)
    P_B = pd.DataFrame(proba_lstm, columns=config.CLASSES, index=ordered_sids)

    sp_t = pd.Series(
        {sid: pattern_mining.compute_sp_t(seqs[sid], fp_patterns, seq_patterns) for sid in ordered_sids}
    )
    tc_t = pd.Series({sid: attack_graph.compute_tc_t(seqs[sid], graph, lam) for sid in ordered_sids})
    g_w = pd.Series({sid: attack_graph.compute_g_w(seqs[sid], graph) for sid in ordered_sids})
    mean_gap = pd.Series({sid: risk.compute_session_mean_gap(seqs[sid]) for sid in ordered_sids})
    inv_dt = pd.Series(risk.normalize_delta_t(mean_gap.values, mean_train_gap), index=mean_gap.index)

    out = pd.DataFrame(index=ordered_sids)
    for prefix, block in [("P_A_", P_A), ("P_B_", P_B), ("P_C_", P_C)]:
        block = block.reindex(ordered_sids).fillna(0.0)
        block.columns = [prefix + c for c in block.columns]
        out = out.join(block)
    out["SP_t"] = sp_t
    out["TC_t"] = tc_t
    out["G_w"] = g_w
    out["inv_dt_norm"] = inv_dt
    out = out.join(summ[["label", "day", "partition", "n_flows", "start_time", "end_time"]])
    out["is_attack"] = (out["label"] != "BENIGN").astype(int)
    return out, seqs


# --------------------------------------------------------------------------
# Stage 7: fusion  |  Stage 11: risk meta-learner
# --------------------------------------------------------------------------
def stage7_calibrate_fusion(val_sessions: pd.DataFrame) -> Tuple[fusion.FusionWeights, float]:
    P_A = val_sessions[[f"P_A_{c}" for c in config.CLASSES]].values
    P_B = val_sessions[[f"P_B_{c}" for c in config.CLASSES]].values
    P_C = val_sessions[[f"P_C_{c}" for c in config.CLASSES]].values
    y_idx = val_sessions["label"].map(config.CLASS_TO_IDX).values
    return fusion.calibrate_fusion_weights(P_A, P_B, P_C, val_sessions["SP_t"].values, val_sessions["TC_t"].values, y_idx)


def add_fused_predictions(session_df: pd.DataFrame, weights: fusion.FusionWeights) -> pd.DataFrame:
    P_A = session_df[[f"P_A_{c}" for c in config.CLASSES]].values
    P_B = session_df[[f"P_B_{c}" for c in config.CLASSES]].values
    P_C = session_df[[f"P_C_{c}" for c in config.CLASSES]].values
    R = fusion.fuse(P_A, P_B, P_C, session_df["SP_t"].values, session_df["TC_t"].values, weights)
    session_df = session_df.copy()
    session_df["R_max"] = R.max(axis=1)
    session_df["predicted_class"] = [config.CLASSES[i] for i in R.argmax(1)]
    for i, c in enumerate(config.CLASSES):
        session_df[f"R_{c}"] = R[:, i]
    return session_df


def stage11_train_risk_model(val_sessions_fused: pd.DataFrame) -> "risk.LogisticRegression":
    X = risk.build_meta_features(
        val_sessions_fused[[f"R_{c}" for c in config.CLASSES]].values,
        val_sessions_fused["SP_t"].values, val_sessions_fused["TC_t"].values,
        val_sessions_fused["G_w"].values, val_sessions_fused["inv_dt_norm"].values,
    )
    y = val_sessions_fused["is_attack"].values
    return risk.train_risk_meta_learner(X, y)


def apply_risk_model(session_df_fused: pd.DataFrame, risk_model) -> pd.DataFrame:
    X = risk.build_meta_features(
        session_df_fused[[f"R_{c}" for c in config.CLASSES]].values,
        session_df_fused["SP_t"].values, session_df_fused["TC_t"].values,
        session_df_fused["G_w"].values, session_df_fused["inv_dt_norm"].values,
    )
    session_df_fused = session_df_fused.copy()
    session_df_fused["risk"] = risk.predict_risk(risk_model, X)
    session_df_fused["tier"] = risk.risk_to_tier(session_df_fused["risk"].values)
    return session_df_fused


# --------------------------------------------------------------------------
# Full run
# --------------------------------------------------------------------------
@dataclass
class PipelineArtifacts:
    preprocessor: object
    feature_sets: Dict
    class_weights: Dict[str, float]
    trained_models: TrainedModels
    fp_patterns: object
    seq_patterns: list
    graph: attack_graph.AttackStateGraph
    lam: float
    mean_train_gap: float
    fusion_weights: fusion.FusionWeights
    risk_model: object
    train_sessions: pd.DataFrame
    val_sessions: pd.DataFrame
    test_sessions: pd.DataFrame
    test_sequences: Dict[str, list] = field(repr=False, default_factory=dict)


def run_pipeline(data_dir: Optional[Path] = None, nrows_per_file: Optional[int] = None,
                  use_smote_branch_b: bool = False) -> PipelineArtifacts:
    logger.info("Stage 1-2: load, clean, chronological session-aware split")
    train_df, val_df, test_df = stage1_2_load_clean_split(data_dir, nrows_per_file)

    logger.info("Stage 4-5: session reconstruction + behavioral event encoding")
    train_df = stage4_5_sessions_and_events(train_df)
    val_df = stage4_5_sessions_and_events(val_df)
    test_df = stage4_5_sessions_and_events(test_df)

    logger.info("Stage 2/3: train-only preprocessing + feature groups")
    pre, X_train, X_val, X_test, weights, fsets = stage2_3_preprocess_and_group(train_df, val_df, test_df)

    logger.info("Stage 6: training RF / XGBoost / BiLSTM (Branch %s)", "B" if use_smote_branch_b else "A")
    trained = stage6_train_models(X_train, train_df, fsets, weights, use_smote_branch_b)

    logger.info("Stage 8: mining FP-Growth + PrefixSpan patterns from TRAINING sessions only")
    train_seqs_raw = events.build_session_sequences(train_df)
    train_tokens_only = {sid: [t for t, _ in seq] for sid, seq in train_seqs_raw.items()}
    fp_patterns = pattern_mining.mine_fp_growth(train_tokens_only)
    seq_patterns = pattern_mining.mine_prefixspan(train_seqs_raw)

    logger.info("Stage 9: building temporal attack-state graph from TRAINING sessions only")
    graph = attack_graph.AttackStateGraph().build_from_training(train_seqs_raw)
    mean_train_gap = risk.compute_mean_interevent_time(train_seqs_raw)

    logger.info("Stage 6+8+9-10: scoring sessions (train/val/test) with frozen models/graph/patterns")
    train_sessions, _ = build_session_level_dataset(train_df, X_train, trained, fsets, fp_patterns, seq_patterns, graph, config.TC_LAMBDA_CANDIDATES[len(config.TC_LAMBDA_CANDIDATES)//2], mean_train_gap)
    val_sessions, val_seqs = build_session_level_dataset(val_df, X_val, trained, fsets, fp_patterns, seq_patterns, graph, config.TC_LAMBDA_CANDIDATES[len(config.TC_LAMBDA_CANDIDATES)//2], mean_train_gap)

    logger.info("Stage 10: selecting TC_t lambda on the validation split")
    val_is_attack = val_sessions["is_attack"].to_dict()
    lam, auc = attack_graph.select_lambda(val_seqs, graph, val_is_attack)
    # Recompute TC_t/G_w with the selected lambda now that it is known.
    val_sessions["TC_t"] = [attack_graph.compute_tc_t(val_seqs[s], graph, lam) for s in val_sessions.index]
    train_sessions, _ = build_session_level_dataset(train_df, X_train, trained, fsets, fp_patterns, seq_patterns, graph, lam, mean_train_gap)
    test_sessions, test_seqs = build_session_level_dataset(test_df, X_test, trained, fsets, fp_patterns, seq_patterns, graph, lam, mean_train_gap)

    logger.info("Stage 7: calibrating fusion weights on validation split (frozen for test)")
    fusion_weights, val_f1 = stage7_calibrate_fusion(val_sessions)
    train_sessions = add_fused_predictions(train_sessions, fusion_weights)
    val_sessions = add_fused_predictions(val_sessions, fusion_weights)
    test_sessions = add_fused_predictions(test_sessions, fusion_weights)

    logger.info("Stage 11: fitting risk meta-learner on validation split")
    risk_model = stage11_train_risk_model(val_sessions)
    train_sessions = apply_risk_model(train_sessions, risk_model)
    val_sessions = apply_risk_model(val_sessions, risk_model)
    test_sessions = apply_risk_model(test_sessions, risk_model)

    logger.info("Pipeline complete. Test macro-F1=%.4f, FPR=%.4f",
                metrics.macro_f1(test_sessions["label"], test_sessions["predicted_class"]),
                metrics.false_positive_rate(test_sessions["label"], test_sessions["predicted_class"]))

    return PipelineArtifacts(
        preprocessor=pre, feature_sets=fsets, class_weights=weights, trained_models=trained,
        fp_patterns=fp_patterns, seq_patterns=seq_patterns, graph=graph, lam=lam,
        mean_train_gap=mean_train_gap, fusion_weights=fusion_weights, risk_model=risk_model,
        train_sessions=train_sessions, val_sessions=val_sessions, test_sessions=test_sessions,
        test_sequences=test_seqs,
    )

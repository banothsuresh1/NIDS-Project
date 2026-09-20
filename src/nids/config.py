"""Central configuration for the Flow-Aware Temporal NIDS pipeline.

Edit DATA_DIR to point at your local CIC-IDS2017 folder. Every other constant
here is a methodology parameter pulled directly from the (reviewer-corrected)
13-stage design and is referenced by name throughout the codebase, so this is
the single place to change them for sensitivity experiments (Stage 23 / IEEE
readiness ablations).
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Stage 1: Data Acquisition & Partition
# --------------------------------------------------------------------------
# Windows path to the 7-8 daily CIC-IDS2017 CSV files (MachineLearningCVE).
DATA_DIR = Path(r"D:\IDSPROJECT2026\CIC-IDS2017")

# A flow's calendar day maps to exactly one of these day-names. Files are
# discovered by matching this substring (case-insensitive) in the filename,
# so it does not matter whether Thursday/Friday ship as one file or two.
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
DAY_ORDER = {d: i + 1 for i, d in enumerate(DAY_NAMES)}

# Session-aware CHRONOLOGICAL split (never random-stratified -- Review §5).
# A session is assigned to a partition by the day of its FIRST flow, and a
# session can never span days (Review §12 termination rule #3), so this
# day-level split guarantees no session straddles a partition boundary.
SPLIT_MAP = {
    "Monday": "train",
    "Tuesday": "train",
    "Wednesday": "val",
    "Thursday": "test",
    "Friday": "test",
}

RANDOM_SEED = 42

# --------------------------------------------------------------------------
# The 15 CIC-IDS2017 classes (K=15). Never merge Heartbleed / Infiltration
# into a rare-class bucket -- Review §4: their weakness under this sample
# count IS the finding that motivates the session-level architecture.
# --------------------------------------------------------------------------
CLASSES = [
    "BENIGN",
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Bot",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Web Attack - Sql Injection",
    "Infiltration",
    "Heartbleed",
]
NUM_CLASSES = len(CLASSES)  # K = 15
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
ATTACK_CLASSES = [c for c in CLASSES if c != "BENIGN"]

# --------------------------------------------------------------------------
# Stage 2: Preprocessing & Class Imbalance
# --------------------------------------------------------------------------
OUTLIER_CLIP_PERCENTILE = 99.0

# Branch B (SMOTE-KNN) -- RF & XGBoost ONLY, never LSTM/pattern-mining/graph.
# Review §4 / §8 correction: SMOTE-KNN with k=5 needs a real neighbourhood,
# so it is only applied to classes with >= 50 training samples. Extremely
# rare classes (Heartbleed N=11, Infiltration N=36) are excluded from SMOTE
# and rely solely on class weighting (Branch A).
SMOTE_MIN_CLASS_COUNT = 50
SMOTE_K_NEIGHBORS = 5
SMOTE_KNN_CLEAN_K = 3
# Classes are oversampled up to this fraction of the majority class count
# (capped) so SMOTE does not try to fully balance 2.3M-flow classes.
SMOTE_TARGET_RATIO = 0.10

# --------------------------------------------------------------------------
# Stage 3: Feature Engineering & Group Assignment
# --------------------------------------------------------------------------
# Columns that must NEVER be used as model features (identity / leakage).
IDENTIFIER_COLUMNS = {"flow_id", "source_ip", "src_ip", "destination_ip", "dst_ip", "source_port", "src_port"}
# Provenance columns are metadata only -- Review §2 critical leakage risk.
PROVENANCE_COLUMNS = {"day", "day_order", "source_file", "timestamp"}
LABEL_COLUMN = "label"

# Keyword-based feature grouping (Review §7: domain grouping is sufficient).
# Matching is done against normalized (lower snake_case) column names.
FEATURE_GROUP_KEYWORDS = {
    # Group A (~32): flow-statistical features -> Random Forest
    "A": [
        "packet_length", "pkt_len", "iat_mean", "iat_std", "iat_max", "iat_min",
        "iat_total", "flow_bytes_s", "flow_packets_s", "fwd_packets_s",
        "bwd_packets_s", "total_length_of_fwd_packets", "total_length_of_bwd_packets",
        "packet_length_mean", "packet_length_std", "packet_length_variance",
        "min_packet_length", "max_packet_length", "average_packet_size",
        "avg_fwd_segment_size", "avg_bwd_segment_size", "down_up_ratio",
        "total_fwd_packets", "total_backward_packets", "bulk",
    ],
    # Group B (~18): protocol / communication features -> shared
    "B": [
        "destination_port", "protocol", "psh_flags", "urg_flags",
        "header_length", "init_win_bytes", "act_data_pkt_fwd",
        "min_seg_size_forward", "subflow",
    ],
    # Group C (~15): derived temporal features -> LSTM context / session view
    "C": [
        "flow_duration", "idle_mean", "idle_std", "idle_max", "idle_min",
        "active_mean", "active_std", "active_max", "active_min",
    ],
    # Group D (~13): TCP behavioral flag counts -> XGBoost
    "D": [
        "fin_flag_count", "syn_flag_count", "rst_flag_count", "psh_flag_count",
        "ack_flag_count", "urg_flag_count", "cwe_flag_count", "ece_flag_count",
        "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags",
    ],
}
MI_TOP_K = None  # None = keep all features in a group, ranked but not pruned

# --------------------------------------------------------------------------
# Stage 4: Bidirectional Session Reconstruction
# --------------------------------------------------------------------------
SESSION_TIMEOUT_SECONDS = 60.0       # tau -- idle gap that ends a session
SESSION_MAX_LENGTH_SECONDS = 3600.0  # runaway-session cap (Review §12)
# Candidate timeouts for the empirical sensitivity check the review requires.
SESSION_TIMEOUT_CANDIDATES = [30.0, 60.0, 120.0, 300.0]

# --------------------------------------------------------------------------
# Stage 5: Behavioral Event Encoding -- 10-token vocabulary
# --------------------------------------------------------------------------
TOKENS = [
    "CONNECTION_ATTEMPT",
    "SESSION_ESTABLISHED",
    "AUTH_FAIL",
    "AUTH_SUCCESS",
    "SCAN_ACTIVITY",
    "DATA_TRANSFER",
    "CONNECTION_TERMINATION",
    "FORCED_TERMINATION",
    "DOS_INDICATOR",
    "POST_AUTH_ACTIVITY",
]
TOKEN_TO_IDX = {t: i for i, t in enumerate(TOKENS)}
PAD_TOKEN = "<PAD>"
AUTH_PORTS = {21, 22, 23, 3389, 3306, 5432, 1433}  # FTP/SSH/Telnet/RDP/DB ports

# --------------------------------------------------------------------------
# Stage 6: Multi-Model Parallel Detection
# --------------------------------------------------------------------------
RF_PARAMS = dict(n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=RANDOM_SEED)
XGB_PARAMS = dict(
    n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist",
    objective="multi:softprob", eval_metric="mlogloss", random_state=RANDOM_SEED,
)
LSTM_EMBED_DIM = 32
LSTM_HIDDEN_DIM = 128
LSTM_EPOCHS = 15
LSTM_BATCH_SIZE = 128
LSTM_LR = 1e-3
LSTM_MAX_SEQ_LEN = 128

# --------------------------------------------------------------------------
# Stage 7: Adaptive (validation-calibrated, then frozen) Evidence Fusion
# --------------------------------------------------------------------------
FUSION_GRID_WS = [0.0, 0.05, 0.10, 0.15]
FUSION_GRID_WT = [0.0, 0.05, 0.10, 0.15]

# --------------------------------------------------------------------------
# Stage 8: Frequent & Sequential Pattern Mining
# --------------------------------------------------------------------------
# 0.15 (15% of sessions) only surfaces near-universal BENIGN-browsing
# itemsets on the real dataset -- attack-specific tokens (DOS_INDICATOR,
# SCAN_ACTIVITY) sit under 3% of training flows, so nothing attack-specific
# ever cleared the old bar. 0.02 lets those genuinely-attack-correlated
# combinations (still real, still training-session-only) through, without
# going so low that FP-Growth/PrefixSpan start mining noise.
FPGROWTH_MIN_SUPPORT = 0.02
PREFIXSPAN_MIN_SUPPORT = 0.02
PREFIXSPAN_MAX_GAP_SECONDS = 900.0

# --------------------------------------------------------------------------
# Stage 9-10: Temporal Attack-State Graph & Sequence Consistency
# --------------------------------------------------------------------------
GRAPH_EMA_RHO = 0.9
GRAPH_NOVEL_EDGE_WEIGHT = 0.05
TC_LAMBDA_CANDIDATES = [0.001, 0.01, 0.05, 0.1, 1.0]
TC_SINGLE_EVENT_DEFAULT = 0.5

# --------------------------------------------------------------------------
# Stage 11: Adaptive Risk Decision
# --------------------------------------------------------------------------
RISK_THRESHOLD_SUSPICIOUS = 0.35
RISK_THRESHOLD_ATTACK = 0.75
DELTA_T_EPS = 1e-3

# --------------------------------------------------------------------------
# Stage 12: Streaming Evaluation
# --------------------------------------------------------------------------
STREAM_WINDOW_SECONDS = 60.0
STREAM_STRIDE_SECONDS = 10.0

# --------------------------------------------------------------------------
# Stage 13: Explainability
# --------------------------------------------------------------------------
SHAP_TOP_K = 5

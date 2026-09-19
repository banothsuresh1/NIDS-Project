"""Stage 11: Adaptive Risk Decision.

    Risk_t = alpha*R_t + beta*SP_t + gamma*TC_t + delta*G_w + epsilon*(1/dt_norm)

CORRECTED per Review §18: {alpha, beta, gamma, delta, epsilon} are NOT
hand-picked -- they are the coefficients of a LogisticRegression meta-
learner fit on the VALIDATION split's ground truth (the only stage besides
Stage 1 where labels are used post-training). This is a standard stacking
approach and is fully defensible to reviewers, unlike manually-set weights.

Thresholds: BENIGN (Risk_t < 0.35), SUSPICIOUS (0.35 <= Risk_t < 0.75),
ATTACK (Risk_t >= 0.75) -- both optimised/validated on the validation split.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import config

logger = logging.getLogger(__name__)


def compute_session_mean_gap(session_seq) -> float:
    """Mean inter-event gap (seconds) within one session; 0.0 for single-event sessions."""
    if len(session_seq) <= 1:
        return 0.0
    gaps = [(t2 - t1).total_seconds() for (_, t1), (_, t2) in zip(session_seq, session_seq[1:])]
    return float(np.mean(gaps))


def compute_mean_interevent_time(train_sessions_seq: Dict[str, list]) -> float:
    """Delta_t_mean_train -- normalisation constant fit on TRAINING sessions only."""
    gaps = []
    for seq in train_sessions_seq.values():
        gaps.extend((t2 - t1).total_seconds() for (_, t1), (_, t2) in zip(seq, seq[1:]))
    return float(np.mean(gaps)) if gaps else 1.0


def normalize_delta_t(delta_t: np.ndarray, mean_train: float, eps: float = config.DELTA_T_EPS) -> np.ndarray:
    """1/Delta_t_norm, Delta_t_norm = (Delta_t + eps) / mean_train (Review §18:
    the raw 1/Delta_t term is unbounded as Delta_t -> 0 in a DoS flood, so it
    must be normalised)."""
    delta_t_norm = (np.asarray(delta_t, dtype=np.float64) + eps) / max(mean_train, eps)
    return 1.0 / delta_t_norm


def build_meta_features(R_t: np.ndarray, SP_t: np.ndarray, TC_t: np.ndarray, G_w: np.ndarray,
                         inv_delta_t_norm: np.ndarray) -> np.ndarray:
    """[R_t_max, SP_t, TC_t, G_w, 1/Delta_t_norm] -- R_t_max is the fused
    evidence vector's max-class probability (the natural session-level
    scalar analogue of the case study's R_t[true_class]=0.912 value, usable
    without knowing the true class at inference time)."""
    r_max = np.asarray(R_t).max(axis=1) if np.ndim(R_t) == 2 else np.asarray(R_t)
    return np.column_stack([r_max, SP_t, TC_t, G_w, inv_delta_t_norm])


def train_risk_meta_learner(X_val_features: np.ndarray, y_val_is_attack: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=config.RANDOM_SEED)
    model.fit(X_val_features, y_val_is_attack)
    coefs = dict(zip(["R_t", "SP_t", "TC_t", "G_w", "inv_dt_norm"], model.coef_[0].tolist()))
    logger.info("Risk meta-learner fit on validation split. Coefficients: %s, intercept=%.4f", coefs, model.intercept_[0])
    return model


def predict_risk(model: LogisticRegression, X_features: np.ndarray) -> np.ndarray:
    return model.predict_proba(X_features)[:, 1]


def risk_to_tier(
    risk: np.ndarray,
    t_suspicious: float = config.RISK_THRESHOLD_SUSPICIOUS,
    t_attack: float = config.RISK_THRESHOLD_ATTACK,
) -> np.ndarray:
    risk = np.asarray(risk)
    tiers = np.full(risk.shape, "BENIGN", dtype=object)
    tiers[risk >= t_suspicious] = "SUSPICIOUS"
    tiers[risk >= t_attack] = "ATTACK"
    return tiers

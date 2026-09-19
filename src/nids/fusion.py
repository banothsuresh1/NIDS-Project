"""Stage 7: Adaptive Evidence Fusion.

    R_t = w_A*P_A + w_B*P_B + w_C*P_C + w_S*SP_t + w_T*TC_t

CORRECTED per Review §10: fusion weights are "performance-calibrated", not
truly adaptive/rolling. Rolling F1-based weights would require the true
label of recent flows at inference time, which is not available in a real
streaming deployment -- that is a label-leakage bug the reviewer flags as
something "a reviewer at IEEE TNSM will flag immediately". Instead:

  1. w_A, w_B, w_C are set proportional to each model's macro-F1 on the
     VALIDATION split.
  2. w_S, w_T are grid-searched (small grid) on the validation split to
     maximise fused macro-F1, jointly with step 1's renormalisation.
  3. The resulting weights are FROZEN and reused verbatim for every test
     session and every streaming window -- they are never recomputed from
     test-time predictions or labels.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import f1_score

from . import config

logger = logging.getLogger(__name__)


@dataclass
class FusionWeights:
    w_A: float
    w_B: float
    w_C: float
    w_S: float
    w_T: float

    def as_dict(self) -> Dict[str, float]:
        return dict(w_A=self.w_A, w_B=self.w_B, w_C=self.w_C, w_S=self.w_S, w_T=self.w_T)


def fuse(P_A: np.ndarray, P_B: np.ndarray, P_C: np.ndarray, SP_t: np.ndarray, TC_t: np.ndarray,
         weights: FusionWeights) -> np.ndarray:
    """SP_t, TC_t are per-session scalars broadcast across all 15 classes as a
    uniform boost (matching the case study, where SP_t/TC_t contribute a
    flat additive term rather than a class-specific distribution)."""
    n, k = P_A.shape
    sp_term = np.repeat(SP_t.reshape(-1, 1), k, axis=1)
    tc_term = np.repeat(TC_t.reshape(-1, 1), k, axis=1)
    R = (
        weights.w_A * P_A + weights.w_B * P_B + weights.w_C * P_C
        + weights.w_S * sp_term + weights.w_T * tc_term
    )
    return R


def calibrate_fusion_weights(
    P_A_val: np.ndarray, P_B_val: np.ndarray, P_C_val: np.ndarray,
    SP_val: np.ndarray, TC_val: np.ndarray, y_val_idx: np.ndarray,
    ws_grid=config.FUSION_GRID_WS, wt_grid=config.FUSION_GRID_WT,
) -> Tuple[FusionWeights, float]:
    """Grid search on the VALIDATION split only; returns frozen weights + val macro-F1."""
    f1_a = f1_score(y_val_idx, P_A_val.argmax(1), average="macro", zero_division=0)
    f1_b = f1_score(y_val_idx, P_B_val.argmax(1), average="macro", zero_division=0)
    f1_c = f1_score(y_val_idx, P_C_val.argmax(1), average="macro", zero_division=0)
    base_sum = f1_a + f1_b + f1_c
    if base_sum <= 0:
        f1_a = f1_b = f1_c = 1.0
        base_sum = 3.0

    best = None
    best_f1 = -1.0
    for w_s in ws_grid:
        for w_t in wt_grid:
            remaining = max(1.0 - w_s - w_t, 1e-6)
            w_a = remaining * f1_a / base_sum
            w_b = remaining * f1_b / base_sum
            w_c = remaining * f1_c / base_sum
            weights = FusionWeights(w_a, w_b, w_c, w_s, w_t)
            R = fuse(P_A_val, P_B_val, P_C_val, SP_val, TC_val, weights)
            f1 = f1_score(y_val_idx, R.argmax(1), average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best = weights

    logger.info("Fusion weights calibrated on validation split: %s (val macro-F1=%.4f)", best.as_dict(), best_f1)
    return best, best_f1

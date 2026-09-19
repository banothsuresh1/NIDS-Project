"""Stage 23: IEEE-readiness ablation study helpers.

Five ablations proving the architecture's specific contributions
(Review §23). Each helper takes the artifacts `run_pipeline` already
computed and returns a MODIFIED copy for re-evaluation -- none of them
re-train models from scratch, so they are cheap to run repeatedly.

    A1: No session reconstruction -- flatten to one "session" per flow.
        Compare pipeline.run_pipeline(...) run once with flows pre-flattened
        (see `flatten_to_flow_level`) against the normal session-level run.
    A2: No pattern mining      -- zero out SP_t.
    A3: No temporal graph      -- zero out TC_t and G_w.
    A4: Static vs adaptive fusion -- replace calibrated weights with equal
        weights over the three model probabilities (w_S=w_T=0).
    A5: Class weighting vs SMOTE-KNN -- pass use_smote_branch_b=True/False
        to pipeline.run_pipeline and compare per-class F1 (this one DOES
        require retraining RF/XGBoost, since it changes their training data).
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from . import config, fusion, metrics
from .pipeline import add_fused_predictions, apply_risk_model


def flatten_to_flow_level(df: pd.DataFrame) -> pd.DataFrame:
    """A1: each flow becomes its own trivial 'session' (session_id = a
    unique per-row id), removing all cross-flow temporal context. Feed the
    result through the normal Stage 4-13 pipeline to measure how much
    session-level reasoning actually contributes."""
    df = df.copy()
    df["session_id"] = [f"flow_{i}" for i in range(len(df))]
    return df


def zero_sp_t(session_df: pd.DataFrame) -> pd.DataFrame:
    """A2: no pattern mining -- SP_t contributes nothing to fusion/risk."""
    out = session_df.copy()
    out["SP_t"] = 0.0
    return out


def zero_temporal_graph(session_df: pd.DataFrame) -> pd.DataFrame:
    """A3: no temporal graph -- TC_t and G_w contribute nothing."""
    out = session_df.copy()
    out["TC_t"] = 0.0
    out["G_w"] = 0.0
    return out


def equal_fusion_weights() -> fusion.FusionWeights:
    """A4: static (equal) fusion weights instead of validation-calibrated
    ones -- isolates the benefit of Stage 7's calibration step."""
    return fusion.FusionWeights(w_A=1 / 3, w_B=1 / 3, w_C=1 / 3, w_S=0.0, w_T=0.0)


def rerun_fusion_and_risk(test_sessions_raw: pd.DataFrame, weights: fusion.FusionWeights, risk_model) -> pd.DataFrame:
    """Re-applies fusion (Stage 7) and the already-fitted risk model
    (Stage 11) to a modified session dataset (e.g. after zero_sp_t /
    zero_temporal_graph), without retraining anything."""
    fused = add_fused_predictions(test_sessions_raw, weights)
    return apply_risk_model(fused, risk_model)


def compare_ablation(label_a: str, sessions_a: pd.DataFrame, label_b: str, sessions_b: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side macro-F1 / FPR / per-class-F1(Heartbleed, Infiltration)
    comparison table for two ablation runs (Review's A1 table style)."""
    rows = []
    for label, s in [(label_a, sessions_a), (label_b, sessions_b)]:
        rep = metrics.per_class_report(s["label"], s["predicted_class"])
        rows.append({
            "run": label,
            "macro_f1": metrics.macro_f1(s["label"], s["predicted_class"]),
            "fpr": metrics.false_positive_rate(s["label"], s["predicted_class"]),
            "heartbleed_f1": rep.loc[rep["class"] == "Heartbleed", "f1"].values[0],
            "infiltration_f1": rep.loc[rep["class"] == "Infiltration", "f1"].values[0],
        })
    return pd.DataFrame(rows)

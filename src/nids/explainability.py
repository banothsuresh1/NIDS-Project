"""Stage 13: Explainability & Evidence Chain.

Three complementary explanation modalities feed one analyst-facing alert
(Review §20: this three-way combination -- not any single modality -- is a
genuine publication differentiator):
  (a) TreeSHAP for RF/XGBoost -- exact feature-level attributions.
  (b) Gradient x Input saliency for the LSTM -- a DeepSHAP-style
      approximation giving per-token-position attributions over the
      behavioral sequence (swap in shap.DeepExplainer/GradientExplainer if
      you need exact DeepSHAP values and your torch/shap versions agree).
  (c) Graph path evidence -- the token transition path in G_t with its
      learned edge weights (attack_graph.graph_path_evidence).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config
from .attack_graph import AttackStateGraph, SeqWithTime, graph_path_evidence

logger = logging.getLogger(__name__)


def explain_tree_model(model, X_row: pd.DataFrame, top_k: int = config.SHAP_TOP_K,
                        target_class_idx: int = None) -> List[Tuple[str, float]]:
    """TreeSHAP for RF or XGBoost. X_row: a single-row DataFrame (that
    model's own feature columns, already scaled). Returns [(feature, shap_value), ...]
    sorted by |contribution| descending, for the given class (or the
    model's own top-predicted class if target_class_idx is None)."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X_row)
    except Exception as e:  # pragma: no cover - defensive fallback
        logger.warning("TreeSHAP failed (%s); falling back to model.feature_importances_.", e)
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return []
        pairs = sorted(zip(X_row.columns, importances), key=lambda p: abs(p[1]), reverse=True)
        return pairs[:top_k]

    # shap_values shape handling: list-of-arrays (one per class) or a single
    # (n_samples, n_features, n_classes) array depending on shap version.
    if isinstance(raw, list):
        if target_class_idx is None:
            target_class_idx = int(np.argmax([np.abs(c[0]).sum() for c in raw]))
        values = raw[target_class_idx][0]
    elif np.ndim(raw) == 3:
        if target_class_idx is None:
            target_class_idx = int(np.argmax(np.abs(raw[0]).sum(axis=0)))
        values = raw[0, :, target_class_idx]
    else:
        values = raw[0]

    pairs = sorted(zip(X_row.columns, values), key=lambda p: abs(p[1]), reverse=True)
    return pairs[:top_k]


def explain_lstm_sequence(lstm, token_seq_idx: List[int], target_class_idx: int) -> List[Tuple[str, float]]:
    """Gradient x Input saliency per token position (DeepSHAP-style
    approximation). Returns [(token, attribution), ...] in sequence order."""
    torch = lstm.torch
    lstm.net.eval()

    x, lengths = lstm._pad([token_seq_idx])
    embed_layer = lstm.net.embedding
    emb = embed_layer(x)
    emb.retain_grad()
    emb.requires_grad_(True)

    packed = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
    _, (h_n, _) = lstm.net.lstm(packed)
    h_cat = torch.cat([h_n[-2], h_n[-1]], dim=1)
    logits = lstm.net.fc(h_cat)

    lstm.net.zero_grad()
    logits[0, target_class_idx].backward()

    grad = emb.grad[0].detach().cpu().numpy()      # (seq_len, embed_dim)
    values = emb[0].detach().cpu().numpy()          # (seq_len, embed_dim)
    attribution = (grad * values).sum(axis=1)        # Gradient x Input, per position

    tokens = [config.TOKENS[i] for i in token_seq_idx]
    return list(zip(tokens, attribution[: len(tokens)].tolist()))


def build_alert_report(
    session_id: str,
    risk: float,
    tier: str,
    predicted_class: str,
    rf_top_features: List[Tuple[str, float]],
    xgb_top_features: List[Tuple[str, float]],
    lstm_token_attrib: List[Tuple[str, float]],
    graph_path: List[Tuple[str, str, float]],
    sp_t: float,
    tc_t: float,
    recommended_action: str = "Review session; correlate source IP against recent alerts.",
) -> str:
    """Analyst-facing evidence chain, formatted to match the methodology's
    worked-example alert layout (Stage 13 output contract)."""
    lines = [
        f"Alert for Session {session_id} ({predicted_class})",
        f"  Risk_t: {risk:.3f} [{tier}]",
        f"  RF top features:   {', '.join(f'{f}({v:+.3f})' for f, v in rf_top_features)}",
        f"  XGBoost top features: {', '.join(f'{f}({v:+.3f})' for f, v in xgb_top_features)}",
        f"  LSTM token attributions: {', '.join(f'{t}({v:+.3f})' for t, v in lstm_token_attrib)}",
        f"  Sequential pattern prior SP_t: {sp_t:.3f}",
        f"  Temporal consistency TC_t: {tc_t:.3f}",
        "  Graph path: " + " -> ".join(
            [graph_path[0][0]] + [f"{d}(w={w:.2f})" for _, d, w in graph_path]
        ) if graph_path else "  Graph path: <single-event session>",
        f"  Recommended action: {recommended_action}",
    ]
    return "\n".join(lines)

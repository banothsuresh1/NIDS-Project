"""Stage 3: Feature Engineering & Group Assignment.

Four semantic, mostly non-overlapping groups (Review §7: domain-knowledge
grouping is scientifically defensible and preferred over RFE):
  Group A -- flow-statistical features        -> Random Forest
  Group B -- protocol / communication features -> shared (RF + XGBoost)
  Group C -- derived temporal features          -> LSTM session context
  Group D -- TCP behavioral flag counts          -> XGBoost

Feature selection within each group uses mutual information ranking
computed on the TRAINING split only (never on val/test).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from . import config

logger = logging.getLogger(__name__)

_NON_FEATURE_COLS = config.IDENTIFIER_COLUMNS | config.PROVENANCE_COLUMNS | {
    config.LABEL_COLUMN, "partition", "destination_ip", "dst_ip",
}


def candidate_feature_columns(df: pd.DataFrame) -> List[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in _NON_FEATURE_COLS]


def assign_feature_groups(columns: List[str]) -> Dict[str, List[str]]:
    """Keyword-match each candidate column into Group A/B/C/D.

    A column matching multiple groups' keywords is assigned to the first
    matching group in A, B, C, D order (Group B is explicitly allowed to be
    shared downstream via the RF/XGBoost consumption rule, not via
    duplicate membership here). Columns matching nothing fall back into
    Group A with a warning so nothing is silently dropped from the model.
    """
    groups: Dict[str, List[str]] = {"A": [], "B": [], "C": [], "D": []}
    unmatched: List[str] = []
    for col in columns:
        placed = False
        for g in ["A", "B", "C", "D"]:
            if any(kw in col for kw in config.FEATURE_GROUP_KEYWORDS[g]):
                groups[g].append(col)
                placed = True
                break
        if not placed:
            groups["A"].append(col)
            unmatched.append(col)
    if unmatched:
        logger.warning(
            "%d column(s) did not match any group keyword and were placed in "
            "Group A by default: %s", len(unmatched), unmatched,
        )
    for g, cols in groups.items():
        logger.info("Feature Group %s: %d columns", g, len(cols))
    return groups


def rank_by_mutual_information(
    X_train: pd.DataFrame, y_train: pd.Series, cols: List[str], top_k: Optional[int] = None,
    random_state: int = config.RANDOM_SEED,
) -> List[str]:
    """MI ranking fit on the training split only (Review: never refit on val/test)."""
    if not cols:
        return []
    mi = mutual_info_classif(X_train[cols], y_train, discrete_features=False, random_state=random_state)
    ranked = [c for _, c in sorted(zip(mi, cols), reverse=True)]
    return ranked[:top_k] if top_k else ranked


def build_model_feature_sets(
    X_train: pd.DataFrame, y_train: pd.Series, top_k: Optional[int] = config.MI_TOP_K,
) -> Dict[str, List[str]]:
    """Returns MI-ranked feature lists for RF (A+B), XGBoost (B+D) and LSTM context (C)."""
    groups = assign_feature_groups(list(X_train.columns))
    ranked = {g: rank_by_mutual_information(X_train, y_train, cols, top_k) for g, cols in groups.items()}
    return {
        "rf": ranked["A"] + ranked["B"],
        "xgb": ranked["B"] + ranked["D"],
        "lstm_context": ranked["C"],
        "groups": groups,
        "ranked_groups": ranked,
    }

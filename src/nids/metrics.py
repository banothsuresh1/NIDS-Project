"""Stage 24: Evaluation Metrics.

Primary: Macro-F1, per-class Precision/Recall/F1, False Positive Rate.
Secondary: PR-AUC per class, bootstrap 95% CIs, McNemar's test.
Accuracy is reported LAST / with caution -- BENIGN dominance (~80% of
CIC-IDS2017) makes raw accuracy an uninformative headline number
(Review §24).
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)

from . import config


def per_class_report(y_true: Sequence[str], y_pred: Sequence[str], labels: List[str] = config.CLASSES) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    df = pd.DataFrame({"class": labels, "precision": precision, "recall": recall, "f1": f1, "support": support})
    macro_row = pd.DataFrame({
        "class": ["MACRO_AVG"], "precision": [precision.mean()], "recall": [recall.mean()],
        "f1": [f1.mean()], "support": [support.sum()],
    })
    return pd.concat([df, macro_row], ignore_index=True)


def macro_f1(y_true, y_pred, labels: List[str] = config.CLASSES) -> float:
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def false_positive_rate(y_true: Sequence[str], y_pred: Sequence[str], negative_label: str = "BENIGN") -> float:
    """Treat BENIGN as the negative class and every attack class as positive.
    FPR = FP / (FP + TN), the headline operational KPI (Review §24)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    actual_negative = y_true == negative_label
    predicted_positive = y_pred != negative_label
    fp = np.sum(actual_negative & predicted_positive)
    tn = np.sum(actual_negative & ~predicted_positive)
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


def pr_auc_per_class(y_true_bin: np.ndarray, y_score: np.ndarray, labels: List[str] = config.CLASSES) -> pd.Series:
    """y_true_bin: one-hot (N,K); y_score: predicted probabilities (N,K)."""
    aucs = {}
    for i, cls in enumerate(labels):
        if y_true_bin[:, i].sum() == 0:
            aucs[cls] = float("nan")
            continue
        aucs[cls] = average_precision_score(y_true_bin[:, i], y_score[:, i])
    return pd.Series(aucs)


def bootstrap_ci(
    y_true: Sequence, y_pred: Sequence, metric_fn: Callable[[Sequence, Sequence], float],
    n_boot: int = 1000, alpha: float = 0.05, seed: int = config.RANDOM_SEED,
) -> Tuple[float, float, float]:
    """Returns (point_estimate, ci_lower, ci_upper) via case resampling."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    samples = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[b] = metric_fn(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def mcnemar_test(y_true: Sequence, y_pred_a: Sequence, y_pred_b: Sequence) -> Tuple[float, float]:
    """McNemar's test comparing two classifiers' correctness on the same
    sessions (Review §21/§26 G: required for the full system vs best
    baseline comparison). Returns (statistic, p_value)."""
    y_true = np.asarray(y_true)
    correct_a = np.asarray(y_pred_a) == y_true
    correct_b = np.asarray(y_pred_b) == y_true
    b = int(np.sum(correct_a & ~correct_b))  # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))  # a wrong, b right
    if b + c == 0:
        return 0.0, 1.0
    if b + c < 25:
        # exact binomial test for small discordant-pair counts
        p = stats.binomtest(min(b, c), b + c, 0.5).pvalue
        stat = float(min(b, c))
    else:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p = 1 - stats.chi2.cdf(stat, df=1)
    return float(stat), float(p)


def confusion(y_true, y_pred, labels: List[str] = config.CLASSES) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)

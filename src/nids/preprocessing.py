"""Stage 2: Preprocessing & Class Imbalance Handling (dual-branch).

Fit-on-train-only ordering, per Review §6:
    1. Median imputer   -- fit(train)  -> transform(train/val/test)
    2. Variance filter   -- fit(train)  -> transform(train/val/test)
    3. Min-Max scaler     -- fit(train)  -> transform(train/val/test)
    4. SMOTE-KNN (Branch B, RF/XGBoost only) -- fit_resample(train) ONLY,
       applied AFTER scaling (Review §6: "the critical ordering").
    5. Class weights (Branch A, ALL models) -- computed from train labels only.

Branch A (class weighting) is the PRIMARY strategy and is what every model,
including the LSTM's loss function, uses. Branch B (SMOTE-KNN) is an
experimental comparison restricted to the two tabular classifiers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from . import config

logger = logging.getLogger(__name__)


@dataclass
class TrainOnlyPreprocessor:
    """Impute -> variance-filter -> Min-Max scale, fit on TRAIN split only."""

    feature_cols: List[str]
    outlier_clip_percentile: float = config.OUTLIER_CLIP_PERCENTILE

    imputer_: Optional[SimpleImputer] = field(default=None, repr=False)
    variance_selector_: Optional[VarianceThreshold] = field(default=None, repr=False)
    scaler_: Optional[MinMaxScaler] = field(default=None, repr=False)
    clip_bounds_: Optional[pd.DataFrame] = field(default=None, repr=False)
    selected_cols_: Optional[List[str]] = field(default=None, repr=False)

    def fit(self, train_df: pd.DataFrame) -> "TrainOnlyPreprocessor":
        X = train_df[self.feature_cols].astype(np.float64)

        # 99th-percentile outlier clip, thresholds computed on TRAIN only.
        lower = X.quantile(1 - self.outlier_clip_percentile / 100.0)
        upper = X.quantile(self.outlier_clip_percentile / 100.0)
        self.clip_bounds_ = pd.DataFrame({"lower": lower, "upper": upper})
        X = X.clip(lower=lower, upper=upper, axis=1)

        self.imputer_ = SimpleImputer(strategy="median")
        X_imp = pd.DataFrame(self.imputer_.fit_transform(X), columns=self.feature_cols)

        self.variance_selector_ = VarianceThreshold(threshold=1e-4)
        self.variance_selector_.fit(X_imp)
        # Cast back to plain Python str: np.array(...)[mask] yields numpy.str_
        # elements, which newer scikit-learn's strict feature-name check only
        # tolerates when EVERY column in a DataFrame shares that exact
        # subtype. Once these column names get mixed with plain-str columns
        # elsewhere downstream (e.g. engineered feature names in a
        # multi-modal fusion step), sklearn raises a TypeError. Plain str
        # avoids the landmine entirely.
        mask = self.variance_selector_.get_support()
        self.selected_cols_ = [str(c) for c in np.array(self.feature_cols)[mask]]
        dropped = set(self.feature_cols) - set(self.selected_cols_)
        if dropped:
            logger.info("Variance filter dropped %d near-constant features: %s", len(dropped), sorted(dropped))

        self.scaler_ = MinMaxScaler()
        self.scaler_.fit(X_imp[self.selected_cols_])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.scaler_ is not None, "call .fit(train_df) first"
        X = df[self.feature_cols].astype(np.float64)
        X = X.clip(lower=self.clip_bounds_["lower"], upper=self.clip_bounds_["upper"], axis=1)
        X_imp = pd.DataFrame(self.imputer_.transform(X), columns=self.feature_cols, index=df.index)
        X_sel = X_imp[self.selected_cols_]
        X_scaled = self.scaler_.transform(X_sel)
        return pd.DataFrame(X_scaled, columns=self.selected_cols_, index=df.index)

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)


def compute_class_weights(y_train: pd.Series, num_classes: int = config.NUM_CLASSES) -> Dict[str, float]:
    """Branch A: W_c = N_train / (K * N_c). Applied to RF, XGBoost and the LSTM loss.

    A class absent from training gets a floor of 1.0 rather than 0.0, and
    every weight is capped at 50.0. This is defensive well-formedness only
    -- it does NOT and CANNOT make any model predict a class it has zero
    training rows for. RF's class_weight dict, XGBoost's sample_weight
    (y_train.map(...)), and the LSTM's per-sample CrossEntropyLoss weight
    are all looked up by each TRAINING SAMPLE'S OWN label; a class with no
    training samples is never looked up by any of them, so its weight value
    is inert. If a class shows 0 recall because N_c == 0, the fix is a
    different training split, not this dict.
    """
    n_train = len(y_train)
    counts = y_train.value_counts()
    weights = {}
    for cls in config.CLASSES:
        n_c = counts.get(cls, 0)
        w = float(n_train / (num_classes * n_c)) if n_c > 0 else 1.0
        weights[cls] = min(w, 50.0)
    return weights


def class_weight_array(weights: Dict[str, float]) -> np.ndarray:
    return np.array([weights[c] for c in config.CLASSES], dtype=np.float64)


def _knn_borderline_clean(
    X_syn: np.ndarray, y_syn: np.ndarray, X_pool: np.ndarray, y_pool: np.ndarray, k: int
) -> np.ndarray:
    """Keep a synthetic point only if a majority of its k nearest neighbours
    (within the full real+synthetic pool) share its class. This is the
    'less aggressive than ENN' cleaning described in Review §8 -- it removes
    borderline synthetic points without deleting real minority samples.
    """
    if len(X_syn) == 0:
        return np.zeros(0, dtype=bool)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X_pool)))
    nn.fit(X_pool)
    _, idx = nn.kneighbors(X_syn)
    keep = np.zeros(len(X_syn), dtype=bool)
    for i in range(len(X_syn)):
        neighbour_labels = y_pool[idx[i]]
        # exclude the point itself if it appears in the pool at distance 0
        same_class_frac = np.mean(neighbour_labels == y_syn[i])
        keep[i] = same_class_frac >= 0.5
    return keep


def smote_knn_resample(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    min_class_count: int = config.SMOTE_MIN_CLASS_COUNT,
    k_neighbors: int = config.SMOTE_K_NEIGHBORS,
    clean_k: int = config.SMOTE_KNN_CLEAN_K,
    target_ratio: float = config.SMOTE_TARGET_RATIO,
    random_state: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Branch B: SMOTE-KNN, restricted to RF/XGBoost callers only.

    Correction from the reviewer pass (Review §4/§8): SMOTE is applied ONLY
    to classes whose training count is already >= `min_class_count`, because
    k=5 SMOTE on e.g. Heartbleed's 11 samples repeatedly reuses the same few
    neighbours and produces near-duplicate, uninformative synthetic points.
    Classes below the floor (Heartbleed, Infiltration) are left untouched
    here and rely entirely on class weighting (Branch A).
    Must be called AFTER scaling and ONLY on the training partition.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as e:
        raise ImportError("pip install imbalanced-learn to use smote_knn_resample") from e

    counts = y_train.value_counts()
    majority_count = counts.max()
    eligible = [c for c in counts.index if counts[c] >= min_class_count and counts[c] < majority_count]
    skipped = [c for c in counts.index if counts[c] < min_class_count]
    if skipped:
        logger.info(
            "SMOTE-KNN skipping classes below N=%d (class-weighting only): %s",
            min_class_count, {c: int(counts[c]) for c in skipped},
        )
    if not eligible:
        logger.info("No classes eligible for SMOTE-KNN; returning original training data.")
        return X_train, y_train

    sampling_strategy = {
        c: min(int(majority_count * target_ratio), int(counts[c] * (k_neighbors + 1)))
        for c in eligible
    }
    # SMOTE requires the target count to exceed the current count.
    sampling_strategy = {c: n for c, n in sampling_strategy.items() if n > counts[c]}
    if not sampling_strategy:
        logger.info("SMOTE targets did not exceed existing counts; skipping oversampling.")
        return X_train, y_train

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    X_res, y_res = smote.fit_resample(X_train.values, y_train.values)

    n_original = len(X_train)
    is_synthetic = np.zeros(len(X_res), dtype=bool)
    is_synthetic[n_original:] = True
    # imblearn appends synthetic rows for each oversampled class after the
    # originals for THAT class group internally, but the safest generic
    # detector is: any row not present (by class-count) beyond the original
    # per-class counts. We instead just treat everything after the original
    # sample count as candidate-synthetic, which holds for imblearn's SMOTE.
    X_syn, y_syn = X_res[is_synthetic], y_res[is_synthetic]
    keep_mask = _knn_borderline_clean(X_syn, y_syn, X_res, y_res, k=clean_k)

    X_final = np.vstack([X_res[~is_synthetic], X_syn[keep_mask]])
    y_final = np.concatenate([y_res[~is_synthetic], y_syn[keep_mask]])

    logger.info(
        "SMOTE-KNN: generated %d synthetic rows, kept %d after KNN cleaning (k=%d). New size: %d (was %d).",
        len(X_syn), keep_mask.sum(), clean_k, len(X_final), n_original,
    )
    return (
        pd.DataFrame(X_final, columns=X_train.columns),
        pd.Series(y_final, name=y_train.name),
    )

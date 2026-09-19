"""Stage 8: Frequent & Sequential Pattern Mining.

FP-Growth mines UNORDERED co-occurrence patterns ("which tokens tend to
appear together in a session"); PrefixSpan mines ORDERED sequential
patterns under a temporal max_gap constraint ("in what order, with what
timing"). Review §14/§15 is explicit that these are NOT redundant and both
feed the sequential-pattern-prior SP_t used in Stage 7 fusion.

Both miners run on TRAINING sessions' behavioral token sequences ONLY, and
NEVER on SMOTE-KNN synthetic data (Review §8, §11: a synthetic flow has no
real timestamp, so it cannot be a real co-occurring or sequential event).
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# FP-Growth -- unordered co-occurrence patterns
# --------------------------------------------------------------------------
def mine_fp_growth(
    train_session_tokens: Dict[str, List[str]], min_support: float = config.FPGROWTH_MIN_SUPPORT
) -> pd.DataFrame:
    """train_session_tokens: session_id -> list of tokens observed in that
    session (repeats collapsed, since FP-Growth transactions are item SETS).

    Transaction = the complete set of unique behavioral tokens observed in
    one session (Review §14) -- NOT individual flows.
    """
    from mlxtend.frequent_patterns import fpgrowth
    from mlxtend.preprocessing import TransactionEncoder

    transactions = [sorted(set(toks)) for toks in train_session_tokens.values()]
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    onehot = pd.DataFrame(arr, columns=te.columns_)
    patterns = fpgrowth(onehot, min_support=min_support, use_colnames=True)
    patterns = patterns.sort_values("support", ascending=False).reset_index(drop=True)
    logger.info("FP-Growth mined %d frequent itemsets at min_support=%.2f", len(patterns), min_support)
    return patterns


def fp_growth_match_score(session_token_set: set, fp_patterns: pd.DataFrame) -> float:
    """Highest support among mined itemsets that are a subset of this session's tokens."""
    if fp_patterns.empty:
        return 0.0
    best = 0.0
    for itemset, support in zip(fp_patterns["itemsets"], fp_patterns["support"]):
        if itemset.issubset(session_token_set) and support > best:
            best = support
    return float(best)


# --------------------------------------------------------------------------
# PrefixSpan -- ordered, temporally gap-constrained sequential patterns
# --------------------------------------------------------------------------
def mine_prefixspan(
    train_session_seqs: Dict[str, List[Tuple[str, "pd.Timestamp"]]],
    min_support: float = config.PREFIXSPAN_MIN_SUPPORT,
    max_gap_seconds: float = config.PREFIXSPAN_MAX_GAP_SECONDS,
    max_pattern_len: int = 6,
    max_patterns_considered: int = 200,
) -> List[Tuple[List[str], float]]:
    """Two-pass mining: (1) PrefixSpan finds frequent ORDER-only candidate
    sequences ignoring time; (2) each candidate is re-scored on the actual
    session timestamps, keeping only matches that also satisfy max_gap
    between every consecutive pair of matched tokens -- this is the
    temporal max_gap constraint Review §15 requires PrefixSpan to enforce.
    """
    from prefixspan import PrefixSpan

    session_ids = list(train_session_seqs.keys())
    order_only_db = [[tok for tok, _ in train_session_seqs[sid]] for sid in session_ids]
    n_sessions = len(order_only_db)
    if n_sessions == 0:
        return []

    min_count = max(1, math.ceil(min_support * n_sessions))
    ps = PrefixSpan(order_only_db)
    ps.minlen = 2
    ps.maxlen = max_pattern_len
    raw = ps.frequent(min_count)  # list of (count, pattern)
    raw = sorted(raw, key=lambda cp: cp[0], reverse=True)[:max_patterns_considered]

    results: List[Tuple[List[str], float]] = []
    for _, pattern in raw:
        if len(pattern) < 2:
            continue
        temporally_valid_count = 0
        for sid in session_ids:
            if _matches_with_gap(pattern, train_session_seqs[sid], max_gap_seconds):
                temporally_valid_count += 1
        temporal_support = temporally_valid_count / n_sessions
        if temporal_support >= min_support:
            results.append((pattern, temporal_support))

    results.sort(key=lambda ps_: ps_[1], reverse=True)
    logger.info(
        "PrefixSpan: %d order-only candidates -> %d temporally-valid patterns "
        "(min_support=%.2f, max_gap=%.0fs)",
        len(raw), len(results), min_support, max_gap_seconds,
    )
    return results


def _matches_with_gap(pattern: Sequence[str], seq: List[Tuple[str, "pd.Timestamp"]], max_gap_seconds: float) -> bool:
    """Greedy left-to-right subsequence match: pattern must occur in order in
    `seq`, and consecutive MATCHED tokens' timestamps must be within
    max_gap_seconds of each other."""
    p_idx = 0
    last_time = None
    for tok, ts in seq:
        if p_idx >= len(pattern):
            break
        if tok == pattern[p_idx]:
            if last_time is not None:
                gap = (ts - last_time).total_seconds()
                if gap > max_gap_seconds:
                    # gap violated -- restart matching from this token as a
                    # fresh potential start of the pattern.
                    p_idx = 0
                    if tok == pattern[0]:
                        p_idx = 1
                        last_time = ts
                    else:
                        last_time = None
                    continue
            last_time = ts
            p_idx += 1
    return p_idx >= len(pattern)


def prefixspan_match_score(session_seq: List[Tuple[str, "pd.Timestamp"]], seq_patterns: List[Tuple[List[str], float]],
                            max_gap_seconds: float = config.PREFIXSPAN_MAX_GAP_SECONDS) -> float:
    best = 0.0
    for pattern, support in seq_patterns:
        if support > best and _matches_with_gap(pattern, session_seq, max_gap_seconds):
            best = support
    return float(best)


# --------------------------------------------------------------------------
# Combined sequential-pattern prior SP_t (Stage 7 fusion input)
# --------------------------------------------------------------------------
def compute_sp_t(
    session_seq: List[Tuple[str, "pd.Timestamp"]],
    fp_patterns: pd.DataFrame,
    seq_patterns: List[Tuple[List[str], float]],
    max_gap_seconds: float = config.PREFIXSPAN_MAX_GAP_SECONDS,
) -> float:
    token_set = {tok for tok, _ in session_seq}
    fp_score = fp_growth_match_score(token_set, fp_patterns)
    seq_score = prefixspan_match_score(session_seq, seq_patterns, max_gap_seconds)
    return float(np.clip(0.5 * fp_score + 0.5 * seq_score, 0.0, 1.0))

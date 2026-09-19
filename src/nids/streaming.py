"""Stage 12: Streaming Evaluation.

Simulates 60s tumbling windows with a 10s stride over the test partition.
Model weights (RF/XGBoost/LSTM) stay FROZEN -- this is streaming
*evaluation* of pre-trained static models, not online learning
(Review §19: never call this "online learning"; the only thing that may
adapt during the stream is the attack-state graph's EMA edge weights, which
is parameter-free incremental statistics, not model learning).

Design: this module is deliberately decoupled from the model stack. It
takes a `score_session_fn(session_id) -> risk_score` callback (built by
pipeline.py by closing over the already-fitted RF/XGBoost/LSTM, frozen
fusion weights, frozen attack-state graph and risk meta-learner) plus
session start/end metadata, and measures throughput + per-session
detection latency without knowing anything about how the score is computed.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


@dataclass
class StreamingResult:
    per_session: pd.DataFrame
    throughput_events_per_sec: float
    mean_latency_ms_per_event: float
    n_windows: int


def simulate_streaming(
    session_summary: pd.DataFrame,  # columns: session_id, start_time, end_time, n_flows, label
    score_session_fn: Callable[[str], float],
    window_seconds: float = config.STREAM_WINDOW_SECONDS,
    stride_seconds: float = config.STREAM_STRIDE_SECONDS,
    attack_threshold: float = config.RISK_THRESHOLD_ATTACK,
) -> StreamingResult:
    sessions = session_summary.sort_values("start_time").reset_index(drop=True)
    if sessions.empty:
        raise ValueError("session_summary is empty -- nothing to stream.")

    t_min, t_max = sessions["start_time"].min(), sessions["end_time"].max()
    window = pd.Timedelta(seconds=window_seconds)
    stride = pd.Timedelta(seconds=stride_seconds)

    results = []
    total_events = 0
    total_wall_time = 0.0

    scored_ids = set()
    window_start = t_min
    n_windows = 0
    while window_start <= t_max:
        window_end = window_start + window
        in_window = sessions[(sessions["start_time"] >= window_start) & (sessions["start_time"] < window_end)]
        for _, row in in_window.iterrows():
            sid = row["session_id"]
            if sid in scored_ids:
                continue
            scored_ids.add(sid)
            t0 = time.perf_counter()
            risk = score_session_fn(sid)
            elapsed = time.perf_counter() - t0
            total_wall_time += elapsed
            total_events += int(row["n_flows"])
            detection_latency = (
                (row["end_time"] - row["start_time"]).total_seconds() if risk >= attack_threshold else np.nan
            )
            results.append({
                "session_id": sid, "label": row.get("label"), "risk": risk,
                "alert": risk >= attack_threshold, "detection_latency_seconds": detection_latency,
                "window_start": window_start,
            })
        window_start += stride
        n_windows += 1

    # Score any sessions the sliding window never captured (e.g. sessions
    # starting after the last full window) so no test session is silently
    # dropped from the report.
    missed = sessions.loc[~sessions["session_id"].isin(scored_ids)]
    for _, row in missed.iterrows():
        t0 = time.perf_counter()
        risk = score_session_fn(row["session_id"])
        total_wall_time += time.perf_counter() - t0
        total_events += int(row["n_flows"])
        results.append({
            "session_id": row["session_id"], "label": row.get("label"), "risk": risk,
            "alert": risk >= attack_threshold,
            "detection_latency_seconds": (row["end_time"] - row["start_time"]).total_seconds() if risk >= attack_threshold else np.nan,
            "window_start": pd.NaT,
        })

    per_session = pd.DataFrame(results)
    throughput = total_events / total_wall_time if total_wall_time > 0 else float("inf")
    mean_latency_ms = (total_wall_time / max(total_events, 1)) * 1000.0

    logger.info(
        "Streaming evaluation: %d sessions across %d windows (%.0fs window / %.0fs stride). "
        "Throughput=%.1f events/sec, mean latency=%.3f ms/event.",
        len(per_session), n_windows, window_seconds, stride_seconds, throughput, mean_latency_ms,
    )
    return StreamingResult(per_session, throughput, mean_latency_ms, n_windows)


def early_detection_stats(result: StreamingResult) -> Dict[str, float]:
    attack_rows = result.per_session[result.per_session["detection_latency_seconds"].notna()]
    if attack_rows.empty:
        return {"mean_detection_latency_seconds": float("nan"), "n_detected": 0}
    return {
        "mean_detection_latency_seconds": float(attack_rows["detection_latency_seconds"].mean()),
        "median_detection_latency_seconds": float(attack_rows["detection_latency_seconds"].median()),
        "n_detected": int(len(attack_rows)),
    }

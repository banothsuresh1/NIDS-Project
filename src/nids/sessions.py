"""Stage 4: Bidirectional Session Reconstruction.

Groups individual CICFlowMeter flow records into complete network sessions
using a symmetric 5-tuple key so forward and reverse flows of the same
conversation map to the same session (Review §12: 5-tuple design is
correct). A session ends on the FIRST of:
  1. TCP FIN/RST flag observed on the flow that just completed
  2. tau = 60s idle gap before the next flow in the same key-group
  3. end of day / file boundary (sessions never span days)
  4. a hard cap of 3600s total session length (Review §12 addition, guards
     against runaway sessions during streaming)

Sessions are reconstructed independently per (day, session_key), so rule 3
is automatically satisfied by processing one day's flows at a time -- this
is also what keeps the chronological train/val/test split leakage-free
(Review §5: "no session may span train and test partitions").
"""
from __future__ import annotations

import ipaddress
import logging
from typing import List

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

_IP_CANDIDATES = [("source_ip", "destination_ip"), ("src_ip", "dst_ip")]
_PORT_CANDIDATES = [("source_port", "destination_port"), ("src_port", "dst_port")]


def _find_cols(df: pd.DataFrame, candidates):
    for pair in candidates:
        if all(c in df.columns for c in pair):
            return pair
    raise KeyError(f"None of the expected column pairs found: {candidates}")


def _ip_to_int(series: pd.Series) -> pd.Series:
    def conv(v):
        try:
            return int(ipaddress.ip_address(str(v).strip()))
        except ValueError:
            return abs(hash(str(v))) % (2 ** 32)
    return series.map(conv)


def build_session_key(df: pd.DataFrame) -> pd.Series:
    """Vectorized symmetric 5-tuple key: {min/max IP, protocol, min/max port}."""
    src_ip_col, dst_ip_col = _find_cols(df, _IP_CANDIDATES)
    src_port_col, dst_port_col = _find_cols(df, _PORT_CANDIDATES)

    src_ip = _ip_to_int(df[src_ip_col])
    dst_ip = _ip_to_int(df[dst_ip_col])
    ip_lo = np.minimum(src_ip, dst_ip)
    ip_hi = np.maximum(src_ip, dst_ip)

    src_port = df[src_port_col].astype(np.int64)
    dst_port = df[dst_port_col].astype(np.int64)
    port_lo = np.minimum(src_port, dst_port)
    port_hi = np.maximum(src_port, dst_port)

    protocol = df["protocol"].astype(str) if "protocol" in df.columns else "0"

    return (
        ip_lo.astype(str) + "_" + ip_hi.astype(str) + "_" + protocol.astype(str)
        + "_" + port_lo.astype(str) + "_" + port_hi.astype(str)
    )


def reconstruct_sessions(
    df: pd.DataFrame,
    timeout_seconds: float = config.SESSION_TIMEOUT_SECONDS,
    max_length_seconds: float = config.SESSION_MAX_LENGTH_SECONDS,
    terminate_on_fin_rst: bool = False,
) -> pd.DataFrame:
    """Adds 'session_key' and globally unique 'session_id' columns.

    Must be called on a single partition (train/val/test) at a time -- or
    on a single day at minimum -- so sessions never straddle a partition
    boundary. flow_duration is used to approximate each flow's END time
    (Timestamp + Flow Duration) so the next flow's idle gap is measured
    from when the previous flow actually finished, not just when it started.

    On `terminate_on_fin_rst`: the review's abstract termination checklist
    lists "TCP FIN/RST flag observed" as a session-ending condition, but its
    own SSH brute-force worked example (S_001) contradicts that rule -- 5 of
    the 8 flows in that ONE session end in RST (failed auth attempts), and
    the guide explicitly attributes the grouping to the tau timeout alone
    ("No inter-flow gap exceeds tau=60s, so all 8 records form one
    session"). Each CICFlowMeter row is already a fully-terminated atomic
    flow (CICFlowMeter itself closes flows on FIN/RST/its own timeout), so a
    row's RST flag describes how THAT flow ended, not whether the broader
    multi-flow session should stop growing. Defaulting to False reproduces
    the worked example; set True to opt into the review's stricter literal
    rule for a sensitivity comparison.
    """
    df = df.copy()
    df["session_key"] = build_session_key(df)

    duration_col = "flow_duration"
    if duration_col in df.columns:
        # CICFlowMeter reports Flow Duration in microseconds.
        flow_end = df["timestamp"] + pd.to_timedelta(df[duration_col].clip(lower=0), unit="us")
    else:
        flow_end = df["timestamp"]

    if terminate_on_fin_rst:
        fin = df["fin_flag_count"] if "fin_flag_count" in df.columns else 0
        rst = df["rst_flag_count"] if "rst_flag_count" in df.columns else 0
        terminated_prev = ((pd.Series(fin, index=df.index) > 0) | (pd.Series(rst, index=df.index) > 0))
    else:
        terminated_prev = pd.Series(False, index=df.index)

    sort_cols = ["day", "session_key", "timestamp"]
    order = df.sort_values(sort_cols).index
    df = df.loc[order].reset_index(drop=True)
    flow_end = flow_end.loc[order].reset_index(drop=True)
    terminated_prev = terminated_prev.loc[order].reset_index(drop=True)

    session_local_id = np.zeros(len(df), dtype=np.int64)
    group_keys = (df["day"].astype(str) + "||" + df["session_key"]).values
    ts = df["timestamp"].values
    fend = flow_end.values
    term = terminated_prev.values

    current_group = None
    local_counter = -1
    session_start_time = None
    prev_end_time = None
    prev_terminated = False

    for i in range(len(df)):
        g = group_keys[i]
        if g != current_group:
            current_group = g
            local_counter = 0
            session_start_time = ts[i]
            prev_end_time = fend[i]
            prev_terminated = term[i]
            session_local_id[i] = local_counter
            continue

        gap = (ts[i] - prev_end_time) / np.timedelta64(1, "s")
        elapsed = (ts[i] - session_start_time) / np.timedelta64(1, "s")
        starts_new = prev_terminated or (gap > timeout_seconds) or (elapsed > max_length_seconds)
        if starts_new:
            local_counter += 1
            session_start_time = ts[i]
        session_local_id[i] = local_counter
        prev_end_time = fend[i]
        prev_terminated = term[i]

    df["session_id"] = group_keys + "__s" + pd.Series(session_local_id).astype(str)

    n_sessions = df["session_id"].nunique()
    logger.info(
        "Session reconstruction: %d flows -> %d sessions (tau=%.0fs, cap=%.0fs). "
        "Mean flows/session=%.2f",
        len(df), n_sessions, timeout_seconds, max_length_seconds, len(df) / max(n_sessions, 1),
    )
    return df


def session_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per session: start/end time, flow count, dominant (majority) label."""
    agg = df.groupby("session_id").agg(
        day=("day", "first"),
        partition=("partition", "first"),
        n_flows=("session_id", "count"),
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        label=(config.LABEL_COLUMN, lambda s: s.value_counts().idxmax()),
    )
    agg["duration_seconds"] = (agg["end_time"] - agg["start_time"]).dt.total_seconds()
    return agg.reset_index()

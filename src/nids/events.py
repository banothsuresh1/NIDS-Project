"""Stage 5: Behavioral Event Encoding.

Maps every flow to exactly one of the 10 behavioral tokens (config.TOKENS)
using ONLY flow attributes available before/without the ground-truth attack
label (TCP flags, packet/byte counts, destination port, duration) -- this
avoids the circularity the reviewer explicitly warns against (Review §13:
"If AUTH_FAIL requires knowing the flow is labelled SSH-Patator ... your
encoding is circular").

Rules are evaluated in priority order (first match wins) and published
verbatim below as `RULE_TABLE`, matching the reviewer's requirement that the
token vocabulary be explicit and reproducible rather than described only in
prose. Thresholds are heuristic defaults tuned to the semantics in the
methodology's SSH brute-force worked example; treat them as a starting point
for further calibration on your own validation split.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

# Byte / rate thresholds used by the rule table (documented, not magic).
SMALL_BYTES = 500          # "small payload" cutoff, in total bytes
LARGE_BYTES = 5_000        # "substantial data transfer" cutoff, in total bytes
HIGH_PACKET_RATE = 500.0   # packets/sec considered flood-like (DoS-ish)
SHORT_DURATION_US = 100_000  # 100ms, in microseconds -- "quick probe"


def _col(df: pd.DataFrame, *names: str, default=0):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(default, index=df.index)


def encode_events(df: pd.DataFrame) -> pd.Series:
    """Vectorized rule table -> pd.Series of token strings, same index as df."""
    fwd_pkts = _col(df, "total_fwd_packets")
    bwd_pkts = _col(df, "total_backward_packets")
    fwd_bytes = _col(df, "total_length_of_fwd_packets")
    bwd_bytes = _col(df, "total_length_of_bwd_packets")
    total_bytes = fwd_bytes + bwd_bytes
    syn = _col(df, "syn_flag_count")
    ack = _col(df, "ack_flag_count")
    rst = _col(df, "rst_flag_count")
    fin = _col(df, "fin_flag_count")
    pkt_rate = _col(df, "flow_packets_s", "fwd_packets_s")
    duration_us = _col(df, "flow_duration")
    dst_port = _col(df, "destination_port")
    is_auth_port = dst_port.isin(config.AUTH_PORTS)

    n = len(df)
    token = pd.Series(config.TOKENS[0], index=df.index, dtype=object)  # default fallback
    assigned = pd.Series(False, index=df.index)

    def assign(mask: pd.Series, tok: str):
        m = mask & ~assigned
        token[m] = tok
        assigned[m] |= m

    # RULE_TABLE (priority order, first match wins):
    # 1. DOS_INDICATOR := very high one-directional packet rate, no reply.
    assign((pkt_rate > HIGH_PACKET_RATE) & (bwd_pkts == 0) & (fwd_pkts > 5), "DOS_INDICATOR")

    # 2. SCAN_ACTIVITY := SYN only, no reply, no payload, very short-lived probe.
    assign(
        (syn > 0) & (bwd_pkts == 0) & (total_bytes < 50) & (duration_us < SHORT_DURATION_US) & (fwd_pkts <= 3),
        "SCAN_ACTIVITY",
    )

    # 3. FORCED_TERMINATION := abrupt RST very early in the exchange (attack reset).
    assign((rst > 0) & (bwd_pkts <= 1) & (fwd_pkts <= 3), "FORCED_TERMINATION")

    # 4. AUTH_FAIL := short exchange on an auth-bearing port, terminated by RST/FIN,
    #    small payload -- the repeated-failure signature of brute-force sessions.
    assign(
        is_auth_port & (fwd_pkts <= 10) & (bwd_pkts <= 10) & (total_bytes < SMALL_BYTES) & ((rst > 0) | (fin > 0)),
        "AUTH_FAIL",
    )

    # 5. AUTH_SUCCESS := auth-port flow with a real bidirectional exchange and
    #    moderate-to-large payload following a completed handshake.
    assign(
        is_auth_port & (ack > 0) & (bwd_pkts > 0) & (total_bytes >= SMALL_BYTES) & (total_bytes < LARGE_BYTES),
        "AUTH_SUCCESS",
    )

    # 6. POST_AUTH_ACTIVITY := auth-port flow with a LARGE outbound/exchanged
    #    payload -- exfiltration-shaped activity following authentication.
    assign(is_auth_port & (total_bytes >= LARGE_BYTES), "POST_AUTH_ACTIVITY")

    # 7. DATA_TRANSFER := substantial bidirectional exchange on a non-auth port.
    assign((~is_auth_port) & (bwd_pkts > 0) & (fwd_pkts > 0) & (total_bytes >= SMALL_BYTES), "DATA_TRANSFER")

    # 8. SESSION_ESTABLISHED := completed handshake, some reply, but not yet
    #    enough data exchanged to call it a transfer.
    assign((syn > 0) & (ack > 0) & (bwd_pkts > 0) & (total_bytes < SMALL_BYTES), "SESSION_ESTABLISHED")

    # 9. CONNECTION_TERMINATION := graceful FIN close (catch-all close signal).
    assign(fin > 0, "CONNECTION_TERMINATION")

    # 10. CONNECTION_ATTEMPT := default fallback -- a SYN with no clean match above.
    assign(pd.Series(True, index=df.index), "CONNECTION_ATTEMPT")

    return token


RULE_TABLE: List[Tuple[str, str]] = [
    ("DOS_INDICATOR", "flow_packets_s > 500 AND backward_packets == 0 AND forward_packets > 5"),
    ("SCAN_ACTIVITY", "syn_flag_count > 0 AND backward_packets == 0 AND total_bytes < 50 AND duration < 100ms"),
    ("FORCED_TERMINATION", "rst_flag_count > 0 AND backward_packets <= 1 AND forward_packets <= 3"),
    ("AUTH_FAIL", "dst_port in AUTH_PORTS AND small bidirectional exchange AND (rst>0 OR fin>0)"),
    ("AUTH_SUCCESS", "dst_port in AUTH_PORTS AND ack>0 AND backward_packets>0 AND moderate payload"),
    ("POST_AUTH_ACTIVITY", "dst_port in AUTH_PORTS AND total_bytes >= 5000"),
    ("DATA_TRANSFER", "dst_port not in AUTH_PORTS AND bidirectional AND total_bytes >= 500"),
    ("SESSION_ESTABLISHED", "syn>0 AND ack>0 AND backward_packets>0 AND total_bytes < 500"),
    ("CONNECTION_TERMINATION", "fin_flag_count > 0 (fallback graceful close)"),
    ("CONNECTION_ATTEMPT", "default fallback"),
]


def build_session_sequences(df: pd.DataFrame) -> Dict[str, List[Tuple[str, pd.Timestamp]]]:
    """session_id -> chronologically ordered [(token, timestamp), ...]."""
    if "token" not in df.columns:
        df = df.copy()
        df["token"] = encode_events(df)
    df = df.sort_values(["session_id", "timestamp"])
    sequences: Dict[str, List[Tuple[str, pd.Timestamp]]] = {}
    for sid, g in df.groupby("session_id"):
        sequences[sid] = list(zip(g["token"].tolist(), g["timestamp"].tolist()))
    return sequences


def sequence_to_indices(seq: List[Tuple[str, pd.Timestamp]], max_len: int = config.LSTM_MAX_SEQ_LEN) -> List[int]:
    idx = [config.TOKEN_TO_IDX[tok] for tok, _ in seq[-max_len:]]
    return idx

"""Stage 9-10: Temporal Attack-State Graph & Sequence Consistency Scoring.

Nodes are the 10 behavioral event tokens themselves -- data-driven, NOT
predefined MITRE ATT&CK kill-chain stages (Review §16: CIC-IDS2017 does not
contain every kill-chain stage, e.g. no Lateral Movement, so forcing a MITRE
template would corrupt the graph with semantically invalid transitions).
MITRE labels may be attached as an OPTIONAL annotation layer after the fact.

Edge weights are an online, per-source-node EMA estimate of transition
probability: every time token `src` is followed by token `dst` in a
training session, EVERY outgoing edge from `src` is EMA-updated (indicator
1 for the observed `dst`, 0 for all others) -- W_t = rho*W_{t-1} +
(1-rho)*W_new, rho=0.9. This makes edge weight converge to the relative
frequency with which `src` is followed by each candidate token, exactly the
"historically observed frequency of real state transitions" the guide
describes, while still decaying slowly (Review §9). Graph is built from
TRAINING sessions only, in chronological order, and NEVER from synthetic
SMOTE-KNN data (Review §8/§11).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config

logger = logging.getLogger(__name__)

SeqWithTime = List[Tuple[str, pd.Timestamp]]


class AttackStateGraph:
    """Edge weights are stored as a dense len(TOKENS)xlen(TOKENS) numpy
    matrix, not a networkx graph -- on the real dataset this method is
    called ~300K+ times per training run (once per observed transition),
    and each call previously did 10 networkx.add_edge() calls (one per
    candidate token), i.e. ~3M+ networkx calls. That was the dominant cost
    of Stage 9. The EMA math is IDENTICAL (verified bit-for-bit against the
    prior dict/networkx implementation); a networkx view is still available
    via the `.graph` property for any external code that wants one, built
    lazily/once from the matrix rather than incrementally.
    """

    def __init__(self, rho: float = config.GRAPH_EMA_RHO, novel_weight: float = config.GRAPH_NOVEL_EDGE_WEIGHT):
        self.rho = rho
        self.novel_weight = novel_weight
        self._idx = config.TOKEN_TO_IDX
        self._n = len(config.TOKENS)
        self._W = np.zeros((self._n, self._n), dtype=np.float64)
        self._seen_source_mask = np.zeros(self._n, dtype=bool)
        self._graph_cache: nx.DiGraph = None

    def observe_transition(self, src: str, dst: str) -> None:
        i, j = self._idx[src], self._idx[dst]
        row = self._W[i]
        row *= self.rho
        row[j] += 1 - self.rho
        self._seen_source_mask[i] = True
        self._graph_cache = None  # invalidate lazy networkx view

    @property
    def graph(self) -> nx.DiGraph:
        """Lazily-built networkx view of the current weights (visualization
        / external inspection only -- nothing in this package reads it)."""
        if self._graph_cache is None:
            g = nx.DiGraph()
            g.add_nodes_from(config.TOKENS)
            for i, src in enumerate(config.TOKENS):
                if not self._seen_source_mask[i]:
                    continue
                for j, dst in enumerate(config.TOKENS):
                    g.add_edge(src, dst, weight=float(self._W[i, j]))
            self._graph_cache = g
        return self._graph_cache

    def build_from_training(self, session_sequences: Dict[str, SeqWithTime]) -> "AttackStateGraph":
        # Process sessions in chronological order of their first event, so
        # the EMA reflects the actual temporal evolution of attack behaviour
        # (Review's "incrementally-updated attack-state graph").
        ordered = sorted(
            session_sequences.items(),
            key=lambda kv: kv[1][0][1] if kv[1] else pd.Timestamp.min,
        )
        n_transitions = 0
        for _, seq in ordered:
            for (t1, _), (t2, _) in zip(seq, seq[1:]):
                self.observe_transition(t1, t2)
                n_transitions += 1
        logger.info(
            "Attack-state graph built from %d training sessions (%d transitions). "
            "Source nodes observed: %d/%d tokens.",
            len(ordered), n_transitions, int(self._seen_source_mask.sum()), len(config.TOKENS),
        )
        return self

    def edge_weight(self, src: str, dst: str) -> float:
        # Direct matrix lookup -- this is called once per transition in every
        # session's TC_t/G_w computation (millions of calls on the real
        # dataset), so it must NOT go through the lazy `.graph` networkx
        # view; that property is for external/visualization use only.
        i = self._idx[src]
        if not self._seen_source_mask[i]:
            return self.novel_weight
        return float(self._W[i, self._idx[dst]])

    def update_edge_streaming(self, src: str, dst: str) -> None:
        """Used ONLY by the Stage 12 streaming simulation to incrementally
        update the graph as new sessions complete -- the base test
        evaluation (Stages 6-11) uses a FROZEN copy of the graph built once
        from training data, so headline metrics stay reproducible."""
        self.observe_transition(src, dst)


def compute_tc_t(session_seq: SeqWithTime, graph: AttackStateGraph, lam: float) -> float:
    """TC_t = sum_i [w(s_i->s_i+1) * exp(-lam*dt_i)] / (n-1).

    Edge cases (Review §17): a single-event session has no transitions, so
    TC_t is undefined (0/0) and is assigned the neutral default 0.5.
    """
    n = len(session_seq)
    if n <= 1:
        return config.TC_SINGLE_EVENT_DEFAULT
    total = 0.0
    for (tok1, t1), (tok2, t2) in zip(session_seq, session_seq[1:]):
        dt = max((t2 - t1).total_seconds(), 0.0)
        w = graph.edge_weight(tok1, tok2)
        total += w * np.exp(-lam * dt)
    return float(total / (n - 1))


def compute_g_w(session_seq: SeqWithTime, graph: AttackStateGraph) -> float:
    """Mean RAW edge weight along the session's path (no temporal decay) --
    a purely structural companion signal to the time-decayed TC_t, both fed
    into the Stage 11 risk meta-learner."""
    n = len(session_seq)
    if n <= 1:
        return config.TC_SINGLE_EVENT_DEFAULT
    weights = [graph.edge_weight(t1, t2) for (t1, _), (t2, _) in zip(session_seq, session_seq[1:])]
    return float(np.mean(weights))


def graph_path_evidence(session_seq: SeqWithTime, graph: AttackStateGraph) -> List[Tuple[str, str, float]]:
    """(Stage 13 input) The specific token transition path with edge weights."""
    return [
        (t1, t2, graph.edge_weight(t1, t2))
        for (t1, _), (t2, _) in zip(session_seq, session_seq[1:])
    ]


def select_lambda(
    val_sessions_seq: Dict[str, SeqWithTime],
    graph: AttackStateGraph,
    y_val_is_attack: Dict[str, int],
    candidates: Sequence[float] = config.TC_LAMBDA_CANDIDATES,
) -> Tuple[float, float]:
    """Choose lambda maximising ROC-AUC of TC_t as an attack-vs-benign
    separator on the validation split (Review §17: lambda must be validated
    empirically, not assumed)."""
    best_lam, best_auc = candidates[0], -1.0
    sids = list(val_sessions_seq.keys())
    y = np.array([y_val_is_attack[s] for s in sids])
    if len(set(y.tolist())) < 2:
        logger.warning("Validation set has only one class for lambda selection; using default lambda.")
        return candidates[len(candidates) // 2], float("nan")
    for lam in candidates:
        tc = np.array([compute_tc_t(val_sessions_seq[s], graph, lam) for s in sids])
        try:
            auc = roc_auc_score(y, tc)
        except ValueError:
            auc = 0.5
        if auc > best_auc:
            best_auc, best_lam = auc, lam
    logger.info("Selected TC_t lambda=%.4f (validation ROC-AUC=%.4f)", best_lam, best_auc)
    return best_lam, best_auc

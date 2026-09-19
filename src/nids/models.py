"""Stage 6: Multi-Model Parallel Detection.

Three classifiers, each on its designated feature group, all outputting
calibrated probability vectors aligned to config.CLASSES (no hard-decision
labels are passed between stages -- Stage 6 output contract):

  RF       -- Group A+B tabular features, Branch A class weights (primary)
              or Branch B SMOTE-KNN augmented data (experimental comparison).
  XGBoost  -- Group B+D tabular features, same Branch A/B choice as RF.
  BiLSTM   -- session behavioral TOKEN SEQUENCES (Stage 5 output), trained
              with Branch A class weights in the loss ONLY -- it never sees
              SMOTE-KNN synthetic data (Review §8/§11: a synthetic flow has
              no real timestamp or TCP state, so it cannot be a valid
              behavioral token).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from . import config

logger = logging.getLogger(__name__)


def _align_proba(proba: np.ndarray, model_classes: Sequence, target_classes: List[str] = config.CLASSES) -> np.ndarray:
    """Reindex an sklearn/xgboost predict_proba matrix onto the full, fixed
    config.CLASSES column order, filling 0.0 for any class the model never
    saw during training (e.g. Heartbleed/Infiltration under the strict
    chronological split -- see the zero-training-count caveat surfaced by
    pipeline.check_zero_shot_classes).
    """
    out = np.zeros((proba.shape[0], len(target_classes)), dtype=np.float64)
    class_to_col = {c: i for i, c in enumerate(model_classes)}
    for j, cls in enumerate(target_classes):
        if cls in class_to_col:
            out[:, j] = proba[:, class_to_col[cls]]
    return out


# --------------------------------------------------------------------------
# Random Forest (Group A+B)
# --------------------------------------------------------------------------
def train_rf(X_train: pd.DataFrame, y_train: pd.Series, class_weights: Optional[Dict[str, float]] = None):
    params = dict(config.RF_PARAMS)
    if class_weights is not None:
        params["class_weight"] = class_weights
    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)
    return rf


def predict_proba_rf(rf, X) -> np.ndarray:
    return _align_proba(rf.predict_proba(X), rf.classes_)


# --------------------------------------------------------------------------
# XGBoost (Group B+D)
# --------------------------------------------------------------------------
def train_xgb(X_train: pd.DataFrame, y_train: pd.Series, class_weights: Optional[Dict[str, float]] = None):
    import xgboost as xgb

    present_classes = sorted(y_train.unique())
    label_to_idx = {c: i for i, c in enumerate(present_classes)}
    y_idx = y_train.map(label_to_idx).values

    sample_weight = None
    if class_weights is not None:
        sample_weight = y_train.map(class_weights).values.astype(np.float64)

    params = dict(config.XGB_PARAMS)
    params["num_class"] = len(present_classes)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_idx, sample_weight=sample_weight)
    model._nids_present_classes = present_classes  # stash for prediction alignment
    return model


def predict_proba_xgb(model, X) -> np.ndarray:
    proba = model.predict_proba(X)
    return _align_proba(proba, model._nids_present_classes)


# --------------------------------------------------------------------------
# Bidirectional LSTM (session token sequences)
# --------------------------------------------------------------------------
@dataclass
class LSTMConfig:
    vocab_size: int = len(config.TOKENS) + 1  # +1 for <PAD>
    pad_idx: int = len(config.TOKENS)
    embed_dim: int = config.LSTM_EMBED_DIM
    hidden_dim: int = config.LSTM_HIDDEN_DIM
    num_classes: int = config.NUM_CLASSES


class BiLSTMClassifier:
    """Thin, dependency-isolated wrapper around a torch BiLSTM so the rest of
    the pipeline never has to import torch directly. Import is deferred to
    __init__ so environments without torch can still use RF/XGBoost alone.
    """

    def __init__(self, cfg: LSTMConfig = LSTMConfig(), device: Optional[str] = None):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        class _Net(nn.Module):
            def __init__(self, c: LSTMConfig):
                super().__init__()
                self.embedding = nn.Embedding(c.vocab_size, c.embed_dim, padding_idx=c.pad_idx)
                self.lstm = nn.LSTM(c.embed_dim, c.hidden_dim, batch_first=True, bidirectional=True)
                self.fc = nn.Linear(c.hidden_dim * 2, c.num_classes)

            def forward(self, x, lengths):
                emb = self.embedding(x)
                packed = nn.utils.rnn.pack_padded_sequence(
                    emb, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                _, (h_n, _) = self.lstm(packed)
                # h_n: (num_directions, batch, hidden) -> concat fwd/bwd final states
                h_cat = torch.cat([h_n[-2], h_n[-1]], dim=1)
                return self.fc(h_cat)

        self.net = _Net(cfg).to(self.device)

    def _batchify(self, sequences: List[List[int]], batch_size: int):
        n = len(sequences)
        idx = np.arange(n)
        for start in range(0, n, batch_size):
            yield idx[start:start + batch_size]

    def _pad(self, seqs: List[List[int]]):
        torch = self.torch
        lengths = torch.tensor([max(len(s), 1) for s in seqs], dtype=torch.int64)
        maxlen = int(lengths.max().item())
        padded = torch.full((len(seqs), maxlen), self.cfg.pad_idx, dtype=torch.int64)
        for i, s in enumerate(seqs):
            s = s if len(s) > 0 else [self.cfg.pad_idx]
            padded[i, : len(s)] = torch.tensor(s, dtype=torch.int64)
        return padded.to(self.device), lengths.to(self.device)

    def fit(
        self,
        sequences: List[List[int]],
        labels: List[str],
        class_weights: Dict[str, float],
        epochs: int = config.LSTM_EPOCHS,
        batch_size: int = config.LSTM_BATCH_SIZE,
        lr: float = config.LSTM_LR,
        verbose: bool = True,
    ):
        torch = self.torch
        import torch.nn as nn

        y_idx = np.array([config.CLASS_TO_IDX[l] for l in labels], dtype=np.int64)
        weight_tensor = torch.tensor(
            [class_weights.get(c, 0.0) for c in config.CLASSES], dtype=torch.float32
        ).to(self.device)
        # classes absent from training get weight 0 above; CrossEntropyLoss
        # requires nonzero total contribution, so floor at a tiny epsilon.
        weight_tensor = torch.clamp(weight_tensor, min=1e-6)

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.net.train()
        n = len(sequences)
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            total_loss = 0.0
            for batch_idx in self._batchify(list(perm), batch_size):
                batch_seqs = [sequences[i] for i in batch_idx]
                batch_y = torch.tensor(y_idx[batch_idx], dtype=torch.int64).to(self.device)
                x, lengths = self._pad(batch_seqs)
                optimizer.zero_grad()
                logits = self.net(x, lengths)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_idx)
            if verbose:
                logger.info("LSTM epoch %d/%d - loss=%.4f", epoch + 1, epochs, total_loss / n)
        return self

    def predict_proba(self, sequences: List[List[int]], batch_size: int = config.LSTM_BATCH_SIZE) -> np.ndarray:
        torch = self.torch
        self.net.eval()
        out = np.zeros((len(sequences), config.NUM_CLASSES), dtype=np.float64)
        with torch.no_grad():
            for batch_idx in self._batchify(list(range(len(sequences))), batch_size):
                batch_seqs = [sequences[i] for i in batch_idx]
                x, lengths = self._pad(batch_seqs)
                logits = self.net(x, lengths)
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                out[batch_idx] = proba
        return out

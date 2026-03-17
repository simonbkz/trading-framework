"""
Transformer-based regime detector using PyTorch.

Uses a lightweight Transformer encoder over sliding windows of regime features
to classify market regimes. Captures temporal dependencies via self-attention,
which can model regime transitions better than per-bar classifiers.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from regimes.base_regime_model import BaseRegimeModel
from config.regimes import ALL_REGIMES
from features.regime_features import get_feature_columns
from utils.logger import get_logger

log = get_logger(__name__)

# Lazy-loaded torch references (set on first use)
_torch = None
_nn = None


def _ensure_torch():
    global _torch, _nn
    if _torch is None:
        import torch
        import torch.nn as nn
        _torch = torch
        _nn = nn


class TransformerRegimeModel(BaseRegimeModel):
    """
    Transformer encoder for regime classification.

    Takes a sliding window of regime features and outputs per-regime
    probabilities. Uses positional encoding and multi-head self-attention.
    """

    name = "transformer"
    FEATURE_COLS = get_feature_columns()

    def __init__(
        self,
        seq_len: int = 60,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        regimes: Optional[List[str]] = None,
    ):
        super().__init__(regimes)
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self._model = None
        self._scaler = None
        self._device = None
        self._n_features = None
        self._model_state = None  # for pickle

    def __getstate__(self):
        """Custom pickle: save torch model as state_dict CPU tensors."""
        state = self.__dict__.copy()
        if self._model is not None:
            state["_model_state"] = {
                k: v.cpu() for k, v in self._model.state_dict().items()
            }
        state["_model"] = None
        state["_device"] = None
        return state

    def __setstate__(self, state):
        """Custom unpickle: rebuild torch model from state_dict."""
        self.__dict__.update(state)
        if self._model_state is not None and self._n_features is not None:
            _ensure_torch()
            self._device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
            self._model = TransformerEncoderNet(
                n_features=self._n_features,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                n_classes=len(self.regimes),
                seq_len=self.seq_len,
                dropout=self.dropout,
            ).to(self._device)
            self._model.load_state_dict(self._model_state)
            self._model.eval()

    def fit(
        self,
        features: pd.DataFrame,
        labels: Optional[pd.Series] = None,
    ) -> "TransformerRegimeModel":
        _ensure_torch()
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler

        self._device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        log.info("TransformerRegime: training on %s", self._device)

        cols = [c for c in self.FEATURE_COLS if c in features.columns]
        if not cols:
            raise ValueError("No feature columns found. Run add_regime_features() first.")

        feat_df = features[cols].copy()
        if labels is not None:
            labels = labels.reindex(feat_df.index)

        valid_mask = feat_df.notna().all(axis=1)
        if labels is not None:
            valid_mask &= labels.notna()
        feat_df = feat_df[valid_mask]
        if labels is not None:
            labels = labels[valid_mask]

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(feat_df.values)
        self._n_features = X_scaled.shape[1]

        regime_to_idx = {r: i for i, r in enumerate(self.regimes)}
        if labels is not None:
            y_all = np.array([regime_to_idx.get(l, 0) for l in labels.values])
        else:
            y_all = np.zeros(len(X_scaled), dtype=int)

        X_windows, y_windows = self._create_windows(X_scaled, y_all)
        if len(X_windows) < 100:
            log.warning("TransformerRegime: only %d windows, may underfit", len(X_windows))

        split = int(0.9 * len(X_windows))
        X_train, X_val = X_windows[:split], X_windows[split:]
        y_train, y_val = y_windows[:split], y_windows[split:]

        train_ds = TensorDataset(
            _torch.FloatTensor(X_train).to(self._device),
            _torch.LongTensor(y_train).to(self._device),
        )
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        n_classes = len(self.regimes)
        self._model = TransformerEncoderNet(
            n_features=self._n_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            n_classes=n_classes,
            seq_len=self.seq_len,
            dropout=self.dropout,
        ).to(self._device)

        optimizer = _torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = _torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = _nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None
        patience = 10
        no_improve = 0
        val_acc = 0.0

        for epoch in range(self.epochs):
            self._model.train()
            train_loss = 0.0
            for xb, yb in train_dl:
                optimizer.zero_grad()
                logits = self._model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                _nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(xb)
            scheduler.step()
            train_loss /= len(X_train)

            self._model.eval()
            with _torch.no_grad():
                val_logits = self._model(_torch.FloatTensor(X_val).to(self._device))
                val_loss = criterion(val_logits, _torch.LongTensor(y_val).to(self._device)).item()
                val_preds = val_logits.argmax(dim=1).cpu().numpy()
                val_acc = (val_preds == y_val).mean()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 10 == 0:
                log.info(
                    "Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_acc=%.3f",
                    epoch + 1, self.epochs, train_loss, val_loss, val_acc,
                )

            if no_improve >= patience:
                log.info("Early stopping at epoch %d", epoch + 1)
                break

        if best_state is not None:
            self._model.load_state_dict(best_state)
            self._model.to(self._device)

        self._fitted = True
        log.info("TransformerRegime fitted: %d windows, %d features, val_acc=%.3f",
                 len(X_windows), self._n_features, val_acc)
        return self

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        _ensure_torch()
        self._check_fitted()

        cols = [c for c in self.FEATURE_COLS if c in features.columns]
        feat_df = features[cols].copy()
        feat_df = feat_df.ffill().fillna(0)
        X_scaled = self._scaler.transform(feat_df.values)

        n = len(X_scaled)
        n_feat = X_scaled.shape[1]
        proba = np.zeros((n, len(self.regimes)))

        self._model.eval()
        with _torch.no_grad():
            batch_windows = []
            batch_indices = []

            for i in range(n):
                if i < self.seq_len - 1:
                    window = np.zeros((self.seq_len, n_feat))
                    window[self.seq_len - i - 1:] = X_scaled[:i + 1]
                else:
                    window = X_scaled[i - self.seq_len + 1: i + 1]
                batch_windows.append(window)
                batch_indices.append(i)

                if len(batch_windows) >= 256 or i == n - 1:
                    xb = _torch.FloatTensor(np.array(batch_windows)).to(self._device)
                    logits = self._model(xb)
                    p = _torch.softmax(logits, dim=1).cpu().numpy()
                    for j, idx in enumerate(batch_indices):
                        proba[idx] = p[j]
                    batch_windows = []
                    batch_indices = []

        return self._to_prob_frame(proba, features.index, self.regimes)

    def _create_windows(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(X)
        windows = []
        labels = []
        for i in range(self.seq_len, n):
            windows.append(X[i - self.seq_len: i])
            labels.append(y[i])
        return np.array(windows), np.array(labels)


class PositionalEncoding:
    """Lazy-init wrapper — actual nn.Module created on first call."""
    _cls = None

    @classmethod
    def _get_cls(cls):
        if cls._cls is None:
            _ensure_torch()

            class _PE(_nn.Module):
                def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
                    super().__init__()
                    self.dropout = _nn.Dropout(p=dropout)
                    pe = _torch.zeros(max_len, d_model)
                    position = _torch.arange(0, max_len, dtype=_torch.float).unsqueeze(1)
                    div_term = _torch.exp(
                        _torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
                    )
                    pe[:, 0::2] = _torch.sin(position * div_term)
                    pe[:, 1::2] = _torch.cos(position * div_term)
                    pe = pe.unsqueeze(0)
                    self.register_buffer("pe", pe)

                def forward(self, x):
                    x = x + self.pe[:, :x.size(1), :]
                    return self.dropout(x)

            # Register at module level for pickling
            _PE.__qualname__ = "PositionalEncoding._PE"
            _PE.__module__ = __name__
            cls._cls = _PE
        return cls._cls

    def __class_getitem__(cls, item):
        return cls._get_cls()


class TransformerEncoderNet:
    """
    Lightweight Transformer encoder for regime classification.
    Wraps a dynamically-created nn.Module that is pickle-safe.
    """
    _cls = None

    @classmethod
    def _get_cls(cls):
        if cls._cls is None:
            _ensure_torch()
            PE = PositionalEncoding._get_cls()

            class _TEN(_nn.Module):
                def __init__(
                    self,
                    n_features: int,
                    d_model: int = 64,
                    n_heads: int = 4,
                    n_layers: int = 2,
                    n_classes: int = 8,
                    seq_len: int = 60,
                    dropout: float = 0.1,
                ):
                    super().__init__()
                    self.input_proj = _nn.Linear(n_features, d_model)
                    self.pos_encoding = PE(d_model, seq_len, dropout)
                    encoder_layer = _nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=n_heads,
                        dim_feedforward=d_model * 4,
                        dropout=dropout,
                        activation="gelu",
                        batch_first=True,
                    )
                    self.encoder = _nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                    self.classifier = _nn.Sequential(
                        _nn.LayerNorm(d_model),
                        _nn.Linear(d_model, d_model),
                        _nn.GELU(),
                        _nn.Dropout(dropout),
                        _nn.Linear(d_model, n_classes),
                    )

                def forward(self, x):
                    x = self.input_proj(x)
                    x = self.pos_encoding(x)
                    x = self.encoder(x)
                    x = x[:, -1, :]
                    return self.classifier(x)

            _TEN.__qualname__ = "TransformerEncoderNet._TEN"
            _TEN.__module__ = __name__
            cls._cls = _TEN
        return cls._cls

    def __new__(cls, *args, **kwargs):
        net_cls = cls._get_cls()
        return net_cls(*args, **kwargs)

from __future__ import annotations

from typing import Literal

import numpy as np

TargetName = Literal["remaining_reasoning_tokens_log1p", "remaining_steps_log1p", "remaining_steps_linear"]


def flatten_features(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim >= 3:
        x = x.reshape(x.shape[0], -1)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("Features must be finite non-empty [N,F]")
    return x


def transform_target(y, target):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Targets must be non-negative finite vector")
    return np.log1p(y) if target.endswith("_log1p") else y.copy()


def inverse_target(y, target):
    return (
        np.expm1(np.asarray(y, dtype=np.float64))
        if target.endswith("_log1p")
        else np.asarray(y, dtype=np.float64)
    )


class Standardizer:
    def fit(self, x):
        self.mean = x.mean(0)
        self.scale = np.where(x.std(0) < 1e-8, 1.0, x.std(0))
        return self

    def transform(self, x):
        return (x - self.mean) / self.scale


class RidgeEstimator:
    name = "ridge"

    def __init__(self, alpha=1.0, standardize=True):
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = float(alpha)
        self.scaler = Standardizer() if standardize else None
        self.weights = None
        self.input_dim = None

    def fit(self, x, y):
        x = flatten_features(x)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1 or len(y) != len(x) or not np.isfinite(y).all():
            raise ValueError("Features and targets incompatible")
        self.input_dim = x.shape[1]
        if self.scaler:
            self.scaler.fit(x)
            x = self.scaler.transform(x)
        z = np.c_[x, np.ones(len(x))]
        p = np.eye(z.shape[1]) * self.alpha
        p[-1, -1] = 0
        self.weights = np.linalg.solve(z.T @ z + p, z.T @ y)
        return self

    def predict(self, x):
        if self.weights is None:
            raise RuntimeError("Estimator not fitted")
        x = flatten_features(x)
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {x.shape[1]}")
        if self.scaler:
            x = self.scaler.transform(x)
        return np.c_[x, np.ones(len(x))] @ self.weights


def _mlp(dim, hid, drop, seed, zero=False):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    m = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hid),
        nn.GELU(),
        nn.Dropout(drop),
        nn.Linear(hid, hid),
        nn.GELU(),
        nn.Dropout(drop),
        nn.Linear(hid, 1),
    )
    if zero:
        nn.init.zeros_(m[-1].weight)
        nn.init.zeros_(m[-1].bias)
    return m


class _Torch:
    def _check(self, x):
        x = flatten_features(x)
        if self.input_dim is not None and x.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {x.shape[1]}")
        return x


class FFNMLEstimator(_Torch):
    name = "ffn_mlp"

    def __init__(
        self, hidden_dim=256, dropout=0.0, epochs=100, learning_rate=1e-3, weight_decay=1e-4, seed=42
    ):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.input_dim = None

    def fit(self, x, y):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        x = flatten_features(x)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 1 or len(y) != len(x) or not np.isfinite(y).all():
            raise ValueError("Features and targets incompatible")
        self.input_dim = x.shape[1]
        self._module = _mlp(self.input_dim, self.hidden_dim, self.dropout, self.seed)
        dl = DataLoader(
            TensorDataset(torch.from_numpy(x.astype("float32")), torch.from_numpy(y[:, None])),
            batch_size=min(256, len(x)),
            shuffle=True,
        )
        opt = torch.optim.AdamW(
            self._module.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self._module.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad(set_to_none=True)
                loss = ((self._module(xb) - yb) ** 2).mean()
                loss.backward()
                opt.step()
        return self

    def predict(self, x):
        import torch

        x = self._check(x)
        self._module.eval()
        with torch.inference_mode():
            return self._module(torch.from_numpy(x.astype("float32"))).squeeze(-1).cpu().numpy().astype(float)


class RidgePlusFFNEstimator(_Torch):
    name = "ridge_plus_ffn"

    def __init__(
        self,
        alpha=1.0,
        hidden_dim=256,
        dropout=0.0,
        epochs=100,
        learning_rate=1e-3,
        weight_decay=1e-4,
        seed=42,
    ):
        self.ridge = RidgeEstimator(alpha)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.input_dim = None

    def fit(self, x, y):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        x = flatten_features(x)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 1 or len(y) != len(x) or not np.isfinite(y).all():
            raise ValueError("Features and targets incompatible")
        self.ridge.fit(x, y)
        base = self.ridge.predict(x).astype("float32")
        self.input_dim = x.shape[1]
        self._module = _mlp(self.input_dim, self.hidden_dim, self.dropout, self.seed, True)
        dl = DataLoader(
            TensorDataset(torch.from_numpy(x.astype("float32")), torch.from_numpy((y - base)[:, None])),
            batch_size=min(256, len(x)),
            shuffle=True,
        )
        opt = torch.optim.AdamW(
            self._module.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self._module.train()
        for _ in range(self.epochs):
            for xb, rb in dl:
                opt.zero_grad(set_to_none=True)
                loss = ((self._module(xb) - rb) ** 2).mean()
                loss.backward()
                opt.step()
        return self

    def predict(self, x):
        import torch

        x = self._check(x)
        self._module.eval()
        with torch.inference_mode():
            r = self._module(torch.from_numpy(x.astype("float32"))).squeeze(-1).cpu().numpy()
        return self.ridge.predict(x) + r.astype(float)


def build_estimator(kind, **kwargs):
    return {"ridge": RidgeEstimator, "ffn_mlp": FFNMLEstimator, "ridge_plus_ffn": RidgePlusFFNEstimator}[
        kind
    ](**kwargs)

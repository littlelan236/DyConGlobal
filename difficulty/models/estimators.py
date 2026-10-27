from __future__ import annotations

from typing import Literal

import numpy as np

TargetName = Literal["remaining_reasoning_tokens_log1p", "remaining_steps_log1p", "remaining_steps_linear", "submission_remaining_steps_fraction", "submission_remaining_reasoning_tokens_fraction"]


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


def _mlp(dim, hid, drop, seed, zero=False, *, hidden_dims=None, activation='gelu', normalization='layernorm'):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    dims = [hid, hid] if hidden_dims is None else list(hidden_dims)
    if not dims or any(type(d) is not int or d <= 0 for d in dims):
        raise ValueError('hidden_dims must be a nonempty list of positive integers')
    activations = {'gelu': nn.GELU, 'relu': nn.ReLU, 'silu': nn.SiLU}
    if activation not in activations or normalization not in {'layernorm', 'none'}:
        raise ValueError('Unsupported MLP activation/normalization')
    # Identity keeps parameter numbering identical to the legacy two-block layout.
    blocks = [nn.LayerNorm(dim) if normalization == 'layernorm' else nn.Identity()]
    width = dim
    for hidden in dims:
        blocks.extend([nn.Linear(width, hidden), activations[activation](), nn.Dropout(drop)])
        width = hidden
    blocks.append(nn.Linear(width, 1))
    m = nn.Sequential(*blocks)
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


def resolve_device(device):
    """Resolve a requested training device without silent CUDA fallback."""
    import torch

    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    resolved = torch.device(device)
    if resolved.type == 'cuda':
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(f'Requested CUDA device is unavailable: {device}')
        return torch.device('cuda', index)
    if resolved.type != 'cpu':
        raise ValueError('MLP supports only CPU or CUDA')
    return resolved


class FFNMLEstimator(_Torch):
    name = "ffn_mlp"

    def __init__(
        self, hidden_dim=256, dropout=0.0, epochs=100, learning_rate=1e-3, weight_decay=1e-4, seed=42,
        *, device='cpu', batch_size=256, patience=None, min_delta=0.0, scheduler=None,
        hidden_dims=None, activation='gelu', normalization='layernorm', optimizer=None
    ):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.input_dim = None
        self.device = device
        self.batch_size = batch_size
        self.patience = patience
        self.min_delta = min_delta
        self.scheduler_config = scheduler or {'enabled': False}
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.normalization = normalization
        self.optimizer_config = optimizer

    def fit(self, x, y, *, validation_data=None, epoch_callback=None):
        import time
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        x = flatten_features(x)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 1 or len(y) != len(x) or not np.isfinite(y).all():
            raise ValueError("Features and targets incompatible")
        if (self.epochs <= 0 or self.batch_size <= 0 or self.hidden_dim <= 0
                or not 0 <= self.dropout < 1 or self.learning_rate <= 0 or self.weight_decay < 0
                or self.min_delta < 0 or (self.patience is not None and self.patience <= 0)
                or not np.isfinite([self.learning_rate, self.weight_decay, self.min_delta]).all()):
            raise ValueError('Invalid MLP training parameters')
        self.input_dim = x.shape[1]
        self.device_ = str(resolve_device(self.device))
        self._module = self._make_module().to(self.device_)
        validation = None
        if validation_data is not None:
            xv, yv = validation_data
            xv, yv = self._check(xv).astype('float32'), np.asarray(yv, dtype=np.float32)
            if yv.ndim != 1 or len(yv) != len(xv) or not np.isfinite(yv).all():
                raise ValueError('Validation features and targets incompatible')
            validation = (xv, yv)
        if self.patience is not None and validation is None:
            raise ValueError('Early stopping requires validation data')
        generator = torch.Generator().manual_seed(self.seed) if validation is not None else None
        dl = DataLoader(
            TensorDataset(torch.from_numpy(x.astype("float32")),
                          torch.from_numpy(self._training_targets(x, y)[:, None])),
            batch_size=min(self.batch_size, len(x)), shuffle=True, generator=generator,
        )
        from difficulty.training.config import Optimizer
        options = Optimizer.model_validate(self.optimizer_config or {
            'kind': 'adamw', 'learning_rate': self.learning_rate, 'weight_decay': self.weight_decay})
        opt = options.build(self._module.parameters())
        scheduler = None
        if self.scheduler_config.get('enabled'):
            if self.scheduler_config.get('kind') != 'reduce_on_plateau':
                raise ValueError('Unsupported MLP scheduler')
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', factor=self.scheduler_config['factor'],
                patience=self.scheduler_config['patience'], threshold=self.scheduler_config['threshold'],
                min_lr=self.scheduler_config['min_lr'],
            )
        self.history_, self.best_epoch_, self.best_eval_loss_ = [], None, None
        self.finite_gradients_ = False
        best_state, stale, stopping_best = None, 0, float('inf')
        for epoch in range(1, self.epochs + 1):
            started = time.monotonic()
            self._module.train()
            total = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(self.device_), yb.to(self.device_)
                opt.zero_grad(set_to_none=True)
                loss = ((self._module(xb) - yb) ** 2).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite MLP training loss')
                loss.backward()
                if not torch.stack([torch.isfinite(p.grad).all() for p in self._module.parameters()
                                    if p.grad is not None]).all():
                    raise ValueError('Nonfinite MLP gradients')
                self.finite_gradients_ = True
                opt.step()
                total += float(loss.detach()) * len(xb)
            if not torch.stack([torch.isfinite(p).all() for p in self._module.parameters()]).all():
                raise ValueError('Nonfinite MLP parameters')
            eval_loss = self._validation_loss(*validation) if validation is not None else None
            if eval_loss is not None:
                if not np.isfinite(eval_loss):
                    raise ValueError('Nonfinite MLP validation loss')
                if self.best_eval_loss_ is None or eval_loss < self.best_eval_loss_:
                    self.best_epoch_, self.best_eval_loss_ = epoch, eval_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self._module.state_dict().items()}
                if eval_loss < stopping_best - self.min_delta:
                    stopping_best, stale = eval_loss, 0
                else:
                    stale += 1
            row = {'epoch': epoch, 'max_epochs': self.epochs, 'train_loss': total / len(x),
                   'eval_loss': eval_loss, 'best_epoch': self.best_epoch_,
                   'best_eval_loss': self.best_eval_loss_, 'device': self.device_,
                   'learning_rate': float(opt.param_groups[0]['lr']),
                   'epoch_seconds': time.monotonic() - started}
            self.history_.append(row)
            if epoch_callback is not None:
                epoch_callback(dict(row))
            if scheduler is not None and eval_loss is not None:
                scheduler.step(eval_loss)
            if self.patience is not None and stale >= self.patience:
                break
        self.epochs_trained_ = len(self.history_)
        if validation is not None:
            if best_state is None:
                raise ValueError('No finite validation checkpoint')
            self._module.load_state_dict(best_state)
        return self

    def _make_module(self):
        return _mlp(self.input_dim, self.hidden_dim, self.dropout, self.seed,
                    hidden_dims=self.hidden_dims, activation=self.activation, normalization=self.normalization)

    def _training_targets(self, x, y):
        return y

    def _validation_loss(self, x, y):
        prediction = self.predict(x)
        return float(np.mean((prediction - y) ** 2))

    def predict(self, x):
        import torch

        if not hasattr(self, '_module'):
            raise RuntimeError('Estimator not fitted')
        x = self._check(x).astype('float32')
        self._module.eval()
        device = next(self._module.parameters()).device
        with torch.inference_mode():
            batches = [self._module(torch.from_numpy(x[i:i+self.batch_size]).to(device))
                       .squeeze(-1).cpu().numpy() for i in range(0, len(x), self.batch_size)]
        result = np.concatenate(batches).astype(float)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite MLP prediction')
        return result


class FrozenRidgePlusFFNEstimator(FFNMLEstimator):
    """Train only a zero-initialized FFN residual over already fitted Ridge arrays."""

    name = 'ridge_plus_ffn'

    def __init__(self, ridge_model, **kwargs):
        super().__init__(**kwargs)
        self.ridge_model = {k: np.asarray(v, dtype=np.float64).copy() for k, v in ridge_model.items()}
        width = self.ridge_model['coef'].shape
        if (len(width) != 1 or not width[0] or self.ridge_model['intercept'].shape != ()
                or any(self.ridge_model[k].shape != width for k in ('feature_mean', 'feature_scale'))
                or not all(np.isfinite(v).all() for v in self.ridge_model.values())
                or (self.ridge_model['feature_scale'] <= 0).any()):
            raise ValueError('Invalid frozen Ridge parameters')
        for value in self.ridge_model.values():
            value.setflags(write=False)

    def _make_module(self):
        if len(self.ridge_model['coef']) != self.input_dim:
            raise ValueError('Frozen Ridge input width mismatch')
        return _mlp(self.input_dim, self.hidden_dim, self.dropout, self.seed, zero=True,
                    hidden_dims=self.hidden_dims, activation=self.activation, normalization=self.normalization)

    def predict_base(self, x):
        from difficulty.models.ridge_grid import predict_model

        x = self._check(x)
        if x.shape[1] != len(self.ridge_model['coef']):
            raise ValueError('Frozen Ridge input width mismatch')
        return predict_model(self.ridge_model, x)

    def _training_targets(self, x, y):
        return (y - self.predict_base(x)).astype('float32')

    def predict(self, x):
        return self.predict_base(x) + super().predict(x)


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

from difficulty.storage import difficulty_path
"""Strict standalone training configuration and legacy adapter."""
from pathlib import Path
from typing import Literal
import warnings

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

from core.io import identifier, read_json, sha256
from difficulty.probe_config import MLPProbe, MLPScheduler, RidgeMLPProbe, RidgeProbe, TARGETS
from difficulty.datasets.identity import task_key


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class Optimizer(Strict):
    kind: Literal['adamw', 'adam', 'sgd'] = 'adamw'
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    betas: list[float] | None = None
    eps: float | None = Field(default=None, gt=0)
    momentum: float | None = Field(default=None, ge=0, lt=1)

    @model_validator(mode='after')
    def valid(self):
        if self.kind == 'sgd' and (self.betas is not None or self.eps is not None):
            raise ValueError('SGD does not accept betas/eps')
        if self.kind != 'sgd' and self.momentum is not None:
            raise ValueError('Adam/AdamW do not accept momentum')
        if self.betas is not None and (len(self.betas) != 2 or any(not 0 <= b < 1 for b in self.betas)):
            raise ValueError('betas must contain two values in [0,1)')
        return self

    def build(self, parameters):
        import torch
        kwargs = dict(lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.kind == 'sgd':
            kwargs['momentum'] = self.momentum or 0.0
        else:
            kwargs.update(betas=tuple(self.betas or [0.9, 0.999]), eps=self.eps or 1e-8)
        return {'adamw': torch.optim.AdamW, 'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}[self.kind](parameters, **kwargs)


class Run(Strict):
    attempt_id: str = 'auto'
    mode: Literal['dry-run', 'check-data', 'fit'] = 'fit'

    @model_validator(mode='after')
    def valid(self):
        identifier(self.attempt_id)
        return self


class Split(Strict):
    id: str | None = None
    strategy: Literal['event_random_v1', 'task_random_v1'] | None = None
    seed: int | None = Field(default=None, ge=0, lt=2**32)
    validation_fraction: float | None = Field(default=None, gt=0, lt=1)
    test_fraction: float | None = Field(default=None, gt=0, lt=1)

    @model_validator(mode='after')
    def valid(self):
        if self.id is not None:
            identifier(self.id)
            if any(getattr(self, k) is not None for k in ('strategy', 'seed', 'validation_fraction', 'test_fraction')):
                raise ValueError('Existing split ID and new split parameters are mutually exclusive')
        elif ((self.validation_fraction or 0.15) + (self.test_fraction or 0.15)) >= 1:
            raise ValueError('Split fractions must leave training samples')
        return self


class Data(Strict):
    dataset_id: str
    expected_receipt_sha256: str | None = Field(default=None, pattern='^[0-9a-f]{64}$')
    split: Split = Field(default_factory=Split)
    targets: list[str]

    @model_validator(mode='after')
    def valid(self):
        identifier(self.dataset_id)
        if not self.targets or len(set(self.targets)) != len(self.targets) or not set(self.targets) <= set(TARGETS):
            raise ValueError('Targets must be unique supported submission regression targets')
        return self


class Features(Strict):
    public_site: int = Field(default=3, ge=0, lt=32)


class MLP(Strict):
    hidden_dims: list[int] = Field(default_factory=lambda: [1024, 1024])
    activation: Literal['gelu', 'relu', 'silu'] = 'gelu'
    normalization: Literal['layernorm', 'none'] = 'layernorm'
    dropout: float = Field(default=0.1, ge=0, lt=1)

    @model_validator(mode='after')
    def valid(self):
        if not self.hidden_dims or any(type(d) is not int or d <= 0 for d in self.hidden_dims):
            raise ValueError('hidden_dims must be nonempty positive integers')
        return self


class Ridge(Strict):
    refit_train_validation: bool = False
    plots_enabled: bool = True
    layers: list[int] | None = None
    alphas: list[float] = Field(default_factory=lambda: [1.0])
    standardize: Literal[True] = True
    blas_threads: int = Field(default=4, gt=0, le=32)


class Model(Strict):
    kind: Literal['ridge', 'ffn_mlp', 'ridge_plus_ffn'] = 'ffn_mlp'
    mlp: MLP | None = None
    ridge: Ridge | None = None
    ridge_publications: dict[str, str] | None = None

    @model_validator(mode='after')
    def valid(self):
        if self.kind == 'ridge':
            if self.mlp is not None or self.ridge_publications is not None:
                raise ValueError('Ridge does not accept mlp or ridge_publications')
        elif self.ridge is not None:
            raise ValueError('MLP variants do not refit Ridge; use ridge_publications for a frozen base')
        if (self.kind == 'ridge_plus_ffn') != (self.ridge_publications is not None):
            raise ValueError('Only Ridge+MLP requires ridge_publications')
        return self


class EarlyStopping(Strict):
    patience: int = Field(default=100, gt=0)
    min_delta: float = Field(default=0.0, ge=0)


class Training(Strict):
    epochs: int = Field(default=100, gt=0)
    batch_size: int = Field(default=256, gt=0)
    seed: int = Field(default=42, ge=0, lt=2**32)
    device: str = 'auto'
    torch_threads: int = Field(default=4, gt=0, le=32)
    loss: Literal['mse'] = 'mse'
    optimizer: Optimizer = Field(default_factory=Optimizer)
    early_stopping: EarlyStopping = Field(default_factory=EarlyStopping)
    scheduler: MLPScheduler = Field(default_factory=MLPScheduler)


class TensorBoard(Strict):
    enabled: bool = True
    host: Literal['127.0.0.1', '0.0.0.0'] = '127.0.0.1'
    port: int = Field(default=6006, ge=1024, le=65535)
    flush_secs: int = Field(default=5, ge=1, le=60)
    reload_interval: int = Field(default=5, ge=1, le=60)
    keep_after_training: bool = True


class TrainingConfig(Strict):
    schema_version: Literal[2] = 2
    run: Run = Field(default_factory=Run)
    data: Data
    features: Features = Field(default_factory=Features)
    model: Model = Field(default_factory=Model)
    training: Training | None = None
    tensorboard: TensorBoard = Field(default_factory=TensorBoard)

    @model_validator(mode='after')
    def valid(self):
        if self.model.kind == 'ridge' and self.training is not None:
            raise ValueError('Closed-form Ridge does not accept iterative training/optimizer parameters')
        if self.model.ridge_publications is not None and set(self.model.ridge_publications) != set(self.data.targets):
            raise ValueError('Ridge publications must bind every configured target')
        return self


def read_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('Training config must be a mapping')
    if 'schema_version' not in raw:
        if set(raw) - {'mlp_probe', 'ridge_mlp_probe'} or 'mlp_probe' not in raw:
            raise ValueError('Unknown or mixed legacy training config')
        warnings.warn('Legacy training YAML: migrate to schema_version: 2', stacklevel=2)
        MLPProbe.model_validate(raw['mlp_probe'])
        if raw.get('ridge_mlp_probe'):
            RidgeMLPProbe.model_validate({**raw['mlp_probe'], **raw['ridge_mlp_probe']})
        return raw, None
    return raw, TrainingConfig.model_validate(raw)


def resolve_settings(root, config):
    """Resolve dataset identity without ever consulting root-config training defaults."""
    root = Path(root)
    source = difficulty_path(root, 'datasets', config.data.dataset_id)
    digest = sha256(source / 'receipt.json')
    if config.data.expected_receipt_sha256 and config.data.expected_receipt_sha256 != digest:
        raise ValueError('Expected dataset receipt SHA mismatch')
    rows = read_json(source / 'rollouts.json')['rollouts']
    common = dict(dataset_id=config.data.dataset_id, dataset_receipt_sha256=digest,
                  expected_tasks=len({task_key(r) for r in rows}),
                  expected_rollouts=len(rows), expected_samples=sum(bool(s.strip()) for s in
                      (source / 'samples.jsonl').read_text().splitlines()), targets=config.data.targets)
    split = config.data.split
    if split.id:
        frozen = read_json(source / 'splits' / split.id / 'split.json')
        common.update(split_id=split.id, split_strategy=frozen['strategy'], split_seed=frozen['seed'])
    else:
        common.update(split_id=None, split_strategy=split.strategy or 'event_random_v1',
                      split_seed=split.seed if split.seed is not None else 42,
                      validation_fraction=split.validation_fraction or 0.15, test_fraction=split.test_fraction or 0.15)
    if config.model.kind == 'ridge':
        ridge = config.model.ridge or Ridge()
        return RidgeProbe.model_validate({**common, **ridge.model_dump(exclude={'layers'}),
            'layers': ridge.layers if ridge.layers is not None else [config.features.public_site],
            'validation_tasks': 1})
    mlp, train = config.model.mlp or MLP(), config.training or Training()
    options = {**common, 'public_site': config.features.public_site,
               'hidden_dim': mlp.hidden_dims[0], **mlp.model_dump(),
               **train.model_dump(exclude={'optimizer', 'early_stopping', 'loss'}),
               **train.early_stopping.model_dump(), 'optimizer': train.optimizer.model_dump(),
               'learning_rate': train.optimizer.learning_rate, 'weight_decay': train.optimizer.weight_decay}
    if split.id:
        options['split_receipt_sha256'] = sha256(source / 'splits' / split.id / 'receipt.json')
    if config.model.kind == 'ridge_plus_ffn':
        options['ridge_publications'] = config.model.ridge_publications
        return RidgeMLPProbe.model_validate(options)
    return MLPProbe.model_validate(options)

"""Orchestration-only schema; training contracts stay in training.config."""
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
import yaml

from difficulty.probe_config import DEFAULT_TARGETS
from difficulty.training.config import Strict, Data, Model, Ridge


class WorkflowData(Data):
    targets: list[str] = Field(default_factory=lambda: list(DEFAULT_TARGETS))

    @model_validator(mode='after')
    def no_training_split(self):
        if 'split' in self.model_fields_set:
            raise ValueError('Use outer_split/inner_split; data.split belongs to standalone training')
        return self


class Run(Strict):
    mode: Literal['dry-run', 'check-data', 'fit'] = 'fit'


class Outer(Strict):
    strategy: Literal['task_random_v1', 'task_manifest_v1'] = 'task_random_v1'
    seed: int = Field(default=42, ge=0, lt=2**32)
    test_fraction: float = Field(default=0.2, gt=0, lt=1)
    task_partitions: str | None = None

    @model_validator(mode='after')
    def valid_protocol(self):
        if self.strategy == 'task_manifest_v1':
            if not self.task_partitions or 'test_fraction' in self.model_fields_set or 'seed' in self.model_fields_set:
                raise ValueError('Manifest requires task_partitions; random seed/fraction are mutually exclusive')
        elif self.task_partitions is not None:
            raise ValueError('task_partitions requires task_manifest_v1')
        return self


class Inner(Strict):
    strategy: Literal['event_random_v1'] = 'event_random_v1'
    seed: int = Field(default=42, ge=0, lt=2**32)
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)


class Grid(Ridge):
    layers: list[int] | Literal['auto'] = 'auto'
    alphas: list[float] = Field(default_factory=lambda: [10.0**i for i in range(-4, 9)])
    refit_train_validation: Literal[True] = True


class RidgeModel(Model):
    kind: Literal['ridge'] = 'ridge'
    ridge: Grid = Field(default_factory=Grid)


class Evaluation(Strict):
    nmse_normalization: Literal['target_energy', 'variance'] = 'target_energy'


class Visualization(Strict):
    enabled: bool = True
    export_formats: list[Literal['png', 'svg']] = Field(default_factory=lambda: ['png', 'svg'])

    @model_validator(mode='after')
    def valid_formats(self):
        if not self.export_formats or len(set(self.export_formats)) != len(self.export_formats):
            raise ValueError('export_formats must be nonempty and unique')
        return self


class HoldoutConfig(Strict):
    schema_version: Literal[1] = 1
    run: Run = Field(default_factory=Run)
    data: WorkflowData
    outer_split: Outer = Field(default_factory=Outer)
    inner_split: Inner = Field(default_factory=Inner)
    model: RidgeModel = Field(default_factory=RidgeModel)
    evaluation: Evaluation = Field(default_factory=Evaluation)
    visualization: Visualization = Field(default_factory=Visualization)


def read_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    return HoldoutConfig.model_validate(raw)

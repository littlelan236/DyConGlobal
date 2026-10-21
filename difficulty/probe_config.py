"""Strict configuration for the submission-remainder Ridge pilot."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_TARGETS = (
    "submission_remaining_reasoning_tokens_log1p",
    "submission_remaining_steps_linear",
)
TARGETS = DEFAULT_TARGETS + (
    "submission_remaining_steps_log1p",
    "submission_remaining_reasoning_tokens_linear",
)


class RidgeProbe(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dataset_id: str = "submission-shortest375-20261019"
    dataset_receipt_sha256: str = "f43387de4b54ca714c12b4495696f663039f222930e600a35c0474a3b8318a7c"
    split_id: str | None = None
    targets: list[str] = Field(default_factory=lambda: list(DEFAULT_TARGETS))
    layers: list[int] = Field(default_factory=lambda: list(range(32)))
    alphas: list[float] = Field(default_factory=lambda: [10.0**i for i in range(-4, 9)])
    standardize: bool = True
    split_seed: int = 42
    validation_tasks: int = Field(default=48, gt=0)
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    test_fraction: float = Field(default=0.2, gt=0, lt=1)
    expected_tasks: int = Field(default=238, gt=1)
    expected_rollouts: int | None = Field(default=None, gt=1)
    expected_samples: int = Field(default=14384, gt=1)
    split_strategy: Literal["repo_source_quota_v1", "inherit_task_split_v1", "event_random_v1", "task_random_v1"] = "repo_source_quota_v1"
    parent_experiment_id: str | None = None
    parent_receipt_sha256: str | None = None
    selection_metric: Literal["validation_pooled_r2"] = "validation_pooled_r2"
    tie_tolerance: float = Field(default=1e-10, ge=0)
    blas_threads: int = Field(default=4, gt=0, le=32)

    @model_validator(mode="after")
    def valid_grid(self):
        from core.io import identifier

        identifier(self.dataset_id)
        if self.split_id is not None:
            identifier(self.split_id)
        if self.split_strategy == "inherit_task_split_v1":
            if not self.parent_experiment_id or not self.parent_receipt_sha256:
                raise ValueError("Inherited task split requires parent experiment and receipt SHA")
            identifier(self.parent_experiment_id)
            if len(self.parent_receipt_sha256) != 64 or any(
                c not in "0123456789abcdef" for c in self.parent_receipt_sha256
            ):
                raise ValueError("Invalid parent receipt SHA")
            if self.expected_rollouts is None:
                raise ValueError("Expanded selection requires an explicit rollout count")
        elif self.parent_experiment_id is not None or self.parent_receipt_sha256 is not None:
            raise ValueError("Parent experiment is only used by inherited task split")
        if len(self.dataset_receipt_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.dataset_receipt_sha256
        ):
            raise ValueError("A frozen dataset receipt SHA-256 is required")
        if not self.layers or len(set(self.layers)) != len(self.layers) or any(
            k < 0 or k >= 32 for k in self.layers
        ):
            raise ValueError("Public layers must be unique indices in 0..31")
        if not self.alphas or len(set(self.alphas)) != len(self.alphas) or any(
            not math.isfinite(a) or a <= 0 for a in self.alphas
        ):
            raise ValueError("Alphas must be unique finite positive numbers")
        if not self.targets or len(set(self.targets)) != len(self.targets) or any(
            target not in TARGETS for target in self.targets
        ):
            raise ValueError("Unknown or duplicate submission target")
        if not self.standardize:
            raise ValueError("This pilot requires training-only feature standardization")
        if self.split_strategy in {'repo_source_quota_v1', 'inherit_task_split_v1'} and self.validation_tasks >= self.expected_tasks:
            raise ValueError("Validation must leave training tasks")
        if self.split_strategy in {"event_random_v1", "task_random_v1"} and self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("Three-way split fractions must leave training data")
        if not math.isfinite(self.tie_tolerance):
            raise ValueError("Tie tolerance must be finite")
        return self

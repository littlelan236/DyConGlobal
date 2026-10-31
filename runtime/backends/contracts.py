"""SDK-free identities and ownership boundaries for a physical attempt."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExecutionBinding:
    schema_version: int
    benchmark: str
    agent: str
    backend_family: str
    adapter: str
    adapter_revision: str
    driver: str
    subject_kind: str
    evaluation_mode: str
    evaluator: str
    evidence_owner: str
    re_evaluation: bool

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class AttemptBinding:
    """Scheduler-owned identity, seed/deadline and assigned model services.

    SampleContext and the durable context remain the source of truth. Adapters
    receive exactly one physical attempt and cannot expand repetitions/retries.
    """

    task: dict
    sample: Any
    context: dict
    output: Path
    runtime: dict


@dataclass
class BackendRunContext:
    binding: ExecutionBinding
    cfg: dict
    run_id: str
    factory: Any
    adapter: Any
    closed: bool = field(default=False, init=False)


@dataclass(frozen=True)
class PreflightFacts:
    passed: bool
    blockers: tuple[str, ...] = ()
    receipt: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CleanupFacts:
    outcome: str
    owner: str

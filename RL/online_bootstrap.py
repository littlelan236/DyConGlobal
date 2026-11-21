"""Online DyCon/bootstrap training primitives.

This module is intentionally torch-free.  It freezes the semantics shared by
rollout workers, the trainer, and the estimator store so that the GPU service
can execute them without duplicating policy decisions.
"""
from __future__ import annotations

from collections import defaultdict, deque
import json
from dataclasses import dataclass, field
import math
import random
from typing import Any, Iterable, Literal, Mapping, Sequence

ONLINE_OBJECTIVE_VERSION = "online_bootstrap_grpo_v1"
BASELINE_SUPERVISION_VERSION = "all_completion_tokens_v1"
SOURCE_WEIGHT_VERSION = "fixed_source_mass_v1"
ESTIMATOR_POOL_VERSION = "oldest_first_active_pool_v1"
FEATURE_FAMILY_VERSION = "same_base_cross_version_v1"


@dataclass(frozen=True)
class TaskSchedule:
    """One seeded task permutation reused unchanged by every epoch."""

    tasks: tuple[str, ...]
    n: int
    epochs: int
    seed: int
    order: tuple[str, ...]
    batches: tuple[tuple[str, ...], ...]

    @classmethod
    def build(cls, tasks: Sequence[str], *, n: int = 4, epochs: int = 1, seed: int = 42) -> "TaskSchedule":
        if not tasks or len(set(tasks)) != len(tasks):
            raise ValueError("tasks must be non-empty and contain no duplicates")
        if n < 1 or epochs < 1:
            raise ValueError("n and epochs must be positive")
        order = list(tasks)
        random.Random(seed).shuffle(order)
        per_epoch = tuple(tuple(order[i:i + n]) for i in range(0, len(order), n))
        return cls(tuple(tasks), n, epochs, seed, tuple(order), per_epoch * epochs)

    @property
    def steps_per_epoch(self) -> int:
        return len(self.batches) // self.epochs

    @property
    def total_steps(self) -> int:
        return len(self.batches)

    def batch(self, step: int) -> tuple[str, ...]:
        if step < 0 or step >= self.total_steps:
            raise IndexError(step)
        return self.batches[step]

    def manifest(self) -> dict[str, Any]:
        return {"version": "shuffle_once_epoch_v1", "seed": self.seed,
                "tasks": list(self.tasks), "order": list(self.order), "n": self.n,
                "epochs": self.epochs, "steps_per_epoch": self.steps_per_epoch,
                "total_steps": self.total_steps,
                "batches": [list(x) for x in self.batches]}


@dataclass(frozen=True)
class SourceMass:
    dycon_weight: float = 0.5
    baseline_weight: float = 0.5
    version: str = SOURCE_WEIGHT_VERSION

    def normalized(self) -> tuple[float, float]:
        values = (float(self.dycon_weight), float(self.baseline_weight))
        if any(not math.isfinite(x) or x < 0 for x in values) or sum(values) <= 0:
            raise ValueError("source weights must be finite, non-negative, and not both zero")
        total = sum(values)
        return values[0] / total, values[1] / total

    def token_coefficients(self, *, task_count: int, dycon_groups: int,
                           dycon_candidates: int, dycon_tokens: int,
                           baseline_repetitions: int, baseline_tokens: int) -> dict[str, float]:
        alpha, beta = self.normalized()
        if min(task_count, dycon_groups, dycon_candidates, dycon_tokens,
               baseline_repetitions, baseline_tokens) <= 0:
            raise ValueError("all source coefficient denominators must be positive")
        return {"dycon": alpha / (task_count * dycon_groups * dycon_candidates * dycon_tokens),
                "baseline": beta / (task_count * baseline_repetitions * baseline_tokens),
                "alpha": alpha, "beta": beta}


@dataclass(frozen=True)
class BaselineStepExample:
    trajectory_id: str
    task_id: str
    repetition: int
    agent_step: int
    input_token_ids: tuple[int, ...]
    labels: tuple[int, ...]
    loss_mask: tuple[int, ...]
    advantage: int
    trajectory_token_count: int
    supervision: str = BASELINE_SUPERVISION_VERSION


def make_baseline_step(*, trajectory_id: str, task_id: str, repetition: int,
                       agent_step: int, context_token_ids: Sequence[int],
                       completion_token_ids: Sequence[int], resolved: bool,
                       trajectory_token_count: int | None = None) -> BaselineStepExample:
    """Build completion-only labels while retaining the complete differentiable prefix."""
    context = tuple(int(x) for x in context_token_ids)
    completion = tuple(int(x) for x in completion_token_ids)
    if not context or not completion:
        raise ValueError("baseline steps require non-empty context and completion")
    total = len(completion) if trajectory_token_count is None else int(trajectory_token_count)
    if total < len(completion) or total <= 0:
        raise ValueError("trajectory_token_count must cover this completion")
    return BaselineStepExample(
        trajectory_id=str(trajectory_id), task_id=str(task_id), repetition=int(repetition),
        agent_step=int(agent_step), input_token_ids=context + completion,
        labels=(-100,) * len(context) + completion,
        loss_mask=(0,) * len(context) + (1,) * len(completion),
        advantage=1 if resolved else -1, trajectory_token_count=total)


@dataclass(frozen=True)
class RolloutWorkItem:
    trajectory_id: str
    task_id: str
    source: Literal["dycon", "baseline"]
    estimated_seconds: float = 0.0
    candidate_index: int | None = None


class MixedRolloutScheduler:
    """Fair unified queue for DyCon and baseline requests on one vLLM service."""

    def __init__(self, items: Iterable[RolloutWorkItem], *, max_inflight: int = 1,
                 dycon_candidate_portion: int = 1):
        if max_inflight < 1 or dycon_candidate_portion < 1:
            raise ValueError("max_inflight and dycon_candidate_portion must be positive")
        self.items = tuple(items)
        if any(not math.isfinite(x.estimated_seconds) or x.estimated_seconds < 0 for x in self.items):
            raise ValueError("estimated_seconds must be finite and non-negative")
        self.max_inflight = max_inflight
        self.dycon_candidate_portion = dycon_candidate_portion

    def order(self) -> tuple[RolloutWorkItem, ...]:
        """Longest trajectories start first; active trajectories receive round-robin service."""
        by_type: dict[str, deque[RolloutWorkItem]] = {"dycon": deque(), "baseline": deque()}
        for item in sorted(self.items, key=lambda x: (-x.estimated_seconds, x.trajectory_id,
                                                       x.candidate_index if x.candidate_index is not None else -1)):
            by_type[item.source].append(item)
        output: list[RolloutWorkItem] = []
        turn = 0
        while by_type["dycon"] or by_type["baseline"]:
            # Keep both sources admitted when both have work; a source may use a
            # bounded number of DyCon candidates before fairness switches back.
            source = ("dycon" if turn % 2 == 0 and by_type["dycon"] else "baseline")
            if not by_type[source]:
                source = "baseline" if source == "dycon" else "dycon"
            count = self.dycon_candidate_portion if source == "dycon" else 1
            for _ in range(count):
                if not by_type[source]:
                    break
                output.append(by_type[source].popleft())
            turn += 1
        return tuple(output)

    @staticmethod
    def assign_workers(items: Sequence[RolloutWorkItem], worker_ids: Sequence[int | str]) -> tuple[tuple[RolloutWorkItem, int | str], ...]:
        """Assign ordered work with LPT load balancing across rollout workers."""
        workers = tuple(worker_ids)
        if not workers or len(set(workers)) != len(workers):
            raise ValueError("worker_ids must be non-empty and unique")
        loads = {worker: 0.0 for worker in workers}
        assigned = []
        counts = {worker: 0 for worker in workers}
        for item in items:
            worker = min(workers, key=lambda value: (loads[value], counts[value], workers.index(value)))
            assigned.append((item, worker))
            loads[worker] += float(item.estimated_seconds)
            counts[worker] += 1
        return tuple(assigned)

    def manifest(self) -> dict[str, Any]:
        ordered = self.order()
        return {"version": "mixed_shared_vllm_longest_first_v1", "max_inflight": self.max_inflight,
                "dycon_candidate_portion": self.dycon_candidate_portion,
                "order": [x.trajectory_id for x in ordered],
                "sources": [x.source for x in ordered]}


@dataclass(frozen=True)
class HiddenPoint:
    point_id: str
    hidden: Any
    target: float
    model_fingerprint: str
    feature_family: str
    base_fingerprint: str
    tokenizer_fingerprint: str
    site: str
    width: int
    dtype: str
    birth_step: int = 0
    split: str | None = None


def validate_feature_family(points: Sequence[HiddenPoint], *, expected_family: str,
                            expected_base: str, expected_tokenizer: str,
                            expected_site: str, expected_width: int) -> None:
    for p in points:
        if p.feature_family != expected_family or p.base_fingerprint != expected_base:
            raise ValueError("hidden point is outside the estimator feature family/base")
        if p.tokenizer_fingerprint != expected_tokenizer or p.site != expected_site or p.width != expected_width:
            raise ValueError("hidden point feature contract mismatch")


class ActiveEstimatorPool:
    """Fixed-capacity point pool with global oldest-first replacement."""

    def __init__(self, capacity: int, *, tie_seed: int = 42):
        if capacity < 1:
            raise ValueError("estimator pool capacity must be positive")
        self.capacity = capacity
        self.tie_seed = tie_seed
        self._points: dict[str, HiddenPoint] = {}

    @property
    def points(self) -> tuple[HiddenPoint, ...]:
        return tuple(self._points.values())

    def initialize(self, points: Iterable[HiddenPoint]) -> None:
        values = list(points)
        if len(values) != self.capacity:
            raise ValueError("initial estimator points must fill the active pool")
        if len({x.point_id for x in values}) != len(values):
            raise ValueError("duplicate estimator point identity")
        self._points = {x.point_id: HiddenPoint(**{**x.__dict__, "birth_step": 0}) for x in values}

    def ingest(self, points: Iterable[HiddenPoint], *, step: int) -> dict[str, Any]:
        if step < 0:
            raise ValueError("step must be non-negative")
        incoming = []
        seen = set(self._points)
        for point in points:
            if point.point_id in seen or any(x.point_id == point.point_id for x in incoming):
                continue
            incoming.append(HiddenPoint(**{**point.__dict__, "birth_step": step}))
        if len(incoming) > self.capacity:
            raise ValueError("cannot replace more points than active pool capacity")
        if not incoming:
            return {"added": 0, "removed": [], "active": len(self._points), "step": step}
        candidates = list(self._points.values())
        if len(candidates) < len(incoming):
            raise ValueError("active pool is smaller than replacement batch")
        rng = random.Random(self.tie_seed + step)
        cohorts: dict[int, list[HiddenPoint]] = defaultdict(list)
        for candidate in candidates:
            cohorts[candidate.birth_step].append(candidate)
        remove: list[HiddenPoint] = []
        for birth_step in sorted(cohorts):
            cohort = cohorts[birth_step]
            rng.shuffle(cohort)
            take = min(len(incoming) - len(remove), len(cohort))
            remove.extend(cohort[:take])
            if len(remove) == len(incoming):
                break
        for point in remove:
            del self._points[point.point_id]
        for point in incoming:
            self._points[point.point_id] = point
        return {"added": len(incoming), "removed": [x.point_id for x in remove],
                "active": len(self._points), "step": step, "replacement": "oldest_first_uniform_tie_v1"}

    def manifest(self) -> dict[str, Any]:
        return {"version": ESTIMATOR_POOL_VERSION, "capacity": self.capacity,
                "tie_seed": self.tie_seed,
                "points": [{"point_id": p.point_id, "birth_step": p.birth_step, "split": p.split}
                           for p in self.points]}



class EstimatorDatasetStore:
    """Durable estimator history plus a fixed-capacity active sample index."""
    def __init__(self, root: str, *, capacity: int, tie_seed: int = 42,
                 feature_family: str = FEATURE_FAMILY_VERSION):
        from pathlib import Path
        self.root = Path(root)
        self.capacity = int(capacity)
        self.feature_family = str(feature_family)
        self.archive_path = self.root / "history.jsonl"
        self.active_path = self.root / "active.json"
        self.pool = ActiveEstimatorPool(self.capacity, tie_seed=tie_seed)
        # Point IDs are globally immutable across the append-only history.  This
        # prevents a point that was evicted and later replayed from being
        # appended a second time, while still allowing it to re-enter the active
        # pool only through a new logical point ID.
        self._history_ids: set[str] = set()

    @staticmethod
    def _record(point: HiddenPoint) -> dict[str, Any]:
        return {"point_id": point.point_id, "hidden": point.hidden, "target": point.target,
                "model_fingerprint": point.model_fingerprint,
                "feature_family": point.feature_family,
                "base_fingerprint": point.base_fingerprint,
                "tokenizer_fingerprint": point.tokenizer_fingerprint,
                "site": point.site, "width": point.width, "dtype": point.dtype,
                "birth_step": point.birth_step, "split": point.split}

    @staticmethod
    def _atomic_json(path, value) -> None:
        import os
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        descriptor = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def initialize(self, points: Iterable[HiddenPoint]) -> dict[str, Any]:
        values = list(points)
        if self.active_path.exists() or self.archive_path.exists():
            raise ValueError("estimator store already exists; restore it instead")
        if not values:
            raise ValueError("initial estimator points must be non-empty")
        self._validate(values, values[0])
        self.pool.initialize(values)
        self.root.mkdir(parents=True, exist_ok=True)
        import os
        with self.archive_path.open("x", encoding="utf-8") as stream:
            for point in self.pool.points:
                stream.write(json.dumps(self._record(point), sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._history_ids = {point.point_id for point in self.pool.points}
        self._save_active()
        return {"version": "estimator_dataset_store_v2", "initialized": len(values),
                "capacity": self.capacity, "feature_family": self.feature_family}

    def _validate(self, values, reference):
        validate_feature_family(values, expected_family=self.feature_family,
                                expected_base=reference.base_fingerprint,
                                expected_tokenizer=reference.tokenizer_fingerprint,
                                expected_site=reference.site, expected_width=reference.width)
        if any(point.dtype != reference.dtype for point in values):
            raise ValueError("hidden point dtype differs from estimator contract")
        for point in values:
            if (not point.point_id or not point.model_fingerprint or point.width < 1
                    or not math.isfinite(float(point.target)) or point.target < 0
                    or len(point.hidden) != point.width
                    or not all(math.isfinite(float(x)) for x in point.hidden)):
                raise ValueError("invalid estimator point identity, hidden vector or target")

    def _history(self, *, repair_tail=False):
        records = {}
        with self.archive_path.open("r+b" if repair_tail else "rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    if repair_tail and not stream.read(1):
                        stream.seek(offset)
                        stream.truncate()
                        break
                    raise ValueError("estimator history is corrupt")
                point_id = row["point_id"]
                if point_id in records and records[point_id] != row:
                    raise ValueError("estimator point identity changed in history")
                records[point_id] = row
        return records

    def _finish_pending(self):
        import os
        pending_path = self.root / "pending-ingest.json"
        if not pending_path.is_file():
            return
        pending = json.loads(pending_path.read_text(encoding="utf-8"))
        records = self._history(repair_tail=True)
        with self.archive_path.open("a", encoding="utf-8") as stream:
            for row in pending["accepted"]:
                old = records.get(row["point_id"])
                if old is not None and old != row:
                    raise ValueError("pending estimator point conflicts with immutable history")
                if old is None:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._atomic_json(self.active_path, pending["active_after"])
        self._atomic_json(self.root / "ingests" / f"step-{pending['step']:06d}.json", pending)
        pending_path.unlink()

    def restore(self) -> dict[str, Any]:
        """Complete a write-ahead ingest before restoring the ordered active index."""
        if not self.archive_path.is_file() or not self.active_path.is_file():
            raise FileNotFoundError("estimator store has no restorable history/active index")
        self._finish_pending()
        active = json.loads(self.active_path.read_text(encoding="utf-8"))
        if active.get("capacity") != self.capacity or active.get("tie_seed") != self.pool.tie_seed:
            raise ValueError("estimator store capacity/tie seed differs from saved contract")
        records = self._history()
        values = []
        # Preserve index order and its authoritative birth ages. Legacy history
        # may contain unnormalised ages from before write-ahead ingest existed.
        for entry in active.get("points", []):
            row = records.get(entry["point_id"])
            if row is None or "hidden" not in row:
                raise ValueError("estimator history lacks active hidden payload")
            values.append(HiddenPoint(**{**row, "birth_step": entry["birth_step"]}))
        if len(values) != self.capacity or len({p.point_id for p in values}) != self.capacity:
            raise ValueError("estimator active index capacity/identity mismatch")
        self._validate(values, values[0])
        self.pool._points = {point.point_id: point for point in values}
        self._history_ids = set(records)
        return {"restored": len(values), "history_path": str(self.archive_path),
                "active_path": str(self.active_path)}

    def ingest(self, points: Iterable[HiddenPoint], *, step: int) -> dict[str, Any]:
        import hashlib
        if step < 0:
            raise ValueError("step must be non-negative")
        self.restore()
        values = list(points)
        self._validate(values, self.pool.points[0])
        records = self._history()
        unique = {}
        for point in values:
            row = self._record(point)
            row.pop("birth_step")
            if point.point_id in unique and unique[point.point_id] != row:
                raise ValueError("conflicting duplicate estimator point")
            if point.point_id in records:
                old = dict(records[point.point_id])
                old.pop("birth_step")
                if old != row:
                    raise ValueError("estimator point identity is immutable")
            unique[point.point_id] = row
        request_sha = hashlib.sha256(json.dumps(unique, sort_keys=True, allow_nan=False).encode()).hexdigest()
        receipt_path = self.root / "ingests" / f"step-{step:06d}.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["request_sha256"] != request_sha:
                raise ValueError("estimator ingest step replay has different points")
            return {**receipt["event"], "reused": True}
        accepted = [HiddenPoint(**row, birth_step=step) for key, row in unique.items() if key not in records]
        # Compute the entire replacement before any persistent mutation.
        candidate = ActiveEstimatorPool(self.capacity, tie_seed=self.pool.tie_seed)
        candidate._points = dict(self.pool._points)
        event = candidate.ingest(accepted, step=step)
        event.update(historical_appended=len(accepted), history_path=str(self.archive_path),
                     active_path=str(self.active_path))
        pending = {"schema_version": 2, "step": step, "request_sha256": request_sha,
                   "accepted": [self._record(p) for p in accepted],
                   "active_after": candidate.manifest(), "event": event}
        self._atomic_json(self.root / "pending-ingest.json", pending)
        self._finish_pending()
        self.restore()
        return event

    def _save_active(self) -> None:
        self._atomic_json(self.active_path, self.pool.manifest())

    def active_records(self) -> tuple[dict[str, Any], ...]:
        """Return complete active hidden/label records for a fit snapshot."""
        return tuple(self._record(point) for point in self.pool.points)

    def manifest(self) -> dict[str, Any]:
        return {"version": "estimator_dataset_store_v1", "root": str(self.root),
                "feature_family": self.feature_family, "active": self.pool.manifest(),
                "history_exists": self.archive_path.is_file()}

@dataclass(frozen=True)
class OnlineBatchPlan:
    schedule: TaskSchedule
    m: int
    branch_count: int
    source_mass: SourceMass

    def work_items(self, task_ids: Sequence[str]) -> tuple[RolloutWorkItem, ...]:
        items: list[RolloutWorkItem] = []
        for task in task_ids:
            items.append(RolloutWorkItem(f"{task}:dycon", task, "dycon", candidate_index=0))
            for repetition in range(self.m):
                items.append(RolloutWorkItem(f"{task}:baseline:{repetition}", task, "baseline"))
        return tuple(items)

    def manifest(self) -> dict[str, Any]:
        alpha, beta = self.source_mass.normalized()
        return {"objective_version": ONLINE_OBJECTIVE_VERSION, "schedule": self.schedule.manifest(),
                "m": self.m, "branch_count": self.branch_count,
                "trajectory_counts": {"dycon": self.schedule.n, "baseline": self.schedule.n * self.m},
                "source_mass": {"dycon": alpha, "baseline": beta}}


def online_surrogate_loss(*, logp_current: float, logp_rollout: float,
                          advantage: float, clip_epsilon: float = 0.1,
                          logp_kl_reference: float | None = None,
                          kl_beta: float = 0.0, k3_cap: float | None = None) -> dict[str, float]:
    """One raw-T1 token objective with a batch-generation denominator.

    The rollout probability and KL reference are detached inputs.  The caller
    supplies a differentiable current log probability when using torch; this
    pure function is used for contract tests and diagnostics.
    """
    values = (logp_current, logp_rollout, advantage, clip_epsilon, kl_beta)
    if any(not math.isfinite(float(x)) for x in values) or clip_epsilon <= 0:
        raise ValueError("objective inputs must be finite and clip_epsilon positive")
    ratio = math.exp(max(-80.0, min(80.0, logp_current - logp_rollout)))
    clipped = min(1.0 + clip_epsilon, max(1.0 - clip_epsilon, ratio))
    policy = -min(ratio * advantage, clipped * advantage)
    k3 = 0.0
    if kl_beta:
        if logp_kl_reference is None or not math.isfinite(float(logp_kl_reference)):
            raise ValueError("KL reference logprob is required when kl_beta > 0")
        delta = logp_kl_reference - logp_current
        k3 = math.expm1(max(-80.0, min(80.0, delta))) - delta
        if k3_cap is not None:
            if not math.isfinite(k3_cap) or k3_cap <= 0:
                raise ValueError("k3_cap must be positive")
            k3 = min(k3, k3_cap)
    return {"ratio": ratio, "clipped_ratio": clipped, "policy": policy,
            "k3": k3, "loss": policy + kl_beta * k3}


class OnlineBootstrapEngine:
    """One-batch-prefetch coordinator for rollout/update overlap.

    Callbacks are deliberately small adapters around the project's formal run,
    trainer, and estimator services.  The engine never starts a second
    prefetched batch, and publishes a new policy/estimator version only after
    both update and rollout have completed at the barrier.
    """

    def __init__(self, schedule: TaskSchedule, *, rollout, update, resolved_points,
                 refit, max_prefetch: int = 1):
        if max_prefetch != 1:
            raise ValueError("online bootstrap currently permits exactly one prefetched batch")
        self.schedule = schedule
        self.rollout = rollout
        self.update = update
        self.resolved_points = resolved_points
        self.refit = refit

    def run(self) -> list[dict[str, Any]]:
        from concurrent.futures import ThreadPoolExecutor
        history: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="online-rollout") as pool:
            current = self.rollout(self.schedule.batch(0), 0, None, None)
            for step in range(self.schedule.total_steps):
                next_future = (pool.submit(self.rollout, self.schedule.batch(step + 1), step + 1,
                                           step, None)
                               if step + 1 < self.schedule.total_steps else None)
                update_result = self.update(current, step)
                points = tuple(self.resolved_points(current) or ())
                refit_result = self.refit(points, step)
                next_batch = next_future.result() if next_future is not None else None
                history.append({"step": step, "tasks": list(self.schedule.batch(step)),
                                "update": update_result, "resolved_points": len(points),
                                "refit": refit_result, "next_ready": next_batch is not None})
                current = next_batch
        return history


@dataclass
class OnlineCoordinatorState:
    """Durable barrier state; only committed steps advance the cursor."""
    campaign_id: str
    committed_step: int = 0
    policy_version: str = "S0"
    estimator_version: str = "E0"
    prefetched_step: int | None = None
    schedule_sha256: str | None = None
    checkpoint_path: str | None = None
    current_batch_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"version": "online_barrier_state_v1", **self.__dict__}


class OnlineCoordinator:
    """Coordinate one prefetched rollout with one optimizer/update lane.

    The callbacks are the integration boundary for the formal run/serving and
    RL trainer services.  The coordinator itself owns ordering, the barrier,
    checkpoint state, and exactly-once step commits.
    """

    def __init__(self, schedule: TaskSchedule, *, rollout, update, resolved_points,
                 estimator_ingest, estimator_refit, campaign_id: str = "online-bootstrap",
                 state_path: str | None = None):
        self.schedule = schedule
        self.rollout = rollout
        self.update = update
        self.resolved_points = resolved_points
        self.estimator_ingest = estimator_ingest
        self.estimator_refit = estimator_refit
        self.campaign_id = campaign_id
        self.state_path = state_path
        self.state = OnlineCoordinatorState(campaign_id=campaign_id,
                                            schedule_sha256=_schedule_digest(schedule))
        if state_path:
            self._load_state()

    def _load_state(self) -> None:
        import json
        from pathlib import Path
        path = Path(self.state_path)
        if not path.exists():
            return
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schedule_sha256") != _schedule_digest(self.schedule):
            raise ValueError("online coordinator schedule differs from saved state")
        if value.get("campaign_id") != self.campaign_id:
            raise ValueError("online coordinator campaign differs from saved state")
        allowed = {"campaign_id", "committed_step", "policy_version", "estimator_version",
                   "prefetched_step", "schedule_sha256", "checkpoint_path", "current_batch_sha256"}
        self.state = OnlineCoordinatorState(**{k: value[k] for k in allowed if k in value})

    def _save_state(self) -> None:
        if not self.state_path:
            return
        import json, os
        from pathlib import Path
        path = Path(self.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.state.as_dict(), sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def _validate_batch_lineage(self, batch: Any, step: int) -> None:
        """Allow only the current or one-step-prefetched policy lineage."""
        policy = getattr(batch, "policy_version", None)
        estimator = getattr(batch, "estimator_version", None)
        if policy is None and estimator is None:
            return
        allowed_policy = {self.state.policy_version}
        allowed_estimator = {self.state.estimator_version}
        if step > 0:
            allowed_policy.add(f"S{step - 1}")
            allowed_estimator.add(f"E{step - 1}")
        if policy is not None and policy not in allowed_policy:
            raise ValueError(f"rollout policy lineage mismatch at step {step}: {policy}")
        if estimator is not None and estimator not in allowed_estimator:
            raise ValueError(f"rollout estimator lineage mismatch at step {step}: {estimator}")

    def run(self) -> list[dict[str, Any]]:
        from concurrent.futures import ThreadPoolExecutor
        import hashlib
        import json
        from pathlib import Path

        if self.state.committed_step >= self.schedule.total_steps:
            return []
        history: list[dict[str, Any]] = []
        batch_root = None
        if self.state_path:
            batch_root = Path(self.state_path).with_name(Path(self.state_path).name + ".batches")
            batch_root.mkdir(parents=True, exist_ok=True)

        def batch_path(step: int):
            return None if batch_root is None else batch_root / f"step-{step:06d}.json"

        def persist_batch(batch: Any, step: int) -> str | None:
            path = batch_path(step)
            if path is None or not hasattr(batch, "as_dict"):
                return None
            payload = batch.as_dict()
            if int(payload.get("step", step)) != step:
                raise ValueError(f"rollout batch step mismatch: expected {step}")
            encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                 allow_nan=False, separators=(",", ":")).encode()
            digest = hashlib.sha256(encoded).hexdigest()
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(encoded + b"\n")
            import os
            with tmp.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(tmp, path)
            EstimatorDatasetStore._atomic_json(path.with_suffix(".receipt.json"), {
                "campaign_id": self.campaign_id, "schedule_sha256": _schedule_digest(self.schedule),
                "step": step, "sha256": digest})
            return digest

        def restore_batch(step: int):
            path = batch_path(step)
            if path is None or not path.is_file():
                return None
            from RL.online_runtime import OnlineRolloutBatch
            raw = path.read_bytes()
            value = json.loads(raw.decode())
            batch = OnlineRolloutBatch.from_dict(value)
            if batch.step != step:
                raise ValueError(f"persisted rollout batch step mismatch: {batch.step} != {step}")
            digest = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                               allow_nan=False, separators=(",", ":")).encode()).hexdigest()
            receipt_path = path.with_suffix(".receipt.json")
            if not receipt_path.is_file():
                raise ValueError("persisted rollout batch lacks integrity receipt")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (receipt.get("sha256") != digest or receipt.get("campaign_id") != self.campaign_id
                    or receipt.get("schedule_sha256") != _schedule_digest(self.schedule)
                    or batch.tasks != self.schedule.batch(step)):
                raise ValueError("persisted rollout batch identity/checksum mismatch")
            if self.state.current_batch_sha256 and step == self.state.committed_step \
                    and digest != self.state.current_batch_sha256:
                raise ValueError("persisted rollout batch checksum differs from coordinator state")
            return batch, digest

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="online-rollout") as pool:
            current_step = self.state.committed_step
            restored = restore_batch(current_step)
            if restored is not None:
                current, current_digest = restored
                self.state.current_batch_sha256 = current_digest
            else:
                current = self.rollout(self.schedule.batch(current_step),
                                       current_step, self.state.policy_version,
                                       self.state.estimator_version)
                current_digest = persist_batch(current, current_step)
                self.state.current_batch_sha256 = current_digest
                self._save_state()
            self._validate_batch_lineage(current, current_step)
            for step in range(self.state.committed_step, self.schedule.total_steps):
                next_future = None
                if step + 1 < self.schedule.total_steps:
                    persisted_next = restore_batch(step + 1)
                    if persisted_next is not None:
                        next_batch, next_digest = persisted_next
                    else:
                        policy_for_next = self.state.policy_version
                        estimator_for_next = self.state.estimator_version
                        def sample_next(next_step=step + 1,
                                        policy_version=policy_for_next,
                                        estimator_version=estimator_for_next):
                            # Persist the prefetched batch inside the worker as
                            # soon as rollout seals it.  If training fails while
                            # this batch is already complete, restart can reuse
                            # it without issuing a second physical rollout.
                            # Lineage is captured at submission time; the
                            # training thread may advance state before this
                            # worker obtains its turn.
                            value = self.rollout(self.schedule.batch(next_step), next_step,
                                                 policy_version, estimator_version)
                            digest = persist_batch(value, next_step)
                            return value, digest
                        next_future = pool.submit(sample_next)
                        self.state.prefetched_step = step + 1
                        self._save_state()
                        next_batch = next_digest = None
                else:
                    next_batch = next_digest = None
                update_result = self.update(current, step, self.state.policy_version,
                                            self.state.estimator_version)
                if next_future is not None:
                    next_batch, next_digest = next_future.result()
                # Estimator data follows sampling completion, including the cold
                # batch and first prefetched batch at the first barrier. Later
                # barriers admit only their newly sampled batch. Ingest receipts
                # make retries of this boundary deterministic.
                newly_sampled = ([current] if step == 0 else [])
                if next_batch is not None:
                    newly_sampled.append(next_batch)
                ingest_events = []
                point_count = 0
                for sampled in newly_sampled:
                    sampled_step = getattr(sampled, "step", step if sampled is current else step + 1)
                    points = tuple(self.resolved_points(sampled) or ())
                    point_count += len(points)
                    ingest_events.append(self.estimator_ingest(points, sampled_step))
                ingest_result = {"batches": ingest_events, "new_points": point_count}
                estimator_result = self.estimator_refit(step, ingest_result)
                if next_batch is not None:
                    self._validate_batch_lineage(next_batch, step + 1)
                self.state.committed_step = step + 1
                if isinstance(update_result, Mapping) and update_result.get("checkpoint_path"):
                    self.state.checkpoint_path = str(update_result["checkpoint_path"])
                self.state.policy_version = f"S{self.state.committed_step}"
                self.state.estimator_version = f"E{self.state.committed_step}"
                self.state.prefetched_step = None
                self.state.current_batch_sha256 = next_digest
                self._save_state()
                history.append({"step": step, "tasks": list(self.schedule.batch(step)),
                                "update": update_result, "resolved_points": point_count,
                                "estimator_ingest": ingest_result, "estimator_refit": estimator_result,
                                "committed_step": self.state.committed_step,
                                "batch_sha256": current_digest})
                current = next_batch
                current_digest = next_digest
        return history


def _schedule_digest(schedule: TaskSchedule) -> str:
    import hashlib, json
    payload = json.dumps(schedule.manifest(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def dry_run_from_config(config) -> dict[str, Any]:
    """Validate and materialize the online schedule without loading a model."""
    if config.algorithm is None or config.algorithm.name != "online_bootstrap_grpo_v1":
        raise ValueError("online bootstrap algorithm is required")
    options = config.online_bootstrap
    schedule = TaskSchedule.build(options.tasks, n=options.n, epochs=options.epochs,
                                  seed=options.shuffle_seed)
    plan = OnlineBatchPlan(schedule=schedule, m=options.m,
                           branch_count=options.branch_count,
                           source_mass=SourceMass(options.dycon_weight, options.baseline_weight))
    alpha, beta = plan.source_mass.normalized()
    return {"status": "validated", "mode": "dry-run", "objective_version": ONLINE_OBJECTIVE_VERSION,
            "schedule": schedule.manifest(), "plan": plan.manifest(),
            "initial_estimator": options.initial_estimator,
            "estimator": {"feature_family": options.feature_family,
                          "capacity": options.estimator_capacity,
                          "tie_seed": options.estimator_tie_seed},
            "rollout": {"startup_order": options.startup_order,
                        "request_admission": options.request_admission,
                        "max_inflight": options.max_inflight,
                        "dycon_candidate_portion": options.dycon_candidate_portion,
                        "provider": ("shared_vllm_http" if options.rollout_endpoint else "jsonl_manifest"),
                        "endpoint": options.rollout_endpoint,
                        "manifest": options.rollout_manifest,
                        "worker_ids": list(options.rollout_gpus or ["shared"])},
            "source_mass": {"dycon": alpha, "baseline": beta},
            "failure_policy": {"baseline_failure_reward": options.baseline_failure_reward,
                               "verifier_error_retries": options.verifier_error_retries},
            "baseline_training": {"mode": options.baseline_training_mode,
                                  "max_trace_input_tokens": options.max_trace_input_tokens},
            "numerics": {"blas_library": "cublaslt",
                         "bf16_allow_reduced_precision_reduction": config.runtime.bf16_allow_reduced_precision_reduction,
                         "bf16_allow_split_k": config.runtime.bf16_allow_split_k}}

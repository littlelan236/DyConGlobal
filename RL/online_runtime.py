"""Runtime adapters for online bootstrap LoRA training.

The module keeps the online wire contract independent from the rollout service. A
provider can be backed by a shared vLLM queue or a deterministic JSONL manifest;
the trainer consumes the same validated records in both cases.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from RL.online_bootstrap import (HiddenPoint, MixedRolloutScheduler, RolloutWorkItem,
                                 SourceMass, TaskSchedule, validate_feature_family)

ONLINE_ROLLOUT_SCHEMA = "online_bootstrap_rollout_batch_v1"
ONLINE_RUN_WORK_ITEMS_SCHEMA = "online_bootstrap_run_work_items_v1"


def make_online_run_work_items(selection: Mapping[str, Any], run_id: str, *, m: int,
                               base_seed: int, branch_count: int) -> dict[str, Any]:
    """Build a frozen standard-run manifest for one mixed online batch.

    The first repetition of each task is the DyCon consensus trajectory and
    the following ``m`` repetitions are baseline trajectories.  The returned
    object is accepted by ``application.execution.run_plan.create_run`` and
    keeps all ordinary sample/retry identities intact.
    """
    if int(m) < 1 or int(branch_count) < 1:
        raise ValueError("m and branch_count must be positive")
    from rollout.sampling import make_work_items
    manifest = make_work_items(selection, run_id, int(m) + 1, int(base_seed), int(branch_count))
    for item in manifest["items"]:
        repetition = int(item["repetition_index"])
        item["online_source"] = "dycon" if repetition == 0 else "baseline"
        item["candidate_index"] = 0 if repetition == 0 else None
        item["online_repetition"] = 0 if repetition == 0 else repetition - 1
    manifest["schema"] = ONLINE_RUN_WORK_ITEMS_SCHEMA
    manifest["online_contract"] = {"dycon_per_task": 1, "baseline_repetitions_per_task": int(m),
                                    "profile_assignment": "repetition_zero_dycon_rest_baseline_v1"}
    return manifest


@dataclass(frozen=True)
class OnlineRow:
    row_id: str
    task_id: str
    source: str
    group_id: str
    input_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    rollout_logprobs: tuple[float, ...]
    kl_reference_logprobs: tuple[float, ...] | None
    advantage: float
    trajectory_id: str | None = None
    trajectory_token_count: int | None = None
    agent_step: int | None = None
    resolved: bool | None = None
    hidden: tuple[float, ...] | None = None
    target: float | None = None
    hidden_meta: Mapping[str, Any] | None = None
    reward_reason: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OnlineRow":
        required = ("row_id", "task_id", "source", "group_id", "input_token_ids",
                    "completion_token_ids", "rollout_logprobs", "advantage")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError("online rollout row missing " + ", ".join(missing))
        source = str(value["source"])
        if source not in {"dycon", "baseline"}:
            raise ValueError("online rollout source must be dycon or baseline")
        inputs = tuple(int(x) for x in value["input_token_ids"])
        completion = tuple(int(x) for x in value["completion_token_ids"])
        rollout = tuple(float(x) for x in value["rollout_logprobs"])
        kl = value.get("kl_reference_logprobs")
        kl_values = None if kl is None else tuple(float(x) for x in kl)
        if len(inputs) <= len(completion) or not completion or len(rollout) != len(completion):
            raise ValueError("online rollout row has inconsistent token lengths")
        if tuple(inputs[-len(completion):]) != completion:
            raise ValueError("completion tokens must be the suffix of input_token_ids")
        failure_reward = source == "baseline" and value.get("reward_reason") in {"LimitsExceeded", "reasoning_loop"}
        if failure_reward and (float(value["advantage"]) >= 0 or value.get("resolved") is not False):
            raise ValueError("baseline failure reward must be negative and unresolved")
        if source == "baseline" and not failure_reward and float(value["advantage"]) not in {-1.0, 1.0}:
            raise ValueError("baseline advantage must be the resolved/unresolved binary +/-1")
        if source == "baseline" and not failure_reward and value.get("resolved") is not None:
            expected = 1.0 if bool(value["resolved"]) else -1.0
            if float(value["advantage"]) != expected:
                raise ValueError("baseline resolved flag disagrees with advantage")
        if kl_values is not None and len(kl_values) != len(completion):
            raise ValueError("online KL reference length differs from completion")
        if any(x < 0 for x in inputs) or not math.isfinite(float(value["advantage"])):
            raise ValueError("online token IDs must be nonnegative and advantage finite")
        if not all(math.isfinite(x) for x in (*rollout, *(kl_values or ()) )):
            raise ValueError("online rollout logprobs must be finite")
        trajectory_count = value.get("trajectory_token_count")
        if trajectory_count is not None and int(trajectory_count) < len(completion):
            raise ValueError("trajectory_token_count must cover completion")
        hidden = value.get("hidden")
        hidden_values = None if hidden is None else tuple(float(x) for x in hidden)
        return cls(row_id=str(value["row_id"]), task_id=str(value["task_id"]),
                   source=source, group_id=str(value["group_id"]),
                   input_token_ids=inputs, completion_token_ids=completion,
                   rollout_logprobs=rollout, kl_reference_logprobs=kl_values,
                   advantage=float(value["advantage"]),
                   trajectory_id=value.get("trajectory_id"),
                   trajectory_token_count=None if trajectory_count is None else int(trajectory_count),
                   agent_step=None if value.get("agent_step") is None else int(value["agent_step"]),
                   resolved=value.get("resolved"), hidden=hidden_values, target=value.get("target"),
                   hidden_meta=value.get("hidden_meta"), reward_reason=value.get("reward_reason"))

    # Derived training view only; never serialized into sealed rollout evidence.
    supervision_positions: tuple[int, ...] | None = None

    def point(self, *, model_fingerprint: str, feature_family: str,
              base_fingerprint: str, tokenizer_fingerprint: str, site: str,
              width: int, dtype: str, birth_step: int = 0) -> HiddenPoint | None:
        if self.hidden is None or self.target is None:
            return None
        return HiddenPoint(point_id=self.row_id, hidden=list(self.hidden), target=float(self.target),
                           model_fingerprint=model_fingerprint, feature_family=feature_family,
                           base_fingerprint=base_fingerprint, tokenizer_fingerprint=tokenizer_fingerprint,
                           site=site, width=width, dtype=dtype, birth_step=birth_step,
                           split=(self.hidden_meta or {}).get("split"))


@dataclass(frozen=True)
class OnlineRolloutBatch:
    step: int
    tasks: tuple[str, ...]
    policy_version: str
    estimator_version: str
    rows: tuple[OnlineRow, ...]
    resolved_points: tuple[HiddenPoint, ...] = ()
    request_sha256: str | None = None
    excluded_groups: tuple[Mapping[str, Any], ...] = ()
    discarded_trajectories: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OnlineRolloutBatch":
        if value.get("schema") != ONLINE_ROLLOUT_SCHEMA:
            raise ValueError("unsupported online rollout batch schema")
        rows = tuple(OnlineRow.from_dict(row) for row in value.get("rows", ()))
        if not rows and not value.get("discarded_trajectories"):
            raise ValueError("online rollout batch is empty")
        tasks = tuple(str(x) for x in value.get("tasks", ()))
        if not tasks:
            tasks = tuple(dict.fromkeys(row.task_id for row in rows))
        if any(row.task_id not in tasks for row in rows):
            raise ValueError("rollout row task is outside batch task list")
        points = []
        for raw in value.get("resolved_points", ()):
            points.append(HiddenPoint(**raw))
        return cls(step=int(value["step"]), tasks=tasks,
                   policy_version=str(value.get("policy_version", "S0")),
                   estimator_version=str(value.get("estimator_version", "E0")),
                   rows=rows, resolved_points=tuple(points), excluded_groups=tuple(value.get("excluded_groups", ())),
                   discarded_trajectories=tuple(value.get("discarded_trajectories", ())),
                   request_sha256=(None if value.get("request_sha256") is None else str(value["request_sha256"])))

    def as_dict(self) -> dict[str, Any]:
        return {"schema": ONLINE_ROLLOUT_SCHEMA, "step": self.step, "tasks": list(self.tasks),
                "policy_version": self.policy_version, "estimator_version": self.estimator_version,
                "rows": [row.__dict__ for row in self.rows],
                "resolved_points": [point.__dict__ for point in self.resolved_points],
                "request_sha256": self.request_sha256, "excluded_groups": list(self.excluded_groups),
                **({"discarded_trajectories": list(self.discarded_trajectories)} if self.discarded_trajectories else {})}


class RolloutProvider(Protocol):
    def __call__(self, tasks: Sequence[str], step: int, policy_version: str | None,
                 estimator_version: str | None) -> OnlineRolloutBatch: ...


class HttpRolloutClient:
    """Small JSON client for the project's shared online rollout service.

    The service owns dycon run/serving/evaluation lifecycles; this client only
    submits the signed work manifest and returns the sealed batch payload. It
    deliberately uses the standard library so the RL trainer has no extra
    runtime dependency and remains easy to exercise with a local HTTP stub.
    """
    def __init__(self, endpoint: str, *, timeout_seconds: float = 120.0,
                 headers: Mapping[str, str] | None = None):
        endpoint = str(endpoint).strip()
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("rollout endpoint must be an http(s) URL")
        if float(timeout_seconds) <= 0:
            raise ValueError("rollout timeout must be positive")
        self.endpoint = endpoint
        self.timeout_seconds = float(timeout_seconds)
        self.headers = {"Content-Type": "application/json", **dict(headers or {})}

    def __call__(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps(dict(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
        request = Request(self.endpoint, data=body, headers=self.headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
                status = getattr(response, "status", 200)
        except HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"shared rollout service HTTP {exc.code}: {detail[:1000]}") from exc
        except URLError as exc:
            raise RuntimeError(f"shared rollout service unavailable: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("shared rollout service request timed out") from exc
        if status < 200 or status >= 300:
            raise RuntimeError(f"shared rollout service returned HTTP {status}")
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("shared rollout service returned invalid JSON") from exc
        if not isinstance(result, Mapping):
            raise ValueError("shared rollout service response must be a JSON object")
        return result


class SharedVllmRolloutProvider:
    """Adapter boundary for one shared vLLM service and mixed work queue.

    ``request`` owns the project's serving/run lifecycle and receives one
    ordered work manifest. It must return an ``online_bootstrap_rollout_batch_v1``
    payload; no subprocess or ad-hoc launcher is created here.
    """
    def __init__(self, request, *, m: int = 8, dycon_candidate_portion: int = 1,
                 max_inflight: int = 1, estimate_seconds=None,
                 worker_ids: Sequence[int | str] = ("shared",), deployment_resolver=None):
        self.request = request
        self.deployment_resolver = deployment_resolver
        self.m = int(m)
        self.dycon_candidate_portion = int(dycon_candidate_portion)
        self.max_inflight = int(max_inflight)
        self.worker_ids = tuple(worker_ids)
        if not self.worker_ids or len(set(self.worker_ids)) != len(self.worker_ids):
            raise ValueError("worker_ids must be non-empty and unique")
        self.estimate_seconds = estimate_seconds or (lambda _task, _source, _rep: 0.0)

    def __call__(self, tasks: Sequence[str], step: int, policy_version: str | None,
                 estimator_version: str | None) -> OnlineRolloutBatch:
        items = []
        for task in tasks:
            items.append(RolloutWorkItem(f"{task}:dycon", task, "dycon",
                                         float(self.estimate_seconds(task, "dycon", 0)), 0))
            for repetition in range(self.m):
                items.append(RolloutWorkItem(f"{task}:baseline:{repetition}", task, "baseline",
                                             float(self.estimate_seconds(task, "baseline", repetition))))
        ordered = MixedRolloutScheduler(items, max_inflight=self.max_inflight,
                                        dycon_candidate_portion=self.dycon_candidate_portion).order()
        assigned = MixedRolloutScheduler.assign_workers(ordered, self.worker_ids)
        payload = {"schema": ONLINE_ROLLOUT_SCHEMA, "step": step, "tasks": list(tasks),
                   "policy_version": policy_version or "S0",
                   "estimator_version": estimator_version or "E0",
                   "max_inflight": self.max_inflight,
                   "worker_ids": list(self.worker_ids),
                   "work_items": [{"trajectory_id": item.trajectory_id, "task_id": item.task_id,
                                   "source": item.source, "candidate_index": item.candidate_index,
                                   "worker_id": worker}
                                  for item, worker in assigned]}
        if self.deployment_resolver is not None:
            payload["deployment"] = self.deployment_resolver(policy_version or "S0", estimator_version or "E0")
        payload["request_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        result = self.request(payload)
        if result.get("request_sha256") is not None and result.get("request_sha256") != payload["request_sha256"]:
            raise ValueError("shared vLLM response request identity differs from work manifest")
        batch = OnlineRolloutBatch.from_dict(result)
        if batch.step != step or batch.tasks != tuple(tasks):
            raise ValueError("shared vLLM response does not match requested batch")
        if batch.request_sha256 is not None and batch.request_sha256 != payload["request_sha256"]:
            raise ValueError("online rollout batch request identity differs from work manifest")
        validate_trajectory_coverage(batch, payload)
        return batch


def validate_trajectory_coverage(batch: OnlineRolloutBatch, request: Mapping[str, Any]) -> None:
    """Validate trajectories, allowing many decisions/branches/steps per trajectory."""
    if batch.step != int(request["step"]) or batch.tasks != tuple(request["tasks"]):
        raise ValueError("online response step/tasks differ from request")
    if (batch.policy_version, batch.estimator_version) != (request["policy_version"], request["estimator_version"]):
        raise ValueError("online response policy/estimator differs from generating versions")
    if batch.request_sha256 != request["request_sha256"]:
        raise ValueError("online response is not bound to request identity")
    expected = {str(item["trajectory_id"]): (str(item["task_id"]), str(item["source"]))
                for item in request["work_items"]}
    observed = set()
    row_ids = set()
    for row in batch.rows:
        if row.row_id in row_ids:
            raise ValueError("duplicate online row identity")
        row_ids.add(row.row_id)
        if row.trajectory_id not in expected or expected[row.trajectory_id] != (row.task_id, row.source):
            raise ValueError("online row trajectory differs from requested work manifest")
        if row.reward_reason is not None:
            policy = (request.get('deployment') or {}).get('failure_policy') or {}
            if (row.source != 'baseline' or row.reward_reason not in {'LimitsExceeded', 'reasoning_loop'}
                    or row.advantage != policy.get('baseline_failure_reward') or row.resolved is not False):
                raise ValueError('online baseline failure reward differs from bound policy')
        observed.add(row.trajectory_id)
    excluded_ids = set()
    for group in batch.excluded_groups:
        trajectory = group.get("trajectory_id")
        if (expected.get(trajectory) != (group.get("task_id"), "dycon") or
                group.get("reason") not in {"all_scores_equal", "top_k_boundary_tie", "missing_or_nonfinite_score", "no_consensus_in_top_half"}
                or not group.get("group_id")):
            raise ValueError("invalid excluded online decision evidence")
        excluded_ids.add(trajectory)
    discarded_ids = set()
    policy = (request.get('deployment') or {}).get('failure_policy') or {}
    for item in batch.discarded_trajectories:
        tid = item.get('trajectory_id')
        retry_limit = item.get('retry_limit')
        if (tid in discarded_ids or tid in observed or tid in excluded_ids
                or expected.get(tid) != (item.get('task_id'), item.get('source'))
                or item.get('reason') != 'verifier_error_retries_exhausted'
                or not item.get('evaluation_id') or not item.get('rollout_id')
                or type(retry_limit) is not int or retry_limit < 0
                or retry_limit != policy.get('verifier_error_retries')
                or item.get('attempt_count') != retry_limit + 1):
            raise ValueError('invalid discarded online trajectory evidence')
        discarded_ids.add(tid)
    if observed | excluded_ids | discarded_ids != set(expected):
        raise ValueError("shared vLLM response has incomplete trajectory coverage")
    online_row_masses(batch.rows, SourceMass())


class JsonlRolloutProvider:
    """Replay a sealed rollout manifest through the same coordinator contract."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._batches: dict[int, OnlineRolloutBatch] = {}
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    batch = OnlineRolloutBatch.from_dict(json.loads(line))
                except Exception as exc:
                    raise ValueError(f"invalid rollout manifest line {line_number}: {exc}") from exc
                if batch.step in self._batches:
                    raise ValueError(f"duplicate rollout step {batch.step}")
                self._batches[batch.step] = batch
        if not self._batches:
            raise ValueError("rollout manifest is empty")

    def __call__(self, tasks: Sequence[str], step: int, policy_version: str | None,
                 estimator_version: str | None) -> OnlineRolloutBatch:
        if step not in self._batches:
            raise IndexError(f"rollout manifest has no step {step}")
        batch = self._batches[step]
        if tuple(tasks) != batch.tasks:
            raise ValueError(f"rollout task schedule mismatch at step {step}")
        # A prefetched batch is intentionally one policy/estimator version old
        # at the next barrier. OnlineCoordinator validates the permitted
        # one-step lineage; the provider must not reject that legal lag.
        return batch


def _group_spans(rows: Sequence[OnlineRow], source: str) -> list[list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if row.source == source:
            groups.setdefault(row.group_id if source == "dycon" else row.task_id, []).append(index)
    return list(groups.values())


def effective_source_mass(batch, source_mass):
    if not batch.discarded_trajectories or not batch.rows:
        return source_mass
    present = {row.source for row in batch.rows}
    alpha, beta = source_mass.normalized()
    if (alpha if 'dycon' in present else 0) + (beta if 'baseline' in present else 0) == 0:
        return source_mass
    return SourceMass(alpha if 'dycon' in present else 0., beta if 'baseline' in present else 0.)


def online_row_masses(rows: Sequence[OnlineRow], source_mass: SourceMass) -> list[float]:
    """Task -> decision -> branch; task -> trajectory -> completion-token means.

    Each returned coefficient multiplies that row's mean completion-token loss.
    Coefficients are computed once for the complete logical batch, then reused
    during sequential forwards so accumulating gradients preserves the objective.
    """
    masses = [0.0] * len(rows)
    for source, mass in zip(("dycon", "baseline"), source_mass.normalized()):
        tasks = {}
        for index, row in enumerate(rows):
            if row.source == source:
                tasks.setdefault(row.task_id, []).append(index)
        for indices in tasks.values():
            groups = {}
            for index in indices:
                row = rows[index]
                key = row.group_id if source == "dycon" else row.trajectory_id
                if not key:
                    raise ValueError("online baseline rows require trajectory_id")
                groups.setdefault(key, []).append(index)
            for members in groups.values():
                if source == "dycon":
                    for index in members:
                        masses[index] = mass / len(tasks) / len(groups) / len(members)
                else:
                    total = sum(len(rows[i].completion_token_ids) for i in members)
                    declared = {rows[i].trajectory_token_count for i in members}
                    if declared != {total}:
                        raise ValueError("baseline trajectory completion coverage differs from declared token count")
                    steps = [rows[i].agent_step for i in members]
                    if None in steps or len(steps) != len(set(steps)):
                        raise ValueError("baseline rows require unique agent_step per trajectory")
                    if len({rows[i].advantage for i in members}) != 1:
                        raise ValueError("baseline trajectory has inconsistent advantages")
                    for index in members:
                        masses[index] = mass / len(tasks) / len(groups) * len(rows[index].completion_token_ids) / total
    return masses


def online_torch_objective(logp_current: torch.Tensor, rollout_logp: torch.Tensor,
                           kl_reference: torch.Tensor | None, advantages: torch.Tensor,
                           loss_mask: torch.Tensor, sources: Sequence[str],
                           group_spans: Mapping[str, Sequence[Sequence[int]]], *,
                           row_weights: Sequence[float] | None = None,
                           source_mass: SourceMass, clip_epsilon: float = 0.1,
                           kl_beta: float = 0.0, k3_cap: float | None = None,
                           differentiable_terms: dict | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Differentiable online objective with independent per-source mass.

    DyCon groups are step groups; baseline groups are task groups whose rows are
    agent-step completions. The rollout denominator and KL reference are detached.
    """
    import torch
    if logp_current.shape != rollout_logp.shape or logp_current.ndim != 2:
        raise ValueError("online logprob tensors must have equal [row,token] shapes")
    mask = loss_mask.to(device=logp_current.device, dtype=torch.bool)
    if mask.shape != logp_current.shape:
        raise ValueError("online loss mask shape mismatch")
    adv = advantages.to(device=logp_current.device, dtype=torch.float32).detach()
    if adv.numel() != logp_current.shape[0]:
        raise ValueError("online advantage count differs from rows")
    if kl_beta and kl_reference is None:
        raise ValueError("KL reference is required when kl_beta > 0")
    if row_weights is None:
        row_weights = [1.0] * len(sources)
    if len(row_weights) != len(sources) or any(float(x) <= 0 for x in row_weights):
        raise ValueError("online row weights must be positive and row-aligned")
    row_weights_tensor = torch.as_tensor(row_weights, device=logp_current.device, dtype=torch.float32)
    theta = logp_current.float()
    denominator = rollout_logp.float().detach()
    ratio = torch.exp((theta - denominator).clamp(-80, 80))
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    surrogate = torch.minimum(ratio * adv[:, None], clipped * adv[:, None])
    delta = (kl_reference.float().detach() - theta) if kl_reference is not None else torch.zeros_like(theta)
    raw_k3 = torch.expm1(delta.clamp(-80, 80)) - delta
    if k3_cap is not None:
        raw_k3 = raw_k3.clamp_max(float(k3_cap))
    alpha, beta = source_mass.normalized()
    source_losses = {}
    source_k3 = {}
    total = theta.new_zeros(())
    kl_total = theta.new_zeros(())
    for source, weight in (("dycon", alpha), ("baseline", beta)):
        spans = [list(span) for span in group_spans.get(source, ())]
        if not spans or weight == 0:
            continue
        group_values = []
        group_k3 = []
        for indices in spans:
            idx = torch.as_tensor(indices, device=theta.device, dtype=torch.long)
            values = surrogate[idx].masked_fill(~mask[idx], 0)
            counts = mask[idx].sum(dim=1).clamp_min(1)
            branch = values.sum(dim=1) / counts
            weights = row_weights_tensor[idx]
            # Baseline rows are agent-step completions. Weight each step by
            # completion_tokens / trajectory_tokens so one trajectory has one
            # token-normalized contribution, then average repetitions in the
            # task group. DyCon rows retain equal step-group reduction.
            if source == "baseline":
                group_values.append((branch * weights).sum() / weights.sum())
            else:
                group_values.append(branch.mean())
            k3_values = raw_k3[idx].masked_fill(~mask[idx], 0).sum(dim=1) / counts
            group_k3.append((k3_values * weights).sum() / weights.sum() if source == "baseline" else k3_values.mean())
        policy = -torch.stack(group_values).mean()
        penalty = torch.stack(group_k3).mean()
        source_losses[source] = float(policy.detach())
        source_k3[source] = float(penalty.detach())
        kl_total = kl_total + float(weight) * float(kl_beta) * penalty
        total = total + float(weight) * (policy + float(kl_beta) * penalty)
    if differentiable_terms is not None:
        differentiable_terms["weighted_kl"] = kl_total
    if not torch.isfinite(total):
        raise ValueError("nonfinite online objective")
    return total, {"loss": float(total.detach()), "source_losses": source_losses,
                   "source_k3": source_k3, "ratio_mean": float(ratio[mask].mean().detach()),
                   "token_count": int(mask.sum()), "row_count": len(sources),
                   "dycon_rows": sum(x == "dycon" for x in sources),
                   "baseline_rows": sum(x == "baseline" for x in sources),
                   "source_mass": {"dycon": alpha, "baseline": beta}}


def collate_rows(rows: Sequence[OnlineRow], *, device: torch.device) -> dict[str, Any]:
    import torch
    if not rows:
        raise ValueError("cannot collate empty online rows")
    if any(len(row.input_token_ids) <= len(row.completion_token_ids) for row in rows):
        raise ValueError("online rows require a nonempty differentiable prefix")
    max_length = max(len(row.input_token_ids) for row in rows)
    batch = len(rows)
    input_ids = torch.zeros((batch, max_length), dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    rollout = torch.zeros((batch, max_length - 1), dtype=torch.float32, device=device)
    kl = torch.zeros_like(rollout)
    has_kl = all(row.kl_reference_logprobs is not None for row in rows)
    if any((row.kl_reference_logprobs is None) != (not has_kl) for row in rows):
        raise ValueError("online batch mixes rows with and without KL references")
    for i, row in enumerate(rows):
        length = len(row.input_token_ids)
        input_ids[i, :length] = torch.tensor(row.input_token_ids, device=device)
        attention[i, :length] = 1
        start = length - len(row.completion_token_ids)
        target_positions = row.supervision_positions or tuple(range(start, length))
        if (len(target_positions) != len(row.completion_token_ids)
                or tuple(sorted(set(target_positions))) != target_positions
                or any(p < 1 or p >= length for p in target_positions)
                or tuple(row.input_token_ids[p] for p in target_positions) != row.completion_token_ids):
            raise ValueError("invalid trace supervision positions")
        target_positions = torch.tensor(target_positions, device=device)
        labels[i, target_positions] = torch.tensor(row.completion_token_ids, device=device)
        mask[i, target_positions] = True
        # selected_logprobs consumes labels[:, 1:], hence completion at position
        # start is read from logits at start-1 and stored at start-1.
        positions = target_positions - 1
        rollout[i, positions] = torch.tensor(row.rollout_logprobs, device=device)
        if has_kl:
            kl[i, positions] = torch.tensor(row.kl_reference_logprobs, device=device)
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels,
            "loss_mask": mask, "rollout_logprobs": rollout,
            "kl_reference_logprobs": kl if has_kl else None}


def online_selected_logprobs(model, tensors, block_size=16):
    """Use the shared chunked head while retaining the complete prefix graph."""
    import torch
    if hasattr(model, "get_output_embeddings"):
        from RL.grpo.logprobs import model_target_logprobs
        values = model_target_logprobs(model, tensors, block_size=block_size)
        mask = tensors["loss_mask"][:, 1:].bool()
        selected = torch.zeros(mask.shape, dtype=values.dtype, device=values.device)
        return selected.masked_scatter(mask, values), mask
    # Small protocol models used by numerical tests can expose logits directly.
    from RL.grpo.objective import selected_logprobs
    outputs = model(input_ids=tensors["input_ids"], attention_mask=tensors["attention_mask"])
    return selected_logprobs(outputs.logits, tensors["input_ids"], tensors["labels"], tensors["loss_mask"])


class TorchOnlineUpdater:
    """Single-GPU LoRA updater for one prefetched online batch."""
    def __init__(self, model, settings, *, source_mass: SourceMass, kl_beta: float = 0.0,
                 clip_epsilon: float = 0.1, k3_cap: float | None = None, reference_scorer=None,
                 on_stage=None, baseline_training_mode="agent_step", max_trace_input_tokens=32768):
        from RL.training import optimizer_and_scheduler
        self.model = model
        self.settings = settings
        self.optimizer, self.scheduler = optimizer_and_scheduler(model, settings)
        self.source_mass = source_mass
        self.kl_beta = kl_beta
        self.clip_epsilon = clip_epsilon
        self.k3_cap = k3_cap
        self.reference_scorer = reference_scorer
        self.on_stage = on_stage
        if baseline_training_mode not in {"agent_step", "prefix_trace"}:
            raise ValueError("invalid baseline training mode")
        self.baseline_training_mode = baseline_training_mode
        self.max_trace_input_tokens = max_trace_input_tokens

    def _stage(self, name, **fields):
        if self.on_stage is not None:
            try:
                self.on_stage(name, **fields)
            except Exception as exc:
                import warnings
                warnings.warn("Online stage callback failed: " + str(exc))

    def __call__(self, batch: OnlineRolloutBatch, step: int, policy_version: str | None = None,
                 estimator_version: str | None = None) -> dict[str, Any]:
        if batch.step != step:
            raise ValueError("online update step does not match rollout batch")
        import torch
        import time
        started = time.monotonic()
        from RL.grpo.objective import selected_logprobs
        self.model.train()
        device = next(self.model.parameters()).device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        rows = batch.rows
        source_mass = effective_source_mass(batch, self.source_mass)
        masses = online_row_masses(rows, source_mass)
        if not rows or not any(masses):
            self.optimizer.zero_grad(set_to_none=True)
            return {'step': step, 'loss': 0., 'gradient_norm': 0., 'row_count': 0,
                    'token_count': 0, 'update_seconds': time.monotonic() - started,
                    'skipped_update': True, 'skip_reason': 'no_weighted_valid_trajectories',
                    'discarded_trajectories': list(batch.discarded_trajectories),
                    'policy_version': policy_version, 'estimator_version': estimator_version,
                    'rollout_policy_version': batch.policy_version,
                    'rollout_estimator_version': batch.estimator_version}
        original_rows = rows
        trace_receipt = None
        build_started = time.monotonic()
        if self.baseline_training_mode == "prefix_trace":
            from RL.online_trace import build_training_traces
            rows, masses, trace_receipt = build_training_traces(
                rows, masses, max_input_tokens=self.max_trace_input_tokens)
            for module in self.model.modules():
                if isinstance(module, torch.nn.Dropout) and module.p:
                    raise ValueError("prefix_trace requires disabled training dropout")
        build_seconds = time.monotonic() - build_started
        self.optimizer.zero_grad(set_to_none=True)
        alpha, beta = source_mass.normalized()
        metrics = {"loss": 0.0, "source_losses": {}, "source_k3": {},
                   "token_count": 0, "row_count": len(rows),
                   "dycon_rows": sum(r.source == "dycon" for r in rows),
                   "baseline_rows": sum(r.source == "baseline" for r in rows),
                   "source_mass": {"dycon": alpha, "baseline": beta},
                   "forward_rows_max": 1}
        if trace_receipt is not None:
            metrics["trace_training"] = trace_receipt
        metrics["discarded_trajectories"] = list(batch.discarded_trajectories)
        metrics["trace_build_seconds"] = build_seconds
        metrics["original_row_count"] = len(original_rows)
        metrics["source_compute_seconds"] = {}
        ratio_sum = 0.0
        from RL.gradient_attribution import GradientAttribution
        attribution = GradientAttribution([p for p in self.model.parameters() if p.requires_grad])
        for ordinal, (row, coefficient) in enumerate(zip(rows, masses)):
            if not coefficient:
                continue
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row_started = time.monotonic()
            tensors = collate_rows([row], device=device)
            self._stage("online_row_forward", inflight_update=step + 1,
                        online_row=ordinal, online_rows_total=len(rows),
                        online_source=row.source, input_tokens=len(row.input_token_ids))
            if self.kl_beta and tensors["kl_reference_logprobs"] is None and self.reference_scorer is not None:
                tensors["kl_reference_logprobs"] = self.reference_scorer(tensors)
            # Full historical context stays in the autograd graph. Each row is
            # an agent-step completion, never an independently detached suffix.
            current, mask = online_selected_logprobs(self.model, tensors,
                block_size=getattr(getattr(self.settings, "memory", None), "loss_chunk_tokens", 16))
            unit_mass = SourceMass(1.0, 0.0) if row.source == "dycon" else SourceMass(0.0, 1.0)
            terms = {}
            loss, local = online_torch_objective(
                current, tensors["rollout_logprobs"], tensors["kl_reference_logprobs"],
                torch.tensor([row.advantage], device=device), mask, [row.source],
                {row.source: [[0]]}, source_mass=unit_mass,
                clip_epsilon=self.clip_epsilon, kl_beta=self.kl_beta, k3_cap=self.k3_cap,
                differentiable_terms=terms)
            self._stage("online_row_backward", online_row=ordinal)
            attribution.backward(loss * coefficient, row.source,
                                 terms["weighted_kl"] * coefficient if self.kl_beta else None)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            metrics["source_compute_seconds"][row.source] = metrics["source_compute_seconds"].get(row.source, 0.) + time.monotonic() - row_started
            metrics["loss"] += coefficient * local["loss"]
            mass = alpha if row.source == "dycon" else beta
            for field in ("source_losses", "source_k3"):
                metrics[field][row.source] = metrics[field].get(row.source, 0.0) + coefficient / mass * local[field][row.source]
            ratio_sum += local["ratio_mean"] * local["token_count"]
            metrics["token_count"] += local["token_count"]
            del current, mask, tensors, loss, terms
        metrics["ratio_mean"] = ratio_sum / max(metrics["token_count"], 1)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self._stage("optimizer_update", online_row=len(rows), online_rows_total=len(rows))
        metrics["gradient_attribution"] = attribution.summary()
        del attribution
        optimizer_started = time.monotonic()
        norm = torch.nn.utils.clip_grad_norm_(params, float(self.settings.grad_clip), error_if_nonfinite=True)
        self.optimizer.step(); self.scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            metrics["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            metrics["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        metrics["optimizer_seconds"] = time.monotonic() - optimizer_started
        metrics.update({"step": step, "gradient_norm": float(norm),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "policy_version": policy_version, "estimator_version": estimator_version,
                        "rollout_policy_version": batch.policy_version,
                        "rollout_estimator_version": batch.estimator_version,
                        "excluded_groups": len(batch.excluded_groups),
                        "update_seconds": time.monotonic() - started})
        return metrics


def manifest_sha256(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

from core.io import hash_json, owned_path, read_json, sha256, write_json
from RL.online_evidence import LocalEvidenceError, native_baseline_row, online_terminal_label, _online_outcome
from RL.online_failures import verifier_discard
from RL.online_runtime import OnlineRow
from rollout.retry import RETRY_STATUSES, default_final_ids

TASK_GRPO_VERSION = "dataset_task_grpo_v1"
TASK_GRPO_EPSILON = 1e-8
TASK_REWARD_STATS_FILE = "task-reward-stats.json"


def prepare(root, run_ids, destination, failure_reward):
    from application.execution.evidence import RunEvidence
    from application.execution.run_plan import load_plan
    from hidden.artifacts import response_records

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "trajectories").mkdir()
    entries, reports = [], {}
    for run_id in dict.fromkeys(run_ids):
        loaded = load_plan(root, run_id)
        cfg, run = loaded["config"], loaded["run"]
        evidence = RunEvidence(root, run_id)
        data = evidence.read()
        work_items = loaded["work_items"]["items"]
        accepted = default_final_ids(root, run_id, data, plan=loaded, verify_files=False)
        if accepted is None:
            accepted = data["rollouts"]
        dycon_samples = {item["sample_key"] for item in work_items if item.get("online_source") == "dycon"}
        candidates = {rid: None for rid in accepted
                      if data["rollouts"][rid].get("sample_key") not in dycon_samples}
        inventories = {}
        counts = Counter()
        for item in work_items:
            if item["sample_key"] in dycon_samples:
                continue
            work = run / "rollout/tasks" / item["instance_id"] / "repetitions" / item["repetition_id"]
            for path in sorted(work.glob("attempts/*/attempt.json")):
                attempt = read_json(path)
                status = attempt.get("agent_exit_status")
                if status not in RETRY_STATUSES:
                    continue
                rid = attempt.get("rollout_id")
                if not rid:
                    counts["missing_physical_rollout"] += 1
                    continue
                row = data["rollouts"].get(rid)
                if row is None or any(row.get(k) != attempt.get(k) for k in ("sample_key", "attempt_id")):
                    raise ValueError("Physical attempt differs from registered rollout")
                if rid in candidates:
                    counts["deduplicated"] += 1
                candidates[rid] = status
        native_root = Path(cfg["paths"]["nas_root"]) / cfg["capture"]["export"]["root"] / run_id
        deployment = {"adapter_identity_sha256": (cfg["serving"].get("lora") or {}).get("identity_sha256")}
        request = {"deployment": deployment, "policy_version": "S0"}
        for rid, failure in sorted(candidates.items()):
            row = data["rollouts"][rid]
            task = data["tasks"][row["task_key"]]
            if task["partition"] == "excluded" or row.get("provenance", {}).get("quarantined"):
                counts["excluded"] += 1
                continue
            failure = failure or (row.get("termination_reason") if row.get("termination_reason") in RETRY_STATUSES else None)
            resolved = False if failure else online_terminal_label(row, data, rid)
            if not failure and (type(resolved) is not bool or verifier_discard(_online_outcome(row, data, rid))):
                counts["missing_valid_evaluation"] += 1
                continue
            path = owned_path(root, row["trajectory_path"])
            key = hash_json([run_id, rid])
            try:
                trajectory = read_json(path)
                responses = list(response_records(trajectory))
                if any((message.get("extra") or {}).get("dycon_parallel") for _, message, _ in responses):
                    counts["dycon_trajectory"] += 1
                    continue
                tokenizers = set()
                rows = [native_baseline_row(native_root, response, cfg, request,
                            task_id=task["instance_id"], trajectory_id=key, ordinal=ordinal,
                            resolved=resolved, reward_override=failure_reward if failure else None,
                            reward_reason=failure, tokenizers=tokenizers,
                            offline_export=True, inventories=inventories)
                        for ordinal, (_, _, response) in enumerate(responses)]
                if not rows:
                    raise LocalEvidenceError("missing_native_agent_responses")
            except (LocalEvidenceError, FileNotFoundError) as exc:
                counts[str(exc) if isinstance(exc, LocalEvidenceError) else "missing_capture"] += 1
                continue
            if len(tokenizers) != 1 or None in tokenizers:
                raise ValueError("Missing or inconsistent baseline tokenizer identity")
            total = sum(len(row.completion_token_ids) for row in rows)
            shard = destination / "trajectories" / (key + ".json")
            write_json(shard, _encode_rows(rows, total))
            entries.append({"id": key, "task": task["instance_id"], "failure": failure,
                            "run": run_id, "rollout": rid, "trajectory_sha256": row["trajectory_sha256"],
                            "bytes": shard.stat().st_size, "tokens": total, "raw_reward": rows[0].advantage,
                            "model": cfg["model"]["fingerprint"], "tokenizer": next(iter(tokenizers))})
            counts["imported_" + (failure or ("resolved" if resolved else "unresolved"))] += 1
        reports[run_id] = {"evidence_revision": data.get("revision"), "counts": dict(counts)}
    manifest = {"schema_version": 2, "probability": "raw_t1", "runs": reports, "entries": entries}
    write_json(destination / "manifest.json", manifest)
    return manifest


def _encode_rows(rows, total, *, initial_prefix=()):
    """Store only the token suffix after the common prefix with the previous step."""
    encoded = []
    previous = tuple(initial_prefix)
    for row in rows:
        tokens = row.input_token_ids
        prefix = next((i for i, (a, b) in enumerate(zip(previous, tokens)) if a != b),
                      min(len(previous), len(tokens)))
        step = asdict(replace(row, input_token_ids=tokens[prefix:], trajectory_token_count=total))
        step["prefix_tokens"] = prefix
        encoded.append(step)
        previous = tokens
    return encoded


def _decode_rows(data, version, *, initial_prefix=()):
    previous = tuple(initial_prefix)
    for step in data:
        if version == 2:
            prefix = step["prefix_tokens"]
            if type(prefix) is not int or not 0 <= prefix <= len(previous):
                raise ValueError("Invalid offline baseline prefix length")
            previous = previous[:prefix] + tuple(step["input_token_ids"])
            step["input_token_ids"] = previous
        yield OnlineRow.from_dict(step)



def _manifest_snapshot(dataset):
    payload = (Path(dataset) / "manifest.json").read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def _read_shard(dataset, entry):
    shard = owned_path(dataset, "trajectories/" + entry["id"] + ".json")
    if "bytes" in entry and shard.stat().st_size != entry["bytes"]:
        raise ValueError("Offline baseline trajectory size changed")
    return read_json(shard)


def _finite_reward(value):
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError("Offline baseline raw reward must be a finite number")
    return float(value)


def _trajectory_reward(dataset, entry):
    if "raw_reward" in entry:
        return _finite_reward(entry["raw_reward"])
    steps = _read_shard(dataset, entry)
    if not steps:
        raise ValueError(f"Empty baseline trajectory: {entry['id']}")
    rewards = set()
    for row in steps:
        if (row.get("source"), row.get("task_id"), row.get("trajectory_id")) != (
                "baseline", entry["task"], entry["id"]):
            raise ValueError(f"Baseline trajectory identity mismatch: {entry['id']}")
        rewards.add(_finite_reward(row["advantage"]))
    if len(rewards) != 1:
        raise ValueError(f"Baseline trajectory has inconsistent raw rewards: {entry['id']}")
    return rewards.pop()


def _reward_summary(histogram):
    distribution = sorted(histogram.items())
    count = sum(n for _, n in distribution)
    # Population statistics with equal mass per trajectory.
    mean = math.fsum(r * (n / count) for r, n in distribution)
    std = math.hypot(*(math.sqrt(n / count) * (r - mean) for r, n in distribution))
    return {"count": count, "mean": mean, "std": std, "histogram": [list(x) for x in distribution]}


def _load_task_reward_stats(dataset, manifest, digest):
    path = Path(dataset) / TASK_REWARD_STATS_FILE
    if not path.is_file():
        raise ValueError(f"Missing {path}; run dycon rl-prepare-baseline-stats --dataset {dataset}")
    stats = read_json(path)
    if stats.get("version") != TASK_GRPO_VERSION or stats.get("manifest_sha256") != digest:
        raise ValueError("Baseline task reward statistics do not match this dataset snapshot; "
                         "run dycon rl-prepare-baseline-stats --dataset " + str(dataset))
    _validate_task_reward_stats(stats, manifest)
    return stats


def _validate_task_reward_stats(stats, manifest):
    counts = Counter(entry["task"] for entry in manifest["entries"])
    tasks = stats["tasks"]
    if not counts or set(counts) != set(tasks):
        raise ValueError("Invalid baseline task reward statistics")
    for task, row in tasks.items():
        histogram = {}
        for reward, count in row["histogram"]:
            reward = _finite_reward(reward)
            if type(count) is not int or count < 1 or reward in histogram:
                raise ValueError(f"Invalid baseline reward histogram: {task}")
            histogram[reward] = count
        if sum(histogram.values()) != counts[task] or row != _reward_summary(histogram):
            raise ValueError(f"Invalid baseline reward statistics: {task}")


def prepare_task_reward_stats(dataset):
    """Seal one task reward distribution per frozen dataset (no token or shard hashing)."""
    from core.locking import file_lock

    dataset = Path(dataset)
    path = dataset / TASK_REWARD_STATS_FILE
    with file_lock(dataset / ".task-reward-stats.lock"):
        manifest, digest = _manifest_snapshot(dataset)
        if manifest["schema_version"] not in {1, 2} or manifest["probability"] != "raw_t1":
            raise ValueError("Unsupported offline baseline manifest")
        if path.is_file():
            existing = read_json(path)
            if existing.get("version") == TASK_GRPO_VERSION and existing.get("manifest_sha256") == digest:
                _validate_task_reward_stats(existing, manifest)
                return existing
        histograms, seen = defaultdict(Counter), set()
        for entry in manifest["entries"]:
            if entry["id"] in seen:
                raise ValueError("Duplicate offline baseline trajectory")
            seen.add(entry["id"])
            histograms[entry["task"]][_trajectory_reward(dataset, entry)] += 1
        if not seen:
            raise ValueError("Baseline dataset has no trajectories")
        stats = {"version": TASK_GRPO_VERSION, "manifest_sha256": digest,
                 "tasks": {task: _reward_summary(hist) for task, hist in sorted(histograms.items())}}
        if sha256(dataset / "manifest.json") != digest:
            raise ValueError("Baseline dataset manifest changed while preparing reward statistics")
        write_json(path, stats)
        return stats


def validate_advantage_stats(settings):
    if settings is not None and settings.baseline_weight and settings.advantage_mode == "task_grpo":
        dataset = Path(settings.dataset)
        manifest, digest = _manifest_snapshot(dataset)
        _load_task_reward_stats(dataset, manifest, digest)


def _active_task_ids(stats):
    return {task for task, summary in stats["tasks"].items() if summary["std"] > 0.0}


def _task_grpo_rows(rows, stats):
    values = {}
    output = []
    for row in rows:
        key = (row.task_id, row.trajectory_id)
        if key not in values:
            task = stats["tasks"][row.task_id]
            raw = row.advantage
            if not any(reward == raw for reward, _ in task["histogram"]):
                raise ValueError(f"Baseline reward differs from task statistics: {key}")
            advantage = 0.0 if task["std"] == 0.0 else (raw - task["mean"]) / (task["std"] + TASK_GRPO_EPSILON)
            values[key] = (raw, advantage)
        raw, advantage = values[key]
        if row.advantage != raw:
            raise ValueError(f"Baseline trajectory has inconsistent raw rewards: {key}")
        output.append(replace(row, advantage=advantage))
    raw_rewards, advantages = zip(*values.values())
    return tuple(output), {
        "mode": "task_grpo", "trajectories": len(values), "tasks": len({task for task, _ in values}),
        "raw_reward_min": min(raw_rewards), "raw_reward_max": max(raw_rewards),
        "advantage_min": min(advantages), "advantage_max": max(advantages),
        "zero_advantage_trajectories": sum(value == 0.0 for value in advantages),
    }


class BaselinePool:
    def __init__(self, path, *, seed, n, m, coverage_window=10, targets=None, eligible_tasks=None):
        self.path = Path(path)
        self.manifest = read_json(self.path / "manifest.json")
        if self.manifest["schema_version"] not in {1, 2} or self.manifest["probability"] != "raw_t1":
            raise ValueError("Unsupported offline baseline manifest")
        self.seed, self.n, self.m = seed, n, m
        self.coverage_window = coverage_window
        self.targets = {key: count for key, count in (targets or {}).items() if count}
        self.tasks = {}
        ids = set()
        for entry in self.manifest["entries"]:
            if entry["id"] in ids:
                raise ValueError("Duplicate offline baseline trajectory")
            ids.add(entry["id"])
            if eligible_tasks is None or entry["task"] in eligible_tasks:
                self.tasks.setdefault(entry["task"], []).append(entry)

    def _order(self, values, key, cycle, lane):
        return sorted(values, key=lambda value: hash_json([self.seed, lane, cycle, key(value)]))

    def batch(self, cursor):
        if not self.tasks:
            return (), cursor, {"trajectory_counts": {}}
        cycle, offset = cursor.get("cycle", 0), cursor.get("offset", 0)
        positions = dict(cursor.get("positions", {}))
        tasks = self._order(self.tasks, str, cycle, "task")
        selected = tasks[offset:offset + self.n]
        rows, counts = [], Counter()
        coverage = {key: set(cursor.get("coverage", {}).get(key, [])) for key in self.targets}
        coverage_step = cursor.get("coverage_step", 0) + 1
        for task in selected:
            pool = self.tasks[task]
            position = positions.get(task, {})
            round_id, used = position.get("round", 0), set(position.get("used", []))
            order = self._order(pool, lambda entry: entry["id"], round_id, task)
            remaining = [entry for entry in order if entry["id"] not in used]
            chosen = []
            for status, target in self.targets.items():
                candidates = [entry for entry in remaining
                              if entry["failure"] == status and entry["id"] not in coverage[status]]
                chosen.extend(candidates[:min(max(target - len(coverage[status]), 0), self.m - len(chosen))])
            preferred = {entry["id"] for entry in chosen}
            chosen.extend([entry for entry in remaining if entry["id"] not in preferred][:self.m - len(chosen)])
            used.update(entry["id"] for entry in chosen)
            positions[task] = ({"round": round_id + 1, "used": []} if len(used) == len(order)
                               else {"round": round_id, "used": sorted(used)})
            for entry in chosen:
                rows.extend(_decode_rows(_read_shard(self.path, entry),
                                         entry.get("schema_version", self.manifest["schema_version"])))
                status = entry["failure"]
                counts[status or "normal"] += 1
                if status in coverage and len(coverage[status]) < self.targets[status]:
                    coverage[status].add(entry["id"])
        next_offset = offset + len(selected)
        if next_offset >= len(tasks):
            cycle, next_offset = cycle + 1, 0
        cursor = {"cycle": cycle, "offset": next_offset, "positions": positions,
                  "coverage_step": coverage_step, "coverage": {key: sorted(ids) for key, ids in coverage.items()}}
        metrics = {"trajectory_counts": dict(counts)}
        if coverage_step == self.coverage_window:
            metrics["coverage_shortfall"] = {key: target - len(coverage[key]) for key, target in self.targets.items()}
            cursor.update(coverage_step=0, coverage={})
        return tuple(rows), cursor, metrics

def _binding_and_stats(settings, tokenizer_fingerprint):
    dataset = Path(settings.dataset)
    manifest, digest = _manifest_snapshot(dataset)
    if any(entry["tokenizer"] != tokenizer_fingerprint for entry in manifest["entries"]):
        raise ValueError("Offline baseline tokenizer differs from training")
    binding_config = settings.model_dump(exclude={"dataset"})
    if settings.clip_epsilon == 0.1:
        binding_config.pop("clip_epsilon")
    if settings.max_trace_completion_tokens is None:
        binding_config.pop("max_trace_completion_tokens")
    # Preserve the exact historical contract for direct mode.
    if settings.advantage_mode == "direct":
        binding_config.pop("advantage_mode")
        return {"manifest_sha256": digest, "config": binding_config}, None
    stats = _load_task_reward_stats(dataset, manifest, digest)
    return {"manifest_sha256": digest, "config": binding_config,
            "advantage": {"version": TASK_GRPO_VERSION, "stats_sha256": hash_json(stats),
                          "zero_variance": "skip_task"}}, stats


def binding(settings, model_fingerprint, tokenizer_fingerprint):
    if settings is None or not settings.baseline_weight:
        return None
    return _binding_and_stats(settings, tokenizer_fingerprint)[0]


def build_updater(config, model, directory, contract, optimizer, scheduler, *, memory_controller=None, on_stage=None):
    settings = config.offline_baseline
    if settings is None or not settings.baseline_weight:
        return None
    identity = contract["grpo"]["reference_identity"]
    source_binding, task_stats = _binding_and_stats(settings, identity["tokenizer_fingerprint"])
    if source_binding != contract["offline_baseline"]:
        raise ValueError("Offline baseline dataset or advantage statistics changed")
    from RL.policy_reference import InitialPolicyScorer, computation
    from RL.online_runtime import OnlineRolloutBatch, SourceMass, TorchOnlineUpdater, online_row_masses
    from RL.gradient_attribution import GradientAttribution

    eligible_tasks = _active_task_ids(task_stats) if task_stats is not None else None
    skipped_tasks = len(task_stats["tasks"]) - len(eligible_tasks) if task_stats is not None else 0
    pool = BaselinePool(settings.dataset, seed=settings.seed, n=settings.tasks_per_update,
                        m=settings.trajectories_per_task, coverage_window=settings.coverage_window,
                        targets={"LimitsExceeded": settings.limits_exceeded_target,
                                 "reasoning_loop": settings.reasoning_loop_target},
                        eligible_tasks=eligible_tasks)
    scorer = (InitialPolicyScorer(model, Path(directory) / "grpo/reference/adapter",
                                  block_size=config.training.memory.loss_chunk_tokens)
              if settings.kl_beta and pool.tasks else None)
    source_mass = SourceMass(settings.dycon_weight, settings.baseline_weight)
    dycon_mass, baseline_mass = source_mass.normalized()
    offloader = memory_controller.offloader if memory_controller is not None else None
    updater = TorchOnlineUpdater(model, config.training, source_mass=source_mass,
                                 kl_beta=settings.kl_beta, k3_cap=settings.k3_cap, reference_scorer=scorer,
                                 kl_reference_mode=computation(config),
                                 clip_epsilon=settings.clip_epsilon,
                                 baseline_training_mode="prefix_trace",
                                 max_trace_input_tokens=settings.max_trace_input_tokens,
                                 max_trace_completion_tokens=settings.max_trace_completion_tokens,
                                 optimizer_pair=(optimizer, scheduler),
                                 activation_offloader=offloader,
                                 on_stage=on_stage or (memory_controller.phase if memory_controller is not None else None))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    def accumulate(state, parallel, replay_kl=None, *, oom_replay=False):
        updater._stage("baseline_data_read", inflight_update=state["global_step"] + 1)
        rows, cursor, counts = pool.batch(state.get("baseline_cursor", {}))
        updater._stage("baseline_data_ready", agent_step_rows=len(rows))
        if not rows:
            if not dycon_mass:
                raise ValueError("Baseline-only update has no eligible trajectories")
            return {"skipped": True, "advantage": {"mode": settings.advantage_mode,
                                                   "dataset_tasks_skipped_zero_advantage": skipped_tasks}}, cursor, None
        if task_stats is not None:
            rows, counts["advantage"] = _task_grpo_rows(rows, task_stats)
            counts["advantage"]["dataset_tasks_skipped_zero_advantage"] = skipped_tasks
        else:
            counts["advantage"] = {"mode": "direct"}
        attribution = GradientAttribution(parameters) if settings.attribution_every and (state["global_step"] + 1) % settings.attribution_every == 0 else None
        for index, parameter in enumerate(parameters):
            if parameter.grad is not None:
                if dycon_mass:
                    parameter.grad.mul_(dycon_mass)
                    if attribution is not None:
                        attribution._add("dycon", index, parameter.grad)
                else:
                    parameter.grad = None
        if attribution is not None and dycon_mass and config.grpo is not None and config.grpo.kl_beta:
            gradients = [parameter.grad for parameter in parameters]
            try:
                for parameter in parameters:
                    parameter.grad = None
                replay_kl()
                for index, parameter in enumerate(parameters):
                    if parameter.grad is not None:
                        attribution._add("kl", index, parameter.grad * dycon_mass)
            finally:
                for parameter, gradient in zip(parameters, gradients):
                    parameter.grad = gradient
        batch = OnlineRolloutBatch(state["global_step"], tuple(sorted({r.task_id for r in rows})),
                                   "historical", "unused", rows)
        masses = online_row_masses(rows, source_mass)
        from contextlib import nullcontext
        attention = nullcontext()
        if oom_replay and getattr(config.model, "attention_backend", None) == "sdpa":
            from RL.qwen_boundary import retry_chunked_attention
            attention = retry_chunked_attention()
        with attention:
            _, metrics, _ = updater.accumulate(batch, masses, source_mass, state["global_step"],
                                               attribution=attribution, parallel=parallel)
        metrics.update(counts, weight=baseline_mass, dycon_weight=dycon_mass)
        return metrics, cursor, attribution

    accumulate.replay_kl = bool(settings.attribution_every and dycon_mass and config.grpo is not None and config.grpo.kl_beta)
    accumulate.attribution_every = settings.attribution_every
    return accumulate


def baseline_only(config):
    settings = config.offline_baseline
    return settings is not None and settings.dycon_weight == 0 and settings.baseline_weight > 0


def schedule(config):
    settings = config.offline_baseline
    dataset = Path(settings.dataset)
    extra = {}
    if settings.advantage_mode == "task_grpo":
        manifest, digest = _manifest_snapshot(dataset)
        stats = _load_task_reward_stats(dataset, manifest, digest)
        tasks = len(_active_task_ids(stats))
        extra["tasks_skipped_zero_advantage"] = len(stats["tasks"]) - tasks
        if not tasks:
            raise ValueError("Baseline task_grpo has no tasks with differing rewards")
    else:
        manifest = read_json(dataset / "manifest.json")
        tasks = len({entry["task"] for entry in manifest["entries"]})
        if not tasks:
            raise ValueError("Baseline dataset has no tasks")
    steps = (tasks + settings.tasks_per_update - 1) // settings.tasks_per_update
    total = steps * settings.epochs
    if config.training.max_updates is not None and config.training.max_updates != total:
        raise ValueError(f"Baseline max_updates must equal epochs * task batches = {total}")
    return {"tasks": tasks, "updates_per_epoch": steps, "epochs": settings.epochs,
            "max_updates": total, **extra}

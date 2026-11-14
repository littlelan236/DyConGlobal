"""Metadata-frozen schedules with bounded CPU preparation of training windows.

The initial scan reads metadata only. Native tokens and raw reference values are
read in frozen step order by a CPU producer while GPUs train the previous step.
The sealed source data remains available for unconsumed windows. Recovery uses
the frozen plan and committed cursor, rebuilding only unconsumed windows.
"""
from collections import Counter
import copy
import math
from pathlib import Path
import queue
import threading
import time

from core.io import hash_json, read_json, sha256, write_json
from RL.dataset.schema import MASK_POLICY
from .advantage import CONSENSUS_ADVANTAGE_VERSION, compute_advantages, validate_retained
from .data import source_groups, freeze_group, tensor_groups
from .schedule import order_groups, shuffle_metadata


def _advantages(value, config):
    settings = config.grpo.model_dump(include={"top_k", "top_k_fraction", "score_scale_floor",
                                                "boundary_tie_sigma"})
    if config.grpo.advantage_policy == CONSENSUS_ADVANTAGE_VERSION:
        retained = [row.get("retained") for row in value["members"]]
        validate_retained(retained, len(value["members"]))
        settings.update(advantage_policy=config.grpo.advantage_policy, retained=retained)
    return compute_advantages([row.get("historical_score") for row in value["members"]], **settings)


def _eligible(value, config):
    group, members = value["group"], value["members"]
    if not group.get("data_ready"):
        return "incomplete_native_group", None
    if group["expected_branch_count"] != len(members) or group["branch_ids"] != [r["branch_sample_id"] for r in members]:
        raise ValueError("Incomplete/misordered streaming group inventory")
    if len(members) < 2:
        return "GRPO groups require at least two branches", None
    if any(row["prompt_count"] + row["completion_count"] > config.validation.max_sequence_length for row in members):
        raise ValueError("Group exceeds explicit sequence limit; no truncation is allowed")
    if any(row["prompt_count"] < 1 or row["completion_count"] < 1 for row in members):
        raise ValueError("GRPO requires nonempty prompt and completion tokens")
    if any((r["prompt_sha256"], r["prompt_count"]) != (members[0]["prompt_sha256"], members[0]["prompt_count"]) for r in members):
        raise ValueError("Streaming group prompt identities differ")
    if any(not row.get("action") for row in members):
        return "missing_action", None
    result = _advantages(value, config)
    return (None, result) if result.metadata["status"] == "ok" else (result.metadata["skip_reason"], None)


def _compact(value):
    """Keep only fields needed to reload native tokens and raw reference values."""
    group_keys = ("group_id", "branch_ids", "expected_branch_count", "data_ready")
    row_keys = ("group_id", "branch_sample_id", "native_request_id", "prompt_sha256",
                "completion_sha256", "prompt_count", "completion_count", "tokenizer_fingerprint",
                "mask_spans", "native_evidence", "action", "historical_score", "retained",
                "session_id", "decision_id", "branch_id", "agent_step", "message_index")
    return {"group": {key: value["group"][key] for key in group_keys},
            "members": [{key: row[key] for key in row_keys if key in row} for row in value["members"]],
            "frozen_advantages": list(value["frozen_advantages"]),
            "advantage_diagnostics": copy.deepcopy(value["advantage_diagnostics"])}


def _plan_entry(value):
    return {"group_id": value["group"]["group_id"],
            "branch_ids": list(value["group"]["branch_ids"]),
            "frozen_advantages": list(value["frozen_advantages"]),
            "advantage_diagnostics": copy.deepcopy(value["advantage_diagnostics"])}


def create_plan(config):
    """Freeze eligibility, advantages and length-bucket order without token IO."""
    if config.algorithm.dataset_id == "synthetic-grpo-smoke-v1":
        raise ValueError("Step streaming requires native dataset metadata")
    raw, proof = source_groups(config)
    eligible, exclusions, advantages = [], [], []
    seen_groups, seen_branches = set(), set()
    total = 0
    for value in raw:
        total += 1
        gid = value["group"]["group_id"]
        bids = [r["branch_sample_id"] for r in value["members"]]
        if gid in seen_groups or seen_branches.intersection(bids) or len(set(bids)) != len(bids):
            raise ValueError("Duplicate streaming group/branch identity")
        seen_groups.add(gid); seen_branches.update(bids)
        reason, result = _eligible(value, config)
        if reason:
            exclusions.append({"group_id": gid, "reason": reason})
            continue
        value["frozen_advantages"] = result.advantages
        value["advantage_diagnostics"] = {**result.metadata, "order": result.order, "ranks": result.ranks}
        advantages.extend(result.advantages)
        eligible.append(_compact(value))
    if not eligible:
        raise ValueError("No eligible GRPO groups after metadata scan")
    ordered = order_groups(eligible, seed=config.grpo.shuffle_seed, batch_size=config.grpo.group_batch_size,
                           algorithm=config.grpo.shuffle_algorithm,
                           prompt_length=lambda value: value["members"][0]["prompt_count"])
    plan_groups = [_plan_entry(value) for value in ordered]
    ids = [value["group_id"] for value in plan_groups]
    settings = config.grpo.model_dump(include={"top_k", "top_k_fraction", "score_scale_floor", "boundary_tie_sigma"})
    if config.grpo.advantage_policy == CONSENSUS_ADVANTAGE_VERSION:
        settings["advantage_policy"] = config.grpo.advantage_policy
    proof.update(objective_version="offline_grpo_k3_v1", preflight=config.preflight.model_dump(),
                 mask_policy=MASK_POLICY, advantage_policy=config.grpo.advantage_policy,
                 advantage_config_sha256=hash_json(settings), group_order_seed=config.grpo.shuffle_seed,
                 group_batch_size=config.grpo.group_batch_size, max_epochs=1, probability_mode="raw_t1",
                 clip_epsilon=config.grpo.clip_epsilon, kl_beta=config.grpo.kl_beta,
                 eligible_group_ids=ids, eligible_groups_sha256=hash_json(sorted(ids)),
                 group_order_sha256=hash_json(plan_groups), group_order_schema="step_stream_v1",
                 group_count=len(ids), branch_count=len(advantages),
                 planned_updates=math.ceil(len(ids) / config.grpo.group_batch_size),
                 exclusions=exclusions, exclusion_reasons=dict(Counter(x["reason"] for x in exclusions)),
                 total_groups=total, group_coverage=len(ids) / total,
                 advantage_distribution={"min":min(advantages), "max":max(advantages),
                     "mean":sum(advantages)/len(advantages), "sum":math.fsum(advantages),
                     "positive":sum(a > 0 for a in advantages), "negative":sum(a < 0 for a in advantages)},
                 **shuffle_metadata(config.grpo.shuffle_algorithm))
    from .execution import enabled, parent_record, from_config, apply_proof
    if enabled(config):
        apply_proof(proof, from_config([parent_record(g) for g in ordered], config))
    if config.training.max_updates != proof["planned_updates"]:
        raise ValueError(f"GRPO max_updates must equal one epoch's planned_updates={proof['planned_updates']}")
    return {"schema_version":1, "proof":proof, "groups":plan_groups}


def validate_plan(plan):
    proof = plan["proof"]
    ids = [g["group_id"] for g in plan["groups"]]
    if (plan.get("schema_version") != 1 or ids != proof["eligible_group_ids"]
            or len(ids) != proof["group_count"] or len(set(ids)) != len(ids)
            or hash_json(plan["groups"]) != proof["group_order_sha256"]):
        raise ValueError("Streaming plan identity/order checksum mismatch")
    if "execution_plan" in proof:
        from .execution import validate
        execution = validate(proof["execution_plan"])
        if (proof.get("execution_schedule_sha256") != execution["sha256"]
                or {p["parent_group_id"] for p in execution["parents"]} != set(ids)
                or proof["planned_updates"] != len(execution["windows"])):
            raise ValueError("Streaming execution plan/input mismatch")
    return plan


class StreamingReference:
    def __init__(self):
        self.values = {}

    def __getitem__(self, group_id):
        return self.values[group_id]


class StreamingGroups:
    """One CPU producer, two queued windows, and one active GPU window per rank."""
    def __init__(self, config, plan, *, rank=0, world_size=1, progress_path=None):
        self.config = config
        self.plan = validate_plan(plan)
        self.proof = plan["proof"]
        self.rank, self.world_size = rank, world_size
        self.progress_path = Path(progress_path) if progress_path else None
        self.reference = StreamingReference()
        self.source = None
        self._queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._error = None
        self._thread = None
        self.prepared_windows = 0
        self.consumed_windows = 0
        self._branch_prefix = [0]
        self._token_prefix = [0]
        from RL.dataset.artifacts import directory, verify_local
        from .stream_resume import checked_path
        dataset = directory(config.algorithm.dataset_root, "rl-dataset", config.algorithm.dataset_id)
        receipt = verify_local(dataset, "rl-dataset", verify_files=False)
        # Metadata scans are allowed; payloads of consumed groups are never read.
        for name in ("dataset.json", "groups.jsonl", "branches.jsonl"):
            if sha256(checked_path(dataset, name)) != receipt["files"][name]["sha256"]:
                raise ValueError("Streaming frozen dataset metadata checksum mismatch: " + name)
        self._dataset = dataset
        self._dataset_receipt = sha256(dataset / "receipt.json")
        self._metadata_hashes = {name: receipt["files"][name]["sha256"]
                                 for name in ("dataset.json", "groups.jsonl", "branches.jsonl")}
        raw, source_proof = source_groups(config)
        raw_by_id = {value["group"]["group_id"]: value for value in raw}
        if source_proof.get("dataset_receipt_sha256") != self.proof.get("dataset_receipt_sha256"):
            raise ValueError("Streaming dataset receipt differs from the frozen plan")
        self.execution_plan = self.proof.get("execution_plan")
        self._ordered = []
        for entry in plan["groups"]:
            value = raw_by_id.get(entry["group_id"])
            if (value is None or value["group"].get("branch_ids") != entry["branch_ids"]
                    or [r["branch_sample_id"] for r in value["members"]] != entry["branch_ids"]):
                raise ValueError("Streaming source group differs from the frozen plan")
            value["frozen_advantages"] = entry["frozen_advantages"]
            value["advantage_diagnostics"] = entry["advantage_diagnostics"]
            self._ordered.append(value)
            self._branch_prefix.append(self._branch_prefix[-1] + len(value["members"]))
            self._token_prefix.append(self._token_prefix[-1] + sum(r["completion_count"] for r in value["members"]))

        if self.execution_plan:
            from .execution import parent_record
            by_id = {parent_record(v)["parent_group_id"]: parent_record(v) for v in self._ordered}
            if list(sorted(by_id.values(), key=lambda p: p["parent_group_id"])) != self.execution_plan["parents"]:
                raise ValueError("Streaming source metadata differs from 8x16 frozen parents")

    def __len__(self):
        return len(self.execution_plan["chunks"]) if self.execution_plan else len(self.plan["groups"])

    def _progress(self, stage):
        if self.progress_path:
            write_json(self.progress_path, {"stage":stage, "rank":self.rank, "updated_at":time.time(),
                "prepared_windows":self.prepared_windows, "consumed_windows":self.consumed_windows,
                "queue_depth":self._queue.qsize(), "error":str(self._error) if self._error else None})

    def cursor(self, count):
        if self.execution_plan:
            from .execution import cursor
            return cursor(self.execution_plan, count)
        if type(count) is not int or not 0 <= count <= len(self):
            raise ValueError("Streaming group cursor out of bounds")
        ids = self.proof["eligible_group_ids"][:count]
        return {"segment_id":"streaming", "segment_ordinal":count,
                "last_consumed_group_id":ids[-1] if ids else None,
                "consumed_group_ids_sha256":hash_json(ids)}

    def validate_state(self, state):
        if self.execution_plan:
            from .execution import validate_state
            return validate_state(self.execution_plan, state)
        count = state["group_cursor"]
        batch = self.config.grpo.group_batch_size
        if (type(count) is not int or type(state["global_step"]) is not int
                or not 0 <= count <= len(self) or (count != len(self) and count % batch)
                or state["global_step"] != math.ceil(count/batch)
                or state["consumed_groups"] != count
                or state["consumed_branches"] != self._branch_prefix[count]
                or state["consumed_tokens"] != self._token_prefix[count]):
            raise ValueError("Streaming checkpoint cursor/counters mismatch")
        if any(state.get(k) != v for k,v in self.cursor(count).items()):
            raise ValueError("Streaming checkpoint consumed group identity mismatch")

    def _put(self, value):
        while not self._stop.is_set():
            try:
                self._queue.put(value, timeout=.5)
                return True
            except queue.Full:
                pass
        return False

    def _produce(self, start):
        import torch
        try:
            if self.execution_plan:
                self._produce_execution(start)
                return
            batch = self.config.grpo.group_batch_size
            for offset in range(start, len(self), batch):
                if self._stop.is_set():
                    return
                self._progress("preparing_window")
                window, values = [], {}
                for value in self._ordered[offset:offset+batch]:
                    raw = copy.deepcopy(value)
                    from RL.dataset.loader import native_tokens
                    for row in raw["members"]:
                        p, c = native_tokens(row, audit=self.config.preflight.audit_native_tokens,
                                             verify_hashes=True)
                        row.update(prompt_token_ids=p, completion_token_ids=c, audit_enabled=True)
                    group = next(tensor_groups([raw]))
                    window.append(group)
                    # Only import reference arrays used by this rank's groups.
                    if (len(window)-1) % self.world_size == self.rank:
                        members = {}
                        for row in raw["members"]:
                            members[row["branch_sample_id"]] = torch.from_numpy(self.source.values(
                                row, verify=True).copy()).detach()
                        values[group.group_id] = members
                if not self._put((tuple(window), values)):
                    return
                self.prepared_windows += 1
                self._progress("window_ready")
            self.source.verify_unchanged()
            self._put(None)
        except BaseException as exc:
            self._error = exc
            self._progress("failed")
            self._put(None)

    def _produce_execution(self, start):
        import torch
        from .execution import select_raw, decorate, placeholder, cursor
        from RL.dataset.loader import native_tokens
        cursor(self.execution_plan, start)
        raw_by_id = {v["group"]["group_id"]: v for v in self._ordered}
        parents = {p["parent_group_id"]: p for p in self.execution_plan["parents"]}
        for w in self.execution_plan["windows"]:
            if w["start"] < start:
                continue
            if self._stop.is_set():
                return
            started = time.monotonic()
            window, values = [], {}
            for index, chunk in enumerate(self.execution_plan["chunks"][w["start"]:w["end"]]):
                pid = chunk["parent_group_id"]
                if index % self.world_size != self.rank:
                    group = placeholder(parents[pid], chunk, w, index % self.world_size)
                else:
                    raw = select_raw(raw_by_id[pid], chunk)
                    for row in raw["members"]:
                        p, c = native_tokens(row, audit=self.config.preflight.audit_native_tokens, verify_hashes=True)
                        row.update(prompt_token_ids=p, completion_token_ids=c, audit_enabled=True)
                    group = decorate(next(tensor_groups([raw])), chunk, w, self.rank)
                    values[pid] = {r["branch_sample_id"]: torch.from_numpy(
                        self.source.values(r, verify=True).copy()).detach() for r in raw["members"]}
                window.append(group)
            if not self._put((tuple(window), values, {"io_reference_seconds": time.monotonic() - started})):
                return
            self.prepared_windows += 1
            self._progress("window_ready")
        self.source.verify_unchanged()
        self._put(None)

    def windows(self, start):
        if self.source is None:
            raise RuntimeError("Streaming raw probability source has not been bound")
        if self._thread is not None:
            raise RuntimeError("Streaming windows are single-use")
        self._thread = threading.Thread(target=self._produce, args=(start,), name="grpo-step-producer", daemon=True)
        self._thread.start()
        try:
            while True:
                self._progress("waiting_window")
                while True:
                    try:
                        item = self._queue.get(timeout=1)
                        break
                    except queue.Empty:
                        if self._error:
                            raise self._error
                        if not self._thread.is_alive():
                            raise RuntimeError("Streaming producer exited without a completion marker")
                if self._error:
                    raise self._error
                if item is None:
                    return
                window, values = item[:2]
                self.window_io = item[2] if len(item) == 3 else {}
                self._active_window = window
                self._active_committed = False
                self.reference.values = values
                self._progress("training_window")
                yield window
                if not self._active_committed:
                    raise RuntimeError("Streaming window advanced before a committed optimizer update")
        finally:
            self.close()

    def commit_window(self, window):
        """Acknowledge only the currently yielded window, exactly once."""
        if window is not getattr(self, "_active_window", None) or self._active_committed:
            raise RuntimeError("Invalid or duplicate streaming window commit")
        self._active_committed = True
        self.consumed_windows += 1
        self.reference.values = {}
        self._progress("window_committed")

    def close(self):
        self._stop.set()
        self.reference.values = {}
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("Streaming producer did not stop within 5 seconds")
        # Covers full completion, pause and exception cleanup. This is an identity
        # check, never a claim that every external payload has been consumed.
        if self.source is not None:
            self.source.verify_unchanged()
        if sha256(self._dataset / "receipt.json") != self._dataset_receipt:
            raise ValueError("Streaming dataset receipt changed during training")
        for name, expected in self._metadata_hashes.items():
            if sha256(self._dataset / name) != expected:
                raise ValueError("Streaming dataset metadata changed during training: " + name)
        if self._error is not None:
            raise self._error


def bind_source(config, groups, directory, base, adapter, identity, *, saved_binding=None, primary=True):
    """Open metadata once per rank; keep one self-contained source proof per run."""
    from RL.dataset.probability import open_probability_stream
    from .contract import validate_scoring_identity, validate_probability_binding
    source = open_probability_stream(config.grpo.reference_cache, config.algorithm, groups.proof,
                                     strict=True, scan_dataset=False, metadata_only=True)
    validate_scoring_identity(source.scoring_identity, base=base, proof=groups.proof,
        training_dtype=config.model.dtype, policy=config.grpo.reference_cache.dtype_compatibility, adapter=adapter)
    source_path = Path(directory)/"grpo/source-provenance.json"
    if primary and saved_binding is None:
        write_json(source_path, source.provenance)
    binding = saved_binding or {"schema_version":1, "selection":config.grpo.reference_cache.model_dump(),
        "source_receipt_sha256":source.receipt_sha256, "source_provenance_sha256":sha256(source_path),
        "scoring_identity":source.scoring_identity, "collector_dtype":source.scoring_identity["collector_dtype"],
        "training_compute_dtype":config.model.dtype, "dtype_compatibility":config.grpo.reference_cache.dtype_compatibility,
        "training_initial_adapter":adapter, "preflight":config.preflight.model_dump(),
        "training_snapshot_purpose":"compatibility_evidence_only", "probability_semantics":"raw_t1_completion_targets_v1"}
    if source.provenance.get("adapter_schema") == "canonical_probability_v3" and saved_binding is None:
        binding.update(execution_cohorts=source.provenance["execution_cohorts"],
                       dataset_receipt_sha256=groups.proof["dataset_receipt_sha256"])
    if source.receipt_sha256 != binding["source_receipt_sha256"] or source.scoring_identity != binding["scoring_identity"]:
        raise ValueError("Streaming source receipt/scoring changed since checkpoint")
    if (source.provenance["manifest_sha256"] != read_json(source_path)["manifest_sha256"]
            or source.provenance["index_sha256"] != read_json(source_path)["index_sha256"]):
        raise ValueError("Streaming source manifest/index changed since checkpoint")
    # Reuse the established source proof validator with a single logical anchor.
    surrogate = {"cache_schema_version":2, "reference_identity":identity, "probability_binding":binding,
                 "reference_logprob_cache_sha256":binding["source_receipt_sha256"],
                 "role_binding":{"ratio_denominator":binding["source_receipt_sha256"],
                                 "k3_reference":binding["source_receipt_sha256"]}}
    validate_probability_binding(directory, {"schema_version":2,"probability_binding":binding}, surrogate)
    groups.source = source
    return binding


def attach_source(config, groups, binding, *, directory=None):
    """Open the same immutable source on a non-primary data-parallel rank."""
    from RL.dataset.probability import open_probability_stream
    if config.grpo.reference_cache.source == "model_forward":
        raise ValueError("Step streaming requires an external probability source")
    source = open_probability_stream(config.grpo.reference_cache, config.algorithm,
                                     groups.proof, strict=True, scan_dataset=False, metadata_only=True)
    if source.receipt_sha256 != binding["source_receipt_sha256"]:
        raise ValueError("Streaming source receipt differs across ranks")
    if source.scoring_identity != binding["scoring_identity"]:
        raise ValueError("Streaming source scoring identity differs across ranks")
    if directory is not None:
        saved = read_json(Path(directory) / "grpo/source-provenance.json")
        if (source.provenance["manifest_sha256"] != saved["manifest_sha256"]
                or source.provenance["index_sha256"] != saved["index_sha256"]):
            raise ValueError("Streaming source manifest/index differs across ranks")
    groups.source = source
    return source


def validate_reference(directory, contract):
    """Validate a checkpoint's frozen metadata plan and shared source anchor."""
    from .contract import validate_probability_binding
    directory = Path(directory)
    from .stream_resume import checkpoint_kind, checked_path
    checkpoint_kind(contract)
    for name in ("grpo/stream-plan.json", "grpo/source-provenance.json", "grpo/reference/adapter"):
        checked_path(directory, name)
    ref = contract["grpo"]
    binding = ref["probability_binding"]
    if (contract["schedule"]["grpo"].get("reference_cache") != binding["selection"]
            or binding.get("preflight") != contract["schedule"].get("preflight")):
        raise ValueError("Streaming source selection/preflight differs from frozen schedule")
    if ref.get("cache_schema_version") != 4 or ref.get("reference_mode") != "step_stream_v1":
        raise ValueError("Unsupported streaming reference schema")
    plan_path = directory/"grpo/stream-plan.json"
    if sha256(plan_path) != contract["streaming"]["plan_sha256"]:
        raise ValueError("Streaming plan checksum mismatch")
    plan = validate_plan(read_json(plan_path))
    if plan["proof"] != contract["input"]:
        raise ValueError("Streaming plan/input proof mismatch")
    binding = ref["probability_binding"]
    digest = binding["source_receipt_sha256"]
    if ref["role_binding"] != {"ratio_denominator":digest, "k3_reference":digest}:
        raise ValueError("Streaming reference shared anchor mismatch")
    validate_probability_binding(directory, {"schema_version":2,"probability_binding":binding},
        {**ref,"cache_schema_version":2,"reference_logprob_cache_sha256":digest})
    from RL.checkpoints import verify_adapter
    verify_adapter(directory/"grpo/reference/adapter", contract["adapter"], verify_owner=False)
    return plan

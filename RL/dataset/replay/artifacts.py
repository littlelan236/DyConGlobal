"""Streaming float32 sidecars, checked branch bindings and explicit estimate gates."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np

from core.io import hash_json, owned_path, read_json, sha256, write_json
from ..artifacts import publication, seal, verify_local
from ..build import iter_rows, json_line
from ..loader import iter_groups
from ..materialize import ShardWriter, read_array
from .validation import RAW_ORIGIN, RAW_SEMANTICS, raw_contract, raw_policy, validate_values, verify_identity

BINDINGS = ("branch_sample_id", "group_id", "native_request_id", "prompt_sha256",
            "completion_sha256", "tokenizer_fingerprint", "prompt_count", "completion_count")


class ReplayWriter:
    """One sealed shard per bounded result; journal commits only complete branches."""
    def __init__(self, stage, dataset, identity, max_shard_bytes, *, recover=False, commit_every_branches=32, commit_interval_seconds=5):
        self.stage = Path(stage); self.dataset = Path(dataset); self.identity = identity
        self.records = []; self.finished = set(); self.pending_ids = set()
        self.progress = self.stage/"progress-branches.jsonl"
        self.progress_hash = hashlib.sha256()
        if recover:
            journal = read_json(self.stage / "journal.json")
            size = journal.get("committed_bytes", 0)
            if size == 0: self.progress.touch(exist_ok=True)
            with self.progress.open("r+b") as stream:
                remaining=size
                while remaining:
                    chunk=stream.read(min(remaining,1024*1024))
                    if not chunk: raise ValueError("Recovery replay index truncated")
                    self.progress_hash.update(chunk);remaining-=len(chunk)
                if self.progress_hash.hexdigest()!=journal.get("index_sha256",self.progress_hash.hexdigest()): raise ValueError("Recovery replay index checksum differs")
                stream.truncate(size)
            for record in iter_rows(self.progress):
                values = read_array(self.stage, record["values_ref"])
                validate_values(values, record["completion_count"])
                if record["branch_sample_id"] in self.finished: raise ValueError("Duplicate recovery branch")
                self.records.append(record); self.finished.add(record["branch_sample_id"])
        else:
            self.progress.touch()
        self.index = {r['branch_sample_id']:r for r in self.records}
        self.max_bytes = max_shard_bytes
        self.commit_every = commit_every_branches; self.commit_seconds = commit_interval_seconds
        self.last_commit = time.monotonic(); self.committed_count = len(self.records)
        if not recover: self.checkpoint("prepared")

    def put(self, row, output, values, *, policy_sha256, model_sha256):
        verify_identity(row, output, values)
        bid = row["branch_sample_id"]
        if bid in self.finished: raise ValueError("Duplicate replay branch")
        values = np.asarray(validate_values(values, row["completion_count"], allow_zero_support=False), dtype="<f4")
        validate_values(values, row["completion_count"], allow_zero_support=False)
        # Per-branch seal avoids sharing an uncommitted shard during recovery.
        # Long branches may exceed the configured shard size and are rejected.
        sub = self.stage / "results" / bid
        sub.mkdir(parents=True, exist_ok=True)
        writer = ShardWriter(sub, "<f4", self.max_bytes)
        try: ref = writer.put(values)
        finally: writer.close()
        ref["shard"] = (Path("results") / bid / ref["shard"]).as_posix()
        record = {k: row[k] for k in BINDINGS}
        record.update(values_ref=ref, count=len(values), policy_sha256=policy_sha256,
                      model_sha256=model_sha256, status="replay_estimate",
                      token_identity_verified=bool(row.get("audit_enabled")),
                      zero_support_count=int(np.isneginf(values).sum()),
                      probability_semantics=RAW_SEMANTICS, origin=RAW_ORIGIN,
                      raw_contract_sha256=self.identity["raw_contract_sha256"])
        self.records.append(record); self.finished.add(bid)
        self.index[bid] = record
        raw=(json.dumps(record,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")
        with self.progress.open("ab") as stream:
            stream.write(raw)
        self.progress_hash.update(raw)
        if len(self.records) - self.committed_count >= self.commit_every or time.monotonic() - self.last_commit >= self.commit_seconds:
            self.checkpoint("replaying")

    def verify_completed(self, row, *, policy_sha256, model_sha256):
        record = self.index.get(row['branch_sample_id'])
        if record is None: raise ValueError('Recovery branch missing')
        if any(record[k] != row[k] for k in BINDINGS) or record['policy_sha256']!=policy_sha256 or record['model_sha256']!=model_sha256:
            raise ValueError('Recovery branch model/policy/token binding mismatch')

    def import_record(self, source, record, expected, *, policy_sha256, model_sha256, origin):
        """Reuse verified committed bytes, with no token replay or relabelling.

        Called only by the single coordinator writer. Source journals are read
        to their committed boundary; arrays are hash checked before hardlinking.
        """
        import copy
        import shutil
        if any(record[k] != expected[k] for k in BINDINGS):
            raise ValueError('Imported branch binding mismatch')
        if (record['policy_sha256'] != policy_sha256 or record['model_sha256'] != model_sha256
                or record.get('probability_semantics') != RAW_SEMANTICS
                or record.get('origin') != RAW_ORIGIN
                or record.get('raw_contract_sha256') != self.identity['raw_contract_sha256']
                or record['values_ref']['dtype'] != '<f4'):
            raise ValueError('Imported raw scoring identity mismatch')
        values = read_array(source, record['values_ref'])
        validate_values(values, expected['completion_count'], allow_zero_support=False)
        if record['count'] != len(values) or record['zero_support_count'] != 0:
            raise ValueError('Imported branch count/support mismatch')
        bid = record['branch_sample_id']
        if bid in self.finished:
            previous = self.index[bid]
            self.verify_completed(expected, policy_sha256=policy_sha256, model_sha256=model_sha256)
            if not np.array_equal(read_array(self.stage, previous['values_ref']), values):
                raise ValueError('Conflicting imported probability bytes')
            return False
        row = copy.deepcopy(record)
        source_file = owned_path(source, row['values_ref']['shard'])
        target = self.stage / 'imports' / bid / 'values.f4'
        target.parent.mkdir(parents=True, exist_ok=True)
        # Source ShardWriter creates one sealed array per file. Copy a slice
        # only if importing a packed external shard in a future producer.
        if source_file.stat().st_size == len(values) * 4 and row['values_ref'].get('offset', 0) == 0:
            if not target.exists():
                try: os.link(source_file, target)
                except OSError: shutil.copyfile(source_file, target)
        else:
            values.astype('<f4').tofile(target)
        row['values_ref'].update(shard=target.relative_to(self.stage).as_posix(), offset=0)
        read_array(self.stage, row['values_ref'])
        row['execution_source'] = origin
        self.records.append(row); self.finished.add(bid); self.index[bid] = row
        raw = (json.dumps(row, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
        with self.progress.open('ab') as stream: stream.write(raw)
        self.progress_hash.update(raw)
        if len(self.records) - self.committed_count >= self.commit_every or time.monotonic() - self.last_commit >= self.commit_seconds:
            self.checkpoint('replaying')
        return True

    def checkpoint(self,state):
        with self.progress.open("ab") as stream:
            stream.flush(); os.fsync(stream.fileno())
        write_json(self.stage / "journal.json", {"signature": self.identity, "state": state,
                   "committed_bytes": self.progress.stat().st_size, "index_sha256": self.progress_hash.hexdigest(),
                   "branch_count":len(self.records)})
        self.last_commit = time.monotonic(); self.committed_count = len(self.records)

    def finish(self, sampling, policies, models, runtime):
        expected = []; group_status = {}; counts = {"groups": 0, "branches": 0, "tokens": 0, "zero_support_count": 0}
        indexed = {r["branch_sample_id"]: r for r in self.records}
        branch_keys = {tuple(k) for k in self.identity["branch_keys"]} if "branch_keys" in self.identity else None
        with (self.stage / "branches.jsonl").open("w", encoding="utf-8") as stream, (self.stage / "diagnostics.jsonl").open("w", encoding="utf-8") as diagnostics:
            selected_group_ids = []
            for group in iter_groups(self.dataset, with_tokens=False):
                if self.identity.get("group_ids") and group["group"]["group_id"] not in self.identity["group_ids"]: continue
                full_member_count = len(group["members"])
                if branch_keys is not None:
                    group = {**group, "members": [r for r in group["members"] if (r["group_id"], r["branch_sample_id"]) in branch_keys]}
                    if not group["members"]: continue
                selected_group_ids.append(group["group"]["group_id"])
                ready = len(group["members"]) == full_member_count; counts["groups"] += 1
                for row in group["members"]:
                    bid = row["branch_sample_id"]; expected.append(bid)
                    record = indexed.get(bid)
                    if record is None: raise ValueError("Replay branch missing at publication")
                    if any(record[k] != row[k] for k in BINDINGS): raise ValueError("Replay branch binding mismatch")
                    if policies and record["policy_sha256"] not in policies: raise ValueError("Replay policy binding mismatch")
                    if models and record["model_sha256"] not in models: raise ValueError("Replay model binding mismatch")
                    if record.get("probability_semantics") != RAW_SEMANTICS or record.get("origin") != RAW_ORIGIN: raise ValueError("Raw replay output contract mismatch")
                    read_array(self.stage, record["values_ref"])
                    json_line(stream, record)
                    counts["branches"] += 1; counts["tokens"] += record["count"]
                    counts["zero_support_count"] += record["zero_support_count"]
                    if record["zero_support_count"]:
                        ready = False
                        values=read_array(self.stage,record["values_ref"])
                        json_line(diagnostics, {"branch_sample_id": bid, "reason": "zero_support_under_replay", "count": record["zero_support_count"],
                                  "completion_positions":np.flatnonzero(np.isneginf(values)).tolist(),"policy_sha256":record["policy_sha256"],"model_sha256":record["model_sha256"]})
                group_status[group["group"]["group_id"]] = ready
        if set(expected) != self.finished: raise ValueError("Replay contains extra/unbound branches")
        if not expected: raise ValueError("No eligible complete native groups to replay")
        write_json(self.stage / "replay.json", {"schema_version": 2, "replay_id": self.identity["replay_id"],
                   "dataset_receipt_sha256": self.identity["dataset_receipt_sha256"],
                   "probability_semantics": RAW_SEMANTICS, "origin": RAW_ORIGIN,
                   "raw_contract_sha256": self.identity["raw_contract_sha256"],
                   "cache_receipt_sha256": self.identity.get("cache_receipt_sha256"),
                   "selection_sha256": self.identity["selection_sha256"],
                   "coverage_plan_id": self.identity.get("coverage_plan_id"),
                   "coverage_plan_sha256": self.identity.get("coverage_plan_sha256"),
                   "status": "replay_estimate", "historical_behavior_verified": False})
        write_json(self.stage / "root-sampling.json", sampling)
        write_json(self.stage / "effective-policies.json", policies)
        write_json(self.stage / "models.json", {"models": models, "runtime": runtime})
        write_json(self.stage / "selection.json", {"branch_sample_ids": expected, "group_probability_ready": group_status,
                   "group_ids": selected_group_ids,
                   "coverage": "explicit_missing_branches" if branch_keys is not None else "explicit_complete_groups" if self.identity.get("group_ids") else "all_complete_groups",
                   "coverage_plan_sha256": self.identity.get("coverage_plan_sha256"),
                   "order": "dataset_group_branch_v1"})
        write_json(self.stage / "summary.json", {**counts, "probability_ready_groups": sum(group_status.values()), "dataset_probability_ready": branch_keys is None and all(group_status.values()), "status": "replay_estimate"})
        write_json(self.stage / "validation.json", {"status": "token_and_value_checks_passed", "numerical_validation_id": None,
                   "historical_runtime_status": "not_archived", "historical_behavior_verified":False, "identity": self.identity})
        self.checkpoint("complete")
        return counts


def write_replay(dataset_path, replay_id, root_config, outputs, storage_root, *, runtime_identity=None, max_shard_bytes=268435456):
    """In-process adapter used by CPU contract tests; production uses the worker."""
    sampling = raw_contract(); policy = raw_policy()
    identity = {"schema_version": 2, "replay_id": replay_id, "dataset_receipt_sha256": sha256(Path(dataset_path)/"receipt.json"),
                "raw_contract_sha256": hash_json(sampling), "selection_sha256": "test-selection",
                "probability_semantics": RAW_SEMANTICS}
    with publication(storage_root, "rl-replay", replay_id, signature=identity) as (stage, destination):
        writer = ReplayWriter(stage, dataset_path, identity, max_shard_bytes)
        seen = set()
        for group in iter_groups(dataset_path):
            for row in group["members"]:
                bid = row["branch_sample_id"]; seen.add(bid)
                output, values = outputs[bid]
                writer.put(row, output, values, policy_sha256=policy["policy_sha256"], model_sha256="test")
        if set(outputs) != seen: raise ValueError("Extra replay outputs")
        counts = writer.finish(sampling, {policy["policy_sha256"]: policy}, {}, runtime_identity or {"status": "test_only"})
        seal(stage, "rl-replay", replay_id, dataset_receipt_sha256=identity["dataset_receipt_sha256"], probability_semantics=RAW_SEMANTICS)
    return {"status": "replay_estimate", "replay_id": replay_id, "path": str(destination), **counts}


class ReplayReader:
    def __init__(self, path, dataset_path, *, allow_estimates=False, legacy_diagnostic=False):
        self.path = Path(path); verify_local(path, "rl-replay")
        receipt = read_json(self.path / "replay.json")
        if receipt["dataset_receipt_sha256"] != sha256(Path(dataset_path) / "receipt.json"):
            raise ValueError("Replay is bound to another dataset receipt")
        if receipt.get("probability_semantics") not in ("processed_sampling_v1", RAW_SEMANTICS): raise ValueError("Unsupported replay probability semantics")
        self.semantics = receipt["probability_semantics"]
        if self.semantics == "processed_sampling_v1" and not legacy_diagnostic: raise ValueError("Legacy processed replay is read-only diagnostic data")
        if receipt["status"] != "historical_behavior_verified" and not allow_estimates:
            raise ValueError("Replay estimates require explicit allow_replay_estimates=True")
        self.group_ready = read_json(self.path / "selection.json")["group_probability_ready"]
        self.rows = {}
        for row in iter_rows(self.path / "branches.jsonl"):
            if row["branch_sample_id"] in self.rows: raise ValueError("Duplicate replay branch")
            self.rows[row["branch_sample_id"]] = row
        selected = read_json(self.path / "selection.json")
        group_ids = set(selected.get("group_ids", [])) if selected.get("coverage") == "explicit_complete_groups" else None
        all_groups = list(iter_groups(dataset_path, with_tokens=False))
        expected = [r for g in all_groups if group_ids is None or g["group"]["group_id"] in group_ids for r in g["members"]]
        if selected.get("coverage") == "explicit_missing_branches":
            identity = read_json(self.path / "validation.json")["identity"]
            branch_keys = {tuple(k) for k in identity["branch_keys"]}
            if (receipt.get("coverage_plan_sha256") != identity.get("coverage_plan_sha256")
                    or selected.get("coverage_plan_sha256") != identity.get("coverage_plan_sha256")):
                raise ValueError("Partial replay coverage plan binding mismatch")
            expected = [r for r in expected if (r["group_id"], r["branch_sample_id"]) in branch_keys]
            if len(expected) != len(branch_keys): raise ValueError("Partial replay selection differs from dataset")
        if set(self.rows) != {r["branch_sample_id"] for r in expected}: raise ValueError("Replay member coverage differs from dataset")
        for row in expected:
            ref = self.rows[row["branch_sample_id"]]
            if any(ref[k] != row[k] for k in BINDINGS) or ref["count"] != row["completion_count"] or ref["values_ref"]["dtype"] != "<f4":
                raise ValueError("Replay branch binding/count mismatch")
        expected_ready={}
        for row in expected:
            gid=row["group_id"]
            expected_ready[gid]=expected_ready.get(gid,True) and self.rows[row["branch_sample_id"]]["zero_support_count"]==0
        if selected.get("coverage") == "explicit_missing_branches":
            for group in all_groups:
                gid = group["group"]["group_id"]
                if gid in expected_ready and any(r["branch_sample_id"] not in self.rows for r in group["members"]):
                    expected_ready[gid] = False
        if self.group_ready!=expected_ready: raise ValueError("Replay group probability readiness mismatch")

    def values(self, row):
        ref = self.rows.get(row["branch_sample_id"])
        if ref is None or any(ref[k] != row[k] for k in BINDINGS): raise ValueError("Replay branch identity mismatch")
        if not self.group_ready.get(row["group_id"]): raise ValueError("Replay group has zero-support or missing probabilities")
        values = read_array(self.path, ref["values_ref"]).tolist()
        return validate_values(values, row["completion_count"], allow_zero_support=False)


def check_replay(path, *, dataset_path=None, with_values=False):
    path = Path(path); verify_local(path, "rl-replay")
    rows = list(iter_rows(path/"branches.jsonl")); seen = set(); tokens = 0
    selection = read_json(path/"selection.json")
    policies = read_json(path/"effective-policies.json")
    models = read_json(path/"models.json").get("models",{})
    for row in rows:
        if row["branch_sample_id"] in seen: raise ValueError("Duplicate replay branch")
        seen.add(row["branch_sample_id"]); tokens += row["count"]
        if row["count"] != row["completion_count"] or row["values_ref"]["count"] != row["count"]: raise ValueError("Replay count mismatch")
        if row["values_ref"]["dtype"] != "<f4": raise ValueError("Replay requires float32 arrays")
        if policies and row["policy_sha256"] not in policies: raise ValueError("Replay policy binding mismatch")
        if models and row["model_sha256"] not in models: raise ValueError("Replay model binding mismatch")
        if with_values:
            values = read_array(path, row["values_ref"])
            validate_values(values, row["count"])
            if int(np.isneginf(values).sum()) != row["zero_support_count"]: raise ValueError("Replay zero-support count mismatch")
    if [r["branch_sample_id"] for r in rows] != selection["branch_sample_ids"]: raise ValueError("Replay selection mismatch")
    summary=read_json(path/"summary.json")
    if summary["branches"]!=len(rows) or summary["tokens"]!=tokens: raise ValueError("Replay summary count mismatch")
    if dataset_path is not None: ReplayReader(path, dataset_path, allow_estimates=True, legacy_diagnostic=True)
    return {"status": "verified", "branches": len(rows), "tokens": tokens,
            "check_level": "values" if with_values else "metadata", "historical_behavior_verified": False}

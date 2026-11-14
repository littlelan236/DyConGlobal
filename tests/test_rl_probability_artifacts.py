"""CPU acceptance: actual online capture -> missing trace replay -> portable cache."""
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace as NS
from contextlib import nullcontext

import numpy as np
import pytest
import torch
import yaml

from core.io import hash_json, read_json, sha256, write_json
from RL.config import Grpo, schedule_contract
from RL.dataset.artifacts import directory, seal, verify_local, publication
from RL.dataset.build import iter_rows, json_line
from RL.dataset.config import ReplayConfig, CoveragePlanBinding, ProbabilityConfig
from RL.dataset.loader import iter_groups
from RL.dataset.materialize import read_array
from RL.dataset.probability import ProbabilityStream
from RL.dataset.probability_artifacts import probability_plan, probability_build, check_probability, read_plan
from RL.dataset.probability_identity import split_identity
from RL.dataset.replay.artifacts import ReplayWriter, check_replay
from RL.dataset.replay.inputs import Prefetch, selected_groups
from RL.dataset.replay.service import replay_logprobs
from RL.dataset.replay.validation import raw_contract, raw_policy
from RL.dataset.replay.worker import execute_batches
from RL.grpo.data import prepare, tensor_groups
from RL.grpo.reference import import_reference, load_reference
from RL.grpo.contract import validate_reference_files
from RL.checkpoints import export_adapter, adapter_contract
from RL.modeling import adapter_fingerprint
from test_grpo_external_reference import configuration
from tests.lora_helpers import model_and_targets

STREAM = "raw_t1_output_tokens"
OFFLINE_RUNTIME = {"logprobs_mode": "raw_logprobs", "enable_trace_replay": True,
                   "model_inventory_sha256": "unit-model", "replay_implementation": {"fixture": "raw-trace-cpu-v1"}}


def mixed_dataset(tmp_path, monkeypatch, *, missing=(1,), fail=False):
    """Attach factual sidecars before token_source freezes ledger hashes."""
    import test_rl_dataset as fixture
    from runtime.serving.raw_logprobs import RawLogprobWriter
    original = fixture.write_json
    writers = {}
    def write(path, value):
        if Path(path).parent.name == "requests" and "output_token_ids" in value:
            source_path = tmp_path / "runs" / value["run_id"] / "resolved-config.json"
            source = read_json(source_path); source["serving"]["dtype"] = "float16"
            original(source_path, source)
            branch = int(Path(path).stem)
            if branch not in missing:
                nas = Path(path).parent.parent
                if nas not in writers:
                    writers[nas] = RawLogprobWriter(tmp_path, nas, {}, source_identity={
                        "model_fingerprint": source["model"]["fingerprint"],
                        "checkpoint_inventory_sha256": source["model"]["fingerprint"],
                        "serving_dtype": "float16", "adapter": None})
                value["raw_t1_probability"] = writers[nas].write(value, [-.123456789, -.25, -1.125])
            elif fail:
                value["raw_t1_probability"] = {"status": "failed", "reason": "explicit CPU capture failure"}
        return original(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(fixture, "write_json", write)
        try:
            inputs = fixture.token_source(tmp_path)
        finally:
            for writer in writers.values(): writer.close()
    return fixture.build(inputs), inputs[0]


def source_scoring(dataset):
    binding = read_json(dataset / "dataset.json")["probability_streams"][STREAM]
    return read_json(dataset / binding["manifest"])["scoring_identity"]


def offline_scoring(dataset):
    model = read_json(next((dataset / "source-evidence").glob("*/resolved-config.json")))["model"]
    first = next(iter_rows(dataset / "branches.jsonl"))
    return {"checkpoint_fingerprint": model["fingerprint"], "tokenizer_fingerprint": first["tokenizer_fingerprint"],
            "adapter_fingerprint": None, "collector_dtype": "float16", "effective_model_overrides": {},
            "runtime": OFFLINE_RUNTIME, "implementation": OFFLINE_RUNTIME["replay_implementation"]}


def config_file(tmp_path, dataset, *, backfill=True, allow_partial=False):
    online = read_json(dataset / "dataset.json").get("probability_streams")
    identities = [offline_scoring(dataset)]
    if online: identities.append(source_scoring(dataset))
    config = {"schema_version": 1, "dataset": {"id": dataset.name, "root": str(dataset.parents[2])},
              "output": {"id": "merged", "root": str(tmp_path / "output"), "max_shard_bytes": 1024},
              "plan": {"id": "coverage", "root": str(tmp_path / "plans")},
              "sources": [{"kind": "dataset_stream", "id": dataset.name, "root": str(dataset.parents[2]), "stream": STREAM}],
              "cohort_policy": {"allowed_execution_cohorts": [hash_json(split_identity(i)[1]) for i in identities]},
              "allow_partial": allow_partial, "expected_scoring_identity": identities[0]}
    if backfill:
        config["sources"].append({"kind": "replay", "id": "backfill", "root": str(tmp_path / "replays"), "role": "backfill"})
    file = tmp_path / "probability.yaml"
    file.write_text(yaml.safe_dump(config))
    return config, file


def freeze(tmp_path, config, file):
    report = probability_plan({"project_root": str(tmp_path)}, file)
    config["coverage_plan"] = {**config["plan"], "sha256": report["plan_sha256"]}
    file.write_text(yaml.safe_dump(config))
    return report


def run_backfill(tmp_path, dataset, config, monkeypatch, *, interrupt=False, continuous=False):
    """Use formal dispatcher and actual ReplayWriter; engine is CPU-only."""
    replay_config = {"schema_version": 2, "replay_id": "backfill", "dataset": config["dataset"],
                     "selection": {"coverage_plan": config["coverage_plan"]},
                     "runtime": {"gpus": [0], "max_model_len": 32},
                     "storage": {"output_root": str(tmp_path / "replays"), "max_shard_bytes": 1024,
                                 "commit_every_branches": 1}}
    file = tmp_path / "replay.yaml"; file.write_text(yaml.safe_dump(replay_config))
    submitted = []
    def dispatch(python, job_file, log, seconds, root):
        job = read_json(job_file)
        replay = ReplayConfig.model_validate(job["replay_config"])
        stage = Path(job["stage"])
        writer = ReplayWriter(stage, dataset, job["identity"], 1024, recover=job["recover"], commit_every_branches=1)
        model_key = next(iter(job["plan"]["models"]))
        producer = Prefetch(dataset, None, replay)
        if continuous:
            import hashlib
            from RL.dataset.replay.output import AsyncWriter
            from RL.dataset.replay.worker import execute_continuous
            request_ids = {"raw-" + hashlib.sha256((replay.replay_id + "\0" + bid).encode()).hexdigest()[:40]: bid
                           for gid, bid in job["identity"]["branch_keys"]}
            class TraceEngine:
                def __init__(self): self.active = {}
                def get_num_unfinished_requests(self): return len(self.active)
                def has_unfinished_requests(self): return bool(self.active)
                def add_request(self, rid, prompt, params):
                    assert prompt["prompt_token_ids"] == [10, 11, 12]
                    submitted.append(request_ids[rid]); self.active[rid] = params.trace_decode_token_ids
                def step(self):
                    rid = next(reversed(self.active))
                    tokens = self.active.pop(rid)
                    return [NS(request_id=rid, finished=True, outputs=[NS(token_ids=tokens,
                               logprobs=[{t: NS(logprob=-.75)} for t in tokens])])]
            asynchronous = AsyncWriter(writer, 2)
            runtime = copy.deepcopy(OFFLINE_RUNTIME)
            try:
                execute_continuous(NS(llm_engine=TraceEngine()), lambda **kw: NS(**kw, _validate_trace_replay=lambda *a: None),
                                   producer, asynchronous, lambda row: raw_policy(), model_key, replay, runtime, 100)
                asynchronous.finish(raw_contract(), {raw_policy()["policy_sha256"]: raw_policy()}, job["plan"]["models"], runtime)
            finally:
                asynchronous.close(); producer.close()
            return
        rows = []
        try:
            while True:
                group = producer.get()
                if group is producer.END: break
                if group is None: continue
                rows.extend(group["members"])
                producer.release(group)
        finally:
            producer.close()
        class Engine:
            def generate(self, prompts, params, use_tqdm):
                result = []
                for prompt, param in zip(prompts, params):
                    tokens = param.trace_decode_token_ids
                    assert prompt["prompt_token_ids"] == [10, 11, 12]
                    result.append(NS(finished=True, outputs=[NS(token_ids=tokens,
                        logprobs=[{t: NS(logprob=-.75)} for t in tokens])]))
                return result
        for row in reversed(rows):
            if row["branch_sample_id"] in writer.finished:
                writer.verify_completed(row, policy_sha256=raw_policy()["policy_sha256"], model_sha256=model_key)
                continue
            submitted.append(row["branch_sample_id"])
            execute_batches(Engine(), lambda **kw: NS(**kw), [row], 1, writer, lambda r: raw_policy(), model_key)
            if interrupt and not job["recover"]: raise InterruptedError("CPU test interruption after sealed branch")
        writer.finish(raw_contract(), {raw_policy()["policy_sha256"]: raw_policy()}, job["plan"]["models"], OFFLINE_RUNTIME)
    monkeypatch.setattr("RL.dataset.replay.service.dispatch", dispatch)
    monkeypatch.setattr("hidden.backends.vllm.engine.gpu_reservation", lambda *a, **kw: nullcontext())
    if interrupt:
        with pytest.raises(InterruptedError): replay_logprobs({"project_root": str(tmp_path)}, file)
        assert not directory(tmp_path / "replays", "rl-replay", "backfill").exists()
    result = replay_logprobs({"project_root": str(tmp_path)}, file, recover=interrupt)
    return result, submitted


def open_stream(path, dataset):
    return ProbabilityStream(path, kind="rl-probability", manifest_name="probability.json", index_name="branches.jsonl",
                             dataset_receipt_sha256=sha256(dataset / "receipt.json"))


def import_cache(tmp_path, dataset, artifact, monkeypatch):
    config = configuration(tmp_path)
    config.algorithm.dataset_root = str(dataset.parents[2]); config.algorithm.dataset_id = dataset.name
    config.training.max_updates = 1; config.grpo.top_k = 1
    config.grpo = Grpo.model_validate({**config.grpo.model_dump(), "reference_cache": {
        "source": "probability_artifact", "artifact": {"scope": "rl-probability", "id": artifact.name,
            "root": str(artifact.parents[2])}, "dtype_compatibility": "fp16_source_bf16_training"}})
    raw, proof = prepare(config); groups = list(tensor_groups(raw))
    base = {"fingerprint": proof["dataset_model"]["fingerprint"]}
    model, targets = model_and_targets()
    def forbidden(*a, **kw): raise AssertionError("probability import evaluated a model")
    monkeypatch.setattr(model, "forward", forbidden)
    dest = tmp_path / "attempt"; dest.mkdir()
    adapter = adapter_contract(base, targets, config.lora)
    export_adapter(model, dest / "grpo/reference/adapter", adapter)
    values, reference = import_reference(model, groups, raw, dest, proof, base, hash_json(adapter_fingerprint(model)), config)
    contract = {"grpo": reference, "input": proof, "adapter": adapter, "schedule": schedule_contract(config)}
    validate_reference_files(dest, contract)
    return values, reference, dest, groups, proof, contract


def test_mixed_missing_only_to_portable_training_cache(tmp_path, monkeypatch):
    dataset, source = mixed_dataset(tmp_path, monkeypatch, missing=(1, 3))
    originals = {str(p): sha256(p) for p in dataset.rglob("*") if p.is_file()}
    config, file = config_file(tmp_path, dataset)
    plan = freeze(tmp_path, config, file)
    assert plan["missing_count"] == 2 and plan["valid"] == 4
    result, submitted = run_backfill(tmp_path, dataset, config, monkeypatch, interrupt=True)
    missing = list(iter_rows(Path(plan["path"]) / "missing.jsonl"))
    assert set(submitted) == {r["branch_sample_id"] for r in missing} and len(submitted) == 2
    replay = Path(result["path"])
    assert check_replay(replay, dataset_path=dataset, with_values=True)["branches"] == 2
    selection = read_json(replay / "selection.json")
    assert not any(selection["group_probability_ready"].values())
    assert not read_json(replay / "summary.json")["dataset_probability_ready"]
    built = probability_build({"project_root": str(tmp_path)}, file)
    artifact = Path(built["path"])
    stream = open_stream(artifact, dataset)
    old_binding = read_json(dataset / "dataset.json")["probability_streams"][STREAM]
    online = ProbabilityStream(dataset, kind="rl-dataset", manifest_name=old_binding["manifest"], index_name=old_binding["index"],
                               dataset_receipt_sha256=old_binding["dataset_binding"], stream=STREAM)
    for group in iter_groups(dataset):
        for row in group["members"]:
            if (row["group_id"], row["branch_sample_id"]) in online.records:
                assert stream.values(row).tobytes() == online.values(row).tobytes()
            else: np.testing.assert_array_equal(stream.values(row), np.full(3, -.75, dtype="<f4"))
    assert originals == {str(p): sha256(p) for p in dataset.rglob("*") if p.is_file()}
    assert not next(iter_groups(dataset))["group"]["raw_probability_complete"]
    values, reference, dest, groups, proof, contract = import_cache(tmp_path, dataset, artifact, monkeypatch)
    assert len(reference["probability_binding"]["execution_cohorts"]) == 2
    moved = directory(tmp_path / "portable", "rl-probability", artifact.name)
    moved.parent.mkdir(parents=True); shutil.copytree(artifact, moved)
    shutil.rmtree(replay); shutil.rmtree(source["nas"]); shutil.rmtree(dataset); shutil.rmtree(artifact)
    assert check_probability(moved, with_values=True)["finite_coverage_complete"]
    assert check_probability(moved, sources=True)["source_audit_status"] == "incomplete"
    validate_reference_files(dest, contract)
    load_reference(dest, groups, proof, reference)
    assert sum(len(b) for b in values.values()) == 6


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_single_source_and_zero_missing_worker(tmp_path, monkeypatch, mode):
    dataset, _ = mixed_dataset(tmp_path, monkeypatch, missing=() if mode == "online" else tuple(range(6)))
    config, file = config_file(tmp_path, dataset, backfill=True)
    plan = freeze(tmp_path, config, file)
    if mode == "online":
        def forbidden(*a, **kw): raise AssertionError("zero missing started a worker")
        monkeypatch.setattr("RL.dataset.replay.service.dispatch", forbidden)
        result, submitted = run_backfill(tmp_path, dataset, config, monkeypatch)
        assert result["status"] == "no_missing" and not submitted
        assert not directory(tmp_path / "replays", "rl-replay", "backfill").exists()
    else:
        assert plan["missing_count"] == 6
        run_backfill(tmp_path, dataset, config, monkeypatch)
    artifact = Path(probability_build({"project_root": str(tmp_path)}, file)["path"])
    assert check_probability(artifact, with_values=True)["covered_count"] == 6
    assert len(open_stream(artifact, dataset).records) == 6


def test_plan_missing_reason_nonfinite_and_corruption(tmp_path, monkeypatch):
    dataset, _ = mixed_dataset(tmp_path, monkeypatch, missing=(1,), fail=True)
    config, file = config_file(tmp_path, dataset)
    with pytest.raises(ValueError, match="recorded_failed"):
        probability_plan({"project_root": str(tmp_path)}, file)
    config["allowed_missing_reasons"] = ["absent", "recorded_failed"]
    file.write_text(yaml.safe_dump(config)); plan = freeze(tmp_path, config, file)
    assert plan["recorded_failed"] == 1


@pytest.mark.parametrize("fault", ["hash", "request", "mask", "locator", "processed", "nan", "inf", "positive", "negative_inf"])
def test_invalid_sources_are_never_silently_recomputed(tmp_path, monkeypatch, fault):
    dataset, _ = mixed_dataset(tmp_path, monkeypatch, missing=())
    binding = read_json(dataset / "dataset.json")["probability_streams"][STREAM]
    index = dataset / binding["index"]; rows = list(iter_rows(index))
    record = rows[0]
    if fault == "hash": record["values_ref"]["sha256"] = "0" * 64
    elif fault == "request": record["native_request_id"] = "bad"
    elif fault == "mask": record["mask_spans"] = []
    elif fault == "locator": record["values_ref"]["offset"] = -1
    elif fault == "processed":
        m = read_json(dataset / binding["manifest"]); m["probability_semantics"] = "processed_sampling_v1"
        write_json(dataset / binding["manifest"], m)
    else:
        from RL.dataset.materialize import ShardWriter
        (dataset / "injected").mkdir()
        writer = ShardWriter(dataset / "injected", "<f4", 1024)
        ref = writer.put([{"nan": np.nan, "inf": np.inf, "positive": .1, "negative_inf": -np.inf}[fault], -.2, -.3]); writer.close()
        ref["shard"] = "injected/" + ref["shard"]; record["values_ref"] = ref
    index.write_text("".join(json.dumps(r) + "\n" for r in rows))
    seal(dataset, "rl-dataset", dataset.name)
    config, file = config_file(tmp_path, dataset, backfill=False)
    # Update the intentionally resealed fixture's online content binding.
    content = hash_json({"branches_sha256": sha256(dataset / "branches.jsonl"), "groups_sha256": sha256(dataset / "groups.jsonl")})
    assert content == binding["dataset_binding"]
    if fault == "negative_inf":
        plan = probability_plan({"project_root": str(tmp_path)}, file)
        assert plan["nonfinite"] == 1 and plan["missing_count"] == 0
        artifact = Path(probability_build({"project_root": str(tmp_path)}, file)["path"])
        assert not check_probability(artifact, with_values=True)["finite_coverage_complete"]
        row = next(iter_groups(dataset))["members"][0]
        with pytest.raises(ValueError, match="finite"):
            open_stream(artifact, dataset).values(row)
    else:
        with pytest.raises(ValueError): probability_plan({"project_root": str(tmp_path)}, file)


def test_changed_receipt_plan_and_recovery_signature_refused(tmp_path, monkeypatch):
    dataset, _ = mixed_dataset(tmp_path, monkeypatch)
    config, file = config_file(tmp_path, dataset); report = freeze(tmp_path, config, file)
    bad = CoveragePlanBinding.model_validate({**config["coverage_plan"], "sha256": "0" * 64})
    with pytest.raises(ValueError, match="hash"):
        list(selected_groups(dataset, ReplayConfig.model_validate({"schema_version": 2, "replay_id": "x", "dataset": config["dataset"],
             "selection": {"coverage_plan": bad.model_dump()}, "runtime": {"gpus": [0]}, "storage": {"output_root": str(tmp_path)}})))
    with pytest.raises(ValueError): ReplayConfig.model_validate({"schema_version": 2, "replay_id": "x", "dataset": config["dataset"],
        "selection": {"group_ids": ["x"], "coverage_plan": config["coverage_plan"]}, "runtime": {"gpus": [0]}, "storage": {"output_root": str(tmp_path)}})
    receipt = read_json(dataset / "receipt.json"); receipt["created_at"] = "changed"
    write_json(dataset / "receipt.json", receipt)
    with pytest.raises(ValueError, match="receipt"):
        read_plan(report["path"], dataset=dataset, check_sources=True)


def test_build_recovery_and_same_id_lock(tmp_path, monkeypatch):
    dataset, _ = mixed_dataset(tmp_path, monkeypatch, missing=())
    config, file = config_file(tmp_path, dataset, backfill=False)
    import RL.dataset.probability_artifacts as module
    original = module.seal
    original_yaml = file.read_bytes()
    def interrupted(*a, **kw): raise InterruptedError("before receipt")
    monkeypatch.setattr(module, "seal", interrupted)
    with pytest.raises(InterruptedError): probability_build({"project_root": str(tmp_path)}, file)
    assert not directory(config["output"]["root"], "rl-probability", "merged").exists()
    monkeypatch.setattr(module, "seal", original)
    config["deduplicate_identical"] = True; file.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="signature"):
        probability_build({"project_root": str(tmp_path)}, file, recover=True)
    file.write_bytes(original_yaml)
    artifact = Path(probability_build({"project_root": str(tmp_path)}, file, recover=True)["path"])
    with pytest.raises(FileExistsError): probability_build({"project_root": str(tmp_path)}, file)
    with pytest.raises(InterruptedError):
        with publication(tmp_path / "locks", "rl-probability", "same", signature={}):
            with pytest.raises((BlockingIOError, RuntimeError)):
                with publication(tmp_path / "locks", "rl-probability", "same", signature={}): pass
            raise InterruptedError("incomplete transaction")


def expanded_fixture(dataset, group_sizes, missing_positions):
    """Explicit synthetic group multiplicity; fixed real native token payloads."""
    from RL.dataset.materialize import ShardWriter
    from RL.dataset.probability import BINDINGS
    original_group = next(iter_groups(dataset))
    template = original_group["members"][0]
    scoring = source_scoring(dataset)
    groups, branches, records = [], [], []
    writer = ShardWriter(dataset, "<f4", 1024)
    try:
        for gid, size in enumerate(group_sizes):
            members = []
            for index in range(size):
                row = copy.deepcopy(template)
                row.update(group_id=f"expanded-group-{gid}", branch_sample_id=f"expanded-g{gid}-b{index}")
                row["historical_score"] = float(index)
                row.pop("prompt_token_ids"); row.pop("completion_token_ids")
                row["raw_probability_available"] = (gid, index) not in missing_positions
                if not row["raw_probability_available"]:
                    row["probability_streams"] = {}
                else:
                    record = {k: row[k] for k in (*BINDINGS, "mask_spans")}
                    record.update(values_ref=writer.put([-.123456789, -.25, -1.125]), scoring_identity_sha256=hash_json(scoring))
                    records.append(record)
                members.append(row)
            group = {**original_group["group"], "group_id": f"expanded-group-{gid}", "expected_branch_count": size,
                     "branch_ids": [r["branch_sample_id"] for r in members],
                     "raw_probability_complete": all(r["raw_probability_available"] for r in members)}
            groups.append(group); branches.extend(members)
    finally:
        writer.close()
    for filename, rows in (("groups.jsonl", groups), ("branches.jsonl", branches), ("raw-t1-probability-branches.jsonl", records)):
        with (dataset / filename).open("w", encoding="utf-8") as out:
            for row in rows: json_line(out, row)
    content = hash_json({"branches_sha256": sha256(dataset / "branches.jsonl"), "groups_sha256": sha256(dataset / "groups.jsonl")})
    manifest = read_json(dataset / "raw-t1-probability.json"); manifest["dataset_receipt_sha256"] = content
    write_json(dataset / "raw-t1-probability.json", manifest)
    meta = read_json(dataset / "dataset.json"); meta["probability_streams"][STREAM]["dataset_binding"] = content
    write_json(dataset / "dataset.json", meta); seal(dataset, "rl-dataset", dataset.name)


@pytest.mark.parametrize("sizes,missing", [([32], {(0, 17)}), ([8, 8], {(0, 1), (0, 5), (1, 0), (1, 4)})])
def test_large_and_interleaved_groups_submit_only_missing(tmp_path, monkeypatch, sizes, missing):
    dataset, _ = mixed_dataset(tmp_path, monkeypatch, missing=())
    expanded_fixture(dataset, sizes, missing)
    config, file = config_file(tmp_path, dataset); plan = freeze(tmp_path, config, file)
    before = {r["branch_sample_id"]: r["values_ref"]["sha256"] for r in iter_rows(dataset / "raw-t1-probability-branches.jsonl")}
    _, submitted = run_backfill(tmp_path, dataset, config, monkeypatch, continuous=True)
    assert set(submitted) == {f"expanded-g{g}-b{b}" for g, b in missing}
    assert len(submitted) == len(missing) == plan["missing_count"]
    artifact = Path(probability_build({"project_root": str(tmp_path)}, file)["path"])
    records = list(iter_rows(artifact / "branches.jsonl"))
    assert [r["branch_sample_id"] for r in records] == [r["branch_sample_id"] for r in iter_rows(dataset / "branches.jsonl")]
    assert len(records) == sum(sizes)
    assert {r["branch_sample_id"]: r["values_ref"]["sha256"] for r in records if r["branch_sample_id"] in before} == before


@pytest.mark.parametrize("fault", ["checkpoint", "tokenizer", "adapter", "overrides", "dtype", "runtime", "incomplete", "extra", "duplicate", "wrong_plan"])
def test_backfill_function_cohort_and_exact_missing_contract(tmp_path, monkeypatch, fault):
    dataset, _ = mixed_dataset(tmp_path, monkeypatch, missing=(1,))
    config, file = config_file(tmp_path, dataset); freeze(tmp_path, config, file)
    result, _ = run_backfill(tmp_path, dataset, config, monkeypatch)
    replay = Path(result["path"])
    if fault in {"extra", "duplicate"}:
        rows = list(iter_rows(replay / "branches.jsonl"))
        extra = copy.deepcopy(rows[0])
        if fault == "extra":
            row = next(r for g in iter_groups(dataset, with_tokens=False) for r in g["members"] if r["branch_sample_id"] != extra["branch_sample_id"])
            from RL.dataset.probability import BINDINGS
            extra.update({k: row[k] for k in BINDINGS})
        rows.append(extra)
        (replay / "branches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif fault == "wrong_plan":
        manifest = read_json(replay / "replay.json"); manifest["coverage_plan_sha256"] = "0" * 64
        write_json(replay / "replay.json", manifest)
    else:
        # Explicit schema-2 adapter fixture, retaining actual replay arrays.
        from RL.dataset.probability import BINDINGS
        scoring = offline_scoring(dataset)
        key = {"checkpoint": "checkpoint_fingerprint", "tokenizer": "tokenizer_fingerprint", "adapter": "adapter_fingerprint",
               "overrides": "effective_model_overrides", "dtype": "collector_dtype", "runtime": "runtime", "incomplete": "implementation"}[fault]
        scoring[key] = {"rope_theta": 2} if fault == "overrides" else "bfloat16" if fault == "dtype" else {} if fault == "incomplete" else {"backend": "undeclared"} if fault == "runtime" else "changed"
        manifest = read_json(replay / "replay.json"); manifest.pop("raw_contract_sha256")
        manifest.update(raw_contract=raw_contract_neutral(), scoring_identity=scoring)
        write_json(replay / "replay.json", manifest)
        rows = list(iter_rows(replay / "branches.jsonl"))
        for row in rows:
            row["mask_spans"] = next(r["mask_spans"] for g in iter_groups(dataset, with_tokens=False) for r in g["members"] if r["branch_sample_id"] == row["branch_sample_id"])
            row["scoring_identity_sha256"] = hash_json(scoring)
            if fault == "tokenizer": row["tokenizer_fingerprint"] = "changed"
        (replay / "branches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    seal(replay, "rl-replay", replay.name)
    with pytest.raises(ValueError): probability_build({"project_root": str(tmp_path)}, file)
    assert not directory(config["output"]["root"], "rl-probability", "merged").exists()


def raw_contract_neutral():
    from RL.dataset.probability import RAW_CONTRACT
    return RAW_CONTRACT


def test_partial_artifact_never_changes_training_eligibility(tmp_path, monkeypatch):
    dataset, _ = mixed_dataset(tmp_path, monkeypatch, missing=(1, 3))
    config, file = config_file(tmp_path, dataset, backfill=False, allow_partial=True)
    artifact = Path(probability_build({"project_root": str(tmp_path)}, file)["path"])
    assert check_probability(artifact, with_values=True)["coverage_status"] == "partial"
    assert len(list(iter_groups(dataset))) == 1 and len(next(iter_groups(dataset))["members"]) == 6
    with pytest.raises(ValueError, match="Missing required"):
        import_cache(tmp_path, dataset, artifact, monkeypatch)


def test_repackage_deduplicates_only_explicit_identical_sources(tmp_path, monkeypatch):
    dataset, _ = mixed_dataset(tmp_path, monkeypatch, missing=())
    config, file = config_file(tmp_path, dataset, backfill=False)
    artifact = Path(probability_build({"project_root": str(tmp_path)}, file)["path"])
    config["sources"].append({"kind": "probability", "id": artifact.name, "root": str(artifact.parents[2])})
    config["output"]["id"] = "repacked"
    file.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="duplicate"):
        probability_build({"project_root": str(tmp_path)}, file)
    config["deduplicate_identical"] = True; file.write_text(yaml.safe_dump(config))
    repacked = Path(probability_build({"project_root": str(tmp_path)}, file)["path"])
    assert all(len(r["source_ref"]) == 2 for r in iter_rows(repacked / "branches.jsonl"))
    assert check_probability(repacked, with_values=True)["covered_count"] == 6
    # Canonical-only migration uses exactly the same writer and reader.
    config["sources"] = config["sources"][1:]; config["output"]["id"] = "canonical-copy"
    file.write_text(yaml.safe_dump(config))
    copied = Path(probability_build({"project_root": str(tmp_path)}, file)["path"])
    assert [r["values_ref"]["sha256"] for r in iter_rows(copied / "branches.jsonl")] == [r["values_ref"]["sha256"] for r in iter_rows(artifact / "branches.jsonl")]


def test_partial_superset_covers_all_training_eligible_branches(tmp_path, monkeypatch):
    dataset, _ = mixed_dataset(tmp_path, monkeypatch, missing=())
    expanded_fixture(dataset, [6, 6], {(1, i) for i in range(6)})
    rows = list(iter_rows(dataset / "branches.jsonl"))
    for row in rows:
        if row["group_id"] == "expanded-group-1": row["historical_score"] = 0.0
    (dataset / "branches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    content = hash_json({"branches_sha256": sha256(dataset / "branches.jsonl"), "groups_sha256": sha256(dataset / "groups.jsonl")})
    manifest = read_json(dataset / "raw-t1-probability.json"); manifest["dataset_receipt_sha256"] = content
    write_json(dataset / "raw-t1-probability.json", manifest)
    meta = read_json(dataset / "dataset.json"); meta["probability_streams"][STREAM]["dataset_binding"] = content
    write_json(dataset / "dataset.json", meta); seal(dataset, "rl-dataset", dataset.name)
    config, file = config_file(tmp_path, dataset, backfill=False, allow_partial=True)
    artifact = Path(probability_build({"project_root": str(tmp_path)}, file)["path"])
    values, ref, dest, groups, proof, contract = import_cache(tmp_path, dataset, artifact, monkeypatch)
    assert len(groups) == 1 and len(values[groups[0].group_id]) == 6
    assert proof["exclusion_reasons"] == {"all_scores_equal": 1}
    assert check_probability(artifact)["coverage_status"] == "partial"


def test_online_recorded_available_but_index_omission_is_corruption(tmp_path, monkeypatch):
    dataset, _ = mixed_dataset(tmp_path, monkeypatch, missing=())
    index = dataset / "raw-t1-probability-branches.jsonl"
    index.write_text("".join(json.dumps(r) + "\n" for r in list(iter_rows(index))[1:]))
    seal(dataset, "rl-dataset", dataset.name)
    _, file = config_file(tmp_path, dataset)
    with pytest.raises(ValueError, match="availability"):
        probability_plan({"project_root": str(tmp_path)}, file)


def test_empty_partial_preserves_all_missing_targets(tmp_path, monkeypatch):
    dataset, _ = mixed_dataset(tmp_path, monkeypatch, missing=tuple(range(6)))
    config, file = config_file(tmp_path, dataset, backfill=False, allow_partial=True)
    report = probability_plan({"project_root": str(tmp_path)}, file)
    assert report["absent"] == 6 and report["covered_count"] == 0
    artifact = Path(probability_build({"project_root": str(tmp_path)}, file)["path"])
    assert not open_stream(artifact, dataset).records
    assert check_probability(artifact)["missing_count"] == 6


def test_explicit_group_subset_freezes_full_group_inventory(tmp_path, monkeypatch):
    dataset, _ = mixed_dataset(tmp_path, monkeypatch, missing=())
    expanded_fixture(dataset, [8, 8], {(0, 1), (1, 5)})
    config, file = config_file(tmp_path, dataset)
    config["group_ids"] = ["expanded-group-1"]; file.write_text(yaml.safe_dump(config))
    report = freeze(tmp_path, config, file)
    assert report["target_count"] == 8 and report["missing_count"] == 1
    _, submitted = run_backfill(tmp_path, dataset, config, monkeypatch, continuous=True)
    assert submitted == ["expanded-g1-b5"]
    artifact = Path(probability_build({"project_root": str(tmp_path)}, file)["path"])
    assert {r["group_id"] for r in iter_rows(artifact / "branches.jsonl")} == {"expanded-group-1"}


def test_probability_cli_and_global_scope(tmp_path, monkeypatch, capsys):
    from core.cli import main
    dataset, _ = mixed_dataset(tmp_path, monkeypatch, missing=())
    config, file = config_file(tmp_path, dataset, backfill=False)
    args = ["rl-dataset", "probability-build", "--probability-config", str(file)]
    main(args); output = json.loads(capsys.readouterr().out)
    assert output["probability_id"] == "merged"
    main(["rl-dataset", "probability-check", "merged", "--storage-root", config["output"]["root"], "--with-values"])
    assert json.loads(capsys.readouterr().out)["covered_count"] == 6
    main(["artifact", "verify", "merged", "--scope", "rl-probability", "--storage-root", config["output"]["root"]])
    assert json.loads(capsys.readouterr().out)["scope"] == "rl-probability"

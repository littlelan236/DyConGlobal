"""Sealed raw sources, no-forward imports, shared anchor and exact resume."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from core.io import hash_json, read_json, sha256, write_json
from RL.artifacts import commit_receipt
from RL.config import Grpo, load, schedule_contract
from RL.dataset.artifacts import REQUIRED, directory, seal
from RL.dataset.materialize import ShardWriter
from RL.dataset.probability import RAW_CONTRACT
from RL.dataset.schema import spans
from RL.grpo.data import prepare, tensor_groups
from RL.grpo.reference import (adapter_compatibility, build_reference, import_reference,
                               load_reference, validate_scoring_identity)
from RL.grpo.contract import validate_reference_files, validate_probability_binding, validate_resume
from RL.modeling import adapter_fingerprint
from RL.checkpoints import adapter_contract, export_adapter
from RL.service import _source, _train_grpo
from tests.lora_helpers import LORA, model_and_targets

ROOT = Path(__file__).parents[1]
BASE = {"fingerprint": "a" * 64}


def configuration(tmp_path):
    config, _ = load(ROOT / "RL/experiments/grpo-offline-k3-smoke.yaml", ROOT)
    config.preflight = config.preflight.model_copy(update={k: True for k in type(config.preflight).model_fields})
    config.lora = LORA.model_copy()
    config.runtime.output_root = str(tmp_path / "training")
    return config


def make_source(tmp_path, config, *, vectors=None, online=False, mutate=None, name="raw"):
    raw, proof = prepare(config)
    scoring = {"checkpoint_fingerprint": BASE["fingerprint"],
        "tokenizer_fingerprint": proof["tokenizer_fingerprint"], "adapter_fingerprint": None,
        "collector_dtype": "float16", "effective_model_overrides": {},
        "runtime": {"backend": "explicit_test_fixture"}, "implementation": {"test": "raw-v1"}}
    kind = "rl-dataset" if online else "rl-replay"
    path = directory(tmp_path, kind, name); path.mkdir(parents=True)
    for filename in REQUIRED[kind]:
        target = path / filename
        target.write_text("" if filename.endswith(".jsonl") else "{}")
    writer = ShardWriter(path, "<f4", 1024 * 1024)
    records = []
    for group in raw:
        for row in group["members"]:
            bid = row["branch_sample_id"]; p = row["prompt_token_ids"]; c = row["completion_token_ids"]
            values = vectors[group["group"]["group_id"]][bid].numpy() if vectors else np.arange(len(c), dtype="<f4") / -10 - 4
            records.append({"group_id": group["group"]["group_id"], "branch_sample_id": bid,
                "native_request_id": "synthetic:" + bid, "prompt_sha256": hash_json(p),
                "completion_sha256": hash_json(c), "prompt_count": len(p), "completion_count": len(c),
                "tokenizer_fingerprint": proof["tokenizer_fingerprint"], "mask_spans": spans(len(p), len(c)),
                "scoring_identity_sha256": hash_json(scoring), "values_ref": writer.put(values)})
    writer.close()
    # Deliberately reverse producer completion order; training order must prevail.
    records.reverse()
    manifest = {"schema_version": 2, "replay_id": name,
        "probability_semantics": "raw_t1_sampled_token_v1" if online else "raw_t1_output_tokens_v1",
        "raw_contract": RAW_CONTRACT, "dataset_receipt_sha256": proof["dataset_receipt_sha256"],
        "scoring_identity": scoring, "stream": "raw_t1_output_tokens"}
    if mutate:
        mutate(manifest, records, path)
    manifest_name = "probability.json" if online else "replay.json"
    index_name = "probability-branches.jsonl" if online else "branches.jsonl"
    write_json(path / manifest_name, manifest)
    (path / index_name).write_text("".join(json.dumps(r) + "\n" for r in records))
    if online:
        write_json(path / "dataset.json", {"probability_streams": {"raw_t1_output_tokens": {
            "manifest": manifest_name, "index": index_name, "dataset_binding": proof["dataset_receipt_sha256"]}}})
    seal(path, kind, name)
    return path


def select(config, tmp_path, *, name="raw", online=False):
    value = {"source": "dataset_probability", "stream": "raw_t1_output_tokens"} if online else {
        "source": "probability_artifact", "artifact": {"scope": "rl-replay", "root": str(tmp_path), "id": name}}
    value["dtype_compatibility"] = "fp16_source_bf16_training"
    config.grpo = Grpo.model_validate({**config.grpo.model_dump(), "reference_cache": value})


def import_cache(tmp_path, config, monkeypatch=None):
    raw, proof = prepare(config); groups = list(tensor_groups(raw))
    model, targets = model_and_targets()
    dest = tmp_path / "attempt"; dest.mkdir(parents=True)
    export_adapter(model, dest / "grpo/reference/adapter", adapter_contract(BASE, targets, config.lora))
    if monkeypatch:
        def forbidden(*args, **kwargs):
            raise AssertionError("external import performed a model/reference forward")
        monkeypatch.setattr(model, "forward", forbidden)
        monkeypatch.setattr("RL.grpo.reference.build_reference", forbidden)
    values, ref = import_reference(model, groups, raw, dest, proof, BASE,
        hash_json(adapter_fingerprint(model)), config)
    return values, ref, dest, groups, proof


def test_strict_sources_and_legacy_schedule(tmp_path):
    config = configuration(tmp_path)
    assert config.grpo.reference_cache.source == "model_forward"
    assert "reference_cache" not in schedule_contract(config)["grpo"]
    bad = [{"source": "model_forward", "stream": "raw"},
        {"source": "dataset_probability", "stream": "raw"},
        {"source": "dataset_probability", "stream": "", "dtype_compatibility": "same_dtype"},
        {"source": "probability_artifact", "artifact": {"scope": "run", "root": str(tmp_path), "id": "x"}, "dtype_compatibility": "same_dtype"},
        {"source": "dataset_probability", "stream": "raw", "artifact": {}, "dtype_compatibility": "same_dtype"}]
    for value in bad:
        with pytest.raises(ValueError):
            Grpo.model_validate({"reference_cache": value})


def test_external_import_never_forwards_and_freezes_training_order(tmp_path, monkeypatch):
    config = configuration(tmp_path); make_source(tmp_path, config); select(config, tmp_path)
    values, reference, dest, groups, proof = import_cache(tmp_path, config, monkeypatch)
    assert list(values) == [g.group_id for g in groups]
    assert reference["role_binding"]["ratio_denominator"] == reference["role_binding"]["k3_reference"]
    assert reference["probability_binding"]["collector_dtype"] == "float16"
    assert reference["probability_binding"]["training_compute_dtype"] == "bfloat16"
    assert read_json(dest / "grpo/reference.json")["schema_version"] == 2
    load_reference(dest, groups, proof, reference)
    for group in groups:
        assert list(values[group.group_id]) == group.branch_ids
        for vector in values[group.group_id].values():
            assert vector.dtype == torch.float32 and not vector.requires_grad


@pytest.mark.parametrize("fault", ["missing", "duplicate", "extra", "token", "request", "mask", "hash", "dtype", "nan", "inf", "neginf", "processed", "temperature", "checkpoint", "tokenizer", "adapter", "overrides", "staging"])
def test_bad_external_source_refused(tmp_path, fault):
    config = configuration(tmp_path)
    def mutate(m, rows, path):
        if fault == "missing": rows.pop()
        elif fault == "duplicate": rows.append(rows[0])
        elif fault == "extra": rows.append({**rows[0], "branch_sample_id": "unbound"})
        elif fault == "token": rows[0]["completion_sha256"] = "bad"
        elif fault == "request": rows[0]["native_request_id"] = "wrong"
        elif fault == "mask": rows[0]["mask_spans"] = []
        elif fault == "hash": rows[0]["values_ref"]["sha256"] = "0" * 64
        elif fault == "dtype": rows[0]["values_ref"]["dtype"] = "<f8"
        elif fault in {"nan", "inf", "neginf"}:
            bad = {"nan": np.nan, "inf": np.inf, "neginf": -np.inf}[fault]
            # Use a separate directory to avoid overwriting the original shared shard.
            extra = path / "bad"; extra.mkdir(); writer = ShardWriter(extra, "<f4", 1000)
            ref = writer.put([bad] * rows[0]["completion_count"]); writer.close()
            ref["shard"] = "bad/" + ref["shard"]; rows[0]["values_ref"] = ref
        elif fault == "processed": m["probability_semantics"] = "processed_sampling_v1"
        elif fault == "temperature": m["raw_contract"] = {**RAW_CONTRACT, "temperature": .6}
        elif fault in {"checkpoint", "tokenizer", "adapter", "overrides"}:
            field = {"checkpoint": "checkpoint_fingerprint", "tokenizer": "tokenizer_fingerprint",
                "adapter": "adapter_fingerprint", "overrides": "effective_model_overrides"}[fault]
            m["scoring_identity"][field] = {"rope": "changed"} if fault == "overrides" else "wrong"
            for row in rows:
                row["scoring_identity_sha256"] = hash_json(m["scoring_identity"])
    path = make_source(tmp_path, config, mutate=mutate); select(config, tmp_path)
    if fault == "staging":
        (path / "receipt.json").unlink()
    with pytest.raises((ValueError, KeyError)):
        import_cache(tmp_path, config)


def test_source_superset_and_missing_required_member(tmp_path):
    config = configuration(tmp_path); make_source(tmp_path, config); select(config, tmp_path)
    raw, proof = prepare(config); groups = list(tensor_groups(raw))[:2]
    model, targets = model_and_targets(); dest = tmp_path / "subset"; dest.mkdir()
    export_adapter(model, dest / "grpo/reference/adapter", adapter_contract(BASE, targets, config.lora))
    values, _ = import_reference(model, groups, raw, dest, proof, BASE, hash_json(adapter_fingerprint(model)), config)
    assert list(values) == [g.group_id for g in groups]


def test_fresh_lora_equivalence_and_dtype_are_independent(tmp_path):
    config = configuration(tmp_path); model, _ = model_and_targets()
    adapter = adapter_compatibility(model, start_mode="fresh", adapter_hash="initial")
    assert adapter["kind"] == "base_equivalent"
    scoring = {"checkpoint_fingerprint": BASE["fingerprint"], "tokenizer_fingerprint": "t",
        "adapter_fingerprint": None, "collector_dtype": "float16", "effective_model_overrides": {}}
    kwargs = dict(base=BASE, proof={"tokenizer_fingerprint": "t"}, training_dtype="bfloat16", adapter=adapter)
    validate_scoring_identity(scoring, policy="fp16_source_bf16_training", **kwargs)
    with pytest.raises(ValueError, match="dtype"):
        validate_scoring_identity(scoring, policy="same_dtype", **kwargs)
    with pytest.raises(ValueError, match="adapter"):
        validate_scoring_identity({**scoring, "adapter_fingerprint": "wrong"}, policy="fp16_source_bf16_training", **kwargs)
    for name, p in model.named_parameters():
        if ".lora_B.default." in name:
            with torch.no_grad(): p.fill_(.1)
            break
    with pytest.raises(ValueError, match="zero increment"):
        adapter_compatibility(model, start_mode="fresh", adapter_hash="bad")
    warm = adapter_compatibility(model, start_mode="warm_start", adapter_hash="warm")
    with pytest.raises(ValueError, match="adapter"):
        validate_scoring_identity(scoring, policy="fp16_source_bf16_training", **{**kwargs, "adapter": warm})


def test_binding_tamper_and_unknown_schema_refused(tmp_path):
    config = configuration(tmp_path); make_source(tmp_path, config); select(config, tmp_path)
    _, ref, dest, groups, proof = import_cache(tmp_path, config)
    manifest = read_json(dest / "grpo/reference.json")
    for field in ("role_binding", "probability_binding"):
        bad = copy.deepcopy(ref); bad[field] = {}
        with pytest.raises(ValueError): validate_probability_binding(dest, manifest, bad)
    with pytest.raises(ValueError): validate_probability_binding(dest, {**manifest, "schema_version": 99}, ref)
    (dest / "grpo/source-provenance.json").write_text("{}")
    with pytest.raises(ValueError, match="provenance"):
        load_reference(dest, groups, proof, ref)


def test_external_service_resume_uses_local_cache_and_matches_forward_oracle(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    raw, proof = prepare(config); groups = list(tensor_groups(raw))
    model, targets = model_and_targets(); baseline = tmp_path / "oracle"; baseline.mkdir()
    export_adapter(model, baseline / "grpo/reference/adapter", adapter_contract(BASE, targets, config.lora))
    vectors, _ = build_reference(model, groups, baseline, proof, BASE, hash_json(adapter_fingerprint(model)), 0)
    source = make_source(tmp_path, config, vectors=vectors); select(config, tmp_path)
    def run(name, *, parent=None, stop=None):
        cfg = config.model_copy(deep=True); cfg.run.attempt_id = name
        cfg.validation.stop_after_update = stop
        cfg.run.start.mode = "resume" if parent else "fresh"
        cfg.run.start.checkpoint = str(parent) if parent else None
        origin = _source(cfg); model, targets = model_and_targets(adapter=origin["adapter"] if origin else None)
        dest = tmp_path / name; dest.mkdir()
        result = _train_grpo(cfg, ROOT, dest, BASE, origin, {"attempt_id": name}, model, targets, optimizer_settings=cfg.training)
        return cfg, dest, model, result
    def forbidden(*args, **kwargs): raise AssertionError("reference model forward unexpectedly called")
    monkeypatch.setattr("RL.grpo.reference.build_reference", forbidden)
    _, full, full_model, _ = run("continuous")
    _, paused, _, _ = run("paused", stop=1)
    shutil.rmtree(source)
    monkeypatch.setattr("RL.dataset.probability.open_probability_stream", forbidden)
    cfg, resumed, resumed_model, result = run("resumed", parent=paused / "checkpoints/step-000001")
    assert result["status"] == "completed"
    assert adapter_fingerprint(resumed_model) == adapter_fingerprint(full_model)
    assert (read_json(paused / "grpo-contract.json")["grpo"]["reference_artifact"] ==
            read_json(resumed / "grpo-contract.json")["grpo"]["reference_artifact"])
    contract = read_json(resumed / "grpo-contract.json"); validate_reference_files(resumed, contract)
    before = read_json(paused / "checkpoints/step-000001/state.json")
    changed = cfg.model_copy(deep=True); changed.grpo.reference_cache.dtype_compatibility = "same_dtype"
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_resume(changed, proof, before)


def test_dataset_stream_and_artifact_import_identical_vectors(tmp_path):
    config = configuration(tmp_path)
    make_source(tmp_path, config); select(config, tmp_path)
    offline, _, _, _, _ = import_cache(tmp_path / "offline-import", config)
    online = make_source(tmp_path, config, online=True, name="online")
    raw, proof = prepare(config); groups = list(tensor_groups(raw))
    select(config, tmp_path, online=True)
    config.algorithm.dataset_id = "online"; config.algorithm.dataset_root = str(tmp_path)
    proof["dataset_receipt_sha256"] = sha256(online / "receipt.json")
    model, targets = model_and_targets(); dest = tmp_path / "online-import"; dest.mkdir()
    export_adapter(model, dest / "grpo/reference/adapter", adapter_contract(BASE, targets, config.lora))
    values, ref = import_reference(model, groups, raw, dest, proof, BASE, hash_json(adapter_fingerprint(model)), config)
    for gid in values:
        for bid in values[gid]:
            torch.testing.assert_close(values[gid][bid], offline[gid][bid], atol=0, rtol=0)
    assert ref["probability_binding"]["selection"]["source"] == "dataset_probability"
    validate_reference_files(dest, {"grpo": ref, "input": proof,
        "adapter": adapter_contract(BASE, targets, config.lora), "schedule": schedule_contract(config)})


def test_interrupted_import_is_not_a_cache_commit(tmp_path, monkeypatch):
    config = configuration(tmp_path); make_source(tmp_path, config); select(config, tmp_path)
    original = torch.save
    def interrupted(*args, **kwargs): raise OSError("injected write interruption")
    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(OSError, match="interruption"):
        import_cache(tmp_path, config)
    assert not (tmp_path / "attempt/grpo/reference.json").exists()
    assert not (tmp_path / "attempt/grpo/reference-logprobs.pt").exists()
    assert not list((tmp_path / "attempt/grpo").glob(".reference-*.pt"))


def test_source_drift_and_missing_stream_are_explicit(tmp_path, monkeypatch):
    config = configuration(tmp_path); source = make_source(tmp_path, config); select(config, tmp_path)
    from RL.dataset.probability import ProbabilityStream
    original = ProbabilityStream.values
    def drifting(self, row, **kwargs):
        array = original(self, row, **kwargs)
        with (source / "receipt.json").open("a") as stream: stream.write(" ")
        return array
    monkeypatch.setattr(ProbabilityStream, "values", drifting)
    with pytest.raises(ValueError, match="receipt changed"):
        import_cache(tmp_path, config)
    assert not (tmp_path / "attempt/grpo/reference.json").exists()


def test_explicit_initial_current_diagnostic_does_not_calibrate_anchor(tmp_path):
    from RL.grpo.reference import diagnose_initial_current
    config = configuration(tmp_path); make_source(tmp_path, config); select(config, tmp_path)
    values, _, _, groups, _ = import_cache(tmp_path, config)
    before = copy.deepcopy(values); model, _ = model_and_targets()
    diagnostic = diagnose_initial_current(model, groups, values, branch_count=2, block_size=0)
    assert len(diagnostic["branches"]) == 2 and not diagnostic["anchor_modified"]
    assert diagnostic["branches"][0]["ratio_quantiles"] != [1.] * 5
    for gid in before:
        for bid in before[gid]: torch.testing.assert_close(before[gid][bid], values[gid][bid], atol=0, rtol=0)


def test_external_dry_run_checks_source_without_importing_torch(tmp_path):
    config = configuration(tmp_path); make_source(tmp_path, config); select(config, tmp_path)
    filename = tmp_path / "dry-external.yaml"; filename.write_text(yaml.safe_dump(config.model_dump()))
    script = '''
import importlib.abc, sys
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname == "torch" or fullname.startswith("torch."):
   raise AssertionError("external dry-run imported torch")
sys.meta_path.insert(0, NoTorch())
import RL.service as service
service.memory_plan=lambda c: {}
service.base_identity=lambda *args: {"fingerprint":"a"*64}
result=service.run({"project_root":sys.argv[1]},sys.argv[2])
assert result["status"]=="validated" and "torch" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", script, str(ROOT), str(filename)],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    (tmp_path / "RL/replays/raw/receipt.json").unlink()
    failed = subprocess.run([sys.executable, "-c", script, str(ROOT), str(filename)],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert failed.returncode != 0 and "receipt.json missing" in failed.stderr


def test_warm_start_accepts_only_the_actual_scoring_adapter(tmp_path):
    config = configuration(tmp_path); model, targets = model_and_targets()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".lora_B.default." in name: parameter.fill_(.01)
    fingerprint = hash_json(adapter_fingerprint(model))
    def adapter_identity(manifest, records, path):
        manifest["scoring_identity"]["adapter_fingerprint"] = fingerprint
        for row in records: row["scoring_identity_sha256"] = hash_json(manifest["scoring_identity"])
    make_source(tmp_path, config, mutate=adapter_identity); select(config, tmp_path)
    config.run.start.mode = "warm_start"
    raw, proof = prepare(config); groups = list(tensor_groups(raw))
    dest = tmp_path / "warm-import"; dest.mkdir()
    export_adapter(model, dest / "grpo/reference/adapter", adapter_contract(BASE, targets, config.lora))
    _, reference = import_reference(model, groups, raw, dest, proof, BASE, fingerprint, config)
    assert reference["probability_binding"]["training_initial_adapter"] == {"kind": "adapter", "fingerprint": fingerprint}
    validate_reference_files(dest, {"grpo": reference, "input": proof,
        "adapter": adapter_contract(BASE, targets, config.lora), "schedule": schedule_contract(config)})


def test_explicit_missing_dataset_stream_refused(tmp_path):
    config = configuration(tmp_path); path = make_source(tmp_path, config, online=True, name="online")
    raw, proof = prepare(config)
    proof["dataset_receipt_sha256"] = sha256(path / "receipt.json")
    config.algorithm.dataset_id = "online"; config.algorithm.dataset_root = str(tmp_path)
    select(config, tmp_path, online=True); config.grpo.reference_cache.stream = "missing"
    from RL.dataset.probability import open_probability_stream
    with pytest.raises(ValueError, match="explicitly selected"):
        open_probability_stream(config.grpo.reference_cache, config.algorithm, proof)


def test_native_replay_writer_and_real_online_builder_share_training_reader(tmp_path, monkeypatch):
    """Exercise both other agents' real producer schemas, not a relabeled fixture."""
    import test_rl_dataset as dataset_fixture
    from runtime.serving.raw_logprobs import RawLogprobWriter
    from RL.dataset.loader import iter_groups
    from RL.dataset.replay.artifacts import ReplayWriter
    from RL.dataset.replay.validation import raw_contract, raw_policy
    from RL.dataset.probability import open_probability_stream
    original_write = dataset_fixture.write_json
    writers = {}
    def native_write(path, value):
        if Path(path).parent.name == "requests" and "output_token_ids" in value:
            root = Path(path).parent.parent
            source = read_json(tmp_path / "runs" / value["run_id"] / "resolved-config.json")
            # Construct an explicit FP16 scoring fixture before its dataset receipt.
            source["serving"]["dtype"] = "float16"
            original_write(tmp_path / "runs" / value["run_id"] / "resolved-config.json", source)
            if root not in writers:
                writers[root] = RawLogprobWriter(tmp_path, root, {}, source_identity={
                    "model_fingerprint": source["model"]["fingerprint"],
                    "checkpoint_inventory_sha256": source["model"]["fingerprint"],
                    "serving_dtype": source["serving"]["dtype"], "adapter": None})
            # Attach the actual collector output before the native ledger is sealed,
            # preserving the production order of descriptor -> ledger -> branch hash.
            value["raw_t1_probability"] = writers[root].write(value, [-.1, -.2, -.3])
        return original_write(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(dataset_fixture, "write_json", native_write)
        try:
            source_fixture = dataset_fixture.token_source(tmp_path)
        finally:
            for writer in writers.values(): writer.close()
    ds = dataset_fixture.build(source_fixture); fixture = source_fixture[0]
    native = list(iter_groups(ds))
    config = configuration(tmp_path)
    config.algorithm.dataset_root = str(ds.parents[2]); config.algorithm.dataset_id = ds.name
    config.training.max_updates = 1
    config.grpo.top_k = 1
    raw, proof = prepare(config); groups = list(tensor_groups(raw))
    base = {"fingerprint": proof["dataset_model"]["fingerprint"]}
    select(config, tmp_path, online=True)
    online_reader = open_probability_stream(config.grpo.reference_cache, config.algorithm, proof)
    dtype = online_reader.scoring_identity["collector_dtype"]
    config.grpo.reference_cache.dtype_compatibility = "same_dtype" if dtype == "bfloat16" else "fp16_source_bf16_training"
    model, targets = model_and_targets()
    def materialize(name):
        dest = tmp_path / name; dest.mkdir()
        export_adapter(model, dest / "grpo/reference/adapter", adapter_contract(base, targets, config.lora))
        values, ref = import_reference(model, groups, raw, dest, proof, base,
            hash_json(adapter_fingerprint(model)), config)
        validate_reference_files(dest, {"grpo": ref, "input": proof,
            "adapter": adapter_contract(base, targets, config.lora), "schedule": schedule_contract(config)})
        return values, ref
    online_values, _ = materialize("actual-online")
    source = read_json(tmp_path / "runs" / fixture["run"] / "resolved-config.json")
    scoring = {"model": {k:v for k,v in source["model"].items() if k != "verify_identity"}, "dtype": "float16"}
    models = {hash_json(scoring): scoring}; contract = raw_contract(); policy = raw_policy()
    replay_root = tmp_path / "native-replay"; replay = directory(replay_root, "rl-replay", "native"); replay.mkdir(parents=True)
    identity = {"schema_version": 2, "replay_id": "native", "dataset_receipt_sha256": proof["dataset_receipt_sha256"],
        "raw_contract_sha256": hash_json(contract), "selection_sha256": "explicit-test-selection", "models": models}
    writer = ReplayWriter(replay, ds, identity, 1024)
    for group in native:
        for row in group["members"]:
            writer.put(row, {"token_ids": row["completion_token_ids"]}, online_reader.values(row),
                policy_sha256=policy["policy_sha256"], model_sha256=hash_json(scoring))
    writer.finish(contract, {policy["policy_sha256"]: policy}, models,
        {"logprobs_mode": "raw_logprobs", "enable_trace_replay": True,
         "model_inventory_sha256": base["fingerprint"], "replay_implementation": {"fixture": "explicit-test"}})
    seal(replay, "rl-replay", "native")
    select(config, replay_root, name="native")
    offline_values, ref = materialize("actual-offline")
    for gid in online_values:
        for bid in online_values[gid]:
            torch.testing.assert_close(online_values[gid][bid], offline_values[gid][bid], atol=0, rtol=0)
    assert ref["probability_binding"]["scoring_identity"]["collector_dtype"] == "float16"

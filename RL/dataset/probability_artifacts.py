"""Receipt-bound coverage planning and one self-contained raw probability writer."""
from pathlib import Path
import hashlib
import json

import numpy as np

from core.io import hash_json, owned_path, read_json, sha256, write_json
from .artifacts import directory, publication, require_space, seal, verify_local
from .build import iter_rows, json_line
from .config import load_probability
from .loader import iter_groups
from .materialize import ShardWriter, read_array
from .probability import BINDINGS, ProbabilityStream
from .probability_identity import RAW_CONTRACT, canonical_scoring, split_identity

VERSION = "canonical_probability_v3.1"
SOURCE_KINDS = {"dataset_stream": "rl-dataset", "replay": "rl-replay", "probability": "rl-probability"}


def implementation_identity():
    return {name: sha256(Path(__file__).parent / name) for name in
            ("probability_artifacts.py", "probability_identity.py", "probability.py", "config.py", "materialize.py")}


def _dataset_binding(dataset):
    verify_local(dataset, "rl-dataset")
    return {"id": dataset.name, "receipt_sha256": sha256(dataset / "receipt.json"),
            "branches_sha256": sha256(dataset / "branches.jsonl"),
            "groups_sha256": sha256(dataset / "groups.jsonl")}


def _targets(dataset, config):
    wanted = set(config.group_ids)
    found = set()
    targets = []
    size = 0
    for group in iter_groups(dataset, with_tokens=False):
        gid = group["group"]["group_id"]
        if wanted and gid not in wanted:
            continue
        found.add(gid)
        for row in group["members"]:
            target = {k: row[k] for k in (*BINDINGS, "mask_spans")}
            # Failure is a source fact; absence of a value is never corruption.
            target["recorded_failed"] = _recorded_failed(row)
            size += len(json.dumps(target).encode()) * 4
            if size > config.output.memory_budget_gib * 1024**3 / 4:
                raise ValueError("Target inventory exceeds probability metadata memory budget")
            targets.append(target)
    if wanted - found:
        raise ValueError("Target groups missing or ineligible: " + str(sorted(wanted - found)[:10]))
    if not targets:
        raise ValueError("No complete native token groups are eligible")
    if len({_key(t) for t in targets}) != len(targets):
        raise ValueError("Duplicate target branch identity")
    return targets


def _recorded_failed(row):
    for source in (row.get("probability_streams") or {}).values():
        if source.get("descriptor", {}).get("status") == "failed":
            return True
    evidence = row.get("native_evidence")
    if evidence and not row.get("raw_probability_available", False):
        ledger = owned_path(evidence["root"], evidence["ledger_ref"])
        if evidence.get("ledger_sha256") and sha256(ledger) != evidence["ledger_sha256"]:
            raise ValueError("Native failure ledger checksum changed")
        descriptor = read_json(ledger).get("raw_t1_probability")
        return bool(descriptor and descriptor.get("status") == "failed")
    return False


def _key(row):
    return row["group_id"], row["branch_sample_id"]


def _open_source(spec, dataset, binding, budget):
    kind = SOURCE_KINDS[spec.kind]
    path = directory(spec.root, kind, spec.id)
    receipt = verify_local(path, kind)
    if sum(p["bytes"] for n, p in receipt["files"].items() if n.endswith(".jsonl")) * 4 > budget:
        raise ValueError("Probability source index exceeds metadata memory budget")
    if spec.kind == "dataset_stream":
        if path != dataset or sha256(path / "receipt.json") != binding["receipt_sha256"]:
            raise ValueError("Dataset stream must bind the exact target dataset receipt")
        descriptor = read_json(path / "dataset.json").get("probability_streams", {}).get(spec.stream)
        if descriptor is None:
            # All-absent datasets may have no stream at all, but a declaration or
            # available branch without an index is inconsistent, not missing.
            if any(r.get("raw_probability_available") or r.get("probability_streams")
                   for r in iter_rows(path / "branches.jsonl")):
                raise ValueError("Dataset declares probability values without a stream")
            return None, {"spec": spec.model_dump(), "receipt_sha256": sha256(path / "receipt.json"), "adapter": VERSION}
        expected_binding = hash_json({k: binding[k] for k in ("branches_sha256", "groups_sha256")})
        if descriptor["dataset_binding"] != expected_binding:
            raise ValueError("Online dataset stream content binding mismatch")
        manifest_name, index_name = descriptor["manifest"], descriptor["index"]
        dataset_sha = descriptor["dataset_binding"]
    else:
        manifest_name = "replay.json" if spec.kind == "replay" else "probability.json"
        index_name = "branches.jsonl"
        dataset_sha = binding["receipt_sha256"]
    source = ProbabilityStream(path, kind=kind, manifest_name=manifest_name, index_name=index_name,
                               dataset_receipt_sha256=dataset_sha, stream=spec.stream)
    if spec.kind == "dataset_stream":
        expected = {_key(r) for r in iter_rows(dataset / "branches.jsonl") if r.get("raw_probability_available")}
        if set(source.records) != expected:
            raise ValueError("Online stream coverage disagrees with recorded probability availability")
    elif source.provenance.get("adapter_schema") == "native_replay_v2":
        from .replay.artifacts import check_replay
        check_replay(path, dataset_path=dataset, with_values=True)
    manifest = source.provenance["manifest"]
    if spec.kind != "dataset_stream" and manifest.get("replay_id" if spec.kind == "replay" else "probability_id") != spec.id:
        raise ValueError("Source manifest ID mismatch")
    source.validate_dataset_rows(r for g in iter_groups(dataset, with_tokens=False, data_ready_only=False) for r in g["members"])
    return source, {"spec": spec.model_dump(), "receipt_sha256": source.receipt_sha256,
                    "adapter": VERSION, "manifest_sha256": source.provenance["manifest_sha256"]}


def _policy(config, sources):
    function = None
    dtype = None
    cohorts = {}
    identities = []
    expected = split_identity(config.expected_scoring_identity) if config.expected_scoring_identity is not None else None
    for source in sources:
        if source is None:
            continue
        if source.provenance["manifest"].get("schema_version") == 3:
            manifest = source.provenance["manifest"]
            identities.extend((manifest["scoring_function_identity"], c) for c in manifest["execution_cohorts"].values())
        else:
            identities.append(split_identity(source.scoring_identity))
    if not identities and expected:
        identities.append(expected)
    if expected:
        function, expected_cohort = expected
        dtype = expected_cohort["collector_dtype"]
    for candidate, cohort in identities:
        if function is not None and candidate != function:
            raise ValueError("Probability scoring function mismatch")
        if dtype is not None and cohort["collector_dtype"] != dtype:
            raise ValueError("Probability sources cannot mix collector compute dtype")
        function, dtype = candidate, cohort["collector_dtype"]
        cohorts[hash_json(cohort)] = cohort
    if function is None:
        raise ValueError("No source scoring identity; declare expected_scoring_identity for all-missing planning")
    allowed = set(config.cohort_policy.allowed_execution_cohorts)
    if allowed and not set(cohorts) <= allowed:
        raise ValueError("Execution cohort is not explicitly allowed: " + str(sorted(set(cohorts) - allowed)))
    if len(cohorts) > 1 and not allowed:
        raise ValueError("Multiple execution cohorts require explicit allowed_execution_cohorts: " + str(sorted(cohorts)))
    return function, cohorts


def _read_values(source, target):
    record = source.records[_key(target)]
    if any(record[k] != target[k] for k in (*BINDINGS, "mask_spans")):
        raise ValueError("Probability branch token/request/mask binding mismatch")
    values = read_array(source.path, record["values_ref"])
    if (values.dtype != np.dtype("<f4") or len(values) != target["completion_count"]
            or np.isnan(values).any() or np.isposinf(values).any() or (values > 0).any()):
        raise ValueError("Invalid raw float32 completion logprobs")
    return values


def _coverage(targets, sources, config, *, frozen=None):
    rows = []
    counts = {k: 0 for k in ("valid", "absent", "recorded_failed", "nonfinite")}
    for index, target in enumerate(targets):
        candidates = []
        for number, source in enumerate(sources):
            if source is not None and _key(target) in source.records:
                values = _read_values(source, target)
                record = source.records[_key(target)]
                candidates.append({"source": number, "values_sha256": record["values_ref"]["sha256"],
                                   "negative_infinity_count": int(np.isneginf(values).sum())})
        if frozen is not None:
            previous = frozen[index]
            existing = previous["candidates"]
            for candidate in existing:
                if candidate not in candidates:
                    raise ValueError("Frozen existing probability values changed")
            extras = [c for c in candidates if c not in existing]
            if existing and extras:
                raise ValueError("Backfill emitted an extra already-covered branch")
        if candidates:
            if len(candidates) > 1 and (not config.deduplicate_identical or len({c["values_sha256"] for c in candidates}) != 1):
                raise ValueError("Conflicting or undeclared duplicate probability sources")
            status = "nonfinite" if candidates[0]["negative_infinity_count"] else "valid"
        else:
            status = "recorded_failed" if target["recorded_failed"] else "absent"
            if status not in config.allowed_missing_reasons:
                raise ValueError("Missing probability reason requires explicit permission in config: " + status)
        counts[status] += 1
        rows.append({**target, "status": status, "candidates": candidates})
    return rows, counts


def _selection(stage, targets):
    with (stage / "targets.jsonl").open("w", encoding="utf-8") as out:
        for target in targets:
            json_line(out, target)
    selection = {"order": "dataset_group_branch_v1", "target_count": len(targets),
                 "inventory": "targets.jsonl", "inventory_sha256": sha256(stage / "targets.jsonl")}
    write_json(stage / "selection.json", selection)
    return hash_json(selection)


def _write_coverage(stage, rows, counts):
    missing = [r for r in rows if r["status"] in {"absent", "recorded_failed"}]
    with (stage / "missing.jsonl").open("w", encoding="utf-8") as out:
        for row in missing:
            json_line(out, row)
    coverage = {**counts, "target_count": len(rows), "covered_count": counts["valid"] + counts["nonfinite"],
                "missing_count": len(missing), "missing_sha256": sha256(stage / "missing.jsonl"),
                "coverage_status": "partial" if missing else "complete",
                "finite_coverage_complete": not missing and not counts["nonfinite"]}
    write_json(stage / "coverage.json", coverage)
    return coverage


def _snapshot(stage, config, source, descriptors, sources):
    write_json(stage / "resolved-config.json", config.model_dump())
    (stage / "source-config.yaml").write_bytes(source.read_bytes())
    write_json(stage / "sources.json", descriptors)
    for index, (descriptor, stream) in enumerate(zip(descriptors, sources)):
        if descriptor.get("planned"):
            continue
        dest = stage / "provenance" / str(index)
        dest.mkdir(parents=True, exist_ok=True)
        source_path = directory(descriptor["spec"]["root"], SOURCE_KINDS[descriptor["spec"]["kind"]], descriptor["spec"]["id"])
        (dest / "receipt.json").write_bytes((source_path / "receipt.json").read_bytes())
        if stream is not None:
            write_json(dest / "source.json", stream.provenance)


def _unchanged(dataset, binding, sources, descriptors):
    if _dataset_binding(dataset) != binding:
        raise ValueError("Dataset changed during probability operation")
    for source, descriptor in zip(sources, descriptors):
        if descriptor.get("planned"):
            continue
        path = directory(descriptor["spec"]["root"], SOURCE_KINDS[descriptor["spec"]["kind"]], descriptor["spec"]["id"])
        verify_local(path, SOURCE_KINDS[descriptor["spec"]["kind"]])
        if sha256(path / "receipt.json") != descriptor["receipt_sha256"]:
            raise ValueError("Probability source receipt changed")
        if source is not None:
            source.verify_unchanged()


def _inputs(config, *, include_backfill):
    dataset = directory(config.dataset.root, "rl-dataset", config.dataset.id)
    binding = _dataset_binding(dataset)
    targets = _targets(dataset, config)
    sources, descriptors = [], []
    budget = int(config.output.memory_budget_gib * 1024**3 / 4)
    for spec in config.sources:
        if spec.role == "backfill" and not include_backfill:
            sources.append(None)
            descriptors.append({"spec": spec.model_dump(), "planned": True})
            continue
        stream, descriptor = _open_source(spec, dataset, binding, budget)
        if stream is not None:
            budget -= (stream.path / stream.provenance["index_name"]).stat().st_size * 4
            if budget < 0:
                raise ValueError("Combined source indexes exceed probability memory budget")
        sources.append(stream)
        descriptors.append(descriptor)
    return dataset, binding, targets, sources, descriptors


def probability_plan(cfg, file, *, recover=False):
    config, source = load_probability(file, cfg["project_root"])
    if config.coverage_plan is not None:
        raise ValueError("Create a plan without coverage_plan; bind its returned hash for build/replay")
    dataset, binding, targets, sources, descriptors = _inputs(config, include_backfill=False)
    function, cohorts = _policy(config, sources)
    rows, counts = _coverage(targets, sources, config)
    signature = {"version": VERSION, "implementation": implementation_identity(), "config": config.model_dump(),
                 "source_config_sha256": sha256(source), "dataset": binding, "sources": descriptors}
    with publication(config.plan.root, "rl-probability-plan", config.plan.id, recover=recover, signature=signature) as (stage, destination):
        if any(p.is_symlink() for p in stage.rglob("*")):
            raise ValueError("Staging symlinks are forbidden")
        selection_hash = _selection(stage, targets)
        coverage = _write_coverage(stage, rows, counts)
        with (stage / "coverage-branches.jsonl").open("w", encoding="utf-8") as out:
            for row in rows:
                json_line(out, row)
        # Planned backfills have no receipt yet and no fabricated provenance.
        _snapshot(stage, config, source, descriptors, sources)
        write_json(stage / "plan.json", {"schema_version": 1, "plan_id": config.plan.id, "version": VERSION,
                   "dataset": binding, "selection_sha256": selection_hash, "sources": descriptors,
                   "config": config.model_dump(), "scoring_function_identity": function,
                   "execution_cohorts": cohorts, "coverage": coverage,
                   "coverage_branches_sha256": sha256(stage / "coverage-branches.jsonl"),
                   "implementation": implementation_identity()})
        _unchanged(dataset, binding, sources, descriptors)
        seal(stage, "rl-probability-plan", config.plan.id)
    return {"status": "planned", "plan_id": config.plan.id, "path": str(destination),
            "plan_sha256": sha256(destination / "plan.json"), "receipt_sha256": sha256(destination / "receipt.json"),
            "scoring_function_sha256": hash_json(function), "execution_cohort_ids": sorted(cohorts),
            **coverage, "model_load": False}


def read_plan(path, *, expected_hash=None, dataset=None, check_sources=False):
    path = Path(path)
    verify_local(path, "rl-probability-plan")
    plan = read_json(path / "plan.json")
    if (plan.get("schema_version") != 1 or plan.get("plan_id") != path.name
            or read_json(path / "receipt.json").get("artifact_id") != plan.get("plan_id")
            or plan.get("version") != VERSION
            or expected_hash is not None and sha256(path / "plan.json") != expected_hash
            or hash_json(read_json(path / "selection.json")) != plan["selection_sha256"]
            or sha256(path / "coverage-branches.jsonl") != plan["coverage_branches_sha256"]):
        raise ValueError("Coverage plan identity/hash mismatch")
    selection = read_json(path / "selection.json")
    targets = list(iter_rows(path / "targets.jsonl"))
    coverage_rows = list(iter_rows(path / "coverage-branches.jsonl"))
    missing = list(iter_rows(path / "missing.jsonl"))
    if (not targets or selection["inventory_sha256"] != sha256(path / "targets.jsonl")
            or selection["target_count"] != len(targets) or len({_key(r) for r in targets}) != len(targets)
            or [{k: r[k] for k in targets[0]} for r in coverage_rows] != targets
            or missing != [r for r in coverage_rows if r["status"] in {"absent", "recorded_failed"}]
            or read_json(path / "coverage.json") != plan["coverage"]
            or sha256(path / "missing.jsonl") != plan["coverage"]["missing_sha256"]):
        raise ValueError("Coverage plan frozen selection mismatch")
    if dataset is not None and _dataset_binding(Path(dataset)) != plan["dataset"]:
        raise ValueError("Coverage plan dataset receipt/inventory changed")
    if check_sources:
        for descriptor in plan["sources"]:
            if descriptor.get("planned"):
                continue
            spec = descriptor["spec"]
            source = directory(spec["root"], SOURCE_KINDS[spec["kind"]], spec["id"])
            verify_local(source, SOURCE_KINDS[spec["kind"]])
            if sha256(source / "receipt.json") != descriptor["receipt_sha256"]:
                raise ValueError("Coverage plan source receipt changed")
    return {"plan": plan, "targets": targets, "rows": coverage_rows, "missing": missing,
            "summary": {"status": "verified", "plan_sha256": sha256(path / "plan.json"),
                        "receipt_sha256": sha256(path / "receipt.json"), **plan["coverage"]}}


def replay_selection(binding, dataset):
    path = directory(binding.root, "rl-probability-plan", binding.id)
    result = read_plan(path, expected_hash=binding.sha256, dataset=dataset, check_sources=True)
    if result["plan"]["implementation"] != implementation_identity():
        raise ValueError("Coverage plan implementation changed; create a new plan ID")
    return result


def probability_build(cfg, file, *, recover=False):
    config, source = load_probability(file, cfg["project_root"])
    has_backfill = any(s.role == "backfill" for s in config.sources)
    frozen = None
    if config.coverage_plan:
        dataset = directory(config.dataset.root, "rl-dataset", config.dataset.id)
        frozen = replay_selection(config.coverage_plan, dataset)
    dataset, binding, targets, sources, descriptors = _inputs(config, include_backfill=has_backfill and (frozen is None or bool(frozen["missing"])))
    if config.coverage_plan:
        frozen = replay_selection(config.coverage_plan, dataset)
        original_config = {**config.model_dump(), "coverage_plan": None}
        if frozen["plan"]["config"] != original_config or frozen["targets"] != targets:
            raise ValueError("Build configuration/targets differ from frozen coverage plan")
        if frozen["plan"]["implementation"] != implementation_identity():
            raise ValueError("Coverage plan implementation changed; create a new plan ID")
        for expected, actual in zip(frozen["plan"]["sources"], descriptors):
            if not expected.get("planned") and expected != actual:
                raise ValueError("Build source receipt differs from frozen plan")
        missing_keys = {_key(r) for r in frozen["missing"]}
        for spec, stream in zip(config.sources, sources):
            if spec.role != "backfill":
                continue
            if not missing_keys:
                if stream is not None:
                    raise ValueError("No-missing plan must not consume a backfill replay")
                continue
            manifest = stream.provenance["manifest"]
            if (manifest.get("coverage_plan_sha256") != config.coverage_plan.sha256
                    or manifest.get("coverage_plan_id") != config.coverage_plan.id
                    or set(stream.records) != missing_keys):
                raise ValueError("Backfill replay plan binding/missing branch coverage mismatch")
    elif has_backfill:
        raise ValueError("Backfill build requires a frozen coverage plan")
    function, cohorts = _policy(config, sources)
    if frozen and function != frozen["plan"]["scoring_function_identity"]:
        raise ValueError("Backfill scoring function differs from coverage plan")
    rows, counts = _coverage(targets, sources, config, frozen=frozen["rows"] if frozen else None)
    if (counts["absent"] or counts["recorded_failed"]) and not config.allow_partial:
        raise ValueError("Missing target probability branches; explicit allow_partial required")
    require_space(config.output.root, sum(r["completion_count"] * 4 for r in targets))
    max_bytes = min(config.output.max_shard_bytes, int(config.output.memory_budget_gib * 1024**3 / 8))
    signature = {"version": VERSION, "implementation": implementation_identity(), "config": config.model_dump(),
                 "source_config_sha256": sha256(source), "dataset": binding, "sources": descriptors}
    with publication(config.output.root, "rl-probability", config.output.id, recover=recover, signature=signature) as (stage, destination):
        if any(p.is_symlink() for p in stage.rglob("*")):
            raise ValueError("Staging symlinks are forbidden")
        selection_hash = _selection(stage, targets)
        coverage = _write_coverage(stage, rows, counts)
        _snapshot(stage, config, source, descriptors, sources)
        writer = ShardWriter(stage, "<f4", max_bytes)
        try:
            with (stage / "branches.jsonl").open("w", encoding="utf-8") as out:
                for row in rows:
                    if not row["candidates"]:
                        continue
                    candidate = row["candidates"][0]
                    number = candidate["source"]
                    stream = sources[number]
                    original = stream.records[_key(row)]
                    values = _read_values(stream, row)
                    ref = writer.put(values)
                    if ref["sha256"] != candidate["values_sha256"]:
                        raise ValueError("Repackaging changed original float32 bytes")
                    if stream.provenance["manifest"].get("schema_version") == 3:
                        cohort_id, origin = original["execution_cohort_id"], original["origin"]
                    else:
                        _, cohort = split_identity(stream.scoring_identity)
                        cohort_id = hash_json(cohort)
                        origin = "online_rollout" if config.sources[number].kind == "dataset_stream" else "offline_native_trace_scoring"
                    references = [{"source_index": c["source"], "source_id": descriptors[c["source"]]["spec"]["id"],
                                   "receipt_sha256": descriptors[c["source"]]["receipt_sha256"],
                                   "locator": sources[c["source"]].records[_key(row)],
                                   "values_sha256": c["values_sha256"]} for c in row["candidates"]]
                    record = {k: row[k] for k in (*BINDINGS, "mask_spans")}
                    record.update(values_ref=ref, scoring_function_sha256=hash_json(function),
                                  execution_cohort_id=cohort_id, origin=origin, source_ref=references,
                                  finite_count=int(np.isfinite(values).sum()),
                                  negative_infinity_count=int(np.isneginf(values).sum()))
                    if origin == "offline_native_trace_scoring":
                        record.update(status="replay_estimate", historical_behavior_verified=False)
                    json_line(out, record)
        finally:
            writer.close()
        manifest = {"schema_version": 3, "probability_id": config.output.id, "stream": "raw_t1_output_tokens",
                    "probability_semantics": "raw_t1_output_tokens_v1", "raw_contract": RAW_CONTRACT,
                    "values_dtype": "<f4", "dataset": binding, "selection_sha256": selection_hash,
                    "scoring_function_identity": function, "execution_cohorts": cohorts,
                    "coverage_status": coverage["coverage_status"],
                    "finite_coverage_complete": coverage["finite_coverage_complete"], "index": "branches.jsonl",
                    "coverage_plan_sha256": config.coverage_plan.sha256 if config.coverage_plan else None,
                    "historical_behavior_verified": False}
        canonical_scoring(manifest)
        write_json(stage / "probability.json", manifest)
        write_json(stage / "summary.json", coverage)
        _unchanged(dataset, binding, sources, descriptors)
        if frozen:
            replay_selection(config.coverage_plan, dataset)
        seal(stage, "rl-probability", config.output.id)
        check_probability(stage, with_values=True, allow_stage=True)
    return {"status": "committed", "probability_id": config.output.id, "path": str(destination),
            "receipt_sha256": sha256(destination / "receipt.json"), **coverage, "model_load": False}


def check_probability(path, *, with_values=False, sources=False, allow_stage=False, metadata_only=False):
    path = Path(path)
    receipt = verify_local(path, "rl-probability", verify_files=not metadata_only)
    if metadata_only:
        for name, expected in receipt["files"].items():
            if name.endswith((".json", ".jsonl", ".yaml")) and sha256(owned_path(path, name)) != expected["sha256"]:
                raise ValueError("Canonical probability metadata checksum mismatch: " + name)
    manifest = read_json(path / "probability.json")
    canonical_scoring(manifest)
    if (receipt.get("artifact_id") != manifest["probability_id"]
            or not allow_stage and (path.name.startswith(".") or manifest["probability_id"] != path.name)
            or hash_json(read_json(path / "selection.json")) != manifest["selection_sha256"]):
        raise ValueError("Canonical probability identity/selection mismatch")
    selection = read_json(path / "selection.json")
    targets = list(iter_rows(path / "targets.jsonl"))
    if (selection["inventory_sha256"] != sha256(path / "targets.jsonl")
            or len(targets) != selection["target_count"] or len({_key(t) for t in targets}) != len(targets)):
        raise ValueError("Canonical target inventory mismatch")
    records = list(iter_rows(path / "branches.jsonl"))
    indexed = {_key(r): r for r in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicate canonical probability branch")
    expected_order = []
    missing = []
    nonfinite = 0
    descriptors = read_json(path / "sources.json")
    for target in targets:
        record = indexed.get(_key(target))
        if record is None:
            missing.append(_key(target))
            continue
        expected_order.append(_key(target))
        ref = record["values_ref"]
        if (any(record[k] != target[k] for k in (*BINDINGS, "mask_spans"))
                or record["scoring_function_sha256"] != hash_json(manifest["scoring_function_identity"])
                or record["execution_cohort_id"] not in manifest["execution_cohorts"]
                or record["origin"] not in {"online_rollout", "offline_native_trace_scoring"}
                or ref["dtype"] != "<f4" or ref["count"] != target["completion_count"]
                or ref["shard"] not in receipt["files"] or not ref["shard"].startswith("shards/")
                or type(ref["offset"]) is not int or ref["offset"] < 0
                or type(ref["count"]) is not int or ref["count"] < 1
                or (ref["offset"] + ref["count"]) * 4 > receipt["files"][ref["shard"]]["bytes"]
                or any(type(record[k]) is not int or record[k] < 0 for k in ("finite_count", "negative_infinity_count"))
                or record["finite_count"] + record["negative_infinity_count"] != target["completion_count"]
                or not record["source_ref"]):
            raise ValueError("Canonical branch binding/cohort/value contract mismatch")
        for source in record["source_ref"]:
            descriptor = descriptors[source["source_index"]]
            if (source["receipt_sha256"] != descriptor["receipt_sha256"]
                    or source["values_sha256"] != ref["sha256"]
                    or source["source_id"] != descriptor["spec"]["id"]):
                raise ValueError("Canonical source provenance mismatch")
        if record["origin"] == "offline_native_trace_scoring" and (record.get("status") != "replay_estimate" or record.get("historical_behavior_verified") is not False):
            raise ValueError("Offline probability provenance cannot claim historical recording")
        nonfinite += bool(record["negative_infinity_count"])
        if with_values:
            values = read_array(path, ref)
            if (np.isnan(values).any() or np.isposinf(values).any() or (values > 0).any()
                    or int(np.isfinite(values).sum()) != record["finite_count"]
                    or int(np.isneginf(values).sum()) != record["negative_infinity_count"]):
                raise ValueError("Invalid canonical logprob values/counts")
    coverage = read_json(path / "coverage.json")
    if (list(indexed) != expected_order or [_key(r) for r in iter_rows(path / "missing.jsonl")] != missing
            or coverage["target_count"] != len(targets) or coverage["covered_count"] != len(records)
            or coverage["missing_count"] != len(missing) or coverage["nonfinite"] != nonfinite
            or coverage["coverage_status"] != ("partial" if missing else "complete")
            or coverage["finite_coverage_complete"] != (not missing and not nonfinite)
            or manifest["coverage_status"] != coverage["coverage_status"]
            or manifest["finite_coverage_complete"] != coverage["finite_coverage_complete"]
            or coverage["missing_sha256"] != sha256(path / "missing.jsonl")
            or read_json(path / "summary.json") != coverage):
        raise ValueError("Canonical probability coverage/order mismatch")
    # Provenance is local evidence; normal verify never follows external roots.
    for index, descriptor in enumerate(descriptors):
        if descriptor.get("planned"):
            continue
        raw = (path / "provenance" / str(index) / "receipt.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != descriptor["receipt_sha256"]:
            raise ValueError("Canonical local source receipt copy mismatch")
        evidence_path = path / "provenance" / str(index) / "source.json"
        if evidence_path.is_file():
            evidence = read_json(evidence_path)
            original_receipt = json.loads(raw)
            original_manifest = evidence["manifest_bytes"].encode("utf-8")
            if (evidence["receipt_bytes"].encode("utf-8") != raw
                    or hashlib.sha256(original_manifest).hexdigest() != evidence["manifest_sha256"]
                    or json.loads(original_manifest) != evidence["manifest"]
                    or original_receipt["files"][evidence["manifest_name"]]["sha256"] != evidence["manifest_sha256"]
                    or original_receipt["files"][evidence["index_name"]]["sha256"] != evidence["index_sha256"]):
                raise ValueError("Canonical local source manifest/index provenance mismatch")
            for name, content in evidence.get("metadata_bytes", {}).items():
                if hashlib.sha256(content.encode("utf-8")).hexdigest() != original_receipt["files"][name]["sha256"]:
                    raise ValueError("Canonical local source identity metadata checksum mismatch")
    audit = []
    if sources:
        for descriptor in descriptors:
            if descriptor.get("planned"):
                continue
            spec = descriptor["spec"]
            try:
                source_path = directory(spec["root"], SOURCE_KINDS[spec["kind"]], spec["id"])
                verify_local(source_path, SOURCE_KINDS[spec["kind"]])
                if sha256(source_path / "receipt.json") != descriptor["receipt_sha256"]:
                    raise ValueError("Source receipt changed")
                audit.append({"id": spec["id"], "status": "verified"})
            except (OSError, ValueError) as error:
                audit.append({"id": spec["id"], "status": "unavailable_or_changed", "error": str(error)[:300]})
    return {"status": "verified", "check_level": "values" if with_values else "metadata",
            "scoring_function_sha256": hash_json(manifest["scoring_function_identity"]),
            "execution_cohort_ids": sorted(manifest["execution_cohorts"]),
            "source_audit_status": "not_requested" if not sources else "verified" if all(r["status"] == "verified" for r in audit) else "incomplete",
            "source_audit": audit, **coverage}

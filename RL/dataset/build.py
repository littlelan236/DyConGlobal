"""Streaming, all-slot step inventories with exact native evidence references."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import hashlib
import os
from pathlib import Path

from core.action_identity import canonical_action, ActionIdentityUnavailable
from core.io import hash_json, read_json, sha256, write_json, owned_path
from hidden.token_evidence import load_token_session, native_branch
from steering.datasets.selection import select_sources
from steering.datasets.sessions import full_decision
from .artifacts import directory, publication, seal, verify_local, verify_dependencies, require_space
from .config import load
from .schema import MASK_POLICY, group_id, branch_id, spans, validate


def json_line(stream, value):
    stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def iter_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _lengths(counter):
    n = sum(counter.values())
    if not n:
        return {"count": 0, "max": None, "p50": None, "p95": None, "p99": None}
    result = {"count": n, "min": min(counter), "max": max(counter), "sum": sum(k*v for k,v in counter.items())}
    for label, percentile in (("p50", .5), ("p95", .95), ("p99", .99)):
        cumulative = 0
        for length in sorted(counter):
            cumulative += counter[length]
            if cumulative >= max(1, n * percentile):
                result[label] = length
                break
    return result


def _read_native(session, decision, candidate, message_index):
    dependencies = {}
    result = native_branch(session, decision, candidate, message_index, dependencies)
    return result, dependencies


def _restore_stream(path, proof):
    """Validate only committed bytes, then discard an interrupted rollout tail."""
    size = proof["bytes"]; digest = hashlib.sha256()
    with path.open("r+b") as stream:
        remaining = size
        while remaining:
            chunk = stream.read(min(remaining, 1024*1024))
            if not chunk: raise ValueError("Recovery checkpoint stream is truncated")
            digest.update(chunk); remaining -= len(chunk)
        if digest.hexdigest() != proof["sha256"]: raise ValueError("Recovery checkpoint stream checksum differs")
        stream.truncate(size)


def _checkpoint(stage, streams, next_source, counters, cohort, owners, origins, deps):
    files = {}
    for name, stream in streams.items():
        stream.flush(); os.fsync(stream.fileno())
        path = stage/(name+".jsonl")
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    write_json(stage/"checkpoint.json", {"next_source": next_source, "files": files,
               "counters": {k:list(v.items()) for k,v in counters.items()}, "cohort": cohort,
               "request_owners": owners, "origin_groups": [[list(k),v] for k,v in origins.items()], "dependencies": deps})


def build_dataset(cfg, config_file, *, dry_run=False, recover=False):
    dc, source_file = load(config_file, cfg["project_root"])
    audit = dc.audit.enabled
    if dry_run and recover: raise ValueError("Recovery uses the frozen selection; omit dry-run")
    target = directory(dc.storage.dataset_root, "rl-dataset", dc.dataset_id)
    signature = {"config": dc.model_dump(), "config_sha256": sha256(source_file),
                 "implementation_sha256": sha256(__file__)}
    if recover:
        state = read_json(target.with_name("."+dc.dataset_id+".staging")/"build-state.json")
        if state["signature"] != signature: raise ValueError("Recovery build inputs or implementation differ")
        frozen = state["frozen"]
    else:
        frozen = select_sources(cfg, dc.sources.runs, rollout_list=dc.sources.binding_file,
                            rollout_ids=dc.sources.rollout_ids, all_submitted=dc.sources.all_submitted,
                            verify_source_hashes=audit)
    deps = frozen["dependencies"]
    # Explicit submitted selection is not a resolved-only filter.
    for source in frozen["sources"]:
        if "swebench" not in source["benchmark"].lower().replace("-", "").replace("_", ""):
            raise ValueError("Selected source is not an evidenced SWE-bench task")
        if frozen["configs"][source["source_run_id"]]["parallel"]["mode"] != "parallel_consensus":
            raise ValueError("Only explicitly allowed parallel_consensus sources are supported")
    if dry_run:
        return {"status": "dry-run", "dataset_id": dc.dataset_id, "selected_rollouts": len(frozen["sources"]),
                "selection_sha256": hash_json(frozen["selection"]), "check_level": "selection_only",
                "data_ready": False, "grpo_ready": False}
    require_space(dc.storage.dataset_root, 0)
    counts = Counter(); sizes = Counter(); excluded = Counter(); prompt_lengths = Counter(); output_lengths = Counter()
    counts["selected_rollouts"] = len(frozen["sources"])
    common_cohort = None
    request_owners = {}; origin_groups = {}
    with publication(dc.storage.dataset_root, "rl-dataset", dc.dataset_id, signature=signature, recover=recover) as (stage, destination), ExitStack() as stack:
        native_pool = stack.enter_context(ThreadPoolExecutor(max_workers=dc.storage.ledger_read_workers, thread_name_prefix="rl-ledger"))
        next_source = 0
        if recover:
            checkpoint = read_json(stage/"checkpoint.json")
            for name, proof in checkpoint["files"].items(): _restore_stream(stage/(name+".jsonl"), proof)
            next_source = checkpoint["next_source"]
            counts, sizes, excluded, prompt_lengths, output_lengths = [Counter(dict(checkpoint["counters"][name])) for name in ("counts","sizes","excluded","prompts","outputs")]
            common_cohort = checkpoint["cohort"]; request_owners = checkpoint["request_owners"]
            request_owners = {k:tuple(v) for k,v in request_owners.items()}
            origin_groups = {tuple(k):(v[0],tuple(v[1])) for k,v in checkpoint["origin_groups"]}
            deps = checkpoint["dependencies"]
            if audit:
                for name, digest in deps.items():
                    if sha256(name) != digest: raise ValueError("Recovery source dependency changed")
        streams = {name: stack.enter_context((stage / (name + ".jsonl")).open("a" if recover else "w", encoding="utf-8"))
                   for name in ("rollouts", "groups", "branches", "aliases", "exclusions")}
        counters = {"counts":counts,"sizes":sizes,"excluded":excluded,"prompts":prompt_lengths,"outputs":output_lengths}
        if not recover:
            write_json(stage/"build-state.json", {"signature":signature, "frozen":frozen})
            _checkpoint(stage,streams,0,counters,common_cohort,request_owners,origin_groups,deps)
        (stage / "source-config.yaml").write_bytes(source_file.read_bytes())
        write_json(stage / "resolved-config.json", dc.model_dump())
        write_json(stage / "selection.json", frozen["selection"])
        for run, snapshot in frozen["snapshots"].items():
            write_json(stage / "source-evidence" / run / "evidence.json", snapshot)
            write_json(stage / "source-evidence" / run / "resolved-config.json", frozen["configs"][run])
        for source_ordinal, source in enumerate(frozen["sources"]):
            if source_ordinal < next_source: continue
            run = source["source_run_id"]; scfg = frozen["configs"][run]
            trajectory = read_json(source["trajectory_path"])
            json_line(streams["rollouts"], source)
            sessions = {}
            for mi, message in enumerate(trajectory.get("messages", [])):
                embedded = (message.get("extra") or {}).get("dycon_parallel")
                if not isinstance(embedded, dict):
                    continue
                sid, did = embedded["session_id"], embedded["decision_id"]
                if sid not in sessions:
                    sessions[sid] = load_token_session(source, scfg, sid, deps, audit=audit)
                session, missing = sessions[sid]
                base = {k: source.get(k) for k in ("source_run_id", "rollout_id", "trajectory_sha256", "task_key", "benchmark", "revision", "instance_id", "attempt_id", "repetition_id")}
                base.update(session_id=sid, decision_id=did, message_index=mi, agent_step=embedded.get("step_id"), parent_prefix_sha256=embedded.get("parent_prefix_sha256"),
                            terminal_reward=source.get("terminal_reward"),
                            terminal_scoring_exclusion=source.get("terminal_scoring_exclusion"))
                if session:
                    decision_path = owned_path(session["base"], "parallel/decisions/" + did + ".json")
                    if not decision_path.is_file():
                        decision, decision_sha = embedded, None
                        session = None
                        missing = "full_decision_unavailable"
                    elif audit:
                        decision, decision_sha = full_decision(session, embedded)
                    else:
                        decision, decision_sha = read_json(decision_path), None
                else:
                    decision, decision_sha = embedded, None
                n = decision.get("branch_count"); candidates = decision.get("candidates") or []
                if type(n) is not int or n < 1 or sorted(c.get("branch_id") for c in candidates) != list(range(n)):
                    raise ValueError("Complete decision branch slots are required even when evidence is unavailable")
                counts["raw_groups"] += 1; counts["raw_branches"] += n; sizes[n] += 1
                rows = []; prompt = None; prompt_sha = None
                native_futures = ({c["branch_id"]: native_pool.submit(_read_native, session, decision, c, mi)
                                   for c in candidates} if session else {})
                for candidate in sorted(candidates, key=lambda c: c["branch_id"]):
                    bid = candidate["branch_id"]
                    row = {**base, "branch_id": bid, "native_request_id": candidate.get("native_request_id"),
                           "api_attempt_id": candidate.get("api_attempt_id"), "seed": candidate.get("seed"),
                           "selected": bid == decision.get("selected_branch_id"),
                           "retained": candidate.get("retained", bid in (decision.get("consensus") or {}).get("retained_branch_ids", [])),
                           "historical_score": candidate.get("score"), "historical_exclusion_reason": candidate.get("exclusion_reason"),
                           "loop_escape": candidate.get("loop_escape"), "decision_sha256": decision_sha,
                           "native_evidence": None, "prompt_count": None, "completion_count": None,
                           "mask_spans": None, "reasoning_spans": None, "action_spans": None,
                           "token_usable": False, "data_ready": False, "exclusion_reason": missing}
                    reason = missing
                    if session:
                        (native, native_reason), native_deps = native_futures[bid].result()
                        deps.update(native_deps)
                        reason = native_reason
                        if native:
                            p, c = native["prompt_token_ids"], native["output_token_ids"]
                            validate(p, c, native.get("request_sha256"), native.get("output_sha256"), verify_hashes=audit)
                            if audit and prompt is not None and p != prompt:
                                raise ValueError("Same decision has different actual native prompts")
                            if prompt is None:
                                prompt, prompt_sha = p, hash_json(p)
                            row_prompt_sha = hash_json(p)
                            output_sha = hash_json(c)
                            evidence = {"root": str(session["root"]), "ledger_ref": candidate["ledger_ref"], "ledger_sha256": candidate.get("ledger_sha256")}
                            row.update(native_evidence=evidence, prompt_sha256=row_prompt_sha, completion_sha256=output_sha,
                                       recorded_request_sha256=native.get("request_sha256"), recorded_output_sha256=native.get("output_sha256"),
                                       prompt_count=len(p), completion_count=len(c), native_finish=native.get("finish_reason"),
                                       mask_spans=spans(len(p), len(c)), tokenizer_fingerprint=native["tokenizer_fingerprint"])
                            from .online_probabilities import bind_online_probability
                            from runtime.serving.raw_logprobs import STREAM
                            probability = bind_online_probability(native, session, row, scfg, deps)
                            row["probability_streams"] = {STREAM: probability} if probability else {}
                            row["raw_probability_available"] = probability is not None
                            prompt_lengths[len(p)] += 1; output_lengths[len(c)] += 1
                            cohort = {"model": scfg["model"], "tokenizer_fingerprint": native["tokenizer_fingerprint"],
                                      "generation": {k:v for k,v in scfg.get("generation", {}).items() if k != "seed"},
                                      "serving_dtype": scfg.get("serving", {}).get("dtype"), "steering": scfg.get("steering")}
                            if common_cohort is None:
                                common_cohort = cohort
                            elif audit and cohort != common_cohort:
                                raise ValueError("Incompatible model/tokenizer/policy cohorts; build separate datasets")
                        response = candidate.get("response") or {}
                        choices = response.get("choices") or []
                        action_hash = None
                        if len(choices) == 1:
                            response_message = choices[0].get("message") or {}
                            try:
                                action_hash, action = canonical_action(response_message)
                            except ActionIdentityUnavailable as exc:
                                reason = reason or str(exc)
                            else:
                                row.update(action_hash=action_hash, action=action,
                                           reasoning_text=response_message.get("reasoning_content"), content=response_message.get("content"))
                                if choices[0].get("finish_reason") not in {"stop", "tool_calls", "function_call"}:
                                    reason = reason or "incomplete_response"
                        else:
                            reason = reason or "missing_full_response"
                        if audit and row["selected"] and action_hash is not None:
                            if canonical_action(message)[0] != action_hash:
                                raise ValueError("Historical winner differs from canonical action")
                            if candidate.get("native_request_id") not in session["finalization"].get("accepted_native_request_ids", []):
                                raise ValueError("Winner native request absent from finalization")
                    row["exclusion_reason"] = reason
                    row["token_usable"] = reason is None and row["native_evidence"] is not None
                    rows.append(row)
                base["prompt_sha256"] = prompt_sha
                gid = group_id(base)
                scoring_exclusion = source.get("terminal_scoring_exclusion")
                ready = n >= dc.quality.min_branches and all(r["token_usable"] for r in rows) and not scoring_exclusion
                origin = tuple(r["native_request_id"] for r in rows)
                origin_signature = (sid, did, prompt_sha, hash_json([
                    {k:r.get(k) for k in ("completion_sha256","tokenizer_fingerprint","native_finish","action_hash")} for r in rows]))
                if all(origin) and origin in origin_groups:
                    previous = origin_groups[origin]
                    if audit and previous[1] != origin_signature:
                        raise ValueError("Ambiguous native request collision across decisions")
                    json_line(streams["aliases"], {**base, "alias_group_id": gid, "canonical_group_id": previous[0], "policy": "same_native_request_inventory_v1"})
                    counts["alias_groups"] += 1
                    continue
                if all(origin):
                    origin_groups[origin] = (gid, origin_signature)
                member_ids = []
                for row in rows:
                    request = row["native_request_id"]
                    owner = (sid, did, row["branch_id"])
                    if audit and request and request in request_owners and request_owners[request] != owner:
                        raise ValueError("Native request is bound to multiple decision/branch identities")
                    if request:
                        request_owners[request] = owner
                    row.update(group_id=gid, branch_sample_id=branch_id(gid, row["branch_id"], request), data_ready=ready, audit_enabled=audit)
                    member_ids.append(row["branch_sample_id"])
                    json_line(streams["branches"], row)
                    counts["usable_branches"] += int(row["token_usable"])
                    counts["training_branches"] += int(ready)
                    counts["selected_branches"] += int(row["selected"])
                    counts["retained_branches"] += int(row["retained"] is True)
                    if row["exclusion_reason"]:
                        excluded[row["exclusion_reason"]] += 1
                        json_line(streams["exclusions"], {"group_id": gid, "branch_sample_id": row["branch_sample_id"], "reason": row["exclusion_reason"], "severity": "data"})
                group = {**base, "group_id": gid, "expected_branch_count": n, "available_branch_count": sum(r["token_usable"] for r in rows),
                         "branch_ids": member_ids, "data_ready": ready, "grpo_ready": False,
                         "raw_probability_complete": all(r.get("raw_probability_available", False) for r in rows),
                         "historical_selected_branch_id": decision.get("selected_branch_id"), "prompt_ref": next((r["native_evidence"] for r in rows if r["native_evidence"]), None)}
                json_line(streams["groups"], group)
                counts["training_groups"] += int(ready)
                if not ready:
                    group_reason = scoring_exclusion or ("incomplete_native_group" if n >= 2 else "single_branch")
                    if scoring_exclusion:
                        excluded[scoring_exclusion] += 1
                    json_line(streams["exclusions"], {"group_id": gid, "reason": group_reason, "severity": "group"})
                for stream in streams.values():
                    stream.flush()
                write_json(stage / "progress.json", {"state": "building", "rollout_id": source["rollout_id"],
                           "decision_id": did, "counts": dict(counts)})
            for sid, (session, _) in sessions.items():
                if session:
                    write_json(stage / "source-evidence" / run / sid / "token-publication.json", {"publication": session["publication"], "proof_version": session["proof_version"]})
            _checkpoint(stage,streams,source_ordinal+1,counters,common_cohort,request_owners,origin_groups,deps)
        for stream in streams.values():
            stream.flush()
        summary = {**counts, "group_sizes": dict(sizes), "exclusion_reasons": dict(excluded),
                   "prompt_lengths": _lengths(prompt_lengths), "completion_lengths": _lengths(output_lengths),
                   "supervised_lengths": _lengths(output_lengths), "alias_count": counts["alias_groups"],
                   "behavior_logprobs": "not_extracted", "audit_enabled": audit, "grpo_ready": False}
        write_json(stage / "summary.json", summary)
        write_json(stage / "dependencies.json", {"files": deps})
        from .online_probabilities import freeze_dataset_stream
        probability_streams = freeze_dataset_stream(stage, max_shard_bytes=dc.storage.max_shard_bytes)
        write_json(stage / "dataset.json", {"schema_version": 1, "dataset_id": dc.dataset_id,
                   "probability_streams": probability_streams,
                   "mask_policy": MASK_POLICY, "quality": dc.quality.model_dump(), "cohort": common_cohort,
                   "data_ready": counts["training_groups"] > 0, "grpo_ready": False,
                   "audit": {"enabled": audit, "source_hashes_verified": audit, "identity_verified": audit, "token_hashes_verified": audit}})
        seal(stage, "rl-dataset", dc.dataset_id, selection_sha256=hash_json(frozen["selection"]))
    return {"status": "published", "dataset_id": dc.dataset_id, "path": str(destination), "summary": summary, "receipt_sha256": sha256(destination / "receipt.json")}


def check_dataset(path, level="full"):
    path = Path(path); receipt = verify_local(path, "rl-dataset")
    audit = read_json(path / "dataset.json").get("audit", {}).get("enabled", False)
    checked = verify_dependencies(path) if audit and level in {"sources", "full"} else 0
    branches = iter(iter_rows(path / "branches.jsonl")); count = 0; groups = 0
    for group in iter_rows(path / "groups.jsonl"):
        actual = []
        for _ in range(group["expected_branch_count"]):
            row = next(branches, None)
            if row is None or row["group_id"] != group["group_id"]:
                raise ValueError("Incomplete or misordered group membership")
            actual.append(row["branch_sample_id"]); count += 1
            if audit and level == "full" and row["native_evidence"]:
                from .loader import native_tokens
                native_tokens(row)
            if level == "full" and row.get("probability_streams"):
                from .online_probabilities import OnlineProbabilityReader
                OnlineProbabilityReader(path).record(row)
        if actual != group["branch_ids"]:
            raise ValueError("Group/member order mismatch")
        groups += 1
    if next(branches, None) is not None:
        raise ValueError("Unbound branch rows")
    return {"status": "verified", "dataset_id": receipt["artifact_id"], "groups": groups, "branches": count, "dependencies_checked": checked, "check_level": level}

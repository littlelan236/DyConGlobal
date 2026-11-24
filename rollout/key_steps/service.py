"""Formal CLI planning, selection, restoration and continuation sealing."""

import copy
import json
import math
import random
from collections import defaultdict, deque
from pathlib import Path

import yaml

from core.io import hash_json, read_json, run_path, write_json
from rollout.key_steps.peaks import VERSION, detect, signal
from rollout.key_steps.state import ReplayMismatch, replay, rows


def publish(path, value):
    value = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    value = {**value, "sha256": hash_json(value)}
    path = Path(path)
    if path.exists() and read_json(path) != value:
        raise ValueError("Immutable key-step artifact already exists with different inputs")
    write_json(path, value)
    return value


def verify(value, schema):
    if value.get("schema") != schema or value.get("sha256") != hash_json(
        {k: v for k, v in value.items() if k != "sha256"}
    ):
        raise ValueError("Key-step artifact identity mismatch: " + schema)
    return value


def plan(cfg, source_run, output):
    from application.execution.evidence import RunEvidence
    from application.execution.run_plan import load_plan
    from rollout.retry import default_final_ids
    from steering.datasets.selection import select_sources

    loaded = load_plan(cfg["project_root"], source_run)
    scfg = loaded["config"]
    settings = scfg.get("key_steps") or {}
    if settings.get("mode") != "source" or scfg["parallel"]["branch_count"] != 8:
        raise ValueError("Expected a new key-step source run with B=8 evidence")
    evidence = RunEvidence(cfg["project_root"], source_run).read()
    ids = default_final_ids(cfg["project_root"], source_run, evidence)
    if not ids:
        raise ValueError("Source run has no accepted final trajectories")
    frozen = select_sources(cfg, [source_run], rollout_ids=ids, verify_source_hashes=False)
    sources = []
    for source in frozen["sources"]:
        out = (
            run_path(cfg["project_root"], source_run)
            / "rollout/tasks"
            / source["instance_id"]
            / "repetitions"
            / source["repetition_id"]
            / "attempts"
            / source["attempt_id"]
        )
        boundaries = rows(out / "replay-boundaries.jsonl")
        if not boundaries:
            continue
        source = {
            **source,
            "attempt_dir": str(out),
            "target_step": max(b["step"] for b in boundaries),
            "trajectory_bytes": Path(source["trajectory_path"]).stat().st_size,
            "boundary_sha256": hash_json(boundaries),
        }
        sources.append(source)
    publish(
        Path(output) / "plan.json",
        {
            "schema": "key-step-plan-v1",
            "source_run": source_run,
            "source_config_hash": scfg["config_hash"],
            "settings": settings,
            "sources": sources,
        },
    )
    scan = publish(
        Path(output) / "scan-inventory.json",
        {
            "schema": "key-step-scan-v1",
            "branch_count": 8,
            "source_run": source_run,
            "source_config_hash": scfg["config_hash"],
            "children": [
                {
                    "child_id": s["rollout_id"],
                    "task_key": s["task_key"],
                    "instance_id": s["instance_id"],
                    "family": scfg["selection"]["tasks"][s["task_key"]]["instance"]["repo"],
                    "seed": int(hash_json([s["rollout_id"], "scan"])[:15], 16),
                    "source": s,
                }
                for s in sources
            ],
        },
    )
    options = {**settings, "mode": "scan", "inventory": scan}
    Path(output, "scan.yaml").write_text(yaml.safe_dump(options, sort_keys=False), encoding="utf8")
    return {
        "sources": len(sources),
        "output": str(Path(output).resolve()),
        "scan_config": str(Path(output, "scan.yaml").resolve()),
    }


def source_decisions(source, scfg):
    from hidden.token_evidence import load_token_session
    from steering.datasets.sessions import full_decision

    if Path(source["trajectory_path"]).stat().st_size != source["trajectory_bytes"]:
        raise ValueError("Canonical source trajectory size changed")
    trajectory = read_json(source["trajectory_path"])
    sessions = {}
    records = []
    for mi, message in enumerate(trajectory.get("messages", [])):
        summary = (message.get("extra") or {}).get("dycon_parallel")
        if not summary:
            continue
        sid = summary["session_id"]
        if sid not in sessions:
            sessions[sid], missing = load_token_session(source, scfg, sid, {}, audit=False)
            if sessions[sid] is None:
                raise ValueError("Source token evidence: " + str(missing))
        session = sessions[sid]
        decision, sha = full_decision(session, summary)
        records.append(
            {
                "step": summary["step_id"],
                "message_index": mi,
                "decision": decision,
                "decision_path": str(
                    session["base"] / "parallel/decisions" / (summary["decision_id"] + ".json")
                ),
                "decision_sha256": hash_json(decision),
                "ledger_root": str(session["root"]),
            }
        )
    return trajectory, records


def select(cfg, plan_path, scan_run, output):
    from application.execution.run_plan import load_plan

    planned = verify(read_json(plan_path), "key-step-plan-v1")
    root = cfg["project_root"]
    scanned = load_plan(root, scan_run)
    scan_settings = scanned["config"].get("key_steps") or {}
    if (
        scan_settings.get("mode") != "scan"
        or scan_settings["inventory"]["source_run"] != planned["source_run"]
    ):
        raise ValueError("Scan belongs to another source")
    source_cfg = load_plan(root, planned["source_run"])["config"]
    if source_cfg["config_hash"] != planned["source_config_hash"]:
        raise ValueError("Source config changed")
    scans = {}
    for item in scanned["work_items"]["items"]:
        work = (
            run_path(root, scan_run)
            / "rollout/tasks"
            / item["instance_id"]
            / "repetitions"
            / item["repetition_id"]
        )
        if not (work / "result.json").exists():
            continue
        result = read_json(work / "result.json")
        receipt = work / "attempts" / result["attempt_id"] / "replay-validation.json"
        if receipt.exists():
            scans[item["continuation"]["source"]["rollout_id"]] = read_json(receipt)
    children = []
    audits = []
    for source in planned["sources"]:
        rid = source["rollout_id"]
        trajectory, decisions = source_decisions(source, source_cfg)
        boundaries = rows(Path(source["attempt_dir"]) / "replay-boundaries.jsonl")
        if hash_json(boundaries) != source["boundary_sha256"]:
            raise ValueError("Source boundary records changed")
        bystep = {r["step"]: r for r in boundaries}
        eligible = {r["step"] for r in scans.get(rid, {}).get("boundaries", []) if r["eligible"]}
        signals = []
        usable = {}
        reasons = {}
        for row in decisions:
            t = row["step"]
            branches = row["decision"]["candidates"]
            sig = signal(t, branches)
            signals.append(sig)
            entropy = []
            valid = sig["complete"] and t in eligible and t in bystep
            for b in branches:
                stat = b.get("generation_entropy") or {}
                e = stat.get("entropy_mean")
                valid &= (
                    stat.get("semantics") == "sampler_distribution_entropy_nats_v1"
                    and type(e) in (int, float)
                    and math.isfinite(e)
                    and stat.get("entropy_count", 0) > 0
                    and bool(b.get("response"))
                    and bool(b.get("ledger_ref"))
                    and not b.get("reasoning_loop")
                    and b.get("native_finish_reason") != "abort"
                )
                if e is not None:
                    entropy.append(e)
            if valid and len(branches) == 8:
                usable[t] = {**row, "entropy": math.fsum(entropy) / 8}
            else:
                reasons[t] = "replay_or_full_candidate_score_entropy_evidence_unavailable"
        peaks, params = detect(signals)
        adaptive = sorted(p["step"] for p in peaks if p["step"] in usable)
        k = len(adaptive)
        rng = random.Random(int(hash_json([planned["source_config_hash"], rid, "uniform-v1"])[:16], 16))
        methods = {
            "adaptive": adaptive,
            "random": sorted(rng.sample(sorted(usable), k)),
            "entropy": sorted(sorted(usable, key=lambda t: (-usable[t]["entropy"], t))[:k]),
        }
        audit = {
            "rollout_id": rid,
            "K": k,
            "U": sorted(usable),
            "peaks": peaks,
            "parameters": params,
            "methods": methods,
            "excluded": reasons,
        }
        audits.append(audit)
        for t in sorted(set().union(*[set(v) for v in methods.values()])):
            row = usable[t]
            parent = "p-" + hash_json([source["source_run_id"], rid, t])[:32]
            for b in range(8):
                children.append(
                    {
                        "child_id": "c-" + hash_json([parent, b])[:32],
                        "parent_id": parent,
                        "task_key": source["task_key"],
                        "instance_id": source["instance_id"],
                        "family": source_cfg["selection"]["tasks"][source["task_key"]]["instance"]["repo"],
                        "source_native_request_id": next(
                            c["native_request_id"]
                            for c in row["decision"]["candidates"]
                            if c["branch_id"] == b
                        ),
                        "step": t,
                        "branch": b,
                        "methods": [m for m, v in methods.items() if t in v],
                        "seed": int(hash_json([rid, t, b, "continuation-v1"])[:15], 16),
                        "source": source,
                        **{
                            key: row[key]
                            for key in ("message_index", "decision_path", "decision_sha256", "ledger_root")
                        },
                    }
                )
    inventory = publish(
        Path(output) / "inventory.json",
        {
            "schema": "key-step-inventory-v1",
            "branch_count": 8,
            "source_run": planned["source_run"],
            "source_config_hash": planned["source_config_hash"],
            "selector": VERSION,
            "scan_run": scan_run,
            "audits": audits,
            "children": children,
        },
    )
    options = {**planned["settings"], "mode": "continuation", "inventory": inventory}
    Path(output, "continuation.yaml").write_text(yaml.safe_dump(options, sort_keys=False), encoding="utf8")
    return {
        "children": len(children),
        "parents": len({x["parent_id"] for x in children}),
        "output": str(Path(output).resolve()),
        "continuation_config": str(Path(output, "continuation.yaml").resolve()),
    }


def continue_agent(agent, settings, child, out):
    source = child["source"]
    boundaries = rows(Path(source["attempt_dir"]) / "replay-boundaries.jsonl")
    if hash_json(boundaries) != source["boundary_sha256"]:
        raise ReplayMismatch("Boundary evidence changed")
    target = source["target_step"] if settings["mode"] == "scan" else child["step"]
    boundary = replay(
        agent.env, source["attempt_dir"], settings, target, Path(out) / "replay-validation.json"
    )
    if settings["mode"] == "scan":
        return {"exit_status": "ReplayValidated", "submission": ""}
    raw = read_json(source["trajectory_path"])
    history = raw["messages"][: boundary["message_index"]]
    if hash_json(agent.model._prepare_messages_for_api(history)) != boundary["history_sha256"]:
        raise ReplayMismatch("Historical prompt changed")
    decision = read_json(child["decision_path"])
    if hash_json(decision) != child["decision_sha256"]:
        raise ReplayMismatch("Historical candidates changed")
    branch = next(b for b in decision["candidates"] if b["branch_id"] == child["branch"])
    from hidden.artifacts import read_request

    native = read_request(child["ledger_root"], branch)
    if (
        native["request_id"] != branch["native_request_id"]
        or native["branch_id"] != child["branch"]
        or native["step_id"] != child["step"]
        or not native.get("output_token_ids")
    ):
        raise ReplayMismatch("Injected native branch identity mismatch")
    state = boundary["agent"]
    agent.messages = copy.deepcopy(history)
    # Source query increments n_calls before querying. Boundary is before that.
    agent.n_calls = state["n_calls"]
    agent.cost = state["cost"]
    agent.n_consecutive_format_errors = state["format_errors"]
    agent._loop_history = deque(state["loop_history"], maxlen=agent._loop_history.maxlen)
    agent._loop_step = state["loop_step"]
    agent.extra_template_vars = state["extra_template_vars"]
    agent._continuation_ref = {
        "schema": "continuation-prefix-v1",
        "trajectory_path": source["trajectory_path"],
        "trajectory_sha256": source["trajectory_sha256"],
        "trajectory_bytes": source["trajectory_bytes"],
        "prefix_message_count": len(history),
        "history_sha256": boundary["history_sha256"],
        "source_step": child["step"],
        "child_id": child["child_id"],
        "parent_id": child["parent_id"],
    }
    import time

    agent._start_time = time.time() - state["elapsed_seconds"]
    from rollout.loop_escape import LoopEscapeHistory, SlotHistory

    saved = state["loop_escape"]
    agent.model._dycon_loop_escape = LoopEscapeHistory(
        decisions=saved["decisions"], slots={int(k): SlotHistory(**v) for k, v in saved["slots"].items()}
    )
    agent.model._dycon_capture_step = child["step"] + 1
    agent.model._dycon_metric_step = child["step"] + 1
    agent.model.inject_saved_response(
        branch["response"],
        branch.get("cost", 0.0),
        {
            "ledger_root": child["ledger_root"],
            "ledger_ref": branch["ledger_ref"],
            "native_request_id": branch["native_request_id"],
            "parent_id": child["parent_id"],
            "child_id": child["child_id"],
            "source_step": child["step"],
        },
    )
    write_json(
        Path(out) / "continuation-source.json",
        {
            "child": child,
            "prefix_message_count": len(history),
            "source_boundary": boundary["step"],
            "injected_native_request_id": native["request_id"],
        },
    )
    return agent.continue_run()


def seal(cfg, run_id, output):
    """Seal terminal reward groups; ungraded/infra members stay null."""
    from application.execution.evidence import RunEvidence
    from application.execution.run_plan import load_plan
    from benchmark_evaluation.swebench_settlement import training_label

    loaded = load_plan(cfg["project_root"], run_id)
    settings = loaded["config"].get("key_steps") or {}
    if settings.get("mode") != "continuation":
        raise ValueError("Expected a continuation run")
    evidence = RunEvidence(cfg["project_root"], run_id).read()
    bysample = defaultdict(list)
    for rid, row in evidence["rollouts"].items():
        bysample[row.get("sample_key")].append((rid, row))
    groups = defaultdict(list)
    methods = defaultdict(set)
    for item in loaded["work_items"]["items"]:
        child = item["continuation"]
        parent = child["parent_id"]
        result_path = (
            loaded["run"]
            / "rollout/tasks"
            / item["instance_id"]
            / "repetitions"
            / item["repetition_id"]
            / "result.json"
        )
        result = read_json(result_path) if result_path.exists() else {}
        matches = [
            (rid, r)
            for rid, r in bysample[item["sample_key"]]
            if r.get("attempt_id") == result.get("attempt_id")
        ]
        row = matches[0][1] if len(matches) == 1 else {}
        reward = None
        if result.get("disposition") == "accepted":
            if row.get("submission_status") != "submitted" and row:
                reward = 0
            elif matches:
                label = training_label(row, evidence, matches[0][0])
                if type(label) is bool:
                    reward = int(label)
        groups[parent].append(
            {
                "child_id": child["child_id"],
                "branch_id": child["branch"],
                "reward": reward,
                "result_ref": str(result_path),
                "trajectory_ref": row.get("trajectory_path"),
                "trajectory_sha256": row.get("trajectory_sha256"),
                "native_prefix_ref": {"root": child["ledger_root"], "decision": child["decision_path"]},
                "source_ref": child,
                "terminal_status": result.get("exit_status"),
            }
        )
        for method in child["methods"]:
            methods[method].add(parent)
    from RL.dataset.continuation import member_turns

    for members in groups.values():
        for member in members:
            member["native_coverage"] = False
            if member["trajectory_ref"] is None:
                continue
            try:
                turns = member_turns(member, cfg["project_root"])
                member["native_coverage"] = True
                member["turn_count"] = len(turns)
                member["completion_tokens"] = sum(len(t["completion_token_ids"]) for t in turns)
            except (ValueError, KeyError, OSError, StopIteration) as exc:
                member["native_evidence_error"] = str(exc)[:500]
    sealed = []
    for parent, members in sorted(groups.items()):
        members.sort(key=lambda x: x["branch_id"])
        complete = (
            len(members) == 8
            and [m["branch_id"] for m in members] == list(range(8))
            and all(m["reward"] is not None for m in members)
        )
        rewards = [m["reward"] for m in members] if complete else []
        mean = sum(rewards) / 8 if complete else None
        std = math.sqrt(sum((x - mean) ** 2 for x in rewards) / 8) if complete else None
        for member in members:
            member["advantage"] = (member["reward"] - mean) / (std + 1e-8) if complete else None
        sealed.append(
            {
                "parent_id": parent,
                "complete": complete,
                "zero_variance": std == 0 if complete else None,
                "members": members,
                "native_coverage": all(m["native_coverage"] for m in members),
                "train_eligible": complete and std > 0 and all(m["native_coverage"] for m in members),
            }
        )
    # Physical children shared by methods retain one result; pair completeness
    # is evaluated per original trajectory, not by silently filling failed points.
    failed_sources = {
        m["source_ref"]["source"]["rollout_id"]
        for g in sealed
        if not g["complete"] or not g["native_coverage"]
        for m in g["members"]
    }
    for group in sealed:
        group["paired_eligible"] = not any(
            m["source_ref"]["source"]["rollout_id"] in failed_sources for m in group["members"]
        )
        group["train_eligible"] &= group["paired_eligible"]
    publish(
        Path(output) / "groups.json",
        {
            "schema": "continuation-groups-v1",
            "run_id": run_id,
            "project_root": cfg["project_root"],
            "config_hash": loaded["config"]["config_hash"],
            "branch_count": 8,
            "groups": sealed,
            "methods": {k: sorted(v) for k, v in methods.items()},
            "data_ready": bool(sealed) and all(g["complete"] and g["native_coverage"] for g in sealed),
            "collection_complete": bool(sealed) and all(g["complete"] for g in sealed),
            "grpo_ready": False,
            "mask_policy": "per_turn_prompt_0_completion_1_v1",
        },
    )
    return {
        "groups": len(sealed),
        "complete": sum(g["complete"] for g in sealed),
        "train_eligible": sum(g["train_eligible"] for g in sealed),
        "grpo_ready": False,
        "path": str(Path(output, "groups.json").resolve()),
    }

"""Small real-schema published captures. No model, GPU or container invocation."""

import copy
import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from application.execution.evidence import RunEvidence
from core.io import hash_json, sha256, write_json
from rollout.consensus import consensus_options, select_consensus


def make_source(
    root,
    run="run-a",
    *,
    single_class=False,
    missing_response=None,
    partial_branch=None,
    ambiguous=None,
    bad_request=None,
    no_closes=False,
    startup=False,
    task_name="unit__repo-1",
    sharded=False,
    tokenizer="unit-tokenizer",
    fingerprint="unit-model",
    no_winner=False,
    mode="parallel_consensus",
    representative_policy=None,
    retention_rule="effective_count_v1",
):
    root = Path(root)
    model = root / "model"
    write_json(model / "config.json", {"text_config": {"num_hidden_layers": 2, "hidden_size": 3}})
    task = {
        "benchmark": "swebench",
        "revision": "unit-v1",
        "instance_id": task_name,
        "repo": "unit/repo",
        "base_commit": "abc",
        "partition": "train",
        "instance": {"instance_id": task_name, "repo": "unit/repo", "base_commit": "abc"},
    }
    task_key = hash_json({k: task[k] for k in ("benchmark", "revision", "instance_id")})
    rid, sid, did = "r-unit", ("startup-unit" if startup else "cap-" + run), "dec-unit"
    nas = root / "nas" / "hidden/online" / run
    session = nas / "sessions" / sid
    options = {
        "retention_ratio": 0.5,
        "min_retained_branches": 2,
        "rule": "exact_action",
        "tie_policy": "estimator_score",
    }
    if representative_policy is not None:
        options["representative_policy"] = representative_policy
    config = {
        "project_root": str(root),
        "paths": {"nas_root": str(root / "nas")},
        "model": {"path": str(model), "fingerprint": fingerprint},
        "capture": {"capture_stage": "post_steering", "export": {"enabled": True, "root": "hidden/online"}},
        "parallel": {
            "consensus": options,
            "mode": mode,
            "entropy": {"retention_rule": retention_rule, "full_retention_threshold": 10},
        },
        "steering": {"enabled": False},
        "serving": {"dtype": "float32"},
    }
    decision_options = consensus_options(config["parallel"])
    write_json(root / "runs" / run / "resolved-config.json", config)
    candidates, events, arrays_by_sample = [], [], {}
    actions = ["a", "b", "a", "a", "b", "a"] if not single_class else ["a"] * 6
    for bid, action in enumerate(actions):
        native = f"native-{run}-{bid}"
        position = 100 + bid
        sample = hash_json(
            {"session": sid, "request": native, "anchor": "think_close", "position": position}
        )[:32]
        response = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{"function": {"name": "bash", "arguments": {"command": action}}}],
                    },
                }
            ]
        }
        if bid == partial_branch:
            response["choices"][0]["finish_reason"] = "length"
        if bid == missing_response or no_winner:
            response = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant"}}]}
        candidate = {
            "branch_id": bid,
            "score": float(bid),
            "response": response,
            "completion_tokens": [11, 8, 6, 15, 7, 9][bid],
            "status": "failed" if no_winner else "eligible",
            "native_request_id": native,
            "close_sample_id": None if ambiguous == bid else sample,
            "absolute_position": position,
            "token_id": 42,
            "score_identity": {
                "session_id": sid,
                "decision_id": did,
                "branch_id": bid,
                "step_id": 0,
                "api_attempt_id": f"api-{bid}",
                "request_sha256": "request-sha",
            },
        }
        candidates.append(candidate)
        value = np.random.default_rng(bid + 10).normal(size=3).astype(np.float32)
        value[0] += 2 if action == "a" else -2
        buffer = io.BytesIO()
        np.savez(buffer, layer_0=value, layer_2=value * 2)
        target = nas / "raw" / (sample + ".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(buffer.getvalue())
        event = {
            "session_id": sid,
            "decision_id": did,
            "branch_id": bid,
            "request_id": "wrong-request" if bad_request == bid else native,
            "message_index": 0,
            "step_id": 0,
            "parent_prefix_sha256": "parent-sha",
            "query_id": "query-unit",
            "api_attempt_id": f"api-{bid}",
            "request_sha256": "request-sha",
            "run_id": run,
            "sample_id": sample,
            "anchor": "think_close",
            "phase": "generation_input",
            "absolute_position": position,
            "token_id": 42,
            "status": "persisted",
            "model_fingerprint": fingerprint,
            "tokenizer_fingerprint": tokenizer,
            "capture_stage": "post_steering",
            "serving_dtype": "float32",
            "generation_index": 2,
            "storage_shapes": {"layer_0": [3], "layer_2": [3]},
            "storage_dtypes": {"layer_0": "float32", "layer_2": "float32"},
            "tensor": target.relative_to(nas).as_posix(),
            "sha256": sha256(target),
        }
        events.append(event)
        arrays_by_sample[sample] = value
    proxy = [
        SimpleNamespace(
            branch_id=c["branch_id"],
            message={},
            response=SimpleNamespace(model_dump=lambda mode="json", response=c["response"]: response),
        )
        for c in candidates
        if c["status"] == "eligible"
    ]
    winner, consensus = select_consensus(
        proxy,
        {c["branch_id"]: c["score"] for c in candidates},
        0.5,
        2,
        "estimator_score",
        representative_policy=representative_policy,
        retention_rule=decision_options.get("retention_rule"),
        full_retention_threshold=decision_options.get("full_retention_threshold", 10),
        token_lengths={c["branch_id"]: c["completion_tokens"] for c in candidates},
    )
    for meta in consensus.get("branch_metadata", []):
        candidates[meta["branch_id"]].update(meta)
    selected = winner.branch_id if winner else None
    decision = {
        "schema_version": 1,
        "session_id": sid,
        "decision_id": did,
        "step_id": 0,
        "query_id": "query-unit",
        "parent_prefix_sha256": "parent-sha",
        "branch_count": 6,
        "selection_mode": mode,
        "selected_branch_id": selected,
        "consensus_config": decision_options,
        "consensus": consensus,
        "candidates": candidates,
    }
    embedded = copy.deepcopy(decision)
    for candidate in embedded["candidates"]:
        candidate.pop("response")
    message = (
        copy.deepcopy(candidates[selected]["response"]["choices"][0]["message"])
        if selected is not None
        else {"role": "assistant"}
    )
    message["extra"] = {"dycon_parallel": embedded}
    trajectory = root / "runs" / run / "rollout/trajectories" / rid / "raw.traj.json"
    write_json(trajectory, {"messages": [message]})
    trajectory_sha = sha256(trajectory)
    if sharded:
        shard = nas / "shards" / "unit.tar"
        shard.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(shard, "w") as archive:
            for event in events:
                archive.add(nas / event["tensor"], arcname=Path(event["tensor"]).name)
        shard_sha = sha256(shard)
        for event in events:
            event["tensor_locator"] = {
                "shard": "shards/unit.tar",
                "member": Path(event["tensor"]).name,
                "member_sha256": event["sha256"],
                "shard_sha256": shard_sha,
            }
    index_rows = []
    for bid, event in enumerate(events):
        close_events = [] if no_closes else [event]
        if ambiguous == bid and not no_closes:
            other = copy.deepcopy(event)
            other["absolute_position"] += 10
            other["sample_id"] = hash_json(
                {
                    "session": sid,
                    "request": event["request_id"],
                    "anchor": "think_close",
                    "position": other["absolute_position"],
                }
            )[:32]
            close_events.append(other)
        index_rows.append(
            {
                "decision_id": did,
                "branch_id": bid,
                "selected": bid == selected,
                "native_request_id": candidates[bid]["native_request_id"],
                "rollout_id": rid,
                "trajectory_sha256": trajectory_sha,
                "close_events": close_events,
                "close_sample_ids": [event["sample_id"] for event in close_events],
            }
        )
    write_json(session / "parallel/decisions" / (did + ".json"), decision)
    write_json(session / "parallel/branch-index.json", index_rows)
    write_json(session / "parallel/branch-coverage.json", {"candidate_count": 6})
    branch_files = {p.relative_to(nas).as_posix(): sha256(p) for p in (session / "parallel").rglob("*.json")}
    write_json(
        session / "parallel/branch-receipt.json",
        {"schema_version": 1, "rollout_id": rid, "trajectory_sha256": trajectory_sha, "files": branch_files},
    )
    write_json(session / "session.json", {"session_id": sid})
    write_json(session / "receipt.json", {"session_id": sid, "status": "stopped"})
    write_json(
        session / "finalization.json",
        {
            "session_id": sid,
            "rollout_id": rid,
            "trajectory_path": str(trajectory),
            "trajectory_sha256": trajectory_sha,
            "model_fingerprint": fingerprint,
            "accepted_native_request_ids": [candidates[selected]["native_request_id"]]
            if selected is not None
            else [],
            "responses": {candidates[selected]["native_request_id"]: {"message_index": 0}}
            if selected is not None
            else {},
        },
    )
    write_json(nas / "inventory.json", {"close_token_ids": [42], "tokenizer_fingerprint": tokenizer})
    names = [
        "session.json",
        "receipt.json",
        "finalization.json",
        "parallel/branch-index.json",
        "parallel/branch-coverage.json",
        "parallel/branch-receipt.json",
    ]
    write_json(
        root / "runs" / run / "hidden/publication.json",
        {
            "run_id": run,
            "nas_locator": str(nas),
            "sessions": [
                {
                    "session_id": sid,
                    "rollout_id": rid,
                    "trajectory_sha256": trajectory_sha,
                    "files": {
                        name: {"nas_locator": str(session / name), "sha256": sha256(session / name)}
                        for name in names
                    },
                }
            ],
        },
    )
    patch = root / "runs" / run / "rollout/patches" / rid / "patch.diff"
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_text("", encoding="utf-8")
    row = {
        "run_id": run,
        "task_key": task_key,
        "attempt_id": "attempt-0",
        "trajectory_path": trajectory.relative_to(root).as_posix(),
        "trajectory_sha256": trajectory_sha,
        "patch_path": patch.relative_to(root).as_posix(),
        "patch_sha256": sha256(patch),
        "model_fingerprint": fingerprint,
        "submission_status": "submitted",
        "termination_reason": "Submitted",
        "evaluation_status": "not_run",
        "resolved": None,
        "selected_evaluation_id": None,
        "provenance": {},
    }
    RunEvidence(root, run).register_rollout(rid, row, task)
    return {
        "root": root,
        "run": run,
        "rid": rid,
        "sid": sid,
        "nas": nas,
        "session": session,
        "row": row,
        "task": task,
        "task_key": task_key,
        "arrays": arrays_by_sample,
        "cfg": {"project_root": str(root), "paths": {}},
        "decision": decision,
    }

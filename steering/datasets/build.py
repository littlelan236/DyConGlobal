"""All-candidate, historical-action datasets with immutable provenance."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from core.action_identity import ActionIdentityUnavailable, canonical_action
from core.io import hash_json, read_json, sha256, write_json
from rollout.consensus import CONSENSUS_MODES, consensus_options, validate_consensus_evidence
from steering.artifacts import iter_jsonl, seal, verify_dependencies, verify_local, write_jsonl
from steering.datasets.features import inventory_for, select_sites
from steering.datasets.labels import LABEL_RULE, branch_action, label_branch
from steering.datasets.paths import publication
from steering.datasets.selection import select_sources
from steering.datasets.sessions import add_dependency, choose_close, full_decision, load_session


def _exclude(row, reason):
    return {
        **row,
        "action_label": row.get("label"),
        "action_y": row.get("y"),
        "label": "excluded",
        "y": None,
        "exclusion_reason": reason,
    }


def _model_inventory(cfg, dependencies):
    path = Path(cfg["model"]["path"]) / "config.json"
    if not path.is_file():
        explicit = cfg.get("capture", {}).get("model_inventory")
        if explicit is None:
            raise ValueError("Frozen model config is needed to identify decoder_output/final_norm sites")
        return explicit
    model = read_json(path)
    add_dependency(dependencies, path)
    return model.get("text_config", model)


def build_dataset(
    cfg,
    dataset_id,
    *,
    source_runs,
    rollout_ids=(),
    rollout_list=None,
    all_submitted=False,
    storage_root=None,
    sites=None,
):
    frozen = select_sources(
        cfg, source_runs, rollout_ids=rollout_ids, rollout_list=rollout_list, all_submitted=all_submitted
    )
    dependencies = frozen["dependencies"]
    samples, exclusions, decisions, aliases, rollouts = [], [], [], [], []
    seen_captures, common_space, common_inventory = {}, None, None
    with publication(cfg, "steering_dataset", dataset_id, storage_root) as (stage, destination):
        write_json(stage / "selection.json", frozen["selection"])
        for run, snapshot in frozen["snapshots"].items():
            write_json(stage / "source-evidence" / run / "evidence.json", snapshot)
            write_json(stage / "source-evidence" / run / "resolved-config.json", frozen["configs"][run])
        for source_index, source in enumerate(frozen["sources"]):
            run, rid = source["source_run_id"], source["rollout_id"]
            source_cfg = frozen["configs"][run]
            model = _model_inventory(source_cfg, dependencies)
            write_json(stage / "source-evidence" / run / "model-inventory.json", model)
            publication_path = Path(source["run_directory"]) / "hidden/publication.json"
            published = read_json(publication_path) if publication_path.is_file() else None
            trajectory = read_json(source["trajectory_path"])
            if not isinstance(trajectory.get("messages"), list):
                raise ValueError("Canonical trajectory must contain a messages list")
            sessions = {}
            source_summary = {
                **source,
                "decision_count": 0,
                "candidate_count": 0,
                "included_count": 0,
                "excluded_count": 0,
                "nonparallel_assistant_messages": 0,
            }
            for message_index, message in enumerate(trajectory["messages"]):
                embedded = (message.get("extra") or {}).get("dycon_parallel")
                if not isinstance(embedded, dict):
                    source_summary["nonparallel_assistant_messages"] += int(
                        message.get("role") == "assistant"
                    )
                    continue
                if message.get("role") != "assistant":
                    raise ValueError("Parallel decision must belong to a canonical assistant response")
                sid, did = embedded["session_id"], embedded["decision_id"]
                if sid not in sessions:
                    sessions[sid] = load_session(
                        source,
                        source_cfg,
                        sid,
                        published,
                        stage / "source-evidence" / run / sid,
                        dependencies,
                    )
                session, missing_session = sessions[sid]
                base = {
                    k: source[k]
                    for k in (
                        "source_run_id",
                        "rollout_id",
                        "trajectory_sha256",
                        "task_key",
                        "benchmark",
                        "revision",
                        "instance_id",
                        "attempt_id",
                        "repetition_id",
                    )
                }
                base.update(
                    session_id=sid,
                    decision_id=did,
                    message_index=message_index,
                    agent_step=embedded.get("step_id"),
                    parent_prefix_sha256=embedded.get("parent_prefix_sha256"),
                )
                if session is None:
                    candidates = embedded.get("candidates") or []
                    count = embedded.get("branch_count")
                    if type(count) is not int or sorted(c.get("branch_id") for c in candidates) != list(
                        range(count)
                    ):
                        raise ValueError(
                            "Unavailable session still requires a complete canonical candidate denominator"
                        )
                    for candidate in candidates:
                        exclusions.append(
                            _exclude({**base, "branch_id": candidate["branch_id"]}, missing_session)
                        )
                    decisions.append(
                        {
                            **base,
                            "branch_count": count,
                            "included_count": 0,
                            "excluded_count": count,
                            "status": missing_session,
                            "selected_branch_id": embedded.get("selected_branch_id"),
                        }
                    )
                    source_summary["decision_count"] += 1
                    source_summary["candidate_count"] += count
                    source_summary["excluded_count"] += count
                    continue
                decision, decision_sha = full_decision(session, embedded)
                if decision.get("selection_mode") not in CONSENSUS_MODES:
                    raise ValueError("Steering agreement labels require a historical consensus decision")
                validate_consensus_evidence(decision, consensus_options(source_cfg["parallel"]))
                candidates = {candidate["branch_id"]: candidate for candidate in decision["candidates"]}
                parsed = {bid: branch_action(candidate) for bid, candidate in candidates.items()}
                selected = decision.get("selected_branch_id")
                if selected is not None and (type(selected) is not int or selected not in candidates):
                    raise ValueError("Historical selected branch is outside the candidate set")
                winner_hash, winner_action, winner_reason = parsed.get(
                    selected, (None, None, "missing_winner_action")
                )
                try:
                    actual_hash, actual_action = canonical_action(message)
                except ActionIdentityUnavailable:
                    actual_hash, actual_action = None, None
                if winner_hash != actual_hash or winner_action != actual_action:
                    raise ValueError("Historical winner action differs from the canonical assistant action")
                consensus = decision.get("consensus") or {}
                if (
                    consensus.get("consensus_action") is not None
                    and consensus["consensus_action"] != winner_action
                ):
                    raise ValueError("Consensus action differs from the actual canonical winner")
                retained = consensus.get("retained_branch_ids", [])
                if len(retained) != len(set(retained)) or not set(retained) <= candidates.keys():
                    raise ValueError("Invalid historical estimator-retained set")
                if selected is not None:
                    native = candidates[selected].get("native_request_id")
                    accepted = session["finalization"].get("accepted_native_request_ids", [])
                    if native is not None and native not in accepted:
                        raise ValueError("Canonical selected request is absent from finalization")
                    response_binding = session["finalization"].get("responses", {}).get(native)
                    if response_binding and response_binding.get("message_index") != message_index:
                        raise ValueError("Finalization response/message index mismatch")
                counts, action_counts = Counter(), Counter()
                for bid, candidate in sorted(candidates.items()):
                    digest, action, reason = parsed[bid]
                    label = label_branch(
                        action_hash=digest,
                        winner_hash=winner_hash,
                        unavailable_reason=reason or (winner_reason if winner_hash is None else None),
                    )
                    row = {
                        **base,
                        **label,
                        "branch_id": bid,
                        "selected": bid == selected,
                        "retained": bid in retained,
                        "score": candidate.get("score"),
                        "estimator_exclusion_reason": candidate.get("exclusion_reason"),
                        "loop_excluded": (candidate.get("loop_escape") or {}).get("excluded"),
                        "action_hash": digest,
                        "action": action,
                        "consensus_action_hash": winner_hash,
                        "consensus_action": winner_action,
                        "native_request_id": candidate.get("native_request_id"),
                        "close_sample_id": candidate.get("close_sample_id"),
                        "decision_sha256": decision_sha,
                        "branch_receipt_sha256": sha256(session["base"] / "parallel/branch-receipt.json"),
                        "classification": session["classification"],
                    }
                    action_counts[label["label"]] += 1
                    event, hidden_reason, method = choose_close(
                        candidate, session["index"].get((did, bid)), decision, session, message_index
                    )
                    if label["label"] == "excluded" or hidden_reason:
                        exclusions.append(
                            _exclude(
                                row,
                                label.get("label_reason") if label["label"] == "excluded" else hidden_reason,
                            )
                        )
                        counts["excluded"] += 1
                        continue
                    if event.get("model_fingerprint") != source["model_fingerprint"]:
                        raise ValueError("Hidden model fingerprint differs from the selected trajectory")
                    inventory = inventory_for(event, model)
                    selected_inventory = select_sites(inventory, sites)
                    space = {
                        "model_fingerprint": source["model_fingerprint"],
                        "tokenizer_fingerprint": event["tokenizer_fingerprint"],
                        "inventory": inventory,
                        "capture_stage": event.get("capture_stage"),
                        "serving_dtype": event.get("serving_dtype"),
                        "intervention": source_cfg.get("steering", {"enabled": False}),
                    }
                    if common_space is None:
                        common_space, common_inventory = space, selected_inventory
                    elif space != common_space:
                        raise ValueError("Incompatible model/tokenizer/site/dtype/steering feature spaces")
                    capture_identity = (str(session["root"].resolve()), sid, did, bid, event["sample_id"])
                    if capture_identity in seen_captures:
                        aliases.append(
                            {
                                "alias": {**base, "branch_id": bid, "sample_id": event["sample_id"]},
                                "canonical_uid": seen_captures[capture_identity],
                            }
                        )
                        counts["aliases"] += 1
                        continue
                    row.update(
                        {
                            k: event.get(k)
                            for k in (
                                "sample_id",
                                "token_id",
                                "phase",
                                "absolute_position",
                                "generation_index",
                                "capture_stage",
                                "serving_dtype",
                                "tokenizer_fingerprint",
                                "model_fingerprint",
                                "graph_execution",
                                "api_attempt_id",
                                "request_sha256",
                                "tensor_locator",
                                "tensor",
                                "sha256",
                            )
                        }
                    )
                    row.update(
                        tensor_root=str(session["root"]),
                        close_binding=method,
                        source_event_sha256=hash_json(event),
                        site_inventory_sha256=hash_json(selected_inventory),
                    )
                    row["uid"] = hash_json([run, rid, sid, did, bid, event["sample_id"]])
                    row["ordinal"] = len(samples)
                    seen_captures[capture_identity] = row["uid"]
                    samples.append(row)
                    counts[label["label"]] += 1
                decisions.append(
                    {
                        **base,
                        "branch_count": decision["branch_count"],
                        "selected_branch_id": selected,
                        "retained_branch_ids": retained,
                        "consensus_action_hash": winner_hash,
                        "consensus_config": decision["consensus_config"],
                        "decision_sha256": decision_sha,
                        "included_count": counts["positive"] + counts["negative"],
                        "excluded_count": counts["excluded"],
                        "alias_count": counts["aliases"],
                        "positive_count": counts["positive"],
                        "negative_count": counts["negative"],
                        "action_label_counts": dict(action_counts),
                        "all_valid_actions_agree": action_counts["positive"] > 0
                        and action_counts["negative"] == 0,
                    }
                )
                source_summary["decision_count"] += 1
                source_summary["candidate_count"] += decision["branch_count"]
                source_summary["included_count"] += counts["positive"] + counts["negative"]
                source_summary["excluded_count"] += counts["excluded"]
            rollouts.append(source_summary)
            print(
                f"dataset: {source_index + 1}/{len(frozen['sources'])} rollouts, "
                f"{len(samples)} included, {len(exclusions)} excluded",
                flush=True,
            )
        if not decisions:
            raise ValueError("Selected trajectories contain no consensus decisions")
        counts = Counter(row["label"] for row in samples)
        excluded_counts = Counter(row["exclusion_reason"] for row in exclusions)
        groups = {}
        for field in ("source_run_id", "task_key", "decision_id", "retained"):
            grouped = defaultdict(Counter)
            for row in samples:
                key = (
                    str(row[field])
                    if field != "decision_id"
                    else "/".join((row["source_run_id"], row["rollout_id"], row[field]))
                )
                grouped[key][row["label"]] += 1
            groups[field] = {key: dict(value) for key, value in sorted(grouped.items())}
        summary = {
            "rollout_count": len(rollouts),
            "decision_count": len(decisions),
            "candidate_count": sum(row["branch_count"] for row in decisions),
            "sample_count": len(samples),
            "excluded_count": len(exclusions),
            "alias_count": len(aliases),
            "class_counts": {"negative": counts["negative"], "positive": counts["positive"]},
            "nonselected_positive_count": sum(row["y"] == 1 and not row["selected"] for row in samples),
            "filtered_positive_count": sum(row["y"] == 1 and not row["retained"] for row in samples),
            "retained_negative_count": sum(row["y"] == 0 and row["retained"] for row in samples),
            "unanimous_decision_count": sum(row.get("all_valid_actions_agree", False) for row in decisions),
            "excluded_reasons": dict(excluded_counts),
            "groups": groups,
        }
        manifest = {
            "schema_version": 1,
            "schema": "consensus-branch-close-v1",
            "dataset_id": dataset_id,
            "label_rule": LABEL_RULE,
            "sample_unit": "one-decision-branch-think-close",
            "candidate_scope": "all",
            "sample_count": len(samples),
            "decision_count": len(decisions),
            "sites": common_inventory or [],
            "space": common_space,
            "selection_sha256": hash_json(frozen["selection"]),
            "sample_order_sha256": hash_json([row["uid"] for row in samples]),
            "storage_root": str(destination.parent.parent.parent),
            "immutable_metadata": True,
            "cache_rebuildable": True,
            "tensors_copied": False,
        }
        write_json(stage / "dataset.json", manifest)
        write_json(stage / "rollouts.json", {"rollouts": rollouts, "capture_aliases": aliases})
        write_jsonl(stage / "samples.jsonl", samples)
        write_jsonl(stage / "excluded.jsonl", exclusions)
        write_jsonl(stage / "decisions.jsonl", decisions)
        write_json(stage / "summary.json", summary)
        write_json(
            stage / "dependencies.json",
            {
                "files": dependencies,
                "note": "Mutable run evaluation evidence and whole publication manifests are snapshot provenance only.",
            },
        )
        verify_dependencies(stage)
        seal(stage, "steering-dataset", dataset_id, hash_json(manifest))
    return {
        "dataset_id": dataset_id,
        "path": str(destination),
        **{k: v for k, v in summary.items() if k != "groups"},
    }


def check_dataset(directory, *, with_tensors=False, check_sources=True):
    directory = Path(directory)
    receipt = verify_local(directory, "steering-dataset", directory.name)
    manifest, summary = read_json(directory / "dataset.json"), read_json(directory / "summary.json")
    if receipt["identity"] != hash_json(manifest) or manifest["dataset_id"] != directory.name:
        raise ValueError("Dataset manifest/receipt identity mismatch")
    samples = list(iter_jsonl(directory / "samples.jsonl"))
    excluded = list(iter_jsonl(directory / "excluded.jsonl"))
    decisions = list(iter_jsonl(directory / "decisions.jsonl"))
    uids = [row["uid"] for row in samples]
    if len(uids) != len(set(uids)) or [row["ordinal"] for row in samples] != list(range(len(samples))):
        raise ValueError("Dataset sample identity/order is invalid")
    if hash_json(uids) != manifest["sample_order_sha256"] or len(samples) != manifest["sample_count"]:
        raise ValueError("Dataset sample order/count mismatch")
    if len(excluded) != summary["excluded_count"] or len(decisions) != manifest["decision_count"]:
        raise ValueError("Dataset excluded/decision denominator mismatch")
    if summary["candidate_count"] != len(samples) + len(excluded) + summary["alias_count"]:
        raise ValueError("All-candidate coverage does not reconcile")
    for row in samples:
        expected = label_branch(action_hash=row["action_hash"], winner_hash=row["consensus_action_hash"])
        if row["label"] != expected["label"] or row["y"] != expected["y"]:
            raise ValueError("Dataset action agreement label mismatch")
    dependency_count = verify_dependencies(directory) if check_sources else None
    if with_tensors:
        from hidden.artifacts import iter_arrays_grouped
        from steering.datasets.features import vectors

        for _key, arrays in iter_arrays_grouped(
            [(i, row["tensor_root"], row) for i, row in enumerate(samples)]
        ):
            vectors(arrays, manifest["sites"])
    return {
        "status": "verified",
        "dataset_id": directory.name,
        "sample_count": len(samples),
        "excluded_count": len(excluded),
        "decision_count": len(decisions),
        "source_files_verified": dependency_count,
        "with_tensors": with_tensors,
        "receipt_sha256": sha256(directory / "receipt.json"),
    }

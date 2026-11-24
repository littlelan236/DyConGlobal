"""Read-only comparison of physical exits and matched final submitted tasks."""

from __future__ import annotations

import re
import math
from collections import Counter
from pathlib import Path

from application.execution.evidence import RunEvidence
from application.execution.run_plan import load_plan
from core.io import hash_json, now, owned_path, read_json, sha256
from enrichment.reader import TrajectoryReader

TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
EXIT_ALIASES = {
    "limitsexceeded": "limits_exceeded",
    "contextlimitexceeded": "context_limit_exceeded",
    "contextlimitexceedederror": "context_limit_exceeded",
    "contextwindowexceeded": "context_limit_exceeded",
    "contextwindowexceedederror": "context_limit_exceeded",
    "reasoningloop": "reasoning_loop",
}
EXIT_CATEGORIES = ("limits_exceeded", "context_limit_exceeded", "reasoning_loop")


def _exit_category(status):
    return EXIT_ALIASES.get(re.sub(r"[^a-z0-9]", "", str(status).lower()))


def _read_optional(path):
    return read_json(path) if path.exists() else {}


def _status(record):
    # Capture/transport failures must not hide an already recorded agent exit.
    return record.get("agent_exit_status") or record.get("exit_status")


def _check_binding(record, run_id, item, attempt_id=None):
    expected = {"run_id": run_id, "task_key": item["task_key"],
                "sample_key": item["sample_key"], "repetition_id": item["repetition_id"],
                "instance_id": item["instance_id"]}
    if attempt_id is not None:
        expected["attempt_id"] = attempt_id
    for key, value in expected.items():
        if key in record and record[key] != value:
            raise ValueError(f"Comparison evidence {key} mismatch: {run_id}/{item['instance_id']}")


def _repetition_selection(loaded, repetitions):
    """Use matching ordinals for whole runs, or one explicitly selected sample per task."""
    repetitions = {} if repetitions is None else dict(repetitions)
    unknown = set(repetitions) - set(loaded)
    if unknown:
        raise ValueError(f"Repetition selectors reference unrequested runs: {sorted(unknown)}")
    counts = {run_id: value["plan"]["repeat_count"] for run_id, value in loaded.items()}
    if not repetitions:
        if len(set(counts.values())) != 1:
            raise ValueError("Runs have different repeat_count values; select one repetition per run "
                             "with --repetition RUN_ID=INDEX (zero-based)")
        return "all_repetitions", {run_id: None for run_id in loaded}
    selected = {}
    for run_id, count in counts.items():
        value = repetitions.get(run_id, 0 if count == 1 else None)
        if value is None:
            raise ValueError(f"Select one repetition for every multi-repetition run: {run_id}")
        if isinstance(value, str) and re.fullmatch(r"(?:rep-)?[0-9]+", value):
            value = int(value.removeprefix("rep-"))
        if type(value) is not int or not 0 <= value < count:
            raise ValueError(f"Invalid repetition for {run_id}: {value!r}; expected index 0..{count - 1}")
        selected[run_id] = f"rep-{value:03d}"
    return "selected_repetition", selected


def _snapshot(root, run_id, loaded, repetition_id):
    if loaded["plan"]["operation"] != "rollout":
        raise ValueError(f"run compare requires a rollout run: {run_id}")
    evidence = RunEvidence(root, run_id).read()
    all_items = {(i["task_key"], i["repetition_id"]): i for i in loaded["work_items"]["items"]}
    items = {key: item for key, item in all_items.items()
             if repetition_id is None or item["repetition_id"] == repetition_id}
    attempts, by_sample = {}, {}
    for rid, row in evidence["rollouts"].items():
        key = (row["task_key"], row.get("repetition_id"))
        if key not in all_items:
            raise ValueError(f"Rollout outside frozen run tasks: {run_id}/{rid}")
        _check_binding(row, run_id, all_items[key], row["attempt_id"])
        if key not in items:
            continue
        physical = (*key, row["attempt_id"])
        if physical in attempts:
            raise ValueError(f"Duplicate canonical physical attempt: {run_id}/{physical}")
        attempts[physical] = row.get("termination_reason")
        by_sample.setdefault(key, {})[rid] = row

    submitted, accepted, accepted_results, selected_ids = {}, {}, {}, {}
    migrated = (loaded["run"] / "migration/source.json").exists()
    for key, item in items.items():
        task_dir = loaded["run"] / "rollout/tasks" / item["instance_id"]
        work = task_dir / "repetitions" / item["repetition_id"]
        # The one-time schema-2 migration retained physical artifact locations.
        # Never read schema-1 archives or infer a final rollout from ordering.
        scopes = [work, task_dir] if migrated and item["repetition_index"] == 0 else [work]
        for scope in scopes:
            for directory in sorted((scope / "attempts").glob("*")):
                if not directory.is_dir() or not re.fullmatch(r"(?:rep-\d+-)?attempt-\d+", directory.name):
                    continue
                aid = directory.name
                context = _read_optional(directory / "attempt-context.json")
                fact = _read_optional(directory / "agent-outcome.json")
                result = _read_optional(directory / "attempt.json")
                for record in (context, fact, result):
                    _check_binding(record, run_id, item, aid)
                if context and not fact and not result and not (directory / "attempt-started.json").exists():
                    continue  # Reserved IDs are not physical attempts yet.
                physical = (*key, aid)
                status = _status(fact) or _status(result) or attempts.get(physical)
                if not status and (directory / "raw.traj.json").exists():
                    status = read_json(directory / "raw.traj.json").get("info", {}).get("exit_status")
                attempts[physical] = status

        final_scopes = [work] if work.exists() and (work / "attempts").exists() else scopes
        cursor = next((scope / "result.json" for scope in final_scopes if (scope / "result.json").exists()), None)
        if cursor is None:
            continue
        result = read_json(cursor)
        _check_binding(result, run_id, item)
        if result.get("attempt_id"):
            physical = (*key, result["attempt_id"])
            if not attempts.get(physical):
                attempts[physical] = _status(result)
        rid = result.get("rollout_id")
        if not rid:
            continue
        row = by_sample.get(key, {}).get(rid)
        if row is None or row["attempt_id"] != result.get("attempt_id"):
            raise ValueError(f"Final result has no matching canonical rollout: {run_id}/{rid}")
        pair_key = (key[0], item["repetition_index"] if repetition_id is None else None)
        selected_ids[pair_key] = rid
        if (result.get("disposition", "accepted") == "accepted"
                and not row.get("provenance", {}).get("quarantined")):
            accepted[pair_key] = row
            accepted_results[pair_key] = result
        if (row["submission_status"] == "submitted" and _status(result) == "Submitted"
                and result.get("disposition", "accepted") == "accepted"
                and not row.get("provenance", {}).get("quarantined")):
            submitted[pair_key] = row

    raw_statuses = Counter(s for s in attempts.values() if s)
    counts = Counter(_exit_category(s) for s in attempts.values())
    physical_count = len(attempts)
    category_counts = {k: counts[k] for k in EXIT_CATEGORIES}
    target_count = sum(category_counts.values())
    exits = {
        "physical_attempt_count": physical_count,
        "known_outcome_count": sum(raw_statuses.values()),
        "unknown_outcome_count": physical_count - sum(raw_statuses.values()),
        "exit_count": target_count,
        "exit_rate": target_count / physical_count if physical_count else None,
        "by_reason": {k: {"count": n, "rate": n / physical_count if physical_count else None}
                      for k, n in category_counts.items()},
        "raw_status_counts": dict(sorted(raw_statuses.items())),
    }
    return {"loaded": loaded, "evidence": evidence, "submitted": submitted, "accepted": accepted,
            "accepted_results": accepted_results,
            "selected_ids": selected_ids, "exits": exits}


def _trajectory_metrics(root, row):
    path = owned_path(root, row["trajectory_path"])
    if sha256(path) != row["trajectory_sha256"]:
        raise ValueError(f"Canonical trajectory changed: {path}")
    reader = TrajectoryReader(path, hydrate_responses=False)
    steps = list(reader.steps())
    responses = list(reader.responses())
    known = dict.fromkeys(TOKEN_FIELDS, 0)
    missing = dict.fromkeys(TOKEN_FIELDS, 0)
    for item in responses:
        usage = (item["response"] or {}).get("usage") or {}
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if usage.get("total_tokens") is None and all(type(n) is int and n >= 0 for n in (prompt, completion)):
            usage = {**usage, "total_tokens": prompt + completion}
        for field in TOKEN_FIELDS:
            value = usage.get(field)
            if type(value) is int and value >= 0:
                known[field] += value
            else:
                missing[field] += 1
    absent = sum(not isinstance(step.get("response"), dict) for step in steps)
    calls = reader.data.get("info", {}).get("model_stats", {}).get("api_calls")
    absent = max(absent, max(0, calls - len(responses)) if type(calls) is int else 0)
    complete = {field: missing[field] == 0 and absent == 0 and bool(responses) for field in TOKEN_FIELDS}
    return {"steps": len(steps),
            **{field: known[field] if complete[field] else None for field in TOKEN_FIELDS},
            "known_tokens": known, "missing_usage_responses": missing,
            "responses_without_evidence": absent, "persisted_responses": len(responses),
            "token_coverage_complete": all(complete.values())}


def _change(value, baseline):
    delta = value - baseline if value is not None and baseline is not None else None
    return {"delta": delta, "relative_change": delta / baseline if delta is not None and baseline else None}


def _comparison_resolution(row, evidence, rollout_id, source):
    """Read the selected evaluation only; never change settled/training labels."""
    settled = row.get("resolved")
    if source == "settled":
        return settled, "settled"
    evaluation = evidence.get("evaluations", {}).get(row.get("selected_evaluation_id"), {})
    outcome = evaluation.get("outcomes", {}).get(rollout_id, {})
    # Official solution booleans take precedence over the raw execution fallback.
    # The latter also represents confirmed invalid/missing patches without a report.
    for name in ("official_result", "raw_outcome"):
        record = outcome.get(name)
        value = record.get("resolved") if isinstance(record, dict) else None
        if type(value) is bool:
            return value, name
    value = outcome.get("resolved")
    if type(value) is bool:
        return value, "evaluation.resolved"
    if type(settled) is bool:
        return settled, "settled_fallback"
    return None, "unknown"


def compare_runs(root, run_ids, *, baseline=None, repetitions=None, resolved_source="settled"):
    """Join task x repetition pairs across explicit runs; write no run files."""
    if resolved_source not in {"settled", "solution"}:
        raise ValueError("resolved_source must be settled or solution")
    root = Path(root).resolve()
    run_ids = list(run_ids)
    if len(run_ids) < 2 or len(run_ids) != len(set(run_ids)):
        raise ValueError("Specify at least two distinct run IDs")
    if baseline is not None and baseline not in run_ids:
        raise ValueError("Baseline must be one of the compared runs")
    loaded = {rid: load_plan(root, rid) for rid in run_ids}
    mode, selected = _repetition_selection(loaded, repetitions)
    snapshots = {rid: _snapshot(root, rid, loaded[rid], selected[rid]) for rid in run_ids}
    common = sorted(set.intersection(*(set(s["submitted"]) for s in snapshots.values())))
    count = len(common)
    task_keys = sorted({key[0] for key in common})
    tasks = [{"task_key": key, **{field: snapshots[run_ids[0]]["evidence"]["tasks"][key][field]
                                 for field in ("benchmark", "revision", "instance_id")}}
             for key in task_keys]
    samples = [{"task_key": key[0], "paired_repetition_index": key[1],
                "repetition_by_run": {rid: snapshots[rid]["submitted"][key]["repetition_id"] for rid in run_ids}}
               for key in common]
    results = []
    for run_id, snapshot in snapshots.items():
        rows = []
        for key in common:
            row = snapshot["submitted"][key]
            resolved_value, value_source = _comparison_resolution(
                row, snapshot["evidence"], snapshot["selected_ids"][key], resolved_source)
            rows.append({"task_key": key[0], "repetition_id": row["repetition_id"],
                         "repetition_index": row["repetition_index"], "paired_repetition_index": key[1],
                         "sample_key": row["sample_key"], "rollout_id": snapshot["selected_ids"][key],
                         "attempt_id": row["attempt_id"], "trajectory_sha256": row["trajectory_sha256"],
                         "evaluation_status": row["evaluation_status"], "resolved": resolved_value,
                         "settled_resolved": row["resolved"], "resolved_value_source": value_source,
                         "selected_evaluation_id": row.get("selected_evaluation_id"),
                         **_trajectory_metrics(root, row)})
        resolved = sum(row["resolved"] is True for row in rows)
        metrics = {"resolved_count": resolved, "resolved_rate": resolved / count if count else None,
                   "unresolved_count": sum(row["resolved"] is False for row in rows),
                   "unknown_evaluation_count": sum(row["resolved"] is None for row in rows),
                   "steps": sum(row["steps"] for row in rows)}
        for field in TOKEN_FIELDS:
            metrics[field] = (sum(row[field] for row in rows)
                              if all(row[field] is not None for row in rows) else None)
        results.append({"run_id": run_id,
                        "repeat_count": snapshot["loaded"]["plan"]["repeat_count"],
                        "selected_repetition_ids": ([selected[run_id]] if selected[run_id] is not None else
                                                     [f"rep-{i:03d}" for i in range(snapshot["loaded"]["plan"]["repeat_count"])]),
                        "submitted_task_count": len({key[0] for key in snapshot["submitted"]}),
                        "submitted_sample_count": len(snapshot["submitted"]),
                        "exits": snapshot["exits"], "common_submitted": metrics,
                        "token_coverage": {"complete": all(r["token_coverage_complete"] for r in rows),
                                           "incomplete_task_count": len({r["task_key"] for r in rows
                                                                         if not r["token_coverage_complete"]}),
                                           "incomplete_sample_count": sum(not r["token_coverage_complete"] for r in rows),
                                           "known_tokens": {k: sum(r["known_tokens"][k] for r in rows)
                                                            for k in TOKEN_FIELDS}},
                        "samples": rows,
                        "source": {"plan_sha256": snapshot["loaded"]["plan"]["plan_sha256"],
                                   "evidence_revision": snapshot["evidence"]["revision"],
                                   "evidence_sha256": hash_json(snapshot["evidence"])}})
    if baseline is not None:
        reference = next(r for r in results if r["run_id"] == baseline)
        for result in results:
            changes = {}
            for scope, fields in (("common_submitted", ("resolved_count", "resolved_rate", "steps", *TOKEN_FIELDS)),
                                  ("exits", ("physical_attempt_count", "exit_count", "exit_rate"))):
                changes[scope] = {k: _change(result[scope][k], reference[scope][k]) for k in fields}
            for rate in (changes["common_submitted"]["resolved_rate"], changes["exits"]["exit_rate"]):
                rate["percentage_point_change"] = rate["delta"] * 100 if rate["delta"] is not None else None
            changes["exits"]["by_reason"] = {
                key: {field: _change(result["exits"]["by_reason"][key][field],
                                     reference["exits"]["by_reason"][key][field]) for field in ("count", "rate")}
                for key in EXIT_CATEGORIES}
            for reason in changes["exits"]["by_reason"].values():
                rate = reason["rate"]
                rate["percentage_point_change"] = rate["delta"] * 100 if rate["delta"] is not None else None
            result["changes_from_baseline"] = changes
    token_savings = _token_savings_concentration(
        {row["run_id"]: row["samples"] for row in results}, run_ids, baseline)
    return {"schema_version": 2, "generated_at": now(), "baseline_run_id": baseline,
            "comparison_mode": mode, "resolved_source": resolved_source,
            "common_submitted_task_count": len(task_keys), "common_submitted_tasks": tasks,
            "common_submitted_sample_count": count, "common_submitted_samples": samples,
            "resolved_rate_denominator": count,
            "resolved_rate_unit": "task_repetition_pair",
            "policy": {
                "exits": "All observed physical attempts within selected repetitions, deduplicated by task/repetition/attempt; "
                         "rate denominator includes started attempts with unknown outcomes, separately reported.",
                "exit_aliases": EXIT_ALIASES,
                "cohort": "Final accepted submitted task/repetition pairs. Equal repeat counts pair matching ordinals by default; "
                          "explicit per-run selections pair each run's chosen repetition by task identity only.",
                "steps": "Canonical assistant messages in the final selected trajectory, including submission.",
                "tokens": "Persisted canonical responses, including selected FormatError responses; "
                          "exclude unselected parallel candidates, previous attempts and auxiliary requests.",
                "resolved_rate": ("Resolved sample count / common submitted pair count; the denominator includes unknown results. "
                                  + ("Use settled row labels. " if resolved_source == "settled" else
                                     "Ignore audit status; use selected evaluation official_result.resolved, then "
                                     "raw_outcome.resolved, then evaluation.resolved, then settled row fallback; "
                                     "only boolean values count, missing results stay unknown. ")
                                  + "Unique task counts are reported separately, without any-success task aggregation."),
                "changes": "Relative change=(value-baseline)/baseline; null for zero baseline or missing data. "
                           "Resolved percentage-point change=100*(resolved-baseline_resolved)/common_pair_count.",
                "missing": "Missing token evidence yields null totals with known subtotals; empty cohort has null rates.",
                "snapshot": "Read-only observation; live runs can change while the command reads evidence."},
            "runs": results,
            "token_savings_concentration": token_savings}


def _reward_value(row, result):
    for source in (result, row):
        value = source.get("raw_reward", source.get("reward"))
        if isinstance(value, dict):
            value = value.get("reward")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def _accepted_sample_metrics(root, row, result, ordinal):
    trajectory = _trajectory_metrics(root, row)
    return {
        "selection_ordinal": ordinal,
        "repetition_id": row["repetition_id"],
        "repetition_index": row["repetition_index"],
        "sample_key": row["sample_key"],
        "rollout_id": result["rollout_id"],
        "attempt_id": row["attempt_id"],
        "trajectory_sha256": row["trajectory_sha256"],
        "reward": _reward_value(row, result),
        **{field: trajectory[field] for field in ("steps", *TOKEN_FIELDS)},
    }


def _aggregate_accepted_metrics(samples):
    metrics = {"sample_count": len(samples), "steps": sum(item["steps"] for item in samples)}
    for field in TOKEN_FIELDS:
        values = [item[field] for item in samples]
        metrics[field] = sum(values) if values and all(value is not None for value in values) else None
    rewards = [item["reward"] for item in samples]
    metrics["reward_sum"] = sum(rewards) if rewards and all(value is not None for value in rewards) else None
    metrics["reward_mean"] = (metrics["reward_sum"] / len(rewards)
                               if metrics["reward_sum"] is not None else None)
    return metrics


def _token_savings_concentration(sample_series, run_ids, baseline):
    """Measure whether a run's token savings versus baseline are concentrated."""
    if baseline is None:
        return {}
    if baseline not in run_ids:
        raise ValueError("Baseline must be one of the compared runs")

    def key(sample):
        return (sample.get("task_key"), sample.get("paired_repetition_index",
                                                       sample.get("selection_ordinal")))

    baseline_samples = {key(sample): sample for sample in sample_series.get(baseline, [])}
    output = {}
    for run_id in run_ids:
        if run_id == baseline:
            output[run_id] = None
            continue
        savings = []
        for sample in sample_series.get(run_id, []):
            base = baseline_samples.get(key(sample))
            if base is None:
                continue
            current_tokens = sample.get("total_tokens")
            baseline_tokens = base.get("total_tokens")
            if isinstance(current_tokens, int) and isinstance(baseline_tokens, int):
                savings.append(max(0, baseline_tokens - current_tokens))
        positive = [value for value in savings if value > 0]
        total = sum(positive)
        ordered = sorted(positive, reverse=True)
        top_count = max(1, math.ceil(len(ordered) * 0.10)) if ordered else 0
        output[run_id] = {
            "baseline_run_id": baseline,
            "trajectory_count": len(savings),
            "positive_saving_trajectory_count": len(positive),
            "total_positive_token_savings": total,
            "top_10_percent_savings_share": (sum(ordered[:top_count]) / total if total else None),
            "largest_trajectory_savings_share": (ordered[0] / total if total else None),
            "hhi": (sum((value / total) ** 2 for value in positive) if total else None),
        }
    return output


def accepted_task_counts(root, run_ids, *, baseline=None, repetitions=None):
    """Count final accepted repetitions for tasks shared by all selected runs."""
    root = Path(root).resolve()
    run_ids = list(run_ids)
    if len(run_ids) < 2 or len(run_ids) != len(set(run_ids)):
        raise ValueError("Specify at least two distinct run IDs")
    if baseline is not None and baseline not in run_ids:
        raise ValueError("Baseline must be one of the compared runs")
    loaded = {rid: load_plan(root, rid) for rid in run_ids}
    mode, selected = _repetition_selection(loaded, repetitions)
    snapshots = {rid: _snapshot(root, rid, loaded[rid], selected[rid]) for rid in run_ids}

    inventories = {}
    for run_id in run_ids:
        inventory = {}
        for item in loaded[run_id]["work_items"]["items"]:
            inventory.setdefault(item["task_key"], {
                key: item.get(key) for key in ("benchmark", "revision", "instance_id")
            })
        inventories[run_id] = inventory
    common_keys = sorted(set.intersection(*(set(value) for value in inventories.values())))
    # Align the accepted cohort by count per task. Repetition IDs need not match
    # across runs; each run contributes its first minimum-count accepted samples.
    accepted_by_task = {}
    for task_key in common_keys:
        per_run = {
            run_id: sorted((key for key in snapshots[run_id]["accepted"] if key[0] == task_key),
                           key=lambda pair: pair[1] if pair[1] is not None else -1)
            for run_id in run_ids
        }
        minimum = min(len(keys) for keys in per_run.values())
        accepted_by_task[task_key] = {run_id: keys[:minimum] for run_id, keys in per_run.items()}

    rows = []
    for task_key in common_keys:
        metadata = next((snapshots[run_id]["evidence"]["tasks"].get(task_key)
                         for run_id in run_ids
                         if task_key in snapshots[run_id]["evidence"].get("tasks", {})), None)
        metadata = metadata or next(inventories[run_id][task_key] for run_id in run_ids)
        accepted_by_run = {
            run_id: sum(1 for key in snapshots[run_id]["accepted"] if key[0] == task_key)
            for run_id in run_ids
        }
        rows.append({
            "task_key": task_key,
            "benchmark": metadata.get("benchmark"),
            "revision": metadata.get("revision"),
            "instance_id": metadata.get("instance_id"),
            "accepted_by_run": accepted_by_run,
            "common_accepted_samples_by_run": {
                run_id: [
                    _accepted_sample_metrics(root, snapshots[run_id]["accepted"][pair],
                                              snapshots[run_id]["accepted_results"][pair], ordinal)
                    for ordinal, pair in enumerate(accepted_by_task[task_key][run_id])
                ]
                for run_id in run_ids
            },
            "minimum_accepted_count": min(accepted_by_run.values()),
            "common_accepted_count": min(accepted_by_run.values()),
            "common_accepted_metrics_by_run": {
                run_id: _aggregate_accepted_metrics([
                    _accepted_sample_metrics(root, snapshots[run_id]["accepted"][pair],
                                              snapshots[run_id]["accepted_results"][pair], ordinal)
                    for ordinal, pair in enumerate(accepted_by_task[task_key][run_id])
                ])
                for run_id in run_ids
            },
        })
    rows.sort(key=lambda row: (-row["minimum_accepted_count"],
                               row["instance_id"] or "", row["task_key"]))

    run_summaries = []
    for run_id in run_ids:
        accepted_counts = Counter(key[0] for key in snapshots[run_id]["accepted"])
        run_summaries.append({
            "run_id": run_id,
            "repeat_count": loaded[run_id]["plan"]["repeat_count"],
            "selected_repetition_ids": ([selected[run_id]] if selected[run_id] is not None
                                         else [f"rep-{i:03d}" for i in range(loaded[run_id]["plan"]["repeat_count"])]),
            "accepted_task_count": sum(value > 0 for value in accepted_counts.values()),
            "accepted_sample_count": sum(accepted_counts.values()),
        })
    if baseline is not None:
        reference = next(summary for summary in run_summaries if summary["run_id"] == baseline)
        for summary in run_summaries:
            summary["changes_from_baseline"] = {
                field: _change(summary[field], reference[field])
                for field in ("accepted_task_count", "accepted_sample_count")
            }
    sample_series = {
        run_id: [
            {"task_key": pair[0], **_accepted_sample_metrics(
                root, snapshots[run_id]["accepted"][pair],
                snapshots[run_id]["accepted_results"][pair], ordinal)}
            for ordinal, pair in enumerate(
                (pair for task_key in common_keys for pair in accepted_by_task[task_key][run_id])
            )
        ]
        for run_id in run_ids
    }
    token_savings = _token_savings_concentration(sample_series, run_ids, baseline)
    common_metrics = {
        run_id: _aggregate_accepted_metrics(sample_series[run_id])
        for run_id in run_ids
    }
    if baseline is not None:
        reference = common_metrics[baseline]
        for run_id in run_ids:
            common_metrics[run_id]["changes_from_baseline"] = {
                field: _change(common_metrics[run_id][field], reference[field])
                for field in ("sample_count", "steps", *TOKEN_FIELDS, "reward_sum", "reward_mean")
            }
        for row in rows:
            reference_count = row["accepted_by_run"][baseline]
            reference_metrics = row["common_accepted_metrics_by_run"][baseline]
            for run_id in run_ids:
                metrics = row["common_accepted_metrics_by_run"][run_id]
                metrics["changes_from_baseline"] = {
                    field: _change(metrics[field], reference_metrics[field])
                    for field in ("sample_count", "steps", *TOKEN_FIELDS, "reward_sum", "reward_mean")
                }
            row["changes_from_baseline"] = {
                run_id: {
                    "accepted_count": _change(row["accepted_by_run"][run_id], reference_count),
                    "metrics": row["common_accepted_metrics_by_run"][run_id].get("changes_from_baseline", {}),
                }
                for run_id in run_ids
            }
    return {
        "schema_version": 1,
        "generated_at": now(),
        "comparison_mode": mode,
        "baseline_run_id": baseline,
        "run_ids": run_ids,
        "runs": run_summaries,
        "common_task_count": len(rows),
        "common_accepted_task_count": sum(row["minimum_accepted_count"] > 0 for row in rows),
        "common_accepted_count": sum(row["minimum_accepted_count"] for row in rows),
        "common_accepted_sample_task_count": sum(row["common_accepted_count"] > 0 for row in rows),
        "common_accepted_sample_count": sum(row["common_accepted_count"] for row in rows),
        "common_accepted_metrics_by_run": common_metrics,
        "common_accepted_samples_by_run": {
            run_id: [
                {"task_key": pair[0], **_accepted_sample_metrics(
                    root, snapshots[run_id]["accepted"][pair],
                    snapshots[run_id]["accepted_results"][pair], ordinal)}
                for ordinal, pair in enumerate(
                    (pair for task_key in common_keys for pair in accepted_by_task[task_key][run_id])
                )
            ]
            for run_id in run_ids
        },
        "tasks": rows,
        "token_savings_concentration": token_savings,
        "policy": {
            "accepted": "Final result pointers with disposition=accepted and no quarantine, counted once per selected repetition.",
            "task_identity": "benchmark + revision + instance_id (task_key)",
            "common_tasks": "Only task identities present in every selected run are listed; zero minima are retained.",
            "minimum": "Per-task minimum accepted repetition count across all compared runs.",
            "paired_metrics": "Steps, persisted prompt/completion/total tokens and raw reward use the per-task minimum accepted count from each run; repetition IDs need not match.",
            "baseline": "Relative change=(value-baseline)/baseline; null for zero baseline or missing data.",
            "steps_tokens": "Canonical assistant steps and selected-response usage; unselected parallel candidate branches are excluded.",
            "snapshot": "Read-only observation of frozen plans and run evidence.",
        },
    }


def format_accepted_markdown(report):
    """Render the accepted-by-task report as a stable, scan-friendly table."""
    runs = report["run_ids"]
    lines = [
        f"Common accepted tasks (minimum > 0): {report['common_accepted_task_count']}",
        f"Common accepted count (sum of per-task minima): {report['common_accepted_count']}",
        f"Common accepted tasks with metrics: {report['common_accepted_sample_task_count']}",
        f"Common accepted trajectories: {report['common_accepted_sample_count']}",
        f"Shared task identities: {report['common_task_count']}",
        f"Repetition mode: {report['comparison_mode']}",
    ]
    if report.get("baseline_run_id"):
        lines.append(f"Baseline: {report['baseline_run_id']}; relative changes are shown in JSON.")
    for summary in report["runs"]:
        lines.append(f"- {summary['run_id']}: repeat_count={summary['repeat_count']}; "
                     f"accepted_tasks={summary['accepted_task_count']}; "
                     f"accepted_samples={summary['accepted_sample_count']}; "
                     f"selected={','.join(summary['selected_repetition_ids'])}")
    def compact_metrics(metrics):
        def value(item):
            return "N/A" if item is None else f"{item:g}"
        return (f"n={metrics['sample_count']}; steps={metrics['steps']}; "
                f"tokens={value(metrics['total_tokens'])}; reward={value(metrics['reward_sum'])}/"
                f"{value(metrics['reward_mean'])}")

    lines += ["", "Trajectory metrics use the per-task minimum accepted count; repetition IDs may differ across runs.",
              "| Run | Samples | Steps | Prompt tokens | Completion tokens | Total tokens | Reward sum | Reward mean |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for run_id in runs:
        metrics = report["common_accepted_metrics_by_run"][run_id]
        lines.append("| " + run_id + " | " + " | ".join(
            str(metrics[key]) if metrics[key] is not None else "N/A"
            for key in ("sample_count", "steps", "prompt_tokens", "completion_tokens", "total_tokens",
                        "reward_sum", "reward_mean")) + " |")
    if report.get("baseline_run_id"):
        def change_percent(value):
            relative = value.get("relative_change") if value else None
            return "N/A" if relative is None else f"{relative:+.2%}"
        lines += ["", "| Run | Accepted task change | Accepted sample change | Steps change | Prompt token change | Completion token change | Total token change | Reward sum change | Reward mean change |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for summary in report["runs"]:
            accepted_changes = summary.get("changes_from_baseline", {})
            metric_changes = report["common_accepted_metrics_by_run"][summary["run_id"]].get("changes_from_baseline", {})
            cells = [change_percent(accepted_changes.get(field)) for field in ("accepted_task_count", "accepted_sample_count")]
            cells += [change_percent(metric_changes.get(field)) for field in ("steps", "prompt_tokens", "completion_tokens", "total_tokens", "reward_sum", "reward_mean")]
            lines.append("| " + summary["run_id"] + " | " + " | ".join(cells) + " |")
    if report.get("token_savings_concentration"):
        lines += ["", "| Run | Positive saving trajectories | Total positive token savings | Top 10% savings share | Largest trajectory share | HHI |",
                  "|---|---:|---:|---:|---:|---:|"]
        for run_id, metrics in report["token_savings_concentration"].items():
            if metrics is None:
                continue
            lines.append("| " + run_id + " | " + " | ".join(
                str(metrics[key]) if metrics[key] is not None else "N/A"
                for key in ("positive_saving_trajectory_count", "total_positive_token_savings",
                            "top_10_percent_savings_share", "largest_trajectory_savings_share", "hhi")) + " |")
    lines += ["", "| Task | " + " | ".join(f"Accepted / common metrics: {run_id}" for run_id in runs)
              + " | Minimum accepted | Common accepted |", "|---|" + "---:|" * (len(runs) + 2)]
    for row in report["tasks"]:
        task = row["instance_id"] or row["task_key"]
        counts = " | ".join(
            f"{row['accepted_by_run'][run_id]} / {compact_metrics(row['common_accepted_metrics_by_run'][run_id])}"
            for run_id in runs)
        lines.append(f"| {task} | {counts} | {row['minimum_accepted_count']} | {row['common_accepted_count']} |")
    lines += ["", "Rows are shared task identities, sorted by minimum accepted count descending. "
              "Zero minima are retained for auditability. Each selected trajectory record is also available in JSON."]
    return "\n".join(lines)

def format_markdown(report):
    def number(value):
        return f"{value:,}" if value is not None else "N/A"

    def percent(value):
        return f"{value:.2%}" if value is not None else "N/A"

    def change(row, scope, field):
        if "changes_from_baseline" not in row:
            return ""
        value = row["changes_from_baseline"][scope][field]
        if field.endswith("rate"):
            pp = value["percentage_point_change"]
            return f" ({pp:+.2f} pp)" if pp is not None else " (N/A)"
        relative = value["relative_change"]
        return f" ({relative:+.2%})" if relative is not None else " (N/A)"

    lines = [f"Common submitted tasks (unique): {report['common_submitted_task_count']}",
             f"Common submitted task/repetition pairs: {report['common_submitted_sample_count']} "
             "(resolved rate denominator)",
             f"Repetition mode: {report['comparison_mode']}",
             f"Resolved source: {report.get('resolved_source', 'settled')} (same common submitted denominator)"]
    if report.get("resolved_source") == "solution":
        lines.append("Original solution booleans ignore audit status; settled labels and training eligibility are unchanged.")
    for row in report["runs"]:
        lines.append(f"- {row['run_id']}: repeat_count={row['repeat_count']}; "
                     f"selected={','.join(row['selected_repetition_ids'])}")
    if report["baseline_run_id"]:
        lines.append(f"Baseline: {report['baseline_run_id']}; changes in parentheses; rate changes in percentage points.")
    lines += ["", "| Run | Physical attempts | Limit/loop exits | Exit rate | Unknown outcomes |",
              "|---|---:|---:|---:|---:|"]
    for row in report["runs"]:
        ex = row["exits"]
        lines.append(f"| {row['run_id']} | {number(ex['physical_attempt_count'])}"
                     f"{change(row, 'exits', 'physical_attempt_count')} | {number(ex['exit_count'])}"
                     f"{change(row, 'exits', 'exit_count')} | {percent(ex['exit_rate'])}"
                     f"{change(row, 'exits', 'exit_rate')} | {ex['unknown_outcome_count']} |")
    lines += ["", "| Run | LimitsExceeded count / rate | Context limit count / rate | Reasoning loop count / rate |",
              "|---|---:|---:|---:|"]
    for row in report["runs"]:
        values = []
        for key, value in row["exits"]["by_reason"].items():
            count_change, rate_change = "", ""
            if "changes_from_baseline" in row:
                reason = row["changes_from_baseline"]["exits"]["by_reason"][key]
                relative = reason["count"]["relative_change"]
                pp = reason["rate"]["percentage_point_change"]
                count_change = f" ({relative:+.2%})" if relative is not None else " (N/A)"
                rate_change = f" ({pp:+.2f} pp)" if pp is not None else " (N/A)"
            values.append(f"{value['count']}{count_change} / {percent(value['rate'])}{rate_change}")
        lines.append("| " + row["run_id"] + " | " + " | ".join(values) + " |")
    lines += ["", "| Run | Resolved | Resolved rate | Steps | Prompt tokens | Completion tokens | Total tokens | Unknown evaluations |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["runs"]:
        m = row["common_submitted"]
        cells = [(percent(m[k]) if k == "resolved_rate" else number(m[k])) + change(row, "common_submitted", k)
                 for k in ("resolved_count", "resolved_rate", "steps", *TOKEN_FIELDS)]
        lines.append("| " + row["run_id"] + " | " + " | ".join(cells) + f" | {m['unknown_evaluation_count']} |")
    if report.get("token_savings_concentration"):
        lines += ["", "| Run | Positive saving trajectories | Total positive token savings | Top 10% savings share | Largest trajectory share | HHI |",
                  "|---|---:|---:|---:|---:|---:|"]
        for run_id, metrics in report["token_savings_concentration"].items():
            if metrics is None:
                continue
            lines.append("| " + run_id + " | " + " | ".join(
                str(metrics[key]) if metrics[key] is not None else "N/A"
                for key in ("positive_saving_trajectory_count", "total_positive_token_savings",
                            "top_10_percent_savings_share", "largest_trajectory_savings_share", "hhi")) + " |")
    lines += ["", "Exit rates use all physical attempts within selected repetitions. "
              "Resolved/step/token metrics use only common final submitted task/repetition pairs.",
              "Tokens count selected canonical responses only. N/A means missing evidence, an empty denominator, or a zero baseline."]
    return "\n".join(lines)

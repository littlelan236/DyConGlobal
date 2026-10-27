"""Immutable per-decision exact-action diversity and actual-winner support."""

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from action.analyzers.trajectories import verify_frozen_sources
from action.datasets.trajectories import read_messages, select_trajectories
from core.action_identity import ActionIdentityUnavailable, canonical_action
from core.io import artifact_dir, hash_json, identifier, now, owned_path, read_json, sha256, write_json
from core.locking import exclusive_artifact

LEGACY_SCHEMA = "exact-action-branch-statistics-v1"
SCHEMA = "exact-action-branch-statistics-v2"
BASE_METRICS = ("action_types", "winner_count")
ENTROPY_METRICS = ("entropy", "normalized_entropy", "effective_action_count",
                   "winner_support", "winner_uncertainty", "winner_surprisal")
CODE_FILES = (
    "action/analyzers/branches.py", "action/evaluation/branch_histograms.py",
    "action/evaluation/branch_curves.py",
    "action/datasets/trajectories.py", "action/analyzers/trajectories.py",
    "difficulty/datasets/selection.py", "application/execution/evidence.py",
    "core/action_identity.py", "core/io.py", "core/locking.py", "core/cli.py",
)


def _full_decision(root, config, run, embedded):
    """Use the source run's frozen spool/NAS locators, never current capture config."""
    session, decision = identifier(embedded["session_id"]), identifier(embedded["decision_id"])
    export = config.get("capture", {}).get("export", {})
    roots = []
    if export.get("spool_root"):
        roots.append(Path(root) / export["spool_root"] / run)
    if export.get("root") and config.get("paths", {}).get("nas_root"):
        roots.append(Path(config["paths"]["nas_root"]) / export["root"] / run)
    for directory in roots:
        path = owned_path(directory, f"sessions/{session}/parallel/decisions/{decision}.json")
        if path.is_file():
            raw = path.read_bytes()
            full = json.loads(raw)
            # The embedded summary is the canonical step's binding to the full receipt.
            for key, value in embedded.items():
                expected = full.get(key)
                if key == "candidates":
                    expected = [{k: v for k, v in b.items() if k != "response"}
                                for b in full.get("candidates", [])]
                    value = [{k: v for k, v in b.items() if k != "response"} for b in value]
                if expected != value:
                    raise ValueError(f"Embedded/full decision mismatch: {decision}/{key}")
            import hashlib
            return full, {"kind": "full", "path": str(path),
                          "sha256": hashlib.sha256(raw).hexdigest()}
    return embedded, {"kind": "embedded", "sha256": hash_json(embedded)}


def _action(branch):
    response = branch.get("response")
    if response is not None:
        choices = response.get("choices") or []
        message = choices[0].get("message") if choices else None
        try:
            _, action = canonical_action(message)
        except ActionIdentityUnavailable as exc:
            if branch.get("action") is not None:
                raise ValueError("Saved action exists but raw response has no comparable action") from exc
            return None, str(exc)
        if branch.get("action") is not None and branch["action"] != action:
            raise ValueError("Saved action differs from raw response")
        return action, None
    if branch.get("action") is not None:
        # Revalidate compact canonical action evidence rather than treating arbitrary text as a vote.
        actions = json.loads(branch["action"])
        _, action = canonical_action({"tool_calls": [{"function": entry} for entry in actions]})
        if action != branch["action"]:
            raise ValueError("Saved action is not canonical")
        return action, None
    # A compact summary strips raw responses, including filtered-out actions.
    return None, "raw_response_and_action_unavailable"


def decision_record(embedded, decision, message, scope, binding, proof):
    candidates = decision.get("candidates")
    count = decision.get("branch_count")
    if not isinstance(candidates, list) or type(count) is not int or count < 1:
        raise ValueError("Decision has no complete branch inventory")
    ids = [b.get("branch_id") for b in candidates]
    if any(type(b) is not int for b in ids) or sorted(ids) != list(range(count)):
        raise ValueError("Decision branch IDs must cover 0..branch_count-1 exactly once")
    consensus = decision.get("consensus")
    retained = None
    if isinstance(consensus, dict) and "retained_branch_ids" in consensus:
        retained = consensus["retained_branch_ids"]
        if (not isinstance(retained, list) or any(type(b) is not int for b in retained)
                or len(set(retained)) != len(retained) or not set(retained) <= set(ids)):
            raise ValueError("Invalid recorded estimator-retained branch IDs")
        if consensus.get("retained_count") != len(retained):
            raise ValueError("Retained count differs from recorded IDs")
        for b in candidates:
            if "retained" in b and b["retained"] != (b["branch_id"] in retained):
                raise ValueError("Branch retained flag differs from consensus receipt")
    if scope != "all" and retained is None:
        raise ValueError("No recorded estimator-retained set; use --branch-scope all for this run")
    actions, unavailable = {}, {}
    needed = set(ids if scope in {"all", "both"} else retained)
    selected = decision.get("selected_branch_id")
    if type(selected) is not int or selected not in ids:
        raise ValueError("Canonical parallel step has no selected branch")
    for b in candidates:
        action, reason = _action(b)
        bid = b["branch_id"]
        if reason == "raw_response_and_action_unavailable" and bid in needed | {selected}:
            # Full records distinguish a failed/no-output branch from lost compact evidence.
            if proof["kind"] == "full" or "response" in b:
                reason = "missing_response"
            else:
                raise ValueError(f"Incomplete action evidence for branch {bid}; restore full decision receipt")
        if action is not None:
            actions[str(bid)] = action
        else:
            unavailable[str(bid)] = reason
    try:
        _, actual = canonical_action(message)
    except ActionIdentityUnavailable:
        actual = None
    winner = actions.get(str(selected))
    if winner != actual:
        raise ValueError("Selected branch action differs from canonical assistant message")
    if isinstance(consensus, dict):
        if consensus.get("consensus_action") is not None and consensus["consensus_action"] != winner:
            raise ValueError("Consensus action differs from canonical winner")
        if consensus.get("votes"):
            votes = consensus["votes"]
            if len({v["action"] for v in votes}) != len(votes):
                raise ValueError("Duplicate action vote group")
            counts = Counter(actions[str(b)] for b in retained if str(b) in actions)
            if counts != Counter({v["action"]: v["count"] for v in votes}):
                raise ValueError("Recorded consensus votes differ from branch actions")
    return {**binding, "decision_id": identifier(embedded["decision_id"]),
            "session_id": identifier(embedded["session_id"]), "step_id": embedded.get("step_id"),
            "branch_count": count, "retained_branch_ids": retained,
            "selected_branch_id": selected, "actions": actions, "unavailable": unavailable,
            "consensus_status": (consensus or {}).get("status"), "proof": proof}


def step_statistics(record, scope, *, legacy=False):
    actions = record["actions"]
    row = {k: record[k] for k in ("source_run_id", "rollout_id", "instance_id", "cohort",
                                 "assistant_index", "message_index", "step_id", "decision_id",
                                 "session_id", "branch_count", "selected_branch_id", "consensus_status")}
    winner = actions.get(str(record["selected_branch_id"]))
    for view in ("all", "retained"):
        if scope != "both" and scope != view:
            continue
        ids = list(range(record["branch_count"])) if view == "all" else record["retained_branch_ids"]
        counts = Counter(actions[str(b)] for b in ids if str(b) in actions)
        valid = sum(counts.values())
        k = len(counts)
        entropy = -math.fsum((n / valid) * math.log(n / valid) for n in counts.values()) if valid else None
        normalized = (entropy / math.log(k) if k > 1 else 0.0) if valid else None
        effective = math.exp(entropy) if entropy is not None else None
        winner_count = counts.get(winner, 0) if winner is not None else None
        support = winner_count / valid if winner_count is not None and valid else None
        winner_uncertainty = (1.0 - winner_count / valid) if winner_count is not None and valid else None
        winner_surprisal = (-math.log(winner_count / valid)
                            if winner_count is not None and winner_count > 0 and valid else None)
        row.update({f"branches_{view}": len(ids), f"valid_actions_{view}": valid,
                    f"unavailable_{view}": len(ids) - sum(counts.values()),
                    f"action_types_{view}": len(counts),
                    f"winner_count_{view}": winner_count,
                    f"group_sizes_{view}": sorted(counts.values(), reverse=True)})
        if not legacy:
            row.update({f"entropy_{view}": entropy,
                    f"normalized_entropy_{view}": normalized,
                    f"effective_action_count_{view}": effective,
                    f"winner_support_{view}": support,
                    f"winner_uncertainty_{view}": winner_uncertainty,
                    f"winner_surprisal_{view}": winner_surprisal,
                    f"winner_surprisal_status_{view}": ("unavailable" if support is None else
                                                        "positive_infinity" if support == 0 else "finite")})
    if scope == "both" and not legacy:
        for metric in ENTROPY_METRICS:
            a, b = row.get(f"{metric}_all"), row.get(f"{metric}_retained")
            row[f"delta_{metric}"] = b - a if a is not None and b is not None else None
    return row


def metric_summary(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    counts = Counter(values)
    view = key.rsplit("_", 1)[1]
    integer_histogram = key.startswith(("action_types_", "winner_count_"))
    upper = max((r[f"branches_{view}"] for r in rows), default=0) if integer_histogram else None
    result = {"steps": len(rows), "valid_steps": len(values), "unavailable_steps": len(rows) - len(values),
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "bin_upper": upper,
            "histogram": ({str(i): counts.get(i, 0) for i in range(upper + 1)}
                          if upper is not None else {})}
    if key.startswith("winner_surprisal_"):
        result["positive_infinity_steps"] = sum(r.get(f"winner_surprisal_status_{view}") == "positive_infinity"
                                                for r in rows)
    return result


def summarize(rows, trajectories, scope, *, legacy=False):
    views = ("all", "retained") if scope == "both" else (scope,)
    metrics = BASE_METRICS if legacy else BASE_METRICS + ENTROPY_METRICS
    keys = [f"{metric}_{view}" for metric in metrics for view in views]
    result = {"schema": LEGACY_SCHEMA if legacy else SCHEMA,
              "step_weighting": "one canonical parallel assistant step, equal weight",
              "steps": len(rows), "trajectories": len(trajectories), "branch_scope": scope,
              "metrics": {k: metric_summary(rows, k) for k in keys},
              "branch_coverage": {view: {
                  "outputs": sum(r[f"branches_{view}"] for r in rows),
                  "valid_actions": sum(r[f"valid_actions_{view}"] for r in rows),
                  "unavailable": sum(r[f"unavailable_{view}"] for r in rows),
                  "steps_with_unavailable": sum(r[f"unavailable_{view}"] > 0 for r in rows),
                  "branch_count_histogram": {str(k): v for k, v in sorted(Counter(r[f"branches_{view}"] for r in rows).items())},
              } for view in views}}
    for field, name in (("source_run_id", "by_source_run"), ("cohort", "by_cohort")):
        groups = sorted({r[field] for r in rows if r[field] is not None})
        group_keys = keys + (["delta_" + metric for metric in ENTROPY_METRICS]
                             if scope == "both" and not legacy else [])
        result[name] = {g: {k: metric_summary([r for r in rows if r[field] == g], k) for k in group_keys}
                        for g in groups}
    result["trajectories_without_parallel_steps"] = sum(t["parallel_steps"] == 0 for t in trajectories)
    if scope == "both" and not legacy:
        result["filter_effects"] = {metric: metric_summary(rows, "delta_" + metric)
                                     for metric in ENTROPY_METRICS}
    return result


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


@exclusive_artifact("action", "analysis_id")
def analyze_branches(cfg, analysis_id, run_id=None, rollout_ids=(), rollout_list=None,
                     scope="both", termination_reason=None, formats=("png", "svg"), dpi=180,
                     line_mode="both", progress_points=101, line_metrics=()):
    if scope not in {"all", "retained", "both"}:
        raise ValueError("Invalid branch scope")
    if not formats or set(formats) - {"png", "svg"} or not 50 <= dpi <= 600:
        raise ValueError("Use png/svg formats and dpi in [50, 600]")
    if line_mode not in {"per-run", "all-runs", "both", "none", "per-trajectory"}:
        raise ValueError("Invalid line mode")
    if type(progress_points) is not int or not 2 <= progress_points <= 1001:
        raise ValueError("progress-points must be an integer in [2, 1001]")
    root = Path(cfg["project_root"])
    out = artifact_dir(cfg, "action", analysis_id)
    if out.exists():
        raise FileExistsError(f"Use a new analysis ID: {out}")
    implementation = Path(__file__).resolve().parents[2]
    before = {f: (implementation / f).read_bytes() for f in CODE_FILES}
    selection = select_trajectories(cfg, run_id, rollout_ids, rollout_list)
    snapshots = selection.pop("source_snapshots")
    configs = {}
    for run in selection["source_runs"]:
        locator = run["config_source"]
        if locator is None:
            raise ValueError("Source run needs its frozen resolved-config.json")
        path = owned_path(root, locator["path"])
        if sha256(path) != locator["sha256"]:
            raise ValueError("Frozen source config changed during selection")
        configs[run["run_id"]] = read_json(path)
    rows, records, trajectories, excluded = [], [], [], []
    seen = set()
    for source in selection["sources"]:
        if termination_reason is not None and source["termination_reason"] != termination_reason:
            excluded.append({"source_run_id": source["source_run_id"], "rollout_id": source["rollout_id"],
                             "termination_reason": source["termination_reason"]})
            continue
        messages = read_messages(root, source)
        total_assistant_steps = sum(m.get("role") == "assistant" for m in messages)
        assistant_index, steps = 0, 0
        for mi, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            index = assistant_index
            assistant_index += 1
            embedded = message.get("extra", {}).get("dycon_parallel")
            if not embedded:
                continue
            run = source["source_run_id"]
            key = (run, embedded["session_id"], embedded["decision_id"])
            if key in seen:
                raise ValueError("Duplicate canonical parallel decision")
            seen.add(key)
            decision, proof = _full_decision(root, configs[run], run, embedded)
            binding = {k: source[k] for k in ("source_run_id", "rollout_id", "instance_id", "cohort")}
            record = decision_record(embedded, decision, message, scope,
                                     {**binding, "assistant_index": index, "message_index": mi}, proof)
            records.append(record)
            row = step_statistics(record, scope)
            row.update({"trajectory_assistant_steps": total_assistant_steps,
                        "parallel_index": steps,
                        "normalized_progress": index / (total_assistant_steps - 1)
                        if total_assistant_steps > 1 else 0.0})
            rows.append(row)
            steps += 1
        trajectories.append({**source, "assistant_steps": assistant_index, "parallel_steps": steps,
                             "nonparallel_assistant_steps": assistant_index - steps})
    if not rows:
        raise ValueError("Selected trajectories contain no canonical parallel steps after filtering")
    from action.evaluation.branch_curves import resolve_line_metrics
    resolved_metrics = resolve_line_metrics(rows, line_mode, line_metrics)
    summary = summarize(rows, trajectories, scope)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".branch-stats-", dir=out.parent) as temporary:
        stage = Path(temporary)
        for run in selection["source_runs"]:
            write_json(stage / run["snapshot_path"], snapshots[run["run_id"]])
            run["snapshot_sha256"] = sha256(stage / run["snapshot_path"])
            write_json(stage / "source-config" / (run["run_id"] + ".json"), configs[run["run_id"]])
        for name, content in before.items():
            path = stage / "code" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        analysis = {"schema": SCHEMA, "analysis_id": analysis_id, "created_at": now(),
                    "source_runs": selection["source_runs"], "branch_scope": scope,
                    "termination_reason": termination_reason, "excluded_trajectories": excluded,
                    "formats": list(dict.fromkeys(formats)), "dpi": dpi,
                    "line_mode": line_mode, "progress_points": progress_points,
                    "line_metrics": resolved_metrics,
                    "curve_schema_version": 4,
                    "plot_unit": "source_run_id + rollout_id; no task grouping",
                    "statistics": {
                        "log_base": "e (nats)", "probabilities": "action group count / valid action branches",
                        "entropy": "H=-sum(p_i*ln(p_i)); null if N=0",
                        "normalized_entropy": "H/ln(K) if K>1; 0 if K=1; null if N=0; measures evenness",
                        "effective_action_count": "E=exp(H)", "winner_support": "q=n_winner/N",
                        "winner_uncertainty": "U=1-q", "winner_surprisal": "I=-ln(q)",
                        "zero_support": "q=0: I=null with status=positive_infinity; excluded from finite means",
                        "filter_effect": "delta_metric=retained-all; both scopes required",
                    },
                    "curve_aggregation": {
                        "per_run": "normalized progress on 0..1; interpolate per trajectory then equal trajectory weight",
                        "per_trajectory": "one figure per source_run_id + rollout_id; raw assistant steps; no averaging or interpolation",
                        "normalized": "x=assistant_index/(total_assistant_steps-1); singleton x=0",
                        "interpolation": "per-trajectory linear interpolation on an inclusive 0..1 grid; no extrapolation or bridging missing values",
                        "across_trajectories": "equal weight per trajectory at each grid point, separately within each source run",
                        "all_runs": "one figure comparing all explicitly selected source runs; never discover or select other runs",
                    },
                    "code": {f: sha256(stage / "code" / f) for f in before},
                    "action_unit": "one ordered tool-call sequence per branch; core.action_identity.canonical_action",
                    "retention": "historical recorded retained_branch_ids; no rescoring or threshold simulation",
                    "winner": "actual canonical selected action; never the largest all-branch vote group",
                    "missing": "no-action branches excluded from action types, not a synthetic extra action; missing winner is null"}
        write_json(stage / "analysis.json", analysis)
        write_json(stage / "selection.json", selection)
        write_json(stage / "summary.json", summary)
        write_json(stage / "trajectories.json", trajectories)
        for filename, entries in (("steps.jsonl", rows), ("decisions.jsonl", records)):
            with (stage / filename).open("w", encoding="utf-8") as stream:
                for row in entries:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        write_csv(stage / "steps.csv", rows)
        write_csv(stage / "trajectories.csv", trajectories)
        bins = [{"metric": k, "value": int(v), "step_count": n,
                 "step_fraction": n / s["valid_steps"] if s["valid_steps"] else None}
                for k, s in summary["metrics"].items() for v, n in s["histogram"].items()]
        write_csv(stage / "histogram_counts.csv", bins)
        from action.evaluation.branch_histograms import render_histograms
        render_histograms(stage / "plots", summary, analysis)
        from action.evaluation.branch_curves import build_curve_tables, build_trajectory_curves, render_line_plots
        raw_curves, normalized_curves = build_curve_tables(rows, trajectories, progress_points)
        trajectory_curves = build_trajectory_curves(rows)
        (stage / "curves").mkdir()
        for filename, entries in curve_exports(raw_curves, normalized_curves, analysis, trajectory_curves):
            write_csv(stage / "curves" / filename, entries)
        render_line_plots(stage / "plots", raw_curves, normalized_curves, analysis, trajectory_curves, trajectories)
        verify_frozen_sources(root, stage, analysis, selection)
        _verify_decision_sources(root, records, selection)
        for run in selection["source_runs"]:
            if sha256(owned_path(root, run["config_source"]["path"])) != run["config_source"]["sha256"]:
                raise ValueError("Source config changed during analysis")
        changed = [f for f, raw in before.items() if (implementation / f).read_bytes() != raw]
        if any(f != "core/cli.py" for f in changed):
            raise ValueError("Branch statistics implementation changed during analysis: " + ", ".join(changed))
        write_json(stage / "branch-receipt.json", {
            "schema": SCHEMA, "analysis_id": analysis_id, "completed_at": now(),
            "files": {str(p.relative_to(stage)): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()},
        })
        stage.rename(out)
    return {"analysis_id": analysis_id, "output": str(out), "trajectories": len(trajectories),
            "steps": len(rows), "metrics": summary["metrics"]}


def curve_exports(raw_curves, normalized_curves, analysis, trajectory_curves=()):
    """Match plotted axes while retaining the raw-step audit and legacy exports."""
    if analysis.get("curve_schema_version", 2) < 3:
        return [("per-run.csv", raw_curves), ("all-runs-normalized.csv", normalized_curves)]
    if analysis.get("curve_schema_version", 2) >= 4 and analysis["line_mode"] == "per-trajectory":
        return [("per-trajectory.csv", trajectory_curves)]
    if analysis.get("task_ids"):
        return [("per-run.csv", raw_curves), ("all-runs-task.csv", raw_curves), ("raw-steps.csv", raw_curves)]
    return [("per-run.csv", normalized_curves), ("all-runs-normalized.csv", normalized_curves),
            ("raw-steps.csv", raw_curves)]


def _verify_decision_sources(root, records, selection):
    embedded_sources = {}
    for record in records:
        proof = record["proof"]
        if proof["kind"] == "full":
            if sha256(proof["path"]) != proof["sha256"]:
                raise ValueError("Full decision source checksum mismatch")
        else:
            key = (record["source_run_id"], record["rollout_id"])
            if key not in embedded_sources:
                source = next(s for s in selection["sources"]
                              if (s["source_run_id"], s["rollout_id"]) == key)
                embedded_sources[key] = read_messages(root, source)
            embedded = embedded_sources[key][record["message_index"]]["extra"]["dycon_parallel"]
            if hash_json(embedded) != proof["sha256"]:
                raise ValueError("Embedded decision source checksum mismatch")


def check_branch_analysis(cfg, analysis_id):
    root = Path(cfg["project_root"])
    out = artifact_dir(cfg, "action", analysis_id)
    receipt = read_json(out / "branch-receipt.json")
    if receipt.get("schema") not in {LEGACY_SCHEMA, SCHEMA} or receipt.get("analysis_id") != analysis_id:
        raise ValueError("Invalid branch analysis receipt identity")
    for name, digest in receipt["files"].items():
        if sha256(owned_path(out, name)) != digest:
            raise ValueError(f"Branch analysis checksum mismatch: {name}")
    analysis, selection = read_json(out / "analysis.json"), read_json(out / "selection.json")
    verify_frozen_sources(root, out, analysis, selection)
    records = [json.loads(line) for line in (out / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [json.loads(line) for line in (out / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    _verify_decision_sources(root, records, selection)
    legacy = analysis["schema"] == LEGACY_SCHEMA
    trajectories = read_json(out / "trajectories.json")
    if analysis.get("task_ids") and any(t["instance_id"] not in analysis["task_ids"] for t in trajectories):
        raise ValueError("Single-task output includes another task")
    lengths = {(t["source_run_id"], t["rollout_id"]): t["assistant_steps"] for t in trajectories}
    expected_rows, counters = [], Counter()
    for record in records:
        row = step_statistics(record, analysis["branch_scope"], legacy=legacy)
        if not legacy:
            key = (row["source_run_id"], row["rollout_id"])
            length = lengths[key]
            row.update({"trajectory_assistant_steps": length, "parallel_index": counters[key],
                        "normalized_progress": row["assistant_index"] / (length - 1) if length > 1 else 0.0})
            counters[key] += 1
        expected_rows.append(row)
    if expected_rows != rows:
        raise ValueError("Step statistics disagree with frozen branch actions")
    expected = summarize(rows, trajectories, analysis["branch_scope"], legacy=legacy)
    # JSON object keys stringify integer branch counts.
    if hash_json(expected) != hash_json(read_json(out / "summary.json")):
        raise ValueError("Branch histogram summary mismatch")
    if not legacy:
        from action.evaluation.branch_curves import build_curve_tables, build_trajectory_curves, trajectory_plot_index
        raw_curves, normalized_curves = build_curve_tables(rows, trajectories, analysis["progress_points"])
        trajectory_curves = build_trajectory_curves(rows)
        for filename, entries in curve_exports(raw_curves, normalized_curves, analysis, trajectory_curves):
            with (out / "curves" / filename).open(encoding="utf-8-sig", newline="") as stream:
                stored = list(csv.DictReader(stream))
            canonical = [{k: "" if v is None else str(v) for k, v in r.items()} for r in entries]
            if canonical != stored:
                raise ValueError("Curve values/denominators differ from frozen steps: " + filename)
        if analysis.get("curve_schema_version", 2) >= 4 and analysis["line_mode"] == "per-trajectory":
            index = trajectory_plot_index(trajectories, analysis)
            if read_json(out / "plots/trajectory-index.json") != index:
                raise ValueError("Per-trajectory plot coverage differs from selected trajectories")
            for entry in index["entries"]:
                for filename in entry["files"]:
                    if not owned_path(out, filename).is_file() or filename not in receipt["files"]:
                        raise ValueError("Missing per-trajectory figure: " + filename)
    implementation = Path(__file__).resolve().parents[2]
    matches = all((implementation / f).is_file() and sha256(implementation / f) == h
                  for f, h in analysis["code"].items())
    return {"analysis_id": analysis_id, "verified": True, "steps": len(rows),
            "trajectories": expected["trajectories"],
            "current_implementation_matches_snapshot": matches}

"""Receipt-bound parallel-score consistency evaluation beside the plot workflow."""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from action.datasets.trajectories import select_trajectories
from application.execution.evidence import RunEvidence
from core.io import artifact_dir, identifier, now, sha256, write_json
from core.locking import exclusive_artifact
from difficulty.evaluation.instability import aggregate_steps, metric_names, summarize_groups
from difficulty.evaluation.instability import evaluate_step as evaluate_step_metrics
from difficulty.evaluation.parallel import (
    ParallelEvidenceUnavailable,
    collect_parallel_scores,
    list_parallel_trajectories,
)

CODE_FILES = ["difficulty/evaluation/parallel_instability.py", "difficulty/evaluation/instability.py",
              "difficulty/evaluation/parallel_identity.py", "difficulty/evaluation/parallel.py"]


def _select(cfg, run_id, rollout_ids, task_ids, attempt_id):
    if rollout_ids and task_ids or attempt_id and not task_ids:
        raise ValueError("Use rollout IDs or task IDs; --attempt-id requires --task-id")
    if len(set(rollout_ids)) != len(rollout_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError("Duplicate explicit trajectory/task selection")
    available = list_parallel_trajectories(cfg, run_id)
    if rollout_ids:
        ids = list(rollout_ids)
    elif task_ids:
        ids = []
        for task in task_ids:
            matches = [r for r in available if r["task_id"] == task
                       and (attempt_id is None or r["attempt_id"] == attempt_id)]
            if len(matches) != 1:
                choices = ", ".join(f'{r["attempt_id"]}={r["rollout_id"]}' for r in matches) or "none"
                raise ValueError(f"Task {task} is ambiguous/missing; specify attempt or rollout. Matches: {choices}")
            ids.append(matches[0]["rollout_id"])
    else:
        eligible = RunEvidence(cfg["project_root"], run_id).read()["rollouts"]
        ids = [r["rollout_id"] for r in available if r["rollout_id"] in eligible]
    if not ids or set(ids) - {r["rollout_id"] for r in available}:
        raise ValueError("No trajectories selected, or requested rollout belongs to another run")
    return select_trajectories(cfg, run_id, ids)


def _csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not rows:
            stream.write("no_rows\n")
            return
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False)
                             if isinstance(v, (list, dict)) else v for k, v in row.items()})


def _markdown(summary):
    lines = ["# Parallel score conditional consistency", "", f'Run: `{summary["run_id"]}`', "",
             "All estimates are stored predictions. This report measures reasoning sensitivity at fixed action,",
             "not prediction accuracy, ground-truth noise variance, or counterfactual rollout success.", "",
             f'Reasoning identity: **{summary["parameters"]["reasoning_identity"]}**; '
             f'score space: **{summary["score_space"]}**.', "",
             "Aggregation: action group -> step -> rollout -> task; tasks have equal weight.",
             "Confidence intervals resample whole tasks or repositories, retaining temporal dependence.", ""]
    for view, report in summary["views"].items():
        lines += [f"## {view}", "", "| Metric | Task-balanced value | Task bootstrap 95% CI |",
                  "|---|---:|---|"]
        for name, metric in report["aggregate"].items():
            if name == "relative":
                continue
            value, ci = metric["mean"], metric["task_bootstrap"]["ci95"]
            lines.append(f'| {name} | {value if value is not None else "unavailable"} | {ci} |')
        relative = report["aggregate"]["relative"]
        lines += ["", f'Relative Q = {relative["ratio"]}; same gap = {relative["same_gap"]}; '
                  f'different gap = {relative["different_gap"]}; mixed steps = {relative["mixed_steps"]}.',
                  f'Task CI: {relative["task_bootstrap"]["ci95"]}; '
                  f'repo CI: {relative["repo_bootstrap"]["ci95"]}.', "",
                  f'Supported action groups: {report["groups"]["supported_groups"]}; '
                  f'all eligible action groups: {report["groups"]["action_groups"]}.', ""]
    lines += ["## Coverage", "", "```json", json.dumps(summary["coverage"], indent=2, ensure_ascii=False), "```", "",
              "Multiplier rates are unavailable outside log1p score space. Undefined metrics remain null,",
              "never zero. Relative Q divides paired task-balanced means, not averaged per-step ratios.", ""]
    return "\n".join(lines)


@exclusive_artifact("difficulty", "evaluation_id")
def evaluate_parallel_instability(cfg, run_id, evaluation_id, *, rollout_ids=(), task_ids=(),
                                  attempt_id=None, reasoning_identity="text", score_space="auto",
                                  multipliers=(1.25, 1.5, 2., 3.), bootstrap_samples=2000, seed=42,
                                  skip_unavailable=False):
    identifier(run_id)
    if reasoning_identity not in {"text", "tokens"} or score_space not in {"auto", "log1p", "native"}:
        raise ValueError("Invalid reasoning identity or score space")
    multipliers = sorted(set(float(k) for k in multipliers))
    if not multipliers or any(not math.isfinite(k) or k <= 1 for k in multipliers):
        raise ValueError("Multipliers must be finite and greater than one")
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 0 or seed < 0:
        raise ValueError("Bootstrap samples and seed must be nonnegative integers")
    out = artifact_dir(cfg, "difficulty", evaluation_id, "evaluations")
    if out.exists():
        raise FileExistsError(f"Use a new evaluation ID: {out}")
    selection = _select(cfg, run_id, rollout_ids, task_ids, attempt_id)
    all_branches, all_groups, all_steps, sources, unavailable = [], [], [], [], []
    targets, estimators, endpoints = set(), set(), set()
    for source in selection["sources"]:
        rid = source["rollout_id"]
        try:
            data = collect_parallel_scores(cfg, run_id, rid, identity_mode=reasoning_identity)
        except ParallelEvidenceUnavailable as exc:
            if not skip_unavailable:
                raise
            unavailable.append({"rollout_id": rid, "task_id": source["instance_id"], "reason": str(exc)})
            continue
        for name in ("trajectory_sha256", "task_key", "model_fingerprint", "run_id"):
            if source[name] != data["source"][name]:
                raise ValueError("Trajectory identity changed during evaluation")
        targets.add(data["target"])
        estimators.add(data["estimator_sha256"])
        endpoints.add(data["endpoint"])
        if len(targets) > 1 or len(estimators) > 1 or len(endpoints) > 1:
            raise ValueError("Cannot pool different predictor targets, checkpoints or endpoints")
        log1p = score_space == "log1p" or score_space == "auto" and data["target"].endswith("_log1p")
        context = {"task_id": source["instance_id"], "task_key": source["task_key"],
                   "repo": source["repo"], "rollout_id": rid}
        by_step = defaultdict(list)
        for branch in data["branches"]:
            row = {**context, **branch}
            all_branches.append(row)
            by_step[row["step_id"]].append(row)
        for step in data["steps"]:
            for view in ("deduplicated", "sampling"):
                groups, metrics = evaluate_step_metrics(by_step[step["step_id"]], view=view,
                                                       log1p=log1p, multipliers=multipliers)
                coords = {**context, "step_id": step["step_id"], "decision_id": step["decision_id"]}
                all_groups.extend({**coords, **g} for g in groups)
                all_steps.append({**coords, **metrics})
        sources.append({**source, "session_id": data["session_id"], "hidden_root": data["hidden_root"],
                        "source_hashes": data["source_hashes"], "identity_source_files": data["identity_source_files"],
                        "canonical_steps": len(data["steps"]), "excluded_decisions": data["excluded_decisions"]})
    if not sources:
        raise ValueError("No published action-bound trajectory was available")
    target = next(iter(targets))
    log1p = score_space == "log1p" or score_space == "auto" and target.endswith("_log1p")
    coverage = {"requested_rollouts": len(selection["sources"]), "evaluated_rollouts": len(sources),
                "unavailable_rollouts": unavailable,
                "canonical_steps": sum(s["canonical_steps"] for s in sources),
                "branch_candidates": len(all_branches),
                "eligible_finite_candidates": sum(r["status"] == "eligible" and r["score"] is not None
                                                  for r in all_branches),
                "identity_status_counts": dict(Counter(r["identity_status"] for r in all_branches))}
    summary = {"schema_version": 1, "evaluation_id": evaluation_id, "created_at": now(), "run_id": run_id,
               "target": target, "endpoint": next(iter(endpoints)), "estimator_sha256": next(iter(estimators)),
               "score_space": "log1p" if log1p else "native_nonlog_or_unspecified",
               "multiplier_rates_available": log1p,
               "parameters": {"reasoning_identity": reasoning_identity, "requested_score_space": score_space,
                              "multipliers": multipliers, "bootstrap_samples": bootstrap_samples, "seed": seed,
                              "skip_unavailable": skip_unavailable},
               "coverage": coverage, "views": {},
               "weighting": "equal action group -> step -> rollout -> task; whole task/repo block bootstrap",
               "interpretation": "Conditional reasoning sensitivity, not accuracy or calibrated noise variance"}
    names = metric_names(multipliers, log1p)
    task_reports = []
    for view in ("deduplicated", "sampling"):
        rows = [r for r in all_steps if r["view"] == view]
        groups = [r for r in all_groups if r["view"] == view]
        summary["views"][view] = {"aggregate": aggregate_steps(rows, names, bootstrap_samples=bootstrap_samples, seed=seed),
                                  "groups": summarize_groups(groups)}
        for task_key, task in sorted({(r["task_key"], r["task_id"]) for r in rows}):
            task_rows = [r for r in rows if r["task_key"] == task_key]
            metrics = aggregate_steps(task_rows, names, bootstrap_samples=0, seed=seed)
            task_reports.append({"task_id": task, "task_key": task_key, "repo": task_rows[0]["repo"], "view": view,
                                 **{k: v["mean"] for k, v in metrics.items() if k != "relative"},
                                 "relative_ratio": metrics["relative"]["ratio"],
                                 "relative_mixed_steps": metrics["relative"]["mixed_steps"],
                                 "supported_steps": metrics["mean_pair_gap"]["steps"]})
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".parallel-instability-", dir=out.parent) as temporary:
        stage = Path(temporary)
        write_json(stage / "selection.json", selection)
        write_json(stage / "sources.json", sources)
        write_json(stage / "summary.json", summary)
        for name, rows in [("branches", all_branches), ("groups", all_groups), ("steps", all_steps),
                           ("tasks", task_reports)]:
            _csv(stage / (name + ".csv"), rows)
            with (stage / (name + ".jsonl")).open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        (stage / "report.md").write_text(_markdown(summary), encoding="utf-8")
        write_json(stage / "receipt.json", {
            "schema_version": 1, "state": "complete", "created_at": summary["created_at"], "parameters": summary["parameters"],
            "code": {name: sha256(Path(__file__).resolve().parents[2] / name) for name in CODE_FILES},
            "files": {p.name: sha256(p) for p in sorted(stage.iterdir()) if p.is_file()}})
        stage.rename(out)
    return {"output": str(out), "report": str(out / "report.md"), "coverage": coverage,
            "primary": summary["views"]["deduplicated"]["aggregate"]}

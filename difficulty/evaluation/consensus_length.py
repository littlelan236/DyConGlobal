"""Compare stored estimator scores and completion lengths inside the historical winner vote."""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LinearRegression

from action.datasets.trajectories import read_messages, select_trajectories
from core.action_identity import canonical_action
from core.io import artifact_dir, identifier, now, owned_path, sha256, write_json
from core.locking import exclusive_artifact
from difficulty.evaluation.instability import aggregate_metrics, describe
from rollout.consensus import CONSENSUS_MODES


METRICS = (
    "same_branch", "optima_overlap", "both_unique", "length_tied", "score_tied",
    "token_excess", "relative_token_excess", "spearman", "kendall_tau_b",
    "actual_token_regret", "actual_is_estimator_best",
)
CODE_FILES = (
    "difficulty/evaluation/consensus_length.py", "difficulty/evaluation/instability.py",
    "action/datasets/trajectories.py", "core/action_identity.py", "core/cli.py",
)


def _branch_map(decision):
    count = decision.get("branch_count")
    candidates = decision.get("candidates", [])
    ids = [row.get("branch_id") for row in candidates]
    if (type(count) is not int or count < 1 or len(ids) != count
            or any(type(bid) is not int for bid in ids) or sorted(ids) != list(range(count))):
        raise ValueError("Invalid consensus branch inventory")
    return dict(zip(ids, candidates))


def evaluate_decision(decision, message):
    """Return one step and its winning-vote candidates; leave undefined metrics null."""
    consensus = decision.get("consensus") or {}
    winner_hash = consensus.get("consensus_action_hash")
    selected = decision.get("selected_branch_id")
    policy = (decision.get("consensus_config") or {}).get("representative_policy") or "estimator_score"
    step = {"status": "no_winner", "consensus_status": consensus.get("status"),
            "action_hash": winner_hash, "representative_policy": policy, "selected_branch_id": selected,
            "winner_count": 0, "length_count": 0, "score_count": 0, "paired_count": 0,
            "shortest_branch_ids": None, "estimator_best_branch_ids": None,
            "shortest_branch_id": None, "estimator_branch_id": None,
            **dict.fromkeys(METRICS)}
    if winner_hash is None:
        if consensus.get("votes") and consensus.get("status") not in {"insufficient_branches", "no_eligible_branch"}:
            raise ValueError("Consensus votes are present without a winner binding")
        return step, []
    candidates = _branch_map(decision)
    votes = [vote for vote in consensus.get("votes", []) if vote.get("action_hash") == winner_hash]
    if len(votes) != 1:
        raise ValueError("Expected exactly one recorded winner vote")
    ids = votes[0].get("branch_ids")
    retained = consensus.get("retained_branch_ids", [])
    if (not isinstance(ids, list) or not ids or any(type(bid) is not int for bid in ids)
            or len(set(ids)) != len(ids) or votes[0].get("count") != len(ids)
            or not set(ids) <= set(retained) or not set(ids) <= candidates.keys()):
        raise ValueError("Winner vote does not match the retained candidate inventory")
    if selected is not None and (type(selected) is not int or selected not in ids):
        raise ValueError("Selected branch is outside the recorded winner vote")
    if selected is not None and canonical_action(message)[0] != winner_hash:
        raise ValueError("Canonical assistant action differs from the recorded winner vote")

    branches, lengths, scores = [], {}, {}
    for bid in sorted(ids):
        row = candidates[bid]
        if row.get("action_hash") != winner_hash or row.get("retained") is not True:
            raise ValueError("Winner candidate does not match the recorded action or retention")
        length, score = row.get("completion_tokens"), row.get("score")
        if type(length) is int and length >= 0:
            lengths[bid] = length
        if type(score) in (float, int) and math.isfinite(score):
            scores[bid] = score
        branches.append({"branch_id": bid, "completion_tokens": lengths.get(bid),
                         "length_source": row.get("length_source"), "score": scores.get(bid),
                         "selected": bid == selected})

    n = len(ids)
    complete_lengths, complete_scores = len(lengths) == n, len(scores) == n
    step.update(winner_count=n, length_count=len(lengths), score_count=len(scores),
                paired_count=len(lengths.keys() & scores.keys()),
                status=("singleton" if n == 1 else "comparable") if complete_lengths and complete_scores
                else "missing_length_and_score" if not complete_lengths and not complete_scores
                else "missing_length" if not complete_lengths else "missing_score")
    shortest, best = None, None
    if complete_lengths:
        length = min(lengths.values())
        shortest = [bid for bid in lengths if lengths[bid] == length]
        step.update(shortest_branch_ids=shortest, length_tied=len(shortest) > 1,
                    actual_token_regret=lengths[selected] - length if selected is not None else None)
    if complete_scores:
        score = min(scores.values())
        best = [bid for bid in scores if scores[bid] == score]
        estimator_branch = best[0]  # lengths/scores were populated in branch ID order.
        step.update(estimator_best_branch_ids=best, estimator_branch_id=estimator_branch,
                    score_tied=len(best) > 1,
                    actual_is_estimator_best=selected in best if selected is not None else None)
    if shortest is not None and best is not None:
        length_branch = min(shortest, key=lambda bid: (scores[bid], bid))
        excess = lengths[estimator_branch] - length
        step.update(shortest_branch_id=length_branch, same_branch=length_branch == estimator_branch,
                    optima_overlap=not set(shortest).isdisjoint(best),
                    both_unique=len(shortest) == len(best) == 1, token_excess=excess,
                    relative_token_excess=excess / length if length else None)
        if len(set(lengths.values())) > 1 and len(set(scores.values())) > 1:
            xs, ys = list(lengths.values()), list(scores.values())
            step.update(spearman=float(spearmanr(xs, ys).statistic),
                        kendall_tau_b=float(kendalltau(xs, ys).statistic))
    for branch in branches:
        bid = branch["branch_id"]
        branch.update(is_shortest=bid in shortest if shortest is not None else None,
                      is_estimator_best=bid in best if best is not None else None)
    return step, branches


def evaluate_r2(groups, *, seed):
    """Score length-only regressions on held-out tasks with equal hierarchical weights."""
    by_task = defaultdict(lambda: defaultdict(list))
    for task, rollout, members in groups:
        by_task[task][rollout].append(members)
    tasks = sorted(by_task)
    fold_count = min(5, len(tasks))
    result = {"raw_score_r2": None, "within_step_score_r2": None, "tasks": len(tasks),
              "rollouts": sum(len(v) for v in by_task.values()), "steps": len(groups),
              "branches": sum(len(members) for _, _, members in groups),
              "folds": fold_count if fold_count > 1 else 0, "fold_task_counts": [],
              "seed": seed, "status": "insufficient_tasks"}
    if fold_count < 2:
        return result
    Random(seed).shuffle(tasks)
    fold, weights, raw, centered = [], [], [], []
    for i, task in enumerate(tasks):
        rollouts = by_task[task]
        for steps in rollouts.values():
            for members in steps:
                values = np.asarray([(b["completion_tokens"], b["score"]) for b in members], dtype=float)
                raw.extend(values)
                deviations = values - values[0]
                centered.extend(deviations - deviations.mean(axis=0))
                weight = 1 / (len(rollouts) * len(steps) * len(members))
                weights.extend([weight] * len(members))
                fold.extend([i % fold_count] * len(members))
    fold, weights = np.asarray(fold), np.asarray(weights)
    result["fold_task_counts"] = [len(tasks[i::fold_count]) for i in range(fold_count)]
    for field, values, intercept in (("raw_score_r2", raw, True),
                                     ("within_step_score_r2", centered, False)):
        x, y = np.asarray(values).T
        # The within-step target is a deviation from that step's mean score.
        baseline = np.average(y, weights=weights) if intercept else 0.0
        total = np.dot(weights, (y - baseline) ** 2) if np.any(y != y[0]) else 0.0
        if total == 0:
            continue
        prediction = np.empty_like(y)
        for i in range(fold_count):
            train, test = fold != i, fold == i
            model = LinearRegression(fit_intercept=intercept)
            model.fit(x[train, None], y[train], sample_weight=weights[train])
            prediction[test] = model.predict(x[test, None])
        result[field] = float(1 - np.dot(weights, (y - prediction) ** 2) / total)
    result["status"] = "ok" if result["raw_score_r2"] is not None else "constant_score"
    return result


def _write_csv(path, rows, empty_columns):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or empty_columns
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                             for key, value in row.items()})


def _report(summary):
    def show(x):
        if x is None:
            return "n/a"
        if isinstance(x, list):
            return "[" + ", ".join(show(v) for v in x) + "]"
        return f"{x:.4g}" if isinstance(x, float) else str(x)

    lines = ["# Winner-vote length vs. estimator", "", f'Run: `{summary["run_id"]}`', "",
             "Compare only historical winner-vote members. Lower stored estimator score is better.",
             "Length is recorded completion tokens. Positive rank correlation means shorter branches",
             "tend to score better. Scores are predictions; this does not evaluate downstream success.", "",
             "Primary metrics use multi-member groups; singleton groups are counted separately.",
             "A missing value excludes the affected metric, without selecting a winner from a partial group.",
             "Spearman and Kendall tau-b require variation in both quantities.", "",
             "| Metric | Task-balanced mean | Tasks | Steps | Task 95% CI | Repo 95% CI |",
             "|---|---:|---:|---:|---|---|"]
    for metric, data in summary["metrics"].items():
        lines.append(f'| {metric} | {show(data["mean"])} | {data["tasks"]} | {data["steps"]} | '
                     f'{show(data["task_bootstrap"]["ci95"])} | {show(data["repo_bootstrap"]["ci95"])} |')
    r2 = summary["r2"]
    lines += ["", "Out-of-sample R^2 from length-only linear regressions:",
              f'- Raw estimator score: **{show(r2["raw_score_r2"])}** (baseline: weighted mean score).',
              f'- Within-step score deviation: **{show(r2["within_step_score_r2"])}** (baseline: zero deviation).',
              f'- {r2["tasks"]} tasks, {r2["steps"]} complete multi-member steps, '
              f'{r2["branches"]} branches, {r2["folds"]} task-held-out folds.',
              "Each task is fully held out once. Predictions and squared errors are pooled with equal task",
              "weight, then equal trajectory, step, and branch weight within each parent."]
    lines += ["", "Token excess across eligible steps (unweighted distribution): " + ", ".join(
        f"{key}={show(value)}" for key, value in summary["token_excess_distribution"].items()),
        "", "Coverage:", "", "```json", json.dumps(summary["coverage"], indent=2), "```", ""]
    return "\n".join(lines)


@exclusive_artifact("difficulty", "evaluation_id")
def evaluate_consensus_length(cfg, run_id, evaluation_id, *, rollout_ids=(), bootstrap_samples=2000, seed=42):
    """Read immutable canonical decisions and publish a standalone evaluation receipt."""
    if type(bootstrap_samples) is not int or type(seed) is not int or min(bootstrap_samples, seed) < 0:
        raise ValueError("Bootstrap samples and seed must be nonnegative integers")
    root = Path(cfg["project_root"])
    out = artifact_dir(cfg, "difficulty", evaluation_id, "evaluations")
    if out.exists():
        raise FileExistsError(f"Use a new evaluation ID: {out}")
    selection = select_trajectories(cfg, run_id, rollout_ids)
    locator = selection["source_runs"][0]["config_source"]
    if locator is None:
        raise ValueError("Source requires a frozen resolved-config.json")
    config_bytes = owned_path(root, locator["path"]).read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != locator["sha256"]:
        raise ValueError("Frozen config changed during selection")
    parallel = json.loads(config_bytes).get("parallel", {})
    if not parallel.get("enabled") or parallel.get("mode") not in CONSENSUS_MODES:
        raise ValueError("Source run did not enable parallel consensus")

    steps, branches, trajectories, r2_groups = [], [], [], []
    score_identities, decisions = set(), set()
    for source in selection["sources"]:
        context = {"task_key": source["task_key"], "task_id": source["instance_id"],
                   "repo": source["repo"], "rollout_id": source["rollout_id"]}
        statuses = Counter()
        for index, message in enumerate(read_messages(root, source)):
            if message.get("role") != "assistant":
                continue
            decision = (message.get("extra") or {}).get("dycon_parallel")
            if not decision or decision.get("selection_mode") not in CONSENSUS_MODES:
                statuses["nonconsensus_assistant"] += 1
                continue
            session, decision_id = (identifier(decision[key]) for key in ("session_id", "decision_id"))
            key = (session, decision_id)
            if key in decisions:
                raise ValueError(f"Duplicate canonical consensus decision: {key}")
            decisions.add(key)
            try:
                step, members = evaluate_decision(decision, message)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid consensus evidence in {source['rollout_id']}/{decision_id}: {exc}") from exc
            binding = {**context, "session_id": session, "decision_id": decision_id,
                       "message_index": index, "step_id": decision.get("step_id")}
            statuses[step["status"]] += 1
            if members:
                identity = tuple(decision.get(k) for k in ("estimator_sha256", "target", "endpoint"))
                if any(not isinstance(v, str) or not v for v in identity):
                    raise ValueError(f"Missing score identity for {decision_id}")
                score_identities.add(identity)
            steps.append({**binding, **step})
            branches.extend({**binding, **row} for row in members)
            if step["status"] == "comparable":
                r2_groups.append((context["task_key"], context["rollout_id"], members))
        trajectories.append({**context, "termination_reason": source["termination_reason"],
                             "status_counts": dict(statuses)})
    if not steps:
        raise ValueError("Selected trajectories contain no canonical consensus decisions")
    frozen_estimator = parallel.get("estimator_sha256")
    if len(score_identities) > 1 or any(identity[0] != frozen_estimator for identity in score_identities):
        raise ValueError("Score identity changed or differs from the frozen estimator; select a homogeneous cohort")

    primary = [row for row in steps if row["winner_count"] > 1]
    coverage = {"trajectories": len(trajectories), "tasks": len({t["task_key"] for t in trajectories}),
                "consensus_steps": len(steps), "multi_member_steps": len(primary),
                "singleton_groups": sum(r["winner_count"] == 1 for r in steps),
                "trajectories_without_consensus": sum(
                    sum(t["status_counts"].values()) == t["status_counts"].get("nonconsensus_assistant", 0)
                    for t in trajectories),
                "step_status": dict(Counter(row["status"] for row in steps)),
                "consensus_status": dict(Counter(row["consensus_status"] or "missing" for row in steps)),
                "representative_policy": dict(Counter(row["representative_policy"] for row in steps)),
                "winner_branches": len(branches), "lengths": sum(r["length_count"] for r in steps),
                "scores": sum(r["score_count"] for r in steps), "paired": sum(r["paired_count"] for r in steps)}
    identity = next(iter(score_identities), None)
    created = now()
    params = {"bootstrap_samples": bootstrap_samples, "seed": seed}
    summary = {"schema_version": 1, "run_id": run_id, "created_at": created, "parameters": params,
               "score_identity": dict(zip(("estimator_sha256", "target", "endpoint"), identity)) if identity else None,
               "weighting": "equal steps within a rollout, rollouts within a task, then equal tasks",
               "coverage": coverage, "metrics": aggregate_metrics(primary, METRICS, require_repo=True, **params),
               "r2": evaluate_r2(r2_groups, seed=seed),
               "token_excess_distribution": describe([r["token_excess"] for r in primary
                                                      if r["token_excess"] is not None])}
    implementation = Path(__file__).resolve().parents[2]
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".consensus-length-", dir=out.parent) as temporary:
        stage = Path(temporary)
        write_json(stage / "selection.json", selection)
        write_json(stage / "trajectories.json", trajectories)
        write_json(stage / "summary.json", summary)
        _write_csv(stage / "steps.csv", steps, ["decision_id"])
        _write_csv(stage / "branches.csv", branches, ["decision_id", "branch_id", "completion_tokens", "score"])
        (stage / "report.md").write_text(_report(summary), encoding="utf-8")
        write_json(stage / "receipt.json", {"schema_version": 1, "state": "complete", "created_at": created,
                   "source_run_id": run_id, "parameters": params,
                   "code": {name: sha256(implementation / name) for name in CODE_FILES},
                   "files": {p.name: sha256(p) for p in sorted(stage.iterdir())}})
        stage.rename(out)
    return {"output": str(out), "report": str(out / "report.md"), "coverage": coverage}

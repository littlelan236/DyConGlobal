"""Plot stored online scores aligned to one finalized canonical trajectory.

This analysis never loads a predictor, hidden tensors, or a model service.
"""

import csv
import math
import statistics
from pathlib import Path
from tempfile import TemporaryDirectory

from action.datasets.trajectories import read_actions, select_trajectories
from application.execution.evidence import RunEvidence
from core.config import load_snapshot
from core.io import artifact_dir, identifier, now, owned_path, read_json, sha256, write_json
from core.locking import exclusive_artifact
from difficulty.evaluation.visualize import plot_parallel_scores
from hidden.artifacts import read_parallel_proof


class ParallelEvidenceUnavailable(ValueError):
    """A trajectory has no published action-bound parallel data yet."""


def list_parallel_trajectories(cfg, run_id):
    """List catalog identities, including attempts; never silently pick the latest."""
    identifier(run_id)
    catalog = RunEvidence(cfg["project_root"], run_id).read()
    run_catalog = catalog
    return sorted(
        [{"rollout_id": rid, "task_id": catalog["tasks"][r["task_key"]]["instance_id"],
          "attempt_id": r["attempt_id"], "termination_reason": r["termination_reason"],
          "resolved": r["resolved"], "quarantined": bool(r["provenance"].get("quarantined"))}
         for rid, r in run_catalog["rollouts"].items()],
        key=lambda r: (r["task_id"], r["attempt_id"], r["rollout_id"]),
    )


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _session(root, rollout_id):
    matches = []
    for path in sorted((root / "sessions").glob("*/finalization.json")):
        final = read_json(path)
        if final.get("rollout_id") == rollout_id:
            matches.append((path, final))
    if not matches:
        raise ParallelEvidenceUnavailable(f"No published finalized session for {rollout_id}")
    if len(matches) != 1:
        raise ValueError(f"Expected one published finalized session for {rollout_id}; found {len(matches)}")
    return matches[0]


def collect_parallel_scores(cfg, run_id, rollout_id, *, identity_mode=None, publication_id=None, repair_id=None):
    """Join receipt-checked decisions with finalized actual assistant steps."""
    root = Path(cfg["project_root"])
    selection = select_trajectories(cfg, run_id, [rollout_id])
    source = selection["sources"][0]
    if source["run_id"] != run_id:
        raise ValueError("Requested rollout belongs to a different run")
    frozen = load_snapshot(root, run_id)
    if not frozen.get("parallel", {}).get("enabled"):
        raise ValueError("Source run did not enable parallel decoding")
    published = Path(frozen["paths"]["nas_root"]) / frozen["capture"]["export"]["root"] / run_id
    publication = None
    if publication_id:
        from hidden.publish_visualization import read_visualization_publication
        publication=read_visualization_publication(frozen,run_id,publication_id,repair_id=repair_id)
        entries=[entry for entry in publication['decisions'] if entry['rollout_id']==rollout_id]
        if not entries: raise ParallelEvidenceUnavailable('Publication has no decisions for the requested rollout')
        final=entries[0]['finalization']
        final_path=publication['directory']/'decisions.json'
        sid=entries[0]['session_id']
    else:
        if repair_id: raise ValueError('--repair-id requires --publication-id for parallel plots')
        final_path, final = _session(published, rollout_id)
        sid=final_path.parent.name
    if (final["trajectory_sha256"] != source["trajectory_sha256"]
            or final["model_fingerprint"] != source["model_fingerprint"]):
        raise ValueError("Finalized session does not bind the canonical trajectory/model")
    if not publication and not (final_path.parent / "parallel/branch-receipt.json").exists():
        raise ParallelEvidenceUnavailable("Session has no published branch receipt")
    if publication:
        entries=[entry for entry in entries if entry.get('score_receipt_bound')]
        evidence={'samples':[{**candidate,'decision_id':entry['decision']['decision_id'],
            'selected':candidate['branch_id']==entry['decision']['selected_branch_id'],
            'rollout_id':rollout_id,'trajectory_sha256':source['trajectory_sha256']}
            for entry in entries for candidate in entry['decision']['candidates']],
            'decisions':[entry['decision'] for entry in entries],
            'receipt':{'rollout_id':rollout_id,'trajectory_sha256':source['trajectory_sha256']}}
    else:
        evidence = read_parallel_proof(published, final_path.parent.name,
                                   rollout_id=rollout_id,
                                   trajectory_sha256=source["trajectory_sha256"])
    receipt = evidence["receipt"]
    if (receipt["rollout_id"] != rollout_id
            or receipt["trajectory_sha256"] != source["trajectory_sha256"]):
        raise ValueError("Branch receipt does not bind the canonical trajectory")
    actions = read_actions(root, source)
    steps = [{"step_id": i, "message_index": action["message_index"], "decision_id": None, "query_step_id": None,
              "selected_branch_id": None, "selected_score": None, "score_span": None,
              "score_std": None, "finite_branches": 0, "missing_branches": 0,
              "eligible_branches": 0, "fallback_reason": None}
             for i, action in enumerate(actions)]
    response_map = final.get("responses")
    if not isinstance(response_map, dict):
        raise ValueError("Finalization lacks response-to-step bindings; recover/finalize this capture first")
    accepted = set(final["accepted_native_request_ids"])
    bound = {}
    for native, mapping in response_map.items():
        if native not in accepted:
            raise ValueError("Finalization response is not an accepted native request")
        if mapping["is_action_response"]:
            step = mapping["agent_step_index"]
            if (not isinstance(step, int) or not 0 <= step < len(steps)
                    or steps[step]["message_index"] != mapping["message_index"] or step in bound):
                raise ValueError("Finalization response-to-action-step binding mismatch")
            bound[step] = native
    if not publication and set(bound) != set(range(len(steps))):
        raise ValueError("Finalization does not cover all actual assistant steps")
    indexed = {(r["decision_id"], r["branch_id"]): r for r in evidence["samples"]}
    if len(indexed) != len(evidence["samples"]):
        raise ValueError("Duplicate branch identities")
    decisions = evidence["decisions"]
    if not decisions:
        raise ParallelEvidenceUnavailable("No receipt-bound parallel decisions in this session")
    from difficulty.evaluation.parallel_identity import CandidateIdentityReader, IdentityUnavailable

    identity_reader = CandidateIdentityReader(published, identity_mode) if identity_mode else None
    branches, excluded, targets, endpoints = [], [], set(), set()
    seen_decisions = set()
    expected_estimator = frozen["parallel"].get("estimator_sha256")
    for decision in decisions:
        did = decision["decision_id"]
        if did in seen_decisions:
            raise ValueError("Duplicate decision identity")
        seen_decisions.add(did)
        candidates = decision["candidates"]
        if (len(candidates) != decision["branch_count"]
                or len({c["branch_id"] for c in candidates}) != len(candidates)):
            raise ValueError("Incomplete or duplicate decision candidates")
        if decision.get("selection_mode") == "parallel_consensus":
            from rollout.consensus import validate_consensus_evidence

            validate_consensus_evidence(decision, frozen["parallel"].get("consensus"))
        for c in candidates:
            record = indexed.get((did, c["branch_id"]))
            if (record is None or record["rollout_id"] != rollout_id
                    or record["trajectory_sha256"] != source["trajectory_sha256"]
                    or record["native_request_id"] != c.get("native_request_id")
                    or record["score"] != c.get("score")
                    or record["selected"] != (c["branch_id"] == decision["selected_branch_id"])):
                raise ValueError("Decision disagrees with its published branch index")
        winner = next((c for c in candidates if c["branch_id"] == decision["selected_branch_id"]), None)
        if winner is None:
            if decision["selected_branch_id"] is not None:
                raise ValueError("Selected branch is absent from candidates")
            excluded.append({"decision_id": did, "step_id": decision["step_id"], "reason": "no_winner"})
            continue
        native = winner.get("native_request_id")
        mapping = response_map.get(native)
        if not mapping and publication:
            mapping=final.get('decision_mappings',{}).get(native)
        if not mapping:
            if publication:
                excluded.append({'decision_id':did,'step_id':decision['step_id'],'reason':'native_mapping_unavailable'})
                continue
            raise ValueError(f"Selected decision {did} is absent from the final trajectory bindings")
        if not mapping["is_action_response"]:
            excluded.append({"decision_id": did, "step_id": decision["step_id"],
                             "reason": "format_error_response_not_an_action"})
            continue
        step = mapping["agent_step_index"]
        if (type(step) is not int or not 0<=step<len(steps)
                or steps[step]['message_index']!=mapping['message_index']):
            raise ValueError('Parallel canonical decision action mapping mismatch')
        row = steps[step]
        if row["decision_id"] is not None or bound.get(step,native if publication else None) != native:
            raise ValueError("Parallel decision-to-actual-step binding mismatch")
        if expected_estimator and decision.get("estimator_sha256") != expected_estimator:
            raise ValueError("Decision predictor identity differs from the frozen run")
        if decision.get("target"):
            targets.add(decision["target"])
        if decision.get("endpoint"):
            endpoints.add(decision["endpoint"])
        values = [c["score"] for c in candidates if _finite(c.get("score"))]
        row.update(decision_id=did, query_step_id=decision["step_id"], selected_branch_id=winner["branch_id"],
                   selected_score=winner["score"] if _finite(winner.get("score")) else None,
                   score_span=max(values) - min(values) if values else None,
                   score_std=statistics.pstdev(values) if values else None,
                   finite_branches=len(values), missing_branches=len(candidates) - len(values),
                   eligible_branches=sum(c["status"] == "eligible" for c in candidates),
                   fallback_reason=decision.get("fallback_reason"))
        if decision.get("selection_mode") == "parallel_consensus":
            row["consensus"] = decision["consensus"]
        for c in candidates:
            value = c.get("score")
            branches.append({"step_id": step, "query_step_id": decision["step_id"],
                             "decision_id": did, "branch_id": c["branch_id"],
                             "selected": c["branch_id"] == winner["branch_id"],
                             "score": value if _finite(value) else None,
                             "status": c["status"], "reason": c.get("reason"),
                             "invalid_score": repr(value) if value is not None and not _finite(value) else None,
                             "native_request_id": c.get("native_request_id"),
                             "close_sample_id": c.get("close_sample_id"),
                             "fallback_reason": decision.get("fallback_reason")})
            if publication:
                points=[p for p in publication['branch-point-index'] if p['rollout_id']==rollout_id
                        and p['recovery'].get('decision_id')==did and p['recovery'].get('branch_id')==c['branch_id']]
                scope=next(s for s in publication['coverage']['sessions'] if s['rollout_id']==rollout_id)
                capture=next((r['close_capture'] for r in scope.get('branches',[])
                              if r['decision_id']==did and r['branch_id']==c['branch_id']),{})
                branches[-1].update(hidden_available=bool(points),score_available=_finite(value),
                    target=decision.get('target'),endpoint=decision.get('endpoint'),
                    estimator_sha256=decision.get('estimator_sha256'),
                    close_status=capture.get('status'),close_reason=capture.get('reason'))
            if decision.get("selection_mode") == "parallel_consensus":
                branches[-1].update({k: c.get(k) for k in ("retained", "estimator_rank", "exclusion_reason")})
            if identity_reader:
                identity = {"action_hash": None, "action": None, "reasoning_hash": None,
                            "reasoning_evidence": identity_mode, "identity_status": "not_eligible_or_finite",
                            "parent_prefix_sha256": decision.get("parent_prefix_sha256")}
                if c["status"] == "eligible" and _finite(value):
                    try:
                        identity.update(identity_reader.read(c, decision), identity_status="available")
                    except IdentityUnavailable as exc:
                        identity["identity_status"] = str(exc)
                branches[-1].update(identity)
    if len(targets) > 1 or len(endpoints) > 1:
        raise ValueError("Cannot mix predictor targets/endpoints in one plot")
    if not branches:
        raise ParallelEvidenceUnavailable("No parallel decision belongs to an actual action step")
    return {"selection": selection, "source": source, "steps": steps, "branches": branches,
            "target": next(iter(targets), "unspecified stored score"),
            "endpoint": next(iter(endpoints), None), "excluded_decisions": excluded,
            "session_id": sid, "hidden_root": str(published),
            "estimator_sha256": expected_estimator,
            "identity_source_files": identity_reader.sources if identity_reader else {},
            "source_hashes": {"run_snapshot": sha256(root / "runs" / run_id / "resolved-config.json"),
                              "finalization": sha256(final_path),
                              "branch_receipt": sha256(publication['directory']/'receipt.json') if publication else
                                  sha256(final_path.parent / "parallel/branch-receipt.json")}}


@exclusive_artifact("difficulty", "plot_id")
def visualize_parallel(cfg, run_id, plot_id, *, rollout_id=None, task_id=None,
                       attempt_id=None, step_start=0, step_end=None, publication_id=None, repair_id=None):
    """Publish PNG/SVG plus auditable score tables under a new analysis ID."""
    identifier(run_id)
    if bool(rollout_id) == bool(task_id) or (attempt_id and not task_id):
        raise ValueError("Choose one --rollout-id or --task-id; --attempt-id requires --task-id")
    if step_start < 0 or (step_end is not None and step_end < step_start):
        raise ValueError("Step range must be nonnegative and inclusive end >= start")
    out = artifact_dir(cfg, "difficulty", plot_id, "evaluations")
    if out.exists():
        raise FileExistsError(f"Use a new plot ID: {out}")
    if task_id:
        matches = [r for r in list_parallel_trajectories(cfg, run_id)
                   if r["task_id"] == task_id and (attempt_id is None or r["attempt_id"] == attempt_id)]
        if len(matches) != 1:
            choices = ", ".join(f'{r["attempt_id"]}={r["rollout_id"]}' for r in matches) or "none"
            raise ValueError(f"Expected one trajectory; specify --attempt-id/--rollout-id. Matches: {choices}")
        rollout_id = matches[0]["rollout_id"]
    data = collect_parallel_scores(cfg, run_id, rollout_id, publication_id=publication_id,repair_id=repair_id)
    steps = [r for r in data["steps"] if r["step_id"] >= step_start
             and (step_end is None or r["step_id"] <= step_end)]
    if not steps:
        raise ValueError("Requested range contains no actual trajectory steps")
    included = {r["step_id"] for r in steps}
    branches = [r for r in data["branches"] if r["step_id"] in included]
    source = data["source"]
    summary = {"schema_version": 1, "plot_id": plot_id, "created_at": now(), "run_id": run_id,
               "rollout_id": rollout_id, "task_id": source["instance_id"], "session_id": data["session_id"],
               "target": data["target"], "endpoint": data["endpoint"], "score_space": "native predictor output",
               "estimator_sha256": data["estimator_sha256"], "source_hashes": data["source_hashes"],
               "canonical_steps": len(data["steps"]), "plotted_steps": len(steps),
               "step_range": [steps[0]["step_id"], steps[-1]["step_id"]],
               "selected_scores": sum(r["selected_score"] is not None for r in steps),
               "branch_candidates": len(branches), "finite_branch_scores": sum(r["score"] is not None for r in branches),
               "missing_branch_scores": sum(r["score"] is None for r in branches),
               "steps_without_parallel_decision": sum(r["decision_id"] is None for r in steps),
               "excluded_decisions": data["excluded_decisions"],
               "interpretation": "Branch spread and selected-path estimates, not ground truth or calibrated noise variance"}
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".parallel-plot-", dir=out.parent) as temporary:
        stage = Path(temporary)
        plot_parallel_scores(steps, branches, stage, target=data["target"],
                             title=f'{source["instance_id"]} | {rollout_id}',
                             subtitle=f'{run_id} | {source["termination_reason"]} | steps are 0-based')
        for name, rows in [("steps.csv", steps), ("branches.csv", branches)]:
            with (stage / name).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else list(data["branches"][0]))
                writer.writeheader()
                writer.writerows(rows)
        write_json(stage / "scores.json", {"steps": steps, "branches": branches})
        write_json(stage / "selection.json", data["selection"])
        write_json(stage / "summary.json", summary)
        write_json(stage / "receipt.json", {
            "schema_version": 1, "state": "complete", "created_at": summary["created_at"],
            "source_hashes": data["source_hashes"],
            "code": {str(p): sha256(Path(__file__).resolve().parents[2] / p)
                     for p in ["difficulty/evaluation/parallel.py", "difficulty/evaluation/visualize.py"]},
            "files": {p.name: sha256(p) for p in sorted(stage.iterdir()) if p.is_file()},
        })
        stage.rename(out)
    return {"output": str(out), "plot": str(out / "trajectory.png"), **summary}

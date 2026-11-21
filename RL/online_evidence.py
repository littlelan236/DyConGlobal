"""Seal online rows from the existing canonical/evaluation/native capture APIs."""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from typing import Any, Mapping

from core.io import hash_json, owned_path, read_json, sha256
from RL.online_bootstrap import EstimatorDatasetStore, HiddenPoint
from RL.online_runtime import OnlineRolloutBatch, OnlineRow, validate_trajectory_coverage


def online_terminal_label(row, evidence, rollout_id):
    """Use the official raw boolean for online rewards, retaining audit sidecars."""
    outcome = evidence.get("evaluations", {}).get(row.get("selected_evaluation_id"), {}).get("outcomes", {}).get(rollout_id)
    if not isinstance(outcome, dict):
        return None
    # The existing run lifecycle grades an empty submission as a supported
    # negative without launching the official test harness.
    audit = outcome.get("verifier_audit") or {}
    if (audit.get("classification") == "ValidOutcome"
            and audit.get("reason_code") == "missing_patch"
            and audit.get("effective_resolved") is False
            and outcome.get("resolved") is False):
        return False
    official = outcome.get("official_result")
    if isinstance(official, dict):
        value = official.get("resolved")
        if type(value) is bool:
            return value
        # An official null must never become an implicit negative reward.
        return None
    raw = outcome.get("raw_outcome", outcome)
    value = raw.get("resolved")
    return value if raw.get("status") == "completed" and type(value) is bool else None


def _online_outcome(row, evidence, rollout_id):
    return evidence.get('evaluations', {}).get(row.get('selected_evaluation_id'), {}).get('outcomes', {}).get(rollout_id, {})


def ensure_evaluated(root, run_id):
    """Run the standard harness after rollout; terminal labels never default false."""
    from application.execution.evidence import RunEvidence
    from application.execution.run_plan import load_plan
    from application.execution.harness import prepare, run
    loaded = load_plan(root, run_id)
    evidence = RunEvidence(root, run_id)
    evidence.verify()
    data = evidence.read()
    selected = evidence.select()["rollouts"]
    if len(selected) != len(loaded["work_items"]["items"]):
        raise ValueError("online evaluation lacks an accepted final trajectory for every work item")
    from RL.online_failures import baseline_failure_reward, verifier_discard
    profiles = {(item.get('instance_id'), item.get('repetition_id')): item.get('online_source')
                for item in loaded['work_items']['items']}
    needs_grading = {rid: row for rid, row in selected.items()
        if baseline_failure_reward(loaded['config'], profiles.get((row.get('instance_id'), row.get('repetition_id'))), row) is None}
    # instance_id may be stored only in the task catalog on canonical rows.
    for rid, row in list(needs_grading.items()):
        task = data.get('tasks', {}).get(row.get('task_key'), {})
        source = profiles.get((task.get('instance_id'), row.get('repetition_id')))
        if baseline_failure_reward(loaded['config'], source, row) is not None:
            needs_grading.pop(rid)
    if all(type(online_terminal_label(row, data, rid)) is bool or verifier_discard(_online_outcome(row, data, rid))
           for rid, row in needs_grading.items()):
        return {"state": "reused", "evaluation_ids": sorted({row["selected_evaluation_id"] for row in needs_grading.values() if row.get("selected_evaluation_id")})}
    # SWE rollout and post-hoc official grading are distinct project services.
    # Other qualified live-verifier backends must already supply terminal labels.
    if loaded["config"].get("benchmark", {}).get("backend", "swebench") != "swebench":
        raise ValueError("online live-verifier trajectory lacks valid terminal evaluation")
    if (loaded["run"] / "harness/summary.json").is_file():
        raise ValueError("online official evaluation is ineligible; inspect harness validity evidence before retry")
    if len(needs_grading) == len(selected):
        prepare(root, run_id)
    else:
        prepare(root, run_id, rollout_ids=tuple(needs_grading))
    result = run(root, run_id, "online-" + hash_json(run_id)[:24])
    evidence.verify()
    data = evidence.read()
    selected = evidence.select()["rollouts"]
    if any(type(online_terminal_label(row, data, rid)) is not bool
           and not verifier_discard(_online_outcome(row, data, rid))
           for rid, row in selected.items() if rid in needs_grading):
        raise ValueError("online official evaluation lacks a raw boolean result")
    return {"state": "evaluated", "evaluation_ids": sorted({row["selected_evaluation_id"] for row in needs_grading.values() if row.get("selected_evaluation_id")})}


def native_completion(root, evidence, cfg, request):
    """Read exact tokens and the generating policy's raw-T1 vector, never retokenize."""
    from hidden.artifacts import read_request
    from runtime.serving.raw_logprobs import read_values
    native = read_request(root, evidence)
    prompt, output = native.get("prompt_token_ids"), native.get("output_token_ids")
    if (not prompt or not output or hash_json(prompt) != native.get("request_sha256")
            or hash_json(output) != native.get("output_sha256")):
        raise ValueError("online native token evidence is incomplete or changed")
    if native.get("run_id") != cfg["run_id"]:
        raise ValueError("online native request belongs to another run")
    descriptor = native.get("raw_t1_probability")
    if not descriptor:
        raise ValueError("online native request lacks generating raw-T1 probabilities")
    values = read_values(root, descriptor, output_ids=output, record=native)
    source = descriptor["source_identity"]
    if (source.get("model_fingerprint") != cfg["model"]["fingerprint"]
            or source.get("serving_dtype") != cfg["serving"]["dtype"]):
        raise ValueError("online probability model/dtype differs from frozen serving")
    expected_adapter = (request.get("deployment") or {}).get("adapter_identity_sha256")
    actual_adapter = source.get("adapter")
    if actual_adapter != expected_adapter:
        raise ValueError("online probability adapter differs from generating deployment")
    if request["policy_version"] != "S0" and expected_adapter is None:
        raise ValueError("updated online policy has no immutable serving adapter binding")
    if not all(math.isfinite(float(x)) for x in values):
        raise ValueError("online rollout contains nonfinite required probabilities")
    return native, tuple(float(x) for x in values)


class NativeRunAssembler:
    """Consume accepted repetitions and preserve their recorded decision advantages.

    ``feature`` binds the initial estimator target/site. Successful canonical
    points are read through sealed native trace publication; losing candidates
    can train the policy but never inherit winner suffix labels.
    """
    def __init__(self, cfg, *, feature: Mapping[str, Any], advantage_settings: Mapping[str, Any]):
        self.cfg = cfg
        self.feature = dict(feature)
        self.advantage_settings = dict(advantage_settings)

    def __call__(self, request, execution, *, run_id):
        from application.execution.evidence import RunEvidence
        from application.execution.run_plan import load_plan
        from benchmark_evaluation.swebench_settlement import training_label
        from hidden.artifacts import read_request, response_records
        from hidden.token_evidence import load_token_session, native_branch
        from RL.grpo.groups import advantages_for_members
        from rollout.retry import verify_final
        loaded = load_plan(self.cfg["project_root"], run_id)
        cfg, run = loaded["config"], loaded["run"]
        if execution.get("exit_code") != 0:
            raise RuntimeError("online source run has no successful supervisor exit receipt")
        evidence = RunEvidence(cfg["project_root"], run_id)
        evidence.verify()
        data = evidence.read()
        requested = {(item["task_id"], item["source"]): [] for item in request["work_items"]}
        for item in request["work_items"]:
            requested[(item["task_id"], item["source"])].append(item["trajectory_id"])
        rows, points, sources, excluded, discarded = [], [], [], [], []
        deps = {}
        for item in loaded["work_items"]["items"]:
            task_id, source = item["instance_id"], item["online_source"]
            repetition = int(item["online_repetition"])
            trajectory_id = requested[(task_id, source)][repetition]
            result_path = run / "rollout/tasks" / task_id / "repetitions" / item["repetition_id"] / "result.json"
            result = read_json(result_path)
            rid = verify_final(Path(cfg["project_root"]), run_id, item, result, data)
            record = data["rollouts"][rid]
            from RL.online_failures import baseline_failure_reward, verifier_discard
            reward_override = baseline_failure_reward(cfg, source, record)
            outcome = _online_outcome(record, data, rid)
            if reward_override is None and verifier_discard(outcome):
                discarded.append({'trajectory_id': trajectory_id, 'task_id': task_id,
                    'source': source, 'rollout_id': rid,
                    'reason': 'verifier_error_retries_exhausted',
                    'evaluation_id': record['selected_evaluation_id'],
                    'retry_limit': outcome['online_rl_discard']['retry_limit'],
                    'attempt_count': outcome['online_rl_discard']['attempt_count']})
                continue
            resolved = False if reward_override is not None else online_terminal_label(record, data, rid)
            if type(resolved) is not bool:
                raise ValueError("online trajectory lacks a valid official terminal evaluation")
            path = owned_path(cfg["project_root"], record["trajectory_path"])
            trajectory = read_json(path)
            native_root = Path(cfg["paths"]["nas_root"]) / cfg["capture"]["export"]["root"] / run_id
            canonical = list(response_records(trajectory))
            if not canonical:
                raise ValueError("online canonical trajectory has no native agent responses")
            step_rows = []
            for ordinal, (message_index, message, response) in enumerate(canonical):
                if source == "baseline":
                    native, values = native_completion(native_root, response, cfg, request)
                    step_rows.append(OnlineRow(
                        native["request_id"], task_id, source, task_id,
                        tuple(native["prompt_token_ids"] + native["output_token_ids"]),
                        tuple(native["output_token_ids"]), values, None, reward_override if reward_override is not None else (1. if resolved else -1.),
                        trajectory_id=trajectory_id, agent_step=ordinal, resolved=resolved,
                        reward_reason=record.get("termination_reason") if reward_override is not None else None))
                    continue
                embedded = (message.get("extra") or {}).get("dycon_parallel")
                if not embedded:
                    raise ValueError("online DyCon response lacks complete decision evidence")
                sid, did = embedded["session_id"], embedded["decision_id"]
                source_record = {**record, "source_run_id": run_id, "rollout_id": rid,
                                 "trajectory_path": str(path), "run_directory": str(run)}
                session, reason = load_token_session(source_record, cfg, sid, deps, audit=True)
                if session is None:
                    raise ValueError("online DyCon native session unavailable: " + str(reason))
                from steering.datasets.sessions import full_decision
                decision, _ = full_decision(session, embedded)
                candidates = decision.get("candidates", [])
                if [c.get("branch_id") for c in candidates] != list(range(decision["branch_count"])):
                    raise ValueError("online DyCon decision has incomplete candidate slots")
                members = [{"historical_score": candidate.get("score"),
                                    "retained": candidate.get("retained", candidate["branch_id"] in
                                                (decision.get("consensus") or {}).get("retained_branch_ids", []))}
                           for candidate in candidates]
                advantages = advantages_for_members(members, **self.advantage_settings)
                if advantages.metadata["status"] != "ok":
                    excluded.append({"trajectory_id": trajectory_id, "task_id": task_id,
                                     "group_id": f"{trajectory_id}:{did}",
                                     "reason": advantages.metadata["skip_reason"],
                                     "decision_sha256": sha256(session["base"] / "parallel/decisions" / (did + ".json"))})
                    continue
                # Missing scores follow the existing group exclusion policy;
                # only trainable groups require complete probability vectors.
                natives = []
                for candidate in candidates:
                    native, reason = native_branch(session, decision, candidate, message_index, deps)
                    if native is None or reason:
                        raise ValueError("online DyCon candidate lacks complete native evidence: " + str(reason))
                    native, values = native_completion(native_root, candidate, cfg, request)
                    natives.append((native, values))
                for (native, values), advantage in zip(natives, advantages.advantages):
                    step_rows.append(OnlineRow(native["request_id"], task_id, source, f"{trajectory_id}:{did}",
                        tuple(native["prompt_token_ids"] + native["output_token_ids"]),
                        tuple(native["output_token_ids"]), values, None, advantage,
                        trajectory_id=trajectory_id, agent_step=ordinal, resolved=resolved,
                        reward_reason=record.get("termination_reason") if reward_override is not None else None))
            if source == "baseline":
                total = sum(len(row.completion_token_ids) for row in step_rows)
                step_rows = [replace(row, trajectory_token_count=total) for row in step_rows]
            rows.extend(step_rows)
            if resolved and training_label(record, data, rid) is True:
                sid = read_request(native_root, canonical[0][2])["session_id"]
                points.extend(self._resolved_points(cfg, run_id, sid, rid, request))
            sources.append({"trajectory_id": trajectory_id, "rollout_id": rid,
                            "trajectory_sha256": record["trajectory_sha256"],
                            "evaluation_id": record.get("selected_evaluation_id"), "resolved": resolved,
                            "reward_policy": "baseline_limit_reward_v1" if reward_override is not None else "official_raw_boolean_v1",
                            "reward_override": reward_override,
                            "verifier_audit": outcome.get("verifier_audit")})
        batch = OnlineRolloutBatch(int(request["step"]), tuple(request["tasks"]),
            request["policy_version"], request["estimator_version"], tuple(rows), tuple(points),
            request_sha256=request["request_sha256"], excluded_groups=tuple(excluded),
            discarded_trajectories=tuple(discarded))
        validate_trajectory_coverage(batch, request)
        output = run / "RL/online-batch.json"
        EstimatorDatasetStore._atomic_json(output, batch.as_dict())
        EstimatorDatasetStore._atomic_json(output.with_suffix(".receipt.json"), {
            "request_sha256": request["request_sha256"], "batch_sha256": sha256(output),
            "plan_sha256": loaded["plan"]["plan_sha256"], "sources": sources,
            "evidence_sha256": sha256(evidence.path), "native_dependencies": deps})
        return batch.as_dict()

    def _resolved_points(self, cfg, run_id, sid, rid, request):
        from enrichment.online_trace import publication_directory, verify_publication
        from hidden.artifacts import read_arrays
        from hidden.publish_finalized import publish_finalized
        from difficulty.models.estimators import transform_target
        feature = self.feature
        publication_id = "online-success-" + hash_json([request["request_sha256"], sid])[:24]
        publish_finalized(cfg, run_id, publication_id, [sid], eligibility="resolved_only",
                          coverage_scope="canonical_native")
        trace = publication_directory(cfg["project_root"], run_id, publication_id)
        native_root = Path(cfg["paths"]["nas_root"]) / cfg["capture"]["export"]["root"] / run_id
        verify_publication(trace, native_root)
        target = feature["target"]
        field = "remaining_reasoning_tokens_after" if "reasoning_tokens" in target else "remaining_steps"
        for event in read_json(trace / "index.json"):
            if event["rollout_id"] != rid or event.get("anchor") != "think_close" or event.get("censored"):
                continue
            if "fraction" in target:
                raise ValueError("online fraction targets require explicit native trajectory denominator")
            if event.get("tokenizer_fingerprint") != feature["tokenizer_fingerprint"]:
                raise ValueError("online successful anchor uses another tokenizer")
            arrays = read_arrays(native_root, event)
            site = str(feature["site"])
            vector = arrays[site]
            if vector.shape != (int(feature["width"]),):
                raise ValueError("online successful anchor width differs from initial estimator")
            value = float(transform_target([event[field]], target)[0])
            from RL.online_initial import new_point_split
            fingerprint = request["deployment"]["policy_fingerprint"]
            yield HiddenPoint(event["sample_id"], vector.tolist(), value, fingerprint,
                feature["feature_family"], cfg["model"]["fingerprint"], feature["tokenizer_fingerprint"],
                site, int(feature["width"]), feature["dtype"], birth_step=int(request["step"]) + 1,
                split=new_point_split(event["sample_id"], feature.get("new_point_split")))

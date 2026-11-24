"""Seal online rows from the existing canonical/evaluation/native capture APIs."""
from __future__ import annotations

from dataclasses import replace
import math
import time
from pathlib import Path
from typing import Any, Mapping

from core.io import hash_json, identifier, owned_path, read_json, sha256
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


def _work_result(run, item):
    return run / "rollout/tasks" / item["instance_id"] / "repetitions" / item["repetition_id"] / "result.json"


def _sampling_discard(item, result):
    from rollout.retry import online_sampling_discard
    discard = online_sampling_discard(result)
    if discard is not None and (result.get("sample_key") != item["sample_key"]
            or result.get("repetition_id") != item["repetition_id"]):
        raise ValueError("Discarded rollout sample identity mismatch")
    return discard


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
    items = loaded["work_items"]["items"]
    skipped = 0
    for item in items:
        if not item.get("online_source"):
            continue
        path = _work_result(loaded["run"], item)
        if path.is_file() and _sampling_discard(item, read_json(path)) is not None:
            skipped += 1
    if len(selected) + skipped != len(items):
        raise ValueError("online evaluation lacks a terminal result for every work item")
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
    if len(needs_grading) == len(selected) and not skipped:
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


class LocalEvidenceError(ValueError):
    """Missing trajectory data; callers must still reject inconsistent evidence."""


def native_completion(root, evidence, cfg, request, *, offline_export=False, inventories=None):
    """Read exact tokens and the generating policy's raw-T1 vector, never retokenize."""
    from hidden.artifacts import read_request
    from runtime.serving.raw_logprobs import read_values
    ledger = evidence.get("ledger_ref")
    if not offline_export and ledger and not owned_path(root, ledger).is_file():
        raise LocalEvidenceError("native_ledger_missing")
    native = (read_request(root, evidence, offline_export=True, inventories=inventories)
              if offline_export else read_request(root, evidence))
    if native.get("run_id") != cfg["run_id"]:
        raise ValueError("online native request belongs to another run")
    prompt, output = native.get("prompt_token_ids"), native.get("output_token_ids")
    if not prompt or not output:
        raise LocalEvidenceError("missing_native_tokens")
    if not offline_export and (hash_json(prompt) != native.get("request_sha256")
                               or hash_json(output) != native.get("output_sha256")):
        raise ValueError("online native tokens changed")
    descriptor = native.get("raw_t1_probability")
    if not descriptor:
        raise LocalEvidenceError("missing_raw_probabilities")
    try:
        values = read_values(root, descriptor, output_ids=output, record=native, verify_hashes=not offline_export)
    except FileNotFoundError as exc:
        if exc.filename is None or Path(exc.filename) != owned_path(root, descriptor["locator"]):
            raise
        raise LocalEvidenceError("raw_probability_shard_missing") from exc
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
    if descriptor["finite_count"] != len(values):
        raise LocalEvidenceError("nonfinite_required_probabilities")
    return native, tuple(float(x) for x in values)


def native_baseline_row(native_root, response, cfg, request, *, task_id,
                        trajectory_id, ordinal, resolved, reward_override=None,
                        reward_reason=None, tokenizers=None, **read_options):
    """Build one completion-supervised row from native rollout evidence."""
    native, values = native_completion(native_root, response, cfg, request, **read_options)
    if tokenizers is not None:
        tokenizers.add(native.get("tokenizer_fingerprint"))
    reward = reward_override if reward_override is not None else (1. if resolved else -1.)
    return OnlineRow(
        native["request_id"], task_id, "baseline", task_id,
        tuple(native["prompt_token_ids"] + native["output_token_ids"]),
        tuple(native["output_token_ids"]), values, None, reward,
        trajectory_id=trajectory_id, agent_step=ordinal, resolved=resolved,
        reward_reason=reward_reason if reward_override is not None else None)


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
            result_path = _work_result(run, item)
            result = read_json(result_path)
            discard_context = {"step": int(request["step"]), "trajectory_id": trajectory_id,
                "task_id": task_id, "source": source, "result_path": str(result_path.relative_to(run)),
                "result_sha256": sha256(result_path)}
            sampling_failure = _sampling_discard(item, result)
            if sampling_failure is not None:
                discarded.append({**discard_context, **sampling_failure,
                    "attempt_id": result.get("attempt_id"), "rollout_id": result.get("rollout_id"),
                    "session_id": result.get("capture_session_id")})
                continue
            rid = verify_final(Path(cfg["project_root"]), run_id, item, result, data)
            record = data["rollouts"][rid]
            native_root = Path(cfg["paths"]["nas_root"]) / cfg["capture"]["export"]["root"] / run_id
            stop_failure = result.get("capture_stop_retry")
            if stop_failure is not None:
                sid = identifier(result["capture_session_id"])
                receipt = native_root / "sessions" / sid / "receipt.json"
                if not receipt.is_file():
                    manifest = read_json(run / "hidden/publication.json")
                    if (manifest.get("schema_version") != 1 or manifest.get("run_id") != run_id
                            or Path(manifest["nas_locator"]).resolve() != native_root.resolve()):
                        raise ValueError("Capture stop discard publication identity mismatch")
                    entries = [entry for entry in manifest["sessions"] if entry["session_id"] == sid]
                    if len(entries) > 1 or any(entry["rollout_id"] != rid
                            or entry["trajectory_sha256"] != record["trajectory_sha256"]
                            or "receipt.json" in entry["files"] for entry in entries):
                        raise ValueError("Capture stop discard session proof mismatch")
                    discarded.append({**discard_context, **stop_failure, "stage": "evidence",
                        "retry_count": stop_failure["attempt_count"] - 1,
                        "rollout_id": rid, "session_id": sid, "missing_file": "receipt.json"})
                    continue
            from RL.online_failures import baseline_failure_reward, verifier_discard
            reward_override = baseline_failure_reward(cfg, source, record)
            outcome = _online_outcome(record, data, rid)
            if reward_override is None and verifier_discard(outcome):
                discarded.append({**discard_context, 'stage': 'evaluation', 'rollout_id': rid,
                    'retry_count': outcome['online_rl_discard']['attempt_count'] - 1,
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
            def collect():
                step_excluded, step_deps, step_points = [], {}, []
                canonical = list(response_records(trajectory))
                if not canonical:
                    raise LocalEvidenceError("missing_native_agent_responses")
                step_rows = []
                for ordinal, (message_index, message, response) in enumerate(canonical):
                    if source == "baseline":
                        step_rows.append(native_baseline_row(
                            native_root, response, cfg, request, task_id=task_id,
                            trajectory_id=trajectory_id, ordinal=ordinal, resolved=resolved,
                            reward_override=reward_override,
                            reward_reason=record.get("termination_reason")))
                        continue
                    embedded = (message.get("extra") or {}).get("dycon_parallel")
                    if not embedded:
                        raise LocalEvidenceError("missing_dycon_decision")
                    sid, did = embedded["session_id"], embedded["decision_id"]
                    source_record = {**record, "source_run_id": run_id, "rollout_id": rid,
                                     "trajectory_path": str(path), "run_directory": str(run)}
                    session, reason = load_token_session(source_record, cfg, sid, step_deps, audit=True)
                    if session is None:
                        if reason in {"unpublished_session", "publication_token_proof_incomplete",
                                      "native_session_evidence_unavailable"}:
                            raise LocalEvidenceError(reason)
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
                        step_excluded.append({"trajectory_id": trajectory_id, "task_id": task_id,
                                         "group_id": f"{trajectory_id}:{did}",
                                         "reason": advantages.metadata["skip_reason"],
                                         "decision_sha256": sha256(session["base"] / "parallel/decisions" / (did + ".json"))})
                        continue
                    # Missing scores follow the existing group exclusion policy;
                    # only trainable groups require complete probability vectors.
                    natives = []
                    for candidate in candidates:
                        native, reason = native_branch(session, decision, candidate, message_index, step_deps)
                        if native is None or reason:
                            raise LocalEvidenceError(reason or "missing_native_candidate")
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
                if resolved and training_label(record, data, rid) is True:
                    sid = read_request(native_root, canonical[0][2])["session_id"]
                    step_points = list(self._resolved_points(cfg, run_id, sid, rid, request))
                return step_rows, step_points, step_excluded, step_deps

            failures = []
            for attempt in range(2):
                try:
                    step_rows, step_points, step_excluded, step_deps = collect()
                    break
                except LocalEvidenceError as exc:
                    failures.append(str(exc))
                    if attempt == 0:
                        time.sleep(0.25)
            else:
                discarded.append({**discard_context, "stage": "evidence",
                    "reason": "native_evidence_retries_exhausted", "rollout_id": rid,
                    "session_id": result.get("capture_session_id"), "failure_codes": failures,
                    "retry_limit": 1, "attempt_count": 2, "retry_count": 1})
                continue
            rows.extend(step_rows)
            points.extend(step_points)
            excluded.extend(step_excluded)
            deps.update(step_deps)
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
        native_root = Path(cfg["paths"]["nas_root"]) / cfg["capture"]["export"]["root"] / run_id
        session_root = native_root / "sessions" / identifier(sid)
        # Check exactly the session files this publication needs. Errors in an
        # existing sealed publication or unrelated inventory still propagate.
        for name in ("receipt.json", "coverage.json", "finalization.json"):
            if not (session_root / name).is_file():
                raise LocalEvidenceError("estimator_session_" + name.removesuffix(".json") + "_missing")
        publish_finalized(cfg, run_id, publication_id, [sid], eligibility="resolved_only",
                          coverage_scope="canonical_native")
        trace = publication_directory(cfg["project_root"], run_id, publication_id)
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

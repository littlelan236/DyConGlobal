from __future__ import annotations

import concurrent.futures
import inspect
import time
from collections import Counter
from pathlib import Path

from application.execution.evidence import RunEvidence
from core.io import hash_json, now, read_json, update_status, write_json
from core.locking import exclusive_evaluation
from runtime.containers.factory import ContainerFactory, TaskContext
from runtime.containers.dependencies import classify_dependency_failure, from_exception, record_dependency_failure
from benchmark_evaluation.swebench_settlement import POLICY as SCORING_POLICY, project


FAILURE_TYPES = ("invalid_patch", "network_error", "timeout", "dependency_error", "other_error")
MAX_NETWORK_RETRIES = 2
NETWORK_RETRY_DELAY_SECONDS = 1


def classify_harness_failure(*, exception=None, evidence=""):
    """Classify unsuccessful harness execution using exception and official logs.

    Patch failures are completed negative outcomes; network failures are
    retryable ConnectionError outcomes. Unknown execution failures stay
    infra_error with an explicit other_error classification.
    """
    text = evidence or ""
    dependency = from_exception(exception) if exception is not None else None
    dependency = dependency or classify_dependency_failure(text)
    if dependency:
        return {'failure_type': 'dependency_error', 'failure_detail': dependency['detail'],
                'dependency_failure': dependency}
    if exception is not None:
        text = f"{text}\n{exception!r}\n{exception}"
    lowered = text.lower()
    if any(marker in lowered for marker in (
        "patch apply failed", "only garbage was found in the patch input",
        "failed to apply patch", "patch failed", "missingpatch",
    )):
        failure_type = "invalid_patch"
    elif any(marker in lowered for marker in (
        "timeout error:", "test timed out", "timed out after", "deadline exceeded",
    )):
        failure_type = "timeout"
    elif any(marker in lowered for marker in (
        "connectionerror", "connection reset", "readtimeout", "connecttimeout",
        "connection aborted", "remote end closed connection", "socket.gaierror",
        "httpsconnectionpool", "temporary failure in name resolution",
        "name or service not known", "network is unreachable", "connection refused",
        "docker.errors.apierror", "cannot connect to the docker daemon", "docker daemon",
        "proxyerror", "502 bad gateway", "503 service unavailable",
    )):
        failure_type = "network_error"
    else:
        failure_type = "other_error"
    detail = next((line.strip() for line in text.splitlines()
                   if line.strip() and any(marker in line.lower() for marker in (
                       "patch apply failed", "only garbage", "failed to apply patch",
                       "timeout", "timed out", "connection", "network", "error",
                   ))), "")
    return {"failure_type": failure_type, "failure_detail": detail[:1000]}


def _evaluation_evidence(output: Path, evaluation_task_id: str, rollout_id: str, instance_id: str):
    base = output / "harness" / evaluation_task_id / rollout_id / instance_id
    parts = []
    for name in ("run_instance.log", "test_output.txt"):
        path = base / name
        if path.exists():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def _dependency_evidence(cfg, output, evaluation_task_id, rollout_id, instance_id, evidence='', exception=None):
    failure = from_exception(exception) if exception is not None else None
    if failure:
        return failure
    directory = output / 'environment' / evaluation_task_id / instance_id
    path = directory / 'dependency-failure.json'
    saved = read_json(path) if path.exists() else None
    if saved and saved.get('stage') in {'preflight', 'bootstrap'}:
        return saved
    failure = classify_dependency_failure(evidence,
        wheelhouse=cfg['environment'].get('wheelhouse'),
        offline=not cfg['environment'].get('network_fallback', False),
        log_path=output / 'harness' / evaluation_task_id / rollout_id / instance_id / 'test_output.txt')
    if failure:
        context = TaskContext(cfg['evaluation_run_id'], instance_id, evaluation_task_id, 'harness', directory)
        return record_dependency_failure(cfg, context, failure)
    return saved


def install_harness_adapter(cfg: dict, factory: ContainerFactory, rows: list[dict], output: Path):
    import importlib.metadata

    from swebench.harness import run_evaluation as official
    from swebench.harness.test_spec import python as sb_python
    from swebench.harness.test_spec.test_spec import TestSpec

    if importlib.metadata.version("swebench") != "4.1.0":
        raise RuntimeError("Harness adapter is validated only against swebench 4.1.0")
    required = {"test_spec", "pred", "client", "run_id", "timeout"}
    if not required.issubset(inspect.signature(official.run_instance).parameters):
        raise RuntimeError("SWE harness interface changed")
    instances = {r["instance_id"]: r for r in rows}
    saved_official = {
        key: getattr(official, key)
        for key in ["build_container", "cleanup_container", "RUN_EVALUATION_LOG_DIR", "exec_run_with_timeout"]
    }
    original_script = TestSpec.eval_script
    script_details = {}
    original_get = sb_python.requests.get
    original_requirements = sb_python.get_requirements_by_commit

    # Requirements metadata is advisory for the project's locked prebuilt SWE
    # images. If all remote candidates fail, use the image environment and keep
    # the failure visible in the evaluation log instead of aborting spec build.
    def cached_requirements(repo, commit):
        # Network/metadata failures are infrastructure errors, never fabricated
        # empty dependency metadata. Real 404s still reach the official fallback.
        return original_requirements(repo, commit)

    # Keep exact downloaded metadata content in the project, then prefer it offline.
    cache = Path(
        cfg["environment"].get(
            "requirements_cache", str(Path(cfg["project_root"]) / "data/swebench/requirements-cache")
        )
    )
    cache.mkdir(parents=True, exist_ok=True)

    from benchmark_evaluation.metadata import make_cached_get

    cached_get = make_cached_get(original_get, cache, cfg["timeouts"].get("metadata_request_seconds", 30))

    def build_container(test_spec, client, run_id, logger, nocache, force_rebuild):
        if force_rebuild or nocache:
            raise ValueError("Project harness uses locked prebuilt images, not implicit rebuilds")
        iid = test_spec.instance_id
        ctx = TaskContext(
            cfg.get("evaluation_run_id", output.name),
            iid,
            run_id,
            "harness",
            output / "environment" / run_id / iid,
        )
        return factory.create(instances[iid], ctx)

    # This is scoped to a dedicated evaluation process, not docker SDK globally.
    official.build_container = build_container
    official.cleanup_container = lambda client, container, logger: (
        factory.close(container) if container else None
    )
    official.RUN_EVALUATION_LOG_DIR = output / "harness"
    sb_python.requests.get = cached_get
    sb_python.get_requirements_by_commit = cached_requirements
    def wrapped_script(spec):
        from benchmark_evaluation.swebench_receipts import get_test_command

        raw = original_script.fget(spec)
        script_details[spec.instance_id] = get_test_command(raw)
        return factory.policy_for(instances[spec.instance_id]).wrap_eval(raw)

    TestSpec.eval_script = property(wrapped_script)

    def execute_eval(container, cmd, timeout):
        import hashlib

        started = time.monotonic()
        out = factory.execute(container, "source /opt/miniconda3/bin/activate && " + cmd, timeout)
        # 137 proves SIGKILL, not whether timeout/OOM/cancellation caused it.
        timed_out = out["returncode"] == 124
        iid = container.instance["instance_id"]
        command_info = script_details.get(iid)
        write_json(container.context.output_dir / "eval-execution.json", {
            "schema_version": 1, "instance_id": iid,
            "evaluation_task_id": container.context.attempt_id,
            "command": cmd, "returncode": out["returncode"], "timed_out": timed_out,
            "elapsed_seconds": time.monotonic() - started,
            "output_sha256": hashlib.sha256(out["output"].encode()).hexdigest(),
            "test_command_sha256": command_info["command_sha256"] if command_info else None,
            "test_command": command_info, "at": now(),
        })
        if not timed_out:
            try:
                factory.probe(container, "environment-after-eval.json")
            except Exception as exc:
                write_json(container.context.output_dir / "post-eval-probe-error.json", {"error": str(exc)})
                raise
        return out["output"], timed_out, time.monotonic() - started

    official.exec_run_with_timeout = execute_eval

    def restore():
        for key, value in saved_official.items():
            setattr(official, key, value)
        sb_python.requests.get = original_get
        sb_python.get_requirements_by_commit = original_requirements
        TestSpec.eval_script = original_script

    official.dycon_restore = restore
    return official


@exclusive_evaluation
def evaluate(cfg, run, eval_id, gold=False, factory=None, incremental=False):
    from core.io import owned_path, sha256

    output = Path(run) / "harness"
    if incremental:
        output = output / "evaluations" / eval_id
    # Snapshot selection is authoritative even if the live manifest has since changed.
    selected = cfg.get("selection")
    if incremental and selected:
        selected = {**selected, "rollouts": {rid: row for rid, row in selected["rollouts"].items()
                    if row.get("evaluation_status") != "completed"}}
        selected["rollout_ids"] = sorted(selected["rollouts"])
        selected["tasks"] = {row["task_key"]: selected["tasks"][row["task_key"]]
                              for row in selected["rollouts"].values()}
    if not selected or not selected.get("rollout_ids"):
        raise ValueError("Evaluation requires a run snapshot with explicit rollout selection")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(f"Harness output already exists: {output}")
    import hashlib
    evaluation_owner = "harness-" + hashlib.sha256(f"{run.name}:{eval_id}".encode()).hexdigest()[:24]
    cfg = {**cfg, "evaluation_run_id": evaluation_owner}
    write_json(output / "resolved-config.json", cfg)
    rows = {}
    for rid, row in selected["rollouts"].items():
        task = selected["tasks"][row["task_key"]]
        if task["instance_id"] in rows and rows[task["instance_id"]] != task["instance"]:
            raise ValueError("Same instance ID has incompatible benchmark definitions")
        rows[task["instance_id"]] = task["instance"]
        patch = owned_path(cfg["project_root"], row["patch_path"])
        if sha256(patch) != row["patch_sha256"]:
            raise ValueError("Patch changed after run selection was frozen")
    factory = factory or ContainerFactory(cfg)
    official = None
    outcomes = {}
    update_status(output / "status.json", "evaluation")
    try:
        factory.lock_images(list(rows.values()))
        official = install_harness_adapter(cfg, factory, list(rows.values()), output)

        def worker(rid, row):
            iid = selected["tasks"][row["task_key"]]["instance_id"]
            patch = (
                rows[iid].get("patch", "")
                if gold
                else owned_path(cfg["project_root"], row["patch_path"]).read_text()
            )
            result = {
                "patch_sha256": hash_json(patch) if gold else row["patch_sha256"],
                "status": "infra_error",
                "resolved": None,
            }
            if not patch.strip():
                from benchmark_evaluation.swebench_verifier import POLICY_VERSION, digest
                empty = {
                    **result, "status": "completed", "resolved": False,
                    "reason": "MissingPatch", **classify_harness_failure(evidence="MissingPatch"),
                    "attempts": [],
                }
                audit = {"policy_version": POLICY_VERSION,
                    "official_result": {"completed": False, "resolved": None},
                    "classification": "VerifierError" if gold else "ValidOutcome",
                    "reason_code": "gold_reference_patch_missing" if gold else "missing_patch",
                    "effective_resolved": None if gold else False,
                    "evidence_digest": digest({"instance_id": iid, "patch_sha256": result["patch_sha256"], "empty": True}),
                    "cause_confirmed": True, "diagnostic_status": "confirmed"}
                audit_path = output / "verifier" / "submission" / f"{rid}.json"
                write_json(audit_path, audit)
                empty = project(empty, audit)
                empty["verifier_audit"].update(path=str(audit_path), sha256=sha256(audit_path))
                return rid, empty
            prediction = {"instance_id": iid, "model_name_or_path": rid, "model_patch": patch}
            timeout = rows[iid].get("evaluation_timeout_seconds", cfg["timeouts"]["evaluation_seconds"])
            attempts = []
            from RL.online_failures import failure_policy, verifier_error
            online_retries = failure_policy(cfg).get('verifier_error_retries')
            retry_limit = MAX_NETWORK_RETRIES if online_retries is None else online_retries
            for attempt in range(retry_limit + 1):
                identity = {"evaluation": eval_id, "rollout": rid}
                if attempt:
                    identity["attempt"] = attempt + 1
                evaluation_task_id = "task-" + hash_json(identity)[:24]
                started_at = now()
                graded = None
                try:
                    factory.deadline.check()
                    # Spec construction can fetch metadata and must also be retried.
                    spec = official.make_test_spec(rows[iid], namespace="swebench")
                    graded = official.run_instance(
                        spec, prediction, False, False, factory.client,
                        evaluation_task_id, timeout,
                    )
                    logs = _evaluation_evidence(output, evaluation_task_id, rid, iid)
                    dependency = _dependency_evidence(cfg, output, evaluation_task_id, rid, iid, logs)
                    if dependency:
                        attempt_result = {'failure_type': 'dependency_error',
                            'failure_detail': dependency['detail'], 'dependency_failure': dependency}
                    elif graded["completed"]:
                        attempt_result = {"status": "completed", "resolved": bool(graded["resolved"])}
                    else:
                        attempt_result = classify_harness_failure(
                            evidence=logs)
                except Exception as exc:
                    # Classify the exception separately: an old git-apply log must
                    # not turn a later connection failure into a patch failure.
                    attempt_result = classify_harness_failure(exception=exc)
                    dependency = _dependency_evidence(cfg, output, evaluation_task_id, rid, iid, exception=exc)
                    if dependency:
                        attempt_result.update(failure_type='dependency_error',
                            failure_detail=dependency['detail'], dependency_failure=dependency)
                    attempt_result["error"] = repr(exc)

                failure_type = attempt_result.get("failure_type")
                if failure_type == "invalid_patch":
                    attempt_result.update(status="completed", resolved=False)
                elif failure_type == "network_error":
                    attempt_result.update(status="ConnectionError", resolved=None)
                elif failure_type:
                    attempt_result.update(status="infra_error", resolved=None)
                # Retry policy uses physical execution facts, before projecting
                # score eligibility. A failed retry must not contaminate a later
                # success with stale audit, status or dependency fields.
                retryable = attempt_result['status'] == 'ConnectionError' or (
                    attempt_result.get('dependency_failure', {}).get('reason') == 'dependency_download_network_error')
                from benchmark_evaluation.swebench_audit import collect
                from benchmark_evaluation.swebench_verifier import classify

                audit_input = collect(output, evaluation_task_id, rid, rows[iid],
                                      {"patch_sha256": result["patch_sha256"], **attempt_result,
                                       **({"official_result": {"completed": graded["completed"],
                                                              "resolved": graded["resolved"]}}
                                          if graded is not None else {})}, gold=gold)
                audit = classify(audit_input)
                audit_path = output / "verifier" / evaluation_task_id / f"{rid}.json"
                write_json(audit_path, audit)
                attempt_result = project(attempt_result, audit)
                attempt_result["verifier_audit"].update(path=str(audit_path), sha256=sha256(audit_path))
                attempts.append({
                    "attempt": attempt + 1, "evaluation_task_id": evaluation_task_id,
                    "started_at": started_at, "finished_at": now(), **attempt_result,
                })
                # Replace the final outcome so successful retries cannot retain
                # stale network failure_type/error fields.
                result = {"patch_sha256": result["patch_sha256"], **attempt_result}
                if online_retries is not None:
                    retryable = verifier_error(attempt_result)
                if not retryable or attempt == retry_limit:
                    break
                time.sleep(NETWORK_RETRY_DELAY_SECONDS * (attempt + 1))
            if online_retries is not None and verifier_error(result):
                result['online_rl_discard'] = {
                    'reason': 'verifier_error_retries_exhausted',
                    'retry_limit': online_retries, 'attempt_count': len(attempts)}
            result["attempts"] = attempts
            return rid, result

        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["execution"]["harness_workers"]) as pool:
            futures = [pool.submit(worker, rid, row) for rid, row in selected["rollouts"].items()]
            for future in concurrent.futures.as_completed(futures):
                rid, result = future.result()
                outcomes[rid] = result
                write_json(output / "progress.json", outcomes)
                if result.get('failure_type') == 'dependency_error':
                    update_status(output / 'status.json', 'evaluation',
                        dependency_failure_count=sum(x.get('failure_type') == 'dependency_error' for x in outcomes.values()))
        # Incremental evaluations must carry forward all prior outcomes and
        # source rows. Otherwise a later batch would hide earlier resolved
        # trajectories from run-level evidence and cross-run comparison.
        cumulative_outcomes = dict(outcomes)
        historical_rows = {}
        if incremental:
            evaluation_root = output.parent
            for summary_path in evaluation_root.glob("*/summary.json"):
                if summary_path.parent == output:
                    continue
                try:
                    prior = read_json(summary_path)
                    cumulative_outcomes = {**prior.get("outcomes", {}), **cumulative_outcomes}
                    resolved_cfg = summary_path.parent / "resolved-config.json"
                    if resolved_cfg.exists():
                        prior_cfg = read_json(resolved_cfg)
                        prior_sel = prior_cfg.get("selection", {})
                        historical_rows.update(prior_sel.get("rollouts", {}))
                except (OSError, ValueError, TypeError):
                    continue
        for rid, row in list(cumulative_outcomes.items()):
            if row.get("scoring_policy") != SCORING_POLICY:
                # Carrying an old unprojected score into a new incremental
                # evaluation must not bypass current settlement. Re-evaluation
                # can replace it; historical artifacts remain unchanged.
                cumulative_outcomes[rid] = project(row, {
                    "classification": "Review", "reason_code": "legacy_unsettled_evaluation",
                    "effective_resolved": None, "policy_version": None,
                    "official_result": {"completed": row.get("status") == "completed",
                                        "resolved": row.get("resolved")}})
        resolved = sum(row["resolved"] is True for row in cumulative_outcomes.values())
        summary = {
            "evaluation_id": eval_id,
            "gold": gold,
            "incremental": bool(incremental),
            "requested": len(cumulative_outcomes),
            "resolved": resolved,
            "network_retries": sum(max(0, len(row.get("attempts", [])) - 1) for row in cumulative_outcomes.values()),
            "resolve_rate": resolved / len(cumulative_outcomes) if cumulative_outcomes else None,
            "graded": sum(row["status"] == "completed" for row in cumulative_outcomes.values()),
            "failure_counts": dict(Counter(row["failure_type"] for row in cumulative_outcomes.values()
                                            if row.get("failure_type"))),
            "dependency_failure_count": sum(row.get("failure_type") == "dependency_error" for row in cumulative_outcomes.values()),
            "dependency_reason_counts": dict(Counter(row["dependency_failure"]["reason"] for row in cumulative_outcomes.values()
                                                      if row.get("dependency_failure"))),
            "verifier_audit_mode": "settlement",
            "scoring_policy": SCORING_POLICY,
            "scoring_eligible": sum(row.get("scoring_eligible") is True for row in cumulative_outcomes.values()),
            "reward_unknown": sum(row.get("resolved") is None for row in cumulative_outcomes.values()),
            "verifier_classification_counts": dict(Counter(
                row.get("verifier_audit", {}).get("classification", "NotAudited")
                for row in cumulative_outcomes.values())),
            "outcomes": cumulative_outcomes,
            "finished_at": now(),
        }
        write_json(output / "summary.json", summary)
        if not gold:
            evidence = RunEvidence(cfg["project_root"], run.name)
            with evidence.transaction() as data:
                all_rows = {**historical_rows, **selected["rollouts"]}
                all_tasks = dict(selected.get("tasks", {}))
                for row in historical_rows.values():
                    task_key = row["task_key"]
                    if task_key not in all_tasks:
                        prior_cfg = next((read_json(p.parent / "resolved-config.json") for p in (output.parent).glob("*/summary.json") if p.parent / "resolved-config.json" in [p.parent / "resolved-config.json"]), None)
                        if prior_cfg:
                            all_tasks.update(prior_cfg.get("selection", {}).get("tasks", {}))
                for rid, row in all_rows.items():
                    if rid not in cumulative_outcomes:
                        continue
                    if rid not in data["rollouts"]:
                        imported = dict(row, run_id=run.name, evaluation_status="not_run", resolved=None, selected_evaluation_id=None)
                        imported["provenance"] = {**row.get("provenance", {}), "source_run_id": row["run_id"]}
                        data["rollouts"][rid] = imported
                        if row["task_key"] in all_tasks:
                            data["tasks"][row["task_key"]] = all_tasks[row["task_key"]]
            evidence.record_evaluation(
                eval_id, output / "summary.json", cumulative_outcomes,
                {"harness": "swebench==4.1.0", "run_id": run.name,
                 "config_hash": cfg["config_hash"], "source_run_id": selected.get("source_run_id", run.name)},
            )
        update_status(
            output / "status.json", "complete" if summary["graded"] == summary["requested"] else "incomplete",
            dependency_failure_count=summary['dependency_failure_count']
        )
        return summary
    except BaseException as exc:
        update_status(output / "status.json", "failed", error=str(exc))
        raise
    finally:
        try:
            factory.cleanup_owned(evaluation_owner)
        finally:
            if official is not None:
                official.dycon_restore()

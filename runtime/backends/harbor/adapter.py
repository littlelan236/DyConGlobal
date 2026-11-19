"""One DyCon physical attempt to one isolated Harbor Trial, with shared bridge."""
from __future__ import annotations

import copy
import socket
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from benchmark_evaluation.harbor import trial_outcome
from core.cancellation import Cancelled
from core.deadline import Deadline
from core.io import hash_json, now, read_json, sha256, write_json
from runtime.backends.base import BackendAdapter
from runtime.backends.contracts import PreflightFacts
from runtime.backends.harbor.task_dependencies import apt_mirror_warmup_required
from runtime.serving.capture_lifecycle import drain_capture
from runtime.serving.diagnostics import sanitize_error_text


def _agent_exit_status(limit_reason, result, *, terminalbench):
    if limit_reason in {"reasoning_loop", "native_reasoning_loop",
                        "repeated_reasoning_action_loop", "all_parallel_branches_reasoning_loop"}:
        return "reasoning_loop"
    if limit_reason:
        return "LimitsExceeded"
    exception = result.get("exception_info")
    if terminalbench and (exception or {}).get("exception_type") == "AgentTimeoutError":
        return "AgentTimeout"
    return "OpenHandsError" if exception else "AgentCompleted"


def _attempt_status(cfg, terminal, *, evaluation_status=None, capture_incomplete=False):
    from rollout.retry import disposition

    # A rejected generation belongs to this attempt. Its score and capture are
    # still recorded, but the scheduler must be able to select a fresh seed.
    if disposition(cfg, terminal) == "retry_pending":
        return terminal
    if capture_incomplete:
        return "CaptureError"
    if evaluation_status in {"unknown", "Review"}:
        return "Review"
    if evaluation_status == "infra_error":
        return "VerifierError"
    return terminal


def _capture_error(stage, exc):
    diagnostic = getattr(exc, "capture_diagnostic", None)
    return {"stage": stage, "error_type": type(exc).__name__,
            "summary": sanitize_error_text(exc), "request": diagnostic}


def worker_request(cfg, run_id, factory, task, binding, output, operation="execute", channel_dir=None):
    from runtime.backends.harbor.policy import execution_policy, phase_budgets
    policy = execution_policy(cfg)
    profile = cfg["execution_context"]["profile"]
    requested_cpu = task["instance"]["metadata"]["environment"].get("cpus", 1)
    if not factory.cpu_supported and cfg["harbor"]["cpu_quota"] == "strict":
        raise ValueError("Rootless Docker cannot enforce Harbor task CPU quota; an explicit deviation profile is required")
    cpu_mode = "limit" if factory.cpu_supported else "ignore"
    owner = {"run": run_id, "task": task["instance_id"], "sample": binding["sample_key"],
             "attempt": binding["attempt_id"]}
    trial_name = "dycon-hb-" + hash_json({
        "ownership": owner, "operation": operation, "factory_owner": factory.owner_id,
    })[:32]
    task_python_index = cfg.get("harbor", {}).get("task_python_index")
    task_apt_mirror = cfg.get("harbor", {}).get("task_apt_mirror")
    apt_warmup = apt_mirror_warmup_required(task) if task_apt_mirror else False
    task_wheelhouse = None
    task_wheelhouse_manifest_sha256 = None
    configured_wheelhouse = cfg.get("environment", {}).get("wheelhouse")
    if task_python_index and configured_wheelhouse:
        task_wheelhouse_path = Path(configured_wheelhouse)
        if not task_wheelhouse_path.is_absolute():
            task_wheelhouse_path = Path(cfg["project_root"]) / task_wheelhouse_path
        task_wheelhouse_path = task_wheelhouse_path.resolve()
        manifest_path = task_wheelhouse_path / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Configured Harbor task wheelhouse has no manifest.json")
        manifest = read_json(manifest_path)
        expected = manifest.get("manifest_sha256")
        if not isinstance(expected, str) or hash_json({k: v for k, v in manifest.items()
                                                       if k != "manifest_sha256"}) != expected:
            raise ValueError("Configured Harbor task wheelhouse manifest is invalid")
        task_wheelhouse = str(task_wheelhouse_path)
        task_wheelhouse_manifest_sha256 = sha256(manifest_path)
    resource_cache = None
    cache_path = cfg.get("harbor", {}).get("task_resource_cache")
    if cache_path:
        from runtime.backends.harbor.resource_cache import validate_descriptor
        if sha256(cache_path) != cfg["harbor"].get("task_resource_cache_sha256"):
            raise ValueError("Resource cache descriptor changed after run creation")
        resource_cache = validate_descriptor(cache_path)
    preparation = cfg.get("harbor", {}).get("task_preparation") or {}
    task_preparation = {**preparation.get("*", {}), **preparation.get(task["instance_id"], {})}
    env = {"task_preparation": task_preparation, "task_resource_cache": resource_cache, "images": factory.images, "owner_id": factory.owner_id, "benchmark": cfg["benchmark"]["backend"],
           "ownership": owner, "run_id": run_id, "trial_name": trial_name, "terminal_policy": policy,
           "sample_key": binding["sample_key"], "attempt_id": binding["attempt_id"],
           "output": str(output), "dependency_snapshot": profile["dependency_snapshot"],
           "task_python_index": task_python_index, "task_wheelhouse": task_wheelhouse,
           "task_wheelhouse_manifest_sha256": task_wheelhouse_manifest_sha256,
           "task_apt_mirror": task_apt_mirror, "task_apt_mirror_warmup": apt_warmup,
           "channel_dir": str(channel_dir) if channel_dir else None, "cpu_enforcement_policy": cpu_mode,
           "resources": {"requested_cpus": requested_cpu, "applied_cpus": requested_cpu if factory.cpu_supported else None,
                         "memory_enforcement": "limit", "cpu_deviation": cpu_mode == "ignore"}}
    agent_seconds = task["instance"]["metadata"].get("agent", {}).get("timeout_sec", 600)
    return {"operation": operation, "task": task, "binding": binding, "attempt_id": binding["attempt_id"],
            "output": str(output), "environment": env, "trial_name": trial_name,
            "disable_verifier": operation == "probe" or (operation == "execute" and task["benchmark"] == "swebench_pro_v2"),
            "driver": {"cfg": cfg, "ownership": owner, "output": str(output), "agent_environment": {},
                       "agent_seconds": agent_seconds if policy else max(1, agent_seconds - 60),
                       "phase_budgets": phase_budgets(cfg, task["instance"]["metadata"]),
                       "probe": operation == "probe",
                       "probe_host": socket.gethostbyname(socket.gethostname())}}


def run_worker_bridge(context, task, binding, output, runtime, operation="execute", capture=None):
    from rollout.adapters.openhands_bridge import OpenHandsBridge
    from runtime.agents.model_socket import UnixModelChannel
    from runtime.agents.openhands import driver_environment

    cfg = copy.deepcopy(context.cfg)
    cfg["generation"]["seed"] = binding.get("effective_seed", cfg["generation"]["seed"])
    metadata = task["instance"]["metadata"]
    seconds = metadata.get("agent", {}).get("timeout_sec", 600)
    seconds += metadata.get("environment", {}).get("build_timeout_sec", 600)
    seconds += metadata.get("verifier", {}).get("timeout_sec", 600) + 120
    from runtime.backends.harbor.policy import phase_budgets
    budgets = phase_budgets(cfg, metadata)
    if budgets:
        seconds = sum(budgets.values())
    deadline = Deadline(min(seconds, context.factory.deadline.remaining(seconds)))
    # Unix socket addresses have a 108-byte kernel limit; run/task paths are
    # deliberately long. Keep the ephemeral endpoint in the project's temp/.
    channel_dir = Path(cfg["project_root"]) / "temp/model-channels" / hash_json({"binding": binding, "output": str(output)})[:16]
    with OpenHandsBridge(cfg, binding, cfg["execution_context"]["profile"], output, deadline, runtime, capture) as bridge:
        endpoint = urlsplit(bridge.base_url)
        with UnixModelChannel(channel_dir / "model.sock", endpoint.hostname, endpoint.port):
            request = worker_request(cfg, context.run_id, context.factory, task, binding, output, operation, channel_dir)
            env = driver_environment(cfg, bridge.base_url, "", deadline)
            env.pop("BENCHFLOW_PROVIDER_API_KEY")
            request["driver"]["agent_environment"] = env
            response = context.factory.execute_worker(request, secret=bridge.secret,
                cancellation=runtime.get("cancellation"), seconds=deadline.remaining(seconds))
    return response, bridge


class HarborAdapter(BackendAdapter):
    def make_factory(self, cfg, run_id):
        from runtime.backends.harbor.factory import HarborFactory
        return HarborFactory(cfg, run_id)

    def startup(self, context, cfg, task, runtime):
        # ACP/tool and offline model-channel initialization is covered by the
        # no-model preflight. Model requests use the shared authenticated bridge.
        pass

    def preflight(self, context, tasks):
        from catalog.harbor import verify_package, execution_context
        from core.preflight import long_checks_enabled, timed_check
        from runtime.agents.dependencies import verify_snapshot
        from runtime.backends.harbor.dependencies import verify_runtime

        cfg = context.cfg
        deep = long_checks_enabled(cfg)
        checks = {}
        blockers, receipts = [], []
        run = self.run_path(context)
        record = {"passed": False, "blockers": blockers, "tasks": receipts,
                  "long_checks": deep, "checks": checks}
        try:
            # Reconstruct the frozen context with shallow checks first. This
            # retains profile/runtime/manifest identity and capability validation.
            with timed_check(checks, "execution_context"):
                actual = execution_context({**cfg, "execution": {
                    **cfg.get("execution", {}), "preflight": {"long_checks": False}}}, tasks)
                if actual != cfg["execution_context"]:
                    raise ValueError("Harbor execution context differs from frozen identity")
            model_receipt = run / "validation/startup-model-identity.json"
            if model_receipt.exists():
                record["runner_model_identity"] = read_json(model_receipt)
                if "model_sha" in record["runner_model_identity"]:
                    checks["model_sha"] = record["runner_model_identity"]["model_sha"]
            if "model_sha" not in checks:
                from core.model_identity import model_identity_verification_enabled, verify_model
                if model_identity_verification_enabled(cfg):
                    with timed_check(checks, "model_sha", enabled=deep):
                        verify_model(cfg, deep=deep)
                else:
                    checks["model_sha"] = {"status": "skipped", "reason": "model.verify_identity=false",
                                           "seconds": 0.0, "calls": 0}
            with timed_check(checks, "cpu_quota"):
                if (not context.factory.cpu_supported and cfg["harbor"]["cpu_quota"] == "strict"
                        and any(t["instance"]["metadata"]["environment"].get("cpus", 1) > 0
                                for t in tasks.values())):
                    raise ValueError("Rootless Docker cannot enforce Harbor task CPU quota")
            with timed_check(checks, "task_package_sha", enabled=deep):
                for task in tasks.values():
                    verify_package(task, deep=deep)
            with timed_check(checks, "dependency_snapshot_sha", enabled=deep):
                verify_snapshot(cfg["execution_context"]["profile"]["dependency_snapshot"], deep=deep)
            with timed_check(checks, "worker_runtime_sha", enabled=deep):
                verify_runtime((Path(cfg["project_root"]) / cfg["harbor"]["runtime_ref"]).parent, deep=deep)
            with timed_check(checks, "oci_deep_verify", enabled=deep):
                context.factory.lock_images([t["instance"] for t in tasks.values()], deep=deep)
        except Exception as exc:
            blockers.append("Harbor startup validation failed: " + str(exc))
            write_json(run / "validation/harbor-preflight.json", record)
            return PreflightFacts(False, tuple(blockers), record)
        if not deep:
            with timed_check(checks, "environment_audit", enabled=False):
                pass
            record["passed"] = True
            write_json(run / "validation/harbor-preflight.json", record)
            return PreflightFacts(True, (), record)
        with timed_check(checks, "environment_audit"):
            for key, task in tasks.items():
                output = self.run_path(context) / "validation/harbor" / key / uuid.uuid4().hex[:12]
                output.mkdir(parents=True, exist_ok=True)
                binding = {"run_id": context.run_id, "task_key": key, "sample_key": "preflight-" + key[:16],
                           "attempt_id": "preflight", "profile_sha256": context.cfg["execution_context"]["profile_sha256"]}
                try:
                    response, bridge = run_worker_bridge(context, task, binding, output, {}, "probe")
                    if response["result"].get("exception_info") or bridge.records:
                        raise ValueError("Harbor environment probe failed: " +
                                         str(response["result"].get("exception_info", {}).get("exception_message")) +
                                         "; see " + str(output / "worker-response.json"))
                    network = output / "network-probe.json"
                    if task["benchmark"] == "swebench_pro_v2":
                        if not network.is_file() or not read_json(network).get("verified"):
                            raise ValueError("Harbor offline network policy probe failed; see " + str(network))
                    from runtime.backends.harbor.policy import execution_policy
                    if execution_policy(context.cfg):
                        channel = output / "model-channel-probe.json"
                        if not channel.is_file() or not read_json(channel).get("verified"):
                            raise ValueError("Terminal model-channel probe failed; see " + str(channel))
                    receipts.append({"task_key": key, "path": str(output / "worker-response.json"),
                                     "sha256": sha256(output / "worker-response.json"), "model_requests": 0})
                except Exception as exc:
                    blockers.append(task["instance_id"] + ": " + str(exc))
        # Keep a failed audit distinct from a verified or intentionally skipped audit.
        checks["environment_audit"]["status"] = "failed" if blockers else "verified"
        record["passed"] = not blockers
        write_json(self.run_path(context) / "validation/harbor-preflight.json", record)
        return PreflightFacts(not blockers, tuple(blockers), record)

    def execute_attempt(self, context, attempt):
        from rollout.adapters.openhands import canonical_trajectory
        from rollout.artifacts import import_rollout
        from rollout.retry import counts_recorded, disposition, publish_once, record_agent_outcome

        cfg, task, output = context.cfg, attempt.task, attempt.output
        output.mkdir(parents=True, exist_ok=True)
        binding = {**attempt.context, "profile_sha256": cfg["execution_context"]["profile_sha256"],
                   "backend_family": "harbor"}
        started = now()
        publish_once(output / "attempt-started.json", {"attempt_id": binding["attempt_id"], "started_at": started,
                     "retry_kind": attempt.runtime.get("retry_kind", "initial")})
        capture_client = capture = None
        capture_status = "disabled"
        bridge = None
        try:
            if cfg.get("capture", {}).get("enabled"):
                from hidden.backends.vllm.online import CaptureClient, capture_request_metadata
                inventory = attempt.runtime["capture_token_inventory"]
                capture = capture_request_metadata("cap-" + attempt.sample.sample_key + "-" + binding["attempt_id"],
                    task["instance_id"], context.run_id, "pending-" + binding["attempt_id"], 0, cfg["capture"]["anchors"],
                    newline_token_ids=inventory["newline_token_ids"], close_token_ids=inventory["close_token_ids"],
                    tokenizer_fingerprint=inventory["tokenizer_fingerprint"], newline_policy=cfg["capture"]["newline_policy"],
                    capture_select=inventory["capture_select"])
                capture_client = CaptureClient(attempt.runtime["capture_control_url"], timeout=cfg["capture"]["control_seconds"])
                capture_client.start(capture, graph_mode="FULL", prefix_cache_policy="safe_boundary")
                capture_status = "collecting"
                write_json(output / "capture-session.json", {"session_id": capture["session_id"], "metadata": capture})
            response, bridge = run_worker_bridge(context, task, binding, output, attempt.runtime, capture=capture)
            result = response["result"]
            terminalbench = task["benchmark"] == "terminalbench_2_1"
            terminal = _agent_exit_status(bridge.limit_reason, result, terminalbench=terminalbench)
            if counts_recorded(self.run_path(context)):
                record_agent_outcome(self.run_path(context), output, attempt.context, terminal)
            events_path = output / "openhands/acp-events.json"
            events = read_json(events_path).get("events", []) if events_path.exists() else []
            canonical = canonical_trajectory(task["instance_id"], binding, bridge.records, events, terminal)
            canonical["info"]["dycon"]["attempt_context"] = attempt.context
            spec = {}
            if task["benchmark"] == "swebench_pro_v2":
                patch = output / "model.patch"
                if not patch.is_file():
                    raise ValueError("Pro patch transport missing; cannot substitute an empty patch")
                canonical["info"]["submission"] = patch.read_text()
                write_json(output / "patch-binding.json", {"binding": binding, "sha256": sha256(patch), "size": patch.stat().st_size})
            else:
                proof = self._live_proof(task, binding, output, response)
                spec = {"subject": str(proof), "subject_sha256": sha256(proof)}
            raw = output / "openhands/canonical.traj.json"
            write_json(raw, canonical)
            capture_errors = []
            capture_diagnostics = []
            if capture_client:
                capture_stage = "drain"
                try:
                    drain_capture(cfg, output, capture_client, capture["session_id"], capture_diagnostics,
                                  serving_endpoint=attempt.runtime.get("serving_worker_endpoint"))
                    capture_stage = "prepare"
                    prepared = capture_client.prepare(capture["session_id"], raw)
                    write_json(output / "capture-prepare.json", prepared)
                except Exception as exc:
                    capture_errors.append(capture_stage + ": " + type(exc).__name__)
                    capture_diagnostics.append(_capture_error(capture_stage, exc))
                    capture_status = "partial"
            rid = import_rollout(cfg["project_root"], {"source": str(raw), "task": task, "run_id": context.run_id,
                "attempt_id": binding["attempt_id"], "purposes": ["baseline"], "agent": "openhands",
                "model_fingerprint": cfg["model"]["fingerprint"], "sample": attempt.sample.as_dict(),
                "attempt_context": attempt.context, "run_local": True, **spec})
            outcome = None
            if task["benchmark"] == "terminalbench_2_1":
                outcome = trial_outcome(result, task=task, attempt_id=binding["attempt_id"],
                    subject=spec["subject_sha256"], evidence_dir=output, operation="execute")
                report = output / "evaluation-report.json"
                write_json(report, outcome)
            if capture_client and not capture_errors:
                try:
                    finalized = capture_client.finalize(capture["session_id"], rollout_id=rid,
                        trajectory_path=str(raw), trajectory_sha256=sha256(raw),
                        model_fingerprint=cfg["model"]["fingerprint"])
                    capture_status = finalized["capture_status"]
                    write_json(output / "capture-finalization.json", finalized)
                    if capture_status not in {"complete", "not_exported"}:
                        capture_errors.append("finalize returned " + capture_status)
                except Exception as exc:
                    capture_errors.append("finalize: " + type(exc).__name__)
                    capture_diagnostics.append(_capture_error("finalize", exc))
                    capture_status = "partial"
            if capture_diagnostics:
                write_json(output / "capture-errors.json", capture_diagnostics)
            if outcome is not None:
                from application.execution.evidence import RunEvidence
                evaluation_id = "hb-" + hash_json({"binding": binding, "report": sha256(report)})[:24]
                RunEvidence(cfg["project_root"], context.run_id).record_evaluation(evaluation_id, report,
                    {rid: {**outcome, "subject_sha256": spec["subject_sha256"],
                           "status": "Review" if outcome["status"] == "unknown" else outcome["status"]}},
                    {"benchmark": task["benchmark"], "live_verifier": True, "trial_id": response["trial_id"]})
            status = _attempt_status(cfg, terminal,
                evaluation_status=outcome["status"] if outcome else None,
                capture_incomplete=bool(capture_errors) and cfg["capture"]["failure_policy"] == "block_rollout")
            calls = [c for r in bridge.records for c in r.get("model_calls", [])]
            known = [c.get("usage") or {} for c in calls if c.get("usage_known")]
            final = {**attempt.sample.as_dict(), **attempt.context, "rollout_id": rid, "exit_status": status,
                     "agent_exit_status": terminal, "disposition": disposition(cfg, status),
                     "started_at": started, "finished_at": now(), "resolved": outcome["resolved"] if outcome else None,
                     "evaluation_outcome": outcome, "capture_status": capture_status, "capture_errors": capture_errors,
                     "cleanup_outcome": read_json(output / "cleanup.json"), "trial_id": response["trial_id"],
                     "usage": {"prompt_tokens": sum(r.get("prompt_tokens", 0) for r in known),
                               "completion_tokens": sum(r.get("completion_tokens", 0) for r in known),
                               "complete": len(known) == len(calls)}, "timings": {}}
            write_json(output / "attempt.json", final)
            return final
        except Exception as exc:
            if capture_client and not (output / "capture-drain.json").exists():
                try:
                    capture_client.stop(capture["session_id"])
                except Exception as capture_exc:
                    write_json(output / "capture-error-stop.json", _capture_error("failed_attempt_stop", capture_exc))
            status = "Cancelled" if isinstance(exc, Cancelled) else "EnvironmentSetupFailed"
            final = {**attempt.sample.as_dict(), **attempt.context, "exit_status": status,
                     "disposition": disposition(cfg, status), "error": str(exc), "resolved": None,
                     "capture_status": "partial" if capture else "disabled", "started_at": started, "finished_at": now()}
            write_json(output / "attempt.json", final)
            return final

    def recover_attempt(self, cfg, run_id, task, sample, context, output, runtime):
        from application.execution.evidence import RunEvidence
        from application.execution.subject import verify_subject
        from rollout.retry import disposition

        evidence = RunEvidence(cfg["project_root"], run_id).read()
        matches = [(rid, row) for rid, row in evidence["rollouts"].items()
                   if row.get("sample_key") == sample.sample_key and row.get("attempt_id") == context["attempt_id"]]
        if not matches:
            return None  # Scheduler creates a new physical attempt; no tool replay.
        if len(matches) != 1:
            raise ValueError("Ambiguous recovered Harbor physical attempt")
        rid, row = matches[0]
        verify_subject(cfg["project_root"], row)
        canonical = read_json(Path(cfg["project_root"]) / row["trajectory_path"])
        if canonical.get("info", {}).get("dycon", {}).get("attempt_context") != context:
            raise ValueError("Recovered Harbor trajectory binds another physical attempt")
        if not (output / "cleanup.json").is_file() or not read_json(output / "cleanup.json")["complete"]:
            raise ValueError("Harbor cleanup was not confirmed; reconcile owned resources first")
        terminal = canonical["info"]["exit_status"]
        evaluation = evidence["evaluations"].get(row.get("selected_evaluation_id"), {}).get("outcomes", {}).get(rid)
        evaluation_status = None
        if task["benchmark"] == "terminalbench_2_1" and row["resolved"] is None:
            evaluation_status = row.get("evaluation_status")
            if evaluation_status not in {"Review", "unknown", "infra_error"}:
                evaluation_status = "infra_error"
        capture_path = output / "capture-session.json"
        capture_status = "disabled"
        if capture_path.exists():
            from rollout.runner import _recover_capture_status
            session = read_json(capture_path)["session_id"]
            capture_status = _recover_capture_status(cfg, Path(cfg["project_root"]) / "runs" / run_id,
                                                      session, rid, row["trajectory_sha256"])
            if capture_status != "complete":
                capture_status = "pending_recovery"
        status = _attempt_status(cfg, terminal, evaluation_status=evaluation_status,
                                 capture_incomplete=capture_status == "pending_recovery")
        result = {**sample.as_dict(), **context, "rollout_id": rid, "exit_status": status,
                  "agent_exit_status": terminal,
                  "disposition": "blocked" if status == "CaptureError" else disposition(cfg, status),
                  "capture_status": capture_status, "resolved": row["resolved"], "evaluation_outcome": evaluation,
                  "recovered_without_generation": True,
                  "started_at": read_json(output / "attempt-started.json")["started_at"], "finished_at": now()}
        write_json(output / "attempt.json", result)
        return result

    @staticmethod
    def _live_proof(task, binding, output, response):
        proof_dir = output / "live-subject"
        proof_dir.mkdir(exist_ok=True)
        # These bytes prove the executed state/verifier identity; they explicitly
        # do not claim to reconstruct the discarded terminal environment.
        files = []
        for source in sorted((output / "trials").rglob("*")):
            if source.is_file() and not source.is_symlink():
                relative = source.relative_to(output)
                target = proof_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
                files.append({"path": relative.as_posix(), "sha256": sha256(target), "size": target.stat().st_size})
        write_json(proof_dir / "subject.json", {"schema_version": 1, "kind": "live_state", "binding": binding,
                   "trial_id": response["trial_id"], "task_package_sha256": task["instance"]["package_sha256"],
                   "outputs": files, "re_evaluation": False, "complete": True,
                   "re_evaluation_reason": "Live trial/verifier proof; terminal state was not exported as a replay snapshot"})
        return proof_dir / "subject.json"

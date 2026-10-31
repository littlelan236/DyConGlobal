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


def worker_request(cfg, run_id, factory, task, binding, output, operation="execute", channel_dir=None):
    profile = cfg["execution_context"]["profile"]
    requested_cpu = task["instance"]["metadata"]["environment"].get("cpus", 1)
    if not factory.cpu_supported and cfg["harbor"]["cpu_quota"] == "strict":
        raise ValueError("Rootless Docker cannot enforce Harbor task CPU quota; an explicit deviation profile is required")
    cpu_mode = "limit" if factory.cpu_supported else "ignore"
    owner = {"run": run_id, "task": task["instance_id"], "sample": binding["sample_key"],
             "attempt": binding["attempt_id"]}
    env = {"images": factory.images, "owner_id": factory.owner_id, "benchmark": cfg["benchmark"]["backend"],
           "ownership": owner, "run_id": run_id,
           "sample_key": binding["sample_key"], "attempt_id": binding["attempt_id"],
           "output": str(output), "dependency_snapshot": profile["dependency_snapshot"],
           "channel_dir": str(channel_dir) if channel_dir else None, "cpu_enforcement_policy": cpu_mode,
           "resources": {"requested_cpus": requested_cpu, "applied_cpus": requested_cpu if factory.cpu_supported else None,
                         "memory_enforcement": "limit", "cpu_deviation": cpu_mode == "ignore"}}
    agent_seconds = task["instance"]["metadata"].get("agent", {}).get("timeout_sec", 600)
    return {"operation": operation, "task": task, "binding": binding, "attempt_id": binding["attempt_id"],
            "output": str(output), "environment": env,
            "disable_verifier": operation == "probe" or (operation == "execute" and task["benchmark"] == "swebench_pro_v2"),
            "driver": {"cfg": cfg, "ownership": owner, "output": str(output), "agent_environment": {},
                       "agent_seconds": max(1, agent_seconds - 60), "probe": operation == "probe",
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
        super().preflight(context, tasks)
        from catalog.harbor import verify_package
        blockers, receipts = [], []
        for key, task in tasks.items():
            verify_package(task)
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
                receipts.append({"task_key": key, "path": str(output / "worker-response.json"),
                                 "sha256": sha256(output / "worker-response.json"), "model_requests": 0})
            except Exception as exc:
                blockers.append(task["instance_id"] + ": " + str(exc))
        record = {"passed": not blockers, "blockers": blockers, "tasks": receipts}
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
            terminal = "AgentCompleted" if not result.get("exception_info") else "OpenHandsError"
            if bridge.limit_reason:
                terminal = "reasoning_loop" if bridge.limit_reason == "reasoning_loop" else "LimitsExceeded"
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
            if capture_client:
                try:
                    # Reuse the existing shared native drain; it owns no agent loop.
                    from runtime.agents.skillsbench import drain_capture
                    drain_capture(cfg, output, capture_client, capture["session_id"])
                    capture_client.prepare(capture["session_id"], raw)
                except Exception as exc:
                    capture_errors.append(str(exc))
                    capture_status = "partial"
            rid = import_rollout(cfg["project_root"], {"source": str(raw), "task": task, "run_id": context.run_id,
                "attempt_id": binding["attempt_id"], "purposes": ["baseline"], "agent": "openhands",
                "model_fingerprint": cfg["model"]["fingerprint"], "sample": attempt.sample.as_dict(),
                "attempt_context": attempt.context, "run_local": True, **spec})
            outcome = None
            if task["benchmark"] == "terminalbench_2_1":
                outcome = trial_outcome(result, task=task, attempt_id=binding["attempt_id"], subject=spec["subject_sha256"])
                report = output / "evaluation-report.json"
                write_json(report, outcome)
                from application.execution.evidence import RunEvidence
                evaluation_id = "hb-" + hash_json({"binding": binding, "report": sha256(report)})[:24]
                RunEvidence(cfg["project_root"], context.run_id).record_evaluation(evaluation_id, report,
                    {rid: {**outcome, "subject_sha256": spec["subject_sha256"]}},
                    {"benchmark": task["benchmark"], "live_verifier": True, "trial_id": response["trial_id"]})
            if capture_client:
                try:
                    capture_status = capture_client.finalize(capture["session_id"], rollout_id=rid,
                        trajectory_path=str(raw), trajectory_sha256=sha256(raw),
                        model_fingerprint=cfg["model"]["fingerprint"])["capture_status"]
                except Exception as exc:
                    capture_errors.append(str(exc))
                    capture_status = "partial"
            status = terminal
            if outcome and outcome["resolved"] is None:
                status = "VerifierError"
            if capture_errors and cfg["capture"]["failure_policy"] == "block_rollout":
                status = "CaptureError"
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
            if capture_client:
                try:
                    capture_client.stop(capture["session_id"])
                except Exception:
                    pass
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
        capture_path = output / "capture-session.json"
        capture_status = "disabled"
        if capture_path.exists():
            from rollout.runner import _recover_capture_status
            session = read_json(capture_path)["session_id"]
            capture_status = _recover_capture_status(cfg, Path(cfg["project_root"]) / "runs" / run_id,
                                                      session, rid, row["trajectory_sha256"])
            if capture_status != "complete":
                return {**sample.as_dict(), **context, "rollout_id": rid, "exit_status": "CaptureError",
                        "disposition": "blocked", "capture_status": "pending_recovery", "resolved": row["resolved"]}
        terminal = canonical["info"]["exit_status"]
        status = terminal
        if task["benchmark"] == "terminalbench_2_1" and row["resolved"] is None:
            status = "VerifierError"
        result = {**sample.as_dict(), **context, "rollout_id": rid, "exit_status": status,
                  "agent_exit_status": terminal, "disposition": disposition(cfg, status),
                  "capture_status": capture_status, "resolved": row["resolved"], "recovered_without_generation": True,
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

"""Native SWE-bench adapter driven by the shared OpenHands bridge."""
from __future__ import annotations

from pathlib import Path

from core.cancellation import Cancelled
from core.io import now, read_json, sha256, write_json
from runtime.backends.base import BackendAdapter
from runtime.backends.contracts import PreflightFacts
from runtime.backends.native.swe_patch import publish_patch


class NativeSWEOpenHandsAdapter(BackendAdapter):
    """Owns one SWE task container and one OpenHands ACP attempt.

    The actual ACP transport is injected by the runtime worker. This keeps the
    scheduler and adapter SDK-free while allowing unit tests to exercise the
    complete evidence/patch contract without Docker or OpenHands installed.
    """

    def make_factory(self, cfg, run_id):
        from runtime.containers.factory import ContainerFactory
        return ContainerFactory(cfg)

    def preflight(self, context, tasks):
        super().preflight(context, tasks)
        from runtime.agents.swebench_openhands import validate_profile
        blockers = []
        profile = context.cfg["execution_context"]["profile"]
        try:
            validate_profile(profile)
        except Exception as exc:
            blockers.append("profile: " + str(exc))
        if (context.cfg.get("capture", {}).get("enabled") or
                context.cfg.get("parallel", {}).get("enabled")) and profile.get("request_roles") != "openhands-1.22.1":
            blockers.append("OpenHands request-role qualification is incomplete")
        receipt = {"schema_version": 1, "benchmark": "swebench", "agent": "openhands",
                   "profile_sha256": context.cfg["execution_context"]["profile_sha256"],
                   "tasks": sorted(tasks), "blockers": blockers, "passed": not blockers}
        write_json(self.run_path(context) / "validation/swebench-openhands-preflight.json", receipt)
        return PreflightFacts(not blockers, tuple(blockers), receipt)

    def execute_attempt(self, context, attempt):
        cfg, task, output = context.cfg, attempt.task, Path(attempt.output)
        output.mkdir(parents=True, exist_ok=True)
        binding = {**attempt.context, "benchmark": "swebench", "agent": "openhands",
                   "backend_family": "native_swe_openhands",
                   "profile_sha256": cfg["execution_context"]["profile_sha256"]}
        started = now()
        write_json(output / "attempt-started.json", {"attempt_id": binding["attempt_id"], "started_at": started})
        capture_client = capture = None
        capture_status = "disabled"
        try:
            if cfg.get("capture", {}).get("enabled"):
                from hidden.backends.vllm.online import CaptureClient, capture_request_metadata
                inventory = attempt.runtime["capture_token_inventory"]
                capture = capture_request_metadata(
                    "cap-" + attempt.sample.sample_key + "-" + binding["attempt_id"], task["instance_id"],
                    context.run_id, "pending-" + binding["attempt_id"], 0, cfg["capture"]["anchors"],
                    newline_token_ids=inventory["newline_token_ids"], close_token_ids=inventory["close_token_ids"],
                    tokenizer_fingerprint=inventory["tokenizer_fingerprint"],
                    newline_policy=cfg["capture"]["newline_policy"], capture_select=inventory["capture_select"])
                capture_client = CaptureClient(attempt.runtime["capture_control_url"], timeout=cfg["capture"]["control_seconds"])
                capture_client.start(capture, graph_mode="FULL", prefix_cache_policy="safe_boundary")
                capture_status = "collecting"
                write_json(output / "capture-session.json", {"session_id": capture["session_id"], "metadata": capture})

            executor = attempt.runtime.get("openhands_executor")
            if executor is None:
                from runtime.backends.native.swe_openhands_worker import execute
                executor = execute
            from core.deadline import Deadline
            from rollout.adapters.openhands_bridge import OpenHandsBridge
            deadline = Deadline(cfg["timeouts"]["model_seconds"])
            with OpenHandsBridge(cfg, binding, cfg["execution_context"]["profile"], output,
                                 deadline, attempt.runtime, capture) as bridge:
                response = executor(cfg, context, task, attempt, context.factory, capture, bridge)
                response.setdefault("records", bridge.records)
            from rollout.adapters.openhands import canonical_trajectory
            records = response.get("records", [])
            events = response.get("events", [])
            terminal = response.get("exit_status", "OpenHandsError")
            canonical = canonical_trajectory(task["instance_id"], binding, records, events, terminal)
            canonical["info"]["dycon"]["attempt_context"] = attempt.context
            patch_text = response.get("patch", "")
            profile = cfg["execution_context"]["profile"]
            patch, patch_receipt = publish_patch(
                output, patch_text, binding=binding, base_commit=task.get("base_commit") or task["instance"].get("base_commit"),
                workspace=profile["workspace"], max_bytes=profile["patch_max_bytes"])
            if not patch_receipt["empty"] and terminal in {"AgentCompleted", "OpenHandsCompleted"}:
                terminal = "Submitted"
            canonical["info"].update({"exit_status": terminal, "submission": patch_text})
            raw = output / "openhands/canonical.traj.json"
            write_json(raw, canonical)
            if capture_client and capture:
                from runtime.agents.skillsbench import drain_capture
                drain_capture(cfg, output, capture_client, capture["session_id"])
                capture_client.prepare(capture["session_id"], raw)
            from rollout.artifacts import import_rollout
            rid = import_rollout(cfg["project_root"], {"source": str(raw), "task": task,
                "run_id": context.run_id, "attempt_id": binding["attempt_id"], "purposes": ["baseline"],
                "agent": "openhands", "model_fingerprint": cfg["model"]["fingerprint"],
                "sample": attempt.sample.as_dict(), "attempt_context": attempt.context, "run_local": True})
            if capture_client and capture:
                capture_status = capture_client.finalize(capture["session_id"], rollout_id=rid,
                    trajectory_path=str(raw), trajectory_sha256=sha256(raw),
                    model_fingerprint=cfg["model"]["fingerprint"])["capture_status"]
            result = {**attempt.sample.as_dict(), **attempt.context, "rollout_id": rid,
                      "exit_status": terminal, "agent_exit_status": terminal,
                      "submission_status": patch_receipt["status"], "patch_sha256": patch_receipt["sha256"],
                      "capture_status": capture_status, "resolved": None,
                      "started_at": started, "finished_at": now(),
                      "cleanup_outcome": response.get("cleanup", {"complete": True})}
            write_json(output / "attempt.json", result)
            return result
        except Cancelled:
            raise
        except Exception as exc:
            if capture_client and capture:
                try:
                    capture_client.stop(capture["session_id"])
                except Exception:
                    pass
            result = {**attempt.sample.as_dict(), **attempt.context, "exit_status": type(exc).__name__,
                      "agent_exit_status": None, "capture_status": capture_status,
                      "resolved": None, "error": str(exc), "started_at": started, "finished_at": now()}
            write_json(output / "attempt.json", result)
            return result

    def recover_attempt(self, cfg, run_id, task, sample, context, output, runtime):
        output = Path(output)
        attempt = output / "attempt.json"
        trajectory = output / "openhands/canonical.traj.json"
        patch = output / "patch-binding.json"
        if not (attempt.exists() and trajectory.exists() and patch.exists()):
            return None
        result = read_json(attempt)
        if result.get("attempt_id") != context["attempt_id"]:
            raise ValueError("Recovered SWE/OpenHands attempt binding differs")
        result["recovered_without_generation"] = True
        return result

    def subject_spec(self, task, output, context, state):
        return {}

    def verify_subject(self, root, row):
        return None

    def subject_failed(self, root, row, state):
        return False

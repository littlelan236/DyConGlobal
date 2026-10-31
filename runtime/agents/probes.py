"""Run-scoped acceptance probes; reuse the production OpenHands bridge."""
from pathlib import Path

import requests

from application.execution.run_plan import load_plan
from core.backends import identity
from core.deadline import Deadline
from core.io import hash_json, identifier, now, read_json, write_json
from core.processes import matches
from hidden.artifacts import event_records
from rollout.adapters.openhands import CONDENSATION_PREFIX, CONDENSATION_SUFFIX
from rollout.adapters.openhands_bridge import OpenHandsBridge


def auxiliary_probe(root, run_id, probe_id):
    """Call the pinned condenser path without owning a primary capture session."""
    identifier(probe_id)
    loaded = load_plan(root, run_id)
    cfg, run = loaded["config"], loaded["run"]
    if (identity(cfg) != ("skillsbench", "openhands") or not cfg["capture"]["enabled"]
            or cfg["serving"]["management"] != "managed"):
        raise ValueError("Auxiliary probe requires a managed SkillsBench capture run")
    process = read_json(run / "serving/process.json")
    if process.get("run_id") != run_id or not matches(process):
        raise ValueError("Run-owned model process is not live")
    target = run / "validation" / ("auxiliary-" + probe_id)
    receipt = target / "receipt.json"
    if receipt.exists():
        return read_json(receipt)
    if target.exists():
        raise FileExistsError("Probe ID already contains unfinished evidence; use a new ID")
    endpoint = cfg["serving"]["base_url"].removesuffix("/v1").rstrip("/")
    nas = Path(cfg["paths"]["nas_root"]) / cfg["capture"]["export"]["root"] / run_id
    spool = Path(root) / cfg["capture"]["export"]["spool_root"] / run_id

    def events():
        # Live events may still be in the local spool. Compare immutable
        # logical identity fields, which survive NAS shard relocation.
        result = {}
        for base in (spool, nas):
            for _, row in event_records(base):
                logical = {key: row.get(key) for key in ("sample_id", "request_id", "anchor",
                           "absolute_position", "token_id", "sha256")}
                sample = row["sample_id"]
                if sample in result and result[sample] != logical:
                    raise ValueError("Conflicting spool/NAS capture sample identity")
                result[sample] = logical
        return result

    with requests.Session() as session:
        session.trust_env = False

        def state():
            response = session.get(endpoint + "/dycon/capture/engine/status", timeout=30)
            response.raise_for_status()
            value = response.json()
            if not value["enabled"] or value["failures"]:
                raise ValueError("Capture engine is disabled or has outstanding failures")
            return value

        before, before_events = state(), events()
        if not before_events:
            raise ValueError("Probe requires nonempty primary capture evidence")
        profile = cfg["execution_context"]["profile"]
        owner_execution = identifier(process['execution_attempt_id'])
        control = read_json(run / 'execution/attempts' / owner_execution / 'control.json')
        auxiliary_session = 'auxiliary-' + probe_id
        bridge = OpenHandsBridge(cfg, {"run_id": run_id, "attempt_id": "validation-" + probe_id,
                                     'session_id': auxiliary_session},
                                 profile, target, Deadline(cfg["timeouts"]["model_seconds"]),
                                 runtime={'execution_attempt_id': owner_execution,
                                          'cancel_token': control['token'], 'request_control_enabled': True})
        raw = bridge.complete({"messages": [{"role": "user", "content": CONDENSATION_PREFIX
            + "\n<EVENT>Tool results:\none\ntwo\nthree\nfour\nfive\nThe task is still running.</EVENT>\n"
            + CONDENSATION_SUFFIX}]})
        if (bridge.decisions != 0 or len(bridge.records) != 1
                or bridge.records[0]["role"] != "condensation"
                or bridge.records[0].get("capture") is not None
                or "dycon_capture" in bridge.records[0]["effective_request"]):
            raise ValueError("Auxiliary call polluted primary decision identity")
        if raw["choices"][0]["finish_reason"] != "stop" or not raw.get("usage", {}).get("completion_tokens"):
            raise ValueError("Auxiliary model did not complete a real response")
        response = session.post(endpoint + "/dycon/capture/engine/flush", json={'session_id': auxiliary_session},
                                timeout=cfg["capture"]["finalize_seconds"])
        response.raise_for_status()
        if response.json()["failures"]:
            raise ValueError("Auxiliary model caused capture drain failure")
        after, after_events = state(), events()
        broker = None
        if cfg.get("parallel", {}).get("enabled"):
            response = session.post(endpoint + "/dycon/parallel/status", json={}, timeout=30)
            response.raise_for_status()
            broker = response.json()
    if not matches(process):
        raise ValueError("Model process identity changed during the probe")
    if any(after_events.get(key) != value for key, value in before_events.items()):
        raise ValueError("Existing primary capture events changed")
    captured = [row for row in after_events.values()
                if str(row.get("request_id", "")).startswith(raw["id"])]
    if captured:
        raise ValueError("Auxiliary model produced primary capture samples")
    result = {"schema_version": 1, "run_id": run_id, "probe_id": probe_id, "at": now(),
              "config_hash": cfg["config_hash"], "serving_process": process,
              "response_id": raw["id"], "response_sha256": hash_json(raw),
              "usage": raw["usage"], "role": "condensation", "primary_decisions": bridge.decisions,
              "auxiliary_capture_samples": len(captured), "primary_events_before": len(before_events),
              "primary_events_after": len(after_events), "existing_primary_events_unchanged": True,
              "primary_event_identity_sha256": hash_json(before_events), "parallel_admission": broker,
              "engine_before_sha256": hash_json(before), "engine_after_sha256": hash_json(after),
              "verified": True}
    result["receipt_sha256"] = hash_json(result)
    write_json(target / "engine-before.json", before)
    write_json(target / "engine-after.json", after)
    write_json(receipt, result)
    return result

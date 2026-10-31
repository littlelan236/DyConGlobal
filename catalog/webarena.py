"""Pinned Verified task import, with evaluator definitions kept off the agent plane."""
from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path

from core.io import hash_json, identifier, read_json, sha256, write_json
from core.locking import file_lock

REVISION = "6473f72db5dcefc97b5725b59e734504edc28a21"
DATASET_SHA256 = "d65275660814663375028e9017e1f929e3c38321041b125795e2713b52243d30"
HARD_SHA256 = "3b0a4df231bb5a0c642215e521c3fa97701a384f52a734dc2db8f617ad0591a7"
SITES = {"shopping", "shopping_admin", "reddit", "gitlab", "map", "wikipedia", "homepage"}
PAYLOAD_FIELDS = {"source_task_id", "task_revision", "intent", "sites", "start_urls", "intent_template_id",
                  "source_task_sha256", "dataset_revision", "dataset_sha256", "hard"}

# Public FinalAgentResponse schema, projected from the pinned types/agent_response.py.
# The internal _FinalAgentResponse and expected alternatives never reach this profile.
FINAL_RESPONSE_SCHEMA = {
    "title": "FinalAgentResponse", "type": "object", "required": ["task_type", "status"],
    "properties": {
        "task_type": {"type": "string", "enum": ["RETRIEVE", "MUTATE", "NAVIGATE"]},
        "status": {"type": "string", "enum": ["SUCCESS", "ACTION_NOT_ALLOWED_ERROR", "PERMISSION_DENIED_ERROR",
            "NOT_FOUND_ERROR", "DATA_VALIDATION_ERROR", "UNKNOWN_ERROR"]},
        "retrieved_data": {"anyOf": [{"type": "array", "items": {"anyOf": [
            {"type": "string"}, {"type": "integer"}, {"type": "number"}, {"type": "boolean"},
            {"type": "object", "additionalProperties": True}, {"type": "null"}]}}, {"type": "null"}], "default": None},
        "error_details": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
    },
}

SYSTEM_TEMPLATE = """You operate a browser to complete a WebArena Verified task.
Use the bash tool only as a transport for one browser command per model response.
It is not a terminal: shell commands, scripts, pipelines, databases, administrator APIs,
and commands outside the browser protocol are unavailable. Use the observed websites.
The initial page and every action observation include an accessibility tree with
element IDs. Act only on IDs in the latest observation. Follow these commands:
browser click [ID]
browser type [ID] [text] [1] (1 presses Enter; 0 does not)
browser hover [ID]
browser press [key combination]
browser scroll [up|down]
browser goto [URL]
browser new_tab
browser tab_focus [tab index]
browser close_tab
browser go_back
browser go_forward
browser stop [JSON]
Emit exactly one tool call per response. Only the chosen action is executed.
When done, submit browser stop with an actual JSON object for task_type, status,
retrieved_data, and error_details. Choose task_type from RETRIEVE, MUTATE, NAVIGATE.
Choose status from SUCCESS, ACTION_NOT_ALLOWED_ERROR, PERMISSION_DENIED_ERROR,
NOT_FOUND_ERROR, DATA_VALIDATION_ERROR, UNKNOWN_ERROR. For RETRIEVE use retrieved_data
as an array of the results you actually found (strings, numbers, booleans or objects,
with no nested arrays). For MUTATE/NAVIGATE use null. Set error_details to null for
SUCCESS or a concise explanation for errors. Your SUCCESS is a claim, not a score.
Never invent a result or infer a reference answer. Browser action budget: {{ max_browser_actions }}.
"""

INSTANCE_TEMPLATE = """Task: {{ task }}

Initial browser observation:
{{ initial_observation }}
"""


def browser_profile():
    return {"schema_version": 1, "benchmark": "webarena_verified", "agent": "minisweagent",
            "revision": REVISION, "observation_type": "accessibility_tree", "action_protocol": "bash_browser_id_v1",
            "system_template": SYSTEM_TEMPLATE, "instance_template": INSTANCE_TEMPLATE,
            "observation_template": "<returncode>{{ output.returncode }}</returncode>\n{{ output.output }}",
            "format_error_template": "Use exactly one bash tool call containing one browser action. "
                "For example browser click [123] or browser stop [JSON_OBJECT]. {{ error }}",
            "final_response_schema": copy.deepcopy(FINAL_RESPONSE_SCHEMA),
            "schema_source": "src/webarena_verified/types/agent_response.py@" + REVISION}


def validate_profile(profile):
    if profile != browser_profile():
        raise ValueError("WebArena mini agent profile differs from the pinned browser protocol")
    return copy.deepcopy(profile)


def validate_payload(payload, instance_id, revision):
    if set(payload) != PAYLOAD_FIELDS or revision != REVISION or payload.get("dataset_revision") != revision:
        raise ValueError("Invalid pinned WebArena task payload")
    source_id = payload["source_task_id"]
    if type(source_id) is not int or not 0 <= source_id < 812 or instance_id != f"wav-{source_id:03d}":
        raise ValueError("WebArena task ID does not match its source ID")
    if type(payload["task_revision"]) is not int or payload["task_revision"] < 1:
        raise ValueError("WebArena requires a positive source task revision")
    if type(payload["intent_template_id"]) is not int or type(payload["hard"]) is not bool:
        raise ValueError("Invalid WebArena template or Hard membership")
    if not isinstance(payload["intent"], str) or not payload["intent"].strip():
        raise ValueError("WebArena task requires a nonempty intent")
    if (not isinstance(payload["sites"], list) or not payload["sites"]
            or len(set(payload["sites"])) != len(payload["sites"]) or not set(payload["sites"]) <= SITES):
        raise ValueError("Invalid WebArena sites")
    if (not isinstance(payload["start_urls"], list) or not payload["start_urls"]
            or not all(isinstance(url, str) and url for url in payload["start_urls"])):
        raise ValueError("WebArena task requires start_urls")
    if (payload["dataset_sha256"] != DATASET_SHA256
            or not re.fullmatch(r"[0-9a-f]{64}", payload["source_task_sha256"])):
        raise ValueError("Invalid WebArena source hash")


def _source(source):
    source = Path(source).resolve()
    dataset = source / "assets/dataset/webarena-verified.json" if source.is_dir() else source
    hard = dataset.parent / "subsets/webarena-verified-hard.json"
    if dataset.is_symlink() or hard.is_symlink() or sha256(dataset) != DATASET_SHA256 or sha256(hard) != HARD_SHA256:
        raise ValueError("WebArena dataset/Hard bytes do not match the pinned official revision")
    tasks, hard_ids = read_json(dataset), read_json(hard)["task_ids"]
    if len(tasks) != 812 or {task["task_id"] for task in tasks} != set(range(812)):
        raise ValueError("Expected all 812 pinned Verified tasks")
    if len(hard_ids) != 258 or len(set(hard_ids)) != 258 or not set(hard_ids) <= set(range(812)):
        raise ValueError("Expected the 258-task official Hard subset")
    return dataset, tasks, set(hard_ids)


def _task_id(value):
    if type(value) is int and 0 <= value < 812:
        return value
    if isinstance(value, str) and re.fullmatch(r"(?:wav-)?\d{1,3}", value):
        return _task_id(int(value.removeprefix("wav-")))
    raise ValueError("Invalid Verified source task ID: " + repr(value))


def _immutable_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Immutable WebArena publication differs: " + str(path))
    else:
        write_json(path, value)


def import_tasks(cfg, source=None, *, subset="full", task_ids=(), partition="test", snapshot_id=None):
    """Import Full/Hard or an explicit subset; publish all evaluator definitions privately."""
    from catalog.store import Catalog, Task

    if subset not in {"full", "hard"}:
        raise ValueError("WebArena subset must be full or hard")
    root = Path(cfg["project_root"]).resolve()
    if source is None:
        source = cfg.get("webarena_images", {}).get("source_root", "benchmarks/webarena-verified")
    dataset, official, hard_ids = _source(root / source)
    requested = [_task_id(value) for value in task_ids]
    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate WebArena task IDs")
    candidates = hard_ids if subset == "hard" else set(range(812))
    if requested and not set(requested) <= candidates:
        raise ValueError("Requested tasks are not members of the selected official subset")
    selected = set(requested) if requested else candidates
    rows = []
    for original in official:
        if original["task_id"] not in selected:
            continue
        payload = {"source_task_id": original["task_id"], "task_revision": original["revision"],
                   "intent": original["intent"], "sites": original["sites"], "start_urls": original["start_urls"],
                   "intent_template_id": original["intent_template_id"], "source_task_sha256": hash_json(original),
                   "dataset_revision": REVISION, "dataset_sha256": DATASET_SHA256,
                   "hard": original["task_id"] in hard_ids}
        rows.append(Task.model_validate({"benchmark": "webarena_verified", "revision": REVISION,
            "instance_id": f"wav-{original['task_id']:03d}", "partition": partition, "instance": payload}).model_dump())
    tasks = {hash_json({key: task[key] for key in ("benchmark", "revision", "instance_id")}): task for task in rows}
    selection_hash = hash_json({"tasks": tasks, "subset": subset})
    snapshot_id = identifier(snapshot_id or "webarena-verified-" + selection_hash[:20])
    private = root / "state/webarena-evaluators" / REVISION / "tasks.json"
    profile = browser_profile()
    profile_path = root / "runtime/agent-profiles" / ("minisweagent-webarena-" + hash_json(profile)[:20]) / "manifest.json"
    target = root / "catalog/snapshots" / snapshot_id / "manifest.json"
    with file_lock(root / "state/webarena-import.lock"):
        private.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        private.parent.chmod(0o700)
        if private.exists() and sha256(private) != DATASET_SHA256:
            raise ValueError("Private Verified evaluator dataset changed")
        if not private.exists():
            # Preserve pinned bytes, rather than reserializing the evaluator definitions.
            raw_bytes = dataset.read_bytes()
            if hashlib.sha256(raw_bytes).hexdigest() != DATASET_SHA256:
                raise ValueError("Pinned WebArena dataset changed during import")
            partial = private.with_name(".tasks.json.partial")
            partial.write_bytes(raw_bytes)
            partial.chmod(0o600)
            partial.replace(private)
        private.chmod(0o600)
        _immutable_json(profile_path, profile)
        manifest = {"schema_version": 1, "benchmark": "webarena_verified", "revision": REVISION,
                    "dataset_sha256": DATASET_SHA256, "subset": subset, "tasks": tasks,
                    "profile_ref": profile_path.relative_to(root).as_posix(), "profile_sha256": sha256(profile_path),
                    "evaluator": {"path": private.relative_to(root).as_posix(), "sha256": DATASET_SHA256,
                                  "revision": REVISION, "task_count": 812}}
        if target.exists() and read_json(target) != manifest:
            raise FileExistsError("Snapshot ID already binds different WebArena content")
        Catalog(root).register_tasks(rows)
        _immutable_json(target, manifest)
    return {"snapshot_ref": target.relative_to(root).as_posix(), "snapshot_sha256": sha256(target),
            "profile_ref": profile_path.relative_to(root).as_posix(), "task_count": len(tasks),
            "subset": subset, "revision": REVISION, "evaluator": manifest["evaluator"]}

"""Online hidden-state capture control and asynchronous export primitives.

The controller is deliberately independent from vLLM's scheduler.  A managed
EasySteer/vLLM integration can call ``record`` after its native capture hook,
while tests and external adapters can use the same lifecycle contract.
"""

from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from core.io import hash_json, now, read_json, sha256, write_json
from core.locking import file_lock

THINK_OPEN_TOKEN_ID = 248068
THINK_NEWLINE_TOKEN_ID = 198


def accept_reasoning_event(state, position: int, anchor: str, stats=None) -> bool:
    """Apply request-scoped phase filtering before a tensor enters export.

    ``state`` is intentionally a tiny mutable mapping so the same rule can be
    used by the vLLM plugin and by CPU/recovery tests without importing vLLM.
    """
    stats = stats if stats is not None else {}
    prompt_length = state.get("prompt_length")
    if prompt_length is not None and position < prompt_length:
        stats["prompt"] = stats.get("prompt", 0) + 1
        return False
    if position <= state.get("last_position", -1):
        stats["duplicate"] = stats.get("duplicate", 0) + 1
        return False
    if state.get("close_seen", False):
        stats["after_close"] = stats.get("after_close", 0) + 1
        return False
    state["last_position"] = position
    if anchor == "think_close":
        state["close_seen"] = True
    return True


class AnchorError(ValueError):
    """Prompt/anchor evidence is not sufficient for an online capture."""


def validate_anchor(prompt_token_ids, anchor: str, *, token_ids: Mapping[str, list[int]] | None = None):
    """Validate an anchor at the exact position used by Qwen generation.

    ``think_open`` is the prompt boundary ``<think>\\n`` and therefore uses
    position -2 for the opening token. ``think_close`` is only valid when the
    close marker is an input token of a later forward; a response ending right
    after the marker has no close-token hidden state to export.
    """
    ids = list(prompt_token_ids or [])
    if anchor == "think_open":
        if len(ids) < 2 or ids[-2:] != [THINK_OPEN_TOKEN_ID, THINK_NEWLINE_TOKEN_ID]:
            raise AnchorError("think_open_unavailable: prompt must end with <think>\\n")
        return {"position": len(ids) - 2, "token_id": THINK_OPEN_TOKEN_ID, "available": True}
    if anchor == "reasoning_newline":
        return {"position": None, "token_id": None, "available": True}
    if anchor != "think_close":
        raise AnchorError(f"unknown capture anchor: {anchor}")
    close = list((token_ids or {}).get("think_close") or (token_ids or {}).get("close_token_ids") or [])
    if not close:
        raise AnchorError("think_close_unavailable: close token IDs were not configured")
    if ids[-len(close) :] != close:
        raise AnchorError("no_followup_forward: </think> was not an input suffix")
    return {"position": len(ids) - len(close), "token_id": close[0], "available": True}


@dataclass(frozen=True)
class CaptureRequest:
    session_id: str
    task_id: str
    run_id: str
    rollout_id: str
    step_id: int
    anchors: tuple[str, ...]
    trajectory_sha256: str | None = None
    capture_target_position: int | None = None
    capture_cache_boundary: int | None = None
    newline_token_ids: tuple[int, ...] = ()
    close_token_ids: tuple[int, ...] = ()
    tokenizer_fingerprint: str | None = None
    newline_policy: str = "all"
    capture_select: dict[str, Any] | None = None
    capture_select_sha256: str | None = None
    message_index: int | None = None
    prompt_token_ids: tuple[int, ...] = ()
    request_sha256: str | None = None
    query_id: str | None = None
    api_attempt_id: str | None = None
    decision_id: str | None = None
    branch_id: int | None = None
    branch_count: int | None = None
    parent_prefix_sha256: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]):
        required = ["session_id", "task_id", "run_id", "rollout_id", "step_id", "anchors"]
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError("capture metadata missing: " + ", ".join(missing))
        anchors = tuple(value["anchors"])
        if not anchors or any(x not in {"think_open", "reasoning_newline", "think_close"} for x in anchors):
            raise ValueError("capture metadata contains an invalid anchor")
        if type(value["step_id"]) is not int or value["step_id"] < 0:
            raise ValueError("step_id must be a nonnegative integer")
        newline_policy = value.get("newline_policy", "all")
        if newline_policy not in {"all", "double_only"}:
            raise ValueError("invalid newline policy")
        selector = value.get("capture_select")
        selector_hash = value.get("capture_select_sha256")
        if selector is not None and selector_hash is not None and hash_json(selector) != selector_hash:
            raise ValueError("capture selector checksum mismatch")
        decision = value.get("decision_id")
        if decision is not None:
            from core.io import identifier

            identifier(decision)
            if (type(value.get("branch_id")) is not int or type(value.get("branch_count")) is not int
                    or not 0 <= value["branch_id"] < value["branch_count"] <= 16):
                raise ValueError("Invalid capture branch identity")
        return cls(
            session_id=str(value["session_id"]),
            task_id=str(value["task_id"]),
            run_id=str(value["run_id"]),
            rollout_id=str(value["rollout_id"]),
            step_id=value["step_id"],
            anchors=anchors,
            trajectory_sha256=value.get("trajectory_sha256"),
            capture_target_position=value.get("capture_target_position"),
            capture_cache_boundary=value.get("capture_cache_boundary"),
            newline_token_ids=tuple(value.get("newline_token_ids") or ()),
            close_token_ids=tuple(value.get("close_token_ids") or ()),
            tokenizer_fingerprint=value.get("tokenizer_fingerprint"),
            newline_policy=newline_policy,
            capture_select=selector,
            capture_select_sha256=selector_hash,
            message_index=value.get("message_index"),
            prompt_token_ids=tuple(value.get("prompt_token_ids") or ()),
            request_sha256=value.get("request_sha256"),
            query_id=value.get("query_id"),
            api_attempt_id=value.get("api_attempt_id"),
            decision_id=decision,
            branch_id=value.get("branch_id"),
            branch_count=value.get("branch_count"),
            parent_prefix_sha256=value.get("parent_prefix_sha256"),
        )

    def to_dict(self):
        value = {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "rollout_id": self.rollout_id,
            "step_id": self.step_id,
            "anchors": list(self.anchors),
            "trajectory_sha256": self.trajectory_sha256,
            "capture_target_position": self.capture_target_position,
            "capture_cache_boundary": self.capture_cache_boundary,
            "newline_token_ids": list(self.newline_token_ids),
            "close_token_ids": list(self.close_token_ids),
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "newline_policy": self.newline_policy,
            "capture_select": self.capture_select,
            "capture_select_sha256": self.capture_select_sha256,
            "message_index": self.message_index,
            "prompt_token_ids": list(self.prompt_token_ids),
            "request_sha256": self.request_sha256,
            "query_id": self.query_id,
            "api_attempt_id": self.api_attempt_id,
        }
        if self.decision_id is not None:
            value.update(decision_id=self.decision_id, branch_id=self.branch_id,
                         branch_count=self.branch_count, parent_prefix_sha256=self.parent_prefix_sha256)
        return value


@dataclass
class _ExportItem:
    sample_id: str
    tensor: dict[str, np.ndarray]
    metadata: dict[str, Any]
    reservation: str | None = None


@dataclass
class _SpillItem:
    """Durable tensor handoff created when the in-memory queue is full."""

    sample_id: str
    tensor_path: Path
    metadata_path: Path
    session_id: str
    reservation: str | None = None


class AsyncExportQueue:
    """Bounded GPU-to-host export queue with atomic NPZ/sidecar writes.

    The capture hook only enqueues CPU staging arrays. Compression and disk I/O
    happen in a background thread. A small journal makes queued items visible
    during shutdown/recovery without blocking the model forward.
    """

    def __init__(self, root: str | Path, maxsize: int = 256, spool=None):
        self.spool = spool
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.staging = self.root / "staging"
        self.staging.mkdir(exist_ok=True)
        self.spill = self.root / "spill"
        self.spill.mkdir(exist_ok=True)
        self.journal = self.root / "queue.jsonl"
        self.queue: queue.Queue[_ExportItem | None] = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._pending_condition = threading.Condition(self._lock)
        self._pending_sessions = {}
        self._closed = False
        self._failed: list[dict[str, Any]] = []
        self._written = 0
        self._worker = threading.Thread(target=self._run, name="dycon-capture-export", daemon=True)
        self._spill_queue: queue.Queue[_SpillItem | None] = queue.Queue()
        self._spill_worker = threading.Thread(target=self._run_spill, name="dycon-capture-spill", daemon=True)
        self._worker.start()
        self._spill_worker.start()
        self._recover_spills()

    def _recover_spills(self):
        """Queue committed spill records after a controller restart."""
        for metadata_path in sorted(self.spill.glob("*.json")):
            try:
                metadata = read_json(metadata_path)
                tensor_path = self.root / metadata["tensor"]
                if not tensor_path.is_file():
                    continue
                item = _SpillItem(
                    metadata["sample_id"], tensor_path, metadata_path,
                    metadata["session_id"], metadata.get("reservation"),
                )
                with self._pending_condition:
                    self._pending_sessions[item.session_id] = self._pending_sessions.get(item.session_id, 0) + 1
                self._spill_queue.put(item)
            except (OSError, KeyError, TypeError, ValueError) as exc:
                with self._lock:
                    self._failed.append({"sample_id": metadata_path.stem, "error": repr(exc)})

    def submit(self, item: _ExportItem, *, block: bool = False) -> bool:
        with self._lock:
            if self._closed:
                raise RuntimeError("capture export queue is closed")
        if self.spool:
            item.reservation = "queue:" + item.sample_id
            self.spool.budget.acquire(item.reservation, sum(v.nbytes for v in item.tensor.values()) + 4096)
            with self._lock:
                size = self.journal.stat().st_size if self.journal.exists() else 0
                self.spool.budget.acquire(self.spool.key("queue.jsonl"), size + 256)
        sid = item.metadata["session_id"]
        with self._pending_condition:
            self._pending_sessions[sid] = self._pending_sessions.get(sid, 0) + 1
        try:
            self.queue.put(item, block=block, timeout=1 if block else 0)
        except queue.Full:
            # Transient writer pressure must not silently discard a hidden
            # event.  Persist it on local NVMe and consume it asynchronously.
            try:
                spill = self._write_spill(item)
                if item.reservation and self.spool:
                    self.spool.budget.release(item.reservation)
                self._spill_queue.put(spill)
                return True
            except Exception as exc:
                if item.reservation and self.spool:
                    self.spool.budget.release(item.reservation)
                with self._pending_condition:
                    self._pending_sessions[sid] -= 1
                    self._pending_condition.notify_all()
                with self._lock:
                    self._failed.append({"sample_id": item.sample_id, "session_id": sid, "error": repr(exc)})
                return False
        with self.journal.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"sample_id": item.sample_id, "status": "queued", "at": now()}) + "\n")
        return True

    def _write_spill(self, item: _ExportItem) -> _SpillItem:
        if self.spool is None:
            raise RuntimeError("capture queue full and no local spool is configured")
        tensor_relative = f"spill/{item.sample_id}.npz"
        metadata_relative = f"spill/{item.sample_id}.json"
        tensor_path = self.root / tensor_relative
        metadata_path = self.root / metadata_relative
        temp = tensor_path.with_suffix(".part")
        reservation = self.spool.key(tensor_relative)
        estimate = sum(value.nbytes + 4096 for value in item.tensor.values()) + 8192
        self.spool.budget.acquire(reservation, estimate)
        try:
            with temp.open("wb") as stream:
                np.savez_compressed(stream, **item.tensor)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, tensor_path)
            value = {
                "schema_version": 1,
                "sample_id": item.sample_id,
                "session_id": item.metadata["session_id"],
                "tensor": tensor_relative,
                "metadata": item.metadata,
                "tensor_sha256": sha256(tensor_path),
                "reservation": reservation,
                "status": "spilled",
            }
            self.spool.write_json(metadata_relative, value, immutable=True)
            with self.journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"sample_id": item.sample_id, "status": "spilled", "at": now()}) + "\n")
            return _SpillItem(item.sample_id, tensor_path, metadata_path, item.metadata["session_id"], reservation)
        except Exception:
            temp.unlink(missing_ok=True)
            tensor_path.unlink(missing_ok=True)
            self.spool.budget.release(reservation)
            raise

    def _run(self):
        while True:
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                return
            try:
                self._persist(item)
                with self._lock:
                    self._written += 1
            except Exception as exc:  # record and continue is the default policy
                with self._lock:
                    self._failed.append(
                        {
                            "sample_id": item.sample_id,
                            "session_id": item.metadata["session_id"],
                            "error": repr(exc),
                        }
                    )
            finally:
                if item.reservation and self.spool:
                    self.spool.budget.release(item.reservation)
                with self._pending_condition:
                    self._pending_sessions[item.metadata["session_id"]] -= 1
                    self._pending_condition.notify_all()
                self.queue.task_done()

    def _run_spill(self):
        while True:
            item = self._spill_queue.get()
            if item is None:
                self._spill_queue.task_done()
                return
            try:
                metadata = read_json(item.metadata_path)
                with np.load(item.tensor_path, allow_pickle=False) as archive:
                    tensors = {name: np.asarray(archive[name]) for name in archive.files}
                if sha256(item.tensor_path) != metadata["tensor_sha256"]:
                    raise ValueError("spill tensor checksum mismatch")
                self._persist(_ExportItem(item.sample_id, tensors, metadata["metadata"]))
                item.metadata_path.unlink(missing_ok=True)
                item.tensor_path.unlink(missing_ok=True)
                if item.reservation and self.spool:
                    self.spool.budget.release(item.reservation)
                with self._lock:
                    self._written += 1
            except Exception as exc:
                with self._lock:
                    self._failed.append({"sample_id": item.sample_id, "session_id": item.session_id, "error": repr(exc)})
            finally:
                with self._pending_condition:
                    self._pending_sessions[item.session_id] = max(0, self._pending_sessions.get(item.session_id, 1) - 1)
                    self._pending_condition.notify_all()
                self._spill_queue.task_done()

    def drain_session(self, session_id, timeout):
        with self._pending_condition:
            if not self._pending_condition.wait_for(
                lambda: self._pending_sessions.get(session_id, 0) == 0, timeout
            ):
                raise TimeoutError("Session capture writer did not drain")

    def _persist(self, item: _ExportItem):
        if self.spool:
            relative = f"raw/{item.sample_id}.npz"
            target = self.root / relative
            target.parent.mkdir(exist_ok=True)
            self.spool.budget.acquire(
                self.spool.key(relative), sum(v.nbytes + 4096 for v in item.tensor.values()) + 4096
            )
            self.spool.write_json(
                f"index/{item.sample_id}.json",
                {**item.metadata, "tensor": relative, "artifact_state": "queued", "sha256": None},
            )
            temp = target.with_suffix(".part")
            try:
                with temp.open("wb") as stream:
                    np.savez_compressed(stream, **item.tensor)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp, target)
                self.spool.budget.acquire(self.spool.key(relative), target.stat().st_size)
                meta = {
                    **item.metadata,
                    "tensor": relative,
                    "sha256": sha256(target),
                    "status": "persisted",
                    "artifact_state": "local_persisted",
                }
                self.spool.write_json(f"index/{item.sample_id}.json", meta)
                self.spool.add(meta)
            finally:
                temp.unlink(missing_ok=True)
            return
        # ``raw`` is the immutable recovery boundary.  Keep a hard-link under
        # the historical ``tensors`` name when possible so existing readers do
        # not need a migration and the tensor bytes are still stored once.
        target = self.root / "raw" / f"{item.sample_id}.npz"
        target.parent.mkdir(exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{item.sample_id}.", suffix=".npz", dir=target.parent)
        os.close(fd)
        temp = Path(temp_name)
        try:
            np.savez_compressed(temp, **item.tensor)
            os.replace(temp, target)
            meta = dict(item.metadata)
            meta.update(
                {
                    "sample_id": item.sample_id,
                    "tensor": str(target.relative_to(self.root)),
                    "sha256": sha256(target),
                    "status": "persisted",
                }
            )
            write_json(self.root / "raw" / f"{item.sample_id}.json", meta)
            write_json(self.root / "index" / f"{item.sample_id}.json", meta)
            legacy = self.root / "tensors" / f"{item.sample_id}.npz"
            legacy.parent.mkdir(exist_ok=True)
            try:
                legacy.unlink(missing_ok=True)
                os.link(target, legacy)
            except OSError:
                # Filesystems mounted without hard-link support still get a
                # readable compatibility copy; the canonical path is raw/.
                np.savez_compressed(legacy, **item.tensor)
            with self.journal.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"sample_id": item.sample_id, "status": "persisted", "at": now()}) + "\n")
        finally:
            temp.unlink(missing_ok=True)

    def status(self):
        with self._lock:
            return {
                "queued": self.queue.qsize(),
                "persisted": self._written,
                "failed": list(self._failed),
                "spill_pending": self._spill_queue.qsize(),
                "spill_files": len(list(self.spill.glob("*.json"))),
                "closed": self._closed,
            }

    def close(self, *, drain: bool = True, timeout: float = 30):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if drain:
            with self.queue.all_tasks_done:
                if not self.queue.all_tasks_done.wait_for(lambda: self.queue.unfinished_tasks == 0, timeout):
                    raise TimeoutError("Local capture writer drain timed out")
            with self._pending_condition:
                if not self._pending_condition.wait_for(lambda: not any(self._pending_sessions.values()), timeout):
                    raise TimeoutError("Spill capture writer drain timed out")
        self.queue.put(None)
        self._worker.join(timeout=timeout)
        self._spill_queue.put(None)
        self._spill_worker.join(timeout=timeout)


@dataclass
class _Session:
    session_id: str
    options: dict[str, Any]
    default_request: CaptureRequest | None = None
    created_at: str = field(default_factory=now)
    status: str = "active"
    requests: dict[str, CaptureRequest] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)
    sample_ids: dict[str, dict] = field(default_factory=dict)
    finalization: dict[str, Any] | None = None
    metrics: dict[str, int] = field(
        default_factory=lambda: {
            "graph_replays": 0,
            "eager_capture_forwards": 0,
            "fallbacks": 0,
            "cache_hit_tokens": 0,
        }
    )


class OnlineCaptureController:
    """Session and request binding facade used by the managed server."""

    def __init__(self, project_root: str | Path, config: Mapping[str, Any]):
        self.root = Path(project_root)
        self.project_root = self.root.resolve()
        self.config = dict(config)
        export = dict(config.get("export", {}))
        self.export_enabled = export.get("enabled", True)
        self.spool = None
        if self.export_enabled and "spool_root" in export:
            from runtime.serving.spool import CaptureSpool

            self.spool = CaptureSpool(
                self.root,
                export,
                config.get("run_id", "standalone"),
                config["nas_root"],
                config.get("reserve_bytes", 0),
            )
        self.export_root = self.spool.root if self.spool else self.root / export.get("root", "hidden/online")
        if not self.export_enabled:
            self.export_root = self.root / "runs" / config.get("run_id", "standalone") / "capture-diagnostics"
        self.queue = AsyncExportQueue(self.export_root, int(export.get("queue_size", 256)), self.spool)
        self.sessions: dict[str, _Session] = {}
        self.request_outputs = {}
        self.request_bindings: dict[str, tuple[str, CaptureRequest]] = {}
        self._lock = threading.RLock()
        self._output_condition = threading.Condition(self._lock)
        self._manifest_lock = threading.Lock()
        self.parallel_decisions = {}

    def _write(self, relative, value, **kwargs):
        if self.spool:
            return self.spool.write_json(relative, value, **kwargs)
        path = self.export_root / relative
        if kwargs.get("immutable") and path.exists():
            if read_json(path) != value:
                raise ValueError("Immutable capture record changed")
            return path
        write_json(path, value)
        return path

    @staticmethod
    def _request_summary(request):
        return {
            k: v
            for k, v in request.to_dict().items()
            if k not in {"prompt_token_ids", "newline_token_ids", "close_token_ids", "capture_select"}
        }

    def failure(self, session_id, reason):
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.metrics["fallbacks"] += 1
                session.options.setdefault("failures", []).append(str(reason))
                self._write(f"sessions/{session_id}/failures.json", session.options["failures"])
        if self.config.get("failure_policy") == "block_rollout":
            raise RuntimeError("CaptureError: " + str(reason))
        return {"recorded": True}

    def _record_manifests(self, session_id, metadata, *, session_status="active"):
        if not self.export_enabled:
            return
        # Per-token sidecars are authoritative. Keep the two global catalogs
        # bounded by sessions, rather than rewriting every token of every run.
        session = self.sessions[session_id]
        if session_status == "active" and session.samples:
            return
        directory = self.export_root / "sessions" / session_id
        entry = {
            "session_id": session_id,
            "status": session_status,
            "backend": "easysteer",
            "task_id": metadata["task_id"],
            "run_id": metadata["run_id"],
            "rollout_id": metadata["rollout_id"],
            "anchors": list(session.default_request.anchors),
            "layout": "token_ledger_tar_v2" if self.spool else "per_token_sidecars_v1",
            "export_root": str(self.spool.nas if self.spool else self.export_root),
            "session_path": str(directory / "session.json"),
            "index_root": str((self.spool.nas if self.spool else self.export_root) / "index"),
            "index_filter": {"session_id": session_id},
            "sample_count": len(session.samples) if session_status != "active" else None,
            "receipt_path": str(directory / "receipt.json"),
            "finalization_path": str(directory / "finalization.json"),
        }
        if session.finalization:
            entry.update(session.finalization)
        if self.spool:
            for key in ("session_path", "receipt_path", "finalization_path"):
                entry[key] = str(self.spool.nas / Path(entry[key]).relative_to(self.export_root))
        entry["published"] = session_status == "complete"
        diagnostic = session.default_request.task_id in {"startup", "startup-mixed", "engine-warmup"}
        entry.update(
            dataset_role="diagnostic" if diagnostic else "rollout",
            eligible_for_training=session_status == "complete" and not diagnostic,
        )
        with self._manifest_lock, file_lock(self.root / "state/product-manifests.lock"):
            for owner in ("enrichment", "hidden"):
                path = self.root / owner / "manifest.json"
                manifest = read_json(path) if path.exists() else {"schema_version": 1}
                manifest.setdefault("online_captures", {})[session_id] = entry
                write_json(path, manifest)

    def start(self, metadata: Mapping[str, Any], **options):
        request = CaptureRequest.from_mapping(metadata)
        with self._lock:
            if request.session_id in self.sessions:
                raise ValueError("capture session already exists")
            merged = {**self.config, **options}
            if str(merged.get("graph_mode", "FULL")).upper() != "FULL":
                raise ValueError("online capture requires graph_mode=FULL")
            self.sessions[request.session_id] = _Session(request.session_id, merged, request)
            session_dir = self.export_root / "sessions" / request.session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            self._write(
                f"sessions/{request.session_id}/session.json",
                {
                    "session_id": request.session_id,
                    "metadata": self._request_summary(request),
                    "options": {
                        "graph_mode": merged.get("graph_mode", "FULL"),
                        "prefix_cache_policy": merged.get("prefix_cache_policy"),
                    },
                    "capture_config_sha256": hash_json(self.config),
                    "started_at": self.sessions[request.session_id].created_at,
                    "export_root": str(self.export_root),
                },
            )
            return self.status(request.session_id)

    def bind(self, request_id: str, metadata: Mapping[str, Any]):
        request = CaptureRequest.from_mapping(metadata)
        with self._lock:
            session = self.sessions.get(request.session_id)
            if session is None or session.status != "active":
                raise KeyError("capture session is not active")
            if request_id in self.request_bindings and self.request_bindings[request_id][1] != request:
                raise ValueError("Native request ID has conflicting capture bindings")
            session.requests[request_id] = request
            self.request_bindings[request_id] = (request.session_id, request)
        return {"request_id": request_id, **request.to_dict()}

    def record(
        self,
        request_id: str,
        tensors: Mapping[str, Any],
        *,
        prompt_token_ids=None,
        anchor="think_open",
        token_ids=None,
        position=None,
        token_id=None,
        token_text=None,
        phase=None,
        absolute_position=None,
        generation_index=None,
        message_index=None,
        token_ref=None,
        graph_replays=0,
        eager_capture_forwards=0,
        fallback_reason=None,
        cache_hit_tokens=0,
        capture_stage="post_steering",
        graph_execution=None,
        batch_kind=None,
    ):
        with self._lock:
            binding = self.request_bindings.get(request_id)
            if binding is None:
                raise KeyError("request has no capture metadata binding")
            session_id, request = binding
            session = self.sessions[session_id]
            try:
                if prompt_token_ids is not None and anchor == "think_open":
                    evidence = validate_anchor(prompt_token_ids, anchor, token_ids=token_ids)
                    if position is None:
                        position, token_id = evidence["position"], evidence["token_id"]
                if anchor == "think_close" and prompt_token_ids is not None:
                    evidence = validate_anchor(prompt_token_ids, anchor, token_ids=token_ids)
                    if position is None:
                        position, token_id = evidence["position"], evidence["token_id"]
            except AnchorError as error:
                if self.config.get("failure_policy", "record_and_continue") == "record_and_continue":
                    failure = {
                        **request.to_dict(),
                        "request_id": request_id,
                        "sample_id": hash_json(
                            {"session": session_id, "request": request_id, "anchor": anchor, "failed": True}
                        )[:32],
                        "anchor": anchor,
                        "status": "failed",
                        "failure_reason": str(error),
                        "captured_at": now(),
                    }
                    session.samples.append(failure)
                    session.metrics["fallbacks"] += 1
                    self._record_manifests(session_id, failure)
                    return failure
                raise
            if anchor not in request.anchors:
                raise AnchorError(f"anchor {anchor} was not requested")
            if anchor == "reasoning_newline" and token_ids and token_id is not None:
                allowed = set(token_ids.get("reasoning_newline") or token_ids.get("newline_token_ids") or [])
                if allowed and token_id not in allowed:
                    raise AnchorError(
                        "reasoning_newline_unavailable: token is outside the tokenizer inventory"
                    )
            arrays = {}
            for name, value in tensors.items():
                if hasattr(value, "detach"):
                    value = value.detach().to(device="cpu", copy=True).numpy()
                arrays[str(name)] = np.asarray(value)
            sample_id = hash_json(
                {"session": session_id, "request": request_id, "anchor": anchor, "position": position}
            )[:32]
            metadata = {
                **request.to_dict(),
                "request_id": request_id,
                "sample_id": sample_id,
                "anchor": anchor,
                "anchor_position": position,
                "anchor_token_id": token_id,
                "token_id": token_id,
                "token_text": token_text,
                "phase": phase,
                "absolute_position": absolute_position if absolute_position is not None else position,
                "generation_index": generation_index,
                "message_index": message_index,
                "token_ref": token_ref,
                "capture_stage": capture_stage,
                "graph_execution": "full" if graph_replays else "eager",
                "cache_hit_tokens": cache_hit_tokens,
                "status": "captured",
                "captured_at": now(),
            }
            if graph_execution is not None:
                if graph_execution not in {"full", "eager", "piecewise"}:
                    raise ValueError("Unknown per-event graph execution")
                metadata["graph_execution"] = graph_execution
            metadata["batch_kind"] = batch_kind
            metadata.update(
                model_fingerprint=self.config.get("model_fingerprint"),
                serving_dtype=self.config.get("serving_dtype"),
                storage_dtypes={name: str(value.dtype) for name, value in arrays.items()},
                storage_shapes={name: list(value.shape) for name, value in arrays.items()},
            )
            import hashlib

            metadata["tensor_content_sha256"] = hash_json(
                {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in arrays.items()}
            )
            previous = session.sample_ids.get(sample_id)
            if previous is not None:
                if previous.get("tensor_content_sha256") != metadata["tensor_content_sha256"]:
                    raise ValueError("Conflicting duplicate capture point")
                return previous
            if self.spool:
                for key in ("prompt_token_ids", "capture_select", "newline_token_ids", "close_token_ids"):
                    metadata.pop(key, None)
                metadata["request_key"] = hash_json(request_id)
            if not self.export_enabled:
                return {"sample_id": sample_id, "status": "not_exported"}
            if not all(np.isfinite(v).all() for v in arrays.values()):
                return self.failure(session_id, "Nonfinite capture tensors")
            accepted = self.queue.submit(_ExportItem(sample_id, arrays, metadata), block=False)
            metadata["status"] = "queued" if accepted else "failed"
            metadata["failure_reason"] = None if accepted else "export_backpressure"
            if not accepted:
                self.failure(session_id, "export_backpressure")
            self._record_manifests(session_id, metadata)
            session.samples.append(metadata)
            session.sample_ids[sample_id] = metadata
            session.metrics["graph_replays"] += int(graph_replays)
            session.metrics["eager_capture_forwards"] += int(eager_capture_forwards)
            session.metrics["cache_hit_tokens"] += int(cache_hit_tokens)
            if fallback_reason:
                session.metrics["fallbacks"] += 1
                metadata["fallback_reason"] = fallback_reason
            return metadata

    def record_file(self, request_id: str, path: str | Path, **metadata):
        """Accept a shared-filesystem staging archive from a vLLM worker.

        Tensor bytes never travel through the HTTP control plane.  The worker
        writes one atomic NPZ in ``staging`` and sends only its relative path;
        this process validates that the path is inside the configured export
        root before handing arrays to the same asynchronous queue used by the
        in-process adapter.
        """
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.export_root / "staging" / candidate
        candidate = candidate.resolve()
        staging = (self.export_root / "staging").resolve()
        if not candidate.is_relative_to(staging):
            raise ValueError("staged capture path escapes the capture staging directory")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        with self._lock:
            if request_id not in self.request_bindings:
                raise KeyError("request has no capture metadata binding")
            _, request = self.request_bindings[request_id]
            position = metadata.get("absolute_position", metadata.get("position"))
            prompt_length = len(request.prompt_token_ids)
            metadata.setdefault("message_index", request.message_index)
            if prompt_length and position is not None:
                metadata["phase"] = "prompt" if position < prompt_length else "generation_input"
                metadata["generation_index"] = position - prompt_length if position >= prompt_length else -1
            metadata["token_ref"] = [
                request.rollout_id,
                request.step_id,
                metadata.get("phase"),
                position,
                metadata.get("token_id"),
            ]
        try:
            with np.load(candidate, allow_pickle=False) as archive:
                tensors = {name: np.asarray(archive[name]) for name in archive.files}
            return self.record(request_id, tensors, **metadata)
        finally:
            candidate.unlink(missing_ok=True)

    def record_output(self, value):
        request_id = value["request_id"]
        with self._lock:
            if request_id not in self.request_bindings:
                # Some outputs expose the external ID; require a unique exact
                # prefix match to the native randomized ID, never batch order.
                matches = [rid for rid in self.request_bindings if rid.startswith(request_id + "-")]
                if len(matches) != 1:
                    raise KeyError("output has no unique native request binding")
                request_id = matches[0]
            session_id, request = self.request_bindings[request_id]
            record = {**request.to_dict(), **value, "request_id": request_id}
            record.pop("metadata", None)
            if request.prompt_token_ids and list(request.prompt_token_ids) != record["prompt_token_ids"]:
                raise ValueError("Native prompt differs from request binding")
            record.update(
                schema_version=2,
                request_key=hash_json(request_id),
                request_sha256=hash_json(record["prompt_token_ids"]),
                output_sha256=hash_json(record["output_token_ids"]),
            )
            relative = (
                f"requests/{record['request_key']}.json"
                if self.spool
                else f"sessions/{session_id}/requests/{hash_json(request_id)}.json"
            )
            stored = dict(record)
            if self.spool and (self.export_root / "inventory.json").exists():
                for key in ("newline_token_ids", "close_token_ids", "capture_select"):
                    stored.pop(key, None)
                stored.update(
                    inventory_ref="inventory.json",
                    inventory_sha256=hash_json(read_json(self.export_root / "inventory.json")),
                )
            path = self._write(relative, stored, immutable=True)
            record = {**record, "ledger_ref": relative, "ledger_sha256": sha256(path)}
            self.request_outputs[request_id] = record
            self._output_condition.notify_all()
            return {"recorded": True, "request_id": request_id}

    def wait_output(self, session_id, step_id, response_id=None, timeout=30):
        def candidates():
            return [
                row
                for row in self.request_outputs.values()
                if row["session_id"] == session_id
                and row["step_id"] == step_id
                and (
                    response_id is None
                    or row["request_id"] == response_id
                    or row["request_id"].startswith(response_id + "-")
                )
            ]

        with self._output_condition:
            if not self._output_condition.wait_for(lambda: bool(candidates()), timeout=timeout):
                raise TimeoutError("Native output receipt did not arrive")
            rows = candidates()
            if len(rows) != 1:
                raise ValueError("Ambiguous native output receipt")
            return dict(rows[0])

    def record_decision(self, decision, timeout=30):
        """Freeze branch outputs and selection before any tool is executed."""
        import copy

        sid, did = decision["session_id"], decision["decision_id"]
        from core.io import identifier

        identifier(did)
        candidates = decision["candidates"]
        if (len(candidates) != decision["branch_count"]
                or sorted(r["branch_id"] for r in candidates) != list(range(len(candidates)))
                or (decision["selected_branch_id"] is not None
                    and decision["selected_branch_id"] not in range(len(candidates)))):
            raise ValueError("Invalid decision candidates/winner")

        def matches(row):
            return [r for r in self.request_outputs.values()
                    if r["session_id"] == sid and r.get("decision_id") == did
                    and r.get("branch_id") == row["branch_id"]
                    and r.get("api_attempt_id") == row["api_attempt_id"]]

        with self._output_condition:
            self._output_condition.wait_for(
                lambda: all(matches(r) for r in candidates if r.get("response") is not None), timeout)
            value = copy.deepcopy(decision)
            native_prompt_hashes = set()
            for row in value["candidates"]:
                receipts = matches(row)
                if row.get("response") is not None and len(receipts) != 1:
                    self._write(f"sessions/{sid}/parallel/failed-{did}.json", value, immutable=True)
                    raise ValueError("Parallel response has no unique native output receipt")
                if receipts:
                    receipt = receipts[0]
                    native_prompt_hashes.add(receipt["request_sha256"])
                    if row.get("native_request_id") not in {None, receipt["request_id"]}:
                        raise ValueError("Parallel score belongs to another native request")
                    if row.get("close_sample_id") is not None:
                        position = row.get("absolute_position")
                        token = row.get("token_id")
                        ids = receipt["prompt_token_ids"] + receipt["output_token_ids"]
                        if (type(position) is not int or not len(receipt["prompt_token_ids"]) <= position < len(ids) - 1
                                or ids[position] != token or token not in receipt.get("close_token_ids", [248069])
                                or row["close_sample_id"] != hash_json({"session": sid,
                                    "request": receipt["request_id"], "anchor": "think_close", "position": position})[:32]):
                            raise ValueError("Parallel score has invalid close token evidence")
                    row.update(native_request_id=receipt["request_id"],
                               ledger_ref=receipt["ledger_ref"], ledger_sha256=receipt["ledger_sha256"],
                               output_sha256=receipt["output_sha256"],
                               num_cached_tokens=receipt.get("num_cached_tokens", 0),
                               prompt_tokens=len(receipt["prompt_token_ids"]),
                               completion_tokens=len(receipt["output_token_ids"]))
                    row["selected"] = row["branch_id"] == value["selected_branch_id"]
            if len(native_prompt_hashes) > 1:
                raise ValueError("Parallel candidates have different native token prompts")
            self._write(f"sessions/{sid}/parallel/decisions/{did}.json", value, immutable=True)
            # Full candidate text is durable on disk; keep only small identities
            # in memory until this session is finalized.
            self.parallel_decisions[did] = {
                **value,
                "candidates": [{k: v for k, v in row.items() if k != "response"}
                               for row in value["candidates"]],
            }
            return value

    def record_parallel_action(self, summary, phase, details=None):
        if phase not in {"started", "completed", "failed"}:
            raise ValueError("Invalid parallel action phase")
        sid, did = summary["session_id"], summary["decision_id"]
        from core.io import identifier

        identifier(did)
        if did not in self.parallel_decisions:
            raise ValueError("Action has no committed decision")
        value = {"session_id": sid, "decision_id": did,
                 "selected_branch_id": summary["selected_branch_id"], "phase": phase,
                 "recorded_at": now(), "details": details or {}}
        self._write(f"sessions/{sid}/parallel/actions/{did}-{phase}.json", value, immutable=True)

    def _finalize_branches(self, session_id, rollout_id, trajectory_sha256):
        session = self.sessions[session_id]
        with self._lock:
            decisions = [v for v in self.parallel_decisions.values() if v["session_id"] == session_id]
        if not decisions:
            previous = self.export_root / f"sessions/{session_id}/parallel/branch-coverage.json"
            if previous.exists():
                return read_json(previous)
            return None
        from enrichment.online_trace import reasoning_targets

        rows, missing = [], []
        for decision in decisions:
            for candidate in decision["candidates"]:
                rid = candidate.get("native_request_id")
                receipt = self.request_outputs.get(rid)
                events = [e for e in session.samples if e["request_id"] == rid and e["anchor"] == "think_close"]
                targets, available = reasoning_targets(receipt, self.config.get("open_token_ids", [248068]),
                                                       receipt.get("close_token_ids", [248069])) if receipt else (set(), [])
                close_positions = {p for p, a in targets if a == "think_close"}
                matched = []
                for event in events:
                    path = self.export_root / "index" / (event["sample_id"] + ".json")
                    if event["absolute_position"] in close_positions and path.exists():
                        stored = read_json(path)
                        if stored.get("status") != "failed" and stored.get("sha256"):
                            matched.append(stored)
                if not matched:
                    missing.append({"decision_id": decision["decision_id"], "branch_id": candidate["branch_id"],
                                    "reason": "missing_close_hidden", "availability": available})
                rows.append({"decision_id": decision["decision_id"], "branch_id": candidate["branch_id"],
                             "selected": candidate["branch_id"] == decision["selected_branch_id"],
                             "native_request_id": rid, "close_sample_ids": [e["sample_id"] for e in matched],
                             "close_events": matched,
                             "score": candidate.get("score"), "rollout_id": rollout_id,
                             "trajectory_sha256": trajectory_sha256,
                             "labels": {"remaining_steps": None, "remaining_reasoning_tokens": None}})
        base = f"sessions/{session_id}/parallel"
        self._write(base + "/branch-index.json", rows, immutable=True)
        coverage = {"expected": len(rows), "observed": len(rows) - len(missing), "missing": missing}
        self._write(base + "/branch-coverage.json", coverage, immutable=True)
        files = {str(p.relative_to(self.export_root)): sha256(p)
                 for p in (self.export_root / base).rglob("*.json") if p.name != "branch-receipt.json"}
        self._write(base + "/branch-receipt.json", {"schema_version": 1, "files": files,
                    "rollout_id": rollout_id, "trajectory_sha256": trajectory_sha256}, immutable=True)
        return coverage

    def finalize(self, session_id, *, rollout_id, trajectory_path, trajectory_sha256, model_fingerprint):
        path = Path(trajectory_path).resolve()
        if not path.is_relative_to(self.project_root):
            raise ValueError("Finalized trajectory is outside the project")
        if sha256(path) != trajectory_sha256:
            raise ValueError("Finalized trajectory checksum mismatch")
        source = read_json(path)
        from hidden.artifacts import read_request, response_records

        assistant = [row for _, row, _ in response_records(source)]
        native_ids = [
            (row.get("extra", {}).get("response") or {}).get("dycon_native_request_id") for row in assistant
        ]
        with self._lock:
            session = self.sessions[session_id]
            if self.config.get("model_fingerprint") not in {None, model_fingerprint}:
                raise ValueError("Finalized trajectory checkpoint mismatch")
            for row, native_id in zip(assistant, native_ids, strict=True):
                if native_id not in self.request_outputs:
                    raise ValueError("Finalized trajectory is missing exact native request evidence")
                response = read_request(self.export_root, row["extra"]["response"])
                receipt = self.request_outputs[native_id]
                if receipt["session_id"] != session_id:
                    raise ValueError("Receipt belongs to a different capture session")
                if any(response.get(key) != receipt[key] for key in ["prompt_token_ids", "output_token_ids"]):
                    raise ValueError("Trajectory tokens differ from native output receipt")
            value = {
                "rollout_id": rollout_id,
                "trajectory_path": str(path),
                "trajectory_sha256": trajectory_sha256,
                "model_fingerprint": model_fingerprint,
                "accepted_native_request_ids": native_ids,
                "finalized_at": now(),
            }
            action_step = 0
            response_map = {}
            for (index, message, _), native_id in zip(response_records(source), native_ids, strict=True):
                valid = message.get("role") == "assistant"
                response_map[native_id] = {
                    "message_index": index,
                    "agent_step_index": action_step,
                    "is_action_response": valid,
                }
                action_step += int(valid)
            value["responses"] = response_map
            session.finalization = value
            self._write(f"sessions/{session_id}/finalization.json", {"session_id": session_id, **value})
        self.queue.drain_session(session_id, self.config.get("finalize_seconds", 120))
        from enrichment.online_trace import reasoning_targets

        expected, observed = set(), set()
        for native_id in native_ids:
            receipt = self.request_outputs[native_id]
            targets, _ = reasoning_targets(
                receipt,
                self.config.get("open_token_ids", [248068]),
                list(receipt.get("close_token_ids") or [248069]),
            )
            expected.update((native_id, p, a) for p, a in targets)
        for sample in session.samples:
            if sample.get("status") != "failed":
                observed.add((sample["request_id"], sample["absolute_position"], sample["anchor"]))
        # Native selector also observes answer newlines; only reasoning targets
        # enter the training dataset. Missing selected reasoning rows fail coverage.
        coverage = {
            "expected": len(expected),
            "observed": len(expected & observed),
            "missing": sorted(expected - observed),
        }
        self._write(f"sessions/{session_id}/coverage.json", coverage)
        if self.spool and self.parallel_decisions:
            self.spool.seal(session_id)
        branch_coverage = self._finalize_branches(session_id, rollout_id, trajectory_sha256)
        complete = (
            not (branch_coverage and branch_coverage["missing"])
            and bool(assistant)
            and not session.options.get("failures")
            and not any(row.get("status") == "failed" for row in session.samples)
            and not coverage["missing"]
            and not any(f.get("session_id") == session_id for f in self.queue.status()["failed"])
        )
        if not self.export_enabled:
            session.status = "not_exported"
        elif self.spool:
            published = self.spool.publish_session(
                session_id, self.config.get("export", {}).get("upload_max_elapsed_seconds", 600)
            )
            session.status = (
                "complete" if complete and published else "pending_upload" if complete else "partial"
            )
        else:
            session.status = "complete" if complete else "partial"
        self._record_manifests(session_id, session.default_request.to_dict(), session_status=session.status)
        with self._lock:
            for did in [key for key, record in self.parallel_decisions.items()
                        if record["session_id"] == session_id]:
                del self.parallel_decisions[did]
        if self.spool and session.status == "complete":
            self.spool.release_session_metadata(session_id)
        return {**value, "capture_status": session.status}

    def prepare_trajectory(self, session_id, trajectory_path):
        """One bounded join before canonical import; includes FormatError responses."""
        from hidden.artifacts import response_records

        path = Path(trajectory_path).resolve()
        if not path.is_relative_to(self.project_root):
            raise ValueError("Trajectory escapes project")
        source = read_json(path)
        rows = list(response_records(source))

        def matches(message, response):
            metadata = (message.get("extra", {}).get("request") or {}).get("dycon_capture", {})
            return [
                r
                for r in self.request_outputs.values()
                if r["session_id"] == session_id
                and (not metadata.get("decision_id") or (
                    r.get("decision_id") == metadata["decision_id"]
                    and r.get("branch_id") == metadata["branch_id"]))
                and (
                    r.get("query_id") == metadata["query_id"]
                    if metadata.get("query_id")
                    else r["step_id"] == metadata.get("step_id")
                )
                and (
                    not metadata.get("api_attempt_id")
                    or r.get("api_attempt_id") == metadata["api_attempt_id"]
                )
                and (
                    (metadata.get("decision_id") and r.get("decision_id") == metadata["decision_id"])
                    or not response.get("id")
                    or r["request_id"] == response["id"]
                    or r["request_id"].startswith(response["id"] + "-")
                )
            ]

        with self._output_condition:
            self._output_condition.wait_for(
                lambda: all(matches(m, r) for _, m, r in rows), self.config.get("finalize_seconds", 120)
            )
            for index, message, response in rows:
                candidates = matches(message, response)
                if len(candidates) != 1:
                    self.failure(
                        session_id, f"response {index} has {len(candidates)} matching native receipts"
                    )
                    continue
                receipt = candidates[0]
                response.update(
                    dycon_native_request_id=receipt["request_id"],
                    ledger_ref=receipt["ledger_ref"],
                    ledger_sha256=receipt["ledger_sha256"],
                    ledger_root=str(self.spool.nas if self.spool else self.export_root),
                )
                response.pop("prompt_token_ids", None)
                response.pop("output_token_ids", None)
                for choice in response.get("choices", []):
                    choice.pop("token_ids", None)
                message["extra"].setdefault("request", {})["message_index"] = index
        write_json(path, source)
        return {
            "responses": len(rows),
            "capture_failures": self.sessions[session_id].options.get("failures", []),
        }

    def mark_behavior_failure(self, session_id, reason, *, expected=None, actual=None):
        with self._lock:
            session = self.sessions[session_id]
            session.status = "failed_behavior"
            session.metrics["fallbacks"] += 1
            entry = {
                "session_id": session_id,
                "status": session.status,
                "failure_reason": reason,
                "expected": expected,
                "actual": actual,
                "at": now(),
            }
            write_json(self.export_root / "sessions" / session_id / "behavior-failure.json", entry)
            return entry

    def stop(self, session_id: str):
        self.queue.drain_session(session_id, self.config.get("finalize_seconds", 120))
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            session.status = "stopped"
            for sample in session.samples:
                if sample.get("status") in {"queued", "captured"}:
                    sample["session_status"] = "stopped"
            if session.samples:
                self._record_manifests(session_id, session.samples[0], session_status="stopped")
            self._write(
                f"sessions/{session_id}/receipt.json",
                {
                    "session_id": session_id,
                    "status": session.status,
                    "metadata": self._request_summary(session.default_request),
                    "samples": len(session.samples),
                    "metrics": dict(session.metrics),
                    "export": self.queue.status(),
                    "files": {
                        "queue.jsonl": sha256(self.queue.journal) if self.queue.journal.exists() else None
                    },
                },
            )
            return self.status(session_id)

    def status(self, session_id: str | None = None):
        with self._lock:
            if session_id is None:
                return {
                    "sessions": [self._session_status(s) for s in self.sessions.values()],
                    "export": self.queue.status(),
                }
            session = self.sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            return self._session_status(session)

    def _session_status(self, session: _Session):
        return {
            "session_id": session.session_id,
            "status": session.status,
            "created_at": session.created_at,
            "requests": len(session.requests),
            "samples": list(session.samples),
            "metrics": dict(session.metrics),
            "export": self.queue.status(),
        }

    def results(self, session_id: str):
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            return {
                "session_id": session_id,
                "samples": list(session.samples),
                "status": session.status,
                "export": self.queue.status(),
            }

    def close(self):
        errors = []
        try:
            self.queue.close(drain=True, timeout=self.config.get("finalize_seconds", 120))
        except Exception as exc:
            errors.append(repr(exc))
        try:
            if self.spool and not self.spool.close(self.config["export"]["upload_max_elapsed_seconds"]):
                errors.append("NAS publication remains pending/recoverable")
        except Exception as exc:
            errors.append(repr(exc))
        if errors:
            write_json(
                self.root / "runs" / self.config.get("run_id", "standalone") / "capture-shutdown-errors.json",
                errors,
            )
            if self.config.get("failure_policy") == "block_rollout":
                from runtime.serving.errors import CaptureError

                raise CaptureError(str(errors))


class CaptureControlServer:
    """Small localhost control plane; it never carries model data tensors."""

    def __init__(self, controller: OnlineCaptureController, host="127.0.0.1", port=0):
        self.controller, self.host, self.port = controller, host, int(port)
        controller_ref = controller

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                return

            def _json(self, status, value):
                body = json.dumps(value, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read(self):
                n = int(self.headers.get("Content-Length", "0"))
                return json.loads(self.rfile.read(n) or b"{}")

            def do_GET(self):
                try:
                    if self.path == "/dycon/capture/status":
                        return self._json(200, controller_ref.status())
                    if self.path.startswith("/dycon/capture/session/") and self.path.endswith("/status"):
                        sid = self.path.split("/")[-2]
                        return self._json(200, controller_ref.status(sid))
                    if self.path.startswith("/dycon/capture/session/") and self.path.endswith("/results"):
                        sid = self.path.split("/")[-2]
                        return self._json(200, controller_ref.results(sid))
                    return self._json(404, {"error": "not_found"})
                except KeyError as exc:
                    return self._json(404, {"error": str(exc)})

            def do_POST(self):
                try:
                    value = self._read()
                    if self.path == "/dycon/capture/session/start":
                        return self._json(200, controller_ref.start(value.pop("metadata", value), **value))
                    if self.path == "/dycon/capture/request/output":
                        return self._json(200, controller_ref.record_output(value))
                    if self.path == "/dycon/capture/failure":
                        return self._json(200, controller_ref.failure(**value))
                    if self.path.startswith("/dycon/capture/session/") and self.path.endswith("/prepare"):
                        return self._json(
                            200, controller_ref.prepare_trajectory(self.path.split("/")[-2], **value)
                        )
                    if self.path == "/dycon/capture/request/wait-output":
                        return self._json(200, controller_ref.wait_output(**value))
                    if self.path.startswith("/dycon/capture/session/") and self.path.endswith("/finalize"):
                        return self._json(200, controller_ref.finalize(self.path.split("/")[-2], **value))
                    if self.path == "/dycon/capture/behavior-failure":
                        return self._json(
                            200,
                            controller_ref.mark_behavior_failure(
                                value["session_id"],
                                value["reason"],
                                expected=value.get("expected"),
                                actual=value.get("actual"),
                            ),
                        )
                    if self.path == "/dycon/capture/request/bind":
                        return self._json(200, controller_ref.bind(value["request_id"], value["metadata"]))
                    if self.path == "/dycon/capture/sample":
                        tensors = {k: np.asarray(v) for k, v in value.pop("tensors").items()}
                        return self._json(
                            200, controller_ref.record(value.pop("request_id"), tensors, **value)
                        )
                    if self.path == "/dycon/capture/sample-file":
                        return self._json(
                            200,
                            controller_ref.record_file(value.pop("request_id"), value.pop("path"), **value),
                        )
                    if self.path.startswith("/dycon/capture/session/") and self.path.endswith("/stop"):
                        return self._json(200, controller_ref.stop(self.path.split("/")[-2]))
                    return self._json(404, {"error": "not_found"})
                except (KeyError, ValueError, AnchorError) as exc:
                    return self._json(400, {"error": str(exc)})
                except Exception as exc:
                    return self._json(500, {"error": repr(exc)})

        self._server = ThreadingHTTPServer((host, self.port), Handler)
        self.port = self._server.server_address[1]
        self._thread: threading.Thread | None = None

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}"

    def start(self):
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="dycon-capture-control", daemon=True
        )
        self._thread.start()
        return self

    def close(self):
        self._server.shutdown()
        self._server.server_close()
        if self._thread:
            self._thread.join(timeout=5)
        self.controller.close()

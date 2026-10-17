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

THINK_OPEN_TOKEN_ID = 248068
THINK_NEWLINE_TOKEN_ID = 198


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
        )

    def to_dict(self):
        return {
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
        }


@dataclass
class _ExportItem:
    sample_id: str
    tensor: dict[str, np.ndarray]
    metadata: dict[str, Any]


class AsyncExportQueue:
    """Bounded GPU-to-host export queue with atomic NPZ/sidecar writes.

    The capture hook only enqueues CPU staging arrays. Compression and disk I/O
    happen in a background thread. A small journal makes queued items visible
    during shutdown/recovery without blocking the model forward.
    """

    def __init__(self, root: str | Path, maxsize: int = 256):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.staging = self.root / "staging"
        self.staging.mkdir(exist_ok=True)
        self.journal = self.root / "queue.jsonl"
        self.queue: queue.Queue[_ExportItem | None] = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._closed = False
        self._failed: list[dict[str, Any]] = []
        self._written = 0
        self._worker = threading.Thread(target=self._run, name="dycon-capture-export", daemon=True)
        self._worker.start()

    def submit(self, item: _ExportItem, *, block: bool = False) -> bool:
        with self._lock:
            if self._closed:
                raise RuntimeError("capture export queue is closed")
        try:
            self.queue.put(item, block=block, timeout=1 if block else 0)
        except queue.Full:
            return False
        with self.journal.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"sample_id": item.sample_id, "status": "queued", "at": now()}) + "\n")
        return True

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
                    self._failed.append({"sample_id": item.sample_id, "error": repr(exc)})
            finally:
                self.queue.task_done()

    def _persist(self, item: _ExportItem):
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
            meta.update({"sample_id": item.sample_id, "tensor": str(target.relative_to(self.root)), "sha256": sha256(target), "status": "persisted"})
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
            return {"queued": self.queue.qsize(), "persisted": self._written, "failed": list(self._failed), "closed": self._closed}

    def close(self, *, drain: bool = True, timeout: float = 30):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if drain:
            self.queue.join()
        self.queue.put(None)
        self._worker.join(timeout=timeout)


@dataclass
class _Session:
    session_id: str
    options: dict[str, Any]
    default_request: CaptureRequest | None = None
    created_at: str = field(default_factory=now)
    status: str = "active"
    requests: dict[str, CaptureRequest] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, int] = field(default_factory=lambda: {"graph_replays": 0, "eager_capture_forwards": 0, "fallbacks": 0, "cache_hit_tokens": 0})


class OnlineCaptureController:
    """Session and request binding facade used by the managed server."""

    def __init__(self, project_root: str | Path, config: Mapping[str, Any]):
        self.root = Path(project_root)
        self.config = dict(config)
        export = dict(config.get("export", {}))
        self.export_root = self.root / export.get("root", "hidden/online")
        self.queue = AsyncExportQueue(self.export_root, int(export.get("queue_size", 256)))
        self.sessions: dict[str, _Session] = {}
        self.request_outputs = {}
        self.request_bindings: dict[str, tuple[str, CaptureRequest]] = {}
        self._lock = threading.RLock()
        self._manifest_lock = threading.Lock()

    def _record_manifests(self, session_id, metadata, *, session_status="active"):
        entry = {
            "session_id": session_id,
            "task_id": metadata["task_id"],
            "run_id": metadata["run_id"],
            "rollout_id": metadata["rollout_id"],
            "step_id": metadata["step_id"],
            "anchor": metadata["anchor"],
            "anchor_position": metadata.get("anchor_position"),
            "anchor_token_id": metadata.get("anchor_token_id"),
            "capture_stage": metadata.get("capture_stage", "post_steering"),
            "graph_execution": metadata.get("graph_execution"),
            "cache_hit_tokens": metadata.get("cache_hit_tokens", 0),
            "capture_target_position": metadata.get("capture_target_position"),
            "capture_cache_boundary": metadata.get("capture_cache_boundary"),
            "status": metadata.get("status", "queued"),
            "tensor": metadata.get("tensor"),
            "token_ref": metadata.get("token_ref"),
            "token_id": metadata.get("token_id"),
            "token_text": metadata.get("token_text"),
            "phase": metadata.get("phase"),
            "absolute_position": metadata.get("absolute_position"),
            "generation_index": metadata.get("generation_index"),
            "message_index": metadata.get("message_index"),
            "trajectory_sha256": metadata.get("trajectory_sha256"),
            "tokenizer_fingerprint": metadata.get("tokenizer_fingerprint"),
            "capture_select_sha256": metadata.get("capture_select_sha256"),
        }
        with self._manifest_lock:
            for owner in ("enrichment", "hidden"):
                path = self.root / owner / "manifest.json"
                manifest = read_json(path) if path.exists() else {"schema_version": 1}
                captures = manifest.setdefault("online_captures", {})
                sample = captures.setdefault(session_id, {"samples": []})
                sample.update({"session_id": session_id, "status": session_status, "backend": "easysteer", "anchors": ["think_open", "reasoning_newline", "think_close"]})
                sample["samples"] = [x for x in sample.get("samples", []) if x.get("sample_id") != metadata["sample_id"]]
                sample["samples"].append(entry | {"sample_id": metadata["sample_id"]})
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
            write_json(
                session_dir / "session.json",
                {
                    "session_id": request.session_id,
                    "metadata": request.to_dict(),
                    "options": merged,
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
            session.requests[request_id] = request
            self.request_bindings[request_id] = (request.session_id, request)
        return {"request_id": request_id, **request.to_dict()}

    def record(self, request_id: str, tensors: Mapping[str, Any], *, prompt_token_ids=None, anchor="think_open", token_ids=None, position=None, token_id=None, token_text=None, phase=None, absolute_position=None, generation_index=None, message_index=None, token_ref=None, graph_replays=0, eager_capture_forwards=0, fallback_reason=None, cache_hit_tokens=0, capture_stage="post_steering"):
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
                        "sample_id": hash_json({"session": session_id, "request": request_id, "anchor": anchor, "failed": True})[:32],
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
                    raise AnchorError("reasoning_newline_unavailable: token is outside the tokenizer inventory")
            arrays = {}
            for name, value in tensors.items():
                if hasattr(value, "detach"):
                    value = value.detach().to(device="cpu", copy=True).numpy()
                arrays[str(name)] = np.asarray(value)
            sample_id = hash_json({"session": session_id, "request": request_id, "anchor": anchor, "position": position})[:32]
            metadata = {**request.to_dict(), "request_id": request_id, "sample_id": sample_id, "anchor": anchor, "anchor_position": position, "anchor_token_id": token_id, "token_id": token_id, "token_text": token_text, "phase": phase, "absolute_position": absolute_position if absolute_position is not None else position, "generation_index": generation_index, "message_index": message_index, "token_ref": token_ref, "capture_stage": capture_stage, "graph_execution": "full" if graph_replays else "eager", "cache_hit_tokens": cache_hit_tokens, "status": "captured", "captured_at": now()}
            accepted = self.queue.submit(_ExportItem(sample_id, arrays, metadata), block=False)
            metadata["status"] = "queued" if accepted else "failed"
            metadata["failure_reason"] = None if accepted else "export_backpressure"
            self._record_manifests(session_id, metadata)
            session.samples.append(metadata)
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
            metadata["token_ref"] = [request.rollout_id, request.step_id, metadata.get("phase"), position, metadata.get("token_id")]
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
            self.request_outputs[request_id] = record
            write_json(self.export_root / "sessions" / session_id / "requests" / (hash_json(request_id) + ".json"), record)
            return {"recorded": True, "request_id": request_id}

    def mark_behavior_failure(self, session_id, reason, *, expected=None, actual=None):
        with self._lock:
            session = self.sessions[session_id]
            session.status = "failed_behavior"
            session.metrics["fallbacks"] += 1
            entry = {"session_id": session_id, "status": session.status,
                     "failure_reason": reason, "expected": expected, "actual": actual,
                     "at": now()}
            write_json(self.export_root / "sessions" / session_id / "behavior-failure.json", entry)
            return entry

    def stop(self, session_id: str):
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            session.status = "stopped"
            for sample in session.samples:
                if sample.get("status") in {"queued", "captured"}:
                    sample["session_status"] = "stopped"
            for sample in session.samples:
                self._record_manifests(session_id, sample, session_status="stopped")
            write_json(
                self.export_root / "sessions" / session_id / "receipt.json",
                {
                    "session_id": session_id,
                    "status": session.status,
                    "metadata": session.requests[next(iter(session.requests))].to_dict()
                    if session.requests
                    else None,
                    "samples": len(session.samples),
                    "metrics": dict(session.metrics),
                    "export": self.queue.status(),
                    "files": {
                        "queue.jsonl": sha256(self.queue.journal)
                        if self.queue.journal.exists()
                        else None
                    },
                },
            )
            return self.status(session_id)

    def status(self, session_id: str | None = None):
        with self._lock:
            if session_id is None:
                return {"sessions": [self._session_status(s) for s in self.sessions.values()], "export": self.queue.status()}
            session = self.sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            return self._session_status(session)

    def _session_status(self, session: _Session):
        return {"session_id": session.session_id, "status": session.status, "created_at": session.created_at, "requests": len(session.requests), "samples": list(session.samples), "metrics": dict(session.metrics), "export": self.queue.status()}

    def results(self, session_id: str):
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            return {"session_id": session_id, "samples": list(session.samples), "status": session.status, "export": self.queue.status()}

    def close(self):
        self.queue.close(drain=True)


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
                    if self.path == "/dycon/capture/behavior-failure":
                        return self._json(200, controller_ref.mark_behavior_failure(
                            value["session_id"], value["reason"], expected=value.get("expected"), actual=value.get("actual")
                        ))
                    if self.path == "/dycon/capture/request/bind":
                        return self._json(200, controller_ref.bind(value["request_id"], value["metadata"]))
                    if self.path == "/dycon/capture/sample":
                        tensors = {k: np.asarray(v) for k, v in value.pop("tensors").items()}
                        return self._json(200, controller_ref.record(value.pop("request_id"), tensors, **value))
                    if self.path == "/dycon/capture/sample-file":
                        return self._json(
                            200,
                            controller_ref.record_file(
                                value.pop("request_id"), value.pop("path"), **value
                            ),
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
        self._thread = threading.Thread(target=self._server.serve_forever, name="dycon-capture-control", daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._server.shutdown()
        self._server.server_close()
        if self._thread:
            self._thread.join(timeout=5)
        self.controller.close()

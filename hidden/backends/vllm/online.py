"""Small client/helpers for wiring rollout requests to the managed capture plane."""

from __future__ import annotations

from typing import Any, Mapping

from core.io import hash_json


def safe_cache_hit_length(num_tokens: int, capture_cache_boundary: int | None) -> int:
    """Return the largest prefix that may be read from APC safely.

    The target token itself is always recomputed. ``None`` retains vLLM's
    normal lookup limit and is used when capture is disabled.
    """
    if num_tokens < 0:
        raise ValueError("num_tokens must be nonnegative")
    if capture_cache_boundary is None:
        return max(0, num_tokens - 1)
    if type(capture_cache_boundary) is not int or capture_cache_boundary < 0:
        raise ValueError("capture_cache_boundary must be a nonnegative integer")
    return min(max(0, num_tokens - 1), capture_cache_boundary)


def build_capture_select(newline_token_ids, close_token_ids, *, include_open=True):
    """Build one native selector for open, reasoning newline, and close rows."""
    newline_ids = sorted({int(x) for x in newline_token_ids})
    close_ids = sorted({int(x) for x in close_token_ids})
    if any(x < 0 for x in [*newline_ids, *close_ids]):
        raise ValueError("capture token IDs must be nonnegative")
    selector = {"generation_tokens": sorted(set(newline_ids + close_ids))}
    if include_open:
        selector["prompt_positions"] = [-2]
    return {"tact_states": selector}


def tokenizer_newline_inventory(tokenizer, *, policy="all"):
    """Return token IDs whose decoded token contains a reasoning newline.

    The inventory is derived from the active tokenizer; IDs are never assumed
    to be stable across checkpoints.
    """
    if policy not in {"all", "double_only"}:
        raise ValueError("unknown newline policy")
    try:
        size = len(tokenizer)
    except TypeError:
        size = tokenizer.vocab_size
    ids = []
    texts = {}
    for token_id in range(size):
        text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        if "\n" not in text:
            continue
        if policy == "double_only" and text != "\n\n":
            continue
        ids.append(token_id)
        texts[str(token_id)] = text
    if not ids:
        raise ValueError("tokenizer has no newline token inventory")
    return ids, texts


def capture_request_metadata(
    session_id: str,
    task_id: str,
    run_id: str,
    rollout_id: str,
    step_id: int,
    anchors,
    trajectory_sha256=None,
    capture_target_position=None,
    capture_cache_boundary=None,
    newline_token_ids=None,
    close_token_ids=None,
    tokenizer_fingerprint=None,
    newline_policy="all",
    capture_select=None,
):
    if type(step_id) is not int or step_id < 0:
        raise ValueError("step_id must be a nonnegative integer")
    anchors = list(anchors)
    if not anchors or any(a not in {"think_open", "reasoning_newline", "think_close"} for a in anchors):
        raise ValueError("invalid capture anchors")
    if newline_policy not in {"all", "double_only"}:
        raise ValueError("invalid newline policy")
    newline_token_ids = sorted({int(x) for x in (newline_token_ids or [])})
    close_token_ids = sorted({int(x) for x in (close_token_ids or [])})
    if any(x < 0 for x in [*newline_token_ids, *close_token_ids]):
        raise ValueError("capture token IDs must be nonnegative")
    selector = capture_select or build_capture_select(
        newline_token_ids if "reasoning_newline" in anchors else [],
        close_token_ids if "think_close" in anchors else [],
        include_open="think_open" in anchors,
    )
    return {
        "session_id": session_id,
        "task_id": task_id,
        "run_id": run_id,
        "rollout_id": rollout_id,
        "step_id": step_id,
        "anchors": anchors,
        "trajectory_sha256": trajectory_sha256,
        "capture_target_position": capture_target_position,
        "capture_cache_boundary": capture_cache_boundary,
        "newline_token_ids": newline_token_ids,
        "close_token_ids": close_token_ids,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "newline_policy": newline_policy,
        "capture_select": selector,
        "capture_select_sha256": hash_json(selector),
    }


class CaptureClient:
    """HTTP control-plane client; tensor payloads stay in the engine process."""

    def __init__(self, base_url: str, timeout: float = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method, path, payload=None):
        import requests

        response = requests.request(method, self.base_url + path, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def start(self, metadata: Mapping[str, Any], **options):
        return self._request("POST", "/dycon/capture/session/start", {"metadata": dict(metadata), **options})

    def bind(self, request_id: str, metadata: Mapping[str, Any]):
        return self._request(
            "POST", "/dycon/capture/request/bind", {"request_id": request_id, "metadata": dict(metadata)}
        )

    def status(self, session_id: str | None = None):
        path = (
            "/dycon/capture/status" if session_id is None else f"/dycon/capture/session/{session_id}/status"
        )
        return self._request("GET", path)

    def stop(self, session_id: str):
        return self._request("POST", f"/dycon/capture/session/{session_id}/stop", {})

    def wait_output(self, session_id, step_id, response_id):
        return self._request(
            "POST",
            "/dycon/capture/request/wait-output",
            {
                "session_id": session_id,
                "step_id": step_id,
                "response_id": response_id,
                "timeout": min(20, self.timeout - 1),
            },
        )

    def finalize(self, session_id, **metadata):
        return self._request("POST", f"/dycon/capture/session/{session_id}/finalize", metadata)

    def prepare(self, session_id, trajectory_path):
        return self._request(
            "POST", f"/dycon/capture/session/{session_id}/prepare", {"trajectory_path": str(trajectory_path)}
        )

    def submit_staged(self, staging_root, request_id: str, tensors: Mapping[str, Any], **metadata):
        """Submit tensor bytes through a shared staging file.

        The control plane receives metadata only.  The server consumes and
        removes the NPZ before placing arrays on its bounded async export
        queue, so large hidden vectors never become JSON request bodies.
        """
        import os
        import tempfile
        from pathlib import Path

        import numpy as np

        root = Path(staging_root)
        root.mkdir(parents=True, exist_ok=True)
        sample = hash_json({"request_id": request_id, **metadata})[:32]
        fd, name = tempfile.mkstemp(prefix=f".{sample}.", suffix=".npz", dir=root)
        os.close(fd)
        path = Path(name)
        try:
            arrays = {}
            for key, value in tensors.items():
                if hasattr(value, "detach"):
                    value = value.detach().to(device="cpu", copy=True).numpy()
                arrays[str(key)] = np.asarray(value)
            np.savez(path, **arrays)
            return self._request(
                "POST",
                "/dycon/capture/sample-file",
                {"request_id": request_id, "path": path.name, **metadata},
            )
        finally:
            path.unlink(missing_ok=True)


def classify_capture_anchor(position, token_id, prompt_length, newline_token_ids, close_token_ids):
    """Classify one native row using its mandatory vLLM labels."""
    if position == prompt_length - 2:
        return "think_open"
    if token_id in set(close_token_ids):
        return "think_close"
    if token_id in set(newline_token_ids):
        return "reasoning_newline"
    return None


def publish_capture_result(
    result,
    rows,
    client: CaptureClient,
    staging_root,
    *,
    newline_token_ids=(),
    close_token_ids=(),
    graph_replays=0,
    cache_hit_tokens=0,
):
    """Publish a native ``CaptureResult`` while preserving row coordinates.

    ``result`` is produced by the pinned EasySteer fetch/assembly code.  The
    function only classifies its labelled rows and forwards layer arrays; it
    does not inspect scheduler order or recreate batching behavior.
    """
    published = []
    newline_ids, close_ids = set(newline_token_ids), set(close_token_ids)
    for index, (row, output) in enumerate(zip(rows, result.outputs, strict=True)):
        prompt_ids = list(output.prompt_token_ids or row["prefix_token_ids"])
        positions = result.sample_positions(index)
        token_ids = result.sample_token_ids(index)
        for row_number, (position, token_id) in enumerate(zip(positions, token_ids, strict=True)):
            anchor = classify_capture_anchor(position, token_id, len(prompt_ids), newline_ids, close_ids)
            if anchor is None:
                continue
            tensors = {
                f"layer_{layer}": result.sample_rows(index, layer)[row_number] for layer in result.layers
            }
            published.append(
                client.submit_staged(
                    staging_root,
                    output.request_id,
                    tensors,
                    anchor=anchor,
                    position=int(position),
                    token_id=int(token_id),
                    absolute_position=int(position),
                    phase="prompt" if position < len(prompt_ids) else "generation_input",
                    generation_index=int(position - len(prompt_ids)) if position >= len(prompt_ids) else -1,
                    graph_replays=graph_replays,
                    cache_hit_tokens=cache_hit_tokens,
                )
            )
    return published

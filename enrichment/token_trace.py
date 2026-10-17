"""Build exact per-token trajectory traces and reasoning-block labels."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from baseline.catalog import Catalog
from core.io import artifact_dir, hash_json, sha256, write_json
from enrichment.reader import TrajectoryReader


def _ids(value):
    return list(value) if isinstance(value, (list, tuple)) and all(type(x) is int for x in value) else None


def _template_prompt(step, reader, tokenizer, cfg):

    # reconstruct() validates the template and exact markers. Its first
    # anchor contains the current prompt prefix when available.
    response = step.get("response") or {}
    actual = _ids(response.get("prompt_token_ids"))
    if actual is not None:
        return actual, "exact_prompt_ids"
    request = step["message"].get("extra", {}).get("request") or {}
    messages = request.get("messages", reader.prepared_prefix(step["message_index"]))
    kwargs = request.get("chat_template_kwargs") or {
        k: cfg["generation"][k] for k in ["enable_thinking", "preserve_thinking"]
    }
    tools = request.get("tools") or [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    try:
        rebuilt = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            **kwargs,
        )
    except Exception:
        # Historical trajectories may contain a serving-specific tool schema
        # that the current local tokenizer cannot render. Preserve the trace
        # as weak text evidence instead of pretending it is exact.
        rendered = "".join(
            f"{m.get('role', '')}:{m.get('content') or ''}:{m.get('reasoning_content') or ''}"
            for m in messages
        ) + "assistant:<think>\n"
        rebuilt = tokenizer.encode(rendered, add_special_tokens=False)
    if hasattr(rebuilt, "keys") and "input_ids" in rebuilt:
        rebuilt = rebuilt["input_ids"]
    if hasattr(rebuilt, "tolist"):
        rebuilt = rebuilt.tolist()
    if isinstance(rebuilt, list) and len(rebuilt) == 1 and isinstance(rebuilt[0], list):
        rebuilt = rebuilt[0]
    if not isinstance(rebuilt, list) or not all(type(x) is int for x in rebuilt):
        raise ValueError("Tokenizer must return one prompt token list")
    reported = (response.get("usage") or {}).get("prompt_tokens")
    evidence = "text_reconstructed"
    if reported is not None:
        evidence = "template_and_prompt_count" if reported == len(rebuilt) else "template_count_mismatch"
    return rebuilt, evidence


def _find(ids, marker, start=0):
    if not marker:
        return None
    for i in range(start, len(ids) - len(marker) + 1):
        if ids[i : i + len(marker)] == marker:
            return i
    return None


def _token_text(tokenizer, token_id):
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


def _append_row(rows, *, token_id, step_id, message_index, phase, request_index, absolute, generation_index, block_id=-1, offset=-1, length=-1, kind=0, newline_count=0, token_text=""):
    rows.append(
        {
            "token_id": int(token_id),
            "step_id": int(step_id),
            "message_index": int(message_index),
            "phase": int(phase),
            "request_token_index": int(request_index),
            "absolute_position": int(absolute),
            "generation_index": int(generation_index),
            "block_id": int(block_id),
            "reasoning_offset": int(offset),
            "reasoning_length": int(length),
            "block_remaining_reasoning_after": -1,
            "remaining_reasoning_after": -1,
            "special_kind": int(kind),
            "newline_count": int(newline_count),
            "token_text": token_text,
        }
    )


def trace_rollout(cfg, rollout_id, tokenizer, dataset_id=None):
    root = Path(cfg["project_root"])
    catalog = Catalog(root)
    selected = catalog.select(
        partition=cfg["benchmark"]["partition"],
        purposes=cfg["benchmark"]["purposes"],
        profile=cfg["benchmark"]["profile"],
    )
    if rollout_id not in selected["rollouts"]:
        raise ValueError(f"Rollout is outside the selected scope: {rollout_id}")
    rollout = selected["rollouts"][rollout_id]
    source = root / rollout["trajectory_path"]
    if sha256(source) != rollout["trajectory_sha256"]:
        raise ValueError("Trajectory checksum mismatch")
    reader = TrajectoryReader(source)
    steps = list(reader.steps())
    open_ids = tokenizer.encode("<think>", add_special_tokens=False)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if not open_ids or tokenizer.decode(open_ids) != "<think>":
        raise ValueError("Tokenizer has no exact <think> marker")
    if not close_ids or tokenizer.decode(close_ids) != "</think>":
        raise ValueError("Tokenizer has no exact </think> marker")

    rows, blocks, step_records = [], [], []
    for step_id, step in enumerate(steps):
        response = step.get("response") or {}
        prompt, prompt_evidence = _template_prompt(step, reader, tokenizer, cfg)
        output = _ids(response.get("output_token_ids"))
        output_evidence = "exact_output_ids"
        if output is None:
            reasoning_ids = tokenizer.encode(step.get("reasoning") or "", add_special_tokens=False)
            content_ids = tokenizer.encode(step.get("content") or "", add_special_tokens=False)
            output = list(reasoning_ids)
            finish = (response.get("choices") or [{}])[0].get("finish_reason")
            # A parser may omit close from text. Reconstruct it only as weak evidence.
            if reasoning_ids and finish != "length":
                output += list(close_ids)
            output += list(content_ids)
            output_evidence = "text_reconstructed"
        evidence = "exact_prompt_and_output_ids" if prompt_evidence == "exact_prompt_ids" and output_evidence == "exact_output_ids" else f"{prompt_evidence}+{output_evidence}"
        prompt_open = _find(prompt, open_ids, max(0, len(prompt) - 16))
        generated_open = _find(output, open_ids)
        if prompt_open is not None and prompt_open + len(open_ids) <= len(prompt):
            content_start = 0
            open_absolute = prompt_open
        elif generated_open is not None:
            content_start = generated_open + len(open_ids)
            open_absolute = len(prompt) + generated_open
        else:
            content_start = 0
            open_absolute = None
        close_start = _find(output, close_ids, content_start)
        content_end = close_start if close_start is not None else len(output)
        complete = close_start is not None and output_evidence == "exact_output_ids"
        block_id = len(blocks)
        content_count = max(content_end - content_start, 0)
        block = {
            "block_id": block_id,
            "step_id": step_id,
            "message_index": step["message_index"],
            "content_start_generation_index": content_start,
            "content_end_generation_index": content_end,
            "content_token_count": content_count,
            "complete": complete,
            "evidence": evidence,
            "open_absolute_position": open_absolute,
            "close_absolute_position": len(prompt) + close_start if close_start is not None else None,
        }
        blocks.append(block)
        step_records.append({"step_id": step_id, "message_index": step["message_index"], "prompt_length": len(prompt), "output_length": len(output), "evidence": evidence, "block_id": block_id})
        for i, token_id in enumerate(prompt):
            kind = 1 if prompt_open is not None and prompt_open <= i < prompt_open + len(open_ids) else 0
            _append_row(rows, token_id=token_id, step_id=step_id, message_index=step["message_index"], phase=0, request_index=i, absolute=i, generation_index=-1, kind=kind, token_text=_token_text(tokenizer, token_id))
        for i, token_id in enumerate(output):
            kind, newline_count, offset, length = 0, 0, -1, -1
            if content_start <= i < content_end:
                text = _token_text(tokenizer, token_id)
                newline_count = text.count("\n")
                kind = 2 if newline_count else 0
                offset, length = i - content_start, content_count
            elif close_start is not None and close_start <= i < close_start + len(close_ids):
                kind = 3
            row_block = block_id if (content_start <= i < content_end or (close_start is not None and close_start <= i < close_start + len(close_ids))) else -1
            _append_row(rows, token_id=token_id, step_id=step_id, message_index=step["message_index"], phase=1, request_index=len(prompt) + i, absolute=len(prompt) + i, generation_index=i, block_id=row_block, offset=offset, length=length, kind=kind, newline_count=newline_count, token_text=_token_text(tokenizer, token_id))

    # Fill global future-reasoning labels in trajectory/block order.
    suffix = 0
    for block in reversed(blocks):
        local_start = block["content_start_generation_index"]
        local_end = block["content_end_generation_index"]
        step_rows = [i for i, r in enumerate(rows) if r["step_id"] == block["step_id"] and r["phase"] == 1 and local_start <= r["generation_index"] < local_end]
        for row_index in step_rows:
            row = rows[row_index]
            local_after = local_end - row["generation_index"] - 1
            row["block_remaining_reasoning_after"] = local_after
            row["remaining_reasoning_after"] = local_after + suffix
        suffix += block["content_token_count"]

    newline_events = [
        {"trace_index": i, "step_id": row["step_id"], "absolute_position": row["absolute_position"], "token_id": row["token_id"], "token_text": row["token_text"], "newline_count": row["newline_count"], "block_id": row["block_id"], "reasoning_offset": row["reasoning_offset"], "reasoning_length": row["reasoning_length"]}
        for i, row in enumerate(rows) if row["special_kind"] == 2
    ]
    if dataset_id is None:
        out = artifact_dir(cfg, "enrichment", rollout_id, "token-traces")
    else:
        out = artifact_dir(cfg, "enrichment", dataset_id) / "token-traces" / rollout_id
    out.mkdir(parents=True, exist_ok=True)
    array_names = ["token_ids", "step_ids", "message_indices", "phases", "request_token_indices", "absolute_positions", "generation_indices", "block_ids", "reasoning_offsets", "reasoning_lengths", "block_remaining_reasoning_after", "remaining_reasoning_after", "special_kinds", "newline_counts"]
    field_map = {"token_ids": "token_id", "step_ids": "step_id", "message_indices": "message_index", "phases": "phase", "request_token_indices": "request_token_index", "absolute_positions": "absolute_position", "generation_indices": "generation_index", "block_ids": "block_id", "reasoning_offsets": "reasoning_offset", "reasoning_lengths": "reasoning_length", "block_remaining_reasoning_after": "block_remaining_reasoning_after", "remaining_reasoning_after": "remaining_reasoning_after", "special_kinds": "special_kind", "newline_counts": "newline_count"}
    arrays = {name: np.asarray([row[field_map[name]] for row in rows], dtype=np.int64) for name in array_names}
    npz_path = out / "token_trace.npz"
    with tempfile.NamedTemporaryFile(dir=out, suffix=".npz", delete=False) as handle:
        temp = Path(handle.name)
    try:
        np.savez_compressed(temp, **arrays)
        temp.replace(npz_path)
    finally:
        temp.unlink(missing_ok=True)
    metadata = {"schema_version": 2, "rollout_id": rollout_id, "trajectory_sha256": rollout["trajectory_sha256"], "model_fingerprint": cfg["model"]["fingerprint"], "tokenizer_name": getattr(tokenizer, "name_or_path", None), "tokenizer_fingerprint": hash_json({"name": getattr(tokenizer, "name_or_path", None), "open_ids": open_ids, "close_ids": close_ids}), "alignment_evidence": sorted({x["evidence"] for x in step_records}), "token_count": len(rows), "steps": step_records, "reasoning_blocks": blocks, "newline_events": newline_events, "files": {"token_trace.npz": sha256(npz_path)}}
    write_json(out / "token_trace.json", metadata)
    metadata["files"]["token_trace.json"] = sha256(out / "token_trace.json")
    write_json(out / "token_trace.receipt.json", metadata)
    return out


def build_token_traces(cfg, dataset_id, tokenizer):
    selected = Catalog(cfg["project_root"]).select(partition=cfg["benchmark"]["partition"], purposes=cfg["benchmark"]["purposes"], profile=cfg["benchmark"]["profile"])
    if not selected["rollout_ids"]:
        raise ValueError("No explicit rollout scope has been registered")
    result = []
    for rollout_id in selected["rollout_ids"]:
        result.append(trace_rollout(cfg, rollout_id, tokenizer, dataset_id=dataset_id))
    return result


def audit_token_traces(cfg, dataset_id):
    """Summarize evidence and marker coverage without re-tokenizing data."""
    root = artifact_dir(cfg, "enrichment", dataset_id) / "token-traces"
    if not root.is_dir():
        raise FileNotFoundError(root)
    report = {
        "dataset_id": dataset_id,
        "rollouts": 0,
        "token_count": 0,
        "newline_events": 0,
        "evidence": {},
        "complete_blocks": 0,
        "incomplete_blocks": 0,
    }
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not (path / "token_trace.json").exists():
            continue
        meta = json.loads((path / "token_trace.json").read_text(encoding="utf-8"))
        report["rollouts"] += 1
        report["token_count"] += int(meta.get("token_count", 0))
        report["newline_events"] += len(meta.get("newline_events", []))
        for evidence in meta.get("alignment_evidence", []):
            report["evidence"][evidence] = report["evidence"].get(evidence, 0) + 1
        blocks = meta.get("reasoning_blocks", [])
        report["complete_blocks"] += sum(bool(x.get("complete")) for x in blocks)
        report["incomplete_blocks"] += sum(not bool(x.get("complete")) for x in blocks)
    return report

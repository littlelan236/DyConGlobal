"""Strict join between token traces and hidden capture index rows."""
from __future__ import annotations

import json

import numpy as np

from core.io import artifact_dir, hash_json, read_json, sha256, write_json
from difficulty.datasets.paths import dataset_directory

_KIND = {"think_open": 1, "reasoning_newline": 2, "think_close": 3}


def _load_trace(cfg, dataset_id, rollout_id):
    path = artifact_dir(cfg, "enrichment", dataset_id) / "token-traces" / rollout_id
    meta = read_json(path / "token_trace.json")
    arrays = np.load(path / "token_trace.npz")
    return path, meta, arrays


def _candidate_indices(meta, arrays, hidden):
    anchor = hidden.get("anchor")
    if anchor not in _KIND:
        raise ValueError(f"Unsupported hidden anchor: {anchor}")
    mask = arrays["special_kinds"] == _KIND[anchor]
    if hidden.get("rollout_id") and hidden["rollout_id"] != meta["rollout_id"]:
        return []
    if hidden.get("step_id") is not None:
        mask &= arrays["step_ids"] == int(hidden["step_id"])
    if hidden.get("anchor_position") is not None:
        mask &= arrays["absolute_positions"] == int(hidden["anchor_position"])
    if hidden.get("anchor_token_id") is not None:
        mask &= arrays["token_ids"] == int(hidden["anchor_token_id"])
    if hidden.get("token_id") is not None:
        mask &= arrays["token_ids"] == int(hidden["token_id"])
    return np.flatnonzero(mask).tolist()


def build_token_hidden_index(cfg, dataset_id, capture_id, index_id, rollout_ids=(), rollout_list=None):
    source_run = cfg.get("source_run_id") or cfg.get("run_id")
    if not source_run:
        raise ValueError("Dataset trajectory selection requires an explicit source run")
    from difficulty.datasets.selection import select_run
    scope = select_run(cfg["project_root"], source_run, rollout_ids, rollout_list)
    selected_ids = set(scope["rollout_ids"])
    from application.execution.evidence import RunEvidence
    all_run_ids = set(RunEvidence(cfg["project_root"], source_run).read()["rollouts"])

    source = artifact_dir(cfg, "enrichment", dataset_id)
    capture = artifact_dir(cfg, "hidden", capture_id)
    enrichment_receipt = read_json(source / "receipt.json")
    hidden_receipt = read_json(capture / "receipt.json")
    if hidden_receipt.get("source_receipt_sha256") != sha256(source / "receipt.json"):
        raise ValueError("Hidden capture was created from a different enrichment dataset")
    hidden_rows = read_json(capture / "index.json")
    if not hidden_rows:
        raise ValueError("No hidden rows to join")
    rows, trace_cache = [], {}
    for hidden in hidden_rows:
        rid = hidden["rollout_id"]
        if rid not in all_run_ids:
            raise ValueError("Hidden trajectory belongs to another source run")
        if rid not in selected_ids:
            continue
        if rid not in trace_cache:
            trace_cache[rid] = _load_trace(cfg, dataset_id, rid)
        _, trace_meta, arrays = trace_cache[rid]
        if trace_meta.get("alignment_evidence") != ["exact_prompt_and_output_ids"]:
            raise ValueError("Exact hidden-token join rejects reconstructed/weak token evidence")
        candidates = _candidate_indices(trace_meta, arrays, hidden)
        if len(candidates) != 1:
            raise ValueError(
                f"Hidden row {hidden.get('sample_id')} matched {len(candidates)} token rows; "
                "absolute position and token ID must be explicit"
            )
        index = candidates[0]
        hidden_path = capture / hidden["path"]
        if sha256(hidden_path) != hidden_receipt["files"][hidden["path"]]:
            raise ValueError("Hidden checksum mismatch")
        rows.append(
            {
                "event_id": hidden.get("sample_id"),
                "sample_id": hidden.get("sample_id"),
                "rollout_id": rid,
                "step_id": int(arrays["step_ids"][index]),
                "message_index": int(arrays["message_indices"][index]),
                "absolute_position": int(arrays["absolute_positions"][index]),
                "generation_index": int(arrays["generation_indices"][index]),
                "token_id": int(arrays["token_ids"][index]),
                "anchor": hidden["anchor"],
                "trace_index": index,
                "block_id": int(arrays["block_ids"][index]),
                "reasoning_offset": int(arrays["reasoning_offsets"][index]),
                "reasoning_length": int(arrays["reasoning_lengths"][index]),
                "block_remaining_reasoning_tokens_after": int(arrays["block_remaining_reasoning_after"][index]),
                "remaining_reasoning_tokens_after": int(arrays["remaining_reasoning_after"][index]),
                "hidden_path": hidden["path"],
                "prefix_sha256": hidden.get("prefix_sha256"),
                "alignment_status": "exact",
                "alignment_evidence": trace_meta["alignment_evidence"],
            }
        )
    out = artifact_dir(cfg, "difficulty", index_id, "datasets")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    receipt = {
        "schema_version": 2,
        "state": "complete",
        "index_id": index_id,
        "dataset_id": dataset_id,
        "capture_id": capture_id,
        "source_receipt_sha256": sha256(source / "receipt.json"),
        "capture_receipt_sha256": sha256(capture / "receipt.json"),
        "row_count": len(rows),
        "alignment_counts": {"exact": len(rows), "weak": 0, "failed": 0},
        "files": {"rows.jsonl": sha256(out / "rows.jsonl")},
        "identity": hash_json({"dataset": enrichment_receipt, "capture": hidden_receipt["input_identity"]}),
    }
    write_json(out / "receipt.json", receipt)
    return out


def audit_token_hidden_index(cfg, index_id):
    """Audit a materialized join without changing its rows."""
    out = dataset_directory(cfg, index_id)
    rows_path = out / "rows.jsonl"
    receipt_path = out / "receipt.json"
    if not rows_path.exists() or not receipt_path.exists():
        raise FileNotFoundError(out)
    counts = {}
    total = 0
    with rows_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            status = row.get("alignment_status", "missing")
            counts[status] = counts.get(status, 0) + 1
    receipt = read_json(receipt_path)
    if receipt.get("files", {}).get("rows.jsonl") != sha256(rows_path):
        raise ValueError("Join rows checksum mismatch")
    return {
        "index_id": index_id,
        "rows": total,
        "alignment_counts": counts,
        "receipt_sha256": sha256(receipt_path),
        "passed": counts.get("failed", 0) == 0 and counts.get("missing", 0) == 0,
    }

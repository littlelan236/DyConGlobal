"""Finalize reasoning events from exact online token receipts, without retokenizing."""

from collections import defaultdict
from pathlib import Path

from core.io import hash_json, owned_path, read_json, sha256, write_json
from hidden.artifacts import event_records, read_arrays, read_request


def publication_directory(project_root, run_id, publication_id=None):
    """Resolve an explicit immutable trace without aliasing another run."""
    from core.io import identifier
    relative = f"runs/{identifier(run_id)}/enrichment/online"
    if publication_id is not None:
        relative = f"runs/{identifier(run_id)}/enrichment/online-publications/{identifier(publication_id)}"
    return owned_path(project_root, relative)


def reasoning_blocks(prompt, output, open_ids, close_ids):
    if not open_ids or not close_ids:
        raise ValueError("Reasoning marker token IDs must be nonempty")

    def last(ids, marker):
        return max(
            (i for i in range(len(ids) - len(marker) + 1) if ids[i : i + len(marker)] == marker), default=-1
        )

    active = last(prompt, open_ids) > last(prompt, close_ids)
    start = 0 if active else None
    blocks, i = [], 0
    while i < len(output):
        if output[i : i + len(close_ids)] == close_ids and active:
            blocks.append({"start": start, "end": i, "close": i, "complete": True})
            active = False
            start = None
            i += len(close_ids)
        elif output[i : i + len(open_ids)] == open_ids:
            if active:
                raise ValueError("Nested reasoning markers in exact output")
            active = True
            start = i + len(open_ids)
            i += len(open_ids)
        else:
            i += 1
    if active:
        blocks.append({"start": start, "end": len(output), "close": None, "complete": False})
    return blocks


def reasoning_targets(request, open_ids, close_ids):
    prompt, output = request["prompt_token_ids"], request["output_token_ids"]
    anchors = request.get("anchors", ["reasoning_newline", "think_close"])
    targets, availability = set(), []
    for block in reasoning_blocks(prompt, output, open_ids, close_ids):
        if "reasoning_newline" in anchors:
            for gi in range(block["start"], min(block["end"], len(output) - 1)):
                if output[gi] in request["newline_token_ids"]:
                    targets.add((len(prompt) + gi, "reasoning_newline"))
        close = block["close"]
        if close is not None and "think_close" in anchors:
            available = close < len(output) - 1
            availability.append(
                {
                    "position": len(prompt) + close,
                    "status": "available" if available else "no_followup_forward",
                }
            )
            if available:
                targets.add((len(prompt) + close, "think_close"))
    return targets, availability


def finalize_token_events(export_root, output, *, open_ids, close_ids,
                          request_evidence=None, event_rows=None, trajectory_paths=None, finalization_paths=None, allow_missing=False):
    """Finalize an export or an explicitly bounded migrated session.

    Optional inputs bind canonical ledger hashes, exact event rows and current
    trajectory locators; original NAS finalization bytes are never rewritten.
    """
    root, out = Path(export_root), Path(output)
    out.mkdir(parents=True, exist_ok=False)
    requests, trajectories = {}, defaultdict(list)
    rejected_requests = set()
    source_files = {}
    publications = {}
    finalizations = {}
    verified_trajectories = set()
    paths = ([(owned_path(root, row['ledger_ref']), row['ledger_sha256']) for row in request_evidence]
             if request_evidence is not None else
             [(path, None) for path in sorted([*(root / "sessions").glob("*/requests/*.json"),
                                               *(root / "requests").glob("*.json")])])
    for path, expected_sha in paths:
        digest = sha256(path)
        if expected_sha is not None and digest != expected_sha:
            raise ValueError('Canonical request ledger checksum mismatch')
        request = read_request(
            root, {"ledger_ref": str(path.relative_to(root)), "ledger_sha256": digest}
        )
        if request["session_id"] == "engine-warmup" or request["session_id"].startswith("startup-"):
            continue
        request["ledger_ref"] = str(path.relative_to(root))
        request["ledger_sha256"] = sha256(path)
        finalization_path = (owned_path(root, finalization_paths[request["session_id"]]) if finalization_paths is not None else root / "sessions" / request["session_id"] / "finalization.json")
        if request.get("schema_version", 1) >= 2 and not finalization_path.exists():
            rejected_requests.add(request["request_id"])
            continue
        if finalization_path.exists():
            if finalization_path not in finalizations:
                finalizations[finalization_path] = read_json(finalization_path)
                source_files[str(finalization_path.relative_to(root))] = sha256(finalization_path)
            finalization = finalizations[finalization_path]
            if request["request_id"] not in finalization["accepted_native_request_ids"]:
                rejected_requests.add(request["request_id"])
                continue
            trajectory_path = (trajectory_paths[finalization['rollout_id']]
                               if trajectory_paths is not None else finalization['trajectory_path'])
            identity = (str(trajectory_path), finalization['trajectory_sha256'])
            if identity not in verified_trajectories:
                if sha256(trajectory_path) != finalization["trajectory_sha256"]:
                    raise ValueError("Finalized source trajectory changed")
                verified_trajectories.add(identity)
            request.update(
                {key: finalization[key] for key in ["rollout_id", "trajectory_sha256", "model_fingerprint"]}
            )
            request.update(finalization.get("responses", {}).get(request["request_id"], {}))
        if request["request_id"] in requests:
            raise ValueError("Duplicate native request receipt")
        if hash_json(request["prompt_token_ids"]) != request.get("request_sha256"):
            raise ValueError("Request token checksum mismatch")
        request["blocks"] = reasoning_blocks(
            request["prompt_token_ids"], request["output_token_ids"], open_ids, close_ids
        )
        requests[request["request_id"]] = request
        trajectories[request["rollout_id"]].append(request)
        source_files[str(path.relative_to(root))] = digest
    for rollout_id, steps in trajectories.items():
        steps.sort(key=lambda step: step["step_id"])
        if len({step["step_id"] for step in steps}) != len(steps):
            raise ValueError("Several requests claim the same trajectory step")
        task_ids = {step["task_id"] for step in steps}
        if len(task_ids) != 1:
            raise ValueError("A trajectory spans several task identities")
        trajectory_hash = hash_json(
            [{"prompt": step["prompt_token_ids"], "output": step["output_token_ids"]} for step in steps]
        )
        suffix, censored = 0, False
        for step_index in range(len(steps) - 1, -1, -1):
            step = steps[step_index]
            step["remaining_steps"] = sum(int(s.get("is_action_response", True)) for s in steps[step_index:])
            step["exact_token_trace_sha256"] = trajectory_hash
            for block in reversed(step["blocks"]):
                block["future_reasoning_tokens"] = suffix
                censored = censored or not block["complete"]
                block["remaining_censored"] = censored
                suffix += block["end"] - block["start"]
    accepted, excluded = [], []
    for path, event in (event_records(root) if event_rows is None else event_rows):
        if event["request_id"] in rejected_requests:
            excluded.append({"sample_id": event["sample_id"], "reason": "not_in_finalized_trajectory"})
            continue
        if event["request_id"] not in requests:
            excluded.append({"sample_id": event["sample_id"], "reason": "diagnostic_or_unfinalized"})
            continue
        request = requests[event["request_id"]]
        prompt, generated = request["prompt_token_ids"], request["output_token_ids"]
        position = event["absolute_position"]
        gi = position - len(prompt)
        all_ids = prompt + generated
        if not 0 <= gi < len(generated) - 1 or all_ids[position] != event["token_id"]:
            raise ValueError("Hidden event has no exact subsequent input-forward evidence")
        if event["request_id"] not in requests:
            excluded.append({"sample_id": event["sample_id"], "reason": "diagnostic_or_unfinalized"})
            continue
        locator = event.get("tensor_locator")
        publication = root / "publications" / (Path(locator["shard"]).stem + ".json") if locator else None
        if publication and publication.exists():
            if str(publication) not in publications:
                publications[str(publication)] = read_json(publication)
            proof = publications[str(publication)]
            relative_index = str(path.relative_to(root))
            if relative_index not in source_files:
                source_files[relative_index] = sha256(path)
            if (
                proof["files"].get(locator["shard"]) != locator["shard_sha256"]
                or proof["files"].get(relative_index) != source_files[relative_index]
                or event["sha256"] != locator["member_sha256"]
            ):
                raise ValueError("Published tensor/index evidence mismatch")
        else:
            read_arrays(root, event)  # legacy exports have no publication receipt
        candidates = [
            (i, block)
            for i, block in enumerate(request["blocks"])
            if (event["anchor"] == "reasoning_newline" and block["start"] <= gi < block["end"])
            or (
                event["anchor"] == "think_close"
                and block["close"] is not None
                and block["close"] <= gi < block["close"] + len(close_ids)
            )
        ]
        if len(candidates) != 1:
            excluded.append(
                {"sample_id": event["sample_id"], "reason": "outside_reasoning_or_repeated_close"}
            )
            continue
        block_index, block = candidates[0]
        local_after = max(0, block["end"] - gi - 1)
        offset = gi - block["start"] if gi < block["end"] else None
        length = block["end"] - block["start"]
        accepted.append(
            {
                **event,
                "rollout_id": request["rollout_id"],
                "trajectory_sha256": request.get("trajectory_sha256"),
                "path": event["tensor"],
                "phase": "generation_input",
                "ledger_ref": request["ledger_ref"],
                "ledger_sha256": request["ledger_sha256"],
                "generation_index": gi,
                "token_ref": [
                    request["rollout_id"],
                    request["step_id"],
                    "generation_input",
                    position,
                    event["token_id"],
                ],
                "prefix_sha256": hash_json(all_ids[: position + 1]),
                "request_sha256": request["request_sha256"],
                "exact_token_trace_sha256": request["exact_token_trace_sha256"],
                "block_id": block_index,
                "reasoning_offset": offset,
                "reasoning_length": length,
                "reasoning_fraction": (offset + 1) / length if offset is not None and length else None,
                "block_remaining_reasoning_tokens_after": local_after,
                "remaining_reasoning_tokens_after": local_after + block["future_reasoning_tokens"],
                "remaining_steps": request["remaining_steps"],
                "alignment_status": "exact",
                "alignment_evidence": "native_prompt_and_output_token_ids",
                "censored": block["remaining_censored"],
            }
        )
        relative = str(path.relative_to(root))
        if relative not in source_files:
            source_files[relative] = sha256(path)
    accepted.sort(
        key=lambda row: (row["task_id"], row["rollout_id"], row["step_id"], row["absolute_position"])
    )
    write_json(out / "index.json", accepted)
    write_json(out / "excluded.json", excluded)
    summaries = {
        key: {
            k: v
            for k, v in value.items()
            if k not in {"prompt_token_ids", "output_token_ids", "newline_token_ids", "capture_select"}
        }
        for key, value in requests.items()
    }
    write_json(out / "requests.json", summaries)
    expected, observed, availability = set(), set(), []
    for row in accepted:
        observed.add((row["request_id"], row["absolute_position"], row["anchor"]))
    for request_id, request in requests.items():
        targets, available = reasoning_targets(request, open_ids, close_ids)
        expected.update((request_id, position, anchor) for position, anchor in targets)
        availability.extend({"request_id": request_id, **row} for row in available)
    coverage = {
        "expected": len(expected),
        "observed": len(observed),
        "missing": sorted(expected - observed),
        "unexpected": sorted(observed - expected),
        "close_availability": availability,
    }
    coverage["policy"] = "available_points" if allow_missing else "complete"
    coverage["expected_count_scope"] = "explicitly_bound_native_requests"
    write_json(out / "coverage.json", coverage)
    receipt = {
        "schema_version": 1,
        "source_root": str(root),
        "source_files": source_files,
        "events": len(accepted),
        "excluded_events": len(excluded),
        "requests": len(requests),
        "labels": "observed trajectory remainder; censored blocks explicitly flagged",
        "files": {
            "index.json": sha256(out / "index.json"),
            "requests.json": sha256(out / "requests.json"),
            "excluded.json": sha256(out / "excluded.json"),
        },
    }
    write_json(out / "receipt.json", receipt)
    if coverage["unexpected"] or (coverage["missing"] and not allow_missing):
        raise ValueError("Online reasoning-event coverage differs from exact token receipts")
    return out

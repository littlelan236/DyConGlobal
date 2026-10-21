"""Causal boundary reconstruction; gold outcomes are labels, never inputs."""

from __future__ import annotations
from application.execution.selection import rollout_selection

import copy
import json
from pathlib import Path

from catalog.store import Catalog
from core.io import artifact_dir, hash_json, read_json, sha256, write_json
from core.locking import exclusive_artifact
from enrichment.reader import TrajectoryReader


def template_messages(messages):
    result = []
    for message in messages:
        if message.get("role") == "exit":
            raise ValueError("Terminal message inside a model request")
        item = {
            k: copy.deepcopy(v)
            for k, v in message.items()
            if k in {"role", "content", "reasoning_content", "tool_calls", "tool_call_id", "name"}
        }
        for call in item.get("tool_calls") or []:
            if isinstance(call.get("function", {}).get("arguments"), str):
                call["function"] = {
                    **call["function"],
                    "arguments": json.loads(call["function"]["arguments"]),
                }
        result.append(item)
    return result


def occurrences(ids, marker):
    return [i for i in range(len(ids) - len(marker) + 1) if ids[i : i + len(marker)] == marker]


def reasoning_count(step, tokenizer):
    response = step.get("response") or {}
    if isinstance(response.get("prompt_token_ids"), list) and isinstance(
        response.get("output_token_ids"), list
    ):
        from enrichment.online_trace import reasoning_blocks

        blocks = reasoning_blocks(
            response["prompt_token_ids"],
            response["output_token_ids"],
            tokenizer.encode("<think>", add_special_tokens=False),
            tokenizer.encode("</think>", add_special_tokens=False),
        )
        return sum(block["end"] - block["start"] for block in blocks), "exact_native_ids"
    usage = response.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    n = details.get("reasoning_tokens")
    if type(n) is int and n >= 0:
        return n, "server_usage"
    if not isinstance(step.get("response"), dict):
        return None, "missing_response"
    if step["reasoning"]:
        return len(tokenizer.encode(step["reasoning"], add_special_tokens=False)), "text_retokenized"
    # No reasoning field is not proof that zero thinking tokens were generated.
    return None, "missing_reasoning"


def tools_schema():
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The bash command to execute"}
                    },
                    "required": ["command"],
                },
            },
        }
    ]


def reconstruct(step, reader, tokenizer, cfg):
    response = step.get("response") or {}
    message = step["message"]
    request = message.get("extra", {}).get("request") or {}
    kwargs = request.get("chat_template_kwargs") or {
        k: cfg["generation"][k] for k in ["enable_thinking", "preserve_thinking"]
    }
    tools = request.get("tools", tools_schema())
    messages = template_messages(request.get("messages", reader.prepared_prefix(step["message_index"])))
    rebuilt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=True, **kwargs
    )
    # Transformers 5 tokenizers may return BatchEncoding even for one chat.
    if hasattr(rebuilt, "keys") and "input_ids" in rebuilt:
        rebuilt = rebuilt["input_ids"]
    if hasattr(rebuilt, "tolist"):
        rebuilt = rebuilt.tolist()
    if isinstance(rebuilt, list) and len(rebuilt) == 1 and isinstance(rebuilt[0], list):
        rebuilt = rebuilt[0]
    if not isinstance(rebuilt, list) or not all(type(x) is int for x in rebuilt):
        raise ValueError("Tokenizer must return a single list of token IDs")
    actual = response.get("prompt_token_ids")
    reported = (response.get("usage") or {}).get("prompt_tokens")
    if actual is not None:
        if actual != rebuilt:
            raise ValueError("Actual prompt IDs differ from reconstructed template")
        prompt, evidence = actual, "exact_prompt_ids"
    else:
        prompt, evidence = rebuilt, "text_reconstructed"
        if reported is not None:
            if reported != len(prompt):
                raise ValueError(f"Prompt count mismatch: {len(prompt)} != {reported}")
            evidence = "template_and_prompt_count"
    open_ids = tokenizer.encode("<think>", add_special_tokens=False)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if (
        not open_ids
        or tokenizer.decode(open_ids) != "<think>"
        or not close_ids
        or tokenizer.decode(close_ids) != "</think>"
    ):
        raise ValueError("Tokenizer has no exact think markers")
    anchors, unavailable = [], []
    opens = occurrences(prompt, open_ids)
    if "think_open" in cfg["enrichment"]["anchors"]:
        if opens:
            pos = opens[-1] + len(open_ids)
            # Last prompt marker must belong to this generation, not prior history.
            suffix = tokenizer.decode(prompt[pos:])
            if suffix.strip() == "":
                anchors.append(
                    {
                        "anchor": "think_open",
                        "occurrence": 0,
                        "prefix_token_ids": prompt[:pos],
                        "evidence": evidence,
                    }
                )
            else:
                unavailable.append({"anchor": "think_open", "reason": "no_current_open_in_prompt"})
        else:
            unavailable.append({"anchor": "think_open", "reason": "no_current_open_in_prompt"})
        if not any(a["anchor"] == "think_open" for a in anchors):
            generated = response.get("output_token_ids")
            generated_opens = occurrences(generated, open_ids) if generated is not None else []
            if generated_opens:
                end = generated_opens[0] + len(open_ids)
                anchors.append(
                    {
                        "anchor": "think_open",
                        "occurrence": 0,
                        "prefix_token_ids": prompt + generated[:end],
                        "evidence": "exact_ids" if actual is not None else evidence,
                    }
                )
                unavailable = [x for x in unavailable if x["anchor"] != "think_open"]
    if "think_close" in cfg["enrichment"]["anchors"]:
        generated = response.get("output_token_ids")
        if generated is not None:
            positions = occurrences(generated, close_ids)
            if cfg["enrichment"]["close_occurrence"] == "first":
                positions = positions[:1]
            for n, index in enumerate(positions):
                anchors.append(
                    {
                        "anchor": "think_close",
                        "occurrence": n,
                        "prefix_token_ids": prompt + generated[: index + len(close_ids)],
                        "evidence": "exact_ids" if actual is not None else evidence,
                    }
                )
            if not positions:
                unavailable.append({"anchor": "think_close", "reason": "close_absent_from_output_ids"})
        else:
            finish = response.get("choices", [{}])[0].get("finish_reason")
            if step["reasoning"].strip() and finish != "length":
                if cfg["enrichment"]["close_occurrence"] == "all":
                    raise ValueError("All-close capture requires actual output token IDs")
                # Reconstructed close assumes the serving reasoning parser removed it.
                # It is explicitly lower evidence; no future action enters the prefix.
                anchors.append(
                    {
                        "anchor": "think_close",
                        "occurrence": 0,
                        "prefix_token_ids": prompt
                        + tokenizer.encode(step["reasoning"], add_special_tokens=False)
                        + close_ids,
                        "evidence": "reasoning_parser_reconstruction",
                    }
                )
            else:
                unavailable.append({"anchor": "think_close", "reason": "missing_or_truncated_reasoning"})
    if cfg["enrichment"]["token_evidence"] == "exact":
        anchors = [a for a in anchors if a["evidence"] in {"exact_ids", "exact_prompt_ids"}]
    return anchors, unavailable


@exclusive_artifact("enrichment", "dataset_id")
def enrich(cfg, dataset_id, tokenizer):
    catalog = Catalog(cfg["project_root"])
    catalog.verify()
    selected = rollout_selection(cfg)
    if not selected["rollout_ids"]:
        raise ValueError("No explicit rollout scope has been registered")
    out = artifact_dir(cfg, "enrichment", dataset_id)
    out.mkdir(parents=True, exist_ok=True)
    identity = hash_json({"selection": selected, "config": cfg["config_hash"]})
    if (out / "receipt.json").exists():
        receipt = read_json(out / "receipt.json")
        if receipt["input_identity"] != identity:
            raise ValueError("Dataset ID already belongs to different inputs/configuration")
        if any(sha256(out / p) != digest for p, digest in receipt["files"].items()):
            raise ValueError("Enrichment artifact changed")
        return out
    files, reports = {}, {}
    for rid, rollout in selected["rollouts"].items():
        if rollout["model_fingerprint"] != cfg["model"]["fingerprint"]:
            raise ValueError(f"Model fingerprint mismatch for {rid}")
        source = Path(cfg["project_root"]) / rollout["trajectory_path"]
        reader = TrajectoryReader(source)
        steps = list(reader.steps())
        responses = list(reader.token_steps())
        counts = [(s["message_index"], reasoning_count(s, tokenizer)) for s in responses]
        samples, missing = [], []
        for s, step in enumerate(steps):
            anchors, unavailable = reconstruct(step, reader, tokenizer, cfg)
            missing.extend({"step_id": s, **x} for x in unavailable)
            for anchor in anchors:
                tail = [n for index, (n, _) in counts if index > step["message_index"]]
                response = step.get("response") or {}
                if isinstance(response.get("prompt_token_ids"), list) and isinstance(
                    response.get("output_token_ids"), list
                ):
                    from enrichment.online_trace import reasoning_blocks

                    blocks = reasoning_blocks(
                        response["prompt_token_ids"],
                        response["output_token_ids"],
                        tokenizer.encode("<think>", add_special_tokens=False),
                        tokenizer.encode("</think>", add_special_tokens=False),
                    )
                    gi = len(anchor["prefix_token_ids"]) - len(response["prompt_token_ids"]) - 1
                    tail.append(sum(max(0, block["end"] - max(block["start"], gi + 1)) for block in blocks))
                elif anchor["anchor"] == "think_open":
                    tail.append(reasoning_count(step, tokenizer)[0])
                tokens = sum(tail) if all(n is not None for n in tail) else None
                success = rollout["resolved"] is True
                prefix = anchor.pop("prefix_token_ids")
                samples.append(
                    {
                        "sample_id": f"{rid}:{s}:{anchor['anchor']}:{anchor['occurrence']}",
                        "rollout_id": rid,
                        "step_id": s,
                        "message_index": step["message_index"],
                        "partition": selected["tasks"][rollout["task_key"]]["partition"],
                        **anchor,
                        "prefix_token_ids": prefix,
                        "prefix_sha256": hash_json(prefix),
                        "labels": {
                            "remaining_steps": len(steps) - s if success else None,
                            "remaining_reasoning_tokens": tokens if success else None,
                        },
                        "terminal_remaining_steps": len(steps) - s,
                        "action_evidence": {"tool_calls": step["tool_calls"], "content": step["content"]},
                        "success_label_available": success,
                        "token_count_sources": sorted(
                            {
                                src
                                for index, (_, src) in counts
                                if index > step["message_index"]
                                or (
                                    index == step["message_index"]
                                    and (anchor["anchor"] == "think_open" or src == "exact_native_ids")
                                )
                            }
                        ),
                    }
                )
        relative = f"{rid}.samples.json"
        write_json(out / relative, samples)
        files[relative] = sha256(out / relative)
        reports[rid] = {
            "steps": len(steps),
            "anchors": len(samples),
            "unavailable": missing,
            "resolved": rollout["resolved"],
            "source_sha256": rollout["trajectory_sha256"],
        }
    write_json(
        out / "receipt.json",
        {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "input_identity": identity,
            "model_fingerprint": cfg["model"]["fingerprint"],
            "catalog_revision": selected["catalog_revision"],
            "selection": selected,
            "config_hash": cfg["config_hash"],
            "files": files,
            "reports": reports,
        },
    )
    report_dir = artifact_dir(cfg, "enrichment", dataset_id, "reports")
    write_json(report_dir / "quality.json", reports)
    return out

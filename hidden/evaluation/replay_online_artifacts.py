"""Replay immutable online events once, retaining native APC across trajectories."""

import argparse
from pathlib import Path

import numpy as np

from core.config import load_snapshot, resolve
from core.io import hash_json, read_json, sha256, write_json
from hidden.adapters.vllm import replay_group
from hidden.artifacts import artifact_bytes, event_records, read_request
from hidden.backends.vllm.engine import configure_environment, gpu_reservation
from hidden.evaluation.compare_vllm_replay import compare_replay


def finalized_request(source, row):
    finalization_path = Path(source) / "sessions" / row["session_id"] / "finalization.json"
    if not finalization_path.exists():
        return row
    finalization = read_json(finalization_path)
    if row["request_id"] not in finalization["accepted_native_request_ids"]:
        return None
    if sha256(finalization["trajectory_path"]) != finalization["trajectory_sha256"]:
        raise ValueError("Finalized replay source trajectory changed")
    return {**row, **{key: finalization[key] for key in ("rollout_id", "trajectory_sha256")}}


def prepare_reference(source, output, event_index=None):
    source, output = Path(source), Path(output)
    receipts = {}
    events = read_json(event_index) if event_index else [row for _, row in event_records(source)]
    paths = (
        sorted({source / row["ledger_ref"] for row in events})
        if events and all(row.get("ledger_ref") for row in events)
        else [*(source / "sessions").glob("*/requests/*.json"), *(source / "requests").glob("*.json")]
    )
    for path in paths:
        row = read_request(
            source, {"ledger_ref": str(path.relative_to(source)), "ledger_sha256": sha256(path)}
        )
        row = finalized_request(source, row)
        if row is None:
            continue
        if row["request_id"] in receipts:
            raise ValueError("Duplicate native request receipt")
        receipts[row["request_id"]] = (row, sha256(path))
    reference = output / "online-reference"
    reference.mkdir(parents=True, exist_ok=False)
    rows, index = [], []
    for event in events:
        receipt, receipt_hash = receipts[event["request_id"]]
        ids = receipt["prompt_token_ids"] + receipt["output_token_ids"]
        pos = event["absolute_position"]
        if not 0 <= pos < len(ids) - 1 or ids[pos] != event["token_id"]:
            raise ValueError("Online event has no exact processed input-token evidence")
        if hash_json(receipt["prompt_token_ids"]) != event["request_sha256"]:
            raise ValueError("Online request checksum mismatch")
        prefix = ids[: pos + 1]
        item = {
            **event,
            "path": event["sample_id"] + ".npz",
            "prefix_sha256": hash_json(prefix),
            "source_receipt_sha256": receipt_hash,
            "trajectory_sha256": receipt.get("trajectory_sha256")
            or event.get("trajectory_sha256")
            or hash_json(ids),
            "trajectory_evidence_kind": "finalized_trajectory"
            if receipt.get("trajectory_sha256") or event.get("trajectory_sha256")
            else "exact_request_token_receipt",
            "serving_dtype": "float16",
        }
        if "tensor_locator" in item:
            item["source_tensor_locator"] = item.pop("tensor_locator")
        (reference / item["path"]).write_bytes(artifact_bytes(source, event))
        item["prompt_length"] = len(receipt["prompt_token_ids"])
        rows.append(
            {
                "sample_id": item["sample_id"],
                "prefix_token_ids": prefix,
                "prefix_sha256": item["prefix_sha256"],
                "rollout_id": receipt.get("rollout_id") or event.get("rollout_id"),
                "session_id": receipt["session_id"],
                "step_id": receipt.get("step_id", event.get("step_id", 0)),
                "absolute_position": pos,
                "branch_id": event.get("branch_id"),
            }
        )
        index.append(item)
    if not rows:
        raise ValueError("No exact online samples")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate online replay sample IDs")
    write_json(reference / "index.json", index)
    write_json(output / "prefixes.json", rows)
    return reference, rows, index


def replay_config(path, overrides, source_run=None):
    cfg = resolve(path, [*overrides, "hidden.dtype=float16", "hidden.adapter=vllm_eager"])
    if source_run is not None:
        frozen = load_snapshot(cfg["project_root"], source_run)
        if frozen["serving"]["dtype"] != "float16":
            raise ValueError("Source rollout must compute in FP16 for this acceptance")
        # Only offline resource settings come from today's config/overrides.
        # Model, generation, steering and site identity come from the source run.
        offline = cfg["hidden"]
        cfg = {
            **frozen,
            "hidden": offline,
            "source_run_config_hash": frozen["config_hash"],
            "replay_resource_overrides": list(overrides),
        }
        cfg["hidden"]["layers"] = list(frozen["capture"]["layers"])
        if cfg["hidden"]["vllm"]["tensor_parallel_size"] != frozen["serving"]["tensor_parallel_size"]:
            raise ValueError("Replay TP must equal source rollout TP")
        cfg["config_hash"] = hash_json({k: v for k, v in cfg.items() if k != "config_hash"})
    return cfg


def prepare_extras(source, path):
    """Bind historical missing-point files to the same exact request ledgers."""
    extras = []
    for row in read_json(path):
        ledger = finalized_request(source, read_request(source, row))
        if ledger is None:
            raise ValueError("Missing point belongs to an unaccepted source request")
        ids = row["prefix_token_ids"]
        original = ledger["prompt_token_ids"] + ledger["output_token_ids"]
        if not ids or len(ids) >= len(original) or original[: len(ids)] != ids:
            raise ValueError("Missing-point prefix differs from exact request ledger")
        extras.append(
            {
                **row,
                "rollout_id": ledger.get("rollout_id") or row.get("rollout_id"),
                "session_id": ledger["session_id"],
                "step_id": ledger.get("step_id", row.get("step_id", 0)),
                "absolute_position": len(ids) - 1,
                "prompt_length": len(ledger["prompt_token_ids"]),
                "online_status": "missing",
                "compared_to_online": False,
            }
        )
    return extras


def replay_to_disk(adapter, rows, metadata, output, layers, chunk_size, sites):
    """One pass; dependency waves and all cache decisions belong to the adapter."""
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate replay sample IDs")
    if set(metadata) != {row["sample_id"] for row in rows}:
        raise ValueError("Replay metadata differs from submitted samples")
    ordered = sorted(
        rows,
        key=lambda row: (
            replay_group(row),
            row.get("step_id", 0),
            row.get("absolute_position", len(row["prefix_token_ids"]) - 1),
            row["sample_id"],
        ),
    )
    output = Path(output)
    replay_dir = output / "trajectory-apc"
    replay_dir.mkdir()
    indices = {"trajectory-apc": [], "offline-only": []}
    metrics_start = len(adapter.metrics)
    for start in range(0, len(ordered), adapter.batch_size):
        batch = ordered[start : start + adapter.batch_size]
        values = adapter.capture_batch(batch, layers, chunk_size)
        for row in batch:
            sid = row["sample_id"]
            item = metadata[sid]
            kind = "offline-only" if item.get("compared_to_online") is False else "trajectory-apc"
            directory = output / kind
            directory.mkdir(exist_ok=True)
            path = directory / (sid + ".npz")
            np.savez_compressed(path, **{f"layer_{site}": values[sid][i] for i, site in enumerate(sites)})
            indices[kind].append({**item, "path": path.name, "sha256": sha256(path)})
        print(f"replayed {min(start + adapter.batch_size, len(ordered))}/{len(ordered)} events", flush=True)
    for kind, index in indices.items():
        if index:
            write_json(output / kind / "index.json", index)
    return adapter.metrics[metrics_start:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--online-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--event-index", type=Path)
    p.add_argument("--model-family", choices=["dense", "moe"], default="dense")
    p.add_argument("--source-run", help="Use immutable source model/generation/steering/site identity")
    p.add_argument(
        "--extra-prefixes",
        type=Path,
        help="Extract missing online points separately; never count them as compared",
    )
    args = p.parse_args()
    cfg = replay_config(args.config, args.set, args.source_run)
    configure_environment(cfg)
    args.output.mkdir(parents=True, exist_ok=False)
    reference, rows, index = prepare_reference(args.online_dir, args.output, args.event_index)
    extras = prepare_extras(args.online_dir, args.extra_prefixes) if args.extra_prefixes else []
    all_rows = rows + extras
    metadata = {row["sample_id"]: row for row in index}
    metadata.update(
        {row["sample_id"]: {k: v for k, v in row.items() if k != "prefix_token_ids"} for row in extras}
    )
    report = {
        "schema_version": 2,
        "replay_policy": "trajectory_apc",
        "replay_source_sha256": sha256(__file__),
        "passed": False,
        "source": str(args.online_dir),
        "samples": len(rows),
        "dtype": "float16",
        "tp": cfg["hidden"]["vllm"]["tensor_parallel_size"],
        "model_fingerprint": cfg["model"]["fingerprint"],
        "replayed_samples": len(all_rows),
        "offline_only_missing_points": len(extras),
        "source_run_config_hash": cfg.get("source_run_config_hash"),
        "shared_gpus": cfg["hidden"]["vllm"].get("allow_shared_gpus", False),
    }
    options = cfg["hidden"]["vllm"]
    with gpu_reservation(
        cfg["project_root"],
        options["gpus"],
        allow_shared=options.get("allow_shared_gpus", False),
        min_free_bytes=options.get("min_free_gpu_bytes", 0),
    ):
        from hidden.adapters.vllm import load_adapter
        from hidden.backends.vllm.capture import native_layers

        adapter = load_adapter(cfg, steering_replay=cfg["steering"]["enabled"])
        if cfg["steering"]["enabled"]:
            from vllm.model_hooks.steering.api import ApplySpec, SteeringSpec, VectorSpec
            from vllm.model_hooks.steering.payloads import DirectionVector

            spec_path = Path(cfg["steering"]["payload"])
            spec = read_json(spec_path)
            if spec["model_fingerprint"] != cfg["model"]["fingerprint"]:
                raise ValueError("Steering replay checkpoint mismatch")
            payload = DirectionVector({spec["layer"]: np.load(spec_path.parent / spec["vector"])})
            lengths = {
                tuple(row["prefix_token_ids"]): metadata[row["sample_id"]]["prompt_length"]
                for row in all_rows
            }
            native_engine = adapter.llm

            class ReplaySteeringEngine:
                def __getattr__(self, name):
                    return getattr(native_engine, name)

                def generate(self, prompts, params, **kwargs):
                    choices = []
                    for prompt in prompts:
                        ids = prompt["prompt_token_ids"]
                        choices.append(
                            SteeringSpec(
                                vectors=[
                                    VectorSpec(
                                        data=payload,
                                        layers=[spec["layer"]],
                                        scale=spec["scale"],
                                        apply=ApplySpec(prompt_window=(lengths[tuple(ids)], len(ids))),
                                    )
                                ]
                            )
                        )
                    return native_engine.generate(prompts, params, steering=choices, **kwargs)

            adapter.llm = ReplaySteeringEngine()
            report["steering_replay"] = {
                "spec_sha256": sha256(spec_path),
                "phase_mapping": "generation to exact prompt suffix",
            }
        sites = native_layers(cfg["hidden"]["layers"], adapter.layer_count)
        report["runtime"] = adapter.metadata
        try:
            windows = replay_to_disk(
                adapter,
                all_rows,
                metadata,
                args.output,
                cfg["hidden"]["layers"],
                cfg["hidden"]["chunk_size"],
                sites,
            )
            report["windows"] = windows
            prompt_tokens = sum(w["prompt_tokens"] for w in windows)
            cached_tokens = sum(w["cached_tokens"] for w in windows)
            report["cache"] = {
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "uncached_prompt_tokens": prompt_tokens - cached_tokens,
                "hit_rate": cached_tokens / prompt_tokens if prompt_tokens else 0,
                "capture_seconds": sum(w["seconds"] for w in windows),
            }
            result = compare_replay(reference, args.output / "trajectory-apc", model_family=args.model_family)
            write_json(args.output / "trajectory-apc/comparison.json", result)
            report["comparison"] = {k: v for k, v in result.items() if k != "metrics"}
            report["passed"] = result["passed"]
        finally:
            write_json(args.output / "report.json", report)
            adapter.close()
    print(
        {
            "passed": report["passed"],
            "comparison": {key: report["comparison"][key] for key in ["worst_cosine", "worst_relative_l2"]},
            "cache": report["cache"],
        },
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

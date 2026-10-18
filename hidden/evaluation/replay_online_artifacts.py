"""Replay immutable online events through the existing isolated FP16 adapter."""

import argparse
from pathlib import Path

import numpy as np

from core.config import load_snapshot, resolve
from core.io import hash_json, read_json, sha256, write_json
from hidden.artifacts import artifact_bytes, event_records, read_request
from hidden.backends.vllm.engine import configure_environment, gpu_reservation
from hidden.evaluation.compare_vllm_replay import compare_replay


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
        finalization_path = source / "sessions" / row["session_id"] / "finalization.json"
        if finalization_path.exists():
            finalization = read_json(finalization_path)
            if row["request_id"] not in finalization["accepted_native_request_ids"]:
                continue
            if sha256(finalization["trajectory_path"]) != finalization["trajectory_sha256"]:
                raise ValueError("Finalized replay source trajectory changed")
            row.update({key: finalization[key] for key in ("rollout_id", "trajectory_sha256")})
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
        if ids[pos] != event["token_id"] or pos >= len(ids) - 1:
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
            }
        )
        index.append(item)
    if not rows:
        raise ValueError("No exact online samples")
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
    phases = ["cold", "warm", "reordered"]
    for phase in phases:
        (args.output / phase).mkdir()
    report = {
        "passed": False,
        "source": str(args.online_dir),
        "samples": len(rows),
        "dtype": "float16",
        "tp": cfg["hidden"]["vllm"]["tensor_parallel_size"],
        "model_fingerprint": cfg["model"]["fingerprint"],
        "phases": {},
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
            prompt_lengths = {row["sample_id"]: row["prompt_length"] for row in index}
            lengths = {tuple(row["prefix_token_ids"]): prompt_lengths[row["sample_id"]] for row in rows}
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
        receipts = {phase: [] for phase in phases}
        windows = {phase: [] for phase in phases}
        try:
            batch_size = min(adapter.batch_size, cfg["hidden"]["vllm"]["max_num_seqs"])
            metadata = {row["sample_id"]: row for row in index}
            for start in range(0, len(rows), batch_size):
                window = rows[start : start + batch_size]
                if adapter.llm.reset_prefix_cache() is False:
                    raise RuntimeError("Native prefix cache reset failed")
                for phase in phases:
                    actual = window[::-1] if phase == "reordered" else window
                    values = adapter.capture_batch(
                        actual, cfg["hidden"]["layers"], cfg["hidden"]["chunk_size"]
                    )
                    windows[phase].append(adapter.metrics[-1])
                    for row in actual:
                        sid = row["sample_id"]
                        path = args.output / phase / (sid + ".npz")
                        np.savez_compressed(
                            path, **{f"layer_{site}": values[sid][i] for i, site in enumerate(sites)}
                        )
                        receipts[phase].append({**metadata[sid], "path": path.name, "sha256": sha256(path)})
                print(f"replayed {min(start + batch_size, len(rows))}/{len(rows)} events", flush=True)
            for phase in phases:
                write_json(args.output / phase / "index.json", receipts[phase])
                result = compare_replay(reference, args.output / phase, model_family=args.model_family)
                result["windows"] = windows[phase]
                write_json(args.output / phase / "comparison.json", result)
                report["phases"][phase] = {k: v for k, v in result.items() if k != "metrics"}
            report["passed"] = all(result["passed"] for result in report["phases"].values())
            if args.extra_prefixes:
                extras = read_json(args.extra_prefixes)
                extra_dir = args.output / "offline-only"
                extra_dir.mkdir()
                extra_index = []
                for start in range(0, len(extras), batch_size):
                    batch = extras[start : start + batch_size]
                    values = adapter.capture_batch(
                        batch, cfg["hidden"]["layers"], cfg["hidden"]["chunk_size"]
                    )
                    for row in batch:
                        sid = row["sample_id"]
                        path = extra_dir / (sid + ".npz")
                        np.savez_compressed(
                            path, **{f"layer_{site}": values[sid][i] for i, site in enumerate(sites)}
                        )
                        extra_index.append(
                            {
                                **{k: v for k, v in row.items() if k != "prefix_token_ids"},
                                "path": path.name,
                                "sha256": sha256(path),
                                "online_status": "missing",
                                "compared_to_online": False,
                            }
                        )
                write_json(extra_dir / "index.json", extra_index)
                report["offline_only_missing_points"] = len(extra_index)
        finally:
            write_json(args.output / "report.json", report)
            adapter.close()
    print(
        {
            "passed": report["passed"],
            "phases": {
                key: {k: v for k, v in value.items() if k in {"worst_cosine", "worst_relative_l2", "layers"}}
                for key, value in report["phases"].items()
            },
        },
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

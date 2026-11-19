"""Standard HF safetensors export with bounded CPU memory and atomic publication."""
from pathlib import Path
import json
import os
import shutil
import struct

from core.io import hash_json, read_json, write_json
from RL.full_artifacts import verify_checkpoint


def _tensor_file(path, name, shape, dtype, chunks):
    import torch
    code = {"float32": "F32", "float16": "F16", "bfloat16": "BF16"}[dtype]
    count = __import__("math").prod(shape)
    size = count * (4 if dtype == "float32" else 2)
    header = json.dumps({"__metadata__": {"format": "pt"}, name: {
        "dtype": code, "shape": shape, "data_offsets": [0, size]}}, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    written = 0
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(header)))
        stream.write(header)
        for chunk in chunks:
            value = chunk.to(getattr(torch, dtype)).contiguous().view(torch.uint8).numpy().tobytes()
            stream.write(value)
            written += len(value)
        if written != size:
            raise ValueError("HF export tensor size mismatch: " + name)
        stream.flush()
        os.fsync(stream.fileno())
    return size


def export(root_config, checkpoint, base_model, output, dtype="bfloat16", dry_run=False):
    root = Path(root_config["project_root"])
    checkpoint = (root / checkpoint).resolve()
    base_model = (root / base_model).resolve()
    output = (root / output).resolve()
    metadata = verify_checkpoint(checkpoint, deep=not dry_run)
    from core.model_identity import describe
    base = describe(base_model)
    if base["fingerprint"] != metadata["contract"]["base_fingerprint"]:
        raise ValueError("Full export base/tokenizer/vision fingerprint mismatch")
    if output.exists():
        raise FileExistsError("Full export output already exists")
    for parent in (output.parent, *output.parents):
        if (parent / "receipt.json").exists() or (parent / "attempt.json").exists():
            raise ValueError("Full export must be outside sealed source artifacts")
    result = {"checkpoint": str(checkpoint), "output": str(output), "dtype": dtype,
              "mode": "dry-run" if dry_run else "export", "base_fingerprint": base["fingerprint"]}
    if dry_run:
        return result
    from safetensors import safe_open
    from RL.full_policy import weight_files
    from RL.full_checkpoint import export_parameter
    from RL.artifacts import commit_receipt
    inventory = read_json(checkpoint / "parameters.json")
    rows = {name: row for row in inventory["parameters"] for name in row["hf_names"]}
    files = weight_files(base_model)
    names = sorted(set(files) | set(rows))
    # Include both tied aliases in the HF export; Transformers restores tying.
    approximate = sum(r["numel"] for r in inventory["parameters"]) * (4 if dtype == "float32" else 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < approximate + 4 * 2**30:
        raise OSError("Insufficient free disk for full HF export")
    stage = output.with_name("." + output.name + ".staging")
    stage.mkdir()
    index, total = {}, 0
    try:
        for i, name in enumerate(names):
            filename = f"model-{i+1:05d}-of-{len(names):05d}.safetensors"
            if name in rows:
                row = rows[name]
                chunks = export_parameter(checkpoint, row)
                shape = row["shape"]
            else:
                with safe_open(files[name], framework="pt", device="cpu") as source:
                    shape = source.get_slice(name).get_shape()
                def source_chunks(name=name, shape=shape):
                    import math
                    with safe_open(files[name], framework="pt", device="cpu") as source:
                        view = source.get_slice(name)
                        count = max(1, 16 * 2**20 // max(4, math.prod(shape[1:]) * 4))
                        for start in range(0, shape[0], count):
                            yield view[start:start+count].reshape(-1)
                chunks = source_chunks()
            total += _tensor_file(stage / filename, name, shape, dtype, chunks)
            index[name] = filename
        for source in base_model.iterdir():
            if source.is_file() and source.suffix in {".json", ".txt", ".model", ".jinja"} and not source.name.endswith("index.json"):
                shutil.copyfile(source, stage / source.name)
        config = read_json(stage / "config.json")
        config["dtype"] = dtype
        config["text_config"]["dtype"] = dtype
        write_json(stage / "config.json", config)
        write_json(stage / "model.safetensors.index.json", {"metadata": {"total_size": total}, "weight_map": index})
        write_json(stage / "full-policy-export.json", {**result, "contract_sha256": hash_json(metadata["contract"]),
            "gdn_execution": metadata["contract"].get("gdn_execution"),
            "initial_policy": metadata["contract"]["initial_policy"], "global_step": metadata["state"]["global_step"]})
        commit_receipt(stage, "rl-full-hf-export-v1", source_checkpoint=str(checkpoint), total_weight_bytes=total)
        os.rename(stage, output)
    except BaseException:
        # Retain failed staging evidence; never replace a checkpoint or published model.
        raise
    return {**result, "weight_bytes": total, "tensor_shards": len(index)}

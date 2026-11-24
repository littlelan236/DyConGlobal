"""Bounded local-shard I/O; all-rank staging and receipt-last publication."""
from pathlib import Path
import os
import shutil
import json
import time

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from core.io import hash_json, read_json, write_json
from RL.artifacts import commit_receipt
from RL.full_policy import local_tensor
from RL.training import capture_rng, restore_rng

CHUNK_ELEMENTS = 16 * 2**20  # 64 MiB per FP32 I/O staging tensor.


def write_tensor(directory, value, counter):
    value = local_tensor(value).detach()
    flat = value.reshape(-1)
    parts = []
    for offset in range(0, flat.numel(), CHUNK_ELEMENTS):
        name = f"tensor-{counter[0]:06d}.safetensors"
        counter[0] += 1
        piece = flat[offset:offset + CHUNK_ELEMENTS].cpu().contiguous()
        save_file({"value": piece}, str(directory / name))
        parts.append({"file": name, "offset": offset, "numel": piece.numel()})
        del piece
    return {"shape": list(value.shape), "dtype": str(value.dtype), "parts": parts}


def read_tensor(directory, description, target):
    if list(target.shape) != description["shape"] or str(target.dtype) != description["dtype"]:
        raise ValueError("Full tensor shard shape/dtype mismatch")
    flat = target.reshape(-1)
    offset = 0
    for part in description["parts"]:
        if part["offset"] != offset or part["numel"] < 1:
            raise ValueError("Full tensor shard has gaps or overlaps")
        value = load_file(str(directory / part["file"]), device="cpu")["value"]
        if value.numel() != part["numel"] or value.dtype != target.dtype:
            raise ValueError("Full tensor shard payload mismatch")
        flat[offset:offset + value.numel()].copy_(value.to(target.device))
        offset += value.numel()
    if offset != flat.numel():
        raise ValueError("Full tensor shard is incomplete")


def write_moment(directory, optimizer, parameter, key, counter):
    if not hasattr(optimizer, "iter_local_state_chunks"):
        return write_tensor(directory, optimizer.state[parameter][key], counter)
    parts, covered = [], 0
    for start, tensor in optimizer.iter_local_state_chunks(parameter, key):
        if start != covered:
            raise ValueError("Adam moment serialization has gaps or overlaps")
        for offset in range(0, tensor.numel(), CHUNK_ELEMENTS):
            name = f"tensor-{counter[0]:06d}.safetensors"
            counter[0] += 1
            piece = tensor[offset:offset + CHUNK_ELEMENTS].detach().cpu().contiguous()
            save_file({"value": piece}, str(directory / name))
            parts.append({"file": name, "offset": covered, "numel": piece.numel()})
            covered += piece.numel()
            del piece
    if covered != local_tensor(parameter).numel():
        raise ValueError("Incomplete serialized Adam moment")
    return {"shape": list(local_tensor(parameter).shape), "dtype": "torch.float32", "parts": parts}


def read_moment(directory, description, optimizer, parameter, key):
    if description["shape"] != list(local_tensor(parameter).shape) or description["dtype"] != "torch.float32":
        raise ValueError("Streamed moment shape/dtype mismatch")
    targets = iter(optimizer.iter_local_state_chunks(parameter, key))
    current = next(targets, None)
    covered = 0
    for part in description["parts"]:
        if part["offset"] != covered or part["numel"] < 1:
            raise ValueError("Streamed moment has gaps or overlaps")
        value = load_file(str(directory / part["file"]), device="cpu")["value"]
        if value.dtype != torch.float32 or value.ndim != 1 or value.numel() != part["numel"]:
            raise ValueError("Invalid streamed moment payload")
        source = 0
        while source < value.numel():
            if current is None or not current[0] <= covered < current[0] + current[1].numel():
                raise ValueError("Streamed restore coverage mismatch")
            start, target = current
            offset = covered - start
            n = min(value.numel() - source, target.numel() - offset)
            target[offset:offset+n].copy_(value[source:source+n])
            covered += n
            source += n
            if offset + n == target.numel():
                current = next(targets, None)
        del value
    if current is not None or covered != local_tensor(parameter).numel():
        raise ValueError("Incomplete restored moment")


def save(directory, policy, optimizer, scheduler, state, contract, inventory, context):
    started = time.monotonic()
    rank, world = context.rank, context.world_size
    step = state["global_step"]
    destination = Path(directory) / "checkpoints" / f"step-{step:06d}"
    stage = destination.with_name("." + destination.name + ".staging")
    parameters = dict(policy.named_parameters())
    local_bytes = sum(local_tensor(p).numel() * 12 for p in parameters.values())
    sizes = context.gather(local_bytes)
    if context.primary:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or stage.exists():
            raise FileExistsError("Full checkpoint destination/staging already exists")
        if shutil.disk_usage(destination.parent).free < sum(sizes) + 2 * 2**30:
            raise OSError("Insufficient disk space for full checkpoint plus 2 GiB reserve")
        stage.mkdir()
    dist.barrier()
    folder = stage / "ranks" / str(rank)
    folder.mkdir(parents=True)
    counter, tensors = [0], {}
    for name, p in parameters.items():
        opt = optimizer.state[p]
        streamed = hasattr(optimizer, "iter_local_state_chunks")
        if set(opt) != ({"step", "tiles"} if streamed else {"step", "exp_avg", "exp_avg_sq"}):
            raise ValueError("Unexpected full AdamW state layout")
        if not streamed and any(local_tensor(opt[k]).dtype != torch.float32 for k in ("exp_avg", "exp_avg_sq")):
            raise ValueError("Full Adam moments must be FP32")
        tensors[name] = {"weight": write_tensor(folder, p, counter),
                         "step": float(opt["step"]),
                         "optimizer": {k: write_moment(folder, optimizer, p, k, counter)
                                       for k in ("exp_avg", "exp_avg_sq")}}
    write_json(folder / "manifest.json", {"schema_version": 1, "rank": rank, "world_size": world,
                                         "parameters": tensors})
    if "gdn_execution" in contract:
        from RL.gdn_backend import active_binding
        from RL.full_backend import evidence
        binding = active_binding()
        if binding is None or binding.execution_contract() != contract["gdn_execution"]:
            raise ValueError("Full checkpoint live GDN identity changed")
        write_json(folder / "gdn.json", evidence(binding))
    torch.save({"rng": capture_rng(), "scheduler": scheduler.state_dict(),
                "optimizer_groups": [{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups]},
               folder / "runtime.pt")
    dist.barrier()
    if context.primary:
        shutil.copytree(Path(directory) / "anchor", stage / "anchor")
        write_json(stage / "parameters.json", inventory)
        write_json(stage / "state.json", {"schema_version": 1, "contract": contract, "state": state})
        receipt = commit_receipt(stage, "rl-full-checkpoint-v1", contract_sha256=hash_json(contract), global_step=step)
        os.rename(stage, destination)
        fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        with (Path(directory) / "checkpoint-metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"global_step": step, "checkpoint_seconds": time.monotonic() - started,
                "file_count": len(receipt["files"]),
                "checkpoint_bytes": sum(row["bytes"] for row in receipt["files"].values())}) + "\n")
    dist.barrier()
    return str(destination)


def restore(path, policy, optimizer, scheduler, context, *, training=True, config=None):
    folder = Path(path) / "ranks" / str(context.rank)
    manifest = read_json(folder / "manifest.json")
    if manifest["world_size"] != context.world_size:
        raise ValueError("Full restore requires the exact world size")
    parameters = dict(policy.named_parameters())
    if set(manifest["parameters"]) != set(parameters):
        raise ValueError("Full restore language parameter inventory mismatch")
    streamed = training and hasattr(optimizer, "iter_local_state_chunks")
    if streamed:
        from RL.optimizer_offload import BACKEND, LAYOUT, coordinated
        recorded = read_json(Path(path) / "state.json")["contract"].get("optimizer_backend", {})
        if recorded.get("backend") != BACKEND or recorded.get("layout") != LAYOUT:
            from RL.optimizer_migration import validate
            if config is None or not validate(config, read_json(Path(path) / "state.json")["contract"]):
                raise ValueError("Full optimizer backend transition requires explicit migration")
        coordinated(context, lambda: optimizer.prepare(extra_bytes=sum(local_tensor(p).numel() * 4 for p in parameters.values())))
    from torch.distributed.tensor import DTensor
    with torch.no_grad():
        for name, p in parameters.items():
            description = manifest["parameters"][name]
            read_tensor(folder, description["weight"], local_tensor(p))
            if streamed:
                if set(description["optimizer"]) != {"exp_avg", "exp_avg_sq"}:
                    raise ValueError("Incomplete restored Adam moments")
                for key, value in description["optimizer"].items():
                    read_moment(folder, value, optimizer, p, key)
                optimizer.state[p]["step"].fill_(description["step"])
                for tile in optimizer.state[p]["tiles"]:
                    tile.version = int(description["step"])
            elif training:
                state = {"step": torch.tensor(description["step"], dtype=torch.float32)}
                for key, value in description["optimizer"].items():
                    local = torch.empty_like(local_tensor(p), dtype=torch.float32)
                    read_tensor(folder, value, local)
                    state[key] = DTensor.from_local(local, p.device_mesh, p.placements,
                        run_check=False, shape=p.shape, stride=p.stride())
                optimizer.state[p] = state
    if training:
        runtime = torch.load(folder / "runtime.pt", map_location="cpu", weights_only=True)
        for target, values in zip(optimizer.param_groups, runtime["optimizer_groups"], strict=True):
            target.update(values)
        scheduler.load_state_dict(runtime["scheduler"])
        restore_rng(runtime["rng"])


def export_parameter(path, row):
    """Yield flattened FP32 parameter chunks in global Shard(0) order on CPU."""
    metadata = read_json(Path(path) / "state.json")
    for rank in range(metadata["contract"]["world_size"]):
        folder = Path(path) / "ranks" / str(rank)
        entry = read_json(folder / "manifest.json")["parameters"][row["name"]]["weight"]
        for part in entry["parts"]:
            yield load_file(str(folder / part["file"]), device="cpu")["value"]

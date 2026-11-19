"""Full-policy cross-backend gradient diagnostic through the common FSDP2 service.

No optimizer, clipping, model update, topology change, truncation or CPU offload.
Reference FP32 gradient shards are temporary; only complete summaries are sealed.
"""
from contextlib import ExitStack
import math
from pathlib import Path
import shutil
import time

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from core.io import hash_json, read_json, sha256, write_json


class Comparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    training_config: str
    group_id: str
    output: str
    modes: list[Literal["fixed_cotangent", "shared_reference_grpo"]] = Field(
        default_factory=lambda: ["fixed_cotangent", "shared_reference_grpo"])


def combine(rows):
    """Combine disjoint shard statistics, without averaging relative errors."""
    return {**{k: math.fsum(r[k] for r in rows) for k in ("reference_sq", "actual_sq", "difference_sq", "dot")},
            "max_abs": max((r["max_abs"] for r in rows), default=0.),
            "elements": sum(r["elements"] for r in rows)}


def describe(row):
    ref, actual, delta = (math.sqrt(max(0., row[k])) for k in ("reference_sq", "actual_sq", "difference_sq"))
    return {**row, "reference_l2": ref, "actual_l2": actual, "difference_l2": delta,
            "relative_l2": delta / ref if ref else (0. if delta == 0 else None),
            "relative_l2_undefined": ref == 0 and delta != 0,
            "cosine": max(-1., min(1., row["dot"] / (ref * actual))) if ref * actual else None}


def statistics(actual, reference, chunk_elements=1024 * 1024):
    """FP64 reductions over every value, with bounded conversion workspace."""
    import torch
    if actual.shape != reference.shape:
        raise ValueError("Gradient shard shapes differ")
    acc = torch.zeros(5, dtype=torch.float64, device=actual.device)
    count = actual.numel()
    a, b = actual.reshape(-1), reference.reshape(-1)
    for start in range(0, count, chunk_elements):
        x = a[start:start + chunk_elements].detach().double()
        y = b[start:start + chunk_elements].detach().to(actual.device, torch.float64)
        if not bool(torch.isfinite(x).all() & torch.isfinite(y).all()):
            raise ValueError("Nonfinite full gradient")
        d = x - y
        acc[:4] += torch.stack((y.square().sum(), x.square().sum(), d.square().sum(), (x * y).sum()))
        if d.numel():
            acc[4] = torch.maximum(acc[4], d.abs().max())
    values = acc.tolist()
    return dict(zip(("reference_sq", "actual_sq", "difference_sq", "dot", "max_abs"), values), elements=count)


def _read(config_file, root):
    import yaml
    from RL.config import load, is_full, schedule_contract
    source = (root / config_file).resolve()
    settings = Comparison.model_validate(yaml.safe_load(source.read_text()))
    if not settings.modes or len(settings.modes) != len(set(settings.modes)):
        raise ValueError("Comparison modes must be nonempty and unique")
    config, training_source = load(settings.training_config, root)
    if not is_full(config) or config.model.dtype != "bfloat16" or config.run.start.mode != "fresh":
        raise ValueError("Gradient comparison requires fresh full-policy BF16 training settings")
    if config.prepared_data is None:
        raise ValueError("Gradient comparison requires an explicit sealed prepared binding")
    from RL.grpo.prepared import PreparedReader, resolve, validate_config
    reader = PreparedReader(resolve(config.prepared_data.model_dump()))
    validate_config(reader, config)
    if settings.group_id not in reader.ids:
        raise ValueError("Comparison parent is missing from sealed prepared data")
    output = (root / settings.output).resolve()
    if not output.is_relative_to(root / "temp") or output == root / "temp":
        raise ValueError("Diagnostic output must be a new directory below project temp/")
    if output.exists():
        raise FileExistsError("Comparison output already exists")
    # Never mutate a sealed source by placing output inside it.
    if any((p / "receipt.json").exists() for p in output.parents):
        raise ValueError("Comparison output is inside a sealed artifact")
    plan = {"settings": settings.model_dump(), "training_schedule": schedule_contract(config),
            "training_config_sha256": sha256(training_source), "comparison_config_sha256": sha256(source),
            "prepared": config.prepared_data.model_dump(), "world_size": len(config.runtime.gpus or [config.runtime.gpu]),
            "scope": "all language parameters; frozen vision excluded", "updates": 0,
            "reference_backend": "reference_pytorch", "actual_backend": "fla_chunk",
            "acceptance": "finite complete gradients; errors are measured, no equivalence threshold"}
    plan["implementation_sha256"] = {name: sha256(root / name) for name in (
        "RL/full_gradient_compare.py", "RL/full_policy.py", "RL/full_training.py", "RL/gdn_backend.py",
        "RL/grpo/objective.py", "RL/grpo/prepared.py", "RL/probe.py", "RL/parallel.py")}
    return settings, config, reader, output, plan


def run(root_config, config_file, *, dry_run=False):
    from core.locking import file_lock
    from RL.full_artifacts import base_identity
    from RL.parallel import launch, devices
    from RL.artifacts import commit_receipt, verify_receipt
    root = Path(root_config["project_root"]).resolve()
    settings, config, reader, output, plan = _read(config_file, root)
    raw = reader.raw(settings.group_id)
    lengths = [len(m["prompt_token_ids"]) + len(m["completion_token_ids"]) for m in raw["members"]]
    plan["input"] = {"group_id": settings.group_id, "branches": len(lengths), "max_sequence_length": max(lengths),
                     "target_tokens": sum(len(m["completion_token_ids"]) for m in raw["members"]),
                     "raw_parent_sha256": hash_json(raw), "synthetic": reader.proof["synthetic"]}
    if dry_run:
        return {"status": "validated", "output": str(output), "plan": plan}
    started = time.monotonic()
    with ExitStack() as locks:
        for gpu in sorted(devices(config)):
            locks.enter_context(file_lock(root / "state/artifact-locks/lora-gpu" / f"{gpu}.lock", nonblocking=True))
        output.parent.mkdir(parents=True, exist_ok=True)
        # Atomic directory creation refuses competing/repeated invocations.
        output.mkdir(exist_ok=False)
        write_json(output / "plan.json", plan)
        write_json(output / "training-config.json", config.model_dump(mode="json"))
        base = base_identity(config, root)
        write_json(output / "base-identity.json", base)
        if shutil.disk_usage(output).free < 2 * 4 * 10_000_000_000 + 4 * 2**30:
            raise OSError("Need space for two temporary complete FP32 gradient sets plus reserve")
        try:
            results = {}
            for backend in ("reference_pytorch", "fla_chunk"):
                work = output / backend
                work.mkdir()
                phase_config = config.model_copy(deep=True)
                phase_config.model.gdn_backend = backend
                phase_config.preflight.check_resources = True
                operation = {"operation": "full_gradient_compare", "output": str(output),
                             "group_id": settings.group_id, "modes": settings.modes,
                             "plan_sha256": hash_json(plan)}
                # Fresh spawn workers isolate backend globals and CUDA contexts, including world 1.
                result = launch(phase_config, root, work, base, None, operation)
                write_json(work / "result.json", result)
                commit_receipt(work, "rl-full-gradient-phase-v1", backend=backend)
                if backend == "reference_pytorch":
                    verify_receipt(work, "rl-full-gradient-phase-v1")
                results[backend] = result
            report = {"schema_version": 1, "status": "completed", "plan_sha256": hash_json(plan),
                      "plan": plan, "base_fingerprint": base["fingerprint"], "results": results,
                      "seconds_before_cleanup": time.monotonic() - started}
            write_json(output / "comparison.json", report)
            # Only this newly created diagnostic's reference tensor payloads are removed.
            removed = 0
            for path in (output / "reference_pytorch").glob("gradients-*"):
                resolved = path.resolve()
                if not resolved.is_relative_to(output / "reference_pytorch") or path.is_symlink():
                    raise ValueError("Gradient cleanup escapes diagnostic output")
                removed += sum(f.stat().st_size for f in path.rglob("*.safetensors"))
                shutil.rmtree(path)
            write_json(output / "gradient-cleanup.json", {"deleted_bytes": removed,
                "note": "reference phase receipt describes now-deleted temporary gradients; comparison summaries remain",
                "seconds_total": time.monotonic() - started})
            commit_receipt(output, "rl-full-gradient-comparison-v1", status="completed")
            return {"status": "completed", "output": str(output), "modes": settings.modes,
                    "gradient_errors": {m: results['fla_chunk']['modes'][m]['all_parameters'] for m in settings.modes}}
        except BaseException:
            write_json(output / "failed.json", {"status": "failed", "note": "No comparison receipt committed"})
            raise


def _gradients(policy, inventory):
    from RL.full_policy import local_tensor
    parameters = dict(policy.named_parameters())
    if set(parameters) != {r["name"] for r in inventory["parameters"]}:
        raise ValueError("Parameter coverage changed")
    for row in inventory["parameters"]:
        p = parameters[row["name"]]
        if p.grad is None:
            raise ValueError("Missing full gradient: " + row["name"])
        g = local_tensor(p.grad).detach()
        if str(g.dtype) != "torch.float32":
            raise ValueError("Full reduced gradient must be FP32")
        yield row["name"], g


def _store_gradients(policy, inventory, path):
    from RL.full_checkpoint import write_tensor
    path.mkdir(parents=True)
    counter, manifest, stats = [0], {}, {}
    for name, grad in _gradients(policy, inventory):
        stats[name] = statistics(grad, grad)
        manifest[name] = write_tensor(path, grad, counter)
    write_json(path.parent / (path.name + "-manifest.json"), manifest)
    return stats


def _compare_gradients(policy, inventory, path):
    from safetensors import safe_open
    manifest = read_json(path.parent / (path.name + "-manifest.json"))
    result = {}
    for name, grad in _gradients(policy, inventory):
        entry = manifest[name]
        if entry["shape"] != list(grad.shape) or entry["dtype"] != str(grad.dtype):
            raise ValueError("Reference gradient shard identity differs")
        offset, parts = 0, []
        flat = grad.reshape(-1)
        for item in entry["parts"]:
            if offset != item["offset"]:
                raise ValueError("Reference gradient has gaps/overlaps")
            with safe_open(path / item["file"], framework="pt", device="cpu") as source:
                value = source.get_tensor("value")
                if value.numel() != item["numel"]:
                    raise ValueError("Reference gradient part size differs")
                parts.append(statistics(flat[offset:offset + value.numel()], value))
                offset += value.numel()
        if offset != grad.numel():
            raise ValueError("Reference gradient is incomplete")
        result[name] = combine(parts)
    if set(result) != set(manifest):
        raise ValueError("Reference gradient parameter coverage differs")
    return result


def _category(name):
    for token in ("embed_tokens", "lm_head", "A_log", "dt_bias", "conv1d", "norm"):
        if token in name:
            return token
    return "other_decoder"


def evaluate(config, root, directory, base, attempt, context):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh
    from RL.full_policy import load, shard_digest, microsteps
    from RL.full_backend import bind, evidence
    from RL.full_training import _group, window_update
    from RL.grpo.prepared import PreparedReader, resolve
    from RL.probe import _configure_numerics
    from RL.training import seed_everything
    from RL.memory import attention_context
    from RL.resources import host_memory
    from RL.grpo.objective import grpo_objective
    started = time.monotonic()
    output = Path(attempt["output"])
    reference_dir = output / "reference_pytorch"
    baseline = config.model.gdn_backend == "reference_pytorch"
    plan = read_json(output / "plan.json")
    if hash_json(plan) != attempt["plan_sha256"]:
        raise ValueError("Comparison plan changed")
    seed_everything(config.run.seed)
    _configure_numerics(config)
    binding = bind(config.model, context)
    mesh = DeviceMesh.from_group(context.data_group, "cuda", mesh_dim_names=("data",))
    policy, inventory = load(config, mesh, context.rank, context.world_size)
    digests = context.gather(shard_digest(policy))
    reader = PreparedReader(resolve(plan["prepared"]))
    if hash_json(reader.raw(attempt["group_id"])) != plan["input"]["raw_parent_sha256"]:
        raise ValueError("Diagnostic parent changed")
    group = _group(reader, attempt["group_id"])
    if group.group_size != plan["input"]["branches"]:
        raise ValueError("Diagnostic branch coverage changed")
    anchor_path = reference_dir / "reference-logprobs.json"
    if baseline:
        branches = []
        policy.eval()
        with torch.no_grad(), attention_context(config.training.memory.attention_policy):
            for b, bid in enumerate(group.branch_ids):
                values = policy(group.branch_batch(b, "cuda:0"), config.training.memory.loss_chunk_tokens or 256)
                branches.append({"branch_id": bid, "target_logprobs": values.float().cpu().tolist()})
        anchors = context.gather(branches)
        if any(peer != branches for peer in anchors):
            raise ValueError("Reference target logprobs disagree across ranks")
        if context.primary:
            write_json(anchor_path, {"branches": branches, "parameter_shards": digests,
                                    "inventory_sha256": hash_json(inventory)})
        dist.barrier()
    anchor = read_json(anchor_path)
    if digests != anchor["parameter_shards"] or hash_json(inventory) != anchor["inventory_sha256"]:
        raise ValueError("Backends do not start from identical full FP32 parameter shards")
    if [r['branch_id'] for r in anchor['branches']] != list(group.branch_ids):
        raise ValueError("Reference branch identity/order differs")
    references = {(group.group_id, bid): torch.tensor(row["target_logprobs"], dtype=torch.float32)
                  for bid, row in zip(group.branch_ids, anchor["branches"], strict=True)}
    policy.train()
    modes = {}
    for mode in attempt["modes"]:
        policy.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        phase_started = time.monotonic()
        observed = []
        with attention_context(config.training.memory.attention_policy):
            if mode == "shared_reference_grpo":
                # Exact production loss/scaling/reduction implementation.
                loss, objective_rows, _ = window_update(policy, (group,), references, config, context)
                objective_rows = [{**r, "metrics": {k:v for k,v in r['metrics'].items() if not k.startswith('_')}} for r in objective_rows]
            else:
                loss = 0.
                objective_rows = []
                for tasks in microsteps((group,), context.world_size):
                    task = tasks[context.rank]
                    b = task[1] if task is not None else 0
                    values = policy(group.branch_batch(b, "cuda:0"), config.training.memory.loss_chunk_tokens or 256)
                    # Same cotangent on both backends: derivative of initial-policy
                    # GRPO with ratio=1, KL=0. No changed forward-dependent weights.
                    coefficient = -float(group.advantages[b]) / len(values)
                    scalar = values.sum() * coefficient * (context.world_size / group.group_size) if task else values.sum()*0
                    if task is not None:
                        loss += float(scalar.detach()) / context.world_size
                        observed.append({"branch_id": group.branch_ids[b], "values": values.detach().cpu().tolist(),
                                         "cotangent_before_world_branch_scaling": coefficient})
                    scalar.backward()
                    del scalar, values
                peers = context.gather({"loss": loss, "branches": observed})
                loss = sum(p["loss"] for p in peers)
                observed = [b for p in peers for b in p["branches"]]
        torch.cuda.synchronize()
        backward_seconds = time.monotonic() - phase_started
        path = reference_dir / ("gradients-" + mode) / f"rank-{context.rank}"
        local = (_store_gradients(policy, inventory, path) if baseline else _compare_gradients(policy, inventory, path))
        peers = context.gather(local)
        rows = {r["name"]: describe(combine([p[r["name"]] for p in peers])) for r in inventory["parameters"]}
        total = describe(combine(list(rows.values())))
        if total['elements'] != inventory['trainable_parameters']:
            raise ValueError("Full gradient elements not covered exactly once")
        categories = {k: describe(combine([v for n,v in rows.items() if _category(n)==k])) for k in sorted({_category(n) for n in rows})}
        forward = None
        if observed:
            deltas = []
            for branch in observed:
                expected = references[(group.group_id, branch['branch_id'])]
                deltas.append(statistics(torch.tensor(branch['values']), expected))
            forward = describe(combine(deltas))
        telemetry = context.gather({"rank": context.rank, "gdn": evidence(binding), "host": host_memory(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        modes[mode] = {"loss": loss, "all_parameters": total, "per_parameter": rows,
            "categories": categories, "forward_logprobs": forward, "observed_branches": observed,
            "objective_rows": objective_rows, "backward_seconds": backward_seconds, "ranks": telemetry}
        if context.primary:
            write_json(Path(directory) / (mode + ".json"), modes[mode])
            print(f"GRADIENT_PHASE backend={config.model.gdn_backend} mode={mode} elements={total['elements']} relative_l2={total['relative_l2']}", flush=True)
    policy.zero_grad(set_to_none=True)
    if context.gather(shard_digest(policy)) != digests:
        raise ValueError("Diagnostic unexpectedly changed model weights")
    return {"backend": config.model.gdn_backend, "inventory": inventory, "initial_parameter_shards": digests,
            "modes": modes, "seconds": time.monotonic()-started, "model_unchanged": True,
            "gradient_stage": "FP32 FSDP2-reduced gradients before clipping; no optimizer"}

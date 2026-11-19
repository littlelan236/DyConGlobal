"""Language parameter boundaries, meta loading, and FSDP2-managed forwards."""
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from core.io import read_json


def parameter_scope(model):
    base = model.base_model
    if (set(dict(model.named_children())) != {"model", "lm_head"}
            or set(dict(base.named_children())) != {"visual", "language_model"}):
        raise ValueError("Unknown Qwen module ownership; explicit language/vision boundary required")
    language = base.language_model
    head = model.get_output_embeddings()
    selected = {id(p) for module in (language, head) for p in module.parameters()}
    vision = {id(p) for p in base.visual.parameters()}
    if selected & vision or {id(p) for p in model.parameters()} != selected | vision:
        raise ValueError("Unknown or overlapping parameter ownership")
    aliases = {}
    for name, p in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(p), []).append(name)
        p.requires_grad_(id(p) in selected)
    rows = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        internal = name.removeprefix("model.")
        no_decay = (p.ndim < 2 or any(a.endswith("embed_tokens.weight") for a in aliases[id(p)])
                    or any(isinstance(module, nn.Embedding) and module.weight is p for module in language.modules()))
        rows.append({"name": internal, "hf_names": aliases[id(p)], "shape": list(p.shape),
                     "numel": p.numel(), "master_dtype": "torch.float32",
                     "weight_decay": not no_decay})
    result = {"schema_version": 1, "scope": "language_model", "parameters": rows,
              "trainable_parameters": sum(p["numel"] for p in rows),
              "frozen_vision_parameters": sum(p.numel() for p in base.visual.parameters()),
              "vision_residency": "immutable_base_on_disk; absent_from_text_execution"}
    if len(rows) != len(selected):
        raise ValueError("Full parameter inventory does not cover each language parameter once")
    return result


class FullPolicy(nn.Module):
    def __init__(self, conditional):
        super().__init__()
        self.language_model = conditional.base_model.language_model
        self.lm_head = conditional.get_output_embeddings()

    def forward(self, batch, block_size=256):
        labels, mask = batch["labels"], batch["loss_mask"].bool()
        if (not torch.equal(labels != -100, mask) or bool(mask[:, 0].any())
                or bool((mask & ~batch["attention_mask"].bool()).any())):
            raise ValueError("Full native label/mask coordinates disagree")
        valid = (labels[:, 1:] != -100).nonzero()
        if not len(valid):
            raise ValueError("Full branch has no targets")
        hidden = self.language_model(input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"], use_cache=False).last_hidden_state
        pieces = []
        for block in valid.split(block_size or len(valid)):
            b, pos = block.unbind(1)
            target = labels[b, pos + 1]
            def project(part, target_ids):
                logits = self.lm_head(part).float()
                return -torch.nn.functional.cross_entropy(logits, target_ids, reduction="none")
            part = hidden[b, pos]
            pieces.append(checkpoint(project, part, target, use_reentrant=False)
                          if self.training and torch.is_grad_enabled() else project(part, target))
        return torch.cat(pieces)


def shard_policy(policy, *, mesh, compute_dtype):
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    precision = MixedPrecisionPolicy(param_dtype=compute_dtype, reduce_dtype=torch.float32)
    for layer in policy.language_model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=precision, reshard_after_forward=True)
    # Root owns embedding/head together, preserving tying and gathering them only
    # around the root forward/backward. Decoder blocks are independent units.
    fully_shard(policy, mesh=mesh, mp_policy=precision, reshard_after_forward=True)
    return policy


def weight_files(path):
    path = Path(path)
    index = path / "model.safetensors.index.json"
    if index.exists():
        return {k: path / v for k, v in read_json(index)["weight_map"].items()}
    from safetensors import safe_open
    single = path / "model.safetensors"
    with safe_open(single, framework="pt", device="cpu") as handle:
        return {k: single for k in handle.keys()}


def local_tensor(parameter):
    return parameter.to_local() if hasattr(parameter, "to_local") else parameter


def load_sharded_base(policy, inventory, path, rank, world_size):
    from safetensors import safe_open
    files = weight_files(path)
    parameters = dict(policy.named_parameters())
    with torch.no_grad():
        for row in inventory["parameters"]:
            name = next((alias for alias in row["hf_names"] if alias in files), None)
            if name is None:
                raise ValueError("Missing base weight: " + row["name"])
            target = local_tensor(parameters[row["name"]])
            if not target.numel():
                continue
            with safe_open(files[name], framework="pt", device="cpu") as source:
                view = source.get_slice(name)
                if list(view.get_shape()) != row["shape"]:
                    raise ValueError("Base weight shape mismatch")
                offset = rank * math.ceil(row["shape"][0] / world_size)
                # FSDP2 uses contiguous Shard(0), including uneven/empty shards.
                for first in range(0, target.shape[0], max(1, 16 * 2**20 // max(4, target[0].numel() * 4))):
                    count = max(1, 16 * 2**20 // max(4, target[0].numel() * 4))
                    last = min(target.shape[0], first + count)
                    target[first:last].copy_(view[offset + first:offset + last].to(target.device, torch.float32))


def load(config, mesh, rank, world_size):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from RL.gdn_backend import install_gdn_backend
    install_gdn_backend(config.model)
    definition = Qwen3_5Config.from_pretrained(config.model.path, local_files_only=True)
    definition.use_cache = definition.text_config.use_cache = False
    definition._attn_implementation = "sdpa"
    definition.text_config._attn_implementation = "sdpa"
    with torch.device("meta"):
        conditional = Qwen3_5ForConditionalGeneration(definition)
        # Real HF configs may construct BF16 parameters, unlike tiny fixtures.
        # Convert metadata before allocating/sharding; optimizer masters stay FP32.
        conditional = conditional.to(dtype=torch.float32)
    inventory = parameter_scope(conditional)
    policy = FullPolicy(conditional)
    del conditional  # Vision stays on immutable disk, never on each rank's GPU.
    rotary = policy.language_model.rotary_emb
    policy.language_model.rotary_emb = type(rotary)(definition.text_config, device="cuda:0")
    policy.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.__future__.set_swap_module_params_on_conversion(True)
    shard_policy(policy, mesh=mesh, compute_dtype=getattr(torch, config.model.dtype))
    policy.to_empty(device="cuda:0")
    # to_empty also touches buffers; rebuild initialized nonpersistent RoPE data.
    policy.language_model.rotary_emb = type(rotary)(definition.text_config, device="cuda:0")
    if config.run.start.mode == "fresh":
        load_sharded_base(policy, inventory, config.model.path, rank, world_size)
    if set(dict(policy.named_parameters())) != {p["name"] for p in inventory["parameters"]}:
        raise ValueError("Sharding changed language parameter aliases/tying")
    return policy, inventory


def activation_offloader(policy, config):
    """Reuse scoped host leases only for decoder checkpoint input activations.

    FSDP2 owns parameter gathering, gradients and optimizer state. Never install
    a global saved-tensor hook around its forward/backward or its weight saves.
    Output-head checkpoint inputs currently remain on device.
    """
    memory = config.training.memory
    if not memory.host_offload.enabled:
        return None
    from RL.memory import BoundaryOffload
    offloader = BoundaryOffload(policy, int(memory.host_budget_gib * 2**30),
        enabled=memory.activation_offload in {"checkpoint_boundary", "memory_pressure"},
        host_settings=memory.host_offload,
        pressure_settings=memory.memory_pressure if memory.activation_offload == "memory_pressure" else None)
    for layer in policy.language_model.layers:
        layer._gradient_checkpointing_func = offloader.checkpoint
    return offloader


def optimizer(policy, inventory, settings, *, pool=None, admission=None):
    parameters = dict(policy.named_parameters())
    groups = [{"params": [parameters[r["name"]] for r in inventory["parameters"] if r["weight_decay"] == decay],
               "weight_decay": settings.weight_decay if decay else 0.0} for decay in (True, False)]
    actual = [p for group in groups for p in group["params"]]
    if len({id(p) for p in actual}) != len(actual) or {id(p) for p in actual} != {id(p) for p in policy.parameters()}:
        raise ValueError("Optimizer must cover every unique language parameter exactly once")
    if any(p.dtype != torch.float32 for p in actual):
        raise ValueError("Full optimizer master shards must be FP32")
    if settings.memory.adam_moment_offload.mode != "disabled":
        from RL.optimizer_offload import StreamedAdamW
        result = StreamedAdamW(groups, lr=settings.learning_rate,
            settings=settings.memory.adam_moment_offload, host_settings=settings.memory.host_offload,
            pool=pool, admission=admission)
    else:
        result = torch.optim.AdamW(groups, lr=settings.learning_rate, betas=(0.9, 0.999), foreach=False)
    from RL.lr_schedule import factor
    schedule = lambda step: factor(step, settings)
    return result, torch.optim.lr_scheduler.LambdaLR(result, schedule)


def microsteps(window, world_size):
    if window and all("execution_chunk" in group.metadata for group in window):
        if world_size not in {2, 4, 8} or len(window) > 8 or any(group.group_size != 16 for group in window):
            raise ValueError("Full FSDP2 8x16 requires up to eight 16-branch groups on 2/4/8 ranks")
        return [tuple((offset + rank, branch) if offset + rank < len(window) else None
                      for rank in range(world_size))
                for offset in range(0, len(window), world_size) for branch in range(16)]
    tasks = [(g, b) for g, group in enumerate(window) for b in range(group.group_size)]
    return [tuple(tasks[i + r] if i + r < len(tasks) else None for r in range(world_size))
            for i in range(0, len(tasks), world_size)]


def global_clip(policy, limit, group):
    import torch.distributed as dist
    total = torch.zeros((), device=local_tensor(next(policy.parameters())).device, dtype=torch.float64)
    missing = []
    for name, p in policy.named_parameters():
        if p.grad is None:
            missing.append(name)
        else:
            grad = local_tensor(p.grad)
            for chunk in grad.reshape(-1).split(1024 * 1024):
                total += chunk.double().square().sum()
    dist.all_reduce(total, group=group)
    norm = float(total.sqrt())
    if not math.isfinite(norm) or missing:
        raise ValueError(f"Invalid full gradient: norm={norm}, missing={missing[:8]}")
    coefficient = min(1.0, limit / (norm + 1e-6))
    for p in policy.parameters():
        local_tensor(p.grad).mul_(coefficient)
    return norm


def shard_digest(policy):
    import hashlib
    digest = hashlib.sha256()
    for name, p in policy.named_parameters():
        local = local_tensor(p).detach()
        digest.update(name.encode())
        for chunk in local.reshape(-1).split(16 * 2**20):
            digest.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_optimizer_state(policy, opt):
    """Check every local master and moment without materializing a full model."""
    if hasattr(opt, "iter_local_state_chunks"):
        return opt.validate()
    finite = torch.ones((), dtype=torch.bool, device=local_tensor(next(policy.parameters())).device)
    count = 0
    for name, parameter in policy.named_parameters():
        state = opt.state[parameter]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Invalid full optimizer state: " + name)
        for value in (parameter, state["exp_avg"], state["exp_avg_sq"]):
            value = local_tensor(value)
            if value.dtype != torch.float32:
                raise ValueError("Full master/moments must remain FP32: " + name)
            for chunk in value.detach().reshape(-1).split(1024 * 1024):
                finite.logical_and_(torch.isfinite(chunk).all())
        count += 1
    if not bool(finite):
        raise ValueError("Nonfinite full master or optimizer moments")
    return {"parameter_tensors": count, "master_moments_dtype": "float32", "all_finite": True}


def parameter_samples(policy):
    """Bounded update evidence, evenly sampling each local master shard."""
    result = {}
    for name, p in policy.named_parameters():
        flat = local_tensor(p).detach().reshape(-1)
        count = min(64, flat.numel())
        # Integer arithmetic avoids float32 rounding n-1 up to n on large shards.
        indices = (torch.arange(count, device=flat.device, dtype=torch.long)
                   * max(0, flat.numel() - 1) // max(1, count - 1))
        result[name] = flat[indices].clone()
    return result

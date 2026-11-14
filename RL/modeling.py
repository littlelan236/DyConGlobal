"""Conditional-generation loading and exact all-linear text-backbone PEFT coverage."""

import hashlib

import torch
from peft import LoraConfig, PeftModel, get_peft_model


def text_targets(model):
    base = model.base_model
    text = base.language_model if hasattr(base, "language_model") else base
    output = model.get_output_embeddings()
    if any(module is output for module in text.modules()):
        raise ValueError("lm_head falls inside the selected text backbone; inspect the model boundary")
    paths = {id(module): name for name, module in model.named_modules()}
    targets = []
    for _, module in text.named_modules():
        if isinstance(module, torch.nn.Linear):
            name = paths[id(module)]
            if not name or any(part in {"visual", "vision", "lm_head"} for part in name.split(".")):
                raise ValueError("Unexpected vision/output module inside the text backbone")
            category = "mlp" if ".mlp." in name else (
                "linear_attention" if ".linear_attn." in name else (
                    "full_attention" if ".self_attn." in name else "extra"))
            targets.append({"path": name, "type": "Linear", "shape": list(module.weight.shape),
                            "in_features": module.in_features, "out_features": module.out_features,
                            "category": category})
    if not targets:
        raise ValueError("Text backbone has no Linear targets")
    return sorted(targets, key=lambda item: item["path"])


def inject(model, settings, *, adapter=None, expected_targets=None):
    targets = text_targets(model)
    if expected_targets is not None and targets != expected_targets:
        raise ValueError("Resolved text backbone targets/shapes differ from the source adapter")
    model.requires_grad_(False)
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=settings.rank, lora_alpha=settings.alpha,
            lora_dropout=settings.dropout, bias="none", target_modules=[item["path"] for item in targets]))
    actual = {name.removeprefix("base_model.model."): module for name, module in model.named_modules()
              if hasattr(module, "lora_A") and "default" in module.lora_A}
    if sorted(actual) != [item["path"] for item in targets]:
        raise ValueError("PEFT did not inject exactly the frozen target list")
    # PEFT minimizes long target sets to suffixes in-place. Preserve the exact
    # resolved boundary in the exported config as well as in our identity file.
    model.peft_config["default"].target_modules = {item["path"] for item in targets}
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    expected_count = settings.rank * sum(t["in_features"] + t["out_features"] for t in targets)
    if (sum(p.numel() for _, p in trainable) != expected_count
            or any(".lora_A.default." not in name and ".lora_B.default." not in name for name, _ in trainable)):
        raise ValueError("Trainable parameter inventory differs from the all-linear LoRA contract")
    if len(model.peft_config) != 1 or "default" not in model.peft_config:
        raise ValueError("Exactly one trainable default adapter is supported")
    return model, targets


def load_base(config):
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real Qwen3.5-9B smoke")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        config.model.path, local_files_only=True, dtype=getattr(torch, config.model.dtype),
        attn_implementation=config.model.attention_backend,
    ).to(torch.device("cuda:0"))
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(config.model.path, local_files_only=True)
    devices = {str(p.device) for p in model.parameters()}
    if devices != {"cuda:0"}:
        raise RuntimeError(f"Model must reside entirely on the selected GPU: {devices}")
    # Some recurrent/norm parameters intentionally remain fp32 in Transformers.
    return model, tokenizer


def enable_checkpointing(model):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()


def parameter_manifest(model, targets):
    params = [{"name": name, "shape": list(p.shape), "numel": p.numel(), "dtype": str(p.dtype),
               "device": str(p.device), "trainable": p.requires_grad} for name, p in model.named_parameters()]
    return {"targets": targets, "target_count": len(targets), "parameters": params,
            "loaded_parameters": sum(p["numel"] for p in params),
            "trainable_parameters": sum(p["numel"] for p in params if p["trainable"])}


def tensor_sha(tensor):
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def adapter_fingerprint(model):
    return {name: tensor_sha(p) for name, p in model.named_parameters() if p.requires_grad}


def frozen_checksums(model):
    # Sample each frozen tensor without making a second copy of the 9B base.
    return {name: {"shape": list(p.shape), "first": tensor_sha(p.detach().reshape(-1)[:64]),
                   "last": tensor_sha(p.detach().reshape(-1)[-64:])}
            for name, p in model.named_parameters() if not p.requires_grad}


class Coverage:
    def __init__(self, model, targets):
        self.modules = {name.removeprefix("base_model.model."): module
                        for name, module in model.named_modules() if hasattr(module, "lora_A")}
        self.rows = {t["path"]: {**t, "executed": False, "gradient_seen": False,
                                "nonzero_B_gradient": False, "max_B_gradient_norm": 0.0}
                     for t in targets}
        self.handles = []
        self.last_gradients = {}
        for name, module in self.modules.items():
            def hook(_module, _inputs, _output, path=name):
                self.rows[path]["executed"] = True
            self.handles.append(module.register_forward_hook(hook))

    def gradients(self):
        self.last_gradients = {}
        for name, module in self.modules.items():
            grad = module.lora_B["default"].weight.grad
            if grad is not None:
                if not torch.isfinite(grad).all():
                    raise ValueError(f"Nonfinite adapter gradient: {name}")
                norm = float(grad.detach().float().norm())
                row = self.rows[name]
                row["gradient_seen"] = True
                row["nonzero_B_gradient"] |= norm > 0
                row["max_B_gradient_norm"] = max(row["max_B_gradient_norm"], norm)
                category = row["category"]
                summary = self.last_gradients.setdefault(category, {"projections": 0, "nonzero_B_gradients": 0,
                                                                   "max_B_gradient_norm": 0.0})
                summary["projections"] += 1
                summary["nonzero_B_gradients"] += int(norm > 0)
                summary["max_B_gradient_norm"] = max(summary["max_B_gradient_norm"], norm)

    def begin_attempt(self):
        import copy
        self._attempt_snapshot = copy.deepcopy((self.rows, self.last_gradients))

    def abort_attempt(self):
        if hasattr(self, "_attempt_snapshot"):
            self.rows, self.last_gradients = self._attempt_snapshot
            del self._attempt_snapshot

    def finish(self, before, after):
        for name, row in self.rows.items():
            prefix = "base_model.model." + name + "."
            row["adapter_changed"] = any(after[key] != value for key, value in before.items()
                                         if key.startswith(prefix))
            if row["executed"] and (not row["nonzero_B_gradient"] or not row["adapter_changed"]):
                raise ValueError(f"Executed projection has no effective LoRA gradient/update: {name}")
        return list(self.rows.values())

    def close(self):
        for handle in self.handles:
            handle.remove()

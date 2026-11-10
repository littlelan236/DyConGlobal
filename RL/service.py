"""The public service used by both dycon lora-train and Python API clients."""

import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

from core.io import hash_json, now, read_json, sha256, write_json
from core.locking import file_lock
from RL.artifacts import base_identity, commit_receipt, dependency_versions, source_identity
from RL.config import attempt_directory, load, schedule_contract


def _source(config):
    """Validate source receipts before loading model weights, with no Torch import."""
    mode = config.run.start.mode
    if mode == "fresh":
        return None
    # checkpoints imports Torch; source validation for dry-run uses receipt/JSON only instead.
    from RL.artifacts import verify_receipt

    path = Path(config.run.start.checkpoint)
    owner = path if mode == "resume" else path.parent
    if mode == "resume" and not (path / "training-state.pt").is_file():
        raise ValueError("resume needs a complete checkpoint; choose warm_start for adapter-only output")
    receipt = verify_receipt(owner, "lora-checkpoint-v1" if mode == "resume" else None)
    if receipt["kind"] not in {"lora-checkpoint-v1", "lora-attempt-v1"}:
        raise ValueError("Unsupported source adapter owner")
    adapter = path / "adapter" if mode == "resume" else path
    if adapter.name != "adapter":
        raise ValueError("warm_start requires an explicitly named adapter/ export")
    identity = read_json(adapter / "identity.json")
    return {"mode": mode, "path": str(path), "owner": str(owner),
            "receipt_sha256": sha256(owner / "receipt.json"),
            "adapter_identity_sha256": sha256(adapter / "identity.json"), "identity": identity,
            "adapter": str(adapter)}


def _preflight(config, root, parent):
    base = base_identity(config.model.path, root)
    fixture = Path(__file__).parent / "fixtures/causal-text-smoke-v1.json"
    if parent:
        contract = parent["identity"]["contract"]
        if contract["base_fingerprint"] != base["fingerprint"] or contract["lora"] != config.lora.model_dump():
            raise ValueError("Source adapter base/tokenizer/LoRA identity mismatch")
        if config.run.start.mode == "resume":
            state = read_json(Path(config.run.start.checkpoint) / "state.json")
            expected = state["contract"]
            if (expected["schedule"] != schedule_contract(config)
                    or expected["input"]["fixture_sha256"] != sha256(fixture)):
                raise ValueError("Resume schedule or fixed input identity mismatch")
            step = state["state"]["global_step"]
            if step >= config.training.max_updates:
                raise ValueError("Source schedule is already complete; use warm_start to begin a new stage")
            stop = config.validation.stop_after_update
            if stop is not None and stop <= step:
                raise ValueError("Pause target must follow the restored global step")
    return base


def run(root_config, training_config):
    """Run/dry-run one immutable engineering attempt; no implicit resume or CPU fallback."""
    root = Path(root_config["project_root"]).resolve()
    config, source = load(training_config, root)
    directory = attempt_directory(config)
    lock = root / "state/artifact-locks/lora-attempt" / (config.run.attempt_id + ".lock")
    with file_lock(lock, nonblocking=True):
        if directory.exists():
            raise FileExistsError(f"Attempt ID already exists, including failed attempts: {directory}")
        parent = _source(config)
        if config.run.mode == "dry-run":
            base = _preflight(config, root, parent)
            return {"mode": "dry-run", "purpose": "engineering_smoke", "status": "validated",
                    "output": str(directory), "base_fingerprint": base["fingerprint"],
                    "fixture_sha256": sha256(Path(__file__).parent / "fixtures/causal-text-smoke-v1.json"),
                    "resolved_config": config.model_dump(), "source": parent and parent["path"],
                    "coverage": "text backbone all Linear; exact paths resolved only when loading the model"}
        directory.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(directory.parent / ("." + config.run.attempt_id + ".lock"), nonblocking=True):
            directory.mkdir(exist_ok=False)
            attempt = {"schema_version": 1, "attempt_id": config.run.attempt_id, "purpose": "engineering_smoke",
                       "start_mode": config.run.start.mode, "status": "initializing", "created_at": now(),
                       "parent": {k: v for k, v in parent.items() if k not in {"identity", "adapter"}} if parent else None}
            write_json(directory / "attempt.json", attempt)
            (directory / "training-config.yaml").write_bytes(source.read_bytes())
            write_json(directory / "resolved-config.json", config.model_dump())
            try:
                base = _preflight(config, root, parent)
                write_json(directory / "model-identity.json", base)
                gpu_lock = root / "state/artifact-locks/lora-gpu" / (str(config.runtime.gpu) + ".lock")
                with file_lock(gpu_lock, nonblocking=True):
                    result = _train(config, root, directory, base, parent, attempt)
                import gc

                import torch

                gc.collect()
                torch.cuda.empty_cache()
                return result
            except BaseException as exc:
                attempt.update(status="failed", ended_at=now(), error=f"{type(exc).__name__}: {exc}")
                write_json(directory / "attempt.json", attempt)
                (directory / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
                # Preserve any committed update-boundary checkpoints; never fabricate success.
                raise


def _select_gpu(config):
    gpu = str(config.runtime.gpu)
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
            raise RuntimeError("GPU selection must precede CUDA initialization; call the service in a fresh process")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    query = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=index,uuid,name,memory.used,memory.free",
                                     "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
    fields = [part.strip() for part in query.split(",")]
    if len(fields) != 5 or int(fields[3]) > 512:
        raise RuntimeError(f"Configured physical GPU {gpu} is not idle: {query}")
    return {"physical_gpu": config.runtime.gpu, "uuid": fields[1], "name": fields[2],
            "initial_used_MiB": int(fields[3]), "initial_free_MiB": int(fields[4])}


def _train(config, root, directory, base, parent, attempt):
    gpu = _select_gpu(config)  # Before importing the training stack.
    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen

    import core
    import RL
    from RL.checkpoints import (
        adapter_contract,
        export_adapter,
        restore_checkpoint,
        save_checkpoint,
        verify_adapter,
    )
    from RL.modeling import (
        Coverage,
        adapter_fingerprint,
        enable_checkpointing,
        frozen_checksums,
        inject,
        load_base,
        parameter_manifest,
    )
    from RL.training import optimizer_and_scheduler, seed_everything, train
    from RL.validation import causal_ce, fixture_batches, probe_logits

    if torch.cuda.device_count() != 1:
        raise RuntimeError("LoRA training requires exactly one visible CUDA device")
    # Acceptance tolerances are frozen before any GPU update, including resume/reload comparisons.
    tolerances = {"adapter_atol": 2e-5, "adapter_rtol": 2e-3, "logits_atol": 0.125,
                  "logits_rtol": 0.01, "loss_atol": 0.05, "loss_rtol": 0.01}
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    seed_everything(config.run.seed)
    source = source_identity(root)
    environment = {"python": sys.version, "platform": platform.platform(), "cuda": torch.version.cuda,
                   **gpu, "versions": dependency_versions(), "source": source,
                   "imports": {"core": str(Path(core.__file__).resolve()), "RL": str(Path(RL.__file__).resolve())},
                   "attention_backend": config.model.attention_backend,
                   "linear_attention": {name: {"module": getattr(getattr(qwen, name, None), "__module__", None),
                                               "name": getattr(getattr(qwen, name, None), "__name__", None)}
                                        for name in ["causal_conv1d_fn", "torch_chunk_gated_delta_rule"]},
                   "tolerances": tolerances}
    environment["determinism"] = {
        "algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    environment["linear_attention"]["execution_backend"] = {
        "causal_conv1d": "reference_pytorch" if source["dependencies"]["causal-conv1d"] is None else "installed_kernel_dispatch",
        "gated_delta_rule": "reference_pytorch" if source["dependencies"]["flash-linear-attention"] is None else "installed_kernel_dispatch",
    }
    write_json(directory / "environment.json", environment)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_base(config)
    checkpoint_keys = set(read_json(Path(config.model.path) / "model.safetensors.index.json")["weight_map"])
    instantiated_keys = set(model.state_dict())
    checkpoint_inventory = {
        "base_loaded_parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_weight_count": len(checkpoint_keys),
        "instantiated_state_count": len(instantiated_keys),
        "checkpoint_weights_not_instantiated": sorted(checkpoint_keys - instantiated_keys),
        "instantiated_states_without_checkpoint_key": sorted(instantiated_keys - checkpoint_keys),
    }
    expected_targets = parent["identity"]["contract"]["targets"] if parent else None
    model, targets = inject(model, config.lora, adapter=parent["adapter"] if parent else None,
                            expected_targets=expected_targets)
    enable_checkpointing(model)
    torch.cuda.synchronize()
    loading = {"seconds": time.monotonic() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    manifest = parameter_manifest(model, targets)
    manifest["checkpoint_inventory"] = checkpoint_inventory
    manifest["extra_backbone_targets"] = [t for t in targets if t["category"] == "extra"]
    write_json(directory / "trainable-parameters.json", manifest)
    batches, input_identity = fixture_batches(tokenizer, config.validation.max_sequence_length)
    adapter_identity = adapter_contract(base, targets, config.lora)
    if parent:
        verify_adapter(parent["adapter"], adapter_identity)
    contract = {"schedule": schedule_contract(config), "input": input_identity,
                "adapter": adapter_identity, "source_fingerprint": source["fingerprint"],
                "dependencies": source["dependencies"], "attention": environment["linear_attention"]}
    contract["determinism"] = environment["determinism"]
    optimizer, scheduler = optimizer_and_scheduler(model, config.training)
    initial_optimizer_entries = len(optimizer.state)
    initial_adapter = adapter_fingerprint(model)
    frozen_before = frozen_checksums(model)
    initial_logits = probe_logits(model, batches[0])
    # These compact outputs are engineering evidence for a later process to compare.
    torch.save(initial_logits, directory / "initial-probe.pt")
    write_json(directory / "initial-adapter.json", initial_adapter)
    state = {"global_step": 0, "consumed_microbatches": 0, "consumed_tokens": 0}
    if config.run.start.mode == "resume":
        state = restore_checkpoint(parent["path"], optimizer, scheduler, contract, adapter_identity)
        if state["consumed_microbatches"] != state["global_step"] * config.training.gradient_accumulation_steps:
            raise ValueError("Fixed fixture resume input cursor is inconsistent with global step")
        expected_tokens = sum(int((batches[i % len(batches)]["labels"][:, 1:] != -100).sum())
                              for i in range(state["consumed_microbatches"]))
        if state["consumed_tokens"] != expected_tokens:
            raise ValueError("Fixed fixture resume token counter is inconsistent")
    initial_state = dict(state)
    if config.run.start.mode == "warm_start":
        source_owner = Path(parent["owner"])
        if source_owner.joinpath("final-probe.pt").is_file():
            expected_logits = torch.load(source_owner / "final-probe.pt", weights_only=True, map_location="cpu")
            torch.testing.assert_close(initial_logits, expected_logits, atol=tolerances["logits_atol"],
                                       rtol=tolerances["logits_rtol"])
        if initial_optimizer_entries != 0 or scheduler.last_epoch != 0 or any(state.values()):
            raise ValueError("Warm-start optimizer/schedule/cursors were not reset")
    coverage = Coverage(model, targets)
    attempt.update(status="training", contract=contract, loading=loading, initial_state=initial_state)
    write_json(directory / "attempt.json", attempt)
    metrics_path = directory / "metrics.jsonl"
    metrics_path.touch(exist_ok=False)
    stop = config.validation.stop_after_update or config.training.max_updates
    total_microbatches = (stop - state["global_step"]) * config.training.gradient_accumulation_steps
    cursor = state["consumed_microbatches"]
    stream = (batches[(cursor + index) % len(batches)] for index in range(total_microbatches))
    def update(current, metrics):
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if current["global_step"] % config.training.checkpoint_every_updates == 0 or current["global_step"] == stop:
            checkpoint = directory / "checkpoints" / f"step-{current['global_step']:06d}"
            save_checkpoint(model, optimizer, scheduler, current, checkpoint, contract, adapter_identity)
            # An eval probe does not consume training RNG, and is not part of the checkpoint payload.
        print(json.dumps({"attempt_id": config.run.attempt_id, **metrics}), flush=True)
    try:
        state = train(model, stream, causal_ce, optimizer, scheduler, max_updates=stop,
                      gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                      grad_clip=config.training.grad_clip, vocab_size=model.config.text_config.vocab_size,
                      max_length=config.validation.max_sequence_length, state=state,
                      on_update=update, on_gradients=coverage.gradients)
        if state["global_step"] != stop:
            raise ValueError("Fixed fixture exhausted before the requested update boundary")
        final_adapter = adapter_fingerprint(model)
        coverage_report = coverage.finish(initial_adapter, final_adapter)
    finally:
        coverage.close()
    frozen_after = frozen_checksums(model)
    if frozen_before != frozen_after:
        raise ValueError("Frozen base tensor checksums changed")
    final_logits = probe_logits(model, batches[0])
    torch.save(final_logits, directory / "final-probe.pt")
    write_json(directory / "final-adapter.json", final_adapter)
    export_adapter(model, directory / "adapter", adapter_identity)
    status = "completed" if stop == config.training.max_updates else "paused"
    validation = {"purpose": "engineering_smoke", "state": state, "loading": loading, "input": input_identity,
                  "frozen_base_unchanged": True, "frozen_checksums": frozen_after,
                  "adapter_changed": initial_adapter != final_adapter,
                  "initial_optimizer_entries": initial_optimizer_entries, "initial_state": initial_state,
                  "optimizer_entries_after_training": len(optimizer.state), "coverage": coverage_report,
                  "source_adapter_verified": bool(parent), "tolerances": tolerances,
                  "logits_bytes_float32_at_training_length": config.validation.max_sequence_length
                  * model.config.text_config.vocab_size * 4}
    write_json(directory / "validation.json", validation)
    if parent:
        if sha256(Path(parent["owner"]) / "receipt.json") != parent["receipt_sha256"]:
            raise ValueError("Source checkpoint changed during training")
        verify_adapter(parent["adapter"], adapter_identity)
    attempt.update(status=status, ended_at=now(), final_state=state)
    write_json(directory / "attempt.json", attempt)
    (directory / "report.md").write_text(
        f"# LoRA engineering smoke: {config.run.attempt_id}\n\n"
        f"Status: {status}. Start: {config.run.start.mode}. Global step: {state['global_step']}.\n\n"
        f"Targets: {len(targets)}. Trainable: {manifest['trainable_parameters']:,}. "
        f"Sequence length: {config.validation.max_sequence_length}.\n\n"
        "Fixed synthetic CE is an engineering check; this is not RL or benchmark quality training.\n",
        encoding="utf-8")
    commit_receipt(directory, "lora-attempt-v1", status=status, attempt_id=config.run.attempt_id,
                   contract_sha256=hash_json(contract))
    return {"attempt_id": config.run.attempt_id, "status": status, "path": str(directory),
            "state": state, "target_count": len(targets), "trainable_parameters": manifest["trainable_parameters"]}

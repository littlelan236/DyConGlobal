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
from RL.resources import memory_plan
from RL.supervision import describe_policy


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
    memory_plan(config)
    base = base_identity(config.model.path, root)
    if config.algorithm is not None:
        from RL.grpo.data import prepare
        from RL.grpo.contract import validate_input, validate_resume, validate_reference_files
        raw, proof = prepare(config)
        validate_input(base, proof)
        if parent:
            identity = parent["identity"]["contract"]
            if identity["base_fingerprint"] != base["fingerprint"] or identity["lora"] != config.lora.model_dump():
                raise ValueError("Source adapter base/tokenizer/LoRA identity mismatch")
            if config.run.start.mode == "resume":
                metadata = read_json(Path(parent["path"]) / "state.json")
                validate_resume(config, proof, metadata)
                validate_reference_files(parent["path"], metadata["contract"])
                if metadata["contract"].get("source_fingerprint") != source_identity(root)["fingerprint"]:
                    raise ValueError("GRPO resume source implementation identity mismatch")
        return {**base, "grpo_dataset": proof}
    fixture = Path(__file__).parent / "fixtures/causal-text-smoke-v1.json"
    if parent:
        contract = parent["identity"]["contract"]
        if contract["base_fingerprint"] != base["fingerprint"] or contract["lora"] != config.lora.model_dump():
            raise ValueError("Source adapter base/tokenizer/LoRA identity mismatch")
        if config.run.start.mode == "resume":
            state = read_json(Path(config.run.start.checkpoint) / "state.json")
            expected = state["contract"]
            if "supervision" not in expected.get("schedule", {}).get("training", {}):
                raise ValueError("Legacy checkpoint lacks a supervision contract; use explicit adapter warm_start")
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
                    "memory_plan": memory_plan(config),
                    "supervision": describe_policy(config.training.supervision),
                    "algorithm": config.algorithm.name if config.algorithm else "causal_ce",
                    "grpo": config.grpo.model_dump() if config.grpo else None,
                    "dataset": base.get("grpo_dataset"),
                    "coverage": "text backbone all Linear; exact paths resolved only when loading the model"}
        directory.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(directory.parent / ("." + config.run.attempt_id + ".lock"), nonblocking=True):
            directory.mkdir(exist_ok=False)
            attempt = {"schema_version": 1, "attempt_id": config.run.attempt_id, "purpose": "engineering_smoke",
                       "start_mode": config.run.start.mode, "status": "initializing", "created_at": now(),
                       "supervision": describe_policy(config.training.supervision),
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
    from RL.memory import LossBackend, attention_context, configure_checkpointing, install_mlp_backend
    from RL.modeling import (
        Coverage,
        adapter_fingerprint,
        frozen_checksums,
        inject,
        load_base,
        parameter_manifest,
    )
    from RL.profiling import UpdateTelemetry
    from RL.supervision import consumed_selected_tokens
    from RL.training import optimizer_and_scheduler, seed_everything, train
    from RL.validation import fixture_batches, probe_logits

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
    # bf16 reduced-precision GEMM reductions depend on token-block dimensions.
    # Full-fp32 accumulation is required for dense/chunked numerical equivalence.
    torch.backends.cuda.preferred_blas_library("cublaslt")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
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
    plan = memory_plan(config)
    environment["memory_plan"] = plan
    environment["qwen_implementation_sha256"] = sha256(Path(qwen.__file__))
    environment["linear_attention"] = {
        name: {"module": getattr(getattr(qwen, name, None), "__module__", None),
               "name": getattr(getattr(qwen, name, None), "__name__", None)}
        for name in ["causal_conv1d_fn", "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"]}
    environment["determinism"] = {
        "algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "bf16_reduction_split_k": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k,
        "blas_library": str(torch.backends.cuda.preferred_blas_library()),
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
    mlps = install_mlp_backend(model, config.training.memory.mlp_chunk_tokens)
    offloader = configure_checkpointing(model, config, plan["host_budget_bytes_per_rank"])
    torch.cuda.synchronize()
    loading = {"seconds": time.monotonic() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    manifest = parameter_manifest(model, targets)
    manifest["checkpoint_inventory"] = checkpoint_inventory
    manifest["extra_backbone_targets"] = [t for t in targets if t["category"] == "extra"]
    write_json(directory / "trainable-parameters.json", manifest)
    if config.algorithm is not None:
        return _train_grpo(config, root, directory, base, parent, attempt, model, targets,
                           optimizer_settings=config.training, environment=environment,
                           loading=loading, mlps=mlps, offloader=offloader)
    batches, input_identity = fixture_batches(tokenizer, config.validation.max_sequence_length,
                                              config.training.supervision)
    write_json(directory / "input-identity.json", input_identity)
    write_json(directory / "selection.json", input_identity["supervision"])
    adapter_identity = adapter_contract(base, targets, config.lora)
    if parent:
        verify_adapter(parent["adapter"], adapter_identity)
    contract = {"schedule": schedule_contract(config), "input": input_identity,
                "adapter": adapter_identity, "source_fingerprint": source["fingerprint"],
                "dependencies": source["dependencies"], "attention": environment["linear_attention"]}
    contract["determinism"] = environment["determinism"]
    contract["memory_backend"] = {"strategy": config.training.memory.model_dump(),
                                  "checkpoint_policy": config.training.checkpoint_policy,
                                  "qwen_implementation_sha256": environment["qwen_implementation_sha256"]}
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
        expected_tokens = consumed_selected_tokens(
            batches, state["consumed_microbatches"], vocab_size=model.config.text_config.vocab_size,
            max_length=config.validation.max_sequence_length, supervision=config.training.supervision)
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
    loss_backend = LossBackend(config.training.memory)
    telemetry = UpdateTelemetry(model, loss_backend, mlps, offloader, directory,
                                config.training.memory.profile_first_update,
                                config.training.memory.attention_policy == "flash_only")
    attempt.update(status="training", contract=contract, loading=loading, initial_state=initial_state)
    write_json(directory / "attempt.json", attempt)
    metrics_path = directory / "metrics.jsonl"
    metrics_path.touch(exist_ok=False)
    stop = config.validation.stop_after_update or config.training.max_updates
    total_microbatches = (stop - state["global_step"]) * config.training.gradient_accumulation_steps
    cursor = state["consumed_microbatches"]
    stream = (batches[(cursor + index) % len(batches)] for index in range(total_microbatches))
    def update(current, metrics):
        metrics.update(telemetry.finish())
        metrics["adapter_gradient_coverage"] = coverage.last_gradients
        if (directory / "kernel-audit.json").exists() and "kernel_audit" not in environment:
            audit = read_json(directory / "kernel-audit.json")
            environment["kernel_audit"] = {key: value for key, value in audit.items() if key != "cuda_kernels"}
            write_json(directory / "environment.json", environment)
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
        with attention_context(config.training.memory.attention_policy):
            state = train(model, stream, loss_backend, optimizer, scheduler, max_updates=stop,
                          gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                          grad_clip=config.training.grad_clip, vocab_size=model.config.text_config.vocab_size,
                          max_length=config.validation.max_sequence_length, state=state,
                          on_update=update, on_gradients=coverage.gradients,
                          on_window_start=telemetry.start, supervision=config.training.supervision)
        if state["global_step"] != stop:
            raise ValueError("Fixed fixture exhausted before the requested update boundary")
        final_adapter = adapter_fingerprint(model)
        coverage_report = coverage.finish(initial_adapter, final_adapter)
    finally:
        coverage.close()
        telemetry.close()
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


def _train_grpo(config, root, directory, base, parent, attempt, model, targets, *,
                optimizer_settings, environment=None, loading=None, mlps=(), offloader=None):
    """Offline objective within the common attempt/model/checkpoint lifecycle."""
    import shutil
    import torch

    from RL.checkpoints import adapter_contract, export_adapter, restore_checkpoint, save_checkpoint, verify_adapter
    from RL.grpo.contract import validate_input, validate_resume, validate_reference_files, receipt_metadata
    from RL.grpo.data import prepare, tensor_groups
    from RL.grpo.backend import GroupObjective
    from RL.grpo.reference import build_reference, load_reference
    from RL.memory import attention_context
    from RL.modeling import adapter_fingerprint, frozen_checksums, Coverage
    from RL.profiling import UpdateTelemetry
    from RL.training import optimizer_and_scheduler, train_grpo

    raw_groups, proof = prepare(config)
    validate_input(base, proof)
    groups = list(tensor_groups(raw_groups))
    write_json(directory / "input-identity.json", proof)
    write_json(directory / "selection.json", {"group_ids": proof["eligible_group_ids"],
                "group_order_sha256": proof["group_order_sha256"]})
    write_json(directory / "frozen-groups.json", raw_groups)
    adapter_identity = adapter_contract(base, targets, config.lora)
    if parent:
        verify_adapter(parent["adapter"], adapter_identity)
    initial_adapter = adapter_fingerprint(model)
    frozen_before = frozen_checksums(model)
    write_json(directory / "initial-adapter.json", initial_adapter)
    (directory / "grpo").mkdir()
    environment = environment or {}
    source = source_identity(root)
    metadata = None
    if config.run.start.mode == "resume":
        metadata = read_json(Path(parent["path"]) / "state.json")
        validate_resume(config, proof, metadata)
        validate_reference_files(parent["path"], metadata["contract"])
        shutil.copytree(Path(parent["path"]) / "grpo", directory / "grpo", dirs_exist_ok=True)
        reference_proof = metadata["contract"]["grpo"]
        reference_values = load_reference(directory, groups, proof, reference_proof)
    else:
        export_adapter(model, directory / "grpo/reference/adapter", adapter_identity)
        with attention_context(config.training.memory.attention_policy):
            reference_values, reference_proof = build_reference(model, groups, directory, proof,
                base, hash_json(initial_adapter), config.training.memory.loss_chunk_tokens)
    contract = {"schedule": schedule_contract(config), "input": proof, "grpo": reference_proof,
                "adapter": adapter_identity, "source_fingerprint": source["fingerprint"],
                "dependencies": source["dependencies"], "attention": environment.get("linear_attention"),
                "determinism": environment.get("determinism"),
                "qwen_implementation_sha256": environment.get("qwen_implementation_sha256")}
    optimizer, scheduler = optimizer_and_scheduler(model, optimizer_settings)
    if metadata is not None:
        if metadata["contract"] != contract:
            raise ValueError("GRPO checkpoint contract differs; exact resume rejected")
        state = restore_checkpoint(parent["path"], optimizer, scheduler, contract, adapter_identity)
    else:
        state = {"global_step": 0, "group_cursor": 0, "consumed_groups": 0,
                 "consumed_branches": 0, "consumed_tokens": 0}
    initial_state = dict(state)
    coverage = Coverage(model, targets)
    objective = GroupObjective(reference_values, config.grpo, config.training.memory.loss_chunk_tokens)
    telemetry = UpdateTelemetry(model, objective, mlps, offloader, directory,
                    config.training.memory.profile_first_update,
                    config.training.memory.attention_policy == "flash_only")
    metrics_path = directory / "metrics.jsonl"
    metrics_path.touch(exist_ok=False)
    attempt.update(status="training", contract=contract, initial_state=initial_state, loading=loading)
    write_json(directory / "attempt.json", attempt)
    write_json(directory / "grpo-contract.json", contract)
    stop = config.validation.stop_after_update or proof["planned_updates"]
    extra_files = {p.relative_to(directory).as_posix(): p for p in (directory / "grpo").rglob("*") if p.is_file()}
    # Metric prefix travels with a checkpoint so resumed reports are reconstructible
    # even if the source attempt is still incomplete after an interruption.
    if metadata:
        source_history = Path(parent["path"]) / "grpo/metrics-prefix.jsonl"
        if not source_history.is_file():
            raise ValueError("GRPO checkpoint metric prefix missing")
        metrics_path.write_bytes(source_history.read_bytes())
    def update(current, metrics):
        metrics.update(telemetry.finish())
        metrics["adapter_gradient_coverage"] = coverage.last_gradients
        metrics["objective_version"] = "offline_grpo_k3_v1"
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        if (current["global_step"] % config.training.checkpoint_every_updates == 0
                or current["global_step"] == stop):
            save_checkpoint(model, optimizer, scheduler, current,
                            directory / "checkpoints" / f"step-{current['global_step']:06d}",
                            contract, adapter_identity, extra_files={**extra_files,
                            "grpo/metrics-prefix.jsonl": metrics_path})
        print(json.dumps({"attempt_id": config.run.attempt_id, **metrics}, allow_nan=False), flush=True)
    try:
        with attention_context(config.training.memory.attention_policy):
            state = train_grpo(model, groups, objective, optimizer, scheduler,
                group_batch_size=config.grpo.group_batch_size, max_epochs=config.grpo.max_epochs,
                max_updates=stop, grad_clip=config.training.grad_clip, state=state,
                on_update=update, on_window_start=telemetry.start, on_gradients=coverage.gradients)
        if state["global_step"] != stop:
            raise ValueError("GRPO dataset exhausted before its planned update boundary")
        final_adapter = adapter_fingerprint(model)
        coverage_report = coverage.finish(initial_adapter, final_adapter)
    finally:
        coverage.close(); telemetry.close()
    frozen_after = frozen_checksums(model)
    if frozen_before != frozen_after:
        raise ValueError("Frozen base changed during GRPO training")
    if state["global_step"] == proof["planned_updates"] and state["group_cursor"] != proof["group_count"]:
        raise ValueError("Completed epoch cursor differs from frozen group count")
    write_json(directory / "final-adapter.json", final_adapter)
    export_adapter(model, directory / "adapter", adapter_identity)
    status = "completed" if state["group_cursor"] == proof["group_count"] else "paused"
    validation = {"purpose": "offline_grpo_k3_v1", "state": state, "input": proof,
        "frozen_base_unchanged": True, "adapter_changed": initial_adapter != final_adapter,
        "coverage": coverage_report, "initial_state": initial_state, "reference": reference_proof}
    write_json(directory / "validation.json", validation)
    from RL.grpo.metrics import read_metric_history
    history = read_metric_history(metrics_path)
    report = {"status": status, "input": proof, "reference": reference_proof,
              "state": state, "updates": history}
    write_json(directory / "grpo-report.json", report)
    (directory / "report.md").write_text(
        f"# Offline GRPO: {config.run.attempt_id}\n\nStatus: {status}. "
        f"Groups: {proof['group_count']}/{proof['total_groups']}; coverage: {proof['group_coverage']:.3f}. "
        f"Branches: {proof['branch_count']}. Consumed cursor: {state['group_cursor']}.\n\n"
        f"Exclusions: {proof['exclusion_reasons']}. Advantage: {proof['advantage_distribution']}.\n\n"
        "Ratio quantiles, clip fractions, K3 penalties and positive/negative logprob changes "
        "are recorded per update in grpo-report.json and metrics.jsonl. "
        "K3 measures the historical completion penalty; it is not an unbiased current-policy KL.\n",
        encoding="utf-8")
    if parent:
        if sha256(Path(parent["owner"]) / "receipt.json") != parent["receipt_sha256"]:
            raise ValueError("Source checkpoint changed during training")
        verify_adapter(parent["adapter"], adapter_identity)
    attempt.update(status=status, ended_at=now(), final_state=state)
    write_json(directory / "attempt.json", attempt)
    commit_receipt(directory, "lora-attempt-v1", status=status, attempt_id=config.run.attempt_id,
                   contract_sha256=hash_json(contract), **receipt_metadata(contract, state))
    return {"attempt_id": config.run.attempt_id, "status": status, "path": str(directory),
            "state": state, "group_count": proof["group_count"], "branch_count": proof["branch_count"]}

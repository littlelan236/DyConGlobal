"""The public service used by both dycon lora-train and Python API clients."""

import json
import os
import platform
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

from core.io import hash_json, now, read_json, sha256, write_json
from core.locking import file_lock
from RL.artifacts import base_identity, commit_receipt, dependency_versions, source_identity, resume_contract_matches
from RL.config import attempt_directory, load, schedule_contract, memory_contract
from RL.resources import memory_plan
from RL.supervision import describe_policy


def _source(config):
    """Validate source receipts before loading model weights, with no Torch import."""
    mode = config.run.start.mode
    if mode == "fresh":
        return None
    # checkpoints imports Torch; source validation for dry-run uses receipt/JSON only instead.
    from RL.artifacts import read_receipt, verify_receipt

    path = Path(config.run.start.checkpoint)
    resuming = mode in {"resume", "append_resume"}
    owner = path if resuming else path.parent
    if resuming and not (path / "training-state.pt").is_file():
        raise ValueError("resume needs a complete checkpoint; choose warm_start for adapter-only output")
    receipt = (verify_receipt if config.preflight.verify_checkpoint else read_receipt)(
        owner, "lora-checkpoint-v1" if resuming else None)
    if receipt["kind"] not in {"lora-checkpoint-v1", "lora-attempt-v1"}:
        raise ValueError("Unsupported source adapter owner")
    adapter = path / "adapter" if resuming else path
    if adapter.name != "adapter":
        raise ValueError("warm_start requires an explicitly named adapter/ export")
    identity = read_json(adapter / "identity.json")
    if not resuming and (owner / "state.json").exists():
        from RL.grpo.stream_resume import execution_mode
        execution_mode(config, read_json(owner / "state.json")["contract"])
    if resuming:
        from RL.grpo.stream_resume import execution_mode
        contract = read_json(path / "state.json")["contract"]
        execution_mode(config, contract)
        if "streaming" in contract:
            from RL.checkpoints import verify_checkpoint
            verify_checkpoint(path)
    return {"mode": mode, "path": str(path), "owner": str(owner),
            "receipt_sha256": sha256(owner / "receipt.json"),
            "adapter_identity_sha256": sha256(adapter / "identity.json"), "identity": identity,
            "adapter": str(adapter)}


def _preflight(config, root, parent):
    from RL.parallel import execution_contract
    if parent and config.run.start.mode in {"resume", "append_resume"}:
        from RL.topology import validate_resume as validate_topology
        validate_topology(config, read_json(Path(parent["path"]) / "state.json"))
    if config.preflight.check_resources:
        memory_plan(config)
    base = base_identity(config.model.path, root, config.preflight.verify_identity)
    prepared_input = None
    if config.algorithm is not None and config.prepared_data is not None and parent is None:
        from RL.grpo.prepared_service import for_training
        from RL.grpo.contract import validate_input
        prepared_input, _, _, _ = for_training(config, root, None)
        validate_input(base, prepared_input.proof)
    if config.algorithm is not None and parent and config.run.start.mode in {"resume", "append_resume"}:
        metadata = read_json(Path(parent["path"]) / "state.json")
        from RL.grpo.stream_resume import execution_mode, resolve_checkpoint
        if execution_mode(config, metadata["contract"]) == "streaming":
            from RL.grpo.streaming import StreamingGroups, bind_source
            from RL.grpo.contract import validate_input, validate_resume
            resolved = resolve_checkpoint(parent["path"], metadata["contract"])
            groups = StreamingGroups(config, resolved["plan"])
            validate_input(base, groups.proof)
            validate_resume(config, groups.proof, metadata)
            groups.validate_state(metadata["state"])
            binding = metadata["contract"]["grpo"]["probability_binding"]
            bind_source(config, groups, resolved["evidence"], base,
                        binding["training_initial_adapter"],
                        metadata["contract"]["grpo"]["reference_identity"],
                        saved_binding=binding, primary=False)
            groups.source.verify_unchanged()
            return {**base, "grpo_dataset": groups.proof,
                    "streaming_resume": {"initial_state": metadata["state"],
                        "metadata_validation": "verified", "tensor_validation": resolved["tensor_validation"],
                        "origin": resolved["origin"]}}
        if "prepared" in metadata["contract"] or config.run.start.mode == "append_resume":
            from RL.grpo.prepared_service import for_training
            from RL.grpo.contract import validate_input
            reader, _, _, _ = for_training(config, root, parent)
            validate_input(base, reader.proof)
            return {**base, "grpo_dataset":reader.proof}
    from RL.grpo.execution import enabled
    if (enabled(config) and config.runtime.data_preparation_async and prepared_input is None
            and not config.preflight.verify_probabilities):
        from RL.grpo.streaming import create_plan
        proof = create_plan(config)["proof"]
        from RL.grpo.contract import validate_input
        validate_input(base, proof)
        return {**base, "grpo_dataset": proof}
    needs_dataset = (enabled(config) or config.preflight.validate_dataset
        or (config.preflight.verify_probabilities and config.run.mode == "dry-run"
            and config.grpo is not None and config.grpo.reference_cache.source != "model_forward")
        or (parent is not None and config.preflight.verify_checkpoint))
    if not needs_dataset:
        return {**base, "grpo_dataset":prepared_input.proof} if prepared_input is not None else base
    if config.algorithm is not None:
        from RL.grpo.data import prepare
        from RL.grpo.contract import validate_input, validate_resume, validate_reference_files
        raw, proof = ((prepared_input, prepared_input.proof) if prepared_input is not None else
                      prepare(config, shuffle_algorithm="stable_random_v1"
                              if parent and config.run.start.mode == "resume" else None))
        if config.preflight.verify_identity:
            validate_input(base, proof)
        if (config.preflight.verify_probabilities and config.run.mode == "dry-run" and config.run.start.mode != "resume"
                and config.grpo.reference_cache.source != "model_forward"):
            from RL.dataset.probability import open_probability_stream, synthetic_branch
            from RL.grpo.contract import validate_scoring_identity
            source = open_probability_stream(config.grpo.reference_cache, config.algorithm, proof)
            scoring = source.scoring_identity
            if config.run.start.mode == "warm_start" and scoring.get("adapter_fingerprint") is None:
                raise ValueError("Warm-start external probabilities require the scoring adapter")
            adapter = ({"kind": "base_equivalent"} if config.run.start.mode == "fresh" else
                       {"kind": "adapter", "fingerprint": scoring["adapter_fingerprint"]})
            # Actual PEFT state/base-equivalence and warm-start adapter tensor identity
            # are checked after loading, before optimizer creation in import_reference.
            validate_scoring_identity(scoring, base=base, proof=proof,
                training_dtype=config.model.dtype,
                policy=config.grpo.reference_cache.dtype_compatibility, adapter=adapter)
            for group in raw:
                for row in group["members"]:
                    source.values(synthetic_branch(row, proof["tokenizer_fingerprint"]) if proof["synthetic"] else row)
            source.verify_unchanged()
        if parent and config.preflight.verify_checkpoint:
            identity = parent["identity"]["contract"]
            if identity["base_fingerprint"] != base["fingerprint"] or identity["lora"] != config.lora.model_dump():
                raise ValueError("Source adapter base/tokenizer/LoRA identity mismatch")
            if config.run.start.mode == "resume":
                metadata = read_json(Path(parent["path"]) / "state.json")
                validate_resume(config, proof, metadata)
                validate_reference_files(parent["path"], metadata["contract"])
        return {**base, "grpo_dataset": proof}
    fixture = Path(__file__).parent / "fixtures/causal-text-smoke-v1.json"
    if parent and config.preflight.verify_checkpoint:
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
    from RL.allocator import configure_cuda_allocator
    allocator = configure_cuda_allocator(config.runtime.cuda_allocator, apply=config.run.mode != "dry-run")
    directory = attempt_directory(config)
    lock = root / "state/artifact-locks/lora-attempt" / (config.run.attempt_id + ".lock")
    with file_lock(lock, nonblocking=True):
        if directory.exists():
            raise FileExistsError(f"Attempt ID already exists, including failed attempts: {directory}")
        parent = _source(config)
        if config.run.mode == "dry-run":
            base = _preflight(config, root, parent)
            from RL.topology import enabled as portable_enabled, execution_record
            return {"mode": "dry-run", "purpose": "engineering_smoke", "status": "validated",
                    "output": str(directory), "base_fingerprint": base["fingerprint"],
                    "fixture_sha256": sha256(Path(__file__).parent / "fixtures/causal-text-smoke-v1.json"),
                    "resolved_config": config.model_dump(), "source": parent and parent["path"],
                    "preflight": config.preflight.model_dump(),
                    "memory_plan": memory_plan(config),
                    "cuda_allocator": allocator,
                    "supervision": describe_policy(config.training.supervision),
                    "algorithm": config.algorithm.name if config.algorithm else "causal_ce",
                    "grpo": config.grpo.model_dump() if config.grpo else None,
                    "dataset": base.get("grpo_dataset"),
                    "streaming_resume": base.get("streaming_resume"),
                    "physical_execution": execution_record(config) if portable_enabled(config) else None,
                    "training_rng_validation": "pending_model_and_update" if portable_enabled(config) else None,
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
            write_json(directory / "cuda-allocator-config.json", allocator)
            try:
                base = _preflight(config, root, parent)
                write_json(directory / "model-identity.json", base)
                from RL.parallel import devices, launch
                selected = devices(config)
                with ExitStack() as locks:
                    for gpu in sorted(selected, key=str):
                        gpu_lock = root / "state/artifact-locks/lora-gpu" / (str(gpu) + ".lock")
                        locks.enter_context(file_lock(gpu_lock, nonblocking=True))
                    if len(selected) > 1:
                        result = launch(config, root, directory, base, parent, attempt)
                    else:
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
    gpu = str(config.runtime.gpu if config.runtime.gpu is not None else config.runtime.gpus[0])
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
            raise RuntimeError("GPU selection must precede CUDA initialization; call the service in a fresh process")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # This must happen before importing torch or touching CUDA.  The helper is
    # torch-free so it also applies in spawned data-parallel workers.
    from RL.allocator import configure_cuda_allocator
    from RL.config import CudaAllocator
    allocator = configure_cuda_allocator(getattr(config.runtime, "cuda_allocator", CudaAllocator()))
    if gpu.startswith("MIG-"):
        from RL.devices import inspect_mig
        result = inspect_mig(gpu, check_idle=getattr(getattr(config, "preflight", None), "check_resources", True))
        result["cuda_allocator"] = allocator
        return result
    if not config.preflight.check_resources:
        return {"physical_gpu": int(gpu), "resource_precheck": "disabled", "cuda_allocator": allocator}
    query = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=index,uuid,name,memory.used,memory.free",
                                     "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
    fields = [part.strip() for part in query.split(",")]
    if len(fields) != 5 or int(fields[3]) > 512:
        raise RuntimeError(f"Configured physical GPU {gpu} is not idle: {query}")
    return {"physical_gpu": int(gpu), "uuid": fields[1], "name": fields[2],
            "initial_used_MiB": int(fields[3]), "initial_free_MiB": int(fields[4]),
            "cuda_allocator": allocator}


def _train(config, root, directory, base, parent, attempt):
    from RL.allocator import capture_allocator_warnings
    with capture_allocator_warnings() as allocator_warnings:
        return _train_impl(config, root, directory, base, parent, attempt,
                           allocator_warnings=allocator_warnings)


def _train_impl(config, root, directory, base, parent, attempt, *, allocator_warnings):
    from RL.parallel import _device_info
    gpu = _device_info or _select_gpu(config)  # Before importing the training stack.
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
    from RL.memory import (LossBackend, MemoryController, attention_context,
                           configure_checkpointing, install_mlp_backend)
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
    if config.model.dtype == "float32":
        tolerances = {"adapter_atol": 2e-6, "adapter_rtol": 2e-4, "logits_atol": 2e-5,
                      "logits_rtol": 2e-5, "loss_atol": 2e-6, "loss_rtol": 2e-5}
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # bf16 reduced-precision GEMM reductions depend on token-block dimensions.
    # Keep reduced-precision reductions disabled by default, but allow Split-K
    # so unsupported cuBLASLt shapes can fall back to cuBLAS (e.g. n=1).
    torch.backends.cuda.preferred_blas_library("cublaslt")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        config.runtime.bf16_allow_reduced_precision_reduction,
        config.runtime.bf16_allow_split_k,
    )
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
    backend = torch.cuda.memory.get_allocator_backend()
    environment["cuda_allocator"] = {**gpu.get("cuda_allocator", {}), "backend_actual": backend,
                                     "logical_device": 0, "support_warnings": allocator_warnings,
                                     "warning_capture": "Python allocator warnings captured; native stderr retained"}
    if config.runtime.cuda_allocator.expandable_segments and backend != "native":
        raise RuntimeError("Configured expandable_segments is unsupported by actual allocator " + backend)
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
    attempt.update(status="loading_model")
    write_json(directory / "attempt.json", attempt)
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
    from RL.parallel import current
    parallel = current()
    if parallel:
        parallel.synchronize_model(model)
    mlps = install_mlp_backend(model, config.training.memory.mlp_chunk_tokens)
    offloader = configure_checkpointing(model, config, plan["host_budget_bytes_per_rank"])
    # The first real allocations happen during model loading; retain warnings
    # observed there rather than claiming that a requested option is effective.
    write_json(directory / "environment.json", environment)
    context = parallel
    memory_controller = MemoryController(
        config, directory, rank=context.rank if context else 0,
        world_size=context.world_size if context else 1,
        physical_gpu=gpu.get("physical_gpu") if isinstance(gpu, dict) else None,
        offloader=offloader, device=next(model.parameters()).device)
    memory_controller.logger.write(event="environment", phase="startup", global_step=None,
        global_step_semantics="not_restored_yet", inflight_step=None, step_attempt=None,
        committed=False, cuda_allocator=environment["cuda_allocator"],
        determinism=environment["determinism"], physical_gpu=gpu.get("physical_gpu"),
        local_world_size=context.world_size if context else 1)
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
                           loading=loading, mlps=mlps, offloader=offloader,
                           memory_controller=memory_controller)
    batches, input_identity = fixture_batches(tokenizer, config.validation.max_sequence_length,
                                              config.training.supervision)
    write_json(directory / "input-identity.json", input_identity)
    write_json(directory / "selection.json", input_identity["supervision"])
    adapter_identity = adapter_contract(base, targets, config.lora)
    if parent and config.preflight.verify_checkpoint:
        verify_adapter(parent["adapter"], adapter_identity)
    contract = {"schedule": schedule_contract(config), "input": input_identity,
                "adapter": adapter_identity,
                "dependencies": source["dependencies"], "attention": environment["linear_attention"]}
    contract["determinism"] = environment["determinism"]
    contract["memory_backend"] = {"strategy": memory_contract(config.training.memory),
                                  "checkpoint_policy": config.training.checkpoint_policy,
                                  "qwen_implementation_sha256": environment["qwen_implementation_sha256"]}
    optimizer, scheduler = optimizer_and_scheduler(model, config.training)
    initial_optimizer_entries = len(optimizer.state)
    initial_adapter = adapter_fingerprint(model)
    frozen_before = frozen_checksums(model) if config.preflight.verify_model_state else None
    initial_logits = probe_logits(model, batches[0]) if config.preflight.run_diagnostics else None
    # These compact outputs are engineering evidence for a later process to compare.
    if config.preflight.run_diagnostics:
        torch.save(initial_logits, directory / "initial-probe.pt")
    write_json(directory / "initial-adapter.json", initial_adapter)
    state = {"global_step": 0, "consumed_microbatches": 0, "consumed_tokens": 0}
    if config.run.start.mode == "resume":
        state = restore_checkpoint(parent["path"], optimizer, scheduler, contract, adapter_identity,
                                   verify=config.preflight.verify_checkpoint)
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
        if config.preflight.run_diagnostics and source_owner.joinpath("final-probe.pt").is_file():
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
    frozen_after = frozen_checksums(model) if config.preflight.verify_model_state else None
    if frozen_before != frozen_after:
        raise ValueError("Frozen base tensor checksums changed")
    final_logits = probe_logits(model, batches[0])
    torch.save(final_logits, directory / "final-probe.pt")
    write_json(directory / "final-adapter.json", final_adapter)
    export_adapter(model, directory / "adapter", adapter_identity)
    status = "completed" if stop == config.training.max_updates else "paused"
    validation = {"purpose": "engineering_smoke", "state": state, "loading": loading, "input": input_identity,
                  "frozen_base_unchanged": True if config.preflight.verify_model_state else None,
                  "frozen_checksums": frozen_after,
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
                optimizer_settings, environment=None, loading=None, mlps=(), offloader=None,
                memory_controller=None):
    """Offline objective within the common attempt/model/checkpoint lifecycle."""
    from RL.parallel import current, execution_contract, follower
    parallel = current()
    from RL.topology import configure as configure_topology
    configure_topology(config, model, parallel)
    if parallel and not parallel.primary:
        return follower(config, root, directory, parent, model, targets,
                        optimizer_settings=optimizer_settings, mlps=mlps, offloader=offloader,
                        memory_controller=memory_controller)
    import shutil
    import torch

    from RL.checkpoints import adapter_contract, export_adapter, restore_checkpoint, save_checkpoint, verify_adapter
    from RL.grpo.contract import validate_input, validate_resume, validate_reference_files, receipt_metadata
    from RL.grpo.data import prepare, tensor_groups
    from RL.grpo.backend import GroupObjective
    from RL.grpo.reference import build_reference, import_reference, load_reference, diagnose_initial_current
    from RL.memory import attention_context
    from RL.modeling import adapter_fingerprint, frozen_checksums, Coverage
    from RL.profiling import UpdateTelemetry
    from RL.training import optimizer_and_scheduler, train_grpo

    attempt.update(status="preparing_data")
    write_json(directory / "attempt.json", attempt)
    preparation_started = time.perf_counter()
    from RL.grpo.stream_resume import execution_mode, resolve_checkpoint, inherit, validate_anchor, write_bundle, restore_training
    parent_metadata = (read_json(Path(parent["path"])/"state.json")
                       if parent and config.run.start.mode in {"resume", "append_resume"} else None)
    mode = execution_mode(config, parent_metadata["contract"] if parent_metadata else None)
    streaming_mode = mode == "streaming"
    legacy = config.run.start.mode == "resume" and mode == "legacy"
    reader = extension = metadata = None
    resolved_stream = None
    stream_groups = None
    if streaming_mode:
        from RL.grpo.streaming import StreamingGroups, create_plan, bind_source
        if parent_metadata is not None:
            metadata = parent_metadata
            resolved_stream = resolve_checkpoint(parent["path"], metadata["contract"])
            plan = resolved_stream["plan"]
        else:
            plan = create_plan(config)
        stream_groups = StreamingGroups(config, plan,
            rank=parallel.rank if parallel else 0,
            world_size=parallel.world_size if parallel else 1,
            progress_path=directory / "grpo/stream-progress.json")
        raw_groups = None
        proof = stream_groups.proof
        groups = stream_groups
        preparation = {"prepared_cache_hit":False, "streaming":True,
                       "source_scan_groups":proof["total_groups"],
                       "materialized_groups":0}
        reference_values = stream_groups.reference
        reference_proof = None
    elif legacy:
        raw_groups, proof = prepare(config, shuffle_algorithm="stable_random_v1")
        groups = list(tensor_groups(raw_groups))
        preparation = {"prepared_cache_hit":False, "legacy_resume":True,
                       "source_scan_groups":proof["total_groups"]}
    else:
        from RL.grpo.prepared_service import for_training
        from RL.grpo.prepared import PreparedGroups, binding
        reader, metadata, extension, preparation = for_training(config, root, parent)
        raw_groups, proof, groups = reader, reader.proof, PreparedGroups(reader)
        if reader.execution_plan:
            from RL.grpo.execution import PreparedExecution
            groups = PreparedExecution(reader, rank=parallel.rank if parallel else 0,
                                       world_size=parallel.world_size if parallel else 1)
    preparation["seconds"] = time.perf_counter() - preparation_started
    if reader is not None or streaming_mode or config.preflight.verify_identity:
        validate_input(base, proof)
    if streaming_mode and metadata is not None:
        validate_resume(config, proof, metadata)
        stream_groups.validate_state(metadata["state"])
    write_json(directory / "input-identity.json", proof)
    write_json(directory / "selection.json", {"group_ids": proof["eligible_group_ids"],
                "group_order_sha256": proof["group_order_sha256"]})
    if legacy:
        write_json(directory / "frozen-groups.json", raw_groups)
    adapter_identity = adapter_contract(base, targets, config.lora)
    if parent and config.preflight.verify_checkpoint:
        verify_adapter(parent["adapter"], adapter_identity)
    initial_adapter = adapter_fingerprint(model)
    frozen_before = frozen_checksums(model) if config.preflight.verify_model_state else None
    write_json(directory / "initial-adapter.json", initial_adapter)
    (directory / "grpo").mkdir()
    if streaming_mode:
        write_json(directory / "grpo/stream-plan.json", plan)
        attempt.update(status="preparing_reference")
        write_json(directory / "attempt.json", attempt)
        if resolved_stream is not None:
            inherit(resolved_stream, directory)
            reference_proof = metadata["contract"]["grpo"]
            validate_anchor(model, directory, metadata["contract"])
            source_binding = reference_proof["probability_binding"]
            bind_source(config, stream_groups, directory, base, source_binding["training_initial_adapter"],
                        reference_proof["reference_identity"], saved_binding=source_binding)
        else:
            export_adapter(model, directory / "grpo/reference/adapter", adapter_identity)
            write_json(directory / "grpo/anchor-tensors.json", initial_adapter)
        from RL.grpo.reference import adapter_compatibility
        if resolved_stream is None:
            anchor_hash = hash_json(initial_adapter)
            anchor_adapter = adapter_compatibility(model, start_mode=config.run.start.mode,
                                                    adapter_hash=anchor_hash)
            source_binding = bind_source(config, stream_groups, directory, base, anchor_adapter,
                                         {"model_fingerprint":base["fingerprint"],
                                          "tokenizer_fingerprint":proof["tokenizer_fingerprint"],
                                          "adapter_hash":anchor_hash,
                                          "objective_version":"offline_grpo_k3_v1"})
            digest = source_binding["source_receipt_sha256"]
            reference_proof = {"cache_schema_version":4, "reference_mode":"step_stream_v1",
                               "reference_identity":{"model_fingerprint":base["fingerprint"],
                                   "tokenizer_fingerprint":proof["tokenizer_fingerprint"],
                                   "adapter_hash":anchor_hash, "objective_version":"offline_grpo_k3_v1"},
                               "probability_binding":source_binding,
                               "reference_logprob_cache_sha256":digest,
                               "role_binding":{"ratio_denominator":digest, "k3_reference":digest}}
    environment = environment or {}
    source = source_identity(root)
    reference_started = time.perf_counter()
    attempt.update(status="preparing_reference")
    write_json(directory / "attempt.json", attempt)
    if streaming_mode:
        preparation["seconds"] = time.perf_counter() - preparation_started
        preparation["model_loading_seconds"] = loading.get("seconds") if loading else None
        write_json(directory / "grpo/reference-preparation.json", {
            "source": config.grpo.reference_cache.source,
            "mode": "step_stream",
            "seconds": preparation["seconds"],
            "queue_windows": 2})
    elif reader is not None:
        from RL.grpo.reference_artifact import ReferenceReader, bind_attempt, publish
        from RL.grpo.prepared import resolve
        parent_reference = None
        resuming = config.run.start.mode in {"resume", "append_resume"}
        if resuming:
            parent_reference = validate_reference_files(parent["path"], metadata["contract"])
            shutil.copytree(Path(parent["path"])/"grpo", directory/"grpo", dirs_exist_ok=True)
        else:
            export_adapter(model, directory/"grpo/reference/adapter", adapter_identity)
        if config.run.start.mode == "resume":
            reference_values = parent_reference
            reference_proof = metadata["contract"]["grpo"]
        else:
            anchor_hash = (parent_reference.manifest["identity"]["adapter_hash"] if parent_reference else hash_json(initial_adapter))
            snapshot = directory/"grpo/reference/adapter"
            if parent_reference is not None:
                from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict, load_peft_weights
                saved = {k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
                try:
                    set_peft_model_state_dict(model, load_peft_weights(str(snapshot), device="cpu"))
                    if hash_json(adapter_fingerprint(model)) != anchor_hash:
                        raise ValueError("Append initial-policy adapter snapshot fingerprint mismatch")
                    reference_values = publish(model, config, reader, base, anchor_hash, adapter_identity,
                                               snapshot, parent=parent_reference)
                finally:
                    set_peft_model_state_dict(model, saved)
            else:
                reference_values = publish(model, config, reader, base, anchor_hash, adapter_identity, snapshot)
            reference_proof = bind_attempt(directory, reference_values, proof)
            if (config.preflight.run_diagnostics and config.grpo.reference_cache.source != "model_forward"
                    and config.grpo.reference_cache.initial_diagnostic_branches):
                from itertools import islice
                count = config.grpo.reference_cache.initial_diagnostic_branches
                selected = list(islice(iter(groups), count))
                with attention_context(config.training.memory.attention_policy):
                    diagnostic = diagnose_initial_current(model, selected, reference_values,
                        branch_count=count, block_size=config.training.memory.loss_chunk_tokens)
                write_json(directory/"grpo/initial-current-diagnostic.json", diagnostic)
        preparation.update(reference_reused_groups=len(parent_reference.owners) if parent_reference else 0,
            reference_new_groups=0 if config.run.start.mode == "resume" else len(reader.local_ids),
            reference_seconds=time.perf_counter()-reference_started)
        preparation["model_loading_seconds"] = loading.get("seconds") if loading else None
        write_json(directory/"grpo/reference-preparation.json", {
            "source":config.grpo.reference_cache.source,
            "seconds":preparation["reference_seconds"],
            "reference_model_forward":config.grpo.reference_cache.source == "model_forward" and preparation["reference_new_groups"] > 0,
            "reused_groups":preparation["reference_reused_groups"],
            "new_groups":preparation["reference_new_groups"]})
        write_json(directory/"grpo/prepared-binding.json", {"prepared":binding(reader.path),
                   "schedule_sha256":reader.schedule["sha256"]})
    elif config.run.start.mode == "resume":
        metadata = read_json(Path(parent["path"]) / "state.json")
        if config.preflight.verify_checkpoint:
            validate_resume(config, proof, metadata)
            validate_reference_files(parent["path"], metadata["contract"])
        shutil.copytree(Path(parent["path"]) / "grpo", directory / "grpo", dirs_exist_ok=True)
        reference_proof = metadata["contract"]["grpo"]
        reference_values = load_reference(directory, groups, proof, reference_proof,
                                          verify=config.preflight.verify_checkpoint)
    elif not streaming_mode:
        export_adapter(model, directory / "grpo/reference/adapter", adapter_identity)
        if config.grpo.reference_cache.source == "model_forward":
            with attention_context(config.training.memory.attention_policy):
                reference_values, reference_proof = build_reference(model, groups, directory, proof,
                    base, hash_json(initial_adapter), config.training.memory.loss_chunk_tokens)
        else:
            reference_values, reference_proof = import_reference(model, groups, raw_groups,
                directory, proof, base, hash_json(initial_adapter), config)
        write_json(directory / "grpo/reference-preparation.json", {
            "source": config.grpo.reference_cache.source,
            "seconds": time.perf_counter() - reference_started,
            "reference_model_forward": config.grpo.reference_cache.source == "model_forward"})
        if (config.preflight.run_diagnostics and config.grpo.reference_cache.source != "model_forward"
                and config.grpo.reference_cache.initial_diagnostic_branches):
            with attention_context(config.training.memory.attention_policy):
                diagnostic = diagnose_initial_current(model, groups, reference_values,
                    branch_count=config.grpo.reference_cache.initial_diagnostic_branches,
                    block_size=config.training.memory.loss_chunk_tokens)
            write_json(directory / "grpo/initial-current-diagnostic.json", diagnostic)
    contract = {"schedule": schedule_contract(config), "input": proof, "grpo": reference_proof,
                "adapter": adapter_identity,
                "dependencies": source["dependencies"], "attention": environment.get("linear_attention"),
                "determinism": environment.get("determinism"),
                "qwen_implementation_sha256": environment.get("qwen_implementation_sha256")}
    if reader is not None:
        contract.update(prepared=binding(reader.path), prepared_schedule_id=reader.schedule["schedule_id"],
                        prepared_schedule_sha256=reader.schedule["sha256"])
        if extension:
            contract["extension"] = extension
    if (not streaming_mode) and (reader is not None or config.preflight.verify_checkpoint or config.preflight.verify_probabilities):
        validate_reference_files(directory, contract)
    if config.grpo.prefix_reuse.mode == "boundary_grad":
        from RL.grpo.prefix import EXECUTION_CONTRACT
        from RL.qwen_boundary import validate_model
        if config.preflight.verify_model_state:
            validate_model(model)
        contract["prefix_execution"] = dict(EXECUTION_CONTRACT)
    if streaming_mode:
        contract["streaming"] = {"version": 1, "window_queue": 2,
                                  "assignment": "window_ordinal_mod_world_size",
                                  "plan_sha256": sha256(directory / "grpo/stream-plan.json")}
    if parallel:
        if reader is None and not streaming_mode:
            raise ValueError("Distributed GRPO requires prepared indexed groups; migrate the legacy attempt first")
        contract["parallel"] = execution_contract(config)
    from RL.topology import bind as bind_topology
    bind_topology(config, metadata, contract, directory)
    optimizer, scheduler = optimizer_and_scheduler(model, optimizer_settings)
    if metadata is not None:
        if config.run.start.mode == "append_resume":
            allowed = {"schedule", "input", "prepared", "prepared_schedule_id", "prepared_schedule_sha256", "grpo", "extension", "source_fingerprint"}
            for key in set(metadata["contract"]) | set(contract):
                if key not in allowed and metadata["contract"].get(key) != contract.get(key):
                    raise ValueError("Append checkpoint immutable training contract mismatch: " + key)
        elif not resume_contract_matches(metadata["contract"], contract):
            write_json(directory / "grpo/resume-contract-diff.json", {
                "differing_fields": [key for key in sorted(set(metadata["contract"]) | set(contract))
                                     if key != "source_fingerprint"
                                     and metadata["contract"].get(key) != contract.get(key)],
                "expected_contract_sha256": hash_json(metadata["contract"]),
                "actual_contract_sha256": hash_json(contract)})
            raise ValueError("GRPO checkpoint contract differs; exact resume rejected")
        if streaming_mode:
            state = restore_training(parent["path"], optimizer, scheduler, contract, groups=stream_groups)
        else:
            state = restore_checkpoint(parent["path"], optimizer, scheduler,
                                       metadata["contract"] if config.run.start.mode == "append_resume" else contract,
                                       adapter_identity, verify=reader is not None or config.preflight.verify_checkpoint)
        if reader is not None:
            state.update(reader.cursor(state.get("consumed_execution_chunks", state["group_cursor"])))
    else:
        state = {"global_step": 0, "group_cursor": 0, "consumed_groups": 0,
                 "consumed_branches": 0, "consumed_tokens": 0}
        if reader is not None:
            state.update(reader.cursor(0))
        elif streaming_mode:
            state.update(stream_groups.cursor(0))
    initial_state = dict(state)
    if parallel:
        parallel.broadcast({"contract": contract, "state": state, "directory": str(directory),
                            "streaming": streaming_mode,
                            "stop": config.validation.stop_after_update or proof["planned_updates"]})
    coverage = Coverage(model, targets)
    objective = GroupObjective(reference_values, config.grpo, config.training.memory.loss_chunk_tokens,
                              offloader=offloader)
    telemetry = UpdateTelemetry(model, objective, mlps, offloader, directory,
                    config.training.memory.profile_first_update,
                    config.training.memory.attention_policy == "flash_only")
    metrics_path = directory / "metrics.jsonl"
    metrics_path.touch(exist_ok=False)
    attempt.update(status="training", contract=contract, initial_state=initial_state, loading=loading)
    write_json(directory / "attempt.json", attempt)
    write_json(directory / "grpo-contract.json", contract)
    stop = config.validation.stop_after_update or proof["planned_updates"]
    extra_files = (write_bundle(directory, contract, origin=resolved_stream["origin"] if resolved_stream else None)
                   if streaming_mode else
                   {p.relative_to(directory).as_posix(): p for p in (directory / "grpo").rglob("*") if p.is_file()})
    if config.runtime.topology_policy == "preserve_updates":
        extra_files["topology-transition.json"] = directory / "topology-transition.json"
    # Metric prefix travels with a checkpoint so resumed reports are reconstructible
    # even if the source attempt is still incomplete after an interruption.
    if metadata:
        source_history = Path(parent["path"]) / "grpo/metrics-prefix.jsonl"
        if not source_history.is_file():
            raise ValueError("GRPO checkpoint metric prefix missing")
        metrics_path.write_bytes(source_history.read_bytes())
    from RL.observability import LiveTelemetry
    live = LiveTelemetry({"project_root":str(root)}, config.run.attempt_id, config.runtime.output_root)
    def start_attempt():
        coverage.begin_attempt()
        telemetry.start()

    def abort_attempt():
        telemetry.abort()
        coverage.abort_attempt()
    def update(current, metrics):
        metrics.update(metrics_schema_version=3 if proof.get("execution_plan") else 2,
                       observed_at=now(), attempt_id=config.run.attempt_id)
        metrics.update(telemetry.finish())
        metrics["adapter_gradient_coverage"] = coverage.last_gradients
        metrics["objective_version"] = "offline_grpo_k3_v1"
        metrics["reference_cache_usage"] = "shared_anchor"
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        live.emit("update_committed", last_completed_update=current["global_step"], inflight_update=None,
                  committed_consumed_groups=current["group_cursor"], completed_groups_in_update=0,
                  **({key: current[key] for key in ("consumed_execution_chunks", "completed_parent_groups",
                       "distinct_parent_groups_seen")} if proof.get("execution_plan") else {}))
        if (current["global_step"] % config.training.checkpoint_every_updates == 0
                or current["global_step"] == stop):
            live.emit("checkpoint")
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
                on_update=update, on_window_start=start_attempt, on_gradients=coverage.gradients,
                on_stage=live.emit, parallel=parallel, memory_controller=memory_controller,
                on_attempt_abort=abort_attempt)
        if state["global_step"] != stop:
            raise ValueError("GRPO dataset exhausted before its planned update boundary")
        final_adapter = adapter_fingerprint(model)
        coverage_report = coverage.finish(initial_adapter, final_adapter)
    finally:
        live.close()
        coverage.close(); telemetry.close()
        if stream_groups is not None:
            stream_groups.close()
    frozen_after = frozen_checksums(model) if config.preflight.verify_model_state else None
    if frozen_before != frozen_after:
        raise ValueError("Frozen base changed during GRPO training")
    if proof.get("execution_plan"):
        from RL.grpo.execution import validate_state
        validate_state(proof["execution_plan"], state)
    if state["global_step"] == proof["planned_updates"] and state["group_cursor"] != proof["group_count"]:
        raise ValueError("Completed epoch cursor differs from frozen group count")
    write_json(directory / "final-adapter.json", final_adapter)
    live.emit("finalizing", inflight_update=None)
    export_adapter(model, directory / "adapter", adapter_identity)
    status = "completed" if state["group_cursor"] == proof["group_count"] else "paused"
    validation = {"purpose": "offline_grpo_k3_v1", "state": state, "input": proof,
        "frozen_base_unchanged": True if config.preflight.verify_model_state else None,
        "adapter_changed": initial_adapter != final_adapter,
        "coverage": coverage_report, "initial_state": initial_state, "reference": reference_proof}
    write_json(directory / "validation.json", validation)
    from RL.grpo.metrics import read_metric_history
    history = read_metric_history(metrics_path)
    report = {"status": status, "input": proof, "reference": reference_proof,
              "state": state, "updates": history}
    if reader is not None or streaming_mode:
        report["preparation"] = preparation
        if extension:
            report["extension"] = extension
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
    if parallel:
        states = parallel.gather(state)
        if any(item != state for item in states):
            raise ValueError("Distributed final cursors differ")
    commit_receipt(directory, "lora-attempt-v1", status=status, attempt_id=config.run.attempt_id,
                   contract_sha256=hash_json(contract), **receipt_metadata(contract, state))
    return {"attempt_id": config.run.attempt_id, "status": status, "path": str(directory),
            "state": state, "group_count": proof["group_count"], "branch_count": proof["branch_count"]}

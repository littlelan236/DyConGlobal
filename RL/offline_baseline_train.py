"""Baseline-only offline trace training through the shared trajectory objective."""
import time
import traceback
from pathlib import Path

from core.io import read_json, write_json, hash_json


def _compute_attempt(accumulate, state, parallel, trace, *, retry, rng):
    # Returning from this frame drops exception/graph references before
    # rollback drains the pinned pool or empties the CUDA allocator.
    from RL.memory import is_cuda_oom
    from RL.training import restore_rng

    try:
        if retry:
            restore_rng(rng)
        result = accumulate(state, parallel, oom_replay=retry)
        return result, {"state": "OK"}
    except Exception as exc:
        return None, {"state": "CUDA_OOM" if is_cuda_oom(exc) else "FATAL",
                      "phase": trace.phase, "rank": parallel.rank if parallel else 0,
                      "error": str(exc)[:8192], "traceback": traceback.format_exc()[-8192:]}


def _compute_update(model, optimizer, accumulate, state, parallel, memory, trace):
    from RL.training import capture_rng, _rollback_grpo

    retry_limit = memory.max_retries if memory else 0
    rng = capture_rng() if retry_limit else None
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()} if retry_limit else None
    step = state["global_step"] + 1
    optimizer.zero_grad(set_to_none=True)
    if memory and memory.offloader:
        memory.offloader.begin_window()
        memory.offloader.set_enabled(memory.mode != "none")

    for retry in range(retry_limit + 1):
        if memory:
            memory.last_decision = {"mode": memory.mode}
            if retry:
                memory.retry((), global_step=step - 1, attempt_id=retry)
                memory.last_decision["attention"] = "chunked_causal_sdpa_for_long_traces"
            memory.begin_attempt((), global_step=step - 1, attempt_id=retry)
        trace("baseline_compute", inflight_update=step, oom_retries=retry)
        result, status = _compute_attempt(accumulate, state, parallel, trace, retry=bool(retry), rng=rng)
        if memory:
            if status["state"] == "CUDA_OOM":
                memory.oom(status["error"], phase=status["phase"])
            memory.pause()
        cleanup = None
        if status["state"] != "OK":
            # Release the failed rank's admission lease before waiting for peers
            # which may need that same host capacity to complete their compute.
            cleanup = _rollback_grpo(model, None, optimizer, buffers, None, memory)
        statuses = parallel.control_step(step, retry, "baseline_compute_result", status) if parallel else [status]
        if memory:
            memory.result(statuses)
        failures = [peer for peer in statuses if peer["state"] != "OK"]
        if not failures:
            return (*result, retry)
        del result
        if memory:
            memory.cleanup(before=True, statuses=statuses)
        if cleanup is None:
            cleanup = _rollback_grpo(model, None, optimizer, buffers, None, memory)
        cleanups = parallel.control_step(step, retry, "baseline_cleanup", cleanup) if parallel else [cleanup]
        if memory:
            memory.cleanup(before=False, statuses=cleanups)
        fatals = [peer for peer in cleanups + failures if peer["state"] == "FATAL"]
        if fatals or retry == retry_limit:
            detail = (fatals or failures)[0]
            raise RuntimeError(f"Baseline step {step} failed on attempt {retry}: {detail}")

    raise AssertionError("Baseline update retry loop exhausted unexpectedly")


def extension_restore_contract(saved, expected, state):
    """Permit only explicit constant-LR baseline epoch extensions."""
    import copy
    if saved.get("objective", {}).get("kind") != "offline_baseline_prefix_trace_v1":
        raise ValueError("Epoch extension requires a baseline checkpoint")
    old, new = saved["schedule"], expected["schedule"]
    if not (new["epochs"] > old["epochs"] and new["max_updates"] > old["max_updates"]
            and state["global_step"] <= old["max_updates"]):
        raise ValueError("Epoch extension must increase epochs and total updates")
    if saved["training"]["scheduler"] != "constant" or expected["training"]["scheduler"] != "constant":
        raise ValueError("Epoch extension currently requires constant scheduler")
    normalized = copy.deepcopy(expected)
    normalized["schedule"]["epochs"] = old["epochs"]
    normalized["schedule"]["max_updates"] = old["max_updates"]
    normalized["training"]["max_updates"] = saved["training"]["max_updates"]
    for section in ("input",):
        normalized[section]["config"]["epochs"] = saved[section]["config"]["epochs"]
    normalized["objective"]["settings"]["epochs"] = saved["objective"]["settings"]["epochs"]
    if normalized != saved:
        raise ValueError("Epoch extension may change only epochs and total updates")
    return saved


def execute(config, directory, base, parent, attempt, model, targets, memory):
    import torch
    from RL.key_steps import enabled
    if enabled(config):
        from RL.key_steps import schedule, binding, build_updater, OBJECTIVE
        settings = config.key_steps
        contract_file = "key-step-contract.json"
    else:
        from RL.offline_baseline import schedule, binding, build_updater
        settings = config.offline_baseline
        OBJECTIVE = "offline_baseline_prefix_trace_v1"
        contract_file = "baseline-contract.json"
    from RL.parallel import current
    from RL.training import optimizer_and_scheduler
    from RL.checkpoints import adapter_contract, export_adapter, save_checkpoint, restore_checkpoint
    from RL.memory import attention_context
    from RL.artifacts import commit_receipt

    plan = schedule(config)
    parallel = current()
    primary = parallel is None or parallel.primary
    shared = Path(directory) if primary else Path(directory).parents[1]
    identity = adapter_contract(base, targets, config.lora)
    manifest = read_json(Path(settings.dataset) / "manifest.json")
    tokenizers = ({m["tokenizer"] for entry in manifest["entries"] for m in entry["members"]}
                  if enabled(config) else {entry["tokenizer"] for entry in manifest["entries"]})
    if len(tokenizers) != 1:
        raise ValueError("Baseline dataset mixes tokenizers")
    source_binding = binding(settings, base["fingerprint"], next(iter(tokenizers)))
    objective_settings = dict(source_binding["config"])
    contract = {"schedule": plan, "input": source_binding, "adapter": identity,
                "objective": {"kind": OBJECTIVE,
                              "settings": objective_settings},
                "training": config.training.model_dump()}
    if enabled(config):
        from RL.parallel import execution_contract
        contract["execution"] = execution_contract(config) if hasattr(config,"runtime") else {"world_size":parallel.world_size if parallel else 1}
        contract["model"] = config.model.model_dump(exclude={"path"}) if hasattr(config.model,"model_dump") else vars(config.model)
        contract["seed"] = getattr(config.run,"seed",None)
    from RL.policy_reference import computation
    if computation(config) != "legacy":
        contract["objective"]["kl_reference_computation"] = computation(config)
    optimizer, scheduler = optimizer_and_scheduler(model, config.training)
    state = {"global_step": 0, "baseline_cursor": {}, "consumed_tokens": 0}
    if enabled(config):
        state.update(consumed_groups=0,consumed_branches=0)
    if parent and config.run.start.mode == "resume":
        # The launcher verified the receipt before spawning the model replicas.
        restore_contract = contract
        if config.run.start.extend_baseline_epochs:
            metadata = read_json(Path(parent["path"]) / "state.json")
            restore_contract = extension_restore_contract(metadata["contract"], contract, metadata["state"])
        state = restore_checkpoint(parent["path"], optimizer, scheduler, restore_contract, identity, verify=False)
    if primary:
        if settings.kl_beta:
            reference = shared / "reference" / "adapter"
            if parent and config.run.start.mode == "resume":
                import shutil
                shutil.copytree(Path(parent["path"]) / "baseline/reference/adapter", reference)
            else:
                export_adapter(model, reference, identity)
        write_json(shared / contract_file, contract)
    if parallel:
        import torch.distributed as dist
        dist.barrier()
    # The shared updater uses this named directory for its immutable KL reference.
    updater_contract = {"grpo": {"reference_identity": {
        "model_fingerprint": base["fingerprint"], "tokenizer_fingerprint": next(iter(tokenizers))}},
        "offline_baseline": source_binding}
    if primary and settings.kl_beta:
        import shutil
        shutil.copytree(shared / "reference/adapter", shared / "grpo/reference/adapter")
    if parallel:
        dist.barrier()
    from RL.trace_logging import TraceLogger
    trace = TraceLogger(directory, parallel.rank if parallel else 0, memory)
    accumulate = build_updater(config, model, shared, updater_contract, optimizer, scheduler,
                               memory_controller=memory, on_stage=trace)
    parameters = [p for p in model.parameters() if p.requires_grad]
    stop = config.validation.stop_after_update or plan["max_updates"]
    attempt.update(status="training", initial_state=dict(state))
    if primary:
        write_json(shared / "attempt.json", attempt)
    model.train()
    with attention_context(config.training.memory.attention_policy):
        while state["global_step"] < stop:
            started = time.monotonic()
            metrics, cursor, _, retries = _compute_update(
                model, optimizer, accumulate, state, parallel, memory, trace)
            trace("gradient_reduce")
            if parallel:
                parallel.reduce_gradients(parameters)
            trace("optimizer_update")
            norm = torch.nn.utils.clip_grad_norm_(parameters, config.training.grad_clip,
                                                  error_if_nonfinite=True)
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state.update(global_step=state["global_step"] + 1, baseline_cursor=cursor)
            if enabled(config):
                groups=len(metrics["advantage"]["parents"])
                state["consumed_groups"]+=groups
                state["consumed_branches"]+=8*groups
            trace("replica_check")
            if parallel:
                peers = parallel.gather(metrics)
                parallel.after_step(model)
            else:
                peers = [metrics]
            state["consumed_tokens"] += sum(m.get("token_count", 0) for m in peers)
            if memory:
                memory.safe_boundary()
                memory.committed(global_step=state["global_step"], window=(), attempt_id=retries)
            if primary:
                result = {"step": state["global_step"], "global_step": state["global_step"],
                          "oom_retries": retries, "loss": sum(m.get("loss", 0.) for m in peers),
                          "gradient_norm": float(norm), "learning_rate": lr,
                          "update_seconds": time.monotonic() - started,
                          "consumed_tokens": state["consumed_tokens"],
                          **({"consumed_groups":state["consumed_groups"],"consumed_branches":state["consumed_branches"]} if enabled(config) else {}),
                          ("key_step_advantage" if enabled(config) else "baseline_advantage"): metrics["advantage"]}
                import json
                with (shared / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(result) + "\n")
                attempt.update(final_state=dict(state), global_step=state["global_step"])
                write_json(shared / "attempt.json", attempt)
                if state["global_step"] % config.training.checkpoint_every_updates == 0 or state["global_step"] == stop:
                    extra = {"baseline/reference/adapter/" + str(p.relative_to(shared / "reference/adapter")): p
                             for p in (shared / "reference/adapter").rglob("*") if p.is_file()}
                    trace("checkpoint_save")
                    save_checkpoint(model, optimizer, scheduler, state,
                                    shared / "checkpoints" / f"step-{state['global_step']:06d}",
                                    contract, identity, extra_files=extra)
                print(json.dumps(result), flush=True)
            trace("update_done")
            if parallel:
                dist.barrier()
    status = "completed" if stop == plan["max_updates"] else "paused"
    if primary:
        export_adapter(model, shared / "adapter", identity)
        attempt.update(status=status, final_state=state)
        write_json(shared / "attempt.json", attempt)
        if enabled(config):
            from RL.key_steps import finalize_attempt
            finalize_attempt(shared, contract, state, status, config, attempt)
        commit_receipt(shared, "lora-attempt-v1", status=status, attempt_id=config.run.attempt_id,
                       contract_sha256=hash_json(contract))
    if memory:
        memory.close()
    return {"attempt_id": config.run.attempt_id, "status": status, "path": str(shared), "state": state}

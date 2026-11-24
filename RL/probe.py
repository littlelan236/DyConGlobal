"""Independent fixed-probe service behind the formal dycon CLI; no training writes."""
import json
import os
from pathlib import Path
import tempfile

from core.io import hash_json, now, read_json, sha256


def register(subparsers):
    parser = subparsers.add_parser("lora-probe", help="Evaluate fixed GRPO parent groups at an explicit checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--group-id", action="append", help="Repeat for each complete parent group; ordered and fixed")
    parser.add_argument("--baseline-report", help="Reuse group IDs and compare against a compatible JSON report")
    parser.add_argument("--gpu", help="Explicit physical GPU index or MIG UUID; required except for --dry-run")
    parser.add_argument("--output", required=True, help="New JSON report path outside sealed RL artifacts")
    parser.add_argument("--plot-output", help="Also draw a .png/.svg/.pdf chart and .data.json (no extra forwards)")
    parser.add_argument("--plot-include-untrained", action="store_true",
                        help="Include untrained/partial selected groups at scheduled steps in the chart")
    parser.add_argument("--dry-run", action="store_true", help="Verify metadata without model loading or report writing")
    plot = subparsers.add_parser("lora-probe-plot", help="Plot an existing fixed probe JSON without model forwards")
    plot.add_argument("--report", required=True)
    plot.add_argument("--output", required=True, help="New .png/.svg/.pdf chart; also writes .data.json")
    plot.add_argument("--include-untrained", action="store_true",
                      help="Include all reported fixed groups, including future/partial groups")


def _path(root, value):
    value = Path(value)
    return (value if value.is_absolute() else root / value).resolve()


def _output_path(root, value, checkpoint, config):
    output = _path(root, value)
    # Receipts seal all of these trees, including externally mounted roots.
    for parent in (output.parent, *output.parents):
        if (parent / "receipt.json").exists() or (parent / "attempt.json").exists():
            raise ValueError("Probe output must be outside attempt/checkpoint/sealed artifact directories")
    for storage in (root, Path(config.runtime.output_root), Path(config.algorithm.dataset_root)):
        if output.is_relative_to(storage.resolve() / "RL"):
            raise ValueError("Write probe reports outside RL source/artifact trees (for example reports/rl-probe)")
    if output.is_relative_to(checkpoint.parent.parent):
        raise ValueError("Probe output cannot modify its source attempt")
    if output.exists():
        raise FileExistsError("Probe output already exists; use a new report path")
    return output


def _publish(output, report, *, binary=False):
    # A hard link publishes the complete fsynced file without ever replacing a
    # report created by a concurrent evaluator. Same directory => same device.
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".probe-", suffix=".json", dir=output.parent)
    try:
        with os.fdopen(fd, "wb" if binary else "w", **({} if binary else {"encoding": "utf-8"})) as stream:
            if binary:
                stream.write(report)
            else:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, output)
    finally:
        os.unlink(name)


def _protocol(config, contract):
    from importlib.metadata import version
    return {"version": "fixed_grpo_preference_v1", "probability_mode": "raw_t1",
            "coordinates": "native_completion_targets", "reduction": "token_mean_branch_mean_group_mean",
            "forward": "complete_branch_teacher_forcing", "mode": "eval_no_grad",
            "dtype": config.model.dtype, "attention_backend": config.model.attention_backend,
            "attention_policy": config.training.memory.attention_policy,
            "mlp_chunk_tokens": config.training.memory.mlp_chunk_tokens,
            "projection_block_size": config.training.memory.loss_chunk_tokens or 256,
            "bf16_reduction": config.runtime.bf16_allow_reduced_precision_reduction,
            "bf16_split_k": config.runtime.bf16_allow_split_k,
            "base_fingerprint": contract["adapter"]["base_fingerprint"],
            "anchor": contract["grpo"]["reference_identity"],
            "advantage_policy": contract["input"]["advantage_policy"],
            "versions": {name: version(name) for name in ("torch", "transformers", "peft")}}


def evaluate(model, groups, protocol):
    """Shared target scorer; one complete branch per forward, no optimizer/RNG restore."""
    import torch
    from RL.grpo.logprobs import model_target_logprobs
    from RL.memory import attention_context
    from RL.probe_data import group_identity
    from RL.probe_metrics import TokenChanges, group_metrics
    model.eval()
    model.requires_grad_(False)
    device = next(model.parameters()).device
    reports, identities = [], []
    token_changes = TokenChanges()
    with torch.no_grad(), attention_context(protocol["attention_policy"]):
        for group, reference in groups:
            identities.append(group_identity(group, reference))
            branches = []
            group_changes = TokenChanges()
            for i, bid in enumerate(group.branch_ids):
                values = model_target_logprobs(model, group.branch_batch(i, device),
                                              protocol["projection_block_size"]).float().cpu()
                anchor = reference[bid].float().cpu()
                if (values.ndim != 1 or values.shape != anchor.shape or values.numel() == 0
                        or not torch.isfinite(values).all() or not torch.isfinite(anchor).all()):
                    raise ValueError("Probe target/reference shape or finite-value mismatch")
                changes = values.double() - anchor.double()
                group_changes.add(changes, float(group.advantages[i]))
                branches.append({"branch_id": bid, "advantage": float(group.advantages[i]),
                    "token_count": values.numel(), "logprob_mean": float(values.double().mean()),
                    "reference_logprob_mean": float(anchor.double().mean()),
                    "logprob_change": float(changes.mean())})
            reports.append({"group_id": group.group_id, "branches": branches,
                            "metrics": group_metrics(branches), "token_changes": group_changes.summary()})
            token_changes.extend(group_changes)
    return reports, identities, token_changes.summary()


def _configure_numerics(config):
    import torch
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.preferred_blas_library("cublaslt")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        config.runtime.bf16_allow_reduced_precision_reduction, config.runtime.bf16_allow_split_k)


def _checkpoint_context(checkpoint):
    from RL.config import Config, schedule_contract
    from RL.checkpoints import verify_checkpoint
    verify_checkpoint(checkpoint)
    receipt_hash = sha256(checkpoint / "receipt.json")
    metadata = read_json(checkpoint / "state.json")
    contract = metadata["contract"]
    if "grpo" not in contract:
        raise ValueError("Fixed probe requires an offline GRPO checkpoint")
    config_path = checkpoint.parent.parent / "resolved-config.json"
    config = Config.model_validate(read_json(config_path))
    if schedule_contract(config) != contract["schedule"]:
        raise ValueError("Checkpoint owner configuration differs from its frozen schedule")
    config_hash = sha256(config_path)
    return metadata, config, receipt_hash, config_hash


def run(root_config, *, checkpoint, group_ids=None, baseline_report=None, gpu=None, output, dry_run=False,
        plot_output=None, plot_include_untrained=False):
    root = Path(root_config["project_root"]).resolve()
    checkpoint = _path(root, checkpoint)
    metadata, config, receipt_hash, config_hash = _checkpoint_context(checkpoint)
    contract = metadata['contract']
    config_path = checkpoint.parent.parent / "resolved-config.json"
    if plot_include_untrained and not plot_output:
        raise ValueError('--plot-include-untrained requires --plot-output')
    baseline_path = _path(root, baseline_report) if baseline_report else None
    baseline_hash = sha256(baseline_path) if baseline_path else None
    baseline = read_json(baseline_path) if baseline_path else None
    if baseline is not None and (baseline.get("kind") != "fixed_grpo_preference" or baseline.get("schema_version") != 1):
        raise ValueError("Unsupported baseline probe report")
    group_ids = list(group_ids or (baseline["group_ids"] if baseline else []))
    if not group_ids or any(not isinstance(g, str) or not g for g in group_ids) or len(set(group_ids)) != len(group_ids):
        raise ValueError("Specify unique fixed --group-id values or reuse --baseline-report group IDs")
    if baseline and baseline["group_ids"] != group_ids:
        raise ValueError("Baseline requires the same ordered fixed group IDs")
    if not set(group_ids).issubset(contract["input"]["eligible_group_ids"]):
        raise ValueError("Probe group IDs must be eligible complete parent IDs in this checkpoint")
    output = _output_path(root, output, checkpoint, config)
    if plot_output:
        from RL.probe_plot import output_paths
        image, data_path = output_paths(root, plot_output)
        if output in (image, data_path):
            raise ValueError('Probe report and plot output paths must be distinct')
    if dry_run:
        return {"mode": "dry-run", "status": "metadata_validated", "checkpoint": str(checkpoint),
                "global_step": metadata["state"]["global_step"], "group_ids": group_ids,
                "output": str(output), "model_loaded": False,
                **({"plot_output": str(image)} if plot_output else {}),
                "coverage": "checkpoint/config/IDs only; selected payloads and baseline scoring compatibility checked on execution"}
    if gpu is None:
        raise ValueError("Specify an idle --gpu explicitly for independent probe evaluation")
    gpu_value = int(gpu) if str(gpu).isdigit() else str(gpu)
    config.runtime = type(config.runtime).model_validate({**config.runtime.model_dump(), "gpu": gpu_value, "gpus": None})
    config.preflight.check_resources = True
    from core.locking import file_lock
    from RL.service import _select_gpu
    with file_lock(root / "state/artifact-locks/lora-gpu" / (str(gpu_value) + ".lock"), nonblocking=True):
        resources = _select_gpu(config)  # Must precede imports that can initialize CUDA.
        import torch
        from RL.artifacts import base_identity
        from RL.modeling import load_base, inject
        from RL.memory import install_mlp_backend
        from RL.probe_data import ProbeGroups
        from RL.probe_metrics import aggregate, compare
        base = base_identity(config.model.path, root, verify_content=False)
        if base["fingerprint"] != contract["adapter"]["base_fingerprint"]:
            raise ValueError("Probe base fingerprint differs from checkpoint")
        _configure_numerics(config)
        protocol = _protocol(config, contract)
        if baseline and baseline["probe_identity"]["protocol"] != protocol:
            raise ValueError("Baseline scoring protocol/anchor differs from current checkpoint")
        model, _ = load_base(config)
        model, _ = inject(model, config.lora, adapter=checkpoint / "adapter", expected_targets=contract["adapter"]["targets"])
        install_mlp_backend(model, config.training.memory.mlp_chunk_tokens)
        with ProbeGroups(config, checkpoint, contract, group_ids) as groups:
            training_schedule = groups.positions(metadata['state'])
            if plot_output:
                # Fail before scoring when the requested chart has no complete trained parents.
                if not plot_include_untrained and not any(
                        g['status_at_checkpoint'] == 'trained' for g in training_schedule['groups']):
                    raise ValueError('No fully trained selected groups to plot; use --plot-include-untrained')
            reports, identities, token_changes = evaluate(model, groups, protocol)
        metrics, support = aggregate(reports)
        identity = {"protocol": protocol, "groups": identities}
        report = {"schema_version": 1, "kind": "fixed_grpo_preference", "created_at": now(),
                  "checkpoint": {"path": str(checkpoint), "receipt_sha256": receipt_hash,
                                 "global_step": metadata["state"]["global_step"]},
                  "group_ids": group_ids, "probe_identity": identity, "probe_sha256": hash_json(identity),
                  "scope": "fixed_subset_of_training_groups; not_held_out_or_downstream_success",
                  "aggregate": metrics, "support": support, "groups": reports,
                  "token_changes": token_changes,
                  "training_schedule": training_schedule,
                  "runtime": resources, "output": str(output)}
        if baseline:
            report["comparison"] = {**compare(report, baseline), "baseline_report": str(baseline_path),
                                    "baseline_report_sha256": baseline_hash}
            if sha256(baseline_path) != baseline_hash:
                raise ValueError("Baseline report changed during probe evaluation")
        if sha256(checkpoint / "receipt.json") != receipt_hash or sha256(config_path) != config_hash:
            raise ValueError("Checkpoint receipt or owner configuration changed during probe evaluation")
        _publish(output, report)
        plots = None
        if plot_output:
            from RL.probe_plot import run as draw_report
            try:
                plots = draw_report(root_config, report=str(output), output=str(plot_output),
                                    include_untrained=plot_include_untrained)
            except Exception as exc:
                raise RuntimeError(f'Probe JSON saved at {output}; chart generation failed: {exc}. '
                                   'Retry with dycon lora-probe-plot --report JSON --output IMAGE') from exc
        return {"status": "complete", "output": str(output), "checkpoint": report["checkpoint"],
                "probe_sha256": report["probe_sha256"], "aggregate": metrics, "support": support,
                **({"plot": plots} if plots else {}),
                **({"comparison": report["comparison"]["aggregate_delta"]} if baseline else {})}

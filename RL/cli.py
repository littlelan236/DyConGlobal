"""Light registration and dispatch through the same public service as Python callers."""


def register(subparsers):
    parser = subparsers.add_parser("rl-train", help="RL training through the common service (LoRA or full FSDP2)")
    parser.add_argument("--training-config", required=True)
    parser = subparsers.add_parser("rl-prepare-key-steps", help="Seal eight-child terminal reward GRPO groups")
    parser.add_argument("--groups", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=["adaptive","random","entropy"], required=True)
    parser = subparsers.add_parser("rl-prepare-baseline", help="Seal historical baseline rollout evidence")
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failure-reward", type=float, default=-3.0)
    parser = subparsers.add_parser("rl-prepare-baseline-stats", help="Prepare offline baseline task reward statistics")
    parser.add_argument("--dataset", required=True)
    parser = subparsers.add_parser("rl-gradient-compare", help="Compare all full-policy reference/FLA gradients without updates")
    parser.add_argument("--comparison-config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser = subparsers.add_parser("rl-export", help="Stream a verified full checkpoint into standard HF shards")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--dry-run", action="store_true")
    parser = subparsers.add_parser("rl-probe", help="Probe a full checkpoint using its fixed FSDP2 topology")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--group-id", action="append", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser = subparsers.add_parser("lora-export", help="Merge a verified project checkpoint into a full serving model on CPU")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--dry-run", action="store_true")
    from RL.probe import register as register_probe
    register_probe(subparsers)
    from RL.observability_cli import register
    register(subparsers)
    parser = subparsers.add_parser("lora-train", help="Qwen3.5-9B LoRA training; offline GRPO supports 1/2/4 GPUs")
    parser.add_argument("--training-config", default="RL/experiments/lora-smoke.yaml")
    parser = subparsers.add_parser("lora-prepare", help="Seal indexed GRPO groups or an explicit append extension (no model load)")
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--prepared-id", required=True)
    parser.add_argument("--shard-groups", type=int, default=16)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--parent-checkpoint", help="Complete new-schema checkpoint to extend")
    source.add_argument("--source-attempt", help="Explicit read-only frozen-groups.json migration")
    parser.add_argument("--extension-id")
    parser.add_argument("--append-seed", type=int)


def execute(args, root_config):
    if args.command == "rl-prepare-key-steps":
        from pathlib import Path
        from RL.key_steps import prepare
        root = Path(root_config["project_root"])
        return prepare(root / args.groups, root / args.output, args.method)
    if args.command == "rl-prepare-baseline":
        from math import isfinite
        from RL.offline_baseline import prepare

        if not isfinite(args.failure_reward) or args.failure_reward >= 0:
            raise ValueError("failure reward must be finite and negative")
        manifest = prepare(root_config["project_root"], args.run_id, args.output, args.failure_reward)
        return {"output": args.output, "trajectories": len(manifest["entries"]), "runs": manifest["runs"]}
    if args.command == "rl-prepare-baseline-stats":
        from pathlib import Path
        from RL.offline_baseline import TASK_REWARD_STATS_FILE, prepare_task_reward_stats

        dataset = (Path(root_config["project_root"]) / args.dataset).resolve()
        stats = prepare_task_reward_stats(dataset)
        tasks = stats["tasks"].values()
        return {"output": str(dataset / TASK_REWARD_STATS_FILE), "version": stats["version"],
                "manifest_sha256": stats["manifest_sha256"], "tasks": len(tasks),
                "trajectories": sum(task["count"] for task in tasks),
                "singleton_tasks": sum(task["count"] == 1 for task in tasks),
                "equal_reward_tasks": sum(task["count"] > 1 and task["std"] == 0.0 for task in tasks)}
    if args.command == "rl-gradient-compare":
        if not args.dry_run:
            _rl_environment(root_config)
        from RL.full_gradient_compare import run
        return run(root_config, args.comparison_config, dry_run=args.dry_run)
    if args.command == "artifact" and args.scope == "rl-full-attempt":
        from RL.full_artifacts import artifact
        return artifact(root_config, args.run_id, args.storage_root, deep=args.artifact_command == "verify")
    if args.command == "rl-export":
        if not args.dry_run:
            _rl_environment(root_config)
        from RL.full_export import export
        return export(root_config, args.checkpoint, args.base_model, args.output, args.dtype, args.dry_run)
    if args.command == "rl-probe":
        if not args.dry_run:
            _rl_environment(root_config)
        from RL.full_probe import run
        return run(root_config, args)
    if args.command == "lora-export":
        from RL.export import export
        return export(root_config, checkpoint=args.checkpoint, base_model=args.base_model,
                      output=args.output, dtype=args.dtype, dry_run=args.dry_run)
    if args.command == "lora-probe":
        from RL.probe import run
        return run(root_config, checkpoint=args.checkpoint, group_ids=args.group_id,
                   baseline_report=args.baseline_report, gpu=args.gpu, output=args.output, dry_run=args.dry_run,
                   plot_output=args.plot_output, plot_include_untrained=args.plot_include_untrained)
    if args.command == "lora-probe-plot":
        from RL.probe_plot import run
        return run(root_config, report=args.report, output=args.output, include_untrained=args.include_untrained)
    if args.command in {"rl-status", "rl-tensorboard", "rl-advantages", "lora-status", "lora-tensorboard", "lora-advantages", "lora-memory"}:
        from RL.observability_cli import execute
        return execute(args, root_config)
    if args.command == "lora-prepare":
        from RL.grpo.prepared_service import create
        if not args.parent_checkpoint and (args.extension_id is not None or args.append_seed is not None):
            raise ValueError("Extension arguments require --parent-checkpoint")
        return create(args.training_config, root_config, args.prepared_id, shard_groups=args.shard_groups,
                      checkpoint=args.parent_checkpoint, extension_id=args.extension_id,
                      append_seed=args.append_seed, source_attempt=args.source_attempt)
    if args.command in {"lora-train", "rl-train"}:
        from RL.config import load, is_full
        config, _ = load(args.training_config, root_config["project_root"])
        if args.command == "lora-train" and is_full(config):
            raise ValueError("Full policies require dycon rl-train")
        if config.run.mode != "dry-run":
            _rl_environment(root_config)
        from RL.service import run

        return run(root_config, args.training_config)
    from RL.artifacts import show, verify

    operation = show if args.artifact_command == "show" else verify
    if not args.run_id:
        raise ValueError("lora-attempt scope requires an attempt ID")
    return operation(root_config, args.scope, args.run_id, storage_root=args.storage_root, dataset_id=args.dataset_id)


def _rl_environment(root_config):
    """Actual execution uses the dedicated locked RL environment; help stays light."""
    import os
    from pathlib import Path
    import sys
    expected = Path(root_config["project_root"]) / ".venvs/rl"
    if Path(sys.prefix).resolve() != expected.resolve():
        executable = expected / "bin/python"
        if not executable.is_file():
            raise RuntimeError("Dedicated .venvs/rl environment is missing")
        os.execv(str(executable), [str(executable), "-m", "core.cli", *sys.argv[1:]])

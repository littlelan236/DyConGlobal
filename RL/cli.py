"""Light registration and dispatch through the same public service as Python callers."""


def register(subparsers):
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
    if args.command in {"lora-status", "lora-tensorboard", "lora-advantages", "lora-memory"}:
        from RL.observability_cli import execute
        return execute(args, root_config)
    if args.command == "lora-prepare":
        from RL.grpo.prepared_service import create
        if not args.parent_checkpoint and (args.extension_id is not None or args.append_seed is not None):
            raise ValueError("Extension arguments require --parent-checkpoint")
        return create(args.training_config, root_config, args.prepared_id, shard_groups=args.shard_groups,
                      checkpoint=args.parent_checkpoint, extension_id=args.extension_id,
                      append_seed=args.append_seed, source_attempt=args.source_attempt)
    if args.command == "lora-train":
        from RL.service import run

        return run(root_config, args.training_config)
    from RL.artifacts import show, verify

    operation = show if args.artifact_command == "show" else verify
    if not args.run_id:
        raise ValueError("lora-attempt scope requires an attempt ID")
    return operation(root_config, args.scope, args.run_id, storage_root=args.storage_root, dataset_id=args.dataset_id)

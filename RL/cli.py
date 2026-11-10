"""Light registration and dispatch through the same public service as Python callers."""


def register(subparsers):
    parser = subparsers.add_parser("lora-train", help="Single GPU Qwen3.5-9B LoRA engineering smoke")
    parser.add_argument("--training-config", default="RL/experiments/lora-smoke.yaml")


def execute(args, root_config):
    if args.command == "lora-train":
        from RL.service import run

        return run(root_config, args.training_config)
    from RL.artifacts import show, verify

    operation = show if args.artifact_command == "show" else verify
    if not args.run_id:
        raise ValueError("lora-attempt scope requires an attempt ID")
    return operation(root_config, args.scope, args.run_id, storage_root=args.storage_root, dataset_id=args.dataset_id)

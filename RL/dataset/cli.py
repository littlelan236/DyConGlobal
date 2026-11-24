def register(subparsers):
    p=subparsers.add_parser("rl-dataset",help="Offline GRPO step datasets and behavior replay")
    q=p.add_subparsers(dest="rl_dataset_command",required=True)
    p = q.add_parser("key-step-plan", help="Freeze B=8 mother trajectories and prepare a CPU replay scan")
    p.add_argument("--source-run", required=True); p.add_argument("--output", required=True)
    p = q.add_parser("key-step-select", help="Select adaptive/random/entropy equal-K points after replay scans")
    p.add_argument("--plan", required=True); p.add_argument("--scan-run", required=True); p.add_argument("--output", required=True)
    p = q.add_parser("key-step-seal", help="Seal terminal reward continuation groups after formal harness scoring")
    p.add_argument("--run", required=True); p.add_argument("--output", required=True)
    probe = q.add_parser("branch-probe", help="Validate inputs for a paired branch probe")
    probe.add_argument("--probe-config", required=True)
    probe.add_argument("--dry-run", action="store_true")
    probe.add_argument("--manifest-output")
    probe.add_argument("--output")
    b=q.add_parser("build");b.add_argument("--dataset-config",required=True);b.add_argument("--dry-run",action="store_true");b.add_argument("--recover",action="store_true")
    c=q.add_parser("check");c.add_argument("dataset_id");c.add_argument("--dataset-root",required=True);c.add_argument("--level",choices=["metadata","sources","full"],default="full")
    m=q.add_parser("materialize");m.add_argument("dataset_id");m.add_argument("--dataset-root",required=True);m.add_argument("--cache-root",required=True)
    r=q.add_parser("replay-logprobs");r.add_argument("--replay-config",required=True);r.add_argument("--dry-run",action="store_true");r.add_argument("--recover",action="store_true")
    g=r.add_mutually_exclusive_group();g.add_argument("--detach",action="store_true");g.add_argument("--cancel",action="store_true");g.add_argument("--status",action="store_true")
    c=q.add_parser("check-logprobs");c.add_argument("replay_id");c.add_argument("--storage-root",required=True);c.add_argument("--with-values",action="store_true");c.add_argument("--dataset-id");c.add_argument("--dataset-root")
    for name in ("probability-plan", "probability-build"):
        c=q.add_parser(name);c.add_argument("--probability-config",required=True);c.add_argument("--recover",action="store_true")
    c=q.add_parser("probability-check");c.add_argument("probability_id");c.add_argument("--storage-root",required=True);c.add_argument("--with-values",action="store_true");c.add_argument("--sources",action="store_true")

def execute(args,cfg):
    if args.rl_dataset_command.startswith("key-step-"):
        from rollout.key_steps import service
        if args.rl_dataset_command == "key-step-plan":
            return service.plan(cfg, args.source_run, args.output)
        if args.rl_dataset_command == "key-step-select":
            return service.select(cfg, args.plan, args.scan_run, args.output)
        return service.seal(cfg, args.run, args.output)
    if args.rl_dataset_command == "branch-probe":
        from .branch_probe import preflight, build_probe_manifest
        if not args.dry_run:
            if not args.output or args.manifest_output:
                raise ValueError("Execution requires --output and cannot use --manifest-output")
            from .branch_probe_online import run
            return run(cfg, args.probe_config, args.output)
        result = preflight(args.probe_config)
        if args.manifest_output:
            result["manifest"] = build_probe_manifest(args.probe_config, args.manifest_output)
        return result
    from pathlib import Path
    from .artifacts import directory
    from .build import build_dataset,check_dataset
    if args.rl_dataset_command in {"probability-plan", "probability-build"}:
        from .probability_artifacts import probability_plan, probability_build
        operation = probability_plan if args.rl_dataset_command == "probability-plan" else probability_build
        return operation(cfg, args.probability_config, recover=args.recover)
    if args.rl_dataset_command=="probability-check":
        from .probability_artifacts import check_probability
        return check_probability(directory(args.storage_root,"rl-probability",args.probability_id), with_values=args.with_values, sources=args.sources)
    if args.rl_dataset_command=="build":return build_dataset(cfg,args.dataset_config,dry_run=args.dry_run,recover=args.recover)
    if args.rl_dataset_command=="check":return check_dataset(directory(args.dataset_root,"rl-dataset",args.dataset_id),args.level)
    if args.rl_dataset_command=="materialize":
        from .materialize import materialize
        return materialize(directory(args.dataset_root,"rl-dataset",args.dataset_id),args.cache_root)
    if args.rl_dataset_command=="replay-logprobs":
        if args.detach or args.cancel or args.status:
            if args.dry_run: raise ValueError("Replay lifecycle flags cannot be combined with dry-run")
            from .replay.jobs import manage
            return manage(cfg,args.replay_config,detach=args.detach,cancel=args.cancel,status=args.status,recover=args.recover)
        from .replay.service import replay_logprobs
        result = replay_logprobs(cfg,args.replay_config,dry=args.dry_run,recover=args.recover)
        if result.get("status") == "blocked": raise ValueError("Replay dry-run blocked: " + str(result.get("blocked", [])))
        return result
    if args.rl_dataset_command=="check-logprobs":
        from .artifacts import directory
        from .replay.artifacts import check_replay
        if bool(args.dataset_id) != bool(args.dataset_root): raise ValueError("Provide dataset-id and dataset-root together")
        dataset=directory(args.dataset_root,"rl-dataset",args.dataset_id) if args.dataset_id else None
        return check_replay(directory(args.storage_root,"rl-replay",args.replay_id),dataset_path=dataset,with_values=args.with_values)

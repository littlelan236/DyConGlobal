"""Light argparse extension for the existing dycon entrypoint."""

from __future__ import annotations

from pathlib import Path


def register(sub):
    dataset = sub.add_parser("steering-dataset", help="Immutable offline consensus branch hidden datasets")
    commands = dataset.add_subparsers(dest="steering_dataset_command", required=True)
    build = commands.add_parser("build")
    build.add_argument("dataset_id")
    build.add_argument("--source-run", action="append", required=True)
    group = build.add_mutually_exclusive_group(required=True)
    group.add_argument("--rollout-id", action="append", default=[])
    group.add_argument("--rollout-list", type=Path)
    group.add_argument(
        "--all-submitted", action="store_true", help="Freeze all currently eligible submitted bindings"
    )
    build.add_argument("--site", action="append")
    build.add_argument("--storage-root", type=Path)
    check = commands.add_parser("check")
    check.add_argument("dataset_id")
    check.add_argument("--dataset-root", type=Path)
    check.add_argument("--with-tensors", action="store_true")
    cache = commands.add_parser("cache")
    cache.add_argument("dataset_id")
    cache.add_argument("--dataset-root", type=Path)
    cache.add_argument("--cache-root", type=Path)
    cache.add_argument("--site", action="append")
    cache.add_argument(
        "--check-data", action="store_true", help="Verify existing exact cache; never build it"
    )
    cache.add_argument("--disk-reserve-gib", type=float, default=2.0)
    visualize = sub.add_parser(
        "steering-visualize", help="LDA, t-SNE, UMAP or immutable frozen-coordinate redraw"
    )
    visualize.add_argument("analysis_id")
    source = visualize.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-id")
    source.add_argument("--redraw-from")
    visualize.add_argument("--dataset-root", type=Path)
    visualize.add_argument("--cache-root", type=Path)
    visualize.add_argument("--analysis-root", type=Path, help="Registered/input root of --redraw-from")
    visualize.add_argument("--storage-root", type=Path)
    visualize.add_argument(
        "--method", choices=["lda", "tsne", "umap"], help="Fit default: lda; redraw inherits source"
    )
    visualize.add_argument(
        "--pca-components", type=int, help="Embedding PCA dimensions (default 50; 0 disables)"
    )
    visualize.add_argument(
        "--perplexity", type=float, help="t-SNE perplexity (default 30; must be < samples)"
    )
    visualize.add_argument("--max-iter", type=int, help="t-SNE iterations (default 1000; >= 250)")
    visualize.add_argument("--n-neighbors", type=int, help="UMAP neighbors (default 15; must be < samples)")
    visualize.add_argument("--min-dist", type=float, help="UMAP minimum distance (default 0.1; in [0,1])")
    visualize.add_argument(
        "--threads", type=int, help="Embedding BLAS/OpenMP limit (default 4; seeded UMAP uses 1)"
    )
    visualize.add_argument("--site", action="append")
    visualize.add_argument("--fit-scope", choices=["all", "grouped"], default="all")
    visualize.add_argument("--split", type=Path)
    visualize.add_argument("--memory-budget-gib", type=float, default=32.0)
    visualize.add_argument("--tol", type=float, default=1e-4)
    visualize.add_argument("--seed", type=int, default=0)
    visualize.add_argument("--max-display-points", type=int)
    visualize.add_argument("--facet", choices=["source_run_id", "retained"])
    visualize.add_argument("--format", dest="formats", choices=["png", "svg", "pdf"], action="append")


def offline_config(path):
    import yaml

    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    return {"project_root": str(path.parent), "paths": raw.get("paths", {})}


def execute(args):
    from steering.datasets.paths import locate

    cfg = offline_config(args.config)
    if args.command == "artifact":
        from steering.artifacts import show

        if not args.run_id:
            raise ValueError("Artifact operations require an explicit steering artifact ID")
        return show(
            cfg, args.scope, args.run_id, storage_root=args.storage_root, with_tensors=args.with_tensors
        )
    if args.command == "steering-dataset":
        if args.steering_dataset_command == "build":
            from steering.datasets.build import build_dataset

            return build_dataset(
                cfg,
                args.dataset_id,
                source_runs=args.source_run,
                rollout_ids=args.rollout_id,
                rollout_list=args.rollout_list,
                all_submitted=args.all_submitted,
                storage_root=args.storage_root,
                sites=args.site,
            )
        if args.steering_dataset_command == "check":
            from steering.datasets.build import check_dataset

            return check_dataset(
                locate(cfg, "steering_dataset", args.dataset_id, args.dataset_root),
                with_tensors=args.with_tensors,
            )
        if args.disk_reserve_gib < 0:
            raise ValueError("Disk reserve cannot be negative")
        from steering.datasets.cache import cache_features, existing_cache, specification

        if args.check_data:
            dataset = locate(cfg, "steering_dataset", args.dataset_id, args.dataset_root)
            identity, spec = specification(dataset, args.site)
            cache = existing_cache(cfg, identity, spec, args.cache_root)
            if cache is None:
                raise FileNotFoundError("No complete exact feature cache exists")
            return {"status": "verified", "cache_id": identity, "path": str(cache)}
        return cache_features(
            cfg,
            args.dataset_id,
            dataset_root=args.dataset_root,
            cache_root=args.cache_root,
            sites=args.site,
            disk_reserve_bytes=int(args.disk_reserve_gib * 1024**3),
        )
    from steering.visualization.service import redraw, visualize

    embedding_options = {
        key: getattr(args, key)
        for key in ("pca_components", "perplexity", "max_iter", "n_neighbors", "min_dist", "threads")
        if getattr(args, key) is not None
    }
    method = args.method or "lda"
    allowed = {
        "lda": set(),
        "tsne": {"pca_components", "perplexity", "max_iter", "threads"},
        "umap": {"pca_components", "n_neighbors", "min_dist", "threads"},
    }
    if not args.redraw_from and set(embedding_options) - allowed[method]:
        raise ValueError("Embedding options do not apply to the selected method")

    common = {
        "storage_root": args.storage_root,
        "sites": args.site,
        "seed": args.seed,
        "formats": tuple(args.formats or ["png", "svg"]),
        "max_display_points": args.max_display_points,
        "facet": args.facet,
    }
    if args.redraw_from:
        if args.split or args.dataset_root or args.cache_root or args.fit_scope != "all" or embedding_options:
            raise ValueError(
                "Redraw inherits frozen fitting/labels; use only analysis source and rendering options"
            )
        return redraw(
            cfg,
            args.analysis_id,
            source_analysis_id=args.redraw_from,
            analysis_root=args.analysis_root,
            method=args.method,
            **common,
        )
    return visualize(
        cfg,
        args.analysis_id,
        dataset_id=args.dataset_id,
        dataset_root=args.dataset_root,
        cache_root=args.cache_root,
        fit_scope=args.fit_scope,
        split_file=args.split,
        memory_budget_bytes=int(args.memory_budget_gib * 1024**3),
        tol=args.tol,
        method=method,
        embedding_options=embedding_options,
        **common,
    )

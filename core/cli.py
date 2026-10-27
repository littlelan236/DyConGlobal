"""Light CLI; heavy model, agent and Docker dependencies are lazy imports."""

import argparse
import json
from pathlib import Path

from catalog.store import Catalog
from core.config import resolve
from core.io import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dycon")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--run-id", dest="analysis_run_id", help="Bind an analysis command to a frozen run")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Create and operate immutable run plans")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    q = run_sub.add_parser("create")
    q.add_argument("run_id")
    q.add_argument("--operation", choices=["rollout", "harness", "enrichment", "hidden", "analysis"], required=True)
    q.add_argument("--tasks-file", type=Path)
    q.add_argument("--task-id", action="append", default=[])
    q.add_argument("--partition")
    q.add_argument("--repeat-count", type=int, default=1)
    q = run_sub.add_parser("show")
    q.add_argument("run_id")
    q = run_sub.add_parser("status")
    q.add_argument("run_id")
    q = run_sub.add_parser("reconcile")
    q.add_argument("run_id")
    q = run_sub.add_parser("start")
    q.add_argument("run_id")
    q.add_argument("--detach", action="store_true")
    q.add_argument("--check-only", action="store_true", help="Run the managed startup acceptance without rollout tasks")
    q = run_sub.add_parser("wait")
    q.add_argument("run_id")
    q = run_sub.add_parser("resume")
    q.add_argument("run_id")
    q.add_argument("--detach", action="store_true")
    q.add_argument("--check-only", action="store_true")
    q = run_sub.add_parser("cancel")
    q.add_argument("run_id")
    q.add_argument("--reason", default="user requested cancellation")
    q = run_sub.add_parser("revise-capture-budget", help="Audit an explicitly authorized offline capture resource revision")
    q.add_argument("run_id")
    q.add_argument("--revision-id", required=True)
    q.add_argument("--max-bytes", type=int, required=True)
    q.add_argument("--high", type=float, required=True)
    q.add_argument("--low", type=float, required=True)
    q.add_argument("--reason", required=True)
    run_sub.add_parser("list")
    harness = sub.add_parser("harness", help="Prepare, run and inspect run-local harness results")
    harness_sub = harness.add_subparsers(dest="harness_command", required=True)
    q = harness_sub.add_parser("prepare")
    q.add_argument("run_id")
    q.add_argument("--rollout-id", action="append", default=[])
    q.add_argument("--source-run")
    q = harness_sub.add_parser("run")
    q.add_argument("run_id")
    q.add_argument("eval_id")
    q.add_argument("--gold", action="store_true")
    q = harness_sub.add_parser("status")
    q.add_argument("run_id")
    q = harness_sub.add_parser("metadata-prefetch", help="Cache exact official requirements before harness execution")
    q.add_argument("--instances", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--proxy", help="Explicit host-side proxy; never changes container network policy")
    q.add_argument("--refresh", action="store_true", help="Revalidate cached bytes against the pinned upstream revision")
    report = sub.add_parser("report", help="Project a run-local report")
    report_sub = report.add_subparsers(dest="report_command", required=True)
    q = report_sub.add_parser("run")
    q.add_argument("run_id")
    capture = sub.add_parser("capture", help="Inspect or recover online capture")
    capture_sub = capture.add_subparsers(dest="capture_command", required=True)
    q = capture_sub.add_parser("status")
    q.add_argument("run_id")
    q = capture_sub.add_parser("recover")
    q.add_argument("run_id")
    q = capture_sub.add_parser("repair")
    q.add_argument("run_id")
    q.add_argument("--rollout-id", action="append", default=[])
    q.add_argument("--session-id", action="append", default=[])
    q.add_argument("--apply", action="store_true")
    q.add_argument("--repair-id")
    q.add_argument("--pointwise", action="store_true",
                   help="Recover all integrity-checked points, including incomplete sessions")
    q = capture_sub.add_parser("archive", help="Archive inactive capture spool to NAS and release local storage")
    q.add_argument("run_id")
    q.add_argument("--archive-id", required=True)
    q.add_argument("--size-only-cleanup", action="store_true", help="Complete an already copied archive using file set and size only")
    q = capture_sub.add_parser("relocate-spool", help="Relocate stopped local capture storage and rebind its disk ledger")
    q.add_argument("--destination", type=Path, required=True)
    q.add_argument("--migration-id", required=True)
    q.add_argument("--apply", action="store_true")
    artifact = sub.add_parser("artifact", help="Inspect or verify run artifacts")
    artifact_sub = artifact.add_subparsers(dest="artifact_command", required=True)
    q = artifact_sub.add_parser("show")
    q.add_argument("run_id", nargs="?")
    q.add_argument("--scope", choices=["run", "difficulty-dataset", "difficulty-split", "difficulty-attempt", "difficulty-prediction", "difficulty-evaluation", "action-analysis"], default="run")
    q.add_argument("--dataset-id")
    q = artifact_sub.add_parser("verify")
    q.add_argument("run_id", nargs="?")
    q.add_argument("--scope", choices=["run", "difficulty-dataset", "difficulty-split", "difficulty-attempt", "difficulty-prediction", "difficulty-evaluation", "action-analysis"], default="run")
    q.add_argument("--dataset-id")
    for cmd in ["config-check", "catalog-check", "runtime-check", "recover"]:
        sub.add_parser(cmd)
    storage = sub.add_parser("docker-storage", help="Apply/check config.yaml rootless Docker data-root")
    storage.add_argument("action", choices=["status", "apply"])
    wheelhouse = sub.add_parser("wheelhouse", help="Build/verify a shared immutable task wheelhouse")
    wheelhouse_sub = wheelhouse.add_subparsers(dest="wheelhouse_command", required=True)
    p = wheelhouse_sub.add_parser("build")
    p.add_argument("publication_id")
    p.add_argument("--source", action="append", default=[], type=Path)
    p.add_argument("--download-build-tools", action="store_true")
    p = wheelhouse_sub.add_parser("verify")
    p.add_argument("path", type=Path)
    sub.add_parser("capture-recover").add_argument("run_id")
    p = sub.add_parser("capture-publish")
    p.add_argument("run_id")
    p.add_argument("--repair-id", required=True)
    p.add_argument("--publication-id", required=True)
    p.add_argument("--resolved-only", action="store_true")
    p.add_argument("--coverage-policy", choices=["complete_native_sessions_v1", "available_native_suffix_v1"], default="complete_native_sessions_v1")
    p = sub.add_parser("capture-reclassify", help="Reclassify finalized branch captures from native token evidence")
    p.add_argument("run_id")
    p.add_argument("--session-id", required=True, action="append")
    p.add_argument("--wait-for-run", action="store_true", help=argparse.SUPPRESS)
    p = sub.add_parser("online-index")
    p.add_argument("--publication-id")
    p.add_argument("run_id")
    p.add_argument("index_id")
    p.add_argument("--rollout-id", action="append", default=[])
    p.add_argument("--rollout-list", type=Path)
    dataset = sub.add_parser("difficulty-dataset", help="Create an immutable difficulty dataset")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    q = dataset_sub.add_parser("compose")
    q.add_argument("dataset_id")
    q.add_argument("--source-dataset", required=True, action="append")
    q = dataset_sub.add_parser("resolved")
    q.add_argument("dataset_id")
    q.add_argument("--source-run", required=True)
    q.add_argument("--index-id", required=True)
    q.add_argument("--rollout-id", action="append", default=[])
    q.add_argument("--rollout-list", type=Path)
    q.add_argument("--label-normalization", choices=["raw", "trajectory_fraction"], default="raw")
    p = sub.add_parser("difficulty-split", help="Freeze a reusable dataset split")
    p.add_argument("dataset_id")
    p.add_argument("split_id")
    p.add_argument("--strategy", choices=["event_random_v1", "task_random_v1", "repo_source_quota_v1", "inherit_task_split_v1"], default="event_random_v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--validation-tasks", type=int, default=48)
    p.add_argument("--parent-attempt-id")
    p.add_argument("--parent-receipt-sha256")
    for cmd in ["tasks-import", "rollouts-import", "evaluations-import"]:
        sub.add_parser(cmd).add_argument("list")
    sub.add_parser("model-register").add_argument("checkpoint")
    skills_images = sub.add_parser("skillsbench-images", help="Prepare pinned SkillsBench images in NAS")
    skills_images.add_argument("action", choices=["plan", "prepare", "status", "verify"])
    skills_images.add_argument("--task-id", action="append", default=[])
    p = sub.add_parser("benchmark-pull")
    p.add_argument("cohort_id")
    p.add_argument("--count", type=int, default=600)
    p.add_argument("--proxy", default="http://127.0.0.1:7890")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--source-revision", default="c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2")
    p.add_argument("--results-revision", default="40f164d5b8f1d249bf95a6df8b74b577fd8e519d")
    p = sub.add_parser("rollout")
    p.add_argument("run_id")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--task-id", action="append", default=[])
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Run the real tool/capture startup gate without admitting tasks",
    )
    p = sub.add_parser("runs-migrate-to-repetition", help="One-time migration of saved runs to schema 2")
    p.add_argument("--source-root", default="runs")
    p.add_argument("--archive-root", default="archive/runs-schema1")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("migration-verify", help="Verify the one-time runs migration")
    p.add_argument("manifest", type=Path)
    p = sub.add_parser("prepare-evaluation")
    p.add_argument("run_id")
    p.add_argument("--source-run", required=True)
    p.add_argument("--rollout-id", action="append", default=[],
                   help="Freeze exactly these canonical rollout IDs")
    p = sub.add_parser("evaluate")
    p.add_argument("run_id")
    p.add_argument("eval_id")
    p.add_argument("--gold", action="store_true")
    sub.add_parser("enrich").add_argument("dataset_id")
    sub.add_parser("token-trace").add_argument("dataset_id")
    sub.add_parser("trajectory-token-audit").add_argument("dataset_id")
    p = sub.add_parser("hidden")
    p.add_argument("dataset_id")
    p.add_argument("capture_id")
    p = sub.add_parser("images-check")
    p.add_argument("--deep", action="store_true")
    p = sub.add_parser("token-join")
    p.add_argument("dataset_id")
    p.add_argument("capture_id")
    p.add_argument("index_id")
    p.add_argument("--source-run")
    p.add_argument("--rollout-id", action="append", default=[])
    p.add_argument("--rollout-list", type=Path)
    sub.add_parser("hidden-token-join-audit").add_argument("index_id")
    p = sub.add_parser("dataset-index")
    p.add_argument("dataset_id")
    p.add_argument("capture_id")
    p.add_argument("index_id")
    p.add_argument("--source-run")
    p.add_argument("--rollout-id", action="append", default=[])
    p.add_argument("--rollout-list", type=Path)
    p = sub.add_parser("action-submission-audit", help="Audit submitted empty/invalid patches from frozen 80-task cohorts")
    p.add_argument("analysis_id", nargs="?")
    p.add_argument("--list", action="store_true", help="Read-only candidate run inventory")
    p.add_argument("--check", action="store_true", help="Verify an existing audit receipt and source hashes")
    p.add_argument("--task-count", type=int, default=80)
    p.add_argument("--min-submitted", type=int, default=30, help="Exclusive threshold: submitted must be greater")
    p.add_argument("--reference-run", default="verified80-parallel-baseline-20261021")
    p = sub.add_parser("action-branch-stats", help="Exact-action diversity and winner-support histograms from frozen parallel decisions")
    from action.evaluation.branch_curves import METRICS
    p.add_argument("analysis_id")
    p.add_argument("--source-run", action="append", default=[])
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--rollout-id", action="append", default=[])
    scope.add_argument("--rollout-list", type=Path)
    p.add_argument("--branch-scope", choices=["all", "retained", "both"], default="both")
    p.add_argument("--termination-reason", help="Optional exact termination reason, e.g. Submitted")
    p.add_argument("--format", dest="formats", choices=["png", "svg"], action="append", default=[])
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--line-mode", choices=["per-run", "all-runs", "both", "none", "per-trajectory"], default="both",
                   help="Normalized run averages, or one raw-step figure per selected rollout with per-trajectory")
    p.add_argument("--line-metric", dest="line_metrics", action="append", default=[], choices=[m for m, _ in METRICS],
                   help="Repeat to choose chart metrics; per-trajectory defaults to H, H/ln(K), K, exp(H), 1-q")
    p.add_argument("--progress-points", type=int, default=101,
                   help="Inclusive 0..1 interpolation grid size (2..1001)")
    p.add_argument("--check", action="store_true", help="Verify existing branch receipt and source checksums")
    p = sub.add_parser("action-review-length", help="Estimate review episodes from frozen submission/action evidence")
    p.add_argument("analysis_id")
    p.add_argument("--audit-id")
    p.add_argument("--action-id")
    p.add_argument("--check", action="store_true")
    p = sub.add_parser("action-classify", help="Create an immutable project-level action analysis and heatmaps")
    p.add_argument("analysis_id")
    p.add_argument("--source-run", action="append", required=True, help="Repeat for multiple source runs; use explicit rollout bindings")
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--rollout-id", action="append", default=[])
    scope.add_argument("--rollout-list", type=Path)
    sub.add_parser("action-check").add_argument("analysis_id")
    p = sub.add_parser("action-plot", help="Inspect default plots or create a frozen plot view without rereading trajectories")
    p.add_argument("analysis_id")
    p.add_argument("--plot", action="append", default=[], choices=["progress-share", "transition-probability"])
    p.add_argument("--plot-id", help="New immutable child plot ID; omit to reuse default figures")
    p.add_argument("--aggregation", choices=["micro", "macro"], default="micro")
    p.add_argument("--format", dest="formats", choices=["png", "svg"], action="append", default=[])
    p.add_argument("--dpi", type=int, default=200)
    p = sub.add_parser("action-index")
    p.add_argument("shared_index_id")
    p.add_argument("index_id")
    p = sub.add_parser("difficulty-predict", help="Apply a published predictor to replay hidden")
    p.add_argument("index_id")
    p.add_argument("predictor")
    p.add_argument("prediction_id")
    p = sub.add_parser("difficulty-evaluate")
    p.add_argument("index_id")
    p.add_argument("predictions")
    p.add_argument("evaluation_id")
    p = sub.add_parser("difficulty-plot-parallel", help="Plot stored branch scores on a finalized trajectory")
    p.add_argument("run_id")
    p.add_argument("plot_id", nargs="?")
    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument("--rollout-id")
    scope.add_argument("--task-id")
    scope.add_argument("--list", action="store_true", help="List catalog trajectory/attempt IDs in this run")
    p.add_argument("--attempt-id")
    p.add_argument("--step-start", type=int, default=0)
    p.add_argument("--step-end", type=int)
    p = sub.add_parser("difficulty-evaluate-parallel", help="Measure score instability at fixed tool actions")
    p.add_argument("run_id")
    p.add_argument("evaluation_id")
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--rollout-id", action="append", default=[])
    scope.add_argument("--task-id", action="append", default=[])
    p.add_argument("--attempt-id")
    p.add_argument("--reasoning-identity", choices=["text", "tokens"], default="text")
    p.add_argument("--score-space", choices=["auto", "log1p", "native"], default="auto")
    p.add_argument("--multipliers", type=float, nargs="+", default=[1.25, 1.5, 2.0, 3.0])
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-unavailable", action="store_true")
    p = sub.add_parser("difficulty-train", help="Train an estimator from a standalone training YAML")
    p.add_argument("--training-config", default="difficulty/training/config.yaml")
    p = sub.add_parser("difficulty-tensorboard", help="Inspect or stop this project's managed TensorBoard")
    p.add_argument("attempt_id")
    p.add_argument("--action", choices=["status", "stop"], default="status")
    p = sub.add_parser("difficulty-ridge")
    p.add_argument("experiment_id")
    p.add_argument("--mode", choices=["dry-run", "check-data", "fit"], default="fit")
    p.add_argument("--split-id")
    p.add_argument("--dataset-id", help="Select a shared dataset; freeze its receipt and actual counts")
    p = sub.add_parser("difficulty-mlp", help="Train/evaluate a frozen single-site MLP attempt")
    p.add_argument("attempt_id")
    p.add_argument("--mode", choices=["dry-run", "check-data", "fit"], default="fit")
    p.add_argument("--split-id")
    p.add_argument("--dataset-id")
    p = sub.add_parser("difficulty-ridge-mlp", help="Train/evaluate MLP residuals over frozen published Ridge checkpoints")
    p.add_argument("attempt_id")
    p.add_argument("--mode", choices=["dry-run", "check-data", "fit"], default="fit")
    p.add_argument("--split-id")
    p.add_argument("--dataset-id")
    p = sub.add_parser("predictor-publish", help="Publish an existing estimator for online branch selection")
    p.add_argument("predictor_id")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment", type=Path)
    source.add_argument("--checkpoint", type=Path)
    p.add_argument("--target", default="reasoning_length")
    p.add_argument("--fit-scope", choices=["refit", "train"], default="refit")
    p.add_argument("--loader", default="runtime.serving.ridge_provider:load")
    p.add_argument("--sites", type=int, nargs="+")
    p.add_argument("--hidden-size", type=int)
    p.add_argument("--endpoint", default="submission")
    a = parser.parse_args(argv)
    if a.command == "difficulty-plot-parallel":
        if a.list and (a.plot_id or a.attempt_id or a.step_start != 0 or a.step_end is not None):
            parser.error("--list takes only RUN_ID; omit PLOT_ID, attempt and step range")
        if not a.list and not a.plot_id:
            parser.error("plotting requires PLOT_ID")
        if a.attempt_id and not a.task_id:
            parser.error("--attempt-id requires --task-id")
    if a.command == "difficulty-evaluate-parallel" and a.attempt_id and not a.task_id:
        parser.error("--attempt-id requires --task-id")
    if a.command == "run" and a.run_command != "create" and a.set:
        parser.error("run status/resume/reconcile accepts only the frozen run plan; use run create for overrides")
    if a.command == "run" and a.run_command == "create":
        root_only = ("generation.", "agent.", "timeouts.")
        forbidden = [item.split("=", 1)[0] for item in a.set if item.startswith(root_only)]
        if forbidden:
            parser.error("generation, agent limits and timeouts belong to root config.yaml: " + ", ".join(forbidden))
    if a.command == "rollout" and a.set:
        parser.error("rollout rejects --set; edit the unique config.yaml or resume its immutable snapshot")
    if a.command == "rollout" and a.resume and a.task_id:
        parser.error("resume uses the immutable task selection")
    explicit_run = a.analysis_run_id or getattr(a, "run_id", None)
    if a.command == "harness" and a.harness_command == "metadata-prefetch":
        if a.analysis_run_id or a.set:
            parser.error("metadata-prefetch reads root config.yaml without run overrides")
        from benchmark_evaluation.metadata import prefetch
        cfg = resolve(a.config)
        result = prefetch(cfg, a.instances, a.output, a.proxy, a.refresh)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result['status'] != 'complete':
            raise SystemExit(2)
        return
    if a.command in {"runs-migrate-to-repetition", "migration-verify"}:
        from application.migration.runs_to_repetition import apply as apply_migration, inventory, verify as verify_migration

        root = Path(a.config).resolve().parent
        if a.command == "migration-verify":
            result = verify_migration(root, a.manifest)
        elif a.dry_run:
            result = inventory(root, root / a.manifest)
        else:
            result = apply_migration(root, root / a.manifest, archive_root=a.archive_root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if a.command == "capture" and a.capture_command == "relocate-spool":
        if a.set or a.analysis_run_id:
            parser.error("capture storage relocation reads root config without run overrides")
        from hidden.relocate_spool import relocate_spool
        result = relocate_spool(resolve(a.config), destination=a.destination,
                                migration_id=a.migration_id, apply=a.apply)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    run_commands = {"rollout", "harness", "capture", "artifact", "report", "capture-recover", "capture-publish", "online-index"}
    if a.command in {"enrich", "token-trace", "trajectory-token-audit", "hidden", "dataset-index", "token-join"}:
        candidate = Path(a.config).resolve().parent / "runs" / a.dataset_id / "plan.json"
        if candidate.exists():
            explicit_run = a.dataset_id
    use_snapshot = ((a.command == "run" and a.run_command != "create" and a.run_command != "list")
                    or (a.command in run_commands and not (
                        a.command == "artifact" and getattr(a, "scope", "run") != "run"))
                    or a.analysis_run_id is not None
                    or (a.command in {"enrich", "token-trace", "trajectory-token-audit", "hidden", "dataset-index", "token-join"} and explicit_run is not None))
    if use_snapshot:
        if a.set:
            parser.error("Existing runs use frozen config; create a new run to change settings")
        from application.execution.run_plan import load_plan
        cfg = load_plan(Path(a.config).resolve().parent, explicit_run)["config"]
    else:
        cfg = resolve(a.config, a.set)
    if a.command == "skillsbench-images":
        if a.set or a.analysis_run_id:
            parser.error("skillsbench-images reads root config.yaml; no run override")
        if not cfg.get("skillsbench_images"):
            parser.error("Configure skillsbench_images in root config.yaml")
        from runtime.images.skillsbench import inventory, prepare, status, verify
        if a.action != "prepare" and a.task_id:
            parser.error("--task-id is only supported by prepare")
        if a.action == "prepare":
            result = prepare(cfg, only=a.task_id)
        else:
            result = {"plan": inventory, "status": status, "verify": verify}[a.action](cfg)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if a.action == "prepare" and result["status"] != "complete":
            raise SystemExit(2)
        if a.action == "verify" and not result["deep_verified"]:
            raise SystemExit(2)
        return
    if a.command == "wheelhouse":
        if a.analysis_run_id or a.set:
            parser.error("wheelhouse reads root config.yaml; use explicit sources, not run overrides")
        from runtime.wheelhouse import build, verify
        result = (build(cfg, a.publication_id, a.source, a.download_build_tools)
                  if a.wheelhouse_command == "build" else verify(a.path))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if a.command == "docker-storage":
        if a.set or a.analysis_run_id:
            parser.error("docker-storage reads the editable root config.yaml; no run snapshots or --set")
        from runtime.docker_storage import apply, status
        result = apply(cfg) if a.action == "apply" else status(cfg)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    catalog = Catalog(cfg["project_root"])
    def publish_artifacts(capture=False):
        if cfg.get("run_id") and not a.command.startswith("action-"):
            from application.execution.artifact import capture_publication, refresh
            if capture:
                capture_publication(cfg["project_root"], cfg["run_id"])
            refresh(cfg["project_root"], cfg["run_id"])

    if a.command == "capture-publish":
        from hidden.publish_online import publish_online
        result = publish_online(cfg, a.run_id, repair_id=a.repair_id, publication_id=a.publication_id, resolved_only=a.resolved_only, coverage_policy=a.coverage_policy)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if a.command == "capture":
        from application.execution.status import run_status
        if a.capture_command == "status":
            result = run_status(cfg["project_root"], a.run_id)
        elif a.capture_command == "archive":
            from hidden.archive_online import archive_online
            result = archive_online(cfg, a.run_id, archive_id=a.archive_id, size_only_cleanup=a.size_only_cleanup)
        elif a.capture_command == "repair":
            from hidden.repair_online import repair_online
            result = repair_online(cfg, a.run_id, rollout_ids=a.rollout_id,
                                   session_ids=a.session_id, apply=a.apply,
                                   repair_id=a.repair_id, pointwise=a.pointwise)
        else:
            from hidden.recover_online import recover_online
            result = recover_online(cfg, a.run_id)
        if a.capture_command == "recover":
            publish_artifacts(capture=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if a.command == "artifact":
        if getattr(a, "scope", "run") != "run":
            if not a.run_id:
                parser.error("Artifact scope requires an artifact ID")
            if a.scope == "action-analysis":
                from action.artifacts import show, verify
            else:
                from difficulty.artifacts import show, verify
            operation = show if a.artifact_command == "show" else verify
            result = operation(cfg["project_root"], a.scope, a.run_id, dataset_id=a.dataset_id)
        else:
            if not a.run_id:
                parser.error("Run artifact operations require RUN_ID")
            from application.execution.artifact import show, verify
            result = show(cfg["project_root"], a.run_id) if a.artifact_command == "show" else verify(cfg["project_root"], a.run_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if a.command == "harness":
        from application.execution.harness import prepare, run
        from application.execution.status import run_status

        if a.harness_command == "prepare":
            result = prepare(cfg["project_root"], a.run_id, a.rollout_id, a.source_run)
        elif a.harness_command == "run":
            result = run(cfg["project_root"], a.run_id, a.eval_id, a.gold)
        else:
            result = run_status(cfg["project_root"], a.run_id)
        publish_artifacts()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if a.command == "report":
        from application.execution.report import report
        result = report(cfg["project_root"], a.run_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if a.command == "run":
        from application.execution.run_plan import create_run, list_runs, show_run
        from application.execution.status import cancel_run, reconcile_run, run_status
        from application.execution.supervisor import launch, wait

        if a.run_command == "create":
            result = {"run_id": a.run_id, "path": str(create_run(
                cfg, a.run_id, operation=a.operation, tasks_file=a.tasks_file,
                task_ids=a.task_id, partition=a.partition, repeat_count=a.repeat_count))}
        elif a.run_command == "revise-capture-budget":
            from application.execution.capture_budget_revision import revise_capture_budget
            result = revise_capture_budget(cfg["project_root"], a.run_id, revision_id=a.revision_id,
                                           max_bytes=a.max_bytes, high=a.high, low=a.low, reason=a.reason)
        elif a.run_command == "show":
            result = show_run(cfg["project_root"], a.run_id)
        elif a.run_command == "status":
            result = run_status(cfg["project_root"], a.run_id)
        elif a.run_command == "list":
            result = list_runs(cfg["project_root"])
        elif a.run_command == "cancel":
            result = cancel_run(cfg["project_root"], a.run_id, a.reason)
        elif a.run_command == "wait":
            result = wait(cfg["project_root"], a.run_id)
        elif a.run_command in {"start", "resume"}:
            result = launch(cfg["project_root"], a.run_id, resume=a.run_command == "resume",
                            detach=a.detach, check_only=a.check_only)
        else:
            result = reconcile_run(cfg["project_root"], a.run_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("exit_code", 0) != 0:
            raise SystemExit(result["exit_code"] if isinstance(result.get("exit_code"), int) else 1)
        return
    if a.command == "config-check":
        result = cfg
    elif a.command == "predictor-publish":
        from runtime.serving.predictors import publish_predictor

        result = publish_predictor(
            cfg, a.predictor_id, experiment=a.experiment, checkpoint=a.checkpoint,
            target=a.target, fit_scope=a.fit_scope, loader=a.loader, sites=a.sites,
            hidden_size=a.hidden_size, endpoint=a.endpoint,
        )
    elif a.command == "catalog-check":
        result = catalog.verify()
    elif a.command == "runtime-check":
        from benchmark_evaluation.selection import assert_runtime_ready

        assert_runtime_ready(cfg)
        result = {"ready": True}
    elif a.command == "capture-recover":
        from hidden.recover_online import recover_online

        result = recover_online(cfg, a.run_id)
    elif a.command == "capture-reclassify":
        from hidden.reclassify_online import reclassify_online

        result = reclassify_online(cfg, a.run_id, a.session_id, wait_for_run=a.wait_for_run)
    elif a.command == "online-index":
        from difficulty.datasets.online import build_published_index

        result = str(build_published_index(cfg, a.run_id, a.index_id, a.rollout_id, a.rollout_list, publication_id=a.publication_id))
    elif a.command == "difficulty-dataset":
        if a.dataset_command == "compose":
            from difficulty.datasets.compose import compose_datasets
            result = compose_datasets(cfg["project_root"], a.dataset_id, a.source_dataset)
        else:
            from difficulty.datasets.resolved import build_resolved_selection
            result = build_resolved_selection(cfg["project_root"], a.source_run, a.index_id, a.dataset_id,
                                              a.rollout_id, a.rollout_list, a.label_normalization)
    elif a.command == "difficulty-split":
        from difficulty.training.splits import create_split
        result = create_split(cfg["project_root"], a.dataset_id, a.split_id,
            strategy=a.strategy, seed=a.seed, validation_fraction=a.validation_fraction,
            test_fraction=a.test_fraction, validation_tasks=a.validation_tasks,
            parent_experiment_id=a.parent_attempt_id, parent_receipt_sha256=a.parent_receipt_sha256)
    elif a.command == "tasks-import":
        result = catalog.register_tasks(read_json(a.list))
    elif a.command == "rollouts-import":
        from rollout.artifacts import import_list

        result = import_list(cfg["project_root"], a.list)
    elif a.command == "evaluations-import":
        from benchmark_evaluation.import_evidence import import_evaluation

        result = [import_evaluation(cfg["project_root"], spec) for spec in read_json(a.list)]
    elif a.command == "model-register":
        from core.model_identity import register

        result = {"fingerprint": register(cfg["project_root"], a.checkpoint)}
    elif a.command == "benchmark-pull":
        from catalog.benchmark import publish

        result = publish(
            cfg,
            a.cohort_id,
            count=a.count,
            proxy=a.proxy,
            workers=a.workers,
            source_revision=a.source_revision,
            results_revision=a.results_revision,
        )
    elif a.command == "rollout":
        if a.resume and a.set:
            raise ValueError("Resume uses immutable snapshot; --set requires a new run")
        import os
        if os.environ.get("DYCON_SUPERVISED") != "1" or os.environ.get("DYCON_EXECUTION_RUN") != str(Path(cfg["project_root"]) / "runs" / a.run_id):
            from application.execution.supervisor import launch
            result = launch(cfg["project_root"], a.run_id, resume=a.resume,
                            check_only=a.check_only, task_ids=a.task_id)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result.get("error") or result.get("exit_code") not in (0,):
                raise SystemExit(1)
            return
        from rollout.runner import run_manifest

        result = run_manifest(
            cfg, a.run_id, a.resume, task_ids=a.task_id, check_only=a.check_only
        )
    elif a.command == "prepare-evaluation":
        from benchmark_evaluation.selection import prepare_run

        result = str(
            prepare_run(
                {**cfg, "source_run_id": a.source_run}, a.run_id, partition=cfg["benchmark"]["partition"],
                rollout_ids=a.rollout_id,
            )
        )
    elif a.command == "evaluate":
        if a.set:
            raise ValueError("Evaluation uses the saved run configuration")
        from benchmark_evaluation.harness import evaluate
        from core.config import load_snapshot

        cfg = load_snapshot(cfg["project_root"], a.run_id)
        result = evaluate(cfg, Path(cfg["project_root"]) / "runs" / a.run_id, a.eval_id, a.gold)
    elif a.command == "enrich":
        from transformers import AutoTokenizer

        from core.model_identity import verify_model
        from enrichment.trajectory import enrich

        verify_model(cfg)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model"]["path"], local_files_only=True, trust_remote_code=cfg["model"]["trust_remote_code"]
        )
        result = str(enrich(cfg, a.dataset_id, tokenizer))
    elif a.command == "token-trace":
        from transformers import AutoTokenizer

        from enrichment.token_trace import build_token_traces

        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model"]["path"], local_files_only=True, trust_remote_code=cfg["model"]["trust_remote_code"]
        )
        result = [str(path) for path in build_token_traces(cfg, a.dataset_id, tokenizer)]
    elif a.command == "trajectory-token-audit":
        from enrichment.token_trace import audit_token_traces

        result = audit_token_traces(cfg, a.dataset_id)
    elif a.command == "hidden":
        import subprocess

        args = [
            cfg["hidden"]["vllm"]["python"],
            "-m",
            "hidden.scripts.extract_vllm",
            "--config",
            cfg["config_source"],
        ]
        if cfg.get("run_id"):
            args += ["--run-id", cfg["run_id"]]
        for override in a.set:
            args += ["--set", override]
        subprocess.run([*args, a.dataset_id, a.capture_id], cwd=cfg["project_root"], check=True)
        publish_artifacts()
        return
    elif a.command == "images-check":
        from runtime.images.nas import NasImageStore

        tasks = catalog.select_tasks(partition=cfg["benchmark"]["partition"])["tasks"]
        if not tasks:
            raise ValueError("Task scope is empty")
        result = {
            k: len(NasImageStore(cfg, t["instance"]["nas_image"]).verify(a.deep)) for k, t in tasks.items()
        }
    elif a.command == "token-join":
        from difficulty.datasets.token_join import build_token_hidden_index

        result = str(build_token_hidden_index({**cfg, "source_run_id": a.source_run or cfg.get("run_id")}, a.dataset_id, a.capture_id, a.index_id, a.rollout_id, a.rollout_list))
    elif a.command == "hidden-token-join-audit":
        from difficulty.datasets.token_join import audit_token_hidden_index

        result = audit_token_hidden_index(cfg, a.index_id)
    elif a.command == "dataset-index":
        from difficulty.datasets.index import build_index

        result = str(build_index({**cfg, "source_run_id": a.source_run or cfg.get("run_id")}, a.dataset_id, a.capture_id, a.index_id, a.rollout_id, a.rollout_list))
    elif a.command == "action-submission-audit":
        from action.analyzers.submissions import audit_submissions, check_audit, inventory
        if a.list and a.check:
            parser.error("Use --list or --check, not both")
        if a.task_count <= 0 or a.min_submitted < 0:
            parser.error("task-count must be positive and min-submitted nonnegative")
        if a.list:
            result = inventory(cfg["project_root"], a.task_count, a.min_submitted, a.reference_run)
        elif not a.analysis_id:
            parser.error("Specify a new analysis ID, or use --list")
        elif a.check:
            result = check_audit(cfg, a.analysis_id)
        else:
            result = audit_submissions(cfg, a.analysis_id, task_count=a.task_count,
                                       min_submitted=a.min_submitted, reference_run=a.reference_run)
    elif a.command == "action-branch-stats":
        from action.analyzers.branches import analyze_branches, check_branch_analysis
        if a.check:
            if (a.source_run or a.rollout_id or a.rollout_list or a.termination_reason or a.formats
                    or a.line_mode != "both" or a.progress_points != 101 or a.line_metrics):
                parser.error("--check reads frozen settings; do not pass source or plot selection arguments")
            result = check_branch_analysis(cfg, a.analysis_id)
        else:
            if not a.source_run:
                parser.error("action-branch-stats requires at least one --source-run")
            result = analyze_branches(cfg, a.analysis_id, run_id=a.source_run,
                                      rollout_ids=a.rollout_id, rollout_list=a.rollout_list,
                                      scope=a.branch_scope, termination_reason=a.termination_reason,
                                      formats=a.formats or ("png", "svg"), dpi=a.dpi,
                                      line_mode=a.line_mode, progress_points=a.progress_points,
                                      line_metrics=a.line_metrics)
    elif a.command == "action-review-length":
        from action.analyzers.review_length import run_review_length, check_review_length
        if a.check:
            result = check_review_length(cfg["project_root"], a.analysis_id)
        else:
            if not a.audit_id or not a.action_id:
                parser.error("action-review-length requires --audit-id and --action-id, or --check")
            result = run_review_length(cfg, a.analysis_id, audit_id=a.audit_id, action_id=a.action_id)
    elif a.command == "action-classify":
        from action.analyzers.trajectories import classify_trajectories

        result = classify_trajectories(cfg, a.analysis_id, a.rollout_id, a.rollout_list, run_id=a.source_run)
    elif a.command == "action-check":
        from action.analyzers.trajectories import check_analysis

        result = check_analysis(cfg, a.analysis_id)
    elif a.command == "action-plot":
        from action.artifacts import plot_analysis
        result = plot_analysis(cfg, a.analysis_id, plot_id=a.plot_id, names=a.plot,
                               aggregation=a.aggregation, formats=a.formats, dpi=a.dpi)
    elif a.command == "action-index":
        from action.datasets.index import create_action_index

        result = str(create_action_index(cfg, a.shared_index_id, a.index_id))
    elif a.command == "difficulty-predict":
        from difficulty.evaluation.predict import predict_hidden
        result = predict_hidden(cfg, a.index_id, a.predictor, a.prediction_id)
    elif a.command == "difficulty-evaluate":
        from difficulty.evaluation.run import evaluate_predictions

        result = str(evaluate_predictions(cfg, a.index_id, a.predictions, a.evaluation_id))
    elif a.command == "difficulty-train":
        from difficulty.training.entrypoint import run
        result = run(cfg, a.training_config)
    elif a.command == "difficulty-tensorboard":
        from difficulty.training.visualization import manage
        result = manage(cfg, a.attempt_id, action=a.action)
    elif a.command == "difficulty-ridge":
        import os

        threads = str((cfg["difficulty"].get("ridge_probe") or {}).get("blas_threads", 4))
        for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS"):
            os.environ[name] = threads
        from difficulty.training.ridge_probe import run_probe

        result = run_probe(cfg, a.experiment_id, mode=a.mode, split_id=a.split_id, dataset_id=a.dataset_id)
    elif a.command in {"difficulty-mlp", "difficulty-ridge-mlp"}:
        from difficulty.training.mlp_probe import run_probe
        result = run_probe(cfg, a.attempt_id, mode=a.mode, split_id=a.split_id, dataset_id=a.dataset_id,
                           kind='ridge_plus_ffn' if a.command == 'difficulty-ridge-mlp' else 'ffn_mlp')
    elif a.command == "difficulty-plot-parallel":
        from difficulty.evaluation.parallel import list_parallel_trajectories, visualize_parallel

        result = list_parallel_trajectories(cfg, a.run_id) if a.list else visualize_parallel(
            cfg, a.run_id, a.plot_id, rollout_id=a.rollout_id, task_id=a.task_id,
            attempt_id=a.attempt_id, step_start=a.step_start, step_end=a.step_end,
        )
    elif a.command == "difficulty-evaluate-parallel":
        from difficulty.evaluation.parallel_instability import evaluate_parallel_instability

        result = evaluate_parallel_instability(
            cfg, a.run_id, a.evaluation_id, rollout_ids=a.rollout_id, task_ids=a.task_id,
            attempt_id=a.attempt_id, reasoning_identity=a.reasoning_identity, score_space=a.score_space,
            multipliers=a.multipliers, bootstrap_samples=a.bootstrap_samples, seed=a.seed,
            skip_unavailable=a.skip_unavailable,
        )
    else:
        from runtime.containers.recovery import recover

        result = recover(cfg)
    if a.command == "rollout":
        import os
        from core.cancellation import request_for
        request = request_for(Path(cfg["project_root"]) / "runs" / a.run_id,
                              os.environ.get("DYCON_EXECUTION_ATTEMPT"))
        if request is not None:
            print(json.dumps({"run_id": a.run_id, "cancelled": True,
                              "partial_results": result}, ensure_ascii=False, indent=2))
            return
    publish_artifacts(capture=a.command in {"rollout", "capture-recover"})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if a.command == "rollout" and read_json(Path(cfg["project_root"])/"runs"/a.run_id/"status.json")["stage"]=="failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()


def module_main(command):
    import sys

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--set", action="append", default=[])
    known, rest = parser.parse_known_args(sys.argv[1:])
    args = ["--config", known.config]
    for override in known.set:
        args += ["--set", override]
    main([*args, command, *rest])

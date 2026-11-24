"""Light CLI; heavy model, agent and Docker dependencies are lazy imports."""

import argparse
import json
from pathlib import Path

from catalog.store import Catalog
from core.config import resolve
from core.io import read_json, sha256, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dycon")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--run-id", dest="analysis_run_id", help="Bind an analysis command to a frozen run")
    sub = parser.add_subparsers(dest="command", required=True)
    from steering.cli import register as register_steering
    register_steering(sub)
    from RL.cli import register as register_lora
    register_lora(sub)
    from RL.dataset.cli import register as register_rl_dataset
    register_rl_dataset(sub)
    harbor = sub.add_parser("harbor", help="Pinned Harbor task/runtime integration")
    hb_sub = harbor.add_subparsers(dest="harbor_command", required=True)
    hb = hb_sub.add_parser("runtime-prepare", help="Publish a pinned isolated worker and hashed dependency lock")
    hb.add_argument("publication_id")
    hb.add_argument("--source", type=Path, required=True)
    hb.add_argument("--uv", type=Path, required=True)
    hb.add_argument("--python", type=Path, required=True)
    hb = hb_sub.add_parser("runtime-verify", help="Verify the worker publication without importing Harbor")
    hb.add_argument("publication", type=Path)
    hb = hb_sub.add_parser("schema-probe", help="Parse task and single-trial schemas in the isolated worker")
    hb.add_argument("--runtime", type=Path, required=True)
    hb.add_argument("--task-path", type=Path, action="append", required=True)
    hb = hb_sub.add_parser("import", help="Freeze explicit official task packages with evaluator separation")
    hb.add_argument("snapshot_id")
    hb.add_argument("--source", type=Path, required=True)
    hb.add_argument("--benchmark", choices=["swebench_pro_v2", "terminalbench_2_1"], required=True)
    hb.add_argument("--task-id", action="append", required=True)
    hb.add_argument("--runtime", type=Path, required=True)
    hb.add_argument("--partition", choices=["train", "val", "test"], default="train")
    hb = hb_sub.add_parser("images-prepare", help="Prepare only frozen task images through the existing NAS store")
    hb.add_argument("snapshot", type=Path)
    hb = hb_sub.add_parser("profile", help="Publish a framework-independent fixed ACP driver profile")
    hb.add_argument("profile_id")
    hb.add_argument("--dependency-snapshot", type=Path, required=True)
    hb.add_argument("--bridge-address", default="127.0.0.1")
    hb = hb_sub.add_parser("qualify", help="Qualify OpenHands request roles from actual frozen Terminal-Bench tool/result evidence")
    hb.add_argument("run_id")
    hb.add_argument("--profile-id", required=True)
    hb.add_argument("--profile-ref", type=Path)
    hb = hb_sub.add_parser("preflight", help="Run real ACP/tool/network/cleanup checks without a model request")
    hb.add_argument("run_id")
    hb = hb_sub.add_parser("controls", help="Run immutable official oracle/nop or exact fresh-replay controls")
    hb.add_argument("run_id")
    hb.add_argument("--control-id", required=True)
    hb.add_argument("--agent", choices=["oracle", "nop", "gold-replay", "empty-replay"], required=True)
    hb.add_argument("--task-id", action="append", default=[])
    hb = hb_sub.add_parser("controls-audit", help="Reclassify saved Terminal-Bench controls into a new audit receipt")
    hb.add_argument("run_id")
    hb.add_argument("--control-id", required=True)
    hb.add_argument("--audit-id", required=True)
    from runtime.backends.harbor.resource_cache import register as register_resource_cache
    register_resource_cache(hb_sub)
    hb = hb_sub.add_parser("reference-publish", help="Publish an explicit derived oracle solution with unchanged verifier")
    hb.add_argument("run_id")
    hb.add_argument("--task-id", required=True)
    hb.add_argument("--reference-id", required=True)
    hb.add_argument("--source", type=Path, required=True)
    hb.add_argument("--reason", required=True)
    hb_sub.choices["controls"].add_argument("--reference-ref", type=Path,
        help="Explicit derived reference publication; single oracle task only")
    webarena = sub.add_parser("webarena", help="WebArena Verified import, runtime and frozen-run preflight")
    wa_sub = webarena.add_subparsers(dest="webarena_command", required=True)
    wa = wa_sub.add_parser("import", help="Import public tasks and isolate pinned official evaluation definitions")
    wa.add_argument("--source", type=Path, required=True)
    wa.add_argument("--subset", choices=["full", "hard"], default="full")
    wa.add_argument("--task-id", action="append", default=[])
    wa = wa_sub.add_parser("preflight", help="Run the frozen browser/site/reset/cleanup checks without model requests")
    wa.add_argument("run_id")
    wa = wa_sub.add_parser("evaluator-prepare", help="Install and pin an isolated official evaluator runtime")
    wa.add_argument("--source", type=Path)
    wa_sub.add_parser("browser-prepare", help="Build and publish the fixed browser worker image into NAS")
    wa = wa_sub.add_parser("environments-prepare", help="Publish an immutable pool from available NAS site images")
    wa.add_argument("--site", action="append", default=[])
    skillsbench = sub.add_parser("skillsbench", help="SkillsBench task import and OpenHands profile checks")
    sb_sub = skillsbench.add_subparsers(dest="skillsbench_command", required=True)
    sb = sb_sub.add_parser('job-start', help='Detach a bounded SkillsBench CLI operation with durable status/logs')
    sb.add_argument('job_id')
    sb.add_argument('--seconds', type=int, default=21600)
    sb.add_argument('args', nargs=argparse.REMAINDER)
    sb = sb_sub.add_parser('job-status', help='Read one detached validation/build status')
    sb.add_argument('job_id')
    sb = sb_sub.add_parser("import", help="Freeze explicit tasks from the pinned official checkout")
    sb.add_argument("snapshot_id")

    sb.add_argument("--oracle-only", action="store_true", help="Freeze validation-only tasks without qualifying agent capabilities")
    sb.add_argument("--task-id", action="append", required=True)
    sb.add_argument("--partition", choices=["train", "val", "test"], default="train")
    sb = sb_sub.add_parser("profile", help="Create a pinned, immutable OpenHands profile")
    sb.add_argument("profile_id")
    sb.add_argument("--bridge-advertise")
    sb.add_argument("--skill-mode", choices=["with-skill", "no-skill"])
    sb.add_argument("--network", choices=["public", "none"])
    sb.add_argument("--cpu-quota", choices=["strict", "unenforced_with_receipt"])
    sb.add_argument("--memory", action="append", default=[], metavar="TASK_ID:MIB",
                    help="Override one task container memory limit in positive integer MiB; repeatable")
    sb.add_argument("--output", action="append", default=[], metavar="TASK_ID:/absolute/path")
    sb.add_argument("--dependency-snapshot", type=Path)
    sb.add_argument("--verifier-python-index",
                    help="Explicit public verifier PyPI index frozen in the profile")
    sb.add_argument("--from-profile", type=Path,
                    help="Copy an existing profile before applying explicit verifier settings")
    sb.add_argument('--verifier-index-compatibility', choices=['generic-pypi-v1', 'generic-pypi-cpu-first-v2'])
    sb.add_argument('--verifier-resource-policy', choices=['skillsbench-verifier-resources-v1'])
    sb.add_argument('--verifier-timeout', action='append', default=[], metavar='TASK_ID:SECONDS')
    proxy = sb.add_mutually_exclusive_group()
    proxy.add_argument("--use-host-proxy", dest="use_host_proxy", action="store_true")
    proxy.add_argument("--no-use-host-proxy", dest="use_host_proxy", action="store_false")
    sb.set_defaults(use_host_proxy=None)
    sb.add_argument("--condenser-max-tokens", type=int)
    sb = sb_sub.add_parser('regrade-pddl', help='Full airport verifier replay; explicit backed-up receipt replacement')
    sb.add_argument('run_id')
    sb.add_argument('regrading_id')
    sb.add_argument('--snapshot', type=Path, required=True)
    sb.add_argument('--samples', type=Path, required=True)
    sb.add_argument('--workers', type=int, default=2)
    sb.add_argument('--replace-original-with-backup', action='store_true')
    sb.add_argument('--apply-only', action='store_true')
    sb = sb_sub.add_parser('verifier-variant', help='Publish an immutable, hash-guarded airport verifier correction')
    sb.add_argument('snapshot_id')
    sb.add_argument('--source-snapshot', type=Path, required=True)
    sb.add_argument('--policy', choices=['pddl-airport-plan-v1','skillsbench-verifier-repair-v1','agentops-py38-py312-v1'], default='pddl-airport-plan-v1')
    sb = sb_sub.add_parser('revalidate-pddl', help='Separately validate explicit historical airport plans; no run mutation')
    sb.add_argument('run_id')
    sb.add_argument('--snapshot', type=Path, required=True)
    sb.add_argument('--samples', type=Path, required=True)
    sb.add_argument('--output', type=Path, required=True)
    sb.add_argument('--validator-python', type=Path, required=True)
    sb.add_argument('--timeout', type=int, default=60)
    sb = sb_sub.add_parser('index-variant', help='Publish hash-guarded verifier index task variants')
    sb.add_argument('snapshot_id')
    sb.add_argument('--source-snapshot', type=Path, required=True)
    sb.add_argument('--policy', choices=['generic-pypi-v1', 'generic-pypi-cpu-first-v2'], default='generic-pypi-v1')
    sb = sb_sub.add_parser("dependencies-build", help="Publish a pinned offline OpenHands closure")
    sb.add_argument("snapshot_id")
    sb.add_argument("--wheelhouse", type=Path)
    sb.add_argument("--cli-archive", type=Path)
    sb.add_argument("--uv", type=Path)
    sb.add_argument("--from-snapshot", type=Path,
                    help="Derive a new condenser-capable publication from a verified closure")
    sb.add_argument("--python-runtime", type=Path, help="Bind an existing portable CPython 3.12 prefix into this new publication")
    sb.add_argument("--verifier-resources", type=Path, help="Extend an unchanged agent closure with a versioned verifier resource specification")
    sb = sb_sub.add_parser("dependencies-verify", help="Verify every published dependency byte")
    sb.add_argument("snapshot", type=Path)
    sb = sb_sub.add_parser("preflight", help="Verify the frozen benchmark/profile/runtime contract")
    sb.add_argument("run_id")
    sb.add_argument("--workers", type=int, default=1, help="Bound task sandbox audit concurrency (1-8); no model requests")
    sb = sb_sub.add_parser('environment-check', help='Audit frozen task environments without model requests')
    sb.add_argument('run_id')
    sb.add_argument('--workers', type=int, default=1)
    sb.add_argument('--capture-resources', action='store_true', help='Archive image-owned verifier build inputs with provenance')
    sb = sb_sub.add_parser('regrade-submissions', help='Replay historical submissions; optionally replace valid canonical outcomes with backups')
    sb.add_argument('run_id')
    sb.add_argument('--source-run', required=True)
    sb.add_argument('--audit-report', type=Path, required=True)
    sb.add_argument('--output', type=Path, required=True)
    sb.add_argument('--workers', type=int, default=3)
    sb.add_argument('--plan-only', action='store_true')
    sb.add_argument('--replace-original-with-backup', action='store_true',
                    help='Replace only valid replay scores in the source run after backup; preserve unavailable/error samples')
    sb = sb_sub.add_parser('audit-rewards', help='Read-only audit of accepted SkillsBench rewards below one')
    sb.add_argument('run_id')
    sb.add_argument('--output', type=Path, required=True, help='New report directory; existing paths are rejected')
    sb = sb_sub.add_parser("qualify", help="Freeze request-role evidence from a real OpenHands tool probe")
    sb.add_argument("run_id")
    sb.add_argument("profile_id")
    sb.add_argument("--profile-ref", type=Path)
    swebench = sub.add_parser("swebench", help="SWE-bench task and agent integration")
    sw_sub = swebench.add_subparsers(dest="swebench_command", required=True)
    sw = sw_sub.add_parser("openhands-profile", help="Create an immutable SWE-bench OpenHands profile")
    sw.add_argument("profile_id")
    sw.add_argument("--bridge-address", required=True)
    sw.add_argument("--dependency-snapshot", type=Path)
    sw.add_argument("--network-mode", choices=["none", "bridge"], default="none")
    sw.add_argument("--workspace", default="/testbed")
    sw.add_argument("--task-cwd", default="/testbed")
    sw.add_argument("--patch-max-bytes", type=int, default=16 * 1024 * 1024)
    sw_sub.add_parser("preflight", help="Validate a frozen SWE-bench OpenHands run").add_argument("run_id")
    sw = sw_sub.add_parser("catalog-sync", help="Regroup SWE tasks and register verified NAS images from pinned sources")
    sw.add_argument("--source", type=Path, required=True)
    sw.add_argument("--image-plan", type=Path, action="append", required=True)
    sw.add_argument("--output", type=Path, required=True)
    sw.add_argument("--apply", action="store_true")
    sw = sw_sub.add_parser("catalog-list", help="Summarize or export SWE catalog tasks by official group")
    sw.add_argument("--group", choices=["verified", "dev", "nonverified", "dev-gold-patch-failed", "verified-gold-patch-failed", "nonverified-gold-patch-failed"])
    sw.add_argument("--output", type=Path)

    sw = sw_sub.add_parser("images-repair", help="Publish frozen SWE task ownership or system-package image repairs")
    sw.add_argument("run_id")
    sw.add_argument("publication_id")
    sw.add_argument("--task-id", action="append", required=True)
    sw.add_argument("--kind", choices=["ownership", "graphics", "gettext"], required=True)

    sw = sw_sub.add_parser("images-pull", help="Resume pinned official Verified/dev missing images to NAS")
    sw.add_argument("store_id")
    sw.add_argument("--plan", type=Path, required=True)
    sw.add_argument("--proxy")
    sw.add_argument("--workers", type=int, default=4)
    sw.add_argument("--attempts", type=int, default=3)
    sw.add_argument("--check-only", action="store_true")

    sb = sb_sub.add_parser("cleanup-network", help="Reconcile a stopped run owned legacy proxy")
    sb.add_argument("run_id")
    sb = sb_sub.add_parser("audit", help="Verify real tools, native capture and winner-only execution")
    sb.add_argument("run_id")
    sb.add_argument("--require-resolved", action="store_true")
    sb.add_argument("--require-parallel", action="store_true")
    sb.add_argument("--rollout-id", action="append", default=[])
    sb = sb_sub.add_parser("auxiliary-probe", help="Validate real condenser isolation on the run-owned live model")
    sb.add_argument("run_id")
    sb.add_argument("--probe-id", required=True)
    sb = sb_sub.add_parser('verifier-control', help='Execute provenance-bound correct/incorrect submissions through the official verifier')
    sb.add_argument('run_id')
    sb.add_argument('--case', required=True, choices=['all','simpo-correct','simpo-wrong-output','simpo-missing-output','simpo-missing-env','simpo-wrong-rerun','latex-correct','latex-wrong-output','latex-missing-resource','latex-compatibility-correct','google-auto-correct','agentops-upstream-blocked','agentops-correct','agentops-wrong','agentops-missing-dependency','agentops-missing-interpreter','fuzz-drivers'])
    sb.add_argument('--retry-failed', action='store_true')
    sb = sb_sub.add_parser("oracle", help="Run the official task oracle and verifier as validation evidence")

    sb.add_argument("run_id")
    sb.add_argument("--workers", type=int, default=1, help="Bound independent official task lifecycles")
    sb.add_argument("--task-id", action="append", default=[])
    sb.add_argument("--retry-failed", action="store_true", help="Preserve failed evidence and create a fresh attempt")
    sb = sb_sub.add_parser('validation-cleanup', help='Reconcile stopped, exactly owned validation containers')
    sb.add_argument('run_id')
    sb.add_argument('--section', choices=['oracle', 'empty-control', 'offline-oracle', 'environments'], default='oracle')
    sb.add_argument('--task-id', action='append', default=[])
    sb = sb_sub.add_parser('empty-control', help='Run unchanged official verifier against a fresh unsolved task')
    sb.add_argument('run_id')
    sb.add_argument('--workers', type=int, default=1)
    sb.add_argument('--task-id', action='append', default=[])
    sb.add_argument('--retry-failed', action='store_true', help='Preserve failed evidence and create a fresh control attempt')
    sb = sb_sub.add_parser("offline-check", help="Install locked OpenHands and run official oracle/verifier with network disabled")
    sb.add_argument("run_id")
    speed = sub.add_parser("serving-token-speed", help="Query active run generation speed per GPU")
    speed.add_argument("--window", type=float, default=2.0, help="Measurement window in seconds")
    speed.add_argument("--timeout", type=float, default=3.0, help="Metrics request timeout in seconds")
    run = sub.add_parser("run", help="Create and operate immutable run plans")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    q = run_sub.add_parser("create")
    q.add_argument("run_id")
    q.add_argument("--operation", choices=["rollout", "harness", "enrichment", "hidden", "analysis"], required=True)
    q.add_argument("--tasks-file", type=Path)
    q.add_argument("--task-id", action="append", default=[])
    q.add_argument("--partition")
    q.add_argument("--repeat-count", type=int, default=1)
    q.add_argument("--from-run", help="Clone frozen rollout or harness configuration and task selection")
    q.add_argument("--reverse-task-order", action="store_true",
                   help="Reverse the source task order in a frozen run clone")
    q.add_argument("--parallel-config", type=Path,
                   help="Freeze a run-local consensus/entropy policy YAML; never edits root config")
    q = run_sub.add_parser("show")
    q.add_argument("run_id")
    q = run_sub.add_parser("status")
    q.add_argument("run_id")
    q = run_sub.add_parser("steps", help="Show the latest recorded step for each running task")
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
    q = run_sub.add_parser("revise-config", help="Preview or audit changes to any configurable run field")
    q.add_argument("run_id")
    q.add_argument("--revision-id", required=True)
    q.add_argument("--set", dest="revision_set", action="append", default=[], metavar="FIELD=YAML_VALUE")
    q.add_argument("--patch", type=Path, help="YAML mapping of configuration updates")
    q.add_argument("--dry-run", action="store_true", help="Validate and show changes without writing")
    q.add_argument("--replan", action="store_true", help="Archive old state and plan new samples for identity changes")
    q.add_argument("--recover", action="store_true", help="Recover an interrupted revision transaction")
    q.add_argument("--expect-config-hash", help="Reject applying changes to a configuration newer than the preview")
    q.add_argument("--reason", required=True)
    q = run_sub.add_parser("queue-resume", help="Resume once after another execution's GPU processes exit")
    q.add_argument("run_id")
    q.add_argument("--after-run", required=True)
    q.add_argument("--request-id", required=True)
    q.add_argument("--reason", required=True)
    q = run_sub.add_parser("add-repetitions", help="Append independent repetitions preserving existing samples")
    q.add_argument("run_id")
    q.add_argument("--request-id", required=True)
    q.add_argument("--count", type=int, help="Number of repetitions to append to every frozen task")
    q.add_argument("--reason", required=True)
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--recover", action="store_true")
    q.add_argument("--expect-plan-hash")
    q = run_sub.add_parser("retry-failed", help="Authorize one fresh physical attempt for selected failures")
    q.add_argument("run_id")
    q.add_argument("--request-id", required=True)
    q.add_argument("--exit-status", action="append", default=[])
    q.add_argument("--accepted-capture-partial", action="store_true",
                   help="Also retry accepted samples with partial capture; union with --exit-status")
    q.add_argument("--task-id", action="append", default=[])
    q.add_argument("--reason", required=True)
    q.add_argument("--sample-key", action="append", default=[], help="Restrict retry to exact task/repetition sample keys")
    q.add_argument("--expect-attempt", action="append", default=[], metavar="SAMPLE_KEY=ATTEMPT_ID", help="Reject changed terminal cursors before writing retry evidence")
    q = run_sub.add_parser("compare-skillsbench", help="Compare accepted SkillsBench repetitions and trajectories across runs")
    q.add_argument("run_ids", nargs="+", metavar="RUN_ID")
    q.add_argument("--baseline", help="Reference run, which must occur in RUN_IDs")
    q.add_argument("--repetition", action="append", default=[], metavar="RUN_ID=REPETITION",
                   help="Select one repetition per run (zero-based index or rep-000); repeat for each multi-repetition run")
    q.add_argument("--format", choices=["json", "markdown"], default="markdown")
    q.add_argument("--output", type=Path, help="Optional new report file; refuses overwrite")
    q = run_sub.add_parser("compare", help="Compare physical exits and common final submitted task/repetition pairs")
    q.add_argument("run_ids", nargs="+", metavar="RUN_ID")
    q.add_argument("--baseline", help="Reference run, which must occur in RUN_IDs")
    q.add_argument("--repetition", action="append", default=[], metavar="RUN_ID=REPETITION",
                   help="Select one repetition per run (zero-based index or rep-000); repeat for each multi-repetition run")
    q.add_argument("--format", choices=["json", "markdown"], default="json")
    q.add_argument("--output", type=Path, help="Optional new report file; refuses overwrite")
    q = run_sub.add_parser("diff-config", help="Compare configurable frozen values in grouped tables")
    q.add_argument("left_run", metavar="LEFT_RUN")
    q.add_argument("right_run", metavar="RIGHT_RUN")
    q.add_argument("--format", choices=["json", "markdown"], default="markdown")
    q.add_argument("--only-changed", action="store_true", help="Show changed configurable fields only")
    q.add_argument("--output", type=Path, help="Optional new report file; refuses overwrite")
    q = run_sub.add_parser("merge", help="Merge accepted samples from SOURCE_RUN into DESTINATION_RUN")
    q.add_argument("source_run", metavar="SOURCE_RUN")
    q.add_argument("destination_run", metavar="DESTINATION_RUN")
    run_sub.add_parser("list")
    run_sub.add_parser("active-status", help="Query all live runs and their progress")
    harness = sub.add_parser("harness", help="Prepare, run and inspect run-local harness results")
    harness_sub = harness.add_subparsers(dest="harness_command", required=True)
    q = harness_sub.add_parser("prepare")
    q.add_argument("run_id")
    q.add_argument("--rollout-id", action="append", default=[])
    q.add_argument("--source-run")
    q.add_argument("--incremental", action="store_true", help="Add newly submitted trajectories while preserving prior harness results")
    q = harness_sub.add_parser("run")
    q.add_argument("run_id")
    q.add_argument("eval_id")
    q.add_argument("--gold", action="store_true")
    q.add_argument("--incremental", action="store_true", help="Evaluate newly submitted trajectories and retain prior evaluations")
    q = harness_sub.add_parser("status")
    q.add_argument("run_id")
    q = harness_sub.add_parser("acceptance", help="Write complete task and cleanup ledger independently of audit completeness")
    q.add_argument("run_id")
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--audit-dir", type=Path)
    q = harness_sub.add_parser("repair-suite", help="Execute a frozen SWE repair validation plan without status polling")
    q.add_argument("--plan", type=Path, required=True)
    q = harness_sub.add_parser("audit", help="Write an immutable SWE-bench validity sidecar without rerunning tests")
    q.add_argument("run_id")
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--scope", choices=["all", "positive", "negative", "infra"], default="all")
    q.add_argument("--rollout-id", action="append", default=[])
    q.add_argument("--labels", type=Path, help="Explicit evidence-bound attribution findings")
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
    q = capture_sub.add_parser("spool-usage", help="Read capture spool quota, local bytes and filesystem free space")
    q.add_argument("run_id", nargs="?", help="Filter local bytes and reservations to one frozen run")
    q.add_argument("--ledger-only", action="store_true", help="Skip recursive local file scan")
    q.add_argument("--scan-seconds", type=float, default=10, help="Local scan time budget (default: 10 seconds)")
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
    q = capture_sub.add_parser("finalize", help="Retry immutable run-end capture publication without model replay")
    q.add_argument("--no-deadline", action="store_true", help="Disable the publication workflow time budget for this invocation")
    q.add_argument("run_id")
    q = capture_sub.add_parser("archive", help="Archive inactive capture spool to NAS and release local storage")
    q.add_argument("run_id")
    q.add_argument("--archive-id", required=True)
    q.add_argument("--destination", type=Path, help="Archive root; defaults to the frozen NAS archive location")
    q.add_argument("--size-only-cleanup", action="store_true", help="Complete an already copied archive using file set and size only")
    q = capture_sub.add_parser("relocate-spool", help="Relocate stopped local capture storage and rebind its disk ledger")
    q.add_argument("--destination", type=Path, required=True)
    q.add_argument("--migration-id", required=True)
    q.add_argument("--apply", action="store_true")
    artifact = sub.add_parser("artifact", help="Inspect or verify run artifacts")
    artifact_sub = artifact.add_subparsers(dest="artifact_command", required=True)
    q = artifact_sub.add_parser("show")
    q.add_argument("run_id", nargs="?")
    q.add_argument("--scope", choices=["run", "difficulty-dataset", "difficulty-split", "difficulty-attempt", "difficulty-prediction", "difficulty-evaluation", "action-analysis", "steering-dataset", "steering-analysis", "lora-attempt", "rl-dataset", "rl-cache", "rl-replay", "rl-probability", "rl-probability-plan", "rl-prepared", "rl-reference", "rl-extension"], default="run")
    q.add_argument("--storage-root", type=Path)
    q.add_argument("--with-tensors", action="store_true")
    q.add_argument("--dataset-id")
    q = artifact_sub.add_parser("verify")
    q.add_argument("run_id", nargs="?")
    q.add_argument("--scope", choices=["run", "difficulty-dataset", "difficulty-split", "difficulty-attempt", "difficulty-prediction", "difficulty-evaluation", "action-analysis", "steering-dataset", "steering-analysis", "lora-attempt", "rl-dataset", "rl-cache", "rl-replay", "rl-probability", "rl-probability-plan", "rl-prepared", "rl-reference", "rl-extension"], default="run")
    q.add_argument("--storage-root", type=Path)
    q.add_argument("--with-tensors", action="store_true")
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
    p.add_argument("--download-plan", type=Path, help="Exact PyPI release JSON plan; verifies published wheel digests")
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
    p = sub.add_parser("capture-trace", help="Freeze explicitly sealed successful native sessions")
    p.add_argument("run_id")
    p.add_argument("publication_id")
    p.add_argument("--session-id", action="append", required=True)
    p.add_argument("--eligibility", choices=["resolved_only", "positive_reward"], default="resolved_only")
    p.add_argument("--coverage-scope", choices=["all_branches", "canonical_native"], default="all_branches",
                   help="canonical_native requires complete selected native coverage; unselected branch gaps remain recorded")
    p = sub.add_parser("online-index")
    p.add_argument("--storage-root", type=Path, help="Root for new dataset/index artifacts; source run stays frozen")
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
    q.add_argument("--storage-root", type=Path, help="Root for dataset/index artifacts")
    q.add_argument("dataset_id")
    q.add_argument("--source-run", required=True)
    q.add_argument("--index-id", required=True)
    q.add_argument("--rollout-id", action="append", default=[])
    q.add_argument("--rollout-list", type=Path)
    q.add_argument("--label-normalization", choices=["raw", "trajectory_fraction"], default="raw")
    q.add_argument("--eligibility", choices=["resolved_only", "positive_reward", "accepted_terminal"], default="resolved_only")
    p = sub.add_parser("difficulty-split", help="Freeze a reusable dataset split")
    p.add_argument("dataset_id")
    p.add_argument("split_id")
    p.add_argument("--strategy", choices=["event_random_v1", "task_random_v1", "repo_source_quota_v1", "inherit_task_split_v1", "task_manifest_v1"], default="event_random_v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--validation-tasks", type=int, default=48)
    p.add_argument("--task-partitions", type=Path, help="Bind task partition manifest; validation stays inside train")
    p.add_argument("--parent-attempt-id")
    p.add_argument("--parent-receipt-sha256")
    for cmd in ["tasks-import", "rollouts-import", "evaluations-import"]:
        sub.add_parser(cmd).add_argument("list")
    sub.add_parser("model-register").add_argument("checkpoint")
    wav_images = sub.add_parser("webarena-images", help="Freeze selected Verified tasks and cache shared site images in NAS")
    wav_images.add_argument("action", choices=["plan", "prepare", "build-wikipedia", "status", "verify"])
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
    p.add_argument("--line-mode", choices=["per-run", "all-runs", "both", "none", "per-trajectory"], default=None,
                   help="Defaults to per-run with entropy mappings; otherwise both")
    p.add_argument("--line-metric", dest="line_metrics", action="append", default=[], choices=[m for m, _ in METRICS],
                   help="Repeat to choose chart metrics; per-trajectory defaults to H, H/ln(K), K, exp(H), 1-q")

    p.add_argument("--plot-layout", choices=["panels", "separate"], default=None,
                   help="Defaults to separate with entropy mappings; otherwise panels")
    p.add_argument("--entropy-mapping-comparison", action=argparse.BooleanOptionalAction, default=None,
                   help="Defaults on for consensus sources: calculate all A/B/C retained counts offline, plot separately and create HTML; use --no-entropy-mapping-comparison to opt out")
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
    p = sub.add_parser("difficulty-plot-trajectories", help="Resolved repetition counts and Ridge/truth curves grouped by task")
    p.add_argument("run_id")
    p.add_argument("plot_id")
    p.add_argument("--attempt-id", required=True)
    p.add_argument("--dataset-id", help="Frozen label dataset; optional for explicitly selected native trajectories")
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--rollout-id", action="append", default=[], help="Explicit canonical trajectory, including unresolved; repeatable")
    scope.add_argument("--rollout-list", help="Frozen JSON rollout IDs or SHA-bound entries")
    p.add_argument("--reuse-predictions-id", help="Reuse receipt/model-verified trajectory predictions while extending coverage")
    p.add_argument("--repair-id", help="Use all receipt-bound repaired action-close hidden, independently of GT")
    p.add_argument('--publication-id', help='Read a visualization-only publication without a dataset')
    p.add_argument("--target", required=True, choices=[
        "submission_remaining_reasoning_tokens_log1p", "submission_remaining_reasoning_tokens_linear",
        "submission_remaining_steps_log1p", "submission_remaining_steps_linear"])
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
    p.add_argument('--publication-id', help='Read receipt-bound derived parallel evidence')
    p.add_argument('--repair-id', help='Check the repair identity bound by --publication-id')
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
    p = sub.add_parser("difficulty-ridge-holdout", help="Dataset Ridge search with outer task holdout and full test evaluation")
    p.add_argument("evaluation_id")
    p.add_argument("--workflow-config", required=True)
    p.add_argument("--reuse-attempt-id")
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
    p = sub.add_parser("difficulty-ridge-test", help="Evaluate frozen Ridge on accepted held-out tasks")
    p.add_argument("evaluation_id")
    p.add_argument("--attempt-id", required=True)
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--task-partitions", required=True, type=Path)
    a = parser.parse_args(argv)
    if a.command in {"lora-train", "lora-prepare", "lora-status", "lora-tensorboard", "lora-advantages", "lora-memory"} or (a.command == "artifact" and a.scope == "lora-attempt"):
        if a.set or a.analysis_run_id:
            parser.error("LoRA uses an independent training YAML; do not pass --set or --run-id")
        from RL.cli import execute as execute_lora
        try:
            result = execute_lora(a, resolve(a.config))
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            parser.error(str(exc))
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return

    if a.command == "rl-dataset" or (a.command == "artifact" and a.scope in {"rl-dataset", "rl-cache", "rl-replay", "rl-probability", "rl-probability-plan", "rl-prepared", "rl-reference", "rl-extension"}):
        if a.set or a.analysis_run_id:
            parser.error("RL datasets use explicit frozen YAML/source bindings; do not pass --set or --run-id")
        try:
            if a.command == "artifact":
                if a.run_id is None: raise ValueError("Artifact scope requires an artifact ID")
                from RL.dataset.artifacts import execute_artifact
                result = execute_artifact(resolve(a.config), a.scope, a.run_id, a.storage_root,
                                          deep=a.artifact_command == "verify")
            else:
                from RL.dataset.cli import execute as execute_rl_dataset
                result = execute_rl_dataset(a, resolve(a.config))
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return

    if (a.command in {"steering-dataset", "steering-visualize"} or
            (a.command == "artifact" and a.scope.startswith("steering-"))):
        if a.set or a.analysis_run_id:
            parser.error("Offline steering reads explicit frozen sources; do not pass --set or --run-id")
        from steering.cli import execute as execute_steering
        try:
            result = execute_steering(a)
        except (ValueError, OSError, KeyError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("status") == "unavailable":
            raise SystemExit(2)
        return
    if a.command == "difficulty-plot-parallel":
        if a.list and (a.plot_id or a.attempt_id or a.step_start != 0 or a.step_end is not None or a.publication_id or a.repair_id):
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
        root_only = ("generation.seed", "timeouts.", "agent.step_limit", "agent.cost_limit",
                     "agent.loop_", "agent.max_consecutive_format_errors")
        run_identity = {"agent.backend", "agent.profile_ref", "benchmark.backend", "benchmark.snapshot_ref"}
        forbidden = [item.split("=", 1)[0] for item in a.set
                     if item.startswith(root_only) and item.split("=", 1)[0] not in run_identity]
        if forbidden:
            parser.error("generation, agent limits and timeouts belong to root config.yaml: " + ", ".join(forbidden))
    if a.command == "rollout" and a.set:
        parser.error("rollout rejects --set; edit the unique config.yaml or resume its immutable snapshot")
    if a.command == "rollout" and a.resume and a.task_id:
        parser.error("resume uses the immutable task selection")
    if a.command == "run" and a.run_command == "compare":
        if a.analysis_run_id:
            parser.error("run compare takes explicit run IDs; do not pass --run-id")
        from application.execution.comparison import compare_runs, format_markdown
        try:
            repetitions = {}
            for selector in a.repetition:
                run_id, separator, repetition = selector.partition("=")
                if not separator or not run_id or not repetition:
                    raise ValueError("--repetition requires RUN_ID=INDEX or RUN_ID=rep-000")
                if run_id in repetitions:
                    raise ValueError(f"Duplicate repetition selector for run: {run_id}")
                repetitions[run_id] = repetition
            result = compare_runs(Path(a.config).resolve().parent, a.run_ids, baseline=a.baseline,
                                  repetitions=repetitions)
            rendered = (format_markdown(result) if a.format == "markdown"
                        else json.dumps(result, ensure_ascii=False, indent=2))
            if a.output:
                with a.output.open("x", encoding="utf-8") as stream:
                    stream.write(rendered + "\n")
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        print(rendered)
        return
    if a.command == "run" and a.run_command == "compare-skillsbench":
        if a.analysis_run_id:
            parser.error("run compare-skillsbench takes explicit run IDs; do not pass --run-id")
        from application.execution.comparison import accepted_task_counts, format_accepted_markdown
        try:
            repetitions = {}
            for selector in a.repetition:
                run_id, separator, repetition = selector.partition("=")
                if not separator or not run_id or not repetition:
                    raise ValueError("--repetition requires RUN_ID=INDEX or RUN_ID=rep-000")
                if run_id in repetitions:
                    raise ValueError(f"Duplicate repetition selector for run: {run_id}")
                repetitions[run_id] = repetition
            result = accepted_task_counts(Path(a.config).resolve().parent, a.run_ids, baseline=a.baseline,
                                          repetitions=repetitions)
            rendered = (format_accepted_markdown(result) if a.format == "markdown"
                        else json.dumps(result, ensure_ascii=False, indent=2))
            if a.output:
                with a.output.open("x", encoding="utf-8") as stream:
                    stream.write(rendered + "\n")
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        print(rendered)
        return
    if a.command == "run" and a.run_command == "add-repetitions":
        if a.analysis_run_id or a.set:
            parser.error("run add-repetitions reads the frozen run without configuration overrides")
        from application.execution.repetitions import add_repetitions
        try:
            result = add_repetitions(Path(a.config).resolve().parent, a.run_id,
                request_id=a.request_id, count=a.count, reason=a.reason,
                dry_run=a.dry_run, recover=a.recover, expect_plan_hash=a.expect_plan_hash)
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return
    if a.command == "run" and a.run_command == "revise-config":
        if a.analysis_run_id or a.set:
            parser.error("run revise-config reads the frozen run; place --set after revise-config")
        from application.execution.config_revision import parse_updates, revise_config
        try:
            result = revise_config(Path(a.config).resolve().parent, a.run_id,
                revision_id=a.revision_id, updates=parse_updates(a.revision_set, a.patch),
                reason=a.reason, replan=a.replan, dry_run=a.dry_run, recover=a.recover,
                expect_config_hash=a.expect_config_hash)
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return
    if a.command == "run" and a.run_command == "diff-config":
        if a.analysis_run_id or a.set:
            parser.error("run diff-config takes two explicit run IDs and reads only their frozen snapshots")
        from application.execution.config_diff import compare_run_configs, format_config_diff_markdown
        try:
            report = compare_run_configs(Path(a.config).resolve().parent, a.left_run, a.right_run,
                                         only_changed=a.only_changed)
            rendered = (format_config_diff_markdown(report) if a.format == "markdown"
                        else json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
            if a.output:
                with a.output.open("x", encoding="utf-8") as stream:
                    stream.write(rendered + "\n")
        except (ValueError, OSError, KeyError) as exc:
            parser.error(str(exc))
        print(rendered)
        return
    if a.command == "run" and a.run_command == "merge":
        if a.analysis_run_id or a.set:
            parser.error("run merge reads only the two frozen run snapshots; do not pass --run-id or --set")
        from application.execution.merge import merge_runs
        try:
            result = merge_runs(Path(a.config).resolve().parent, a.source_run, a.destination_run)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if a.command == "skillsbench" and a.skillsbench_command == "regrade-pddl":
        if a.set or a.analysis_run_id:
            parser.error("regrade-pddl reads the frozen run and explicit verifier correction")
        from runtime.agents.pddl_regrading import regrade
        result = regrade(Path(a.config).resolve().parent, a.run_id, a.regrading_id,
                         a.snapshot, a.samples, workers=a.workers,
                         replace=a.replace_original_with_backup, apply_only=a.apply_only)
        print(json.dumps(result, ensure_ascii=False))
        if result['valid_executions'] != result['sample_count']:
            raise SystemExit(3)
        return
    if a.command == "skillsbench" and a.skillsbench_command == "revalidate-pddl":
        if a.set or a.analysis_run_id:
            parser.error("revalidate-pddl reads the explicit frozen run without overrides")
        if a.timeout <= 0 or a.timeout > 600:
            parser.error("--timeout must be in 1..600 seconds per submitted sample")
        from runtime.agents.pddl_revalidation import revalidate
        result = revalidate(Path(a.config).resolve().parent, a.run_id, a.snapshot,
                            a.samples, a.output, a.validator_python, timeout=a.timeout)
        print(json.dumps(result, ensure_ascii=False))
        if not result["complete"]:
            raise SystemExit(3)
        return
    if a.command == "skillsbench" and a.skillsbench_command == "audit-rewards":
        if a.set or a.analysis_run_id:
            parser.error("audit-rewards uses the explicit frozen run; do not pass --set or --run-id")
        from runtime.agents.accepted_reward_audit import audit_accepted_rewards
        try:
            result = audit_accepted_rewards(Path(a.config).resolve().parent, a.run_id, a.output)
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=__import__('sys').stderr)
            raise SystemExit(1)
        print(json.dumps(result, ensure_ascii=False))
        if not result.get("complete", False):
            raise SystemExit(3)
        return
    if a.command == "skillsbench" and a.skillsbench_command == "regrade-submissions":
        if a.set or a.analysis_run_id:
            parser.error("Regrading uses explicit frozen source and target runs without overrides")
        from runtime.agents.submission_regrading import regrade
        result = regrade(Path(a.config).resolve().parent, a.run_id, a.source_run,
                         a.audit_report, a.output, workers=a.workers, plan_only=a.plan_only,
                         replace_original=a.replace_original_with_backup)
        print(json.dumps(result, ensure_ascii=False))
        if not a.plan_only and not result.get("complete"):
            raise SystemExit(3)
        return
    explicit_run = a.analysis_run_id or getattr(a, "run_id", None)
    if a.command == "harness" and a.harness_command == "audit":
        if a.analysis_run_id or a.set:
            parser.error("harness audit reads frozen evidence without configuration overrides")
        from benchmark_evaluation.swebench_audit import audit_run
        result = audit_run(Path(a.config).resolve().parent, a.run_id, a.output,
                           rollout_ids=a.rollout_id, scope=a.scope, labels=a.labels)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["complete"]:
            raise SystemExit(3)
        return
    if a.command == "harness" and a.harness_command == "acceptance":
        from benchmark_evaluation.swebench_acceptance import collect
        result = collect(Path(a.config).resolve().parent, a.run_id, a.output, audit_dir=a.audit_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if a.command == "harness" and a.harness_command == "repair-suite":
        if a.set or a.analysis_run_id:
            parser.error("repair-suite reads the explicit plan and source frozen run")
        from benchmark_evaluation.swebench_repair_suite import run
        result = run(Path(a.config).resolve().parent, a.plan)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
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
    if a.command == "capture" and a.capture_command == "spool-usage":
        if a.set or a.analysis_run_id:
            parser.error("spool-usage uses root config or its positional frozen run; do not pass --set or --run-id")
        from hidden.spool_usage import spool_usage
        try:
            if a.run_id:
                from application.execution.run_plan import load_plan
                usage_cfg = load_plan(Path(a.config).resolve().parent, a.run_id)["config"]
            else:
                usage_cfg = resolve(a.config)
            result = spool_usage(usage_cfg, a.run_id, ledger_only=a.ledger_only, scan_seconds=a.scan_seconds)
        except (ValueError, OSError, RuntimeError, __import__('sqlite3').Error) as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=__import__('sys').stderr)
            raise SystemExit(1)
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
    if a.command == "serving-token-speed":
        from runtime.serving.token_speed import query_active_gpu_token_speed
        if a.window <= 0 or a.timeout <= 0:
            parser.error("--window and --timeout must be positive")
        print(json.dumps(query_active_gpu_token_speed(resolve(a.config, a.set)["project_root"], window_seconds=a.window, timeout=a.timeout), ensure_ascii=False, indent=2))
        return
    run_commands = {"rollout", "harness", "capture", "artifact", "report", "capture-recover", "capture-publish", "capture-trace", "online-index"}
    if a.command in {"enrich", "token-trace", "trajectory-token-audit", "hidden", "dataset-index", "token-join"}:
        candidate = Path(a.config).resolve().parent / "runs" / a.dataset_id / "plan.json"
        if candidate.exists():
            explicit_run = a.dataset_id
    use_snapshot = ((a.command == "run" and a.run_command not in {"create", "list", "active-status"})
                    or (a.command in run_commands and not (
                        a.command == "artifact" and getattr(a, "scope", "run") != "run"))
                    or (a.command == "webarena" and a.webarena_command == "preflight")
                    or (a.command == "harbor" and a.harbor_command in {"preflight", "controls", "controls-audit", "qualify"})
                    or a.analysis_run_id is not None
                    or (a.command in {"enrich", "token-trace", "trajectory-token-audit", "hidden", "dataset-index", "token-join"} and explicit_run is not None))
    if use_snapshot:
        if a.set:
            parser.error("Existing runs use frozen config; create a new run to change settings")
        from application.execution.run_plan import load_plan
        cfg = load_plan(Path(a.config).resolve().parent, explicit_run)["config"]
    else:
        # Frozen clones validate their overrides against the source snapshot in
        # create_run; the current root may use an entirely different topology.
        cloning = a.command == "run" and a.run_command == "create" and a.from_run is not None
        cfg = resolve(a.config, () if cloning else a.set)
    if a.command == "harbor":
        if a.set or a.analysis_run_id:
            parser.error("harbor registry commands use the root configuration")
        from runtime.backends.harbor.dependencies import prepare_runtime, probe_schemas, verify_runtime
        if a.harbor_command == "resource-cache":
            from runtime.backends.harbor.resource_cache import dispatch
            result = dispatch(cfg["project_root"], a, cfg)
        elif a.harbor_command == "reference-publish":
            from runtime.backends.harbor.reference_solutions import publish_reference
            result = publish_reference(cfg["project_root"], a.run_id, a.task_id, a.reference_id, a.source, a.reason)
        elif a.harbor_command == "runtime-prepare":
            result = prepare_runtime(cfg, a.publication_id, a.source, a.uv, a.python)
        elif a.harbor_command == "runtime-verify":
            result = verify_runtime(a.publication)
        elif a.harbor_command == "schema-probe":
            result = probe_schemas(cfg, a.runtime, a.task_path)
        elif a.harbor_command == "import":
            from catalog.harbor import import_tasks
            result = import_tasks(cfg, a.snapshot_id, a.source, a.benchmark, a.task_id, a.runtime, a.partition)
        elif a.harbor_command == "images-prepare":
            from runtime.images.harbor import prepare
            result = prepare(cfg, a.snapshot)
        elif a.harbor_command == "profile":
            from runtime.backends.harbor.operations import create_profile
            result = create_profile(cfg, a.profile_id, a.dependency_snapshot, a.bridge_address)
        elif a.harbor_command == "qualify":
            from runtime.agents.qualification import qualify_profile
            if cfg["execution_context"]["benchmark"] != "terminalbench_2_1":
                raise ValueError("harbor qualify requires a frozen Terminal-Bench run")
            result = qualify_profile(cfg["project_root"], a.run_id, a.profile_id, a.profile_ref)
        elif a.harbor_command == "preflight":
            from runtime.backends.harbor.operations import preflight
            result = preflight(cfg["project_root"], a.run_id)
        elif a.harbor_command == "controls-audit":
            from runtime.backends.harbor.operations import audit_controls
            result = audit_controls(cfg["project_root"], a.run_id, a.control_id, a.audit_id)
        else:
            from runtime.backends.harbor.operations import controls
            result = controls(cfg["project_root"], a.run_id, a.control_id, a.agent, a.task_id, reference_ref=a.reference_ref)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if a.harbor_command == "preflight" and not result["passed"]:
            raise SystemExit(2)
        return
    if a.command == "webarena":
        if a.set or a.analysis_run_id:
            parser.error("webarena uses root configuration for registry operations and frozen configuration for preflight")
        if a.webarena_command == "import":
            from catalog.webarena import import_tasks
            result = import_tasks(cfg, a.source, subset=a.subset, task_ids=a.task_id)
        elif a.webarena_command == "evaluator-prepare":
            from benchmark_evaluation.webarena import prepare_runtime
            result = prepare_runtime(cfg, source=a.source)
        elif a.webarena_command == "browser-prepare":
            from runtime.images.webarena_browser import prepare_browser
            result = prepare_browser(cfg)
        elif a.webarena_command == "environments-prepare":
            from runtime.images.webarena_environments import prepare_environments
            result = prepare_environments(cfg, sites=a.site)
        else:
            from runtime.containers.webarena_adapter import preflight
            result = preflight(Path(cfg["project_root"]), a.run_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if a.webarena_command == "preflight" and not result.get("passed"):
            raise SystemExit(2)
        return
    if a.command == "skillsbench":
        if a.set or a.analysis_run_id:
            parser.error("skillsbench registry commands read the editable root config; no run override")
        from core.backends import capabilities
        if a.skillsbench_command == 'job-start':
            from runtime.agents.validation_jobs import start
            print(json.dumps(start(cfg['project_root'], a.job_id, a.args, a.seconds)))
        elif a.skillsbench_command == 'job-status':
            from runtime.agents.validation_jobs import status
            print(json.dumps(status(cfg['project_root'], a.job_id)))
        elif a.skillsbench_command == "profile":
            from runtime.agents.openhands import OpenHandsProfile, validate_profile
            from core.io import identifier
            identifier(a.profile_id)
            source = validate_profile(read_json(a.from_profile)) if a.from_profile else {}
            # A new profile must establish its own tool/request evidence.
            source.update(request_roles='unqualified', qualification=None)
            if a.bridge_advertise:
                source.update(bridge_listen=a.bridge_advertise, bridge_advertise=a.bridge_advertise)
            if not source.get('bridge_advertise'):
                parser.error('--bridge-advertise or --from-profile is required')
            for key in ('skill_mode', 'network', 'cpu_quota', 'use_host_proxy',
                        'condenser_max_tokens', 'verifier_python_index', 'verifier_index_compatibility', 'verifier_resource_policy'):
                value = getattr(a, key)
                if value is not None:
                    source[key] = value
            if not a.from_profile and a.use_host_proxy is None:
                source['use_host_proxy'] = source.get('network', 'public') == 'public'
            if a.network == 'none' and a.use_host_proxy is None:
                source['use_host_proxy'] = False
            if a.network == 'none' and a.verifier_python_index is None:
                source['verifier_python_index']=None
                source['verifier_index_compatibility']=None
            outputs = {key: list(value) for key, value in source.get('output_paths', {}).items()}
            for value in a.output:
                if ':' not in value:
                    parser.error('--output must be TASK_ID:/absolute/path')
                task, path = value.split(':', 1)
                outputs.setdefault(task, []).append(path)
            source['output_paths'] = outputs
            overrides = dict(source.get('verifier_timeout_overrides', {}))
            for value in a.verifier_timeout:
                task, separator, seconds = value.partition(':')
                if not separator or not seconds.isdecimal() or int(seconds) < 1:
                    parser.error('--verifier-timeout must be TASK_ID:POSITIVE_SECONDS')
                identifier(task)
                overrides[task] = int(seconds)
            source['verifier_timeout_overrides'] = overrides
            memory = dict(source.get('memory_overrides', {}))
            for value in a.memory:
                task, separator, mib = value.partition(':')
                if not separator or not mib.isdecimal() or int(mib) < 1:
                    parser.error('--memory must be TASK_ID:POSITIVE_MIB')
                identifier(task)
                memory[task] = int(mib)
            source['memory_overrides'] = memory
            if a.dependency_snapshot:
                source.update(dependency_snapshot=str(a.dependency_snapshot.resolve()),
                              skip_agent_install=True, dependency_snapshot_sha256=None)
            if source.get('dependency_snapshot'):
                from runtime.agents.dependencies import verify_snapshot
                dependency_path = Path(source['dependency_snapshot'])
                verify_snapshot(dependency_path)
                dependency_hash = sha256(dependency_path / 'manifest.json')
                if source.get('dependency_snapshot_sha256') not in (None, dependency_hash):
                    raise ValueError('Source profile dependency manifest changed')
                source['dependency_snapshot_sha256'] = dependency_hash
            profile = validate_profile(source)
            target = Path(cfg["project_root"]) / "runtime/agent-profiles" / a.profile_id / "manifest.json"
            if target.exists() and validate_profile(read_json(target)) != profile:
                raise FileExistsError("OpenHands profile is immutable; choose a new profile ID")
            if not target.exists():
                write_json(target, profile)
            cfg["benchmark"].update(backend="skillsbench")
            cfg["agent"].update(backend="openhands", profile_ref=str(target.relative_to(cfg["project_root"])))
            from runtime.agents.openhands import BENCHFLOW_VERSION, BENCHFLOW_WHEEL_SHA256
            print(json.dumps({"profile_ref": str(target.relative_to(cfg["project_root"])),
                "profile_sha256": sha256(target), "benchflow_version": BENCHFLOW_VERSION,
                "benchflow_wheel_sha256": BENCHFLOW_WHEEL_SHA256,
                "cpu_quota_policy": profile["cpu_quota"],
                "request_roles_qualified": False, "capture_parallel_eligible": False},
                ensure_ascii=False, indent=2))
        elif a.skillsbench_command == 'verifier-variant':
            if a.set or a.analysis_run_id:
                parser.error("verifier-variant does not accept configuration overrides")
            if a.policy == 'agentops-py38-py312-v1':
                from catalog.verifier_repairs import publish_variant
                result=publish_variant(cfg,a.snapshot_id,a.source_snapshot,
                    rules_path=Path(cfg['project_root'])/'catalog/verifier_patches/agentops-py38-py312-v1.json')
                print(json.dumps(result,indent=2))
                return
            if a.policy == 'skillsbench-verifier-repair-v1':
                from catalog.verifier_repairs import publish_variant
            else:
                from catalog.skillsbench_verifier import publish_variant
            print(json.dumps(publish_variant(cfg, a.snapshot_id, a.source_snapshot), indent=2))
        elif a.skillsbench_command == 'index-variant':
            from catalog.skillsbench_index import publish_variant
            print(json.dumps(publish_variant(cfg, a.snapshot_id, a.source_snapshot, a.policy), indent=2))
        elif a.skillsbench_command == "dependencies-build":
            from runtime.agents.dependencies import build_snapshot
            result = build_snapshot(cfg, a.snapshot_id, a.wheelhouse, a.cli_archive, a.uv, a.python_runtime, from_snapshot=a.from_snapshot, verifier_resources=a.verifier_resources)
            print(json.dumps({"path": result["path"], "receipt_sha256": result["manifest"]["receipt_sha256"]}))
        elif a.skillsbench_command == "dependencies-verify":
            from runtime.agents.dependencies import verify_snapshot
            result = verify_snapshot(a.snapshot)
            print(json.dumps({"verified": True, "files": len(result["files"]), "receipt_sha256": result["receipt_sha256"]}))
        elif a.skillsbench_command == "import":
            from catalog.skillsbench import import_tasks
            print(json.dumps(import_tasks(cfg, a.snapshot_id, a.task_id, a.partition, oracle_only=a.oracle_only), ensure_ascii=False, indent=2))
        elif a.skillsbench_command == 'verifier-control':
            from runtime.agents.skillsbench import run_oracles
            from runtime.agents.verifier_controls import task_for_case
            if a.case=='all':
                from runtime.agents.verifier_controls import CASES
                results=[]
                for case in sorted(CASES):
                    result=run_oracles(cfg,a.run_id,task_ids=[task_for_case(case)],
                        retry_failed=a.retry_failed,control_case=case)
                    results.append({'case':case,'verified':result['verified']})
                print(json.dumps({'controls':results,'verified':all(r['verified'] for r in results)}))
                if not all(r['verified'] for r in results):raise SystemExit(2)
                return
            result = run_oracles(cfg, a.run_id, task_ids=[task_for_case(a.case)],
                retry_failed=a.retry_failed, control_case=a.case)
            print(json.dumps({k:v for k,v in result.items() if k!='results'}, indent=2))
            if not result['verified']:
                raise SystemExit(2)
        elif a.skillsbench_command == "preflight":
            from runtime.agents.skillsbench import preflight
            result = preflight(cfg["project_root"], a.run_id, workers=a.workers)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result["blockers"]:
                raise SystemExit(2)
        elif a.skillsbench_command == 'environment-check':
            from application.execution.run_plan import load_plan
            from runtime.agents.skillsbench import audit_task_environments
            loaded = load_plan(cfg['project_root'], a.run_id)
            frozen = loaded['config']
            if frozen.get('execution_context', {}).get('benchmark') != 'skillsbench':
                raise ValueError('Environment audit requires a frozen SkillsBench run')
            result = audit_task_environments(frozen, a.run_id, workers=a.workers, capture_resources=a.capture_resources)
            print(json.dumps({'run_id': a.run_id, 'task_count': result['task_count'],
                              'verified': result['verified'], 'failed_tasks': result['failed_tasks']}, indent=2))
            if not result['verified']:
                raise SystemExit(2)
        elif a.skillsbench_command == "qualify":
            from runtime.agents.qualification import qualify_profile
            print(json.dumps(qualify_profile(cfg["project_root"], a.run_id, a.profile_id, a.profile_ref), indent=2))
        elif a.skillsbench_command == "cleanup-network":
            from runtime.agents.network import cleanup_network
            print(json.dumps(cleanup_network(cfg["project_root"], a.run_id), indent=2))
        elif a.skillsbench_command == "audit":
            from runtime.agents.validation import audit_run
            result = audit_run(cfg["project_root"], a.run_id, a.require_resolved, a.require_parallel, a.rollout_id)
            print(json.dumps(result, indent=2))
            if not result['verified']:
                raise SystemExit(2)
        elif a.skillsbench_command == "auxiliary-probe":
            from runtime.agents.probes import auxiliary_probe
            print(json.dumps(auxiliary_probe(cfg["project_root"], a.run_id, a.probe_id), indent=2))
        elif a.skillsbench_command == 'validation-cleanup':
            from runtime.agents.validation_cleanup import reconcile_validation
            print(json.dumps(reconcile_validation(cfg, a.run_id, section=a.section, task_ids=a.task_id), indent=2))
        elif a.skillsbench_command in {"oracle", "offline-check", "empty-control"}:
            from runtime.agents.skillsbench import run_oracles
            oracle_options = {}
            if a.skillsbench_command == "oracle" and (a.workers != 1 or a.task_id or a.retry_failed):
                oracle_options = dict(workers=a.workers, task_ids=a.task_id, retry_failed=a.retry_failed)
            if a.skillsbench_command == 'empty-control':
                oracle_options = dict(workers=a.workers, task_ids=a.task_id, retry_failed=a.retry_failed, empty_control=True)
            result = (run_oracles(cfg, a.run_id, offline=True) if a.skillsbench_command == "offline-check"
                      else run_oracles(cfg, a.run_id, **oracle_options))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result['valid_oracles'] != result['oracle_count'] or not result['oracle_count']:
                raise SystemExit(2)
        elif a.skillsbench_command == "gpu-smoke":
            from runtime.agents.skillsbench import capture_gpu_smoke
            print(json.dumps(capture_gpu_smoke(cfg["project_root"], a.run_id), ensure_ascii=False, indent=2))
        return
    if a.command == "webarena-images":
        if a.set or a.analysis_run_id:
            parser.error("webarena-images reads root config.yaml; no run override")
        if not cfg.get("webarena_images"):
            parser.error("Configure webarena_images in root config.yaml")
        from runtime.images import webarena
        result = (webarena.status(cfg, deep=a.action == "verify")
                  if a.action in {"status", "verify"} else getattr(webarena, a.action.replace("-", "_"))(cfg))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if a.action != "plan" and result["status"] != "complete":
            raise SystemExit(2)
        return
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
        result = (build(cfg, a.publication_id, a.source, a.download_build_tools, a.download_plan)
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
            summary = Path(cfg['project_root']) / 'runs' / a.run_id / 'capture-publication.json'
            result = ({'run_id': a.run_id, 'capture_publication': read_json(summary)} if summary.exists()
                      else run_status(cfg["project_root"], a.run_id))
        elif a.capture_command == "finalize":
            from application.execution.capture_publication import retry_capture_publication
            result = retry_capture_publication(cfg, a.run_id, no_deadline=a.no_deadline)
        elif a.capture_command == "archive":
            from hidden.archive_online import archive_online
            result = archive_online(cfg, a.run_id, archive_id=a.archive_id,
                                    size_only_cleanup=a.size_only_cleanup, destination=a.destination)
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
            result = prepare(cfg["project_root"], a.run_id, a.rollout_id, a.source_run, a.incremental)
        elif a.harness_command == "run":
            result = run(cfg["project_root"], a.run_id, a.eval_id, a.gold, a.incremental)
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
    if a.command == "swebench":
        if a.set or a.analysis_run_id:
            parser.error("swebench profile/preflight commands use the root or frozen configuration")
        if a.swebench_command in {"catalog-sync", "catalog-list"}:
            from catalog.swebench_sync import sync, list_tasks
            if a.swebench_command == "catalog-sync":
                result = sync(cfg, source_path=a.source, image_plans=a.image_plan, output=a.output, apply=a.apply)
            else:
                result = list_tasks(cfg, group=a.group, output=a.output)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if a.swebench_command == "images-repair":
            from runtime.images.swebench_repair import prepare
            result = prepare(cfg["project_root"], a.run_id, a.publication_id, a.task_id, a.kind)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if a.swebench_command == "images-pull":
            from runtime.images.swebench_pull import pull
            result = pull(cfg, a.store_id, plan_path=a.plan, proxy=a.proxy,
                          workers=a.workers, attempts=a.attempts, check_only=a.check_only)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if a.swebench_command == "openhands-profile":
            from core.io import identifier
            from runtime.agents.swebench_openhands import SWEOpenHandsProfile, validate_profile
            identifier(a.profile_id)
            dependency = str(a.dependency_snapshot.resolve()) if a.dependency_snapshot else None
            if a.dependency_snapshot:
                from runtime.agents.dependencies import verify_snapshot
                verify_snapshot(a.dependency_snapshot)
            profile = validate_profile(SWEOpenHandsProfile(
                bridge_advertise=a.bridge_address, bridge_listen=a.bridge_address,
                dependency_snapshot=dependency, workspace=a.workspace, task_cwd=a.task_cwd,
                network_mode=a.network_mode, patch_max_bytes=a.patch_max_bytes,
                request_roles="unqualified").model_dump())
            target = Path(cfg["project_root"]) / "runtime/agent-profiles" / a.profile_id / "manifest.json"
            if target.exists() and validate_profile(read_json(target)) != profile:
                raise FileExistsError("SWE OpenHands profile is immutable; choose a new profile ID")
            if not target.exists():
                write_json(target, profile)
            print(json.dumps({"profile_ref": str(target.relative_to(cfg["project_root"])),
                              "profile_sha256": sha256(target), "request_roles_qualified": False}, indent=2))
        else:
            from application.execution.run_plan import load_plan
            from runtime.backends.registry import adapter_for
            loaded = load_plan(cfg["project_root"], a.run_id)
            adapter = adapter_for(loaded["config"], loaded["config"].get("selection", {}).get("tasks"))
            context = adapter.open_run(loaded["config"], a.run_id)
            try:
                result = adapter.preflight(context, loaded["config"]["selection"]["tasks"])
            finally:
                adapter.close_run(context)
            print(json.dumps(result.receipt, ensure_ascii=False, indent=2))
            if not result.passed:
                raise SystemExit(2)
        return
    if a.command == "run":
        from application.execution.run_plan import create_run, list_runs, show_run
        from application.execution.status import (
            active_run_statuses,
            cancel_run,
            reconcile_run,
            run_status,
            running_step_status,
        )
        from application.execution.supervisor import launch, wait

        if a.run_command == "create":
            result = {"run_id": a.run_id, "path": str(create_run(
                cfg, a.run_id, operation=a.operation, tasks_file=a.tasks_file,
                task_ids=a.task_id, partition=a.partition, repeat_count=a.repeat_count,
                from_run=a.from_run, clone_overrides=a.set, parallel_config=a.parallel_config,
                reverse_task_order=a.reverse_task_order))}
        elif a.run_command == "queue-resume":
            from application.execution.queued_resume import enqueue
            result = enqueue(cfg["project_root"], a.run_id, after_run=a.after_run,
                             request_id=a.request_id, reason=a.reason)
        elif a.run_command == "retry-failed":
            from application.execution.failed_retry import request_failed_retry
            result = request_failed_retry(cfg["project_root"], a.run_id, request_id=a.request_id,
                exit_statuses=a.exit_status, task_ids=a.task_id, sample_keys=a.sample_key,
                expected_attempts=dict(v.split('=',1) for v in a.expect_attempt), reason=a.reason,
                accepted_capture_partial=a.accepted_capture_partial)
        elif a.run_command == "show":
            result = show_run(cfg["project_root"], a.run_id)
        elif a.run_command == "status":
            result = run_status(cfg["project_root"], a.run_id)
        elif a.run_command == "steps":
            result = running_step_status(cfg["project_root"], a.run_id)
        elif a.run_command == "list":
            result = list_runs(cfg["project_root"])
        elif a.run_command == "active-status":
            result = active_run_statuses(cfg["project_root"])
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
        if isinstance(result, dict) and result.get("exit_code", 0) != 0:
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
    elif a.command == "capture-trace":
        from hidden.publish_finalized import publish_finalized
        result = publish_finalized(cfg, a.run_id, a.publication_id, a.session_id, eligibility=a.eligibility,
                                   coverage_scope=a.coverage_scope)
    elif a.command == "online-index":
        from difficulty.datasets.online import build_published_index

        result = str(build_published_index(cfg, a.run_id, a.index_id, a.rollout_id, a.rollout_list, publication_id=a.publication_id, storage_root=a.storage_root))
    elif a.command == "difficulty-dataset":
        if a.dataset_command == "compose":
            from difficulty.datasets.compose import compose_datasets
            result = compose_datasets(cfg["project_root"], a.dataset_id, a.source_dataset)
        else:
            from difficulty.datasets.resolved import build_resolved_selection
            result = build_resolved_selection(cfg["project_root"], a.source_run, a.index_id, a.dataset_id,
                                              a.rollout_id, a.rollout_list, a.label_normalization, eligibility=a.eligibility,
                                              storage_root=a.storage_root or cfg.get("paths", {}).get("artifact_roots", {}).get("difficulty_dataset"))
    elif a.command == "difficulty-split":
        from difficulty.training.splits import create_split
        result = create_split(cfg["project_root"], a.dataset_id, a.split_id,
            strategy=a.strategy, seed=a.seed, validation_fraction=a.validation_fraction,
            test_fraction=a.test_fraction, validation_tasks=a.validation_tasks,
            parent_experiment_id=a.parent_attempt_id, parent_receipt_sha256=a.parent_receipt_sha256,
            task_partitions=a.task_partitions)
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
        # The supervisor may own an external artifact-root run (for example
        # /data3/.../runs/<id>), so comparing against project_root/runs is
        # invalid. The supervisor pins DYCON_EXECUTION_RUN and attempt identity.
        supervised = (os.environ.get("DYCON_SUPERVISED") == "1"
                      and os.environ.get("DYCON_EXECUTION_ATTEMPT")
                      and os.environ.get("DYCON_EXECUTION_RUN"))
        if not supervised:
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
                    or a.line_mode is not None or a.progress_points != 101 or a.line_metrics
                    or a.plot_layout is not None or a.entropy_mapping_comparison is not None):
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
                                      line_metrics=a.line_metrics, plot_layout=a.plot_layout,
                                      entropy_mapping_comparison=a.entropy_mapping_comparison)
    elif a.command == "action-review-length":
        from action.analyzers.review_length import run_review_length, check_review_length
        if a.check:
            result = check_review_length(cfg["project_root"], a.analysis_id)
        else:
            if not a.audit_id or not a.action_id:
                parser.error("action-review-length requires --audit-id and --action-id, or --check")
            result = run_review_length(cfg, a.analysis_id, audit_id=a.audit_id, action_id=a.action_id)
    elif a.command == "action-classify":
        from core.backends import capabilities
        from application.execution.evidence import RunEvidence
        sources = [a.source_run] if isinstance(a.source_run, str) else (a.source_run or [])
        for source_run in sources:
            source_evidence = RunEvidence(cfg["project_root"], source_run).read()
            if (any(t.get("benchmark") == "skillsbench" for t in source_evidence["tasks"].values())
                    or any(r.get("agent") == "openhands" for r in source_evidence["rollouts"].values())):
                capabilities({"benchmark": {"backend": "skillsbench"},
                              "agent": {"backend": "openhands"}}, operation="action")
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
    elif a.command == "difficulty-ridge-test":
        from difficulty.evaluation.accepted_test import run
        result = run(cfg, a.evaluation_id, attempt_id=a.attempt_id, dataset_id=a.dataset_id,
                     task_partitions=a.task_partitions)
    elif a.command == "difficulty-ridge-holdout":
        from difficulty.evaluation.holdout import run
        result = run(cfg, a.evaluation_id, workflow_config=a.workflow_config, reuse_attempt_id=a.reuse_attempt_id)
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
    elif a.command == "difficulty-plot-trajectories":
        from difficulty.evaluation.trajectories import visualize_trajectories
        result = visualize_trajectories(cfg, a.run_id, a.plot_id, attempt_id=a.attempt_id,
                                        dataset_id=a.dataset_id, target=a.target, repair_id=a.repair_id,
                                        reuse_predictions_id=a.reuse_predictions_id,
                                        rollout_ids=a.rollout_id, rollout_list=a.rollout_list,
                                        publication_id=a.publication_id)
    elif a.command == "difficulty-plot-parallel":
        from difficulty.evaluation.parallel import list_parallel_trajectories, visualize_parallel

        result = list_parallel_trajectories(cfg, a.run_id) if a.list else visualize_parallel(
            cfg, a.run_id, a.plot_id, rollout_id=a.rollout_id, task_id=a.task_id,
            attempt_id=a.attempt_id, step_start=a.step_start, step_end=a.step_end,
            publication_id=a.publication_id,repair_id=a.repair_id,
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

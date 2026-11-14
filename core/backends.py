"""Versioned benchmark/agent capabilities and immutable execution identities.

This module deliberately imports no agent SDK. Training provenance never chooses
an execution backend; only the frozen run configuration does.
"""
from __future__ import annotations

import copy
from pathlib import Path

from core.io import hash_json, read_json, sha256

SWE_TASK_BENCHMARKS = frozenset({
    "swebench", "swebench_verified", "swebench_lite", "swebench_full",
    "SWE-bench_Verified", "SWE-bench_NonVerified600", "SWE-bench_Lite", "SWE-bench",
    "legacy-tactRefractor-swebench",
    "fixture",  # Explicit synthetic family used by the repository's regression fixtures.
})


def benchmark_family(name):
    return "swebench" if name in SWE_TASK_BENCHMARKS else name


CAPABILITIES = {
    ("swebench", "minisweagent"): {"subject": "patch", "action_analysis": True},
    ("swebench", "openhands"): {"subject": "patch", "action_analysis": False},
    ("skillsbench", "openhands"): {"subject": "workspace", "action_analysis": False},
    ("webarena_verified", "minisweagent"): {"subject": "web_trace", "action_analysis": False},
    ("swebench_pro_v2", "openhands"): {"subject": "patch", "action_analysis": False},
    ("terminalbench_2_1", "openhands"): {"subject": "live_state", "action_analysis": False},
}


def identity(cfg):
    return (cfg.get("benchmark", {}).get("backend", "swebench"),
            cfg.get("agent", {}).get("backend", "minisweagent"))


def capabilities(cfg, operation=None):
    pair = identity(cfg)
    if pair not in CAPABILITIES:
        raise ValueError(f"Unsupported benchmark/agent combination: {pair}")
    result = copy.deepcopy(CAPABILITIES[pair])
    if operation == "action" and not result["action_analysis"]:
        raise ValueError(f"action analysis does not support {pair[0]}/{pair[1]}")
    if pair[1] == "openhands" and cfg.get("steering", {}).get("enabled"):
        raise ValueError("OpenHands steering has not been qualified")
    return result


def execution_binding(cfg, tasks=None):
    """Resolve the explicit runtime without importing any third-party SDK."""
    from runtime.backends.registry import from_snapshot
    return from_snapshot(cfg, tasks)


def execution_context(cfg, tasks):
    capabilities(cfg)
    benchmark, agent = identity(cfg)
    if benchmark == "swebench" and agent == "openhands":
        from runtime.agents.swebench_openhands import execution_context as swe_context
        return swe_context(cfg, tasks)
    if benchmark in {"swebench_pro_v2", "terminalbench_2_1"}:
        from catalog.harbor import execution_context as harbor_context
        return harbor_context(cfg, tasks)
    if benchmark in {"swebench", "webarena_verified"}:
        expected = "swebench" if benchmark == "swebench" else "webarena_verified"
        if any(benchmark_family(t["benchmark"]) != expected for t in tasks.values()):
            raise ValueError(f"{benchmark} backend cannot execute tasks from another benchmark family")
        if benchmark == "swebench":
            return None  # Preserve legacy run and task hashes byte for byte.
        options = cfg.get("webarena")
        if not options:
            raise ValueError("WebArena Verified requires webarena runtime configuration")
        root = Path(cfg["project_root"]).resolve()

        def project_file(ref):
            path = (root / ref).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("WebArena references must name existing project files")
            return path

        from runtime.containers.webarena_adapter import validate_pool_manifest, validate_reference
        browser_path, pool_path = project_file(options["browser_image_ref"]), project_file(options["environment_pool_ref"])
        browser_ref = validate_reference(read_json(browser_path))
        pool = validate_pool_manifest(read_json(pool_path))
        snapshot_ref = cfg["benchmark"].get("snapshot_ref")
        if not snapshot_ref:
            raise ValueError("WebArena Verified requires benchmark.snapshot_ref")
        snapshot_path = project_file(snapshot_ref)
        snapshot = read_json(snapshot_path)
        if snapshot.get("schema_version") != 1 or snapshot.get("benchmark") != benchmark:
            raise ValueError("Invalid WebArena Verified snapshot")
        if snapshot.get("revision") != pool.get("revision"):
            raise ValueError("WebArena environment pool and task snapshot revisions differ")
        for key, task in tasks.items():
            if task["benchmark"] != benchmark or task["revision"] != snapshot["revision"] or snapshot["tasks"].get(key) != task:
                raise ValueError("Task does not belong to frozen WebArena snapshot")
            if not set(task["instance"]["sites"]) <= set(pool["sites"]):
                raise ValueError("WebArena task requires sites missing from the frozen environment pool")
        evaluator = snapshot.get("evaluator")
        if not evaluator or sha256(project_file(evaluator["path"])) != evaluator["sha256"]:
            raise ValueError("Frozen WebArena evaluator dataset is missing or changed")
        from catalog.webarena import validate_profile
        profile_ref = cfg["agent"].get("profile_ref") or snapshot.get("profile_ref")
        if not profile_ref:
            raise ValueError("WebArena Verified requires an immutable browser agent profile")
        profile_path = project_file(profile_ref)
        profile = validate_profile(read_json(profile_path))
        if snapshot.get("profile_sha256") != sha256(profile_path):
            raise ValueError("Browser profile differs from the imported snapshot")
        from benchmark_evaluation.webarena import runtime_reference
        context = {"schema_version": 1, "benchmark": benchmark, "agent": agent,
                   "revision": snapshot["revision"], "snapshot_sha256": sha256(snapshot_path),
                   "profile_sha256": sha256(profile_path), "profile": profile,
                   "task_keys": sorted(tasks), "decoder": copy.deepcopy(cfg.get("parallel", {})),
                   "generation": copy.deepcopy(cfg["generation"]),
                   "evaluator": copy.deepcopy(evaluator),
                   "evaluator_runtime": runtime_reference(cfg),
                   "evaluator_python": options["evaluator_python"],
                   "evaluator_timeout_seconds": options["evaluator_timeout_seconds"],
                   "webarena_runtime": {"browser_image": browser_ref, "environment_pool": pool,
                       "browser_image_sha256": sha256(browser_path), "environment_pool_sha256": sha256(pool_path),
                       "max_browser_actions": options["max_browser_actions"]}}
        return {**context, "context_sha256": hash_json(context)}
    ref = cfg["agent"].get("profile_ref")
    snapshot_ref = cfg["benchmark"].get("snapshot_ref")
    if not ref or not snapshot_ref:
        raise ValueError("SkillsBench requires benchmark.snapshot_ref and agent.profile_ref")
    root = Path(cfg["project_root"])
    profile_path, snapshot_path = root / ref, root / snapshot_ref
    if (not profile_path.resolve().is_relative_to(root.resolve())
            or not snapshot_path.resolve().is_relative_to(root.resolve())):
        raise ValueError("Benchmark snapshot/profile references must remain inside the project root")
    from runtime.agents.openhands import validate_profile
    profile = validate_profile(read_json(profile_path))
    if profile.get("dependency_snapshot"):
        from runtime.agents.dependencies import verify_snapshot
        dependency = Path(profile["dependency_snapshot"]) / "manifest.json"
        dependency_manifest=verify_snapshot(dependency.parent)
        if profile.get('condenser_max_tokens') and 'stock_condenser_for_minimal_BenchFlow_settings_v1' not in {
                row['policy'] for row in dependency_manifest.get('source_adaptations',[])}:
            raise ValueError('Condenser threshold requires the qualified native-condenser dependency snapshot')
        if profile.get("dependency_snapshot_sha256") not in {None, sha256(dependency)}:
            raise ValueError("OpenHands dependency manifest differs from its profile")
        profile["dependency_snapshot_sha256"] = sha256(dependency)
    snapshot = read_json(snapshot_path)
    if snapshot.get("schema_version") != 1 or snapshot.get("benchmark") != benchmark:
        raise ValueError("Invalid benchmark snapshot")
    for key, task in tasks.items():
        # Partition is mutable Catalog selection metadata, not task content.
        frozen_task = snapshot["tasks"].get(key, {})
        frozen_definition = {k: v for k, v in frozen_task.items() if k != "partition"}
        current_definition = {k: v for k, v in task.items() if k != "partition"}
        if (task["benchmark"] != benchmark or task["revision"] != snapshot["revision"]
                or frozen_definition != current_definition):
            raise ValueError("Task does not belong to frozen benchmark snapshot")
    compatibility = profile.get('verifier_index_compatibility')
    if compatibility != snapshot.get('verifier_index_compatibility'):
        resources=dependency_manifest.get('verifier_resources',{}) if profile.get('dependency_snapshot') else {}
        local=(profile.get('verifier_resource_policy')=='skillsbench-verifier-resources-v1'
            and resources.get('policy')=='skillsbench-verifier-resources-v1'
            and snapshot.get('verifier_correction',{}).get('policy')=='skillsbench-verifier-repair-v1'
            and bool(tasks) and all(resources.get('tasks',{}).get(t['instance_id'],{}).get('package_sha256')
                ==t['instance']['package_sha256'] for t in tasks.values()))
        if not local:
            raise ValueError('Frozen SkillsBench profile and task index compatibility policies differ')
    context = {"schema_version": 1, "benchmark": benchmark, "agent": agent,
               "revision": snapshot["revision"], "snapshot_sha256": sha256(snapshot_path),
               "profile_sha256": sha256(profile_path), "profile": profile,
               "task_keys": sorted(tasks), "decoder": copy.deepcopy(cfg.get("parallel", {})),
               "generation": copy.deepcopy(cfg["generation"])}
    from runtime.agents.openhands import condenser_context
    context.update(condenser_context(cfg, profile))
    if snapshot.get("validation_only"):
        context["validation_only"] = True
    return {**context, "context_sha256": hash_json(context)}

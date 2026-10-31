"""Versioned benchmark/agent capabilities and immutable execution identities.

This module deliberately imports no agent SDK. Training provenance never chooses
an execution backend; only the frozen run configuration does.
"""
from __future__ import annotations

import copy
from pathlib import Path

from core.io import hash_json, read_json, sha256

CAPABILITIES = {
    ("swebench", "minisweagent"): {"subject": "patch", "action_analysis": True},
    ("skillsbench", "openhands"): {"subject": "workspace", "action_analysis": False},
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
        raise ValueError("action analysis does not support SkillsBench/OpenHands")
    if pair[1] == "openhands" and cfg.get("steering", {}).get("enabled"):
        raise ValueError("OpenHands steering has not been qualified")
    return result


def execution_context(cfg, tasks):
    capabilities(cfg)
    benchmark, agent = identity(cfg)
    if benchmark == "swebench":
        if any(t["benchmark"] == "skillsbench" for t in tasks.values()):
            raise ValueError("SkillsBench tasks require skillsbench + openhands")
        return None  # Preserve legacy run and task hashes byte for byte.
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
        if (task["benchmark"] != benchmark or task["revision"] != snapshot["revision"]
                or snapshot["tasks"].get(key) != task):
            raise ValueError("Task does not belong to frozen benchmark snapshot")
    context = {"schema_version": 1, "benchmark": benchmark, "agent": agent,
               "revision": snapshot["revision"], "snapshot_sha256": sha256(snapshot_path),
               "profile_sha256": sha256(profile_path), "profile": profile,
               "task_keys": sorted(tasks), "decoder": copy.deepcopy(cfg.get("parallel", {})),
               "generation": copy.deepcopy(cfg["generation"])}
    if snapshot.get("validation_only"):
        context["validation_only"] = True
    return {**context, "context_sha256": hash_json(context)}

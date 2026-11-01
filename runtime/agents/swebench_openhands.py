"""Pinned OpenHands profile and lifecycle helpers for native SWE-bench tasks."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from core.io import hash_json, read_json, sha256
from runtime.agents.openhands import OpenHandsDriverProfile


class SWEOpenHandsProfile(OpenHandsDriverProfile):
    benchmark: Literal["swebench"] = "swebench"
    workspace: str = "/testbed"
    task_cwd: str = "/testbed"
    network_mode: Literal["none", "bridge"] = "none"
    output_paths: dict[str, list[str]] = Field(default_factory=dict)
    patch_max_bytes: int = Field(default=16 * 1024 * 1024, gt=0)

    @model_validator(mode="after")
    def validate_workspace(self):
        if not self.workspace.startswith("/") or not self.task_cwd.startswith(self.workspace):
            raise ValueError("SWE OpenHands workspace must be an absolute path under /testbed")
        for paths in self.output_paths.values():
            if any(not p.startswith(self.workspace + "/") for p in paths):
                raise ValueError("SWE output paths must remain inside /testbed")
        return self


def validate_profile(raw):
    return SWEOpenHandsProfile.model_validate(raw).model_dump()


def execution_context(cfg, tasks):
    ref = cfg["agent"].get("profile_ref")
    if not ref:
        raise ValueError("SWE-bench OpenHands requires an immutable agent.profile_ref")
    root = Path(cfg["project_root"]).resolve()
    profile_path = (root / ref).resolve()
    if not profile_path.is_relative_to(root) or not profile_path.is_file():
        raise ValueError("SWE-bench OpenHands profile must be project-owned")
    profile = validate_profile(read_json(profile_path))
    dependency_sha = None
    if profile.get("dependency_snapshot"):
        from runtime.agents.dependencies import verify_snapshot
        dependency = Path(profile["dependency_snapshot"]).resolve()
        if not dependency.is_relative_to(root):
            raise ValueError("SWE-bench OpenHands dependency snapshot must be project-owned")
        verify_snapshot(dependency)
        dependency_sha = sha256(dependency / "manifest.json")
        if profile.get("dependency_snapshot_sha256") not in {None, dependency_sha}:
            raise ValueError("SWE-bench OpenHands dependency snapshot hash differs from profile")
        profile["dependency_snapshot_sha256"] = dependency_sha
    revisions = sorted({task["revision"] for task in tasks.values()})
    revision = revisions[0] if len(revisions) == 1 else "mixed"
    context = {"schema_version": 1, "benchmark": "swebench", "agent": "openhands",
               "revision": revision, "revisions": revisions,
               "profile_sha256": sha256(profile_path),
               "profile": profile, "dependency_snapshot_sha256": dependency_sha,
               "task_keys": sorted(tasks), "generation": copy.deepcopy(cfg["generation"]),
               "driver": "openhands-acp-1.22.1", "subject": "patch",
               "evaluator": "swebench-4.1.0"}
    return {**context, "context_sha256": hash_json(context)}


def patch_command(profile):
    import shlex
    cwd = shlex.quote(profile["task_cwd"])
    workspace = shlex.quote(profile["workspace"])
    return (f"cd {cwd} && test \"$(git rev-parse --show-toplevel)\" = {workspace} "
            "&& git add -A -- . ':(exclude).git' "
            "&& git diff --cached --binary --no-ext-diff -- . ':(exclude).git'; "
            "status=$?; git reset -q -- .; exit $status")


def agent_environment(cfg, bridge_url, secret, deadline):
    from runtime.agents.openhands import driver_environment
    return driver_environment(cfg, bridge_url, secret, deadline)

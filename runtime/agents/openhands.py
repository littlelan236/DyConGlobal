"""Pinned BenchFlow/OpenHands profile and lifecycle delegation."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from core.config import Strict

BENCHFLOW_VERSION = "0.6.3"
BENCHFLOW_WHEEL_SHA256 = "e032bcb4894dd3f59f98204366a7282e5ae6058289d1bd8c2fce56e2c972efc6"
CLI_REVISION = "3ca17446c5d9c1e35e054803478a3501ec251ecf"


class OpenHandsProfile(Strict):
    schema_version: Literal[1] = 1
    backend: Literal["openhands"] = "openhands"
    benchflow_version: Literal["0.6.3"] = BENCHFLOW_VERSION
    cli_revision: Literal["3ca17446c5d9c1e35e054803478a3501ec251ecf"] = CLI_REVISION
    sdk_version: Literal["1.22.1"] = "1.22.1"
    tools_version: Literal["1.22.1"] = "1.22.1"
    protocol: Literal["acp"] = "acp"
    # BenchFlow hardening kills all processes of this user before verification.
    # UID 0 would include PID 1 and terminate the sandbox itself.
    sandbox_user: Literal["agent"] = "agent"
    skill_mode: Literal["with-skill", "no-skill"] = "with-skill"
    network: Literal["public", "none"] = "public"
    use_host_proxy: bool = False
    cpu_quota: Literal["strict", "unenforced_with_receipt"] = "strict"
    skip_agent_install: bool = False
    bridge_listen: str
    bridge_advertise: str
    agent_idle_seconds: int = Field(default=600, gt=0)
    condenser_max_tokens: int | None = Field(default=None, ge=1024)
    cleanup_seconds: int = Field(default=60, gt=0)
    # Unknown auxiliary calls cannot silently become training decisions.
    request_roles: Literal["unqualified", "openhands-1.22.1"] = "unqualified"
    dependency_snapshot: str | None = None
    dependency_snapshot_sha256: str | None = None
    qualification: dict | None = None
    output_paths: dict[str, list[str]] = Field(default_factory=dict)
    # Explicit task-level memory adaptations for images whose official verifier
    # exceeds the default sandbox limit; recorded in the execution profile.
    memory_overrides: dict[str, int] = Field(default_factory=dict)
    # Verifier budgets may be extended for deterministic, CPU-heavy official
    # tasks; the override is frozen in the profile and recorded by BenchFlow.
    verifier_timeout_overrides: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_bridge_and_outputs(self):
        import ipaddress
        if self.use_host_proxy and self.network != 'public':
            raise ValueError('Host proxy requires a public-network profile')
        if self.condenser_max_tokens and not self.dependency_snapshot:
            raise ValueError('Condenser threshold requires a qualified dependency snapshot')
        if ipaddress.ip_address(self.bridge_listen) != ipaddress.ip_address(self.bridge_advertise):
            raise ValueError("Bridge must bind the exact host address reachable from the task network")
        if any(not paths for paths in self.output_paths.values()):
            raise ValueError("Declared task outputs must be nonempty")
        from application.execution.subject import safe_output_path
        if self.request_roles == "openhands-1.22.1":
            from core.io import hash_json
            proof = self.qualification or {}
            if (not proof.get("executed_tool_call_ids") or not proof.get("requests")
                    or hash_json({k: v for k, v in proof.items() if k != "receipt_sha256"})
                    != proof.get("receipt_sha256")):
                raise ValueError("Qualified request roles require a verified actual tool/result probe receipt")
        # Offline profiles may still declare the daemon's explicit CPU-quota
        # deviation; the adaptation receipt is precisely what makes this
        # resource limitation auditable without falling back to networking.
        for paths in self.output_paths.values():
            for path in paths:
                safe_output_path(path)
        return self


def validate_profile(raw):
    profile = OpenHandsProfile.model_validate(raw).model_dump()
    return profile


def check_version():
    import importlib.metadata
    if importlib.metadata.version("benchflow") != BENCHFLOW_VERSION:
        raise ValueError("SkillsBench requires isolated benchflow==0.6.3")


def resource_override(cpu_quota, memory_mb, *, cfs_supported, policy):
    if cpu_quota and not cfs_supported:
        if policy != "unenforced_with_receipt":
            raise ValueError("Rootless Docker cannot enforce task CPU quota; an explicit adaptation profile is required")
        return {"compose_override": {"environment": {"CPUS": "0"}},
                "receipt": {"requested_cpu_quota": cpu_quota, "applied_cpu_quota": None,
                            "requested_memory_mb": memory_mb, "memory_limit_preserved": True,
                            "reason": "Docker daemon reports no cpu cfs quota support"}}
    return {"compose_override": {},
            "receipt": {"requested_cpu_quota": cpu_quota, "applied_cpu_quota": cpu_quota,
                        "requested_memory_mb": memory_mb, "memory_limit_preserved": True,
                        "reason": None}}


class BenchFlowAttempt:
    """Own exactly one upstream Rollout; never start a second ACP loop."""

    def __init__(self, config):
        check_version()
        from benchflow.rollout import Rollout
        self.rollout = Rollout(config)

    async def cleanup(self):
        await self.rollout.cleanup()

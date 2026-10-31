"""Explicit, versioned bindings; SDKs are loaded only by the chosen adapter."""
from __future__ import annotations

import importlib

from runtime.backends.contracts import ExecutionBinding

REVISION = "1"
_BINDINGS = {
    ("swebench", "minisweagent"): (
        "native", "native_swe", "mini-swe-agent-2.4.5", "patch", "replay", "swebench-4.1.0", True,
        "runtime.backends.native.swe", "NativeSWEAdapter", "rollout.runner._finish_attempt"),
    ("skillsbench", "openhands"): (
        "benchflow", "benchflow", "openhands-acp-1.22.1", "workspace", "live", "skillsbench-task-verifier", False,
        "runtime.backends.benchflow.adapter", "BenchFlowAdapter", "runtime.agents.skillsbench.execute_attempt"),
    ("webarena_verified", "minisweagent"): (
        "native", "native_webarena", "mini-swe-agent-browser", "web_trace", "artifact_offline", "webarena-verified", True,
        "runtime.backends.native.webarena", "NativeWebArenaAdapter", "rollout.runner._finish_attempt"),
    ("swebench_pro_v2", "openhands"): (
        "harbor", "harbor", "openhands-acp-harbor-1.22.1", "patch", "replay", "official-pro-v2-patch-replay", True,
        "runtime.backends.harbor.adapter", "HarborAdapter", "runtime.backends.harbor.adapter.HarborAdapter"),
    ("terminalbench_2_1", "openhands"): (
        "harbor", "harbor", "openhands-acp-harbor-1.22.1", "live_state", "live", "terminalbench-2.1-task-verifier", False,
        "runtime.backends.harbor.adapter", "HarborAdapter", "runtime.backends.harbor.adapter.HarborAdapter"),
}


def resolve(cfg, tasks=None):
    from core.backends import benchmark_family, capabilities, identity

    capabilities(cfg)
    benchmark, agent = identity(cfg)
    spec = _BINDINGS.get((benchmark, agent))
    if spec is None:
        raise ValueError(f"No qualified execution adapter for {benchmark}/{agent}")
    if tasks is not None and any(benchmark_family(t["benchmark"]) != benchmark for t in tasks.values()):
        raise ValueError(f"{benchmark} backend cannot execute tasks from another benchmark family")
    family, adapter, driver, subject, mode, evaluator, replay, _, _, owner = spec
    return ExecutionBinding(1, benchmark, agent, family, adapter, REVISION, driver,
                            subject, mode, evaluator, owner, replay)


def from_snapshot(cfg, tasks=None):
    """Derive old bindings in memory; never rewrite their config or hashes."""
    binding = resolve(cfg, tasks)
    frozen = cfg.get("execution_binding")
    if frozen is not None and frozen != binding.as_dict():
        raise ValueError("Frozen execution binding is incompatible with this adapter revision")
    return binding


def adapter_for(cfg, tasks=None):
    binding = from_snapshot(cfg, tasks)
    module, name = _BINDINGS[(binding.benchmark, binding.agent)][7:9]
    return getattr(importlib.import_module(module), name)(binding)

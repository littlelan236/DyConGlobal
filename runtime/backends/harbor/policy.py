"""Versioned, run-frozen Terminal execution semantics (legacy is opt-in free)."""
from copy import deepcopy

TERMINAL_POLICY = {
    "name": "terminalbench-live-v1", "user": "root", "uid": 0, "home": "/root",
    "workspace": "official_unchanged", "path": "official_image",
    "cleanup": "owned_acp_relay_only", "agent_budget": "official_full",
    "setup_seconds": 600, "cleanup_seconds": 120,
}


def configured_policy(cfg):
    name = (cfg.get("harbor") or {}).get("terminal_policy")
    if name is None:
        return None
    if name != TERMINAL_POLICY["name"] or cfg["benchmark"]["backend"] != "terminalbench_2_1":
        raise ValueError("terminal_policy requires terminalbench_2_1 and terminalbench-live-v1")
    return deepcopy(TERMINAL_POLICY)


def phase_budgets(cfg, metadata):
    policy = execution_policy(cfg)
    if not policy:
        return None
    return {"environment": metadata.get("environment", {}).get("build_timeout_sec", 600),
            "setup": policy["setup_seconds"],
            "agent": metadata.get("agent", {}).get("timeout_sec", 600),
            "verifier": metadata.get("verifier", {}).get("timeout_sec", 600),
            "cleanup": policy["cleanup_seconds"]}


def execution_policy(cfg):
    policy = cfg.get("execution_context", {}).get("terminal_policy")
    if policy is None:
        return None
    if policy != TERMINAL_POLICY or cfg["benchmark"]["backend"] != "terminalbench_2_1":
        raise ValueError("Invalid frozen Terminal execution policy")
    return deepcopy(policy)

"""Isolated Harbor worker entry. Never imported for static configuration checks."""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
from pathlib import Path


async def execute(request):
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial

    from catalog.harbor import verify_package
    from core.io import write_json

    if request.get("reference_revision"):
        from runtime.backends.harbor.reference_solutions import reference_task
        cfg = request['driver']['cfg']
        official = cfg['selection']['tasks'][request['binding']['task_key']]
        derived, receipt = reference_task(cfg['project_root'], request['reference_revision']['path'], official)
        if derived != request['task'] or receipt != request['reference_revision']:
            raise ValueError('Derived reference request differs from publication')
    verify_package(request["task"])
    output = Path(request["output"])
    operation = request["operation"]
    agent = {"import_path": "runtime.backends.harbor.openhands_acp:DyConOpenHandsACP",
             "skip_model_parsing": True, "kwargs": {"dycon": request["driver"]}}
    from runtime.backends.harbor.policy import execution_policy
    policy = execution_policy(request["driver"]["cfg"])
    if policy:
        agent["override_setup_timeout_sec"] = policy["setup_seconds"]
    if operation in {"oracle", "nop"}:
        agent = {"name": operation}
    if operation == "replay":
        agent = {"import_path": "runtime.backends.harbor.replay:ExactReplayAgent",
                 "skip_model_parsing": True, "kwargs": {"evaluator": request["evaluator"],
                                                          "exact_subject": request["exact_subject"]}}
    config = TrialConfig.model_validate({"task": {"path": request["task"]["instance"]["task_path"]},
        "trial_name": request["trial_name"], "trials_dir": str(output / "trials"), "agent": agent,
        "environment": {"import_path": "runtime.backends.harbor.environment:DyConDockerEnvironment",
            "force_build": False, "delete": True,
            "cpu_enforcement_policy": request["environment"]["cpu_enforcement_policy"],
            "memory_enforcement_policy": "limit", "kwargs": {"dycon": request["environment"]}},
        "verifier": {"disable": request.get("disable_verifier", False)}})
    trial = await Trial.create(config)
    try:
        result = await trial.run()
        result_row = result.model_dump(mode="json")
        result_row["reward_valid"] = result.reward_valid
        result_row["failure_cause"] = result.failure_cause.value if result.failure_cause else None
        return {"worker_protocol": 1, "trial_id": str(trial.id), "result": result_row,
                "binding": request["binding"]}
    finally:
        # Trial is responsible for its verifier ordering and teardown. The
        # parent's label cleanup verifies/crash-recovers any residual resources.
        write_json(output / "trial-identity.json", {"trial_id": str(trial.id),
                   "trial_name": request["trial_name"], "binding": request["binding"]})


def schema_probe(paths):
    import importlib.metadata

    from harbor.models.task.config import TaskConfig
    from harbor.models.trial.config import TrialConfig

    rows = []
    for name in paths:
        path = Path(name)
        task = TaskConfig.model_validate_toml((path / "task.toml").read_text())
        trial = TrialConfig.model_validate({"task": {"path": str(path)}, "trial_name": "schema-probe",
                                           "agent": {"name": "nop"}})
        rows.append({"path": str(path), "task": task.model_dump(mode="json"),
                     "trial": trial.model_dump(mode="json")})
    return {"worker_protocol": 1, "harbor_version": importlib.metadata.version("harbor"), "tasks": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if request.get("worker_protocol") != 1:
        raise ValueError("Unsupported Harbor worker protocol")
    if request["operation"] == "schema":
        result = schema_probe(request["task_paths"])
    elif request["operation"] in {"execute", "probe", "oracle", "nop", "replay"}:
        async def managed():
            task = asyncio.current_task()
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
            return await execute(request)
        result = asyncio.run(managed())
    else:
        raise ValueError("Unsupported Harbor worker operation")
    Path(request["response_path"]).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

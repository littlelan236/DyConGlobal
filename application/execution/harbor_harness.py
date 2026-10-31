"""Official fresh-sandbox Pro replay through the existing harness entrypoint."""
from __future__ import annotations

from application.execution.evidence import RunEvidence
from application.execution.run_plan import load_plan
from application.execution.subject import verify_subject
from benchmark_evaluation.harbor import trial_outcome
from core.io import hash_json, identifier, owned_path, read_json, write_json
from runtime.backends.harbor.adapter import worker_request
from runtime.backends.harbor.factory import HarborFactory


def prepare(root, run_id, rollout_ids=(), source_run=None, incremental=False):
    loaded = load_plan(root, run_id)
    if loaded["config"]["benchmark"]["backend"] != "swebench_pro_v2":
        raise ValueError("Terminal-Bench uses its live task verifier; saved proof is not a replayable environment")
    if source_run not in {None, run_id}:
        raise ValueError("Pro replay currently requires the owning run; cross-run imports must be explicitly bound first")
    evidence = RunEvidence(root, run_id)
    evidence.verify()
    selection = evidence.select(rollout_ids)
    if not selection["rollouts"]:
        raise ValueError("No exact Pro patch subjects selected")
    path = loaded["run"] / "harness/selection.json"
    if path.exists():
        frozen = read_json(path)
        if (hash_json(frozen) != read_json(path.with_name("selection-receipt.json"))["selection_sha256"] or
                (rollout_ids and sorted(rollout_ids) != frozen["rollout_ids"])):
            raise ValueError("Pro harness selection changed")
        for row in frozen["rollouts"].values():
            verify_subject(root, row)
        if incremental:
            raise ValueError("Pro incremental selection requires a new evaluation binding")
        return {"run_id": run_id, "rollout_count": len(frozen["rollout_ids"]), "selection": str(path)}
    for row in selection["rollouts"].values():
        if row["benchmark"] != "swebench_pro_v2" or not row.get("sample_key") or not row.get("attempt_id"):
            raise ValueError("Pro replay requires exact task/sample/physical-attempt identity")
        verify_subject(root, row)
    write_json(path, selection)
    write_json(path.with_name("selection-receipt.json"), {"selection_sha256": hash_json(selection)})
    return {"run_id": run_id, "rollout_count": len(selection["rollout_ids"]), "selection": str(path)}


def run(root, run_id, eval_id, gold=False, incremental=False):
    if gold or incremental:
        raise ValueError("Use harbor controls for official oracle/nop; Pro harness replays only frozen exact subjects")
    loaded = load_plan(root, run_id)
    cfg, directory = loaded["config"], loaded["run"] / "harness" / identifier(eval_id)
    if cfg["benchmark"]["backend"] != "swebench_pro_v2":
        raise ValueError("Terminal-Bench has no post-hoc environment replay")
    if directory.exists():
        raise FileExistsError("Evaluation identity is immutable; use a new eval ID")
    selection = read_json(loaded["run"] / "harness/selection.json")
    if hash_json(selection) != read_json(loaded["run"] / "harness/selection-receipt.json")["selection_sha256"]:
        raise ValueError("Pro harness selection changed")
    directory.mkdir(parents=True)
    factory = HarborFactory(cfg, run_id)
    outcomes = {}
    try:
        factory.lock_images([t["instance"] for t in selection["tasks"].values()])
        for rid, row in selection["rollouts"].items():
            verify_subject(root, row)
            task = selection["tasks"][row["task_key"]]
            output = directory / rid
            output.mkdir()
            binding = {"run_id": run_id, "task_key": row["task_key"], "sample_key": row["sample_key"],
                       "source_attempt_id": row["attempt_id"], "attempt_id": "replay-" + rid,
                       "source_patch_sha256": row["patch_sha256"]}
            request = worker_request(cfg, run_id, factory, task, binding, output, "replay")
            evaluator = cfg["execution_context"]["evaluator"]
            request["evaluator"] = {**evaluator, "path": str(owned_path(root, evaluator["path"]))}
            request["exact_subject"] = {"path": str(owned_path(root, row["patch_path"])),
                "sha256": row["patch_sha256"], "instance_id": task["instance_id"],
                "sample_key": row["sample_key"], "attempt_id": row["attempt_id"]}
            response = factory.execute_worker(request, seconds=cfg["timeouts"]["container_lifetime_seconds"])
            outcome = trial_outcome(response["result"], task=task, attempt_id=row["attempt_id"])
            outcome.update(patch_sha256=row["patch_sha256"], source_sample_key=row["sample_key"],
                           trial_id=response["trial_id"], authoritative=True, new_model_requests=0)
            outcomes[rid] = outcome
        report = directory / "summary.json"
        write_json(report, outcomes)
        RunEvidence(root, run_id).record_evaluation(eval_id, report, outcomes,
            {"benchmark": "swebench_pro_v2", "authoritative": True,
             "evaluator": cfg["execution_context"]["evaluator"], "fresh_sandbox": True})
        return {"run_id": run_id, "evaluation_id": eval_id, "outcomes": outcomes}
    finally:
        factory.cleanup_owned(run_id)

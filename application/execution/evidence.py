"""Run-scoped rollout and evaluation evidence index."""
from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path

from core.io import hash_json, identifier, owned_path, read_json, sha256, write_json
from core.locking import file_lock


class RunEvidence:
    filename = "evidence.json"

    def __init__(self, root, run_id):
        self.root = Path(root).resolve()
        self.run_id = identifier(run_id)
        self.path = self.root / "runs" / self.run_id / self.filename

    def empty(self):
        return {"schema_version": 1, "rollouts": {}, "evaluations": {}, "tasks": {}, "run_id": self.run_id, "revision": 0}

    def read(self):
        if not self.path.exists():
            return self.empty()
        data = read_json(self.path)
        if data.get("schema_version") != 1:
            raise ValueError(f"Unsupported run catalog schema: {self.path}")
        self.validate(data)
        return data

    @contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(self.path.with_suffix(".lock")):
            original = self.read()
            data = copy.deepcopy(original)
            yield data
            self.validate(data)
            if data != original:
                data["revision"] = original["revision"] + 1
                write_json(self.path, data)

    def register_rollout(self, rollout_id, row, task):
        identifier(rollout_id)
        with self.transaction() as data:
            if rollout_id in data["rollouts"]:
                existing = data["rollouts"][rollout_id]
                if existing != row:
                    raise ValueError(f"Conflicting rollout evidence: {rollout_id}")
                return
            key = row["task_key"]
            if key in data["tasks"] and data["tasks"][key] != task:
                raise ValueError("Conflicting run task definition")
            data["tasks"][key] = task
            data["rollouts"][rollout_id] = row

    def record_evaluation(self, eval_id, report_path, outcomes, provenance):
        identifier(eval_id)
        report = owned_path(self.root, report_path)
        if not report.is_relative_to(self.path.parent):
            raise ValueError("Evaluation report must live in its owning run")
        with self.transaction() as data:
            if eval_id in data["evaluations"]:
                raise FileExistsError(eval_id)
            for rid, result in outcomes.items():
                row = data["rollouts"][rid]
                if result["patch_sha256"] != row["patch_sha256"]:
                    raise ValueError("Patch mismatch")
                row.update(evaluation_status=result["status"], resolved=result["resolved"],
                           selected_evaluation_id=eval_id)
            data["evaluations"][eval_id] = {
                "report_path": str(report.relative_to(self.root)),
                "report_sha256": sha256(report), "outcomes": outcomes, "provenance": provenance,
            }

    def validate(self, data):
        from catalog.store import Task
        if data.get("run_id") != self.run_id:
            raise ValueError("Run evidence belongs to another run")
        if type(data.get("revision")) is not int or data["revision"] < 0:
            raise ValueError("Invalid run evidence revision")
        for key, task in data["tasks"].items():
            Task.model_validate(task)
            if key != hash_json({k: task[k] for k in ("benchmark", "revision", "instance_id")}):
                raise ValueError("Run task key does not match benchmark identity")
        for rid, row in data["rollouts"].items():
            identifier(rid)
            if row["task_key"] not in data["tasks"] or row["run_id"] != self.run_id:
                raise ValueError("Run rollout/task binding changed")
            for field in ("trajectory", "patch"):
                owned_path(self.root, row[field + "_path"])
            status, resolved = row["evaluation_status"], row["resolved"]
            if status not in ("not_run", "running", "completed", "infra_error", "ConnectionError"):
                raise ValueError("Invalid evaluation status")
            if (status == "completed") != (type(resolved) is bool):
                raise ValueError("Only completed evaluation has a boolean resolved outcome")
            if resolved and row["submission_status"] != "submitted":
                raise ValueError("Unsubmitted trajectory cannot be resolved")
            eid = row["selected_evaluation_id"]
            if status == "completed" and not eid:
                raise ValueError("Completed outcome needs evaluation evidence")
            if eid:
                outcome = data["evaluations"][eid]["outcomes"][rid]
                if outcome["patch_sha256"] != row["patch_sha256"]:
                    raise ValueError("Evaluation must bind the exact rollout patch")
                if outcome["status"] != status or outcome["resolved"] != resolved:
                    raise ValueError("Rollout outcome disagrees with selected evaluation")
        for evaluation in data["evaluations"].values():
            owned_path(self.root, evaluation["report_path"])
            for rid, outcome in evaluation["outcomes"].items():
                if rid not in data["rollouts"] or outcome["patch_sha256"] != data["rollouts"][rid]["patch_sha256"]:
                    raise ValueError("Evaluation outcome binds an unknown rollout or different patch")

    def select(self, rollout_ids=(), *, partition=None):
        data = self.read()
        requested = list(rollout_ids)
        if len(requested) != len(set(requested)):
            raise ValueError("Duplicate rollout IDs")
        missing = set(requested) - set(data["rollouts"])
        if missing:
            raise ValueError(f"Unknown rollout IDs in run {self.run_id}: {sorted(missing)}")
        if requested:
            ids = requested
        else:
            from rollout.retry import default_final_ids
            final_ids = default_final_ids(self.root, self.run_id, data)
            ids = sorted(data["rollouts"]) if final_ids is None else final_ids
        rows = {}
        for rid in ids:
            row = data["rollouts"][rid]
            task = data["tasks"][row["task_key"]]
            eligible = not row.get("provenance", {}).get("quarantined")
            eligible &= (task["partition"] != "excluded" if partition is None else task["partition"] == partition)
            if not eligible:
                if requested:
                    raise ValueError(f"Excluded/quarantined rollout or partition mismatch: {rid}")
                continue
            rows[rid] = copy.deepcopy(row)
            if requested and row.get("sample_key"):
                result_path = self.path.parent / "rollout/tasks" / task["instance_id"] / "repetitions" / row["repetition_id"] / "result.json"
                final = read_json(result_path) if result_path.exists() else {}
                rows[rid]["selection_role"] = "accepted_final" if final.get("rollout_id") == rid and final.get("disposition", "accepted") == "accepted" else "explicit_intermediate"
        return {"source_run_id": self.run_id, "evidence_revision": data["revision"],
                "evidence_sha256": hash_json(data), "rollout_ids": sorted(rows), "rollouts": rows,
                "tasks": {r["task_key"]: copy.deepcopy(data["tasks"][r["task_key"]]) for r in rows.values()}}

    def verify(self):
        data = self.read()
        for rid, row in data["rollouts"].items():
            for field in ("trajectory", "patch"):
                if sha256(owned_path(self.root, row[field + "_path"])) != row[field + "_sha256"]:
                    raise ValueError(f"Changed {field}: {rid}")
        for evaluation in data["evaluations"].values():
            if sha256(owned_path(self.root, evaluation["report_path"])) != evaluation["report_sha256"]:
                raise ValueError("Changed evaluation report")
        from rollout.retry import verify_counts
        stage_path = self.path.parent / "status.json"
        counts = verify_counts(self.path.parent) if stage_path.exists() and read_json(stage_path).get("stage") in {"complete", "failed", "cancelled"} else {"checked_at_finish": False}
        return {"run_id": self.run_id, "rollouts": len(data["rollouts"]),
                "evaluations": len(data["evaluations"]), "verified": True, "rollout_counts": counts}



def verified_trajectory(root, evidence, run_evidence=None):
    """Resolve capture identity inside its explicit producer run."""
    root = Path(root).resolve()
    rid = evidence["rollout_id"]
    path = Path(evidence.get("trajectory_path", ""))
    if rid.startswith("startup-"):
        path = path if path.is_absolute() else owned_path(root, str(path))
        if not path.resolve().is_relative_to(root / "runs") or sha256(path) != evidence["trajectory_sha256"]:
            raise ValueError("Startup diagnostic trajectory changed")
        return path
    if run_evidence is None:
        run_id = evidence.get("producer_run_id") or evidence.get("run_id")
        if not run_id:
            absolute = path if path.is_absolute() else root / path
            try:
                run_id = absolute.resolve().relative_to(root / "runs").parts[0]
            except (ValueError, IndexError):
                raise ValueError("Capture requires an explicit run identity") from None
        run_evidence = RunEvidence(root, run_id).read()
    elif isinstance(run_evidence, RunEvidence):
        run_evidence = run_evidence.read()
    row = run_evidence["rollouts"][rid]
    if row["trajectory_sha256"] != evidence["trajectory_sha256"]:
        raise ValueError("Capture and canonical trajectory hashes disagree")
    path = owned_path(root, row["trajectory_path"])
    if sha256(path) != row["trajectory_sha256"]:
        raise ValueError("Canonical trajectory bytes changed")
    return path

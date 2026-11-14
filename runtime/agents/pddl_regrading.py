"""Full verifier replay for airport plans, with explicit backed-up receipt replacement.

This task-specific replay restores only text plans against the unchanged image and
frozen PDDL inputs. It cannot be used as arbitrary workspace replay.
"""

from __future__ import annotations

import asyncio
import copy
import csv
import hashlib
import json
import shutil
import traceback
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

from application.execution.run_plan import load_plan
from application.execution.status import _sample_status_projection
from application.execution.subject import read_archive_members
from catalog.skillsbench import verify_package
from catalog.skillsbench_verifier import POLICY, TASK_ID, verify_adaptation
from core.io import hash_json, identifier, now, read_json, sha256, write_json
from core.locking import file_lock
from runtime.agents.pddl_revalidation import _member


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _plan(loaded, snapshot, samples):
    root = Path(loaded["config"]["project_root"])
    run = loaded["run"].resolve()
    for path in (snapshot, samples):
        if not Path(path).resolve().is_relative_to(root.resolve()):
            raise ValueError("Snapshot and selection must be project-owned")
    selection = read_json(samples)
    if selection.get("run_id") != loaded["plan"]["run_id"] or selection.get("task_id") != TASK_ID:
        raise ValueError("Selection belongs to a different run/task")
    variants = [t for t in read_json(snapshot)["tasks"].values() if t["instance_id"] == TASK_ID]
    if len(variants) != 1:
        raise ValueError("Select exactly one corrected airport verifier")
    task = variants[0]
    verify_package(task)
    verify_adaptation(task)
    receipt = read_json(task["instance"]["verifier_adaptation"]["receipt_path"])
    original = receipt["upstream_task"]
    key = hash_json({k: original[k] for k in ("benchmark", "revision", "instance_id")})
    if loaded["config"]["selection"]["tasks"].get(key) != original:
        raise ValueError("Verifier is not derived from this run's frozen task")
    environment = Path(original["instance"]["task_path"]) / "environment"
    problems = read_json(environment / "problem.json")
    projection, _ = _sample_status_projection(loaded, read_json(run / "progress.json"))
    selected, seen = [], set()
    for item in selection["samples"]:
        current = projection.get(item["sample_key"])
        if (item["sample_key"] in seen or not current or current.get("disposition") != "accepted"
                or any(current.get(k) != item[k] for k in ("sample_key", "repetition_id", "attempt_id"))
                or current.get("task_key") != key or current.get("instance_id") != TASK_ID):
            raise ValueError("Duplicate, unaccepted or superseded sample")
        seen.add(item["sample_key"])
        rep, aid = identifier(item["repetition_id"]), identifier(item["attempt_id"])
        attempt = _member(run, f"rollout/tasks/{TASK_ID}/repetitions/{rep}/attempts/{aid}")
        subject_path = _member(attempt, "subject/subject.json")
        subject = read_json(subject_path)
        if sha256(subject_path) != current["subject_sha256"]:
            raise ValueError("Submission subject changed")
        archives = [o for o in subject["outputs"] if o.get("kind") == "directory_archive"]
        if len(archives) != 1 or archives[0]["sha256"] != item["archive_sha256"]:
            raise ValueError("Selection does not identify the exact submitted archive")
        archive = _member(attempt / "subject", archives[0]["path"])
        wanted = {"app/problem.json"}
        for p in problems:
            wanted.update("app/" + p[k] for k in ("domain", "problem", "plan_output"))
        members = read_archive_members(archive, item["archive_sha256"], lambda n: n in wanted)
        members = {m["name"]: m for m in members}
        if set(members) != wanted or json.loads(members["app/problem.json"]["data"]) != problems:
            raise ValueError("Incomplete or changed submitted problem manifest")
        for p in problems:
            for name in ("domain", "problem"):
                if members["app/" + p[name]]["sha256"] != sha256(_member(environment, p[name])):
                    raise ValueError("Submitted PDDL input differs from the frozen input")
        selected.append({"selection": item, "current": current, "attempt": attempt,
                         "subject_path": subject_path, "members": members})
    if not selected:
        raise ValueError("Explicit nonempty selection required")
    return task, original, problems, selected


async def _replay(cfg, run_id, key, task, problems, item, folder):
    from benchflow.environment.manifest import EnvironmentManifest
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.rollout_planes import DefaultRolloutPlanes
    from benchflow.usage_tracking import UsageTrackingConfig

    from runtime.agents.skillsbench import OwnedPlanes, SkillsBenchFactory

    current = item["current"]
    evaluation_attempt = "pddl-regrade-" + hash_json({"output": str(folder), "sample": current["sample_key"]})[:20]
    cfg = copy.deepcopy(cfg)
    cfg["skillsbench_validation"] = True
    record = {**item["selection"], "rollout_id": current["rollout_id"],
        "original_reward": current["raw_reward"], "original_subject_sha256": current["subject_sha256"],
        "task_key": key, "evaluation_attempt_id": evaluation_attempt,
        "evaluation_task_package_sha256": task["instance"]["package_sha256"],
        "started_at": now(), "valid_execution": False, "cleanup_confirmed": False,
        "oracle_executed": False, "model_executed": False, "full_verifier_executed": False}

    class ReplayPlanes(OwnedPlanes):
        async def _export_workspace(self):
            # Replay output is still bound to the original exact submission.
            self.subject_path = item["subject_path"]
            self.subject_sha256 = current["subject_sha256"]

    factory = rollout = planes = None
    try:
        folder.mkdir(parents=True, exist_ok=False)
        factory = SkillsBenchFactory(cfg, run_id)
        await asyncio.to_thread(factory.lock_images, [task])
        image = factory.images[(TASK_ID, "main")]
        record["image"] = image["identity"]
        cfg["skillsbench_service_images"] = {"main": image["tag"]}
        cfg["skillsbench_daemon_capabilities"] = {"cpu_quota_supported": factory.cpu_quota_supported}
        task_source = Path(task["instance"]["task_path"])
        sample = SimpleNamespace(instance_id=TASK_ID, sample_key=current["sample_key"], task_key=key)
        planes = ReplayPlanes(DefaultRolloutPlanes(), cfg, run_id, sample, evaluation_attempt, image, folder)
        planes.task_source = task_source
        mode = cfg["execution_context"]["profile"]["skill_mode"]
        config = RolloutConfig(task_path=task_source, agent="oracle", model=None,
            skill_mode=mode, skills_dir=task_source / "environment/skills" if mode == "with-skill" else None,
            sandbox_user=None, timeout=None, jobs_dir=folder / "benchflow", job_name=evaluation_attempt,
            rollout_name=evaluation_attempt, environment_manifest=EnvironmentManifest(
                name=TASK_ID, image=image["tag"], keep_alive=True, isolation="per_task", owns_lifecycle=True),
            usage_tracking=UsageTrackingConfig(mode="off"), planes=planes)
        rollout = await Rollout.create(config)
        await rollout.setup()
        await rollout.start()
        # Oracle is only a no-model lifecycle mode; never call run()/solve().
        await rollout.install_agent()
        hashes = {}
        for p in problems:
            relative = p["plan_output"]
            if Path(relative).name != relative:
                raise ValueError("Airport replay supports its declared flat text plan paths only")
            data = item["members"]["app/" + relative]["data"]
            local = folder / relative
            local.write_bytes(data)
            await planes.env.upload_file(local, "/app/" + relative)
            hashes["/app/" + relative] = _digest(data)
        # Check the runtime image's task inputs too, without executing submitted files.
        environment = task_source / "environment"
        for p in problems:
            for name in ("domain", "problem"):
                hashes["/app/" + p[name]] = sha256(environment / p[name])
        hashes["/app/problem.json"] = sha256(environment / "problem.json")
        command = "sha256sum " + " ".join(__import__("shlex").quote(p) for p in hashes)
        checked = await planes.env.exec(command, user="root", timeout_sec=30)
        actual = {}
        for line in checked.stdout.splitlines():
            digest, name = line.split(maxsplit=1)
            actual[name.lstrip("*")] = digest
        if checked.return_code or actual != hashes:
            raise ValueError("Runtime plan or reference bytes do not match frozen inputs")
        write_json(folder / "runtime-inputs.json", {"expected": hashes, "observed": actual, "matched": True})
        record["runtime_inputs_verified"] = True
        await rollout.verify()
        result = rollout.result
        record.update(rewards=result.rewards, verifier_error=result.verifier_error,
                      agent_error=result.error, full_verifier_executed=True)
        reports = list((folder / "benchflow").rglob("verifier/ctrf.json"))
        if len(reports) != 1:
            raise ValueError("Expected exactly one full verifier test report")
        ctrf = read_json(reports[0])["results"]
        summary = ctrf["summary"]
        tests = ctrf["tests"]
        if not (summary["tests"] == summary["passed"] == 2
                and summary.get("failed", 0) == 0 and summary.get("skipped", 0) == 0
                and len(tests) == 2 and all(t["status"] == "passed" for t in tests)):
            raise ValueError("Full verifier did not pass both required tests")
        stdout = reports[0].with_name("test-stdout.txt").read_text()
        if stdout.count('"status": "VALID"') != len(problems):
            raise ValueError("Full verifier did not report VALID for every required plan")
        record["ctrf"] = {"path": str(reports[0]), "sha256": sha256(reports[0]), "summary": summary}
        record["test_stdout"] = {"path": str(reports[0].with_name("test-stdout.txt")),
                                 "sha256": sha256(reports[0].with_name("test-stdout.txt"))}
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "exception.txt").write_text(traceback.format_exc())
    finally:
        if rollout is not None:
            try:
                await rollout.cleanup()
            except Exception as exc:
                record["cleanup_error"] = repr(exc)
        if factory is not None:
            try:
                if planes is not None:
                    remaining = factory.client.containers.list(all=True, filters={
                        "label": ["dycon.run=" + run_id, "dycon.epoch=" + planes.ownership["epoch"]]})
                    record["remaining_containers"] = [c.id for c in remaining]
                    record["cleanup_confirmed"] = (folder / "sandbox-cleanup.json").exists() and not remaining
            finally:
                await asyncio.to_thread(factory.cleanup_owned, run_id)
        record["finished_at"] = now()
        record["valid_execution"] = (record.get("full_verifier_executed", False)
            and not record.get("error") and not record.get("verifier_error") and not record.get("agent_error")
            and not record.get("cleanup_error") and record["cleanup_confirmed"]
            and record.get("rewards") == {"reward": 1.0})
        record["receipt_sha256"] = hash_json(record)
        write_json(folder / "result.json", record)
    return record


def _transaction(run, base, changes, *, fail_after=None):
    """Write-ahead byte backups with rollback on errors; old files keep their paths."""
    run, base = Path(run).resolve(), Path(base).resolve()
    backup = base / "backup"
    backup.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, (path, data) in enumerate(changes.items()):
        path = Path(path)
        if path.is_symlink() or not path.resolve().is_relative_to(run):
            raise ValueError("Replacement escaped its run")
        old = path.read_bytes() if path.exists() else None
        saved = backup / f"{index:04d}.bin"
        if old is not None:
            saved.write_bytes(old)
        rows.append({"path": str(path), "backup": str(saved) if old is not None else None,
                     "before_sha256": _digest(old) if old is not None else None,
                     "after_sha256": _digest(data)})
    write_json(base / "replacement-manifest.json", {"schema_version": 1, "files": rows})
    write_json(base / "replacement-state.json", {"phase": "prepared", "at": now()})
    changed = []
    try:
        for index, row in enumerate(rows):
            path = Path(row["path"])
            if (sha256(path) if path.exists() else None) != row["before_sha256"]:
                raise ValueError("Concurrent receipt change before replacement")
            data = changes[path]
            temporary = path.with_name(path.name + ".pddl-regrade-tmp")
            if temporary.exists():
                raise FileExistsError(temporary)
            temporary.write_bytes(data)
            temporary.replace(path)
            changed.append(row)
            if fail_after is not None and index + 1 == fail_after:
                raise RuntimeError("injected transactional failure")
        for row in rows:
            if sha256(row["path"]) != row["after_sha256"]:
                raise ValueError("Replacement write verification failed")
        write_json(base / "replacement-state.json", {"phase": "applied", "at": now(), "files": len(rows)})
    except BaseException:
        for row in reversed(changed):
            path = Path(row["path"])
            if row["backup"] is None:
                path.unlink()
            else:
                temporary = path.with_name(path.name + ".pddl-regrade-restore")
                temporary.write_bytes(Path(row["backup"]).read_bytes())
                temporary.replace(path)
        write_json(base / "replacement-state.json", {"phase": "rolled_back", "at": now()})
        raise
    return rows


def _replacement_changes(loaded, selected, records, base):
    run = loaded["run"].resolve()
    evidence = read_json(run / "evidence.json")
    progress = read_json(run / "progress.json")
    changes = {}
    before_totals = {"reward_sum": 0.0, "resolved": 0}
    record_map = {r["sample_key"]: r for r in records}
    for item in selected:
        current = item["current"]
        record = record_map[current["sample_key"]]
        if not record["valid_execution"]:
            raise ValueError("A full verifier failed; refusing the entire replacement")
        old = item["attempt"]
        raw_path, report_path = old / "evaluation-raw.json", old / "evaluation-report.json"
        result_path = old.parent.parent / "result.json"
        attempt_path = old / "attempt.json"
        original = read_json(attempt_path)
        if (original.get("rollout_id") != current["rollout_id"]
                or original.get("raw_reward") != current["raw_reward"]):
            raise ValueError("Historical attempt changed during replay")
        source_folder = base / "samples" / current["repetition_id"]
        new_raw = read_json(source_folder / "evaluation-raw.json")
        if new_raw.get("completed") is not True or new_raw.get("rewards") != {"reward": 1.0}:
            raise ValueError("Missing full verifier evaluation receipt")
        correction = {"schema_version": 1, "policy": POLICY, "mode": "replace-original-with-backup",
            "regrading_id": base.name, "regraded_at": now(),
            "proof_path": str(source_folder / "result.json"), "proof_sha256": sha256(source_folder / "result.json"),
            "evaluation_task_package_sha256": record["evaluation_task_package_sha256"],
            "original_raw_sha256": sha256(raw_path), "original_report_sha256": sha256(report_path),
            "backup_manifest": str(base / "replacement-manifest.json")}
        new_raw["regrading"] = correction
        new_raw["verifier_evidence"] = str(source_folder / "diagnostics/verifier-execution.json")
        changes[raw_path] = _bytes(new_raw)
        report = read_json(report_path)
        report.update(raw_rewards=record["rewards"], resolved=True, evaluation_status="completed",
                      report_sha256=_digest(changes[raw_path]), verifier_zero_audit=None,
                      evaluated_at=record["finished_at"], regrading=correction)
        changes[report_path] = _bytes(report)
        value = copy.deepcopy(original)
        value.update(raw_reward=record["rewards"], resolved=True, evaluation_status="completed",
                     evaluation_report_sha256=_digest(changes[report_path]), verifier_zero_audit=None,
                     regrading=correction)
        if "raw_evaluation_sha256" in value:
            value["raw_evaluation_sha256"] = _digest(changes[raw_path])
        for ref in (value.get("termination_evidence") or {}).get("evidence_files", []):
            if ref["path"] == "evaluation-raw.json":
                ref["sha256"] = _digest(changes[raw_path])
        changes[attempt_path] = _bytes(value)
        existing_final = read_json(result_path)
        if existing_final.get("attempt_id") != current["attempt_id"]:
            raise ValueError("Repetition final attempt changed")
        changes[result_path] = _bytes(value)
        for name in ("03-artifacts_committed.json", "05-settled.json"):
            path = old / "finalization" / name
            document = read_json(path)
            if name.startswith("03"):
                document["details"]["report_sha256"] = _digest(changes[raw_path])
            else:
                document["details"].update(raw_reward=record["rewards"], resolved=True)
            document["regrading"] = correction
            document.pop("receipt_sha256", None)
            document["receipt_sha256"] = hash_json(document)
            changes[path] = _bytes(document)
        rid, eid = original["rollout_id"], original["evaluation_id"]
        row = evidence["rollouts"][rid]
        if row["selected_evaluation_id"] != eid or row["subject_sha256"] != current["subject_sha256"]:
            raise ValueError("Evidence selection no longer matches original attempt")
        evaluation = evidence["evaluations"][eid]
        if evaluation["report_sha256"] != sha256(report_path) or set(evaluation["outcomes"]) != {rid}:
            raise ValueError("Unexpected evaluation sharing or report identity")
        before_totals["reward_sum"] += current["raw_reward"]["reward"]
        before_totals["resolved"] += bool(row["resolved"])
        row.update(resolved=True, evaluation_status="completed", regrading=correction)
        evaluation["outcomes"][rid].update(resolved=True, rewards=record["rewards"], status="completed")
        evaluation.update(report_sha256=_digest(changes[report_path]))
        evaluation["provenance"]["regrading"] = correction
        progress["samples"][current["sample_key"]] = value
    evidence["revision"] += 1
    changes[run / "evidence.json"] = _bytes(evidence)
    progress["updated_at"] = now()
    # Preserve unrelated counters; these accepted samples do not change dispositions.
    changes[run / "progress.json"] = _bytes(progress)
    return changes, before_totals


def regrade(root, run_id, regrading_id, snapshot, samples, *, workers=2, replace=False, apply_only=False):
    from application.execution.config_revision import _offline

    root, regrading_id = Path(root).resolve(), identifier(regrading_id)
    if not 1 <= workers <= 4:
        raise ValueError("Workers must be in 1..4")
    loaded = load_plan(root, run_id)
    run = loaded["run"].resolve()
    base = run / "regrades" / regrading_id
    with ExitStack() as locks:
        for path in [root / "state/run-locks" / (run_id + ".lock"),
                     run / "execution/supervisor.lock", run / "execution/active-supervisor.lock",
                     run / "validation/.oracle-lifecycle.lock"]:
            locks.enter_context(file_lock(path, nonblocking=True))
        _offline(run)
        task, original, problems, selected = _plan(loaded, snapshot, samples)
        if apply_only:
            if not replace:
                raise ValueError("--apply-only requires explicit replacement")
            proof = read_json(base / "request.json")
            if proof["snapshot_sha256"] != sha256(snapshot) or proof["samples_sha256"] != sha256(samples):
                raise ValueError("Apply request differs from the executed replay")
            records = [read_json(base / "samples" / item["current"]["repetition_id"] / "result.json")
                       for item in selected]
        else:
            if base.exists():
                raise FileExistsError("Regrading IDs are immutable; choose a new ID")
            base.mkdir(parents=True, exist_ok=False)
            write_json(base / "request.json", {"run_id": run_id, "regrading_id": regrading_id,
                "snapshot": str(Path(snapshot).resolve()), "snapshot_sha256": sha256(snapshot),
                "samples": str(Path(samples).resolve()), "samples_sha256": sha256(samples),
                "workers": workers, "replace_original_with_backup": replace, "created_at": now()})
            # Initial full JSON receipts are saved before even starting verifier replay.
            initial = base / "before-replay"
            for name in ("evidence.json", "progress.json", "status.json", "status-summary.json"):
                if (run / name).exists():
                    (initial / "run").mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(run / name, initial / "run" / name)
            for item in selected:
                target = initial / item["current"]["repetition_id"]
                target.mkdir(parents=True)
                for name in ("attempt.json", "evaluation-raw.json", "evaluation-report.json"):
                    shutil.copyfile(item["attempt"] / name, target / name)
                shutil.copytree(item["attempt"] / "finalization", target / "finalization")
                shutil.copyfile(item["attempt"].parent.parent / "result.json", target / "repetition-result.json")
            async def execute():
                from runtime.agents.skillsbench import SkillsBenchFactory

                semaphore = asyncio.Semaphore(workers)
                shared_cfg = copy.deepcopy(loaded["config"])
                network = None
                async def one(item):
                    async with semaphore:
                        record = await _replay(shared_cfg, run_id, item["current"]["task_key"],
                            task, problems, item, base / "samples" / item["current"]["repetition_id"])
                        print(json.dumps({"sample": record["repetition_id"], "valid_execution": record["valid_execution"],
                            "reward": record.get("rewards"), "error": record.get("error")}), flush=True)
                        return record
                try:
                    if shared_cfg["execution_context"]["profile"].get("use_host_proxy"):
                        network = SkillsBenchFactory(copy.deepcopy(shared_cfg), run_id)
                        await asyncio.to_thread(network.lock_images, [])
                        shared_cfg["skillsbench_network_proxy"] = network.cfg["skillsbench_network_proxy"]
                    return await asyncio.gather(*(one(item) for item in selected))
                finally:
                    if network is not None:
                        await asyncio.to_thread(network.cleanup_owned, run_id)
            records = asyncio.run(execute())
        verify_package(original)
        verify_package(task)
        summary = {"run_id": run_id, "regrading_id": regrading_id, "sample_count": len(records),
            "valid_executions": sum(r["valid_execution"] for r in records),
            "full_verifier": True, "original_replaced": False, "finished_at": now(), "output": str(base)}
        write_json(base / "summary.json", summary)
        if replace and summary["valid_executions"] == len(records):
            from application.execution.evidence import RunEvidence
            from application.execution.status import refresh_run_status_summary

            for path in [run / "evidence.lock", run / "progress.lock"]:
                locks.enter_context(file_lock(path, nonblocking=True))
            latest = load_plan(root, run_id)
            _, _, _, latest_selected = _plan(latest, snapshot, samples)
            changes, before = _replacement_changes(latest, latest_selected, records, base)
            # Validate new evidence without acquiring another transaction lock.
            RunEvidence(root, run_id).validate(json.loads(changes[run / "evidence.json"]))
            manifest = _transaction(run, base, changes)
            refreshed = refresh_run_status_summary(root, run_id)
            projection, _ = _sample_status_projection(latest, read_json(run / "progress.json"))
            if not all(projection[r["sample_key"]]["raw_reward"] == {"reward": 1.0}
                       and projection[r["sample_key"]]["resolved"] is True for r in records):
                raise RuntimeError("Replacement projection verification failed; inspect backup journal")
            summary.update(original_replaced=True, replacement_files=len(manifest),
                before_selected=before, after_selected={"reward_sum": float(len(records)), "resolved": len(records)},
                status_summary_refreshed=refreshed is not None,
                backup=str(base / "backup"), historical_analysis_reports_preserved=True)
            write_json(base / "summary.json", summary)
        with (base / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sample_key", "repetition_id", "attempt_id",
                "valid_execution", "original_reward", "rewards", "cleanup_confirmed"])
            writer.writeheader()
            for r in records:
                writer.writerow({k: r.get(k) for k in writer.fieldnames})
        return summary

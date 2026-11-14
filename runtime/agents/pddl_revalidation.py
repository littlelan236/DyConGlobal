"""Separate semantic revalidation of explicitly selected historical airport plans.

This is not full sandbox replay and never updates historical outcomes or rewards.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath

from application.execution.run_plan import load_plan
from application.execution.status import _sample_status_projection
from application.execution.subject import read_archive_members
from catalog.skillsbench import verify_package
from catalog.skillsbench_verifier import MEMBER, POLICY, TASK_ID, verify_adaptation
from core.io import hash_json, identifier, now, read_json, sha256, write_json


def _member(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError("Unsafe relative input path")
    target = root / str(relative)
    if (not target.resolve().is_relative_to(root.resolve())
            or any(p.is_symlink() for p in [target, *target.parents] if p != root and p.is_relative_to(root))):
        raise ValueError("Input path contains a link or escaped its owner")
    return target


def revalidate(root, run_id, snapshot, samples, output, validator_python, *, timeout=60):
    root = Path(root).resolve()
    loaded = load_plan(root, run_id)
    run = loaded["run"].resolve()
    output = Path(output).absolute()
    # New project analysis publication, entirely outside run, package and code trees.
    analysis = root / "analysis"
    if (not output.resolve().is_relative_to(analysis) or output.resolve() == analysis
            or any(p.is_symlink() for p in [output, *output.parents] if p != root and p.is_relative_to(root))):
        raise ValueError("Revalidation output must be a new project analysis directory")
    if output.exists():
        raise FileExistsError("Revalidation output is immutable; choose a new path")
    executable = Path(validator_python).absolute()
    if not executable.is_relative_to(root / ".venvs") or not executable.is_file():
        raise ValueError("Select an explicit project .venvs validator interpreter")
    snapshot = _member(root, str(Path(snapshot).absolute().relative_to(root)))
    samples = _member(root, str(Path(samples).absolute().relative_to(root)))
    publication = read_json(snapshot)
    selected = read_json(samples)
    if selected.get("run_id") != run_id or selected.get("task_id") != TASK_ID:
        raise ValueError("Selection does not bind this run and airport task")
    if not isinstance(selected.get("samples"), list) or not selected["samples"]:
        raise ValueError("Explicit nonempty sample selection required")
    variants = [t for t in publication["tasks"].values() if t["instance_id"] == TASK_ID]
    if len(variants) != 1:
        raise ValueError("Snapshot does not identify exactly one airport variant")
    variant = variants[0]
    verify_package(variant)
    verify_adaptation(variant)
    receipt = read_json(variant["instance"]["verifier_adaptation"]["receipt_path"])
    original = receipt["upstream_task"]
    task_key = hash_json({k: original[k] for k in ("benchmark", "revision", "instance_id")})
    if loaded["config"]["selection"]["tasks"].get(task_key) != original:
        raise ValueError("Corrected verifier is not derived from this run's frozen task")
    environment = Path(original["instance"]["task_path"]) / "environment"
    problem_manifest = environment / "problem.json"
    problems = read_json(problem_manifest)
    wanted = {"app/problem.json"}
    for problem in problems:
        for field in ("domain", "problem", "plan_output"):
            rel = PurePosixPath(problem[field])
            if rel.is_absolute() or ".." in rel.parts or "\\" in str(rel):
                raise ValueError("Unsafe frozen problem manifest")
            wanted.add("app/" + str(rel))
    progress = read_json(run / "progress.json")
    projection, _ = _sample_status_projection(loaded, progress)
    initial_projection = hash_json(projection)
    consumed = {}

    def consume(path):
        path = Path(path)
        digest = sha256(path)
        if str(path) in consumed and consumed[str(path)] != digest:
            raise ValueError("Source changed during preparation")
        consumed[str(path)] = digest
        return digest

    for name in ("plan.json", "selection.json", "config.yaml", "resolved-config.json",
                 "work-items.json", "tasks.json", "progress.json"):
        if (run / name).is_file():
            consume(run / name)
    for path in (snapshot, samples, problem_manifest, Path(__file__),
                 Path(__file__).with_name("pddl_validation_worker.py"), executable.resolve(),
                 Path(variant["instance"]["task_path"]) / MEMBER,
                 Path(variant["instance"]["verifier_adaptation"]["receipt_path"])):
        consume(path)
    prepared, seen = [], set()
    for row in selected["samples"]:
        sample_key = identifier(row["sample_key"])
        rep, attempt_id = identifier(row["repetition_id"]), identifier(row["attempt_id"])
        if sample_key in seen:
            raise ValueError("Duplicate selected sample")
        seen.add(sample_key)
        current = projection.get(sample_key)
        if not current or any(current.get(k) != v for k, v in {
                "instance_id": TASK_ID, "task_key": task_key,
                "repetition_id": rep, "attempt_id": attempt_id}.items()):
            raise ValueError("Selected historical attempt does not match current projection")
        attempt = _member(run, f"rollout/tasks/{TASK_ID}/repetitions/{rep}/attempts/{attempt_id}")
        subject_path = _member(attempt, "subject/subject.json")
        consume(subject_path)
        subject = read_json(subject_path)
        archives = [o for o in subject["outputs"] if o.get("kind") == "directory_archive"]
        if len(archives) != 1:
            raise ValueError("Expected exactly one submitted directory archive")
        archive_row = archives[0]
        archive = _member(attempt / "subject", archive_row["path"])
        if consume(archive) != row["archive_sha256"] or archive_row["sha256"] != row["archive_sha256"]:
            raise ValueError("Selection/subject/archive identity mismatch")
        members = read_archive_members(archive, row["archive_sha256"], lambda name: name in wanted)
        member_map = {m["name"]: m for m in members}
        if set(member_map) != wanted:
            raise ValueError("Missing selected plan or reference input in submission")
        matches = json.loads(member_map["app/problem.json"]["data"]) == problems
        for problem in problems:
            for field in ("domain", "problem"):
                trusted = _member(environment, problem[field])
                digest = consume(trusted)
                matches = matches and member_map["app/" + problem[field]]["sha256"] == digest
        if not matches:
            raise ValueError("Submitted task inputs differ from trusted frozen task inputs")
        prepared.append((row, current, members, member_map, matches))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "selection.json", selected)
    write_json(output / "sources-before.json", consumed)
    results = []
    worker = Path(__file__).with_name("pddl_validation_worker.py")
    for row, current, members, member_map, matches in prepared:
        folder = output / "samples" / row["repetition_id"]
        folder.mkdir(parents=True)
        request_tasks = []
        for problem in problems:
            plan = folder / (identifier(problem["id"]) + ".txt")
            plan.write_bytes(member_map["app/" + problem["plan_output"]]["data"])
            request_tasks.append({"id": problem["id"], "plan": str(plan),
                "domain": str(environment / problem["domain"]),
                "problem": str(environment / problem["problem"])})
        request = {"verifier": str(Path(variant["instance"]["task_path"]) / MEMBER), "tasks": request_tasks}
        write_json(folder / "request.json", request)
        record = {**row, "original_reward": current.get("raw_reward"),
                  "reference_inputs_match": matches,
                  "archive_members": [{k: v for k, v in m.items() if k != "data"} for m in members]}
        try:
            env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
            env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
            with (folder / "worker.log").open("w", encoding="utf-8") as log:
                result = subprocess.run([str(executable), "-I", "-B", str(worker),
                    str(folder / "request.json"), str(folder / "result.json")],
                    cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT,
                    timeout=timeout, check=False)
            if result.returncode:
                raise RuntimeError(f"Validator worker exited {result.returncode}; see worker.log")
            response = read_json(folder / "result.json")
            if len(response["plans"]) != len(problems):
                raise ValueError("Worker did not check every required problem")
            record.update(response)
            success = all(p["valid"] for p in response["plans"])
            record.update(verdict="ValidPlan" if success else "InvalidPlan",
                          revalidated_reward=1 if success else 0)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            record.update(verdict="RevalidationError", revalidated_reward=None,
                          error=f"{type(exc).__name__}: {exc}")
        results.append(record)
        write_json(folder / "record.json", record)
    changed = [path for path, digest in consumed.items() if sha256(path) != digest]
    verify_package(original)
    verify_package(variant)
    final_loaded = load_plan(root, run_id)
    final_projection, _ = _sample_status_projection(final_loaded, read_json(run / "progress.json"))
    if hash_json(final_projection) != initial_projection:
        changed.append("current_sample_projection")
    counts = dict(Counter(r["verdict"] for r in results))
    summary = {"schema_version": 1, "policy": POLICY, "run_id": run_id, "task_id": TASK_ID,
        "checked_at": now(), "snapshot": str(snapshot), "snapshot_sha256": sha256(snapshot),
        "selection_sha256": sha256(samples), "sample_count": len(results),
        "plan_count": sum(len(r.get("plans", [])) for r in results), "counts": counts,
        "complete": not changed and not counts.get("RevalidationError"),
        "source_changes": changed, "historical_outcomes_modified": False,
        "full_container_replay": False, "validation_scope": "plans against hash-verified frozen PDDL inputs",
        "output": str(output)}
    with (output / "results.jsonl").open("w", encoding="utf-8") as stream:
        for record in results:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_key", "repetition_id", "attempt_id", "verdict",
            "revalidated_reward", "task01_status", "task02_status", "reference_inputs_match"])
        writer.writeheader()
        for record in results:
            writer.writerow({**{k: record.get(k) for k in writer.fieldnames if not k.endswith("_status")},
                **{p["id"] + "_status": p["status"] for p in record.get("plans", [])}})
    write_json(output / "summary.json", summary)
    write_json(output / "sources-after.json", {p: sha256(p) for p in consumed})
    return summary

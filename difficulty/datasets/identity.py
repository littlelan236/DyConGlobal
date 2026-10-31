"""Benchmark-aware task identities shared by composition and split code."""


def benchmark(row):
    return row.get("benchmark", "swebench")


def task_key(row):
    family = benchmark(row)
    if family == "skillsbench":
        revision = row.get("revision") or row.get("benchmark_revision")
        if not revision:
            raise ValueError("SkillsBench difficulty rows require a frozen benchmark revision")
        return family, revision, row["instance_id"]
    return family, row.get("repo"), row["instance_id"]


def task_group(row):
    family, *identity = task_key(row)
    if family == "skillsbench":
        return f"skillsbench@{identity[0]}::{identity[1]}"
    return f"{identity[0]}::{identity[1]}"


def task_descriptor(row):
    family, *identity = task_key(row)
    result = {"benchmark": family, "instance_id": row["instance_id"],
              "task_group": task_group(row)}
    if family == "skillsbench":
        result["revision"] = identity[0]
        package = row.get("task_package_sha256") or row.get("package_sha256")
        if package:
            result["task_package_sha256"] = package
    else:
        result["repo"] = row.get("repo")
        if row.get("base_commit") is not None:
            result["base_commit"] = row["base_commit"]
    return result


def validate_same_task(rows):
    """Reject mixed package revisions/base commits under one logical task key."""
    family = benchmark(rows[0])
    if any(benchmark(row) != family or task_key(row) != task_key(rows[0]) for row in rows):
        raise ValueError("Rows disagree on logical task identity")
    if family == "skillsbench":
        packages = {row.get("task_package_sha256") or row.get("package_sha256")
                    for row in rows if row.get("task_package_sha256") or row.get("package_sha256")}
        if len(packages) > 1:
            raise ValueError("Same SkillsBench task revision has conflicting package hashes")
    else:
        commits = {row.get("base_commit") for row in rows if row.get("base_commit") is not None}
        if len(commits) > 1:
            raise ValueError("Same task has conflicting base commits")

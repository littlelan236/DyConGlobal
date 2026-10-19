"""Deterministic repo/source quotas; never split rows of one task."""

import hashlib
import math
from collections import Counter, defaultdict

from core.io import hash_json


def make_split(rollouts, samples, validation_tasks=48, seed=42):
    by_task = {}
    for row in rollouts:
        key = row["repo"] + "::" + row["instance_id"]
        if key in by_task:
            raise ValueError("Input must have exactly one rollout per logical task")
        by_task[key] = row
    by_repo = defaultdict(list)
    for key, row in sorted(by_task.items()):
        by_repo[row["repo"]].append(key)
    quota = {repo: 0 if len(keys) == 1 else max(1, math.floor(0.2 * len(keys)))
             for repo, keys in by_repo.items()}
    minimum = sum(len(keys) > 1 for keys in by_repo.values())
    maximum = sum(len(keys) - 1 for keys in by_repo.values())
    if not minimum <= validation_tasks <= maximum:
        raise ValueError("Validation quota cannot preserve non-singleton repos on both sides")
    while sum(quota.values()) < validation_tasks:
        repo = min((r for r in by_repo if quota[r] < len(by_repo[r]) - 1),
                   key=lambda r: (-(0.2 * len(by_repo[r]) - quota[r]), r))
        quota[repo] += 1
    while sum(quota.values()) > validation_tasks:
        repo = min((r for r in by_repo if quota[r] > 1),
                   key=lambda r: (0.2 * len(by_repo[r]) - quota[r], r))
        quota[repo] -= 1
    validation = set()
    source_quota = {}
    for repo, keys in sorted(by_repo.items()):
        counts = Counter(by_task[key]["source_kind"] for key in keys)
        ideal = {s: quota[repo] * n / len(keys) for s, n in counts.items()}
        allocation = {s: math.floor(x) for s, x in ideal.items()}
        remaining = quota[repo] - sum(allocation.values())
        for source in sorted(counts, key=lambda s: (-(ideal[s] - allocation[s]), s))[:remaining]:
            allocation[source] += 1
        source_quota[repo] = allocation
        for source, n in sorted(allocation.items()):
            candidates = [key for key in keys if by_task[key]["source_kind"] == source]
            candidates.sort(key=lambda key: (
                hashlib.sha256(f"{seed}|{repo}|{by_task[key]['instance_id']}".encode()).hexdigest(),
                by_task[key]["instance_id"],
            ))
            validation.update(candidates[:n])
    tasks = [{"task_group": key, "repo": row["repo"], "instance_id": row["instance_id"],
              "base_commit": row["base_commit"], "source_kind": row["source_kind"],
              "rollout_id": row["rollout_id"], "trajectory_sha256": row["trajectory_sha256"],
              "split": "validation" if key in validation else "train"}
             for key, row in sorted(by_task.items())]
    roles = {t["rollout_id"]: t["split"] for t in tasks}
    hashes = {role: {t["trajectory_sha256"] for t in tasks if t["split"] == role}
              for role in ("train", "validation")}
    if hashes["train"] & hashes["validation"]:
        raise ValueError("Identical trajectory SHA would cross train/validation")
    sample_ids = {role: [s["sample_id"] for s in samples if roles[s["rollout_id"]] == role]
                  for role in ("train", "validation")}
    if not all(sample_ids.values()) or len(validation) != validation_tasks:
        raise ValueError("Empty or incorrect split")
    return {"schema_version": 1, "strategy": "repo_source_quota_v1", "seed": seed,
            "tasks": tasks, "repo_validation_quota": quota, "source_validation_quota": source_quota,
            "sample_ids": sample_ids, "sample_order_sha256": hash_json([s["sample_id"] for s in samples]),
            "task_counts": dict(Counter(t["split"] for t in tasks)),
            "sample_counts": {role: len(ids) for role, ids in sample_ids.items()}}

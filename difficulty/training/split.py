"""Deterministic repo/source quotas; never split rows of one task."""

import hashlib
import math
from collections import Counter, defaultdict

from core.io import hash_json
from difficulty.datasets.identity import task_descriptor, task_group, validate_same_task


def rollout_roles(split):
    """New grouped splits list all rollouts separately from unique tasks."""
    return {r["rollout_id"]: r["split"] for r in split.get("rollouts", split["tasks"])}


def inherit_task_split(rollouts, samples, parent, validation_tasks, seed):
    parents = {t["task_group"]: t for t in parent["tasks"]}
    if len(parents) != len(parent["tasks"]) or parent["seed"] != seed:
        raise ValueError("Duplicate parent task or split seed mismatch")
    by_task = defaultdict(list)
    for row in rollouts:
        by_task[task_group(row)].append(row)
    if set(by_task) != set(parents):
        raise ValueError("Expanded task set must exactly match parent task set")
    expanded = []
    for key, rows in sorted(by_task.items()):
        original = parents[key]
        if not any(r["rollout_id"] == original["rollout_id"] and
                   r["trajectory_sha256"] == original["trajectory_sha256"] for r in rows):
            raise ValueError("Original parent rollout must remain unchanged")
        for row in sorted(rows, key=lambda r: r["rollout_id"]):
            descriptor = task_descriptor(row)
            for field in ("benchmark", "revision", "repo", "instance_id", "base_commit", "task_package_sha256"):
                if field in descriptor and original.get(field) != descriptor[field]:
                    raise ValueError("Task " + field + " differs from parent split")
            expanded.append({**descriptor, "source_kind": row["source_kind"],
                             "rollout_id": row["rollout_id"], "trajectory_sha256": row["trajectory_sha256"],
                             "split": original["split"]})
    roles = {r["rollout_id"]: r["split"] for r in expanded}
    if len(roles) != len(expanded):
        raise ValueError("Duplicate rollout ID in expanded selection")
    hashes = {role: {r["trajectory_sha256"] for r in expanded if r["split"] == role}
              for role in ("train", "validation")}
    if hashes["train"] & hashes["validation"]:
        raise ValueError("Identical trajectory SHA would cross train/validation")
    sample_ids = {role: [s["sample_id"] for s in samples if roles[s["rollout_id"]] == role]
                  for role in ("train", "validation")}
    counts = dict(Counter(t["split"] for t in parents.values()))
    if counts.get("validation") != validation_tasks or not all(sample_ids.values()):
        raise ValueError("Inherited split count mismatch or empty partition")
    if len(set(sample_ids["train"]) | set(sample_ids["validation"])) != len(samples):
        raise ValueError("Duplicate sample ID across expanded split")
    return {"schema_version": 2, "strategy": "inherit_task_split_v1", "seed": seed,
            "tasks": sorted(parents.values(), key=lambda r: r["task_group"]), "rollouts": expanded,
            "parent_split_sha256": hash_json(parent),
            "task_counts": counts, "rollout_counts": dict(Counter(r["split"] for r in expanded)),
            "sample_ids": sample_ids, "sample_counts": {k: len(v) for k, v in sample_ids.items()},
            "sample_order_sha256": hash_json([s["sample_id"] for s in samples])}


def make_split(rollouts, samples, validation_tasks=48, seed=42):
    by_task = {}
    for row in rollouts:
        key = task_group(row)
        if key in by_task:
            raise ValueError("Input must have exactly one rollout per logical task")
        by_task[key] = row
    if any(row.get("benchmark", "swebench") == "skillsbench" for row in rollouts):
        raise ValueError("Repo/source quota split is not defined for SkillsBench datasets")
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
    tasks = [{**task_descriptor(row), "source_kind": row["source_kind"],
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


def make_event_split(samples, validation_fraction=0.2, test_fraction=0.2, seed=42):
    """Deterministically stratified random split of individual step/event rows.

    Rows from one rollout/task may appear in all three partitions, matching the
    original DyCon event-level protocol. Source-kind strata are preserved where
    possible; no hidden or label values are used to choose the assignment.
    """
    import numpy as np

    if not samples or validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Event split requires non-empty samples and positive fractions")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Event split must leave training rows")
    groups = defaultdict(list)
    for i, row in enumerate(samples):
        groups[row.get("source_kind", "unknown")].append(i)
    rng = np.random.default_rng(seed)
    role_by_sample = {}
    for source in sorted(groups):
        indices = np.asarray(groups[source], dtype=np.int64)
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(round(n * test_fraction))) if n >= 3 else max(0, int(round(n * test_fraction)))
        n_val = max(1, int(round(n * validation_fraction))) if n - n_test >= 2 else max(0, int(round(n * validation_fraction)))
        if n_test + n_val >= n:
            n_test = max(1, int(np.floor(n * test_fraction))) if n >= 3 else 0
            n_val = max(1, int(np.floor(n * validation_fraction))) if n - n_test >= 2 else 0
        for i in indices[:n_test]:
            role_by_sample[samples[int(i)]["sample_id"]] = "test"
        for i in indices[n_test:n_test + n_val]:
            role_by_sample[samples[int(i)]["sample_id"]] = "validation"
        for i in indices[n_test + n_val:]:
            role_by_sample[samples[int(i)]["sample_id"]] = "train"
    expected = {s["sample_id"] for s in samples}
    if set(role_by_sample) != expected or set(role_by_sample.values()) != {"train", "validation", "test"}:
        raise ValueError("Event split did not assign every sample to all three roles")
    sample_ids = {role: [s["sample_id"] for s in samples if role_by_sample[s["sample_id"]] == role]
                  for role in ("train", "validation", "test")}
    task_sets = {role: {s["task_group"] for s in samples if role_by_sample[s["sample_id"]] == role}
                 for role in sample_ids}
    rollout_sets = {role: {s["rollout_id"] for s in samples if role_by_sample[s["sample_id"]] == role}
                    for role in sample_ids}
    return {"schema_version": 3, "strategy": "event_random_v1", "seed": seed,
            "validation_fraction": validation_fraction, "test_fraction": test_fraction,
            "sample_ids": sample_ids,
            "sample_counts": {role: len(ids) for role, ids in sample_ids.items()},
            "task_counts": {role: len(task_sets[role]) for role in sample_ids},
            "rollout_counts": {role: len(rollout_sets[role]) for role in sample_ids},
            "source_sample_counts": {role: dict(Counter(samples[i].get("source_kind", "unknown")
                                                         for i, s in enumerate(samples)
                                                         if role_by_sample[s["sample_id"]] == role))
                                     for role in sample_ids},
            "sample_order_sha256": hash_json([s["sample_id"] for s in samples]),
            "role_by_sample": role_by_sample}


def make_task_threeway_split(rollouts, samples, validation_fraction=0.2, test_fraction=0.2, seed=42):
    """Source-stratified three-way split with entire logical tasks as units.

    A task with several rollout sources belongs to a source-profile stratum,
    determined by the sorted set of its source kinds. No labels or hidden values
    influence assignment. Globally rounded task quotas use largest remainders.
    """
    import itertools

    import numpy as np

    if (not rollouts or not samples or not math.isfinite(validation_fraction)
            or not math.isfinite(test_fraction) or validation_fraction <= 0 or test_fraction <= 0
            or validation_fraction + test_fraction >= 1):
        raise ValueError("Task split requires valid fractions and non-empty inputs")
    by_task = defaultdict(list)
    for row in rollouts:
        by_task[task_group(row)].append(row)
    if len({r["rollout_id"] for r in rollouts}) != len(rollouts):
        raise ValueError("Duplicate rollout ID")
    for rows in by_task.values():
        validate_same_task(rows)
    n_test = round(len(by_task) * test_fraction)
    n_val = round(len(by_task) * validation_fraction)
    if min(n_test, n_val, len(by_task) - n_test - n_val) < 1:
        raise ValueError("Task fractions must produce three non-empty partitions")
    strata = defaultdict(list)
    for key, rows in sorted(by_task.items()):
        profile = "|".join(sorted({r["source_kind"] for r in rows}))
        strata[profile].append(key)

    def allocate(counts, total):
        population = sum(counts.values())
        ideal = {key: total * n / population for key, n in counts.items()}
        quota = {key: math.floor(value) for key, value in ideal.items()}
        for key in sorted(counts, key=lambda k: (-(ideal[k] - quota[k]), k))[:total-sum(quota.values())]:
            quota[key] += 1
        return quota

    test_quota = allocate({key: len(keys) for key, keys in strata.items()}, n_test)
    val_quota = allocate({key: len(keys) - test_quota[key] for key, keys in strata.items()}, n_val)
    rng = np.random.default_rng(seed)
    task_roles = {}
    for profile, keys in sorted(strata.items()):
        shuffled = np.array(keys, dtype=object)
        rng.shuffle(shuffled)
        a, b = test_quota[profile], test_quota[profile] + val_quota[profile]
        for role, keys_for_role in (("test", shuffled[:a]), ("validation", shuffled[a:b]),
                                    ("train", shuffled[b:])):
            task_roles.update({str(key): role for key in keys_for_role})
    task_rows, rollout_rows = [], []
    for key, rows in sorted(by_task.items()):
        first = rows[0]
        task_rows.append({**task_descriptor(first), "split": task_roles[key],
                          "source_kinds": sorted({r["source_kind"] for r in rows}),
                          "rollout_ids": sorted(r["rollout_id"] for r in rows)})
        for row in sorted(rows, key=lambda r: r["rollout_id"]):
            rollout_rows.append({"task_group": key, "rollout_id": row["rollout_id"],
                                 "trajectory_sha256": row["trajectory_sha256"], "split": task_roles[key]})
    roles = ("train", "validation", "test")
    hashes = {role: {r["trajectory_sha256"] for r in rollout_rows if r["split"] == role} for role in roles}
    if any(hashes[a] & hashes[b] for a, b in itertools.combinations(roles, 2)):
        raise ValueError("Identical trajectory SHA would cross task partitions")
    rollout_to_task = {r["rollout_id"]: key for key, rows in by_task.items() for r in rows}
    role_by_sample = {}
    for sample in samples:
        if sample["sample_id"] in role_by_sample:
            raise ValueError("Duplicate sample ID")
        key = rollout_to_task[sample["rollout_id"]]
        if sample["task_group"] != key:
            raise ValueError("Sample/task identity mismatch")
        role_by_sample[sample["sample_id"]] = task_roles[key]
    sample_ids = {role: [s["sample_id"] for s in samples if role_by_sample[s["sample_id"]] == role]
                  for role in roles}
    if not all(sample_ids.values()):
        raise ValueError("Task split produced an empty sample partition")
    return {"schema_version": 4, "strategy": "task_random_v1", "seed": seed,
            "validation_fraction": validation_fraction, "test_fraction": test_fraction,
            "tasks": task_rows, "rollouts": rollout_rows,
            "sample_ids": sample_ids, "role_by_sample": role_by_sample,
            "task_counts": dict(Counter(task_roles.values())),
            "rollout_counts": dict(Counter(r["split"] for r in rollout_rows)),
            "sample_counts": {role: len(ids) for role, ids in sample_ids.items()},
            "stratification": "task source-kind profile; largest remainder; no label stratification",
            "source_profile_quotas": {profile: {"test": test_quota[profile],
                                               "validation": val_quota[profile],
                                               "train": len(keys) - test_quota[profile] - val_quota[profile]}
                                      for profile, keys in sorted(strata.items())},
            "task_overlap_counts": {a + "__" + b: 0 for a, b in itertools.combinations(roles, 2)},
            "sample_order_sha256": hash_json([s["sample_id"] for s in samples])}

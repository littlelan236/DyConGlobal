"""Explicit, auditable cleanup for the superseded October 18 capture cohort.

Default is a dry run. Does not infer deletions from timestamps or broad globs.
Run only after the old cohort has exited; preserve submitted bytes verbatim.
"""

import argparse
import shutil
from pathlib import Path

from baseline.catalog import Catalog
from core.disk_budget import DiskBudget
from core.io import now, read_json, sha256, write_json
from core.locking import file_lock
from enrichment.online_trace import finalize_token_events

OLD = "capture-repaired200-20261018-162700"
TESTS = {
    "repair-20261018", "repair-e2e-20261018-c",
    "repair-accept-20261018-a", "repair-accept-20261018-b",
    "capture-repaired200-20261018-162000",
    "capture-verified200-20261018-120720",  # invalid parser/startup attempt
    "speed-baseline-20261018-034421", "speed-baseline-20261018-035007",
    "speed-capture-20261018-035007",
    *{f"repair-accept-20261018-{x}-{y}" for x in ("a", "b")
      for y in ("capture", "combined", "plain", "steering")},
    *{f"throughput-20261018-queue-drain-{i:02}" for i in range(1, 5)},
}
DIAGNOSTICS = {"fullgraph-smoke-20261017", "gpu47-acceptance", "dense-diagnostic"}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    audit = args.audit.resolve()
    nas_parent = Path("/VM/wangtianyi/DyConGlobal/hidden/online").resolve()
    nas = nas_parent / OLD
    catalog = Catalog(root)
    data = catalog.read()
    old = {k: v for k, v in data["rollouts"].items() if v["run_id"] == OLD}
    keep = {k: v for k, v in old.items() if v["submission_status"] == "submitted"}
    failed = {k: v for k, v in old.items() if v["termination_reason"] == "ContextWindowExceededError"}
    assert len(keep) == 37 and len(failed) == 163 and set(keep) | set(failed) == set(old)
    removed = {k: v for k, v in data["rollouts"].items() if k in failed or v["run_id"] in TESTS}
    assert not any(v.get("selected_evaluation_id") for v in removed.values())
    assert not any(set(e.get("outcomes", {})) & set(removed) for e in data["evaluations"].values())
    selection = read_json(root / "runs" / OLD / "selection.json")
    previous = set(selection["task_keys"])
    rest = set(data["tasks"]) - previous
    rerun = {v["task_key"] for v in failed.values()}
    assert len(rest) == 258 and len(rerun) == 163 and not rest & previous
    keys = sorted(rest | rerun)
    assert len(keys) == 421 and not set(keys) & {v["task_key"] for v in keep.values()}
    next_selection = {
        "task_keys": keys, "tasks": {k: data["tasks"][k] for k in keys},
        "source": {"run_id": OLD, "unsubmitted": 163, "other_nas_tasks": 258,
                   "preserved_submitted": 37},
    }
    deletion = set()

    def remove(path):
        path = Path(path).absolute()
        resolved = path.resolve()
        assert (resolved.is_relative_to(root) and resolved != root) or (
            resolved.is_relative_to(nas_parent) and resolved != nas_parent
        ), path
        if path.exists():
            deletion.add(path)

    for rid, row in removed.items():
        for field in ("trajectory_path", "patch_path"):
            parent = (root / row[field]).parent
            assert parent.name == rid
            remove(parent)
        remove(root / "runs" / row["run_id"] / "tasks" / data["tasks"][row["task_key"]]["instance_id"])
    for name in TESTS:
        for base in ("runs", "hidden/spool", "hidden/online", "hidden/reports", "enrichment/outputs", "state/supervisors"):
            remove(root / base / name)
        remove(nas_parent / name)
    for name in DIAGNOSTICS:
        remove(root / "hidden/online" / name)

    sessions = {}
    for path in (nas / "sessions").glob("*/finalization.json"):
        row = read_json(path)
        if row["rollout_id"] in failed:
            sessions[path.parent.name] = row
            remove(path.parent)
    assert len(sessions) == 163
    shard_count = request_count = 0
    for path in (nas / "index").glob("*.json"):
        events = read_json(path)
        deleted_events = [e for e in events if e["session_id"] in sessions]
        assert not deleted_events or len(deleted_events) == len(events), "Mixed-session shard requires repacking"
        if deleted_events:
            remove(path)
            for e in deleted_events:
                remove(nas / e["tensor_locator"]["shard"])
            shard_count += 1
    for path in (nas / "requests").glob("*.json"):
        if read_json(path)["session_id"] in sessions:
            remove(path)
            request_count += 1
    for path in (nas / "publications").glob("*.json"):
        files = read_json(path)["files"]
        flags = [(nas / f) in deletion or bool(set((nas / f).parents) & deletion) for f in files]
        assert not any(flags) or all(
            flag or name in {"inventory.json", "capture-config.json"}
            for name, flag in zip(files, flags)
        ), "Publication mixes deleted and retained task data"
        if any(flags):
            remove(path)
    # Remove the lock siblings of individual immutable files, too.
    for path in list(deletion):
        if path.is_file():
            remove(path.with_name(path.name + ".lock"))

    # Prove all retained trajectory/patch bytes and published NAS bytes survive unchanged.
    proofs = {}
    for row in keep.values():
        for field in ("trajectory", "patch"):
            path = root / row[field + "_path"]
            assert sha256(path) == row[field + "_sha256"]
            proofs[str(path)] = row[field + "_sha256"]
    for path in (nas / "publications").glob("*.json"):
        if path not in deletion:
            for rel, digest in read_json(path)["files"].items():
                proofs[str(nas / rel)] = digest
    for path, digest in proofs.items():
        assert Path(path) not in deletion and not set(Path(path).parents) & deletion
        assert sha256(path) == digest, path
    paths = sorted((p for p in deletion if not set(p.parents) & deletion), key=str)
    stats = {"previous_contextlimit_rollouts": len(failed), "test_rollouts": len(removed)-len(failed),
             "preserved_submitted": len(keep), "removed_nas_shards": shard_count,
             "removed_request_ledgers": request_count, "next_tasks": len(keys)}
    audit.mkdir(parents=True, exist_ok=True)
    write_json(audit / "selection.json", next_selection)
    write_json(audit / "plan.json", {"created_at": now(), "stats": stats,
               "delete": list(map(str, paths)), "removed_rollouts": removed,
               "preserved_rollouts": keep, "preserved_checksums": proofs})
    print(stats, flush=True)
    if not args.apply:
        return
    # Small metadata snapshots permit auditing; deleted trajectory/tensor bytes are not backed up.
    for name in ("baseline", "hidden", "enrichment"):
        shutil.copyfile(root / name / "manifest.json", audit / (name + "-manifest-before.json"))
    for path in paths:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    with catalog.transaction() as current:
        assert current == data, "Catalog changed during offline cleanup"
        for rid in removed:
            del current["rollouts"][rid]
        for group in current["baseline_selections"].values():
            for task, rid in list(group.items()):
                if rid in removed:
                    del group[task]
    for name in ("hidden", "enrichment"):
        path = root / name / "manifest.json"
        with file_lock(path.with_suffix(".lock")):
            manifest = read_json(path)
            for sid, row in list(manifest.get("online_captures", {}).items()):
                legacy = row.get("samples", [{}])[0]
                diagnostic = legacy.get("task_id") in {"smoke-task", "gpu-smoke-task"}
                if sid in sessions or row.get("run_id") in TESTS or diagnostic or any(d in str(row.get("export_root", "")) for d in DIAGNOSTICS):
                    del manifest["online_captures"][sid]
            write_json(path, manifest)
    derived = root / "enrichment/outputs" / OLD
    staging = derived.with_name(OLD + "-retained")
    inventory = read_json(nas / "inventory.json")
    finalize_token_events(nas, staging, open_ids=inventory["open_token_ids"], close_ids=inventory["close_token_ids"])
    assert {r["rollout_id"] for r in read_json(staging / "index.json")} == set(keep)
    shutil.rmtree(derived)
    staging.rename(derived)
    # Historical progress remains a receipt of the old run; annotate intentionally removed products.
    write_json(root / "runs" / OLD / "cleanup.json", {"at": now(), "audit": str(audit),
               "removed_rollout_ids": sorted(failed), "preserved_rollout_ids": sorted(keep)})
    budget = DiskBudget(root, root / "hidden/spool", 50 * 1024**3)
    released = []
    for key, row in budget.allocations().items():
        if any(Path(row["path"]).is_relative_to(root / "hidden/spool" / name) for name in TESTS):
            budget.release(key)
            released.append(key)
    for path, digest in proofs.items():
        assert sha256(path) == digest, path
    assert not set(removed) & set(catalog.read()["rollouts"])
    write_json(audit / "result.json", {"completed_at": now(), "stats": stats,
               "preserved_checksums_verified": len(proofs), "released_reservations": released,
               "retained_coverage": read_json(derived / "coverage.json")})
    print("CLEANUP COMPLETE", flush=True)


if __name__ == "__main__":
    main()

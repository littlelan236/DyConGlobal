"""Gold-patch environment audit and NAS Docker-load timing benchmark."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

from baseline.evaluation.harness import install_harness_adapter
from core.io import hash_json, now, read_json, write_json
from runtime.containers.factory import ContainerFactory


def load_cfg(root: Path, instances: Path, output: Path) -> dict:
    from core.config import resolve

    cfg = resolve(root / "config.yaml", [
        "paths.nas_root=/VM",
        "environment.requirements_cache=state/gold-image-audit/requirements-cache",
        "execution.harness_workers=4",
        "images.load_workers=4",
        "images.cleanup=true",
        "timeouts.evaluation_seconds=900",
    ])
    cfg["evaluation_run_id"] = output.name
    cfg["project_root"] = str(root)
    cfg["audit_instances"] = str(instances)
    return cfg


def audit(root: Path, instances_path: Path, audit_id: str, workers: int, only_file: Path | None = None) -> dict:
    out = root / "state/gold-image-audit" / audit_id
    out.mkdir(parents=True, exist_ok=False)
    instances = read_json(instances_path)
    if only_file is not None:
        wanted = set(read_json(only_file))
        instances = [x for x in instances if x["instance_id"] in wanted]
    wheelhouse = str(root / "runtime/cache/gold-wheels")
    instances = [{**row, "environment_overrides": {**(row.get("environment_overrides") or {}), **({"wheelhouse": wheelhouse, "repo_pins": {"sympy/sympy": ["scipy==1.13.1", "numpy==1.26.4"]}} if row["instance_id"] == "sympy__sympy-23262" else {}), **({"wheelhouse": wheelhouse} if row["instance_id"] == "pylint-dev__pylint-4661" else {})}} for row in instances]
    if not instances or not all(x.get("nas_image") for x in instances):
        raise ValueError("Expected instances with NAS image references")
    cfg = load_cfg(root, instances_path, out)
    write_json(out / "audit-config.json", {k: v for k, v in cfg.items() if k not in {"project_root"}})
    preflight = {"tasks": len(instances), "started_at": now(), "deep_verify": {}, "errors": []}
    from runtime.images.nas import NasImageStore

    for row in instances:
        try:
            store = NasImageStore(cfg, row["nas_image"])
            descriptors = store.verify(deep=False)
            preflight["deep_verify"][row["instance_id"]] = {
                "image_id": store.image_id,
                "requested": store.record.get("requested"),
                "compressed_bytes": store.record.get("compressed_bytes"),
                "manifest_sha256": row["nas_image"]["manifest_sha256"],
                "descriptor_count": len(descriptors),
            }
        except Exception as exc:
            preflight["errors"].append({"instance_id": row["instance_id"], "error": repr(exc)})
    preflight["finished_at"] = now()
    write_json(out / "nas-preflight.json", preflight)
    if preflight["errors"]:
        raise RuntimeError(f"NAS preflight failed for {len(preflight['errors'])} tasks")

    # The gold run is deliberately independent of the catalog: this is an
    # environment audit, and its temporary artifacts must not become baseline data.
    factory = ContainerFactory(cfg)
    output = out / "harness"
    output.mkdir()
    factory.lock_images(instances)
    official = install_harness_adapter(cfg, factory, instances, output)
    started = time.perf_counter()

    def worker(row):
        iid = row["instance_id"]
        task_out = output / "tasks" / iid
        task_out.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        result = {"instance_id": iid, "image_id": factory.images[iid]["image_id"],
                  "patch_sha256": sha256_text(row.get("patch", "")), "status": "infra_error",
                  "resolved": None}
        try:
            spec = official.make_test_spec(row, namespace="swebench")
            t1 = time.perf_counter()
            prediction = {"instance_id": iid, "model_name_or_path": "gold-patch-audit", "model_patch": row.get("patch", "")}
            graded = official.run_instance(spec, prediction, False, False, factory.client,
                                           "audit-" + hash_json({"audit": audit_id, "task": iid})[:24],
                                           row.get("evaluation_timeout_seconds", cfg["timeouts"]["evaluation_seconds"]))
            t2 = time.perf_counter()
            result.update(status="completed" if graded["completed"] else "infra_error",
                          resolved=bool(graded["resolved"]) if graded["completed"] else None,
                          completed=graded.get("completed"), outcome=graded)
            result["spec_seconds"] = t1 - t0
            result["evaluation_seconds"] = t2 - t1
            result["total_seconds"] = t2 - t0
        except Exception as exc:
            result.update(error=repr(exc), total_seconds=time.perf_counter() - t0)
        write_json(task_out / "result.json", result)
        return result

    results = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, row) for row in instances]
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                results.append(result)
                write_json(out / "progress.json", {"completed": i, "total": len(instances), "results": results})
                print(f"{i}/{len(instances)} {result['instance_id']} {result['status']}", flush=True)
    finally:
        try:
            factory.cleanup_owned(out.name)
        finally:
            official.dycon_restore()
            factory.client.close()
    elapsed = time.perf_counter() - started
    timing = collect_timing(cfg, out, results, elapsed)
    summary = {"audit_id": audit_id, "started_at": preflight["started_at"], "finished_at": now(),
               "tasks": len(instances), "workers": workers, "elapsed_seconds": elapsed,
               "completed": sum(r["status"] == "completed" for r in results),
               "resolved": sum(r.get("resolved") is True for r in results),
               "infra_error": sum(r["status"] != "completed" for r in results),
               "results": sorted(results, key=lambda x: x["instance_id"]), "load_timing": timing}
    write_json(out / "summary.json", summary)
    write_json(out / "status.json", {"stage": "complete", "updated_at": now(), "summary": str(out / "summary.json")})
    return summary


def sha256_text(value: str) -> str:
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def collect_timing(cfg, out, results, elapsed):
    # ImageLeaseManager writes one load log per imported digest. Extract Docker's
    # own log size and correlate with task image identity; load wall time is
    # measured by the audit's explicit load timing hook when present.
    state = Path(cfg["project_root"]) / "state/images"
    logs = []
    for p in state.glob("**/*.load.timing.json"):
        try:
            logs.append(read_json(p))
        except (OSError, ValueError):
            pass
    return {"samples": logs, "sample_count": len(logs),
            "docker_load_seconds_total": sum(float(x.get("seconds", 0)) for x in logs),
            "wall_seconds": elapsed,
            "note": "load timing is emitted by ImageLeaseManager around docker load; absent samples indicate images were already present or the run stopped before import"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--instances", type=Path, required=True)
    p.add_argument("--audit-id", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only-file", type=Path)
    a = p.parse_args()
    if a.workers < 1 or a.workers > 4:
        raise ValueError("workers must be 1..4")
    print(json.dumps(audit(a.root.resolve(), a.instances.resolve(), a.audit_id, a.workers, a.only_file.resolve() if a.only_file else None), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

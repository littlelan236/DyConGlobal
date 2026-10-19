"""Retry unsuccessful NAS gold tasks with a frozen config and retained run provenance."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import traceback
from pathlib import Path

import docker

from baseline.evaluation.harness import install_harness_adapter
from core.io import identifier, now, read_json, sha256, write_json
from core.locking import file_lock
from runtime.containers.factory import ContainerFactory
from runtime.images.nas import NasImageStore
from runtime.scripts.consolidate_gold_audits import consolidate
from runtime.scripts.gold_image_audit import audit
from runtime.scripts.nas_cold_gold import purge


def retry(root, base_summary, instances_file, config_snapshot, run_id, workers=4):
    run_id = identifier(run_id)
    out = root / "state/gold-image-audit" / run_id
    out.mkdir(parents=True, exist_ok=False)
    original = read_json(base_summary)
    wanted = {
        r["instance_id"]
        for r in original["results"]
        if r.get("resolved") is not True or not r.get("cold_lifecycle_verified")
    }
    instances = [r for r in read_json(instances_file) if r["instance_id"] in wanted]
    if len(instances) != len(wanted) or len(wanted) != 22:
        raise ValueError("Expected all 22 unsuccessful tasks and exact frozen task definitions")
    cfg = copy.deepcopy(read_json(config_snapshot))
    cfg.update(project_root=str(root), evaluation_run_id=run_id)
    cfg["execution"]["harness_workers"] = workers
    cfg["images"].update(cleanup=True, load_workers=4)
    cfg["require_cold_images"] = True
    selection = out / "instances.json"
    write_json(selection, instances)
    write_json(out / "frozen-config.json", cfg)
    write_json(
        out / "provenance.json",
        {
            "base_summary": str(base_summary),
            "base_summary_sha256": sha256(base_summary),
            "config_source_snapshot": str(config_snapshot),
            "config_snapshot_sha256": sha256(config_snapshot),
            "repair_sources": {
                name: sha256(root / name)
                for name in [
                    "runtime/containers/repairs.py",
                    "baseline/evaluation/metadata.py",
                    "baseline/evaluation/harness.py",
                    "runtime/scripts/gold_image_audit.py",
                ]
            },
        },
    )
    status = {"started_at": now(), "tasks": len(instances), "base_resolved": original["resolved"]}
    summaries = [base_summary]

    def stage(value, **extra):
        status.update(stage=value, updated_at=now(), **extra)
        write_json(out / "status.json", status)
        print(value, flush=True)

    try:
        stage("metadata-preflight")
        factory = ContainerFactory(cfg)
        adapter = install_harness_adapter(cfg, factory, instances, out / "metadata-preflight")

        def build(row):
            try:
                adapter.make_test_spec(row, namespace="swebench")
                return {"instance_id": row["instance_id"], "ok": True}
            except Exception:
                return {"instance_id": row["instance_id"], "ok": False, "error": traceback.format_exc()}

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                metadata = list(pool.map(build, instances))
        finally:
            adapter.dycon_restore()
            factory.client.close()
        write_json(out / "metadata-preflight.json", metadata)
        print(
            "Metadata specs ready: " + str(sum(r["ok"] for r in metadata)) + "/" + str(len(metadata)),
            flush=True,
        )
        for row in instances:
            NasImageStore(cfg, row["nas_image"]).verify()
        pending = instances
        for attempt in range(1, 4):
            if not pending:
                break
            stage("cold-preflight", attempt=attempt)
            preflight = out / ("cold-preflight-" + str(attempt))
            preflight.mkdir()
            # Original blobs were deep-verified on NAS before the initial purge.
            # Every actual import also hashes its complete OCI graph again.
            purge(cfg, pending, preflight, set())
            ids = preflight / "task-ids.json"
            write_json(ids, [r["instance_id"] for r in pending])
            aid = run_id + "-attempt" + str(attempt)
            stage("gold-running", attempt=attempt, audit_id=aid)
            result = audit(root, selection, aid, workers, ids, cold_start=True, cfg=cfg)
            summary_path = root / "state/gold-image-audit" / aid / "summary.json"
            summaries.append(summary_path)
            latest = {r["instance_id"]: r for r in result["results"]}
            pending = [row for row in pending if latest[row["instance_id"]]["status"] != "completed"]
        stage("consolidating")
        result = consolidate(summaries, out / "consolidated-258.json")
        client = docker.DockerClient(base_url=cfg["environment"]["docker_host"], timeout=120)
        try:
            ids = {NasImageStore(cfg, r["nas_image"]).image_id for r in instances}
            remaining = sorted(ids & {im["Id"] for im in client.api.images(all=True)})
            containers = [c["Id"] for c in client.api.containers(all=True) if c["ImageID"] in ids]
        finally:
            client.close()
        stage(
            "passed" if result["passed"] and not remaining and not containers else "needs-investigation",
            total_tasks=result["tasks"],
            total_resolved=result["resolved"],
            cold_lifecycle_verified=result["cold_lifecycle_verified"],
            failed=result["failed"],
            remaining_retry_images=remaining,
            remaining_retry_containers=containers,
            finished_at=now(),
        )
        return 0 if status["stage"] == "passed" else 1
    except Exception:
        stage("failed", error=traceback.format_exc(), finished_at=now())
        raise
    finally:
        # A retry orchestration error must never erase the original 236 successes.
        merged = consolidate(summaries, out / "consolidated-258.json")
        status.update(
            total_tasks=merged["tasks"],
            total_resolved=merged["resolved"],
            cold_lifecycle_verified=merged["cold_lifecycle_verified"],
            failed=merged["failed"],
        )
        write_json(out / "status.json", status)
        write_json(out / "summary.json", status)
        doc = root / "docs" / (run_id + ".md")
        doc.write_text(
            "# 258 tasks：22 个失败样例修复与重测\n\n"
            "修复在统一容器入口恢复指定任务的运行网络；依赖 bootstrap 保持离线。"
            "元数据缓存复用真实下载内容并增加有界重试；所有重试使用同一配置快照。\n\n"
            "下列结果合并原始首轮与本次重试；原始 patch 和评分规则保留。\n\n"
            "```json\n" + json.dumps(status, ensure_ascii=False, indent=2) + "\n```\n"
        )
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--base-summary", type=Path, required=True)
    p.add_argument("--instances", type=Path, required=True)
    p.add_argument("--config-snapshot", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = p.parse_args()
    with file_lock(args.root / "state/gold-image-audit/repair22.lock"):
        raise SystemExit(
            retry(
                args.root.resolve(),
                args.base_summary.resolve(),
                args.instances.resolve(),
                args.config_snapshot.resolve(),
                args.run_id,
                args.workers,
            )
        )

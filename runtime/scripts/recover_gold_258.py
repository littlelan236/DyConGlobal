"""Recover export, verify NAS, purge backed-up images, then run all 258 gold patches."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import docker

from baseline.catalog import Catalog
from core.config import resolve
from core.io import now, read_json, write_json
from core.locking import file_lock
from runtime.containers.factory import ContainerFactory, TaskContext
from runtime.images.nas import NasImageStore
from runtime.scripts.export_task_images_to_nas import export
from runtime.scripts.gold_image_audit import load_cfg
from runtime.scripts.nas_cold_gold import purge, run_cohort


def run(root, signal_fifo=None):
    out = root / "state/gold-image-audit/nas-cold-258-20261018"
    if out.exists():
        previous = read_json(out / "status.json")
        if previous.get("stage") != "failed":
            raise ValueError("Existing recovery run is not failed; refusing to overwrite")
        history = out / ("failed-" + previous["finished_at"].replace(":", "-") + ".json")
        write_json(history, previous)
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    status = {"started_at": now(), "tasks": 258}

    def stage(value, **extra):
        status.update(stage=value, updated_at=now(), **extra)
        write_json(status_path, status)
        print(value, flush=True)

    def notify(value):
        # Optional one-shot event channel. Never wait for a missing reader.
        if signal_fifo:
            import os

            try:
                fd = os.open(signal_fifo, os.O_WRONLY | os.O_NONBLOCK)
                try:
                    os.write(fd, (json.dumps(value) + "\n").encode())
                finally:
                    os.close(fd)
            except OSError:
                pass

    try:
        stage("exporting")
        evidence = root / "state/nas-export-258-recovery-20261018"
        proof = evidence / "verification.json"
        if proof.exists() and read_json(proof).get("verified"):
            receipt = read_json(evidence / "receipt.json")
            print("Reusing completed verified export", flush=True)
        else:
            receipt = export(root, evidence=evidence)
        cfg = resolve(root / "config.yaml")
        catalog = Catalog(root)
        manifest = catalog.read()
        rows = sorted((t["instance"] for t in manifest["tasks"].values()), key=lambda r: r["instance_id"])
        assert len(rows) == 258
        stage("verifying-export")
        for row in rows:
            store = NasImageStore(cfg, row["nas_image"])
            store.verify()
        # Catalog mutation only adds NAS environment metadata; rollout records
        # and task partitions must remain identical to the pre-export snapshot.
        before = read_json(evidence / "catalog-before.json")
        if manifest["rollouts"] != before["rollouts"]:
            raise ValueError("Rollout catalog changed; inspect before image deletion")
        for key, task in manifest["tasks"].items():
            for field in ("instance_id", "partition", "purposes"):
                if task.get(field) != before["tasks"][key].get(field):
                    raise ValueError("Task selection metadata changed")
        write_json(
            evidence / "verification.json",
            {
                "verified": True,
                "tasks": 258,
                "deep_verified": receipt["deep_verified"],
                "receipt": receipt,
                "catalog_metadata_unchanged": True,
                "finished_at": now(),
            },
        )
        # Convert catalog-relative NAS references for the existing multi-store
        # harness config; no catalog paths are rewritten.
        rows = [
            {
                **r,
                "nas_image": {
                    **r["nas_image"],
                    "store": str(Path(cfg["paths"]["nas_root"]) / r["nas_image"]["store"]),
                },
            }
            for r in rows
        ]
        selection = out / "instances.json"
        write_json(selection, rows)
        audit_cfg = load_cfg(root, selection, out)
        stage("testing-nas-roundtrip")
        # Prove real cold imports of both supported root types before deleting
        # the rest of the cohort. Existing stopped containers are not touched here.
        client = docker.DockerClient(base_url=cfg["environment"]["docker_host"], timeout=120)
        used = {c["ImageID"] for c in client.api.containers(all=True)}
        client.close()
        smoke_rows = []
        kinds = set()
        for row in rows:
            store = NasImageStore(audit_cfg, row["nas_image"])
            kind = store.record["root_media_type"]
            if kind not in kinds and store.image_id not in used:
                smoke_rows.append(row)
                kinds.add(kind)
        if len(smoke_rows) != 2:
            raise ValueError("Expected single-manifest and OCI-index smoke tasks")
        for row in smoke_rows:
            smoke = out / "smoke" / row["instance_id"]
            smoke.mkdir(parents=True, exist_ok=True)
            purge(audit_cfg, [row], smoke, set())
            factory_cfg = {
                **audit_cfg,
                "require_cold_images": True,
                "image_events_path": str(smoke / "image-lifecycle.jsonl"),
            }
            factory = ContainerFactory(factory_cfg)
            ctx = TaskContext(out.name + "-smoke", row["instance_id"], "roundtrip", "harness", smoke)
            container = None
            try:
                container = factory.create(row, ctx)
                container.start()
                probe = factory.execute(container, "git rev-parse HEAD && python --version", 60)
                if probe["returncode"]:
                    raise RuntimeError(probe["output"])
            finally:
                if container:
                    factory.close(container)
                factory.cleanup_owned(ctx.run_id)
            if factory.image_manager._exists(NasImageStore(factory_cfg, row["nas_image"]).image_id):
                raise RuntimeError("Smoke image was not cleaned")
            factory.client.close()
            write_json(
                smoke / "roundtrip.json",
                {
                    "passed": True,
                    "instance_id": row["instance_id"],
                    "media_type": NasImageStore(audit_cfg, row["nas_image"]).record["root_media_type"],
                    "probe": probe,
                    "finished_at": now(),
                },
            )
            print(f"ROUNDTRIP PASS {row['instance_id']}", flush=True)
        stage("full-cold-gold-pipeline")
        allowed = set(read_json(root / "state/gold-image-audit/approved-stopped-18.json"))
        # run_cohort independently hashes the entire graph before removing the
        # remaining images and preserves every authorized stopped repository.
        report = run_cohort(root, out, "legacy258", rows, allowed)
        stage(
            "passed" if report["passed"] else "needs-investigation",
            resolved=report["resolved"],
            cold_lifecycle_verified=report["cold_lifecycle_verified"],
            failed=report["failed"],
            remaining_images=report["remaining_images"],
            finished_at=now(),
        )
        write_json(out / "summary.json", report)
        notify(status)
        return 0 if report["passed"] else 1
    except BaseException:
        stage("failed", error=traceback.format_exc(), finished_at=now())
        notify(status)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--signal-fifo", type=Path)
    args = p.parse_args()
    with file_lock(args.root / "state/nas-export-258.lock"):
        with file_lock(args.root / "state/gold-image-audit/nas-cold-258.lock"):
            raise SystemExit(run(args.root.resolve(), args.signal_fifo))

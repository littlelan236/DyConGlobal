"""Finalize the four-task network repair once its harness process exits."""

from __future__ import annotations

import argparse
import json
import os
import select
import traceback
from pathlib import Path

import docker

from core.io import now, read_json, write_json
from runtime.scripts.consolidate_gold_audits import consolidate


def finalize(root: Path, pid: int):
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        fd = None
    if fd is not None:
        select.select([fd], [], [])
        os.close(fd)
    base = root / "state/gold-image-audit"
    out = base / "verified200-network-repair-cold-20261017"
    try:
        retry = read_json(out / "summary.json")
        result = consolidate(
            [
                base / "nas-cold-458-20261017-verified200-attempt1/summary.json",
                out / "summary.json",
            ],
            out / "verified200-consolidated.json",
        )
        cfg = read_json(out / "audit-config.json")
        client = docker.DockerClient(base_url=cfg["environment"]["docker_host"], timeout=120)
        try:
            image_ids = {r["image_id"] for r in retry["results"]}
            remaining_images = sorted(image_ids & {im["Id"] for im in client.api.images(all=True)})
            remaining_containers = [
                c["Id"] for c in client.api.containers(all=True) if c["ImageID"] in image_ids
            ]
        finally:
            client.close()
        status = {
            "at": now(),
            "stage": "passed"
            if result["passed"] and not remaining_images and not remaining_containers
            else "needs-investigation",
            "retry_tasks": retry["tasks"],
            "retry_resolved": retry["resolved"],
            "total_tasks": result["tasks"],
            "total_resolved": result["resolved"],
            "cold_lifecycle_verified": result["cold_lifecycle_verified"],
            "failed": result["failed"],
            "remaining_repair_images": remaining_images,
            "remaining_repair_containers": remaining_containers,
        }
        write_json(out / "final-verification.json", status)
        doc = root / "docs/verified200-cold-network-repair-20261017.md"
        with doc.open("a") as stream:
            stream.write(
                "\n\n重测结束后自动汇总（保留原始评测记录）：\n\n```json\n"
                + json.dumps(status, ensure_ascii=False, indent=2)
                + "\n```\n"
            )
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        write_json(
            out / "final-verification.json", {"at": now(), "stage": "error", "error": traceback.format_exc()}
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    finalize(args.root, args.pid)

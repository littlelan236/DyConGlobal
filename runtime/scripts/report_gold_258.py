"""Generate the 258-task gold and Docker load report after the process exits."""

from __future__ import annotations

import argparse
import json
import os
import select
import statistics
from pathlib import Path

from core.io import now, read_json, write_json


def report(root: Path, pid: int):
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        fd = None
    if fd is not None:
        select.select([fd], [], [])
        os.close(fd)
    out = root / "state/gold-image-audit/nas-cold-258-20261018"
    status = read_json(out / "status.json")
    summary = read_json(out / "summary.json") if (out / "summary.json").exists() else {}
    samples = []
    attempts = []
    for path in summary.get("attempt_summaries", []):
        run = read_json(path)
        attempts.append(
            {
                "summary": path,
                "tasks": run["tasks"],
                "resolved": run["resolved"],
                "infra_error": run["infra_error"],
                "wall_seconds": run["elapsed_seconds"],
            }
        )
        samples.extend(run["load_timing"]["samples"])
    seconds = sorted(s["seconds"] for s in samples)
    timing = {"samples": len(samples), "total_seconds": sum(seconds)}
    if seconds:
        timing.update(
            mean_seconds=statistics.mean(seconds),
            median_seconds=statistics.median(seconds),
            p95_seconds=seconds[min(int(0.95 * len(seconds)), len(seconds) - 1)],
            max_seconds=max(seconds),
        )
    result = {
        "generated_at": now(),
        "stage": status["stage"],
        "tasks": 258,
        "resolved": summary.get("resolved"),
        "cold_lifecycle_verified": summary.get("cold_lifecycle_verified"),
        "failed": summary.get("failed"),
        "remaining_images": summary.get("remaining_images"),
        "error": status.get("error"),
        "attempts": attempts,
        "docker_load_timing": timing,
    }
    write_json(out / "final-report.json", result)
    lines = [
        "# 258 tasks NAS 冷加载 gold patch 评测结果",
        "",
        f"生成时间：{result['generated_at']}。状态：{result['stage']}。",
        "",
        f"Gold resolved：{result['resolved']}/258；冷加载及清理验证：{result['cold_lifecycle_verified']}/258。",
        "",
        "仅全部 resolved、逐任务生命周期验证通过且没有目标镜像残留时标记 passed。",
        "",
        "## 本次运行汇总",
        "",
        "```json",
        json.dumps(result, ensure_ascii=False, indent=2),
        "```",
        "",
        "Docker load 时间包含 NAS blob SHA-256 校验、OCI archive 流式生成和 docker load；"
        "多 worker 累计时间不能直接视为总墙钟时间。预先的导出校验、停止容器备份及 smoke 不计入 gold 的加载样本。",
        "",
        "原始任务、补丁、harness 结果及生命周期证据保留在 state/gold-image-audit/ 中的本次运行目录和逐轮目录。",
    ]
    (root / "docs/nas-258-gold-results-20261018.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    report(args.root, args.pid)

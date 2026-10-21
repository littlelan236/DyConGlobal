"""Freeze and import a report-scoped legacy trajectory cohort into the sole catalog.

This is an offline migration adapter. Normal DyConGlobal execution never reads the
legacy tree. A prepared plan is immutable evidence, not another editable split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from catalog.store import Catalog
from catalog.importers.evaluation import import_evaluation
from rollout.artifacts import import_rollout
from core.io import hash_json, identifier, now, read_json, sha256, write_json
from core.locking import file_lock


def report_runs(report):
    runs = {}
    for line in Path(report).read_text(encoding="utf-8-sig").splitlines():
        cells = [s.strip() for s in line.strip().strip("|").split("|")]
        if len(cells) == 5 and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]+", cells[0]) and cells[1].isdigit():
            runs[identifier(cells[0])] = int(cells[1])
    if not runs or "canonical-baseline" not in runs:
        raise ValueError("Expected the report's per-run table including canonical-baseline")
    return runs


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def outcome_status(row):
    return "resolved" if row["resolved"] is True else row["submission_status"]


def prepare(root, source, report, migration_id):
    root, source = Path(root).resolve(), Path(source).resolve()
    out = root / "catalog/imports" / identifier(migration_id)
    if out.exists():
        raise FileExistsError(out)
    expected = report_runs(report)
    configs, instances, selected, groups = {}, {}, {}, {}
    counts, snapshots = {}, {}
    for run in expected:
        canonical = run == "canonical-baseline"
        cfg_file = source / (
            "baseline/config/resolved-config.json" if canonical else f"runs/{run}/resolved-config.json"
        )
        cfg = read_json(cfg_file)
        configs[run] = {k: cfg.get(k) for k in ["model", "generation", "steering", "agent", "serving"]}
        # Keep only non-secret research/runtime identity, not endpoint credentials.
        configs[run]["serving"] = {
            k: cfg.get("serving", {}).get(k) for k in ["backend", "reasoning_parser", "tool_parser"]
        }
        data_file = Path(cfg["benchmark"]["instances"])
        if not data_file.is_absolute():
            data_file = source / data_file
        if str(data_file) not in snapshots:
            snapshots[str(data_file)] = {"sha256": sha256(data_file), "rows": read_json(data_file)}
        for instance in snapshots[str(data_file)]["rows"]:
            iid = instance["instance_id"]
            if iid in instances and any(
                instance[k] != instances[iid][k] for k in ["repo", "base_commit", "problem_statement"]
            ):
                raise ValueError(f"Conflicting task metadata: {iid}")
            instances.setdefault(iid, instance)
        base = source / ("baseline/traj" if canonical else f"runs/{run}/tasks")
        files = sorted(base.glob("*.traj.json") if canonical else base.glob("*/attempts/*/*.traj.json"))
        counts[run] = len(files)
        for f in files:
            raw = f.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            x = json.loads(raw)
            iid = x.get("instance_id") or f.name.removesuffix(".traj.json")
            info = x.get("info", {})
            result_path = source / f"runs/{run}/tasks/{iid}/result.json"
            result = read_json(result_path) if result_path.exists() else {}
            alias = {
                "path": str(f.relative_to(source)),
                "run_id": run,
                "attempt_id": "canonical" if canonical else f.parent.name,
                "selected": canonical or result.get("attempt_id") == f.parent.name,
            }
            model = cfg["model"]
            identity = {"model_name": Path(model["path"]).name, "checkpoint_verified": False}
            row = groups.setdefault(
                digest,
                {
                    "sha256": digest,
                    "task_id": iid,
                    "aliases": [],
                    "patch_sha256": digest_text(info.get("submission") or ""),
                    "model_identity": identity,
                },
            )
            if row["task_id"] != iid or row["model_identity"] != identity:
                raise ValueError("Duplicate trajectory has conflicting task/model identity")
            row["aliases"].append(alias)
            if alias["selected"]:
                selected[run, iid] = digest
    canonical_manifest = read_json(source / "baseline/manifest.json")
    canonical_rows = {r["instance_id"]: r for r in canonical_manifest["trajectories"]}
    used = {r["task_id"] for r in groups.values()}
    revision = "legacy-snapshot-sha256-" + hash_json({i: instances[i] for i in sorted(used)})
    specs = []
    for digest, row in sorted(groups.items()):
        aliases = row["aliases"]
        canonical = next((a for a in aliases if a["run_id"] == "canonical-baseline"), None)
        primary = aliases[0]
        if canonical:
            legacy = canonical_rows[row["task_id"]]
            if legacy["sha256"] != digest:
                raise ValueError("Canonical manifest does not match trajectory bytes")
            primary = next((a for a in aliases if a["path"] == legacy["source"]), None)
            if primary is None:
                f = source / legacy["source"]
                if sha256(f) != digest:
                    raise ValueError("Canonical source does not match")
                parts = Path(legacy["source"]).parts
                primary = {
                    "path": legacy["source"],
                    "run_id": parts[1],
                    "attempt_id": parts[-2],
                    "selected": True,
                }
                aliases.append(primary)
        instance = instances[row["task_id"]]
        specs.append(
            {
                "source": str(source / primary["path"]),
                "expected_sha256": digest,
                "task": {
                    "benchmark": "legacy-tactRefractor-swebench",
                    "revision": revision,
                    "instance_id": row["task_id"],
                    "repo": instance["repo"],
                    "base_commit": instance["base_commit"],
                    "partition": "train",
                    "exclusion_reason": None,
                    "instance": instance,
                },
                "run_id": primary["run_id"],
                "attempt_id": primary["attempt_id"],
                "purposes": ["training"],
                "model_fingerprint": "legacy-unverified-" + hash_json(row["model_identity"]),
                "legacy": row,
            }
        )
    # Exact trajectory identity for canonical evaluations; selected attempt + exact
    # evaluated patch for other runs. Never spread a result to other same-patch attempts.
    candidates = [("canonical-baseline", source / "baseline/harness/summary.json")]
    candidates += [
        (run, f)
        for run in expected
        if run != "canonical-baseline"
        for f in sorted((source / f"runs/{run}").glob("evaluation/*/summary.json"))
    ]
    evidence = []
    for run, f in candidates:
        summary = read_json(f)
        if summary.get("gold") is not False:
            continue
        predictions = f.parent / "predictions.json"
        if not predictions.exists():
            continue
        predictions_bytes = predictions.read_bytes()
        prediction_rows = json.loads(predictions_bytes)
        prediction_map = {r["instance_id"]: r["model_patch"] for r in prediction_rows}
        if len(prediction_map) != len(prediction_rows):
            raise ValueError("Duplicate prediction task")
        bindings = {}
        for iid, result in summary.get("outcomes", {}).items():
            digest = selected.get((run, iid))
            if not digest or iid not in prediction_map:
                continue
            patch = prediction_map[iid]
            if digest_text(patch) != groups[digest]["patch_sha256"]:
                continue
            if result.get("completed") and type(result.get("resolved")) is not bool:
                raise ValueError("Completed evaluation lacks a boolean outcome")
            bindings[digest] = {"task_id": iid, "patch": patch, "outcome_path": ["outcomes", iid]}
        if bindings:
            evidence.append(
                {
                    "source_run_id": run,
                    "source_report": str(f),
                    "report_sha256": sha256(f),
                    "predictions_sha256": hashlib.sha256(predictions_bytes).hexdigest(),
                    "predictions_source": str(predictions),
                    "bindings": bindings,
                    "finished_at": summary.get("finished_at", ""),
                    "binding_policy": "canonical_trajectory_sha256_and_patch"
                    if run == "canonical-baseline"
                    else "selected_attempt_in_same_run_and_patch",
                }
            )
    # Detect completed contradictory evidence rather than silently choosing success.
    completed = defaultdict(set)
    for e in evidence:
        summary = read_json(e["source_report"])
        for digest, b in e["bindings"].items():
            result = summary["outcomes"][b["task_id"]]
            if result.get("completed"):
                completed[digest].add(result["resolved"])
    if any(len(v) > 1 for v in completed.values()):
        raise ValueError("Conflicting completed results require manual investigation")
    out.mkdir(parents=True)
    shutil.copyfile(report, out / "source-rolloutReport.md")
    write_json(out / "legacy-config-identities.json", configs)
    plan = {
        "schema_version": 1,
        "created_at": now(),
        "migration_id": migration_id,
        "source_root": str(source),
        "report_sha256": sha256(report),
        "expected_files_by_run": expected,
        "actual_files_by_run": counts,
        "total_files": sum(counts.values()),
        "unique_trajectories": len(specs),
        "task_count": len(used),
        "scope": "all_current_attempts_in_report_runs",
        "partition": "train",
        "source_dataset_files": {p: x["sha256"] for p, x in snapshots.items()},
        "rollouts": specs,
        "evaluations": evidence,
    }
    # Freeze evidence now; later application reads only these copies.
    for i, e in enumerate(plan["evaluations"]):
        d = out / "evidence" / f"e-{i:03}"
        d.mkdir(parents=True)
        for name, src, expected_hash in [
            ("summary.json", e["source_report"], e["report_sha256"]),
            ("predictions.json", e["predictions_source"], e["predictions_sha256"]),
        ]:
            shutil.copyfile(src, d / name)
            if sha256(d / name) != expected_hash:
                raise ValueError("Evaluation source changed during snapshot")
        e["report"] = str((d / "summary.json").relative_to(root))
        e["predictions"] = str((d / "predictions.json").relative_to(root))
        for b in e["bindings"].values():
            patch = d / "patches" / (identifier(b["task_id"]) + ".diff")
            patch.parent.mkdir(exist_ok=True)
            patch.write_text(b.pop("patch"), encoding="utf-8")
            b["evaluated_patch"] = str(patch.relative_to(root))
    write_json(out / "plan.json", plan)
    return out / "plan.json"


def apply(root, plan_path):
    root, plan_path = Path(root).resolve(), Path(plan_path).resolve()
    plan = read_json(plan_path)
    out = plan_path.parent
    receipt = out / "receipt.json"
    with file_lock(root / "state/legacy-migration.lock"):
        catalog = Catalog(root)
        if receipt.exists():
            saved = read_json(receipt)
            if saved["plan_sha256"] != sha256(plan_path):
                raise ValueError("Migration plan changed")
            catalog.verify()
            return saved
        # Preflight every source/evidence before publishing any catalog records.
        for spec in plan["rollouts"]:
            if sha256(spec["source"]) != spec["expected_sha256"]:
                raise ValueError("Source trajectory changed after preparation")
        for e in plan["evaluations"]:
            if (
                sha256(root / e["report"]) != e["report_sha256"]
                or sha256(root / e["predictions"]) != e["predictions_sha256"]
            ):
                raise ValueError("Frozen evidence changed")
            predictions = {r["instance_id"]: r["model_patch"] for r in read_json(root / e["predictions"])}
            for digest, b in e["bindings"].items():
                if sha256(root / b["evaluated_patch"]) != digest_text(predictions[b["task_id"]]):
                    raise ValueError("Evaluated patch differs from frozen predictions")
        ids = {}
        for i, spec in enumerate(plan["rollouts"]):
            rid = import_rollout(root, spec)
            ids[spec["expected_sha256"]] = rid
            if i % 100 == 0:
                print(f"Imported {i + 1}/{len(plan['rollouts'])}", flush=True)
        with catalog.transaction() as data:
            for spec in plan["rollouts"]:
                row = data["rollouts"][ids[spec["expected_sha256"]]]
                row["provenance"].update(
                    {
                        "migration_id": plan["migration_id"],
                        "source_occurrences": spec["legacy"]["aliases"],
                        "legacy_model_identity": spec["legacy"]["model_identity"],
                        "legacy_config_identities": str(
                            (out / "legacy-config-identities.json").relative_to(root)
                        ),
                    }
                )
        # Infra errors first; completed evidence takes precedence without turning a
        # later retry failure into a negative label. Contradictions fail in prepare.
        evidence = sorted(plan["evaluations"], key=lambda e: (e["finished_at"], e["source_report"]))
        for phase in [False, True]:
            for i, e in enumerate(evidence):
                report = read_json(root / e["report"])
                bindings = {
                    ids[d]: {
                        "evaluated_patch": str(root / b["evaluated_patch"]),
                        "outcome_path": b["outcome_path"],
                    }
                    for d, b in e["bindings"].items()
                    if bool(report["outcomes"][b["task_id"]].get("completed")) == phase
                }
                if not bindings:
                    continue
                eid = f"{plan['migration_id']}-e{i:03}-{'graded' if phase else 'infra'}"
                current = catalog.read()
                if eid in current["evaluations"]:
                    continue
                import_evaluation(
                    root,
                    {
                        "evaluation_id": eid,
                        "report": str(root / e["report"]),
                        "report_sha256": e["report_sha256"],
                        "bindings": bindings,
                    },
                )
        current = catalog.read()
        fields = [
            "rollout_id",
            "task_id",
            "run_id",
            "attempt_id",
            "partition",
            "purpose",
            "status",
            "submission_status",
            "evaluation_status",
            "resolved",
            "selected_evaluation_id",
            "trajectory_path",
            "trajectory_sha256",
            "model_name",
        ]
        rows, occurrences = [], []
        for spec in plan["rollouts"]:
            rid = ids[spec["expected_sha256"]]
            row = current["rollouts"][rid]
            assert current["tasks"][row["task_key"]]["partition"] == "train" and row["purposes"] == [
                "training"
            ]
            exported = {
                "rollout_id": rid,
                "task_id": spec["task"]["instance_id"],
                "run_id": row["run_id"],
                "attempt_id": row["attempt_id"],
                "partition": "train",
                "purpose": "training",
                "status": outcome_status(row),
                "model_name": spec["legacy"]["model_identity"]["model_name"],
                **{k: row[k] for k in fields if k in row},
            }
            rows.append(exported)
            for alias in spec["legacy"]["aliases"]:
                occurrences.append({"rollout_id": rid, "task_id": exported["task_id"], **alias})
        for name, columns, data in [
            ("trajectory-index.csv", fields, rows),
            (
                "source-occurrences.csv",
                ["rollout_id", "task_id", "path", "run_id", "attempt_id", "selected"],
                occurrences,
            ),
        ]:
            with (out / name).open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=columns)
                w.writeheader()
                w.writerows(data)
        verified = catalog.verify()
        result = {
            "migration_id": plan["migration_id"],
            "completed_at": now(),
            "plan_sha256": sha256(plan_path),
            "report_sha256": plan["report_sha256"],
            "total_source_files": plan["total_files"],
            "unique_trajectories": len(ids),
            "tasks": plan["task_count"],
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "evaluation_counts": dict(Counter(r["evaluation_status"] for r in rows)),
            "graded_resolved_counts": dict(Counter(str(r["resolved"]) for r in rows)),
            "model_counts": dict(Counter(r["model_name"] for r in rows)),
            "catalog_verification": verified,
            "exports": {
                name: sha256(out / name) for name in ["trajectory-index.csv", "source-occurrences.csv"]
            },
        }
        write_json(receipt, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--migration-id", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.root, args.source, args.report, args.migration_id))
    else:
        print(json.dumps(apply(args.root, args.plan), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

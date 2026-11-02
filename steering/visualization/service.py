"""Formal offline analysis lifecycle; redraw only reads a frozen analysis bundle."""

from __future__ import annotations

import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from core.io import hash_json, read_json, sha256, write_json
from steering.artifacts import iter_jsonl, seal, verify_local, write_jsonl
from steering.datasets.cache import existing_cache, specification
from steering.datasets.features import select_sites
from steering.datasets.paths import locate, publication
from steering.visualization.lda import fit_site, load_transform, save_transform
from steering.visualization.manifold import METHODS, fit_embedding, load_descriptor
from steering.visualization.render import display_selection, render_site, write_report

POINT_FIELDS = (
    "ordinal",
    "uid",
    "source_run_id",
    "rollout_id",
    "trajectory_sha256",
    "task_key",
    "benchmark",
    "revision",
    "instance_id",
    "attempt_id",
    "repetition_id",
    "session_id",
    "decision_id",
    "branch_id",
    "sample_id",
    "agent_step",
    "message_index",
    "native_request_id",
    "selected",
    "retained",
    "action_hash",
    "consensus_action_hash",
    "label",
    "y",
    "model_fingerprint",
    "tokenizer_fingerprint",
    "fit_role",
)
CSV_FIELDS = (*POINT_FIELDS, "site_id", "LD1", "displayed", "display_jitter")


def coordinate_fields(method):
    if method == "lda":
        return ("LD1",)
    if method not in METHODS:
        raise ValueError("Unknown steering visualization method")
    return (f"{method.upper()}1", f"{method.upper()}2")


def fit_selection(rows, fit_scope, split_file=None):
    if fit_scope == "all":
        if split_file is not None:
            raise ValueError("A split file requires --fit-scope grouped")
        return (
            np.arange(len(rows)),
            {"fit_scope": "all", "groups": [], "split_source": None},
            ["all"] * len(rows),
        )
    if fit_scope != "grouped" or split_file is None:
        raise ValueError("Grouped fitting requires a complete task-group split file")
    entries = read_json(split_file)
    if isinstance(entries, dict):
        entries = entries.get("groups")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Split must contain a nonempty list of task groups")
    mapping = {}
    for group in entries:
        if not isinstance(group, dict) or set(group) - {
            "task_key",
            "benchmark",
            "revision",
            "instance_id",
            "split",
        }:
            raise ValueError("Split groups use task identity, not branch/sample assignments")
        if group.get("split") not in {"train", "validation"}:
            raise ValueError("Each task group needs split=train or split=validation")
        identity_keys = ("benchmark", "revision", "instance_id")
        identity = (
            {key: group[key] for key in identity_keys} if all(key in group for key in identity_keys) else None
        )
        task_key = group.get("task_key") or (hash_json(identity) if identity is not None else None)
        if task_key is None or (identity is not None and task_key != hash_json(identity)):
            raise ValueError("Split task_key/full task identity mismatch")
        if task_key in mapping:
            raise ValueError("Duplicate/leaking task group across splits")
        mapping[task_key] = group["split"]
    needed = {row["task_key"] for row in rows}
    if set(mapping) != needed:
        raise ValueError("Split must assign every selected benchmark/revision/task exactly once")
    roles = [mapping[row["task_key"]] for row in rows]
    if "train" not in roles or "validation" not in roles:
        raise ValueError("Grouped fitting needs nonempty train and validation task groups")
    indices = np.flatnonzero(np.array(roles) == "train")
    return (
        indices,
        {
            "fit_scope": "grouped",
            "groups": entries,
            "split_source": {"path": str(Path(split_file).resolve()), "sha256": sha256(split_file)},
        },
        roles,
    )


def _finalize(stage, analysis_id, analysis, selection, rows, site_results, *, formats, seed, maximum, facet):
    method = analysis["method"]
    axes = coordinate_fields(method)
    indices, displayed, jitter = display_selection(len(rows), seed, maximum)
    write_json(
        stage / "display-indices.json",
        {
            "seed": seed,
            "max_display_points": maximum,
            "indices": indices.tolist(),
            "statistics_use_all_samples": True,
        },
    )
    analysis.update(
        analysis_id=analysis_id,
        sample_count=len(rows),
        site_results=site_results,
        display={
            "seed": seed,
            "max_display_points": maximum,
            "facet": facet,
            "formats": list(formats),
            "indices_sha256": hash_json(indices.tolist()),
            "jitter": "uniform[-0.12,0.12]" if method == "lda" else "none",
        },
        sample_order_sha256=hash_json([row["uid"] for row in rows]),
    )
    selection["fit_uids"] = [row["uid"] for row in rows if row["fit_role"] in {"all", "train"}]
    write_json(stage / "selection.json", selection)
    write_jsonl(stage / "points-metadata.jsonl", rows)
    summary, plots = {"sample_count": len(rows), "sites": {}, "groups": {}}, []
    for field in ("source_run_id", "task_key", "decision_id", "retained"):
        grouped = defaultdict(Counter)
        for row in rows:
            key = (
                str(row[field])
                if field != "decision_id"
                else "/".join((row["source_run_id"], row["rollout_id"], row[field]))
            )
            grouped[key][row["label"]] += 1
        summary["groups"][field] = {key: dict(value) for key, value in sorted(grouped.items())}
    with (stage / "points.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = CSV_FIELDS if method == "lda" else (*POINT_FIELDS, "site_id", *axes, "displayed")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        labels = np.array([row["y"] for row in rows])
        for site, result in site_results.items():
            stats = {"status": result["status"], "class_counts": [int((labels == y).sum()) for y in (0, 1)]}
            if result["status"] != "available":
                stats["reason"] = result.get("reason")
                summary["sites"][site] = stats
                continue
            coordinates = np.load(stage / "projections" / f"site-{site}.npy", allow_pickle=False)
            if coordinates.shape != (len(rows), len(axes)) or not np.isfinite(coordinates).all():
                raise ValueError("Frozen analysis projection shape/finite check failed")
            stats["all_class_means"] = [
                (
                    float(coordinates[labels == y, 0].mean())
                    if method == "lda"
                    else coordinates[labels == y].mean(axis=0).tolist()
                )
                if (labels == y).any()
                else None
                for y in (0, 1)
            ]
            stats["fit"] = result["metadata"]
            summary["sites"][site] = stats
            for i, row in enumerate(rows):
                writer.writerow(
                    {
                        **row,
                        "site_id": site,
                        **{axis: float(coordinates[i, j]) for j, axis in enumerate(axes)},
                        "displayed": bool(displayed[i]),
                        **({"display_jitter": float(jitter[i])} if method == "lda" else {}),
                    }
                )
            plots += render_site(
                stage / "plots",
                site,
                coordinates,
                rows,
                formats=formats,
                displayed=displayed,
                jitter=jitter,
                facet=facet,
                method=method,
            )
    available = sum(result["status"] == "available" for result in site_results.values())
    analysis["status"] = (
        "complete" if available == len(site_results) else "partial" if available else "unavailable"
    )
    analysis["available_site_count"] = available
    write_json(stage / "analysis.json", analysis)
    write_json(stage / "summary.json", summary)
    write_report(stage, analysis, summary, plots)
    seal(stage, "steering-analysis", analysis_id, hash_json(analysis))
    return analysis["status"]


def visualize(
    cfg,
    analysis_id,
    *,
    dataset_id,
    dataset_root=None,
    cache_root=None,
    storage_root=None,
    sites,
    fit_scope="all",
    split_file=None,
    seed=0,
    formats=("png", "svg"),
    max_display_points=None,
    facet=None,
    tol=1e-4,
    memory_budget_bytes=32 * 1024**3,
    method="lda",
    embedding_options=None,
):
    coordinate_fields(method)
    if method in METHODS and (fit_scope != "all" or split_file is not None):
        raise ValueError(
            "t-SNE/UMAP are full-cohort exploratory embeddings; require --fit-scope all without --split"
        )
    if method == "lda" and embedding_options:
        raise ValueError("Embedding parameters require --method tsne or umap")
    dataset_directory = locate(cfg, "steering_dataset", dataset_id, dataset_root)
    manifest = read_json(dataset_directory / "dataset.json")
    selected_sites = select_sites(manifest["sites"], sites)
    if not sites or not selected_sites:
        raise ValueError("Visualization requires explicit native site IDs")
    cache = None
    for cache_sites in (None, sites):
        identity, spec = specification(dataset_directory, cache_sites)
        cache = existing_cache(cfg, identity, spec, cache_root)
        if cache is not None:
            break
    if cache is None:
        raise ValueError(
            "No verified feature cache for the exact dataset/site/order; run steering-dataset cache"
        )
    raw_rows = list(iter_jsonl(dataset_directory / "samples.jsonl"))
    fit_indices, selection, roles = fit_selection(raw_rows, fit_scope, split_file)
    rows = [
        {**{key: row.get(key) for key in POINT_FIELDS}, "fit_role": role}
        for row, role in zip(raw_rows, roles)
    ]
    labels = np.array([row["y"] for row in rows], dtype=np.int64)
    analysis = {
        "schema_version": 1,
        "schema": "binary-lda-ld1-v1" if method == "lda" else "unsupervised-embedding-2d-v1",
        "method": method,
        "fit_scope": fit_scope,
        "dataset_id": dataset_id,
        "dataset_receipt_sha256": sha256(dataset_directory / "receipt.json"),
        "cache_identity": identity,
        "space": manifest["space"],
        "sites": selected_sites,
        "interpretation": (
            "exploratory supervised visualization; not correctness, generalization or steering payload"
            if method == "lda"
            else "exploratory unsupervised full-cohort embedding; labels used only for plotting; "
            "not correctness, generalization or steering payload"
        ),
    }
    results = {}
    with publication(cfg, "steering_analysis", analysis_id, storage_root) as (stage, destination):
        (stage / "transform").mkdir()
        (stage / "projections").mkdir()
        if split_file is not None:
            write_json(stage / "split-groups.json", read_json(split_file))
        for site in selected_sites:
            site_id = site["id"]
            print(
                f"{method}: fitting native site {site_id}, rows={len(rows)}, train={len(fit_indices)}",
                flush=True,
            )
            values = np.load(cache / f"site-{site_id}.f32.npy", mmap_mode="r", allow_pickle=False)
            try:
                if method == "lda":
                    result = fit_site(
                        values,
                        labels,
                        fit_indices=fit_indices,
                        tol=tol,
                        memory_budget_bytes=memory_budget_bytes,
                    )
                else:
                    result = fit_embedding(
                        values,
                        method=method,
                        seed=seed,
                        memory_budget_bytes=memory_budget_bytes,
                        **(embedding_options or {}),
                    )
            finally:
                values._mmap.close()
            if result["status"] != "available":
                results[site_id] = result
                continue
            transform_path = stage / "transform" / f"site-{site_id}.npz"
            projection_path = stage / "projections" / f"site-{site_id}.npy"
            save_transform(transform_path, result["transform"])
            np.save(projection_path, result["projection"], allow_pickle=False)
            results[site_id] = {
                "status": "available",
                "metadata": result["metadata"],
                "transform_sha256": sha256(transform_path),
                "projection_sha256": sha256(projection_path),
            }
        status = _finalize(
            stage,
            analysis_id,
            analysis,
            selection,
            rows,
            results,
            formats=formats,
            seed=seed,
            maximum=max_display_points,
            facet=facet,
        )
    return {
        "analysis_id": analysis_id,
        "path": str(destination),
        "status": status,
        "sample_count": len(rows),
        "site_status": {site: result["status"] for site, result in results.items()},
    }


def redraw(
    cfg,
    analysis_id,
    *,
    source_analysis_id,
    analysis_root=None,
    storage_root=None,
    sites=None,
    seed=0,
    formats=("png", "svg"),
    max_display_points=None,
    facet=None,
    method=None,
):
    source = locate(cfg, "steering_analysis", source_analysis_id, analysis_root)
    check_analysis(source)
    original = read_json(source / "analysis.json")
    if method is not None and method != original["method"]:
        raise ValueError("Redraw method must match the frozen source analysis; redraw never refits")
    selected = select_sites(original["sites"], sites)
    rows = list(iter_jsonl(source / "points-metadata.jsonl"))
    selection = read_json(source / "selection.json")
    parent_sha = sha256(source / "receipt.json")
    analysis = {
        key: value
        for key, value in original.items()
        if key not in {"analysis_id", "display", "site_results", "status", "available_site_count"}
    }
    analysis.update(
        sites=selected,
        source_analysis={"analysis_id": source_analysis_id, "receipt_sha256": parent_sha},
        operation="redraw_frozen_coordinates_without_refit",
    )
    results = {site["id"]: original["site_results"][site["id"]] for site in selected}
    with publication(cfg, "steering_analysis", analysis_id, storage_root) as (stage, destination):
        (stage / "transform").mkdir()
        (stage / "projections").mkdir()
        for site_id, result in results.items():
            if result["status"] != "available":
                continue
            for relative in (f"transform/site-{site_id}.npz", f"projections/site-{site_id}.npy"):
                shutil.copyfile(source / relative, stage / relative)
        status = _finalize(
            stage,
            analysis_id,
            analysis,
            selection,
            rows,
            results,
            formats=formats,
            seed=seed,
            maximum=max_display_points,
            facet=facet,
        )
        if sha256(source / "receipt.json") != parent_sha:
            raise ValueError("Parent analysis changed during redraw")
    return {
        "analysis_id": analysis_id,
        "path": str(destination),
        "status": status,
        "sample_count": len(rows),
        "operation": "redraw_without_refit",
    }


def check_analysis(directory):
    directory = Path(directory)
    receipt = verify_local(directory, "steering-analysis", directory.name)
    analysis = read_json(directory / "analysis.json")
    axes = coordinate_fields(analysis["method"])
    if receipt["identity"] != hash_json(analysis) or analysis["analysis_id"] != directory.name:
        raise ValueError("Analysis manifest/receipt identity mismatch")
    rows = list(iter_jsonl(directory / "points-metadata.jsonl"))
    if (
        len(rows) != analysis["sample_count"]
        or hash_json([row["uid"] for row in rows]) != analysis["sample_order_sha256"]
    ):
        raise ValueError("Analysis frozen sample order mismatch")
    coordinates = {}
    for site_id, result in analysis["site_results"].items():
        if result["status"] != "available":
            continue
        projection = directory / "projections" / f"site-{site_id}.npy"
        transform = directory / "transform" / f"site-{site_id}.npz"
        if (
            sha256(projection) != result["projection_sha256"]
            or sha256(transform) != result["transform_sha256"]
        ):
            raise ValueError("Analysis coordinate/transform checksum mismatch")
        frozen = (
            load_transform(transform)
            if analysis["method"] == "lda"
            else load_descriptor(transform, analysis["method"])
        )
        matrix = np.load(projection, allow_pickle=False)
        if (
            matrix.shape != (len(rows), len(axes))
            or not np.isfinite(matrix).all()
            or int(frozen["n_components"]) != len(axes)
        ):
            raise ValueError("Analysis must contain finite method-specific coordinates per frozen sample")
        coordinates[site_id] = matrix
    seen = Counter()
    with (directory / "points.csv").open(encoding="utf-8", newline="") as stream:
        for point in csv.DictReader(stream):
            site_id = point["site_id"]
            i = seen[site_id]
            if (
                site_id not in coordinates
                or i >= len(rows)
                or point["uid"] != rows[i]["uid"]
                or int(point["ordinal"]) != i
                or any(float(point[axis]) != coordinates[site_id][i, j] for j, axis in enumerate(axes))
            ):
                raise ValueError("CSV coordinates do not match frozen projections/order")
            seen[site_id] += 1
    if any(seen[site] != len(rows) for site in coordinates):
        raise ValueError("Analysis CSV is missing frozen coordinates")
    return {
        "status": "verified",
        "analysis_id": directory.name,
        "sample_count": len(rows),
        "available_sites": list(coordinates),
        "receipt_sha256": sha256(directory / "receipt.json"),
    }

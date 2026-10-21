"""Select and publish a reproducible non-Verified SWE-bench cohort to NAS.

The command intentionally owns the whole selection and image publication contract:
HF source rows, public agent evidence, OCI descriptor metadata, content-addressed
blobs, per-task image records, a dataset manifest, and catalog registration are
written as one resumable cohort. It does not start Docker or rollout tasks.
"""

from __future__ import annotations

import collections
import concurrent.futures
import copy
import hashlib
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

from catalog.store import Catalog
from core.io import hash_json, identifier, now, read_json, sha256, write_json
from core.locking import file_lock
from runtime.images.oci import descriptor_graph, platform_manifest, verify_blobs
from runtime.images.registry import image_metadata, transfer_blob

HF_DATASET = "SWE-bench/SWE-bench"
HF_REVISION = "c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2"
VERIFIED_REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"
RESULTS_REVISION = "40f164d5b8f1d249bf95a6df8b74b577fd8e519d"
RESULT_EXPERIMENTS = (
    "20250227_sweagent-claude-3-7-20250219",
    "20250522_amazon-q-developer-agent-20250405-dev",
    "20250605_atlassian-rovo-dev",
    "20251027_salesforce_SAGE",
    "20251219_sonar-foundation-agent_claude-opus-4-5",
    "20241103_OpenHands-CodeAct-2.1-sonnet-20241022",
)
DIFFICULTY_BUCKETS = {"easy": {5, 6}, "medium": {3, 4}, "hard": {1, 2}}


def _json_text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _source_rows(revision, dataset=HF_DATASET):
    """Use revision-addressed HF parquet cache; never silently use another revision."""
    import pyarrow.parquet as pq

    cache = Path.home() / ".cache/huggingface/hub"
    snapshot = cache / ("datasets--" + dataset.replace("/", "--")) / "snapshots" / revision
    files = sorted((snapshot / "data").glob("test-*.parquet"))
    if not files:
        from huggingface_hub import hf_hub_download, list_repo_files

        files = [Path(hf_hub_download(dataset, file, repo_type="dataset", revision=revision))
                 for file in list_repo_files(dataset, repo_type="dataset", revision=revision)
                 if file.startswith("data/test-") and file.endswith(".parquet")]
    if not files:
        raise ValueError("No test parquet files in pinned dataset revision")
    return [row for path in files for row in pq.read_table(path).to_pylist()]


def _public_results(revision, proxy):
    import requests

    session = requests.Session()
    session.trust_env = False
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    out = {}
    try:
        for experiment in RESULT_EXPERIMENTS:
            url = (
                "https://raw.githubusercontent.com/SWE-bench/experiments/"
                f"{revision}/evaluation/test/{experiment}/results/results.json"
            )
            response = session.get(url, timeout=(20, 90))
            response.raise_for_status()
            result = response.json()
            out[experiment] = {
                "resolved": sorted(
                    set(result.get("resolved", []))
                    - set(result.get("no_generation", []))
                    - set(result.get("no_logs", []))
                ),
                "no_generation": sorted(set(result.get("no_generation", []))),
                "no_logs": sorted(set(result.get("no_logs", []))),
            }
    finally:
        session.close()
    return out


def _balanced_selection(rows, existing_ids, results, count, verified_ids=()):
    if len(results) != 6:
        raise ValueError("Difficulty requires exactly six result sets")
    blocked = set(existing_ids) | set(verified_ids)
    resolved = {name: set(value["resolved"]) for name, value in results.items()}
    no_output = {
        instance_id
        for value in results.values()
        for instance_id in value["no_generation"] + value["no_logs"]
    }
    candidates = []
    for row in rows:
        instance_id = row["instance_id"]
        if instance_id in blocked or instance_id in no_output:
            continue
        if not row.get("patch") or not row.get("test_patch"):
            continue
        if not row.get("FAIL_TO_PASS") or not row.get("PASS_TO_PASS"):
            continue
        score = sum(instance_id in values for values in resolved.values())
        bucket = next((name for name, values in DIFFICULTY_BUCKETS.items() if score in values), None)
        if bucket is None:
            continue
        candidates.append((bucket, row, score))

    by_bucket_repo = collections.defaultdict(lambda: collections.defaultdict(list))
    for bucket, row, score in candidates:
        by_bucket_repo[bucket][row["repo"]].append((row, score))
    policy = "dycon-nonverified600-v1"
    selected = []
    bucket_counts = {bucket: count // 3 for bucket in DIFFICULTY_BUCKETS}
    if sum(bucket_counts.values()) != count:
        raise ValueError("count must be divisible by three")
    for bucket, quota in bucket_counts.items():
        repos = sorted(by_bucket_repo[bucket])
        if len(repos) < 3:
            raise ValueError(f"Not enough repository diversity for {bucket}: {repos}")
        for repo in repos:
            by_bucket_repo[bucket][repo].sort(
                key=lambda item: hashlib.sha256(
                    f"{policy}:{bucket}:{item[0]['instance_id']}".encode()
                ).hexdigest()
            )
        counts = collections.Counter()
        chosen = []
        while len(chosen) < quota:
            available = [repo for repo in repos if by_bucket_repo[bucket][repo]]
            if not available:
                raise ValueError(f"Only {len(chosen)} candidates available for {bucket}")
            repo = min(available, key=lambda name: (counts[name], name))
            row, score = by_bucket_repo[bucket][repo].pop(0)
            counts[repo] += 1
            chosen.append((row, score))
        for row, score in chosen:
            selected.append(
                {
                    "row": row,
                    "bucket": bucket,
                    "resolution_score": score,
                    "resolution_evidence": {
                        name: row["instance_id"] in values for name, values in resolved.items()
                    },
                }
            )
    if len(selected) != count or len({x["row"]["instance_id"] for x in selected}) != count:
        raise AssertionError("Selection is not unique")
    return selected, {
        "schema_version": 1,
        "policy": policy,
        "requested_count": count,
        "selected_count": len(selected),
        "bucket_counts": dict(collections.Counter(x["bucket"] for x in selected)),
        "repo_counts": dict(collections.Counter(x["row"]["repo"] for x in selected)),
        "source_dataset": HF_DATASET,
        "source_revision": HF_REVISION,
        "results_revision": RESULTS_REVISION,
        "result_experiments": list(RESULT_EXPERIMENTS),
        "difficulty_definition": (
            "easy=resolved by 5-6 of 6 public systems; medium=3-4; "
            "hard=1-2. Excludes all six-system failures and missing-output rows."
        ),
        "quality_filter": "nonempty gold patch/test patch/FAIL_TO_PASS/PASS_TO_PASS; no existing or Verified task",
        "instance_ids": [x["row"]["instance_id"] for x in selected],
    }


def _repository(image):
    if "@" in image:
        repository, reference = image.rsplit("@", 1)
    else:
        repository, reference = image.rsplit(":", 1)
    if not re.fullmatch(r"[a-z0-9_.-]+/[a-z0-9_.-]+", repository):
        raise ValueError(f"Unsupported image repository: {image}")
    return repository, reference


def _task_record(item, image_record, store_name, source_revision, selection_hash):
    row = copy.deepcopy(item["row"])
    instance_id = row["instance_id"]
    row["FAIL_TO_PASS"] = _json_text(row.get("FAIL_TO_PASS", []))
    row["PASS_TO_PASS"] = _json_text(row.get("PASS_TO_PASS", []))
    original_image = row["image"]
    row["image_name"] = original_image
    row["nas_image"] = {
        "store": store_name,
        "manifest": f"images/{instance_id}.json",
        "manifest_sha256": image_record["manifest_sha256"],
        "image_id": image_record["image_id"],
    }
    row["environment_overrides"] = row.get("environment_overrides") or {}
    row["dycon_selection"] = {
        "difficulty_bucket": item["bucket"],
        "difficulty_definition": "six-public-system resolution count",
        "resolution_score": item["resolution_score"],
        "resolution_evidence": item["resolution_evidence"],
        "selection_policy": "dycon-nonverified600-v1",
    }
    return {
        "benchmark": "SWE-bench_NonVerified600",
        "revision": f"hf-{source_revision}-selection-{selection_hash[:16]}",
        "instance_id": instance_id,
        "repo": row["repo"],
        "base_commit": row["base_commit"],
        "partition": "train",
        "exclusion_reason": None,
        "instance": row,
    }


def _write_image_record(path, root, source_image, repository, root_descriptor, descriptors):
    runtime, manifest = platform_manifest(root, root_descriptor, "linux/amd64")
    config = manifest["config"]
    layers = manifest["layers"]
    record = {
        "schema_version": 2,
        "requested": source_image,
        "repository": repository,
        "manifest_digest": root_descriptor["digest"],
        "manifest_size": root_descriptor["size"],
        "root_media_type": root_descriptor["mediaType"],
        "runtime_manifest_digest": runtime["digest"],
        "config": config,
        "layers": layers,
        "compressed_bytes": sum(item["size"] for item in layers),
        "repo_digests": [repository + "@" + root_descriptor["digest"]],
        "platform": "linux/amd64",
        "descriptor_count": len(descriptors),
        "deep_verified": False,
    }
    write_json(path, record)
    return record


def publish(cfg, cohort_id, *, count=600, proxy="http://127.0.0.1:7890", workers=8,
            source_revision=HF_REVISION, results_revision=RESULTS_REVISION):
    identifier(cohort_id)
    if count % 3:
        raise ValueError("count must be divisible by three")
    root = Path(cfg["project_root"]).resolve()
    nas = Path(cfg["paths"]["nas_root"]).resolve()
    if not any(path != Path("/") and os.path.ismount(path) for path in [nas, *nas.parents]):
        raise RuntimeError(f"NAS namespace is not mounted: {nas}")
    if count <= 0 or not 1 <= workers <= 64:
        raise ValueError("Positive count and 1..64 workers required")
    store = nas / cohort_id
    store.mkdir(parents=True, exist_ok=True)
    blobs = store / "blobs/sha256"
    images = store / "images"
    dataset = store / "dataset"
    for path in (blobs, images, dataset):
        path.mkdir(parents=True, exist_ok=True)
    with file_lock(root / "state" / (cohort_id + ".benchmark-pull.lock")):
        if (dataset / "registration.json").exists():
            return verify_cohort(cfg, cohort_id, workers=min(workers, 4))
        catalog = Catalog(root).read()
        existing_ids = {task["instance_id"] for task in catalog["tasks"].values()}
        selection_path = dataset / "selection.json"
        if selection_path.exists():
            selection = read_json(selection_path)
            if selection.get("requested_count") != count or selection.get("source_revision") != source_revision:
                raise ValueError("Existing NAS cohort selection does not match requested parameters")
            if selection.get("results_revision") != results_revision:
                raise ValueError("Existing cohort uses different public results")
            source_rows = _source_rows(source_revision)
            result_rows = read_json(dataset / "source-results.json")
            source_by_id = {row["instance_id"]: row for row in source_rows}
            selected = [
                {
                    "row": source_by_id[instance_id],
                    "bucket": selection["bucket_by_id"][instance_id],
                    "resolution_score": sum(
                        instance_id in set(data["resolved"]) for data in result_rows.values()
                    ),
                    "resolution_evidence": {
                        name: instance_id in set(data["resolved"])
                        for name, data in result_rows.items()
                    },
                }
                for instance_id in selection["instance_ids"]
            ] if selection.get("bucket_by_id") else []
            if not selected:
                raise ValueError("Existing selection lacks resumable bucket_by_id evidence")
            selection_hash = sha256(selection_path)
        else:
            source_rows = _source_rows(source_revision)
            result_rows = _public_results(results_revision, proxy)
            verified_rows = _source_rows(VERIFIED_REVISION, "SWE-bench/SWE-bench_Verified")
            if len(verified_rows) != 500:
                raise ValueError("Expected the complete Verified-500 exclusion list")
            verified_ids = {row["instance_id"] for row in verified_rows}
            selected, summary = _balanced_selection(
                source_rows, existing_ids, result_rows, count, verified_ids
            )
            summary["created_at"] = now()
            summary["source_revision"] = source_revision
            summary["results_revision"] = results_revision
            summary["verified_revision"] = VERIFIED_REVISION
            summary["excluded_verified_ids"] = sorted(verified_ids)
            summary["excluded_existing_ids"] = sorted(existing_ids)
            summary["overlap_verified"] = sorted(verified_ids & {x["row"]["instance_id"] for x in selected})
            summary["source_row_count"] = len(source_rows)
            summary["result_summary"] = {
                name: {key: len(value) for key, value in result.items()}
                for name, result in result_rows.items()
            }
            summary["bucket_by_id"] = {x["row"]["instance_id"]: x["bucket"] for x in selected}
            selection = summary
            write_json(selection_path, summary)
            write_json(dataset / "source-results.json", result_rows)
            write_json(dataset / "source-rows-sha256.json", {
                "source_revision": source_revision,
                "row_count": len(source_rows),
                "rows_sha256": hash_json(source_rows),
            })
            selection_hash = sha256(selection_path)
        selected_rows_path = dataset / "source-selected.json"
        frozen_rows = [item["row"] for item in selected]
        if selected_rows_path.exists():
            if hash_json(read_json(selected_rows_path)) != hash_json(frozen_rows):
                raise ValueError("Resumed source rows differ from frozen selection")
        else:
            write_json(selected_rows_path, frozen_rows)
        if existing_ids.intersection(selection["instance_ids"]):
            raise ValueError("Selected benchmark overlaps current catalog")

        metadata_path = dataset / "image-metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        def prepare_metadata(item):
            instance_id = item["row"]["instance_id"]
            repository, reference = _repository(item["row"]["image"])
            root_descriptor, descriptors = image_metadata(
                repository, blobs, reference=reference, proxy=proxy
            )
            image_record = _write_image_record(
                images / f"{instance_id}.json",
                blobs,
                item["row"]["image"],
                repository,
                root_descriptor,
                descriptors,
            )
            return instance_id, {
                "source_image": item["row"]["image"],
                "repository": repository,
                "reference": reference,
                "record": f"images/{instance_id}.json",
                "record_sha256": sha256(images / f"{instance_id}.json"),
                "manifest_digest": image_record["manifest_digest"],
                "manifest_size": image_record["manifest_size"],
                "descriptors": descriptors,
            }
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pending_metadata = {
                pool.submit(prepare_metadata, item): item["row"]["instance_id"]
                for item in selected if item["row"]["instance_id"] not in metadata
            }
            for future in concurrent.futures.as_completed(pending_metadata):
                instance_id, data = future.result()
                metadata[instance_id] = data
                if len(metadata) % 10 == 0:
                    write_json(metadata_path, metadata)
                    print(f"METADATA {len(metadata)}/{len(selected)}", flush=True)
        write_json(metadata_path, metadata)

        all_descriptors = {}
        for value in metadata.values():
            for item in value["descriptors"]:
                if item["digest"] in all_descriptors and all_descriptors[item["digest"]]["size"] != item["size"]:
                    raise ValueError("Conflicting OCI descriptor sizes across images")
                all_descriptors[item["digest"]] = item
        seed = [
            path / "blobs/sha256"
            for path in nas.iterdir()
            if path.is_dir() and path != store and (path / "blobs/sha256").is_dir()
        ]
        plan = {
            "schema_version": 1,
            "cohort_id": cohort_id,
            "images": len(metadata),
            "unique_descriptors": len(all_descriptors),
            "unique_bytes": sum(item["size"] for item in all_descriptors.values()),
            "source_revision": source_revision,
            "results_revision": results_revision,
            "descriptors": sorted(all_descriptors.values(), key=lambda item: item["digest"]),
        }
        write_json(dataset / "download-plan.json", plan)
        missing_bytes = sum(item["size"] for item in plan["descriptors"]
                            if not (blobs / item["digest"][7:]).is_file())
        if shutil.disk_usage(store).free < missing_bytes + 100 * 1024**3:
            raise BufferError("Insufficient NAS space for cohort plus 100 GiB reserve")
        status_path = dataset / "download-status.json"
        status = read_json(status_path) if status_path.exists() else {
            "schema_version": 1, "complete": False, "completed": [], "started_at": now()
        }
        completed = set(status.get("completed", []))
        progress_lock = threading.Lock()
        last_report = time.monotonic()
        received_bytes = 0

        def progress(size):
            nonlocal last_report, received_bytes
            with progress_lock:
                received_bytes += size
                if time.monotonic() - last_report >= 15:
                    write_json(dataset / "transfer-progress.json", {
                        "received_bytes_this_process": received_bytes, "updated_at": now(),
                        "pid": os.getpid(),
                    })
                    last_report = time.monotonic()

        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(transfer_blob, item, blobs, seed, proxy, progress): item
                for item in plan["descriptors"] if item["digest"] not in completed
            }
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    errors.append({"digest": item["digest"], "error": repr(exc)})
                    write_json(dataset / "transfer-errors.json", errors)
                    continue
                completed.add(item["digest"])
                status.update(completed_count=len(completed), total=len(plan["descriptors"]), last=result,
                              updated_at=now())
                if len(completed) % 25 == 0 or len(completed) == len(plan["descriptors"]):
                    status["completed"] = sorted(completed)
                    write_json(status_path, status)
                    print(f"BLOBS {len(completed)}/{len(plan['descriptors'])}", flush=True)
        if completed != {item["digest"] for item in plan["descriptors"]}:
            raise RuntimeError(f"Not all OCI descriptors downloaded: {errors}")
        status.update(completed=sorted(completed), total=len(completed), complete=True, finished_at=now())
        write_json(status_path, status)

        tasks = []
        for item in selected:
            iid = item["row"]["instance_id"]
            record_path = images / f"{iid}.json"
            record = read_json(record_path)
            record["deep_verified"] = True
            write_json(record_path, record)
            tasks.append(_task_record(item, {
                "manifest_sha256": sha256(record_path),
                "image_id": record["manifest_digest"],
            }, cohort_id, source_revision, selection_hash))
        tasks_path = dataset / "tasks.json"
        write_json(tasks_path, tasks)
        write_json(dataset / "dataset.json", {
            "schema_version": 1,
            "benchmark": "SWE-bench_NonVerified600",
            "cohort_id": cohort_id,
            "selection_sha256": sha256(selection_path),
            "tasks_sha256": sha256(tasks_path),
            "source_results_sha256": sha256(dataset / "source-results.json"),
            "download_plan_sha256": sha256(dataset / "download-plan.json"),
            "created_at": now(),
        })
        # Registration is the final stage: all source and image evidence exists first.
        current_ids = {task["instance_id"] for task in Catalog(root).read()["tasks"].values()}
        if current_ids.intersection(selection["instance_ids"]):
            raise ValueError("Catalog changed while publishing; refusing partial registration")
        keys = Catalog(root).register_tasks(tasks)
        write_json(dataset / "registration.json", {
            "registered_at": now(), "task_count": len(keys), "task_keys": keys,
            "catalog_revision": Catalog(root).read()["catalog_revision"],
        })
        return {
            "cohort_id": cohort_id,
            "store": str(store),
            "task_count": len(tasks),
            "registered": len(keys),
            "difficulty_counts": dict(collections.Counter(item["bucket"] for item in selected)),
            "repo_counts": dict(collections.Counter(item["row"]["repo"] for item in selected)),
            "unique_descriptors": len(plan["descriptors"]),
            "unique_descriptor_bytes": plan["unique_bytes"],
        }


def verify_cohort(cfg, cohort_id, workers=4):
    """Verify all published bytes and task evidence; never rerun model or Docker."""
    from runtime.images.nas import NasImageStore

    store = Path(cfg["paths"]["nas_root"]).resolve() / identifier(cohort_id)
    dataset = store / "dataset"
    manifest = read_json(dataset / "dataset.json")
    for field, name in [
        ("selection_sha256", "selection.json"), ("tasks_sha256", "tasks.json"),
        ("source_results_sha256", "source-results.json"), ("download_plan_sha256", "download-plan.json"),
    ]:
        if sha256(dataset / name) != manifest[field]:
            raise ValueError(f"Published cohort metadata changed: {name}")
    selection = read_json(dataset / "selection.json")
    tasks = read_json(dataset / "tasks.json")
    ids = {task["instance_id"] for task in tasks}
    if len(ids) != len(tasks) or ids != set(selection["instance_ids"]):
        raise ValueError("Cohort task membership mismatch")
    if ids & set(selection["excluded_verified_ids"]) or ids & set(selection["excluded_existing_ids"]):
        raise ValueError("Cohort overlaps Verified/existing exclusion sets")
    buckets = collections.Counter(task["instance"]["dycon_selection"]["difficulty_bucket"] for task in tasks)
    if dict(buckets) != selection["bucket_counts"]:
        raise ValueError("Difficulty balance changed")
    descriptors = []
    for task in tasks:
        image = NasImageStore(cfg, task["instance"]["nas_image"])
        descriptors.extend(descriptor_graph(image.root / "blobs/sha256", {
            "digest": image.image_id, "size": image.record["manifest_size"],
            "mediaType": read_json(image.blob(image.image_id))["mediaType"],
        }))
    verified = verify_blobs(store / "blobs/sha256", descriptors, workers=workers)
    catalog = Catalog(cfg["project_root"]).read()
    for task in tasks:
        key = hash_json({key: task[key] for key in ["benchmark", "revision", "instance_id"]})
        if catalog["tasks"].get(key) != task:
            raise ValueError("Published task does not match catalog")
    report = {
        "schema_version": 1, "cohort_id": cohort_id, "store": str(store),
        "verified_at": now(), "tasks": len(tasks), "verified_blobs": len(verified),
        "difficulty_counts": dict(buckets), "verified_overlap": [], "existing_overlap": [],
        "local_gold_validated": False, "deep_verified": True,
    }
    write_json(dataset / "verification.json", report)
    return report

"""Create task-bound label views referencing packed online hidden rows."""

import math
from pathlib import Path

from application.execution.evidence import RunEvidence
from core.io import artifact_dir, artifact_reference, read_json, sha256, write_json
from core.locking import exclusive_artifact
from difficulty.datasets.selection import select_run
from enrichment.online_trace import publication_directory, verify_publication


@exclusive_artifact("difficulty-dataset", "index_id")
def build_published_index(cfg, run_id, index_id, rollout_ids=(), rollout_list=None, publication_id=None, storage_root=None):
    """Reference published shards directly; no implicit packed tensor copy."""
    from core.config import load_snapshot

    if storage_root is not None:
        cfg = {**cfg, 'paths': {**cfg.get('paths', {}), 'artifact_roots': {
            **cfg.get('paths', {}).get('artifact_roots', {}),
            'difficulty_dataset': str(storage_root)}}}

    frozen = load_snapshot(cfg["project_root"], run_id)
    trace = publication_directory(cfg["project_root"], run_id, publication_id)
    native_root = Path(frozen['paths']['nas_root']) / frozen['capture']['export']['root'] / run_id
    initial = read_json(trace / 'receipt.json')
    receipt = verify_publication(trace, Path(initial['source_root']) if initial.get('origin') == 'migrated' else native_root)
    catalog = RunEvidence(cfg["project_root"], run_id).read()
    selected = select_run(cfg["project_root"], run_id, rollout_ids, rollout_list)
    wanted = set(selected["rollout_ids"])
    samples = []
    requests = read_json(trace / "requests.json")
    for event in read_json(trace / "index.json"):
        if event["rollout_id"] not in catalog["rollouts"]:
            raise ValueError("Published event belongs to another run")
        if event["rollout_id"] not in wanted:
            continue
        rollout = catalog["rollouts"][event["rollout_id"]]
        if rollout["trajectory_sha256"] != event["trajectory_sha256"] or rollout["provenance"].get(
            "quarantined"
        ):
            raise ValueError("Published event does not bind an eligible canonical trajectory")
        action = requests[event['request_id']].get('is_action_response', True)
        exact = event.get('alignment_status') == 'exact'
        eligible = rollout["resolved"] is True and not event["censored"] and action and exact
        samples.append(
            {
                **event,
                "partition": catalog["tasks"][rollout["task_key"]]["partition"],
                "labels": {
                    "remaining_steps": event["remaining_steps"] if eligible else None,
                    "remaining_reasoning_tokens": event["remaining_reasoning_tokens_after"]
                    if eligible
                    else None,
                },
                "selected_evaluation_id": rollout["selected_evaluation_id"],
                "label_status": "non_action_response" if not action else "non_exact_alignment" if not exact else
                                "available" if eligible else "censored" if event["censored"] else "evaluation_ineligible",
                "is_action_response": action,
                "hidden_status": "native_exact" if event.get('alignment_status') == 'exact' else "non_exact_alignment",
            }
        )
    source = Path(frozen["paths"]["nas_root"]) / frozen["capture"]["export"]["root"] / run_id
    if receipt.get('origin') == 'migrated':
        publication_path = Path(cfg['project_root']) / 'runs' / run_id / 'hidden/publication.json'
        if sha256(publication_path) != receipt['source_publication_sha256']:
            raise ValueError('Migrated capture publication changed')
        publication = read_json(publication_path)
        if publication['nas_locator'] != receipt['source_root']:
            raise ValueError('Migrated capture root differs from explicit publication')
        source = Path(receipt['source_root'])
    out = artifact_dir(cfg, "difficulty", index_id, "datasets")
    index_data = {
            "schema_version": 3,
            "dataset_id": run_id,
            "capture_id": run_id,
            "hidden_root": str(source),
            "label_root": artifact_reference(cfg["project_root"], trace),
            "publication_id": publication_id,
            "samples": samples,
            "source_receipt_sha256": sha256(trace / "receipt.json"),
            "split_authority": "source run task snapshot",
            "source_run_id": run_id, "rollout_ids": selected["rollout_ids"],
            "evidence_revision": catalog["revision"], "evidence_sha256": selected["evidence_sha256"],
        }
    coverage = read_json(trace / 'coverage.json') if (trace / 'coverage.json').exists() else {}
    index_data['coverage'] = {
        'path': artifact_reference(cfg['project_root'], trace / 'coverage.json'),
        'sha256': sha256(trace / 'coverage.json') if coverage else None,
        'expected': coverage.get('expected'), 'observed': coverage.get('observed'),
        'native_hidden_missing': len(coverage.get('missing', [])),
        'unexpected': len(coverage.get('unexpected', [])),
    }
    if (out / 'receipt.json').exists():
        prior = read_json(out / 'receipt.json')
        if prior.get('source_receipt_sha256') != index_data['source_receipt_sha256']:
            raise FileExistsError('Index ID already binds another publication')
        if read_json(out / 'index.json') != index_data or sha256(out / 'index.json') != prior['files']['index.json']:
            raise ValueError('Immutable published index checksum or selection mismatch')
        return out
    from uuid import uuid4
    stage = out.with_name('.' + out.name + '.stage')
    if stage.exists():
        stage.rename(stage.with_name(stage.name + '.interrupted-' + uuid4().hex))
    if out.exists():
        raise FileExistsError('Unreceipted index directory needs inspection: ' + str(out))
    stage.mkdir(parents=True)
    write_json(stage / 'index.json', index_data)
    write_json(stage / "receipt.json", {
        "schema_version": 1, "state": "complete", "dataset_id": index_id,
        "source_receipt_sha256": sha256(trace / "receipt.json"),
        "source_run_id": run_id,
        "source_files": {str(trace / "receipt.json"): sha256(trace / "receipt.json"),
                         str(Path(cfg["project_root"]) / "runs" / run_id / "resolved-config.json"):
                         sha256(Path(cfg["project_root"]) / "runs" / run_id / "resolved-config.json")},
        "files": {"index.json": sha256(stage / "index.json")},
    })
    stage.rename(out)
    return out


@exclusive_artifact("difficulty-dataset", "index_id")
def build_online_index(cfg, capture_id, index_id):
    source = artifact_dir(cfg, "hidden", capture_id)
    receipt = read_json(source / "receipt.json")
    if any(sha256(source / name) != digest for name, digest in receipt["files"].items()):
        raise ValueError("Packed online capture checksum mismatch")
    run_id = cfg.get("source_run_id") or cfg.get("run_id")
    if not run_id:
        raise ValueError("Packed dataset selection requires an explicit source run")
    catalog = RunEvidence(cfg["project_root"], run_id).read()
    wanted = set(select_run(cfg["project_root"], run_id)["rollout_ids"])
    partitions = {task["instance_id"]: task["partition"] for task in catalog["tasks"].values()}
    rows = []
    for event in read_json(source / "index.json"):
        if event["rollout_id"] not in catalog["rollouts"]:
            raise ValueError("Packed event belongs to another run")
        if event["rollout_id"] not in wanted:
            continue
        row = dict(event)
        row["remaining_steps_log1p"] = math.log1p(row["remaining_steps"])
        row["remaining_reasoning_tokens_log1p"] = math.log1p(row["remaining_reasoning_tokens_after"])
        row["partition"] = partitions.get(row["task_id"], "diagnostic")
        row["hidden_path"] = str((source / row["path"]).relative_to(Path(cfg["project_root"])))
        rows.append(row)
    out = artifact_dir(cfg, "difficulty", index_id, "datasets")
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "index.json",
        {
            "schema_version": 2,
            "capture_id": capture_id,
            "capture_receipt_sha256": sha256(source / "receipt.json"),
            "samples": rows,
            "split_authority": "source run task snapshot; no token-level random split",
        },
    )
    write_json(out / "receipt.json", {
        "schema_version": 1, "state": "complete", "dataset_id": index_id,
        "capture_receipt_sha256": sha256(source / "receipt.json"),
        "files": {"index.json": sha256(out / "index.json")},
    })
    action = artifact_dir(cfg, "action", index_id, "datasets")
    write_json(
        action / "index.json",
        {
            "shared_index": str((out / "index.json").relative_to(cfg["project_root"])),
            "shared_index_sha256": sha256(out / "index.json"),
            "taxonomy_version": None,
            "samples": [{"sample_id": row["sample_id"], "action_label": None} for row in rows],
        },
    )
    return out

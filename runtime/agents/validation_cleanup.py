"""Reconcile stopped validation sandboxes without rewriting original evidence."""
from pathlib import Path
import uuid

import docker

from core.io import hash_json, now, read_json, sha256, write_json
from core.locking import file_lock


def _environment_auditor_running(project, run_id):
    """Environment audits predate sweep locks; refuse their live CLI owner."""
    import os
    root = Path(project).resolve()
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            if entry.stat().st_uid != os.getuid() or (entry / 'cwd').resolve() != root:
                continue
            args = (entry / 'cmdline').read_bytes().split(b'\0')
            expected = [b'skillsbench', b'environment-check', run_id.encode()]
            if any(args[i:i + 3] == expected for i in range(len(args) - 2)):
                return True
        except (OSError, RuntimeError):
            continue
    return False


def reconcile_validation(cfg, run_id, *, section="oracle", task_ids=None):
    from application.execution.run_plan import load_plan

    loaded = load_plan(cfg["project_root"], run_id)
    frozen, run = loaded["config"], loaded["run"]
    if frozen["benchmark"]["backend"] != "skillsbench":
        raise ValueError("Validation cleanup requires a SkillsBench run")
    if section not in {"oracle", "empty-control", "offline-oracle", "environments"}:
        raise ValueError("Unknown validation section")
    base = run / "validation" / section
    selected = set(task_ids or [t["instance_id"] for t in frozen["selection"]["tasks"].values()])
    known = {t["instance_id"] for t in frozen["selection"]["tasks"].values()}
    if selected - known:
        raise ValueError("Cleanup task absent from frozen selection")
    project = str(Path(cfg["project_root"]).resolve())
    if section == 'environments' and _environment_auditor_running(project, run_id):
        raise ValueError('Refusing cleanup while the environment auditor is active')
    result = {"schema_version": 1, "run_id": run_id, "section": section,
              "at": now(), "attempts": [], "original_receipts_unchanged": True}
    with file_lock(run / 'validation/.oracle-lifecycle.lock', nonblocking=True), \
            file_lock(base / ".sweep.lock", nonblocking=True):
        client = docker.DockerClient(base_url=frozen["environment"]["docker_host"], timeout=10)
        try:
            for task_id in sorted(selected):
                for path in sorted((base / task_id).rglob("ownership-compose.json")):
                    labels = read_json(path)["services"]["main"]["labels"]
                    expected = {"dycon.project": project, "dycon.run": run_id,
                                "dycon.task": task_id, "dycon.role": "skillsbench-environment"}
                    if any(labels.get(k) != v for k, v in expected.items()) or not labels.get("dycon.epoch"):
                        raise ValueError("Validation ownership differs from frozen run")
                    expected.update({k: labels[k] for k in ("dycon.epoch", "dycon.attempt", "dycon.sample")})
                    filters = {"label": [k + "=" + v for k, v in expected.items()]}
                    containers = client.containers.list(all=True, filters=filters)
                    row = {"ownership": expected, "source": str(path.relative_to(run)),
                           "source_sha256": sha256(path),
                           "removed_containers": [], "removed_networks": []}
                    # Inspect all matches before any mutation; running work is never stopped.
                    for container in containers:
                        container.reload()
                        if (any(container.labels.get(k) != v for k, v in expected.items())
                                or container.status not in {"exited", "dead", "created"}):
                            raise ValueError("Refusing cleanup of active or differently owned container")
                    for container in containers:
                        row["removed_containers"].append({"id": container.id,
                            "state": container.attrs.get("State", {}),
                            "image": container.attrs.get("Image")})
                        container.remove(force=False, v=False)
                    for network in client.networks.list(filters=filters):
                        network.reload()
                        if (any(network.attrs.get("Labels", {}).get(k) != v for k, v in expected.items())
                                or network.attrs.get("Containers")):
                            raise ValueError("Refusing cleanup of active or differently owned network")
                        network.remove()
                        row["removed_networks"].append(network.id)
                    row["remaining_containers"] = [c.id for c in client.containers.list(all=True, filters=filters)]
                    row["remaining_networks"] = [n.id for n in client.networks.list(filters=filters)]
                    row["cleanup_confirmed"] = not (row["remaining_containers"] or row["remaining_networks"])
                    result["attempts"].append(row)
        finally:
            client.close()
    result["cleanup_confirmed"] = bool(result["attempts"]) and all(r["cleanup_confirmed"] for r in result["attempts"])
    result["receipt_sha256"] = hash_json(result)
    target = base / "cleanup-reconciliations" / (uuid.uuid4().hex + ".json")
    write_json(target, result)
    return {"run_id": run_id, "section": section, "receipt": str(target),
            "attempt_count": len(result["attempts"]), "cleanup_confirmed": result["cleanup_confirmed"],
            "removed_containers": sum(len(r["removed_containers"]) for r in result["attempts"])}

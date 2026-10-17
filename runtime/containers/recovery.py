"""Remove only orphaned DyConGlobal containers, then reclaim owned images."""

from pathlib import Path

from core.io import identifier
from core.locking import file_lock


def recover(cfg, client=None):
    import docker

    from runtime.images.nas import ImageLeaseManager

    client = client or docker.DockerClient(base_url=cfg["environment"]["docker_host"])
    removed, active = [], []
    containers = client.containers.list(all=True, filters={"label": "dycon.project=" + cfg["project_root"]})
    for container in containers:
        run_id = identifier(container.labels["dycon.run"])
        try:
            with file_lock(
                Path(cfg["project_root"]) / "state/run-locks" / (run_id + ".lock"), nonblocking=True
            ):
                container.remove(force=True)
                removed.append(container.id)
        except BlockingIOError:
            active.append(container.id)
    return {
        "removed_containers": removed,
        "active_containers": active,
        "images": ImageLeaseManager(cfg, client).cleanup(),
    }

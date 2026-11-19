"""Explicit SWE image repairs using shared ownership, lease and OCI APIs."""
import io
from pathlib import Path

from application.execution.run_plan import load_plan
from core.io import identifier, now, read_json, sha256, write_json
from core.locking import file_lock
from runtime.images.nas import NasImageStore, ImageLeaseManager
from runtime.images.ownership import derive_image, validate_repair

SYSTEM_PACKAGES = {
    "gettext": ["gettext"],
    "graphics": ["libgl1", "libxrender1", "libxt6", "libxext6", "libsm6", "libice6"],
}

def prepare(root, run_id, publication_id, task_ids, kind):
    import docker
    loaded = load_plan(root, run_id)
    cfg = loaded["config"]
    publication_id = identifier(publication_id)
    rows = {v["instance_id"]: v["instance"] for v in cfg["selection"]["tasks"].values()}
    selected = sorted(set(task_ids))
    if not selected or set(selected) - rows.keys():
        raise ValueError("Specify known frozen task IDs")
    store = Path(cfg["paths"]["nas_root"]) / publication_id
    store.mkdir(parents=True, exist_ok=True)
    request = {"run_id": run_id, "tasks": selected, "kind": kind,
               "sources": {iid: rows[iid]["nas_image"] for iid in selected}}
    with file_lock(Path(root) / "state/images" / (publication_id + ".lock"), nonblocking=True):
        request_path = store / "request.json"
        if request_path.exists() and read_json(request_path) != request:
            raise ValueError("Image publication identity changed")
        write_json(request_path, request)
        result = {}
        client = docker.DockerClient(base_url=cfg["environment"]["docker_host"], timeout=900)
        leases = ImageLeaseManager(cfg, client)
        try:
            for iid in selected:
                receipt_path = store / "receipts" / (iid + ".json")
                if receipt_path.exists():
                    result[iid] = read_json(receipt_path)
                    NasImageStore(cfg, result[iid]["reference"]).verify()
                    continue
                source = NasImageStore(cfg, rows[iid]["nas_image"])
                source.verify()
                if kind == "ownership":
                    ref = derive_image(source, store, iid, max_id=65535)
                    validation = validate_repair(cfg, source.reference, ref) if ref != source.reference else {"unchanged": True}
                    row = {"reference": ref, "source": source.reference, "validation": validation}
                else:
                    if kind not in SYSTEM_PACKAGES:
                        raise ValueError("Unknown image repair kind")
                    with leases.lease(source):
                        image = client.images.get(source.image_id)
                        base = image.id
                        packages = SYSTEM_PACKAGES[kind]
                        definition = ("FROM " + base + "\nUSER root\n"
                            "RUN test -f /etc/debian_version && apt-get update && "
                            "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "
                            + " ".join(packages) + " && mkdir -p /opt/dycon-image-repair && "
                            "dpkg-query -W > /opt/dycon-image-repair/packages.tsv && "
                            "rm -rf /var/lib/apt/lists/*\n")
                        build_dir = Path(root) / "temp/image-repairs" / publication_id / iid
                        build_dir.mkdir(parents=True, exist_ok=True)
                        (build_dir / "Dockerfile").write_text(definition)
                        tag = "dycon-swe-repair:" + publication_id + "-" + iid
                        built, logs = client.images.build(fileobj=io.BytesIO(definition.encode()),
                            tag=tag, rm=True, forcerm=True, pull=False, network_mode="bridge",
                            timeout=1800, labels={"dycon.image-repair": kind})
                        from runtime.images.export import export_image
                        provenance = {"kind": kind, "source": source.reference,
                                      "packages": packages, "dockerfile_sha256": sha256(build_dir / "Dockerfile"),
                                      "image_id": built.id, "at": now()}
                        ref = export_image(client, built.id, store, provenance,
                                           requested=tag, manifest_name="images/" + iid + ".json")
                        row = {"reference": ref, "source": source.reference, "provenance": provenance}
                        client.images.remove(tag)
                write_json(receipt_path, row)
                result[iid] = row
            write_json(store / "completion.json", {"state": "finished", "results": result, "at": now()})
        finally:
            leases.cleanup()
            client.close()
        return {"publication": str(store), "results": result}

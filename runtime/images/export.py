"""Shared Docker save stream to immutable NAS OCI publication."""
import io, json, os, shutil, tarfile, tempfile
from pathlib import Path
from core.io import read_json, sha256, write_json
from runtime.images.oci import descriptor_graph
from runtime.images.ownership import json_blob, publish_blob

class _Chunks(io.RawIOBase):
    def __init__(self, source):
        self.source, self.buffer = iter(source), bytearray()

    def readable(self):
        return True

    def readinto(self, target):
        while not self.buffer:
            try:
                self.buffer.extend(next(self.source))
            except StopIteration:
                return 0
        count = min(len(target), len(self.buffer))
        target[:count] = self.buffer[:count]
        del self.buffer[:count]
        return count


def export_image(client, image_id, store, provenance, *, requested, manifest_name):
    """Convert a daemon save stream to the project's shared NAS OCI contract."""
    blobs = store / "blobs/sha256"
    blobs.mkdir(parents=True, exist_ok=True)
    entries, manifest = {}, None
    stream = io.BufferedReader(_Chunks(client.api.get_image(image_id)))
    with stream, tarfile.open(fileobj=stream, mode="r|*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            source = tar.extractfile(member)
            if member.name == "manifest.json":
                manifest = json.load(source)
                continue
            if (member.name.startswith("blobs/sha256/") or member.name.endswith(".json") or member.name.endswith("/layer.tar")):
                fd, name = tempfile.mkstemp(prefix=".browser-export-", dir=blobs)
                try:
                    with os.fdopen(fd, "wb") as output:
                        shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                        output.flush()
                        os.fsync(output.fileno())
                    with open(name, "rb") as layer_file:
                        magic = layer_file.read(4)
                    media = "application/vnd.oci.image.layer.v1.tar"
                    if magic.startswith(b"\x1f\x8b"):
                        media += "+gzip"
                    elif magic == b"\x28\xb5\x2f\xfd":
                        media += "+zstd"
                    entries[member.name] = {**publish_blob(Path(name), blobs), "mediaType": media}
                finally:
                    Path(name).unlink(missing_ok=True)
    if not isinstance(manifest, list) or len(manifest) != 1:
        raise ValueError("Browser export must contain exactly one Docker image")
    raw = manifest[0]
    config = {**entries[raw["Config"]], "mediaType": "application/vnd.oci.image.config.v1+json"}
    layers = [{**entries[name], "mediaType": entries[name]["mediaType"]}
              for name in raw["Layers"]]
    value = {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json",
             "config": config, "layers": layers}
    root = {**json_blob(value, blobs), "mediaType": value["mediaType"]}
    graph = descriptor_graph(blobs, root, deep=True)
    record = {"schema_version": 2, "requested": requested,
        "manifest_digest": root["digest"], "manifest_size": root["size"],
        "root_media_type": root["mediaType"], "runtime_manifest_digest": root["digest"],
        "config": config, "layers": layers, "compressed_bytes": sum(d["size"] for d in layers),
        "repo_digests": [], "platform": "linux/amd64", "descriptor_count": len(graph),
        "deep_verified": True, "provenance": provenance}
    path = store / manifest_name
    if path.exists() and read_json(path) != record:
        raise ValueError("Browser image publication already exists with different bytes")
    write_json(path, record)
    return {"store": store.name, "manifest": manifest_name, "manifest_sha256": sha256(path),
            "image_id": root["digest"]}


import hashlib
import io
import json
import tarfile

import pytest

from runtime.images.nas import NasImageStore
from runtime.images.oci import descriptor_graph, platform_manifest

OCI = "application/vnd.oci.image."


def make_graph(tmp_path, indexed=True):
    blobs = tmp_path / "blobs/sha256"
    blobs.mkdir(parents=True)

    def blob(value, media):
        data = value if isinstance(value, bytes) else json.dumps(value).encode()
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        (blobs / digest[7:]).write_bytes(data)
        return {"digest": digest, "size": len(data), "mediaType": media}

    config = blob({"os": "linux", "architecture": "amd64"}, OCI + "config.v1+json")
    layer = blob(b"test-layer-bytes", OCI + "layer.v1.tar")
    manifest = blob(
        {"schemaVersion": 2, "mediaType": OCI + "manifest.v1+json", "config": config, "layers": [layer]},
        OCI + "manifest.v1+json",
    )
    primary = manifest
    if indexed:
        att_config = blob({}, OCI + "config.v1+json")
        att_layer = blob(b"provenance", "application/vnd.in-toto+json")
        att = blob(
            {
                "schemaVersion": 2,
                "mediaType": OCI + "manifest.v1+json",
                "config": att_config,
                "layers": [att_layer],
            },
            OCI + "manifest.v1+json",
        )
        primary = blob(
            {
                "schemaVersion": 2,
                "mediaType": OCI + "index.v1+json",
                "manifests": [
                    {**manifest, "platform": {"os": "linux", "architecture": "amd64"}},
                    {**att, "platform": {"os": "unknown", "architecture": "unknown"}},
                ],
            },
            OCI + "index.v1+json",
        )
    return blobs, primary, manifest, config, layer


@pytest.mark.parametrize("indexed", [True, False])
def test_original_identity_and_complete_graph_survive_archive(tmp_path, indexed):
    blobs, root, runtime, config, layer = make_graph(tmp_path, indexed)
    graph = descriptor_graph(blobs, root, deep=True)
    assert len(graph) == (7 if indexed else 3)
    selected, obj = platform_manifest(blobs, root)
    assert selected["digest"] == runtime["digest"]
    store = NasImageStore.__new__(NasImageStore)
    store.root = tmp_path
    store.image_id = root["digest"]
    store.record = {"manifest_size": root["size"], "config": config, "layers": [layer]}
    stream = io.BytesIO()
    store.write_archive(stream)
    stream.seek(0)
    with tarfile.open(fileobj=stream) as archive:
        index = json.load(archive.extractfile("index.json"))
        assert index["manifests"][0]["digest"] == root["digest"]
        assert len(archive.getmembers()) == len(graph) + 2
        for descriptor in graph:
            content = archive.extractfile("blobs/sha256/" + descriptor["digest"][7:]).read()
            assert "sha256:" + hashlib.sha256(content).hexdigest() == descriptor["digest"]


def test_missing_attestation_blocks_deletion_even_if_runtime_exists(tmp_path):
    blobs, root, runtime, config, layer = make_graph(tmp_path)
    graph = descriptor_graph(blobs, root)
    missing = next(d for d in graph if d["mediaType"] == "application/vnd.in-toto+json")
    (blobs / missing["digest"][7:]).unlink()
    with pytest.raises(FileNotFoundError):
        descriptor_graph(blobs, root)


def test_same_size_corruption_fails_hash_verification(tmp_path):
    blobs, root, runtime, config, layer = make_graph(tmp_path)
    p = blobs / layer["digest"][7:]
    p.write_bytes(b"x" * layer["size"])
    with pytest.raises(ValueError, match="Corrupt OCI blob"):
        descriptor_graph(blobs, root, deep=True)


def test_platform_selection_rejects_wrong_runtime(tmp_path):
    blobs, root, *_ = make_graph(tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        platform_manifest(blobs, root, "linux/arm64")


def test_digest_cannot_escape_blob_namespace(tmp_path):
    with pytest.raises(ValueError):
        descriptor_graph(tmp_path, {"digest": "../unsafe", "size": 0})


def test_blob_symlink_cannot_escape_store(tmp_path):
    from runtime.images.oci import blob_path

    root = tmp_path / "blobs"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"content")
    (root / ("1" * 64)).symlink_to(outside)
    with pytest.raises(ValueError, match="escapes store"):
        blob_path(root, "sha256:" + "1" * 64)


def test_parallel_preflight_checks_unique_blobs_and_rejects_corruption(tmp_path):
    from runtime.images.oci import verify_blobs

    blobs, root, _, _, layer = make_graph(tmp_path)
    descriptors = descriptor_graph(blobs, root)
    assert len(verify_blobs(blobs, descriptors + descriptors)) == len(descriptors)
    (blobs / layer["digest"][7:]).write_bytes(b"x" * layer["size"])
    with pytest.raises(ValueError, match="Corrupt OCI blob"):
        verify_blobs(blobs, descriptors)

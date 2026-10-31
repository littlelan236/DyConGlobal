"""Docker save exports use either legacy layer paths or OCI blob names."""
import gzip
import hashlib
import io
import json
import tarfile
from types import SimpleNamespace

import pytest

from core.io import read_json
from runtime.images.oci import descriptor_graph
from runtime.images.webarena_browser import export_image


@pytest.mark.parametrize("oci_names", [False, True])
@pytest.mark.parametrize("codec", ["tar", "gzip", "zstd"])
def test_export_image_preserves_config_and_layers(tmp_path, oci_names, codec):
    config = json.dumps({"architecture": "amd64", "os": "linux"}).encode()
    layer = b"actual layer bytes"
    if codec == "gzip":
        layer = gzip.compress(layer)
    elif codec == "zstd":
        layer = b"\x28\xb5\x2f\xfd" + layer
    names = (["blobs/sha256/" + hashlib.sha256(raw).hexdigest() for raw in (config, layer)]
             if oci_names else ["config.json", "layer/layer.tar"])
    manifest = [{"Config": names[0], "Layers": names[1:]}]
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        for name, raw in [(names[0], config), (names[1], layer),
                          ("manifest.json", json.dumps(manifest).encode())]:
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            tar.addfile(member, io.BytesIO(raw))
    raw = archive.getvalue()
    client = SimpleNamespace(api=SimpleNamespace(get_image=lambda _: iter(
        raw[i:i + 73] for i in range(0, len(raw), 73))))
    store = tmp_path / "publication"
    reference = export_image(client, "image-id", store, {"source": "test"})
    image = read_json(store / reference["manifest"])
    assert image["config"]["digest"] == "sha256:" + hashlib.sha256(config).hexdigest()
    assert image["layers"][0]["digest"] == "sha256:" + hashlib.sha256(layer).hexdigest()
    suffix = "" if codec == "tar" else "+" + codec
    assert image["layers"][0]["mediaType"] == "application/vnd.oci.image.layer.v1.tar" + suffix
    root = {"digest": image["manifest_digest"], "size": image["manifest_size"],
            "mediaType": image["root_media_type"]}
    assert len(descriptor_graph(store / "blobs/sha256", root, deep=True)) == 3
    target = store / "blobs/sha256" / image["layers"][0]["digest"][7:]
    target.write_bytes(b"!" * len(layer))
    with pytest.raises(ValueError, match="Corrupt OCI blob"):
        descriptor_graph(store / "blobs/sha256", root, deep=True)

import json

import pytest

from core.io import sha256
from runtime.containers.resources import assets, mounts, preparation


def test_legacy_ca_bundle_is_digest_bound_and_restricted_to_its_trust_store(tmp_path):
    source = tmp_path / "runtime/resources/certifi.pem"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture PEM bytes")
    row = {"source": str(source), "sha256": sha256(source),
           "destination": "/testbed/requests/cacert.pem", "transform": "requests-legacy-ca-v1"}

    def config():
        return {"project_root": str(tmp_path), "environment": {"variables": {
            "DYCON_SWEBENCH_RESOURCES": json.dumps([row])}}}

    assert assets(config()) == [row]
    assert mounts(config())[str(source)] == {
        "bind": "/opt/dycon-resources/" + row["sha256"], "mode": "ro"}
    assert preparation(config())
    row["destination"] = "/testbed/requests/adapters.py"
    with pytest.raises(ValueError, match="Invalid immutable test resource"):
        assets(config())
    row["destination"] = "/testbed/requests/cacert.pem"
    source.write_bytes(b"modified PEM bytes")
    with pytest.raises(ValueError, match="Invalid immutable test resource"):
        assets(config())

from pathlib import Path

import pytest
from test_images import Client, store

from runtime.containers.factory import ContainerFactory
from runtime.containers.repairs import environment_overrides
from runtime.images.nas import ImageLeaseManager


def test_cold_mode_rejects_cached_images_without_deleting_unowned_images(cfg):
    cfg = {**cfg, "require_cold_images": True}
    client = Client(cfg["project_root"])
    image = store(1)
    client.images.ids.add(image.image_id)
    manager = ImageLeaseManager(cfg, client)
    with pytest.raises(RuntimeError, match="existing image"):
        with manager.lease(image):
            pytest.fail("A cold audit must never use a cached image")
    assert image.image_id in client.images.ids
    assert client.images.removed == []


def test_dependency_repairs_use_shared_bootstrap_for_all_callers(cfg):
    wheels = Path(cfg["project_root"]) / "runtime/cache/gold-wheels"
    wheels.mkdir(parents=True)
    (wheels / "appdirs-1.4.4-py2.py3-none-any.whl").write_bytes(b"fixture")
    factory = ContainerFactory(cfg, client=object())
    row = {"instance_id": "pylint-dev__pylint-4661", "repo": "pylint-dev/pylint"}
    policy = factory.policy_for(row)
    assert "appdirs==1.4.4" in policy.specs(row["repo"])
    assert "--no-index" in policy.bootstrap(row["repo"])
    assert "appdirs==1.4.4" in policy.bootstrap(row["repo"])
    other = {"instance_id": "ordinary-task", "repo": row["repo"]}
    assert factory.policy_for(other) is factory.policy


def test_sympy_repair_preserves_other_repository_pins(cfg):
    cfg = {**cfg, "environment": {**cfg["environment"], "repo_pins": {"other/repo": ["pytest<8"]}}}
    row = {
        "instance_id": "sympy__sympy-23262",
        "repo": "sympy/sympy",
        "environment_overrides": {"repo_pins": {"sympy/sympy": ["scipy==1.13.1"]}},
    }
    env = environment_overrides(cfg, row)
    assert env["repo_pins"]["other/repo"] == ["pytest<8"]
    assert env["repo_pins"]["sympy/sympy"].count("scipy==1.13.1") == 1
    assert "numpy==1.26.4" in env["repo_pins"]["sympy/sympy"]
    assert row["environment_overrides"]["repo_pins"]["sympy/sympy"] == ["scipy==1.13.1"]

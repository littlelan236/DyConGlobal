import pytest

from runtime.containers.factory import ContainerFactory


@pytest.mark.parametrize(
    "iid,repo",
    [
        ("psf__requests-1921", "psf/requests"),
        ("sphinx-doc__sphinx-10435", "sphinx-doc/sphinx"),
        ("sphinx-doc__sphinx-7985", "sphinx-doc/sphinx"),
        ("sphinx-doc__sphinx-8475", "sphinx-doc/sphinx"),
    ],
)
def test_network_repair_reaches_real_container_spec(cfg, iid, repo):
    factory = ContainerFactory(cfg, client=object())
    factory.lock_images = lambda rows: None
    factory.images[iid] = {"image_id": "sha256:" + "1" * 64}
    spec = factory.spec({"instance_id": iid, "repo": repo})
    assert spec["network_mode"] == "bridge"
    assert factory.policy_for({"instance_id": iid, "repo": repo}).environment["network_fallback"] is False
    # Another task, even from the same repository, keeps the configured policy.
    ordinary = {"instance_id": "sphinx-doc__sphinx-10673", "repo": "sphinx-doc/sphinx"}
    factory.images[ordinary["instance_id"]] = {"image_id": "sha256:" + "2" * 64}
    assert factory.spec(ordinary)["network_mode"] == cfg["environment"].get("network_mode", "none")

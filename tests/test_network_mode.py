from types import SimpleNamespace

from runtime.containers.factory import ContainerFactory


def test_container_spec_uses_single_network_policy(cfg):
    factory = ContainerFactory.__new__(ContainerFactory)
    factory.cfg = {**cfg, "environment": {**cfg["environment"], "network_mode": "none"}}
    factory.images = {"t-1": {"image_id": "sha256:" + "1" * 64}}
    factory._task_policies = {}
    factory._lock = __import__("threading").Lock()
    factory.policy = SimpleNamespace(environment={"user": "root", "platform": "linux/amd64", "cwd": "/testbed", "wheelhouse": None}, variables=lambda: {}, identity=lambda repo: {"policy_hash": "p"})
    factory.policy_for = lambda instance: factory.policy
    factory.lock_images = lambda rows: factory.images
    spec = factory.spec({"instance_id": "t-1", "repo": "org/repo"})
    assert spec["network_mode"] == "none"

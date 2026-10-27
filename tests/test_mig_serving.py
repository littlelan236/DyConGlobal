import pytest

from core.config import Serving
from runtime.serving.resources import (
    device_lock_name,
    normalize_devices,
    reserve_serving_devices,
)


def serving(**overrides):
    values = {
        "base_url": "http://127.0.0.1:8101/v1",
        "api_key_env": "DYCON_API_KEY",
        "management": "managed",
        "python": ".venvs/easysteer/bin/python",
        "gpus": [0, 1, 2, 3],
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.4,
        "max_model_len": 1024,
        "startup_seconds": 10,
        "shutdown_seconds": 10,
        "tool_parser": "qwen3_xml",
        "reasoning_parser": "qwen3",
        "language_model_only": True,
    }
    values.update(overrides)
    return Serving.model_validate(values)


def test_mig_uuid_is_a_single_tp_device():
    device = "MIG-2f2fe213-508a-57bb-b2c7-52f4ae0c1021"
    cfg = serving(gpus=[device], tensor_parallel_size=1)
    assert cfg.gpus == [device]
    assert normalize_devices(cfg.gpus) == (device,)


def test_mig_requires_matching_tp_and_does_not_mix_namespaces():
    device = "MIG-2f2fe213-508a-57bb-b2c7-52f4ae0c1021"
    with pytest.raises(ValueError, match="count"):
        serving(gpus=[device], tensor_parallel_size=2)
    with pytest.raises(ValueError, match="mix"):
        serving(gpus=[0, device], tensor_parallel_size=2)


def test_device_lock_identity_is_stable_and_namespace_specific():
    mig = "MIG-2f2fe213-508a-57bb-b2c7-52f4ae0c1021"
    assert device_lock_name([mig]) == device_lock_name([mig])
    assert device_lock_name([mig]) != device_lock_name(["MIG-2db3e350-5a6e-589c-8630-a7007170eea0"])
    assert device_lock_name([0]) != device_lock_name([mig])


def test_reservation_releases_all_device_locks(tmp_path, monkeypatch):
    import runtime.serving.resources as resources

    device = "MIG-2f2fe213-508a-57bb-b2c7-52f4ae0c1021"
    monkeypatch.setattr(
        resources,
        "inspect_devices",
        lambda python, devices: {"device_kind": "mig", "device_memory": [{"process_count": 0}]},
    )
    with reserve_serving_devices(tmp_path, "python", [device]):
        pass
    with reserve_serving_devices(tmp_path, "python", [device]):
        pass


def test_physical_gpu_sharing_ignores_existing_device_locks(tmp_path, monkeypatch):
    from contextlib import ExitStack
    from hashlib import sha256
    from core.locking import file_lock
    import runtime.serving.resources as resources

    monkeypatch.setattr(resources, "inspect_devices", lambda python, devices: {
        "device_kind": "physical", "device_memory": [{"process_count": 1}],
    })
    with ExitStack() as stack:
        for device in ("2", "3"):
            digest = sha256(device.encode()).hexdigest()[:24]
            stack.enter_context(file_lock(tmp_path / "state" / f"serving-gpu-{digest}.lock", nonblocking=True))
        with reserve_serving_devices(tmp_path, "python", [2, 3]) as info:
            assert info["lock_paths"] == []
            with reserve_serving_devices(tmp_path, "python", [2, 3]):
                pass


def test_mig_still_rejects_overlapping_reservations(tmp_path, monkeypatch):
    import runtime.serving.resources as resources

    device = "MIG-2f2fe213-508a-57bb-b2c7-52f4ae0c1021"
    monkeypatch.setattr(resources, "inspect_devices", lambda python, devices: {
        "device_kind": "mig", "device_memory": [{"process_count": 0}],
    })
    with reserve_serving_devices(tmp_path, "python", [device]):
        with pytest.raises(BlockingIOError):
            with reserve_serving_devices(tmp_path, "python", [device]):
                pass

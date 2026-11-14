"""Explicit MIG UUID selection and per-instance admission, before Torch."""
from types import SimpleNamespace

import pytest

from RL.devices import inspect_mig
from RL.service import _select_gpu

UUID = "MIG-654b3ac2-156b-5fc6-aed8-4ac7394bd99c"


@pytest.mark.parametrize("used,passes", [(128, True), (22000, False)])
def test_mig_admission_uses_selected_instance_and_uuid(used, passes, monkeypatch):
    listing = f"GPU 5: Test (UUID: GPU-parent)\n  MIG 2g.48gb Device 0: (UUID: {UUID})"
    xml = f"""<nvidia_smi_log><gpu><product_name>Test</product_name><mig_devices>
      <mig_device><index>0</index><gpu_instance_id>2</gpu_instance_id><compute_instance_id>0</compute_instance_id>
      <fb_memory_usage><total>48512 MiB</total><used>{used} MiB</used><free>48000 MiB</free></fb_memory_usage></mig_device>
      <mig_device><index>1</index><fb_memory_usage><used>48000 MiB</used></fb_memory_usage></mig_device>
      </mig_devices></gpu></nvidia_smi_log>"""
    def probe(args, **kwargs):
        assert kwargs["timeout"] == 15
        return listing if "-L" in args else xml
    monkeypatch.setattr("RL.devices.subprocess.check_output", probe)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID)
    config = SimpleNamespace(runtime=SimpleNamespace(gpu=UUID))
    if passes:
        value = _select_gpu(config)
        assert value["uuid"] == UUID and value["physical_gpu"] == 5
        assert value["total_MiB"] == 48512 and value["gpu_instance_id"] == 2
        import os
        assert os.environ["CUDA_VISIBLE_DEVICES"] == UUID
    else:
        with pytest.raises(RuntimeError, match="not idle"):
            _select_gpu(config)


def test_missing_mig_instance_is_rejected(monkeypatch):
    monkeypatch.setattr("RL.devices.subprocess.check_output", lambda *a, **k: "GPU 5: Test")
    with pytest.raises(ValueError, match="does not exist"):
        inspect_mig(UUID)

"""GPU resource identity and admission for managed serving."""

from __future__ import annotations

import json
import os
import re
import subprocess
from contextlib import ExitStack, contextmanager
from hashlib import sha256
from pathlib import Path

from core.locking import file_lock

_MIG_UUID = re.compile(r"^MIG-[0-9a-fA-F-]{36}$")


def normalize_devices(devices):
    if not devices:
        raise ValueError("serving.gpus must contain at least one device")
    normalized = []
    for device in devices:
        if type(device) is int:
            if device < 0:
                raise ValueError("serving GPU indices must be nonnegative")
            normalized.append(str(device))
        elif type(device) is str and _MIG_UUID.fullmatch(device):
            normalized.append(device)
        else:
            raise ValueError("serving.gpus entries must be nonnegative integers or MIG UUIDs")
    if len(set(normalized)) != len(normalized):
        raise ValueError("serving.gpus entries must be unique")
    kinds = {"mig" if item.startswith("MIG-") else "physical" for item in normalized}
    if len(kinds) != 1:
        raise ValueError("serving.gpus cannot mix physical GPUs and MIG UUIDs")
    if kinds == {"mig"} and len(normalized) != 1:
        raise ValueError("Managed MIG serving requires exactly one instance and TP=1")
    return tuple(normalized)


def device_lock_name(devices):
    normalized = normalize_devices(devices)
    digest = sha256(",".join(normalized).encode()).hexdigest()[:24]
    return f"serving-gpu-{digest}.lock"


def _nvidia_lines(*args):
    return subprocess.check_output(["nvidia-smi", *args], text=True).splitlines()


def _available_devices():
    physical = set()
    for line in _nvidia_lines("--query-gpu=index,uuid", "--format=csv,noheader,nounits"):
        fields = [part.strip() for part in line.split(",", 1)]
        if len(fields) == 2:
            physical.add(fields[0])
    mig = set()
    for line in _nvidia_lines("-L"):
        match = re.search(r"UUID: (MIG-[0-9a-fA-F-]{36})", line)
        if match:
            mig.add(match.group(1))
    return physical, mig


def _probe_visible_devices(serving_python, devices):
    code = r'''
import json
import os
import torch
from vllm.third_party import pynvml

pynvml.nvmlInit()
rows = []
try:
    for item in os.environ["DYCON_SERVING_PROBE_DEVICES"].split(","):
        handle = (
            pynvml.nvmlDeviceGetHandleByUUID(item)
            if item.startswith("MIG-")
            else pynvml.nvmlDeviceGetHandleByIndex(int(item))
        )
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        rows.append({
            "device": item,
            "total_memory_bytes": int(memory.total),
            "free_memory_bytes": int(memory.free),
            "process_count": len(processes),
        })
    print(json.dumps({"visible_device_count": torch.cuda.device_count(), "device_memory": rows}))
finally:
    pynvml.nvmlShutdown()
'''
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    env["DYCON_SERVING_PROBE_DEVICES"] = ",".join(devices)
    output = subprocess.check_output(
        [str(serving_python), "-c", code], env=env, text=True, stderr=subprocess.STDOUT
    )
    return json.loads(output.strip().splitlines()[-1])


def inspect_devices(serving_python, devices, *, probe=True):
    normalized = normalize_devices(devices)
    physical, mig = _available_devices()
    available = mig if normalized[0].startswith("MIG-") else physical
    missing = [item for item in normalized if item not in available]
    if missing:
        raise ValueError(f"Requested serving device does not exist: {missing}")
    info = {
        "devices": list(normalized),
        "device_kind": "mig" if normalized[0].startswith("MIG-") else "physical",
        "device_count": len(normalized),
    }
    if probe:
        probe_result = _probe_visible_devices(serving_python, normalized)
        if probe_result["visible_device_count"] != len(normalized):
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES count does not match serving.gpus: "
                f"{probe_result['visible_device_count']} != {len(normalized)}"
            )
        info.update(probe_result)
    return info


@contextmanager
def reserve_serving_devices(root, serving_python, devices):
    """Inspect devices; physical GPU sharing is allowed, MIG remains exclusive."""
    normalized = normalize_devices(devices)
    if normalized[0].startswith("MIG-"):
        with ExitStack() as stack:
            lock_paths = []
            for device in sorted(normalized):
                digest = sha256(device.encode()).hexdigest()[:24]
                path = Path(root) / "state" / f"serving-gpu-{digest}.lock"
                stack.enter_context(file_lock(path, nonblocking=True))
                lock_paths.append(str(path))
            info = inspect_devices(serving_python, devices)
            if any(row["process_count"] for row in info["device_memory"]):
                raise RuntimeError("Requested MIG instance is occupied; choose an idle instance")
            info["lock_paths"] = lock_paths
            yield info
        return

    # Physical GPU processes may coexist; vLLM performs the actual memory admission.
    info = inspect_devices(serving_python, devices)
    info["lock_paths"] = []
    yield info

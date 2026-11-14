"""MIG admission for LoRA, without importing Torch or an inference runtime."""
import re
import subprocess
import xml.etree.ElementTree as ET

from runtime.serving.resources import normalize_devices


def inspect_mig(uuid, *, check_idle=True):
    normalize_devices([uuid])
    listing = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=15)
    parent = None
    selected = None
    for line in listing.splitlines():
        match = re.match(r"GPU (\d+):", line)
        if match:
            parent = int(match.group(1))
        match = re.search(r"MIG\s+(\S+)\s+Device\s+(\d+):\s+\(UUID:\s+(MIG-[\da-fA-F-]{36})\)", line)
        if match and match.group(3) == uuid:
            selected = (parent, match.group(1), int(match.group(2)))
    if selected is None or selected[0] is None:
        raise ValueError(f"Configured MIG instance does not exist: {uuid}")
    parent, profile, index = selected
    xml = subprocess.check_output(["nvidia-smi", "-q", "-x", "-i", str(parent)], text=True, timeout=15)
    gpu = ET.fromstring(xml).find("gpu")
    if gpu is None:
        raise RuntimeError("Missing MIG parent inventory")
    devices = [d for d in gpu.findall("mig_devices/mig_device") if d.findtext("index") == str(index)]
    if len(devices) != 1:
        raise RuntimeError("Configured MIG index is absent or ambiguous")
    instance = devices[0]
    def memory(field):
        value = instance.findtext("fb_memory_usage/" + field, "")
        if not re.fullmatch(r"\d+ MiB", value):
            raise RuntimeError("MIG memory inventory is unavailable")
        return int(value.split()[0])
    used, free = memory("used"), memory("free")
    if check_idle and used > 512:
        raise RuntimeError(f"Configured MIG instance {uuid} is not idle: {used} MiB used")
    return {"device_kind": "mig", "physical_gpu": parent, "uuid": uuid,
            "name": gpu.findtext("product_name"), "mig_profile": profile, "mig_index": index,
            "gpu_instance_id": int(instance.findtext("gpu_instance_id")),
            "compute_instance_id": int(instance.findtext("compute_instance_id")),
            "total_MiB": memory("total"), "initial_used_MiB": used, "initial_free_MiB": free}

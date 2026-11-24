"""Content-bound external renderer at the shared container boundary."""
import json
from pathlib import Path

from core.io import sha256

KEY = "DYCON_SWEBENCH_GHOSTSCRIPT"
VALUE = "9.53.3"
PREFIX = "/opt/dycon-tools/ghostscript"
DIGEST = "d9570a94236e25f7ce2024480c67c55748cb07661a040bb9642f5c87efea1801"


def validate(instance, overrides):
    value = overrides.get("variables", {}).get(KEY)
    if value is None:
        return
    if (value != VALUE or instance["instance_id"] != "matplotlib__matplotlib-20488"
            or instance["repo"] != "matplotlib/matplotlib"
            or (instance.get("environment_overrides") or {}).get("variables", {}).get(KEY) != value):
        raise ValueError("Pinned Ghostscript requires explicit matplotlib-20488 configuration")


def mounts(cfg):
    if KEY not in cfg["environment"]["variables"]:
        return {}
    root = Path(cfg["project_root"]) / "runtime/resources/swebench-ghostscript-9.53.3"
    manifest = root / "dycon-files.json"
    if sha256(manifest) != DIGEST:
        raise ValueError("Ghostscript manifest changed")
    for name, digest in json.loads(manifest.read_text()).items():
        file = root / name
        if not file.resolve().is_relative_to(root.resolve()) or sha256(file) != digest:
            raise ValueError("Ghostscript resource changed")
    return {str(root): {"bind": PREFIX, "mode": "ro"}}

def prepare(env):
    if KEY not in env:
        return []
    return [f"ln -s {PREFIX}/bin/gs /usr/local/bin/gs", "test \"$(gs --version)\" = 9.53.3"]

def variables(env):
    if KEY not in env:
        return {}
    return {
        "GS_LIB": PREFIX + "/share/ghostscript/9.53.3/Resource/Init:" + PREFIX + "/share/ghostscript/9.53.3/lib",
    }

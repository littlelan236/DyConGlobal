"""Install project-owned fork Python sources over the matching native wheel."""

import hashlib
import argparse
import importlib.metadata
import json
import shutil
import sysconfig
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--environment", choices=("easysteer", "latent-vllm"), default="easysteer")
args = parser.parse_args()
if importlib.metadata.version("vllm") != "0.29.0":
    raise RuntimeError("Overlay requires the pinned vLLM 0.29.0 wheel")
site = Path(sysconfig.get_paths()["purelib"]) / "vllm"
if not site.resolve().is_relative_to((root / ".venvs" / args.environment).resolve()):
    raise RuntimeError("Refusing to modify vLLM outside the explicitly selected project environment")
source = root / "vendor/easysteer-vllm/vllm"
if args.environment == "latent-vllm":
    # Offline installation consumes the serving-owned pin, never regenerates it.
    manifest_path = root / "vendor/easysteer-runtime.json"
    manifest_bytes = manifest_path.read_bytes()
    pin = json.loads(manifest_bytes)
    for name, digest in pin["source_files_sha256"].items():
        if not name.endswith(".py"): continue
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Vendor source differs from current runtime pin: {name}")
    changed = []
    for name, digest in pin["source_files_sha256"].items():
        if not name.endswith(".py"): continue
        target = site / name
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target); changed.append(name)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest: raise RuntimeError(f"Installed hash mismatch: {name}")
    if manifest_path.read_bytes() != manifest_bytes: raise RuntimeError("Runtime pin changed during offline synchronization; rerun after writer finishes")
    print("Synchronized pinned latent-vllm overlay:", json.dumps(changed))
    raise SystemExit(0)
shutil.copytree(source, site, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
files = {
    p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*.py")
}
(root / "vendor/easysteer-runtime.json").write_text(
    json.dumps(
        {
            "vllm_version": "0.29.0",
            "fork_commit": "7837b07d21c0ddffd043a30001f2c5654290aec7",
            "easysteer_commit": "baecd9698f32aadef68b41536f3db477b513a0d7",
            "wheel_basis": "98dff2a81d747d1dba01a47f939f48c3526d4206",
            "source_files_sha256": files,
            "overlay_path": str(site),
        },
        indent=2,
    )
    + "\n"
)
print("Installed locked fork overlay:", site)

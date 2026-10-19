"""Runtime isolation, identity and native eager engine construction."""

import os
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

from core.io import hash_json, read_json, sha256
from core.locking import file_lock


def serving_interpreter(cfg):
    """Retain the venv path; resolving a Python symlink loses venv identity.

    Read the same authoritative config/overrides without changing legacy
    serving resolution or snapshot hashes.
    """
    value = (Path(cfg["project_root"]) / cfg["serving"]["python"]).absolute()
    if cfg.get("config_source_sha256"):
        return value
    # Old snapshots resolved venv symlinks. Recover only the pinned local venv
    # with the very same interpreter; never consult today's editable config.
    pinned = Path(cfg["project_root"]) / ".venvs/easysteer/bin/python"
    if pinned.exists() and value.exists() and pinned.samefile(value):
        return pinned
    return value


@contextmanager
def gpu_reservation(root, gpus, *, allow_shared=False, min_free_bytes=0):
    if not gpus or len(set(gpus)) != len(gpus) or any(type(g) is not int or g < 0 for g in gpus):
        raise ValueError("Explicit unique physical GPUs are required for vLLM extraction")
    with ExitStack() as stack:
        for gpu in sorted(gpus):
            stack.enter_context(file_lock(Path(root) / f"state/latent-gpu-{gpu}.lock", nonblocking=True))
        inventory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
        )
        uuids = {
            int(line.split(",")[0]): line.split(",")[1].strip()
            for line in inventory.splitlines()
            if line.strip()
        }
        if not set(gpus) <= set(uuids):
            raise ValueError("Requested extraction GPU does not exist")
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True
        )
        if allow_shared:
            if min_free_bytes <= 0:
                raise ValueError("Shared GPU admission requires a positive free-memory floor")
            memory = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True
            )
            free = {
                int(row.split(",")[0]): int(row.split(",")[1].strip()) * 1024**2
                for row in memory.splitlines()
                if row.strip()
            }
            if any(free[g] < min_free_bytes for g in gpus):
                raise RuntimeError("Insufficient free GPU memory for bounded shared replay")
        elif any(line.split(",")[0].strip() in {uuids[g] for g in gpus} for line in apps.splitlines()):
            raise RuntimeError("Extraction GPUs are occupied; existing processes will not be interrupted")
        yield


def configure_environment(cfg):
    options = cfg["hidden"]["vllm"]
    expected = Path(options["python"]).absolute().parent.parent
    if Path(sys.prefix).absolute() != expected:
        raise RuntimeError("Launch extraction with hidden.vllm.python in its isolated environment")
    if expected == serving_interpreter(cfg).parent.parent:
        raise ValueError("Latent extraction must not use the rollout/steering Python environment")
    if "torch" in sys.modules or "vllm" in sys.modules:
        raise RuntimeError("GPU isolation must be configured before importing torch/vLLM")
    path = os.environ.get("PATH", "")
    isolated_bin = str(Path(options["python"]).absolute().parent)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=",".join(map(str, options["gpus"])),
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_PLUGINS="",
        # The pinned runtime rejects batch-invariant mode for Qwen GDN.
        VLLM_BATCH_INVARIANT="0",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_CACHE_ROOT=str(Path(cfg["project_root"]) / "runtime/cache/latent-vllm"),
        TRITON_CACHE_DIR=str(Path(cfg["project_root"]) / "runtime/cache/latent-vllm/triton"),
        FLASHINFER_WORKSPACE_BASE=str(Path(cfg["project_root"]) / "runtime/cache/latent-vllm/flashinfer"),
        TOKENIZERS_PARALLELISM="false",
        PATH=isolated_bin + os.pathsep + "/usr/local/cuda/bin" + os.pathsep + path,
    )


def verify_runtime(root):
    import importlib.metadata

    import vllm

    pin = read_json(Path(root) / "vendor/easysteer-runtime.json")
    if importlib.metadata.version("vllm") != pin["vllm_version"]:
        raise ValueError("Latent vLLM version differs from the pinned runtime")
    installed = Path(vllm.__file__).parent
    # The copied runtime is checked against the existing source inventory;
    # no modifications or second copy of the pin manifest is needed.
    actual = {}
    for name, digest in pin["source_files_sha256"].items():
        if name.endswith(".py"):
            path = installed / name
            if not path.is_file() or sha256(path) != digest:
                raise ValueError(f"Latent runtime source differs from pin: {name}")
            actual[name] = digest
    import easysteer.capture.fetch
    import easysteer.capture.result

    sdk_root = Path(easysteer.capture.fetch.__file__).parent
    sdk = {p.name: sha256(p) for p in sorted(sdk_root.glob("*.py"))}
    project = Path(root)
    implementation = {
        name: sha256(project / name)
        for name in [
            "hidden/adapters/vllm.py",
            "hidden/backends/vllm/capture.py",
            "hidden/backends/vllm/engine.py",
            "hidden/extract.py",
        ]
    }
    return {
        "name": "vllm_eager",
        "vllm_version": pin["vllm_version"],
        "runtime_source_sha256": hash_json(actual),
        "sdk_sha256": sdk,
        "implementation_sha256": implementation,
        "execution": "eager",
        "capture_stream": "tact_states",
    }


def engine_args(cfg):
    v = cfg["hidden"]["vllm"]
    if not v["enable_prefix_caching"]:
        raise ValueError("Offline replay requires hidden.vllm.enable_prefix_caching=true")
    if len(v["gpus"]) != v["tensor_parallel_size"]:
        raise ValueError("One explicit GPU is required per TP rank")
    args = dict(
        model=cfg["model"]["path"],
        dtype=cfg["hidden"]["dtype"],
        trust_remote_code=cfg["model"]["trust_remote_code"],
        tensor_parallel_size=v["tensor_parallel_size"],
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=v["enable_prefix_caching"],
        enable_chunked_prefill=v["enable_chunked_prefill"],
        max_num_seqs=v["max_num_seqs"],
        max_num_batched_tokens=v["max_num_batched_tokens"],
        max_model_len=v["max_model_len"],
        gpu_memory_utilization=v["gpu_memory_utilization"],
        language_model_only=True,
        generation_config="vllm",
        seed=cfg["generation"]["seed"],
    )
    if v["moe_backend"] != "auto":
        args["kernel_config"] = {"moe_backend": v["moe_backend"]}
    if v.get("kv_cache_memory_bytes") is not None:
        args["kv_cache_memory_bytes"] = v["kv_cache_memory_bytes"]
    return args

"""Checkpoint identity uses metadata and all weight shards, never an old project manifest."""

from pathlib import Path

from core.io import hash_json, read_json, sha256, write_json


def model_identity_verification_enabled(cfg):
    """Return whether rollout startup/attempts should verify model identity."""
    return cfg.get("model", {}).get("verify_identity", True)


def describe(path):
    path = Path(path).resolve()
    if not (path / "config.json").is_file():
        raise ValueError("Checkpoint config.json missing")
    files = sorted(
        {
            *path.glob("*.safetensors"),
            *path.glob("pytorch_model*.bin"),
            *path.glob("*.json"),
            *path.glob("*.jinja"),
            *path.glob("*.model"),
            *path.glob("merges.txt"),
            *path.glob("vocab.txt"),
            *path.glob("*.py"),
        }
    )
    if not any(p.suffix in {".safetensors", ".bin"} for p in files):
        raise ValueError("Checkpoint has no model weights")
    inventory = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files}
    return {"fingerprint": hash_json(inventory), "files": inventory}


def verify_model(cfg, *, deep=True):
    from core.config import require_model

    require_model(cfg)
    checkpoint = Path(cfg["model"]["path"])
    identity_path = (
        Path(cfg["project_root"]) / "runtime/model-identities" / (cfg["model"]["fingerprint"] + ".json")
    )
    if not identity_path.is_file():
        raise ValueError("Run model-register to fingerprint this checkpoint before capture/serving")
    identity = read_json(identity_path)
    if (
        identity["fingerprint"] != cfg["model"]["fingerprint"]
        or hash_json(identity["files"]) != identity["fingerprint"]
    ):
        raise ValueError("Checkpoint identity mismatch")
    import re
    if not identity["files"] or "config.json" not in identity["files"]:
        raise ValueError("Checkpoint identity inventory is incomplete")
    for name, record in identity["files"].items():
        if (Path(name).name != name or name in {".", ".."}
                or not isinstance(record.get("bytes"), int) or record["bytes"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", ""))):
            raise ValueError("Invalid checkpoint identity record")
        p = checkpoint / name
        if (not p.resolve().is_relative_to(checkpoint.resolve())
                or not p.is_file() or p.stat().st_size != record["bytes"]
                or (deep and sha256(p) != record["sha256"])):
            raise ValueError(f"Checkpoint changed: {name}")
    return identity


def register(root, path):
    identity = describe(path)
    write_json(Path(root) / "runtime/model-identities" / (identity["fingerprint"] + ".json"), identity)
    return identity["fingerprint"]

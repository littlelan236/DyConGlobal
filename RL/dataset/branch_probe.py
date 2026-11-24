"""Read-only validation of the inputs for a paired branch probe."""
from pathlib import Path

import yaml

from core.io import owned_path, read_json, sha256
from runtime.serving.parallel import artifact_contract
from .artifacts import directory
from .loader import iter_groups


def _load_config(config_path):
    with Path(config_path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)

def _selected_states(config):
    dataset = directory(config["dataset_root"], "rl-dataset", config["dataset_id"])
    wanted = config["group_ids"]
    states = {}
    from .loader import native_tokens
    for row in iter_groups(dataset, with_tokens=False):
        group = row["group"]
        if group["group_id"] not in wanted:
            continue
        prompts = [native_tokens(member)[0] for member in row["members"]]
        if not prompts or any(prompt != prompts[0] for prompt in prompts[1:]):
            raise ValueError(f"Group has no shared native prompt: {group['group_id']}")
        states[group["group_id"]] = {
            "group_id": group["group_id"], "agent_step": group["agent_step"],
            "source_run_id": group["source_run_id"], "session_id": group["session_id"],
            "decision_id": group["decision_id"], "message_index": group["message_index"],
            "prompt_sha256": group["prompt_sha256"],
            "prompt_token_ids": prompts[0], "historical_branches": len(prompts),
        }
    if set(wanted) != states.keys():
        raise ValueError("Requested groups are absent or not data-ready")
    return [states[group_id] for group_id in wanted]

def build_probe_manifest(config_path, output_path):
    config = _load_config(config_path)
    states = _selected_states(config)
    manifest = {"schema_version": 1, "source": {"dataset_id": config["dataset_id"],
        "dataset_receipt_sha256": sha256(directory(config["dataset_root"], "rl-dataset", config["dataset_id"]) / "receipt.json")},
        "sampling": config.get("sampling", {"branches": 4, "seed": 0}),
        "states": states}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(__import__("json").dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return {"status": "manifest_written", "path": str(output), "states": len(states),
            "branches_per_checkpoint": manifest["sampling"]["branches"]}

def preflight(config_path):
    config = _load_config(config_path)
    dataset = directory(config["dataset_root"], "rl-dataset", config["dataset_id"])
    run = Path(config["estimator_run"])
    snapshot = read_json(run / "resolved-config.json")
    parallel = snapshot["parallel"]
    contract = artifact_contract(parallel["estimator_ref"], parallel["estimator_sha256"])
    models = []
    for step in (450, 1240):
        path = Path(config["models"][step])
        receipt_path = path / "lora-export-receipt.json"
        receipt = read_json(receipt_path)
        if receipt["global_step"] != step:
            raise ValueError(f"Export receipt does not identify step {step}")
        for name in ("config.json", "tokenizer.json"):
            if not owned_path(path, name).is_file():
                raise ValueError(f"Missing {name} for step {step}")
        model = read_json(path / "config.json")
        text_config = model.get("text_config", model)
        if text_config["hidden_size"] != contract["hidden_size"]:
            raise ValueError(f"Predictor hidden width differs for step {step}")
        if max(contract["sites"]) >= text_config["num_hidden_layers"]:
            raise ValueError(f"Predictor layer outside model for step {step}")
        models.append({
            "step": step,
            "export_receipt_sha256": sha256(receipt_path),
            "base_matches_predictor": receipt["base_fingerprint"] == contract["model_fingerprint"],
            "tokenizer_sha256": sha256(path / "tokenizer.json"),
        })
    if models[0]["tokenizer_sha256"] != models[1]["tokenizer_sha256"]:
        raise ValueError("Checkpoint tokenizers differ")

    wanted = config["group_ids"]
    if not isinstance(wanted, list) or not wanted or len(set(wanted)) != len(wanted):
        raise ValueError("group_ids must contain distinct dataset group IDs")
    states = {}
    from .loader import native_tokens
    for row in iter_groups(dataset, with_tokens=False):
        group = row["group"]
        group_id = group["group_id"]
        if group_id not in wanted:
            continue
        prompts = [native_tokens(member)[0] for member in row["members"]]
        if not prompts or any(prompt != prompts[0] for prompt in prompts[1:]):
            raise ValueError(f"Group has no shared native prompt: {group_id}")
        states[group_id] = {
            "group_id": group_id, "agent_step": group["agent_step"],
            "source_run_id": group["source_run_id"], "prompt_sha256": group["prompt_sha256"],
            "prompt_tokens": len(prompts[0]), "historical_branches": len(prompts),
        }
        if len(states) == len(wanted):
            break
    if set(wanted) != states.keys():
        raise ValueError("Requested groups are absent or not data-ready")
    return {
        "status": "preflight_verified",
        "dataset_receipt_sha256": sha256(dataset / "receipt.json"),
        "estimator_sha256": parallel["estimator_sha256"],
        "estimator_endpoint": contract["endpoint"], "models": models,
        "states": [states[group_id] for group_id in wanted],
    }

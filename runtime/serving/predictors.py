"""Publish immutable online predictors from existing checkpoints; never train."""

from __future__ import annotations

import importlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np

from core.io import hash_json, identifier, now, owned_path, read_json, sha256, write_json
from core.locking import file_lock
from runtime.serving.parallel import artifact_contract

RIDGE_LOADER = "runtime.serving.ridge_provider:load"
TARGETS = {
    "reasoning_length": "submission_remaining_reasoning_tokens_log1p",
    "remaining_steps": "submission_remaining_steps_linear",
}


def _experiment_checkpoint(directory, target, fit_scope, fingerprint):
    experiment = read_json(directory / "experiment.json")
    selected = read_json(directory / "selected.json")
    metrics = read_json(directory / "metrics.json")
    if experiment["config"]["model"]["fingerprint"] != fingerprint:
        raise ValueError("Training experiment checkpoint identity differs from rollout model")
    trials = [json.loads(line) for line in
              (directory / "trials.jsonl").read_text().splitlines() if line.strip()]
    candidates = [t for t in trials if t["target"] == target and t["status"] == "complete"
                  and t.get("validation", {}).get("r2") is not None
                  and math.isfinite(t["validation"]["r2"])]
    if not candidates:
        raise ValueError("No finite validation R2 for predictor target")
    best = max(t["validation"]["r2"] for t in candidates)
    tolerance = selected["tie_tolerance"]
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid model-selection tolerance")
    winner = min((t for t in candidates if best - t["validation"]["r2"] <= tolerance),
                 key=lambda t: (-t["alpha"], t["layer"]))
    if selected["selection_metric"] != "validation_pooled_r2" or selected["targets"][target] != winner:
        raise ValueError("Saved champion disagrees with validation-only selection")
    if sha256(owned_path(directory, winner["model"])) != winner["model_sha256"]:
        raise ValueError("Selected training checkpoint hash changed")
    if fit_scope not in {"refit", "train"}:
        raise ValueError("Unknown predictor fit scope")
    if metrics[target]["selected"] != winner:
        raise ValueError("Metrics and selected checkpoint disagree")
    model = owned_path(directory, metrics[target]["refit_model"] if fit_scope == "refit" else winner["model"])
    scope = "train_validation" if fit_scope == "refit" else "train"
    with np.load(model, allow_pickle=False) as data:
        expected = {"target": target, "layer": winner["layer"], "alpha": winner["alpha"],
                    "fit_scope": scope, "split_sha256": experiment["split_sha256"],
                    "dataset_receipt_sha256": experiment["input"]["receipt_sha256"]}
        if any(key not in data or data[key].item() != value for key, value in expected.items()):
            raise ValueError("Checkpoint metadata disagrees with experiment selection/refit")
    if hash_json(read_json(directory / "split.json")) != experiment["split_sha256"]:
        raise ValueError("Frozen experimental split changed")
    return model, {
        "experiment": str(directory), "selection_metric": "validation_pooled_r2",
        "layer": winner["layer"], "alpha": winner["alpha"], "fit_scope": scope,
        "validation": winner["validation"], "test": metrics[target].get("test") if fit_scope == "refit" else None,
        "training_receipt_present": (directory / "receipt.json").is_file(),
        "training_completion": "receipt_present" if (directory / "receipt.json").is_file()
        else "numerical_artifacts_only_no_training_completion_receipt",
    }


def publish_predictor(cfg, predictor_id, *, experiment=None, checkpoint=None,
                      target="reasoning_length", fit_scope="refit", loader=RIDGE_LOADER,
                      sites=None, hidden_size=None, endpoint="submission"):
    """Expose provider/checkpoint selection through one CLI and the existing contract.

    Experiment selection uses validation R2, then the requested fit scope.
    An explicit checkpoint and module:callable allow alternative estimators.
    Score units remain those of the trained target (log1p for reasoning length).
    """
    if (experiment is None) == (checkpoint is None):
        raise ValueError("Choose one experiment or checkpoint")
    target = TARGETS.get(target, target)
    if not cfg["model"]["fingerprint"]:
        raise ValueError("Online predictor requires a rollout checkpoint identity")
    root = Path(cfg["project_root"])
    provenance = {"selection": "explicit_checkpoint"}
    evidence = []
    if experiment is not None:
        if loader != RIDGE_LOADER:
            raise ValueError("Automatic Ridge experiment selection requires the Ridge provider")
        directory = (root / experiment).resolve()
        checkpoint, provenance = _experiment_checkpoint(directory, target, fit_scope, cfg["model"]["fingerprint"])
        provenance["selection"] = "validation_best"
        evidence = [directory / name for name in
                    ["experiment.json", "selected.json", "metrics.json", "trials.jsonl", "split.json"]]
    checkpoint = (root / checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    modules = [loader.partition(":")[0]]
    if loader == RIDGE_LOADER:
        from difficulty.models.ridge_grid import load_model

        model = load_model(checkpoint)
        with np.load(checkpoint, allow_pickle=False) as data:
            if "target" in data and data["target"].item() != target:
                raise ValueError("Checkpoint target differs from requested target")
            if "layer" in data:
                layer = int(data["layer"].item())
                if sites is not None and sites != [layer]:
                    raise ValueError("Requested public site differs from trained checkpoint")
                sites = [layer]
            for name in ["alpha", "fit_scope", "native_site"]:
                if name in data:
                    provenance[name] = data[name].item()
        if hidden_size is not None and hidden_size != len(model["coef"]):
            raise ValueError("Requested hidden width differs from trained checkpoint")
        hidden_size = len(model["coef"])
        modules += ["difficulty.models.ridge_grid"]
    if not sites or hidden_size is None:
        raise ValueError("Provider needs explicit public sites and hidden width")
    if ":" not in loader or not callable(getattr(importlib.import_module(modules[0]), loader.split(":", 1)[1], None)):
        raise ValueError("Predictor loader must identify an importable callable")
    implementations = {name: Path(importlib.import_module(name).__file__) for name in modules}
    predictor_id = identifier(predictor_id)
    destination = root / "runtime/parallel-artifacts" / predictor_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(root / "state" / f"predictor-{predictor_id}.lock"):
        if destination.exists():
            raise FileExistsError(destination)
        with tempfile.TemporaryDirectory(prefix=".write-", dir=destination.parent) as scratch:
            stage = Path(scratch) / "artifact"
            stage.mkdir()
            model_file = "model" + checkpoint.suffix
            shutil.copyfile(checkpoint, stage / model_file)
            for path in evidence:
                shutil.copyfile(path, stage / ("source-" + path.name))
            for name, path in implementations.items():
                shutil.copyfile(path, stage / (name + ".source"))
            value = {"schema_version": 1, "loader": loader,
                     "model_fingerprint": cfg["model"]["fingerprint"], "sites": sites,
                     "hidden_size": hidden_size, "target": target, "endpoint": endpoint,
                     "score_space": "log1p_tokens" if target.endswith("_log1p") else "trained_target",
                     "model_file": model_file, "created_at": now(),
                     "implementation_sha256": {name: sha256(path) for name, path in implementations.items()},
                     "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": sha256(checkpoint),
                     "provenance": provenance,
                     "files": {p.name: sha256(p) for p in stage.iterdir()}}
            write_json(stage / "manifest.json", value)
            contract = artifact_contract(stage / "manifest.json", model_fingerprint=cfg["model"]["fingerprint"])
            module, name = loader.split(":")
            predictor = getattr(importlib.import_module(module), name)(stage, contract)
            prediction = np.asarray(predictor.predict(np.zeros((2, len(sites), hidden_size))), dtype=float)
            if prediction.shape != (2,) or not np.isfinite(prediction).all():
                raise ValueError("Provider failed finite batched prediction validation")
            stage.rename(destination)
    return {"manifest": str(destination / "manifest.json"), "manifest_sha256": sha256(destination / "manifest.json"),
            "loader": loader, "target": target, "sites": sites, "provenance": provenance}

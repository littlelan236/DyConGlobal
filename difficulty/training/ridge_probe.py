"""Frozen task split, single-site Ridge grid, model selection and artifacts."""

import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from core.config import assert_frozen
from core.io import artifact_dir, hash_json, now, sha256, write_json
from core.locking import exclusive_artifact
from difficulty.datasets.submission import load_selection, prepare_cache
from difficulty.models.estimators import inverse_target, transform_target
from difficulty.models.ridge_grid import RidgeGrid, load_model, predict_model, save_model
from difficulty.probe_config import RidgeProbe
from difficulty.training.split import make_split

LABELS = {"submission_remaining_reasoning_tokens_log1p": "remaining_reasoning_tokens_to_submission",
          "submission_remaining_steps_linear": "remaining_steps_to_submission"}


def report(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if y.ndim != 1 or prediction.shape != y.shape:
        raise ValueError("Metric arrays must be matching vectors")
    result = {"count": len(y), "r2": None, "mae": None, "rmse": None}
    if len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        result["reason"] = "empty_or_nonfinite"
        return result
    with np.errstate(over="ignore", invalid="ignore"):
        error = y - prediction
        mse = np.mean(error**2)
        variance = np.sum((y - y.mean())**2)
        r2 = 1 - np.sum(error**2) / variance if len(y) >= 2 and variance > 0 else None
        mae, rmse = float(np.mean(np.abs(error))), float(np.sqrt(mse))
    result.update(r2=float(r2) if r2 is not None and np.isfinite(r2) else None,
                  mae=mae if np.isfinite(mae) else None, rmse=rmse if np.isfinite(rmse) else None)
    if result["r2"] is None:
        result["reason"] = "insufficient_count_zero_variance_or_overflow"
    return result


def select_best(trials, tolerance):
    candidates = [t for t in trials if t["status"] == "complete" and t["validation"]["r2"] is not None]
    if not candidates:
        raise ValueError("No finite validation R² candidate")
    best = max(t["validation"]["r2"] for t in candidates)
    tied = [t for t in candidates if best - t["validation"]["r2"] <= tolerance]
    return min(tied, key=lambda t: (-t["alpha"], t["layer"]))


def grouped_reports(samples, y, prediction):
    output = {}
    for field in ("repo", "source_kind", "task_group"):
        groups = defaultdict(list)
        for i, sample in enumerate(samples):
            groups[sample[field]].append(i)
        output[field] = {}
        for key, indices in sorted(groups.items()):
            value = report(y[indices], prediction[indices])
            value["task_count"] = len({samples[i]["task_group"] for i in indices})
            output[field][key] = value
    tasks = list(output["task_group"].values())
    valid_r2 = [t["r2"] for t in tasks if t["r2"] is not None]
    output["task_macro"] = {
        "r2": float(np.mean(valid_r2)) if valid_r2 else None,
        "r2_tasks": len(valid_r2), "r2_excluded_tasks": len(tasks) - len(valid_r2),
        "mae": float(np.mean([t["mae"] for t in tasks])) if all(t["mae"] is not None for t in tasks) else None,
        "rmse": float(np.mean([t["rmse"] for t in tasks])) if all(t["rmse"] is not None for t in tasks) else None,
    }
    return output


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def code_evidence(root):
    names = ["core/config.py", "core/cli.py", "difficulty/probe_config.py",
             "difficulty/datasets/submission.py", "difficulty/training/split.py",
             "difficulty/models/estimators.py", "difficulty/models/ridge_grid.py",
             "difficulty/training/ridge_probe.py"]
    return {name: sha256(root / name) for name in names}


def draw_plots(output, settings, trials, champions):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = output / "plots"
    folder.mkdir()
    for target in settings.targets:
        grid = np.full((len(settings.layers), len(settings.alphas)), np.nan)
        for trial in trials:
            if trial["target"] == target and trial["validation"]["r2"] is not None:
                grid[settings.layers.index(trial["layer"]), settings.alphas.index(trial["alpha"])] = trial["validation"]["r2"]
        fig, ax = plt.subplots(figsize=(12, 8))
        image = ax.imshow(grid, aspect="auto", interpolation="nearest")
        ax.set(xticks=range(len(settings.alphas)), xticklabels=[f"{a:g}" for a in settings.alphas],
               yticks=range(len(settings.layers)), yticklabels=settings.layers,
               xlabel="alpha (sum squared loss)", ylabel="public site", title=target + " validation R²")
        ax.tick_params(axis="x", rotation=60)
        fig.colorbar(image, ax=ax, label="validation pooled R²")
        fig.tight_layout()
        fig.savefig(folder / (target + "-grid.png"), dpi=160)
        plt.close(fig)
        sample_rows, y, prediction = champions[target]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y, prediction, s=4, alpha=0.25)
        bounds = [min(y.min(), prediction.min()), max(y.max(), prediction.max())]
        ax.plot(bounds, bounds, color="black", linewidth=1)
        ax.set(xlabel="target (fit space)", ylabel="prediction", title=target)
        fig.tight_layout()
        fig.savefig(folder / (target + "-validation.png"), dpi=160)
        plt.close(fig)
        by_task = defaultdict(list)
        for i, sample in enumerate(sample_rows):
            by_task[sample["task_group"]].append(i)
        # Deterministic short/median/long validation trajectories, independent of errors.
        tasks = sorted(by_task, key=lambda k: (len(by_task[k]), k))
        chosen = sorted({tasks[0], tasks[len(tasks)//2], tasks[-1]})
        fig, axes = plt.subplots(len(chosen), 1, figsize=(10, 3*len(chosen)), squeeze=False)
        for ax, task in zip(axes[:, 0], chosen):
            indices = sorted(by_task[task], key=lambda i: sample_rows[i]["agent_step"])
            steps = [sample_rows[i]["agent_step"] for i in indices]
            ax.plot(steps, y[indices], label="target")
            ax.plot(steps, prediction[indices], label="prediction")
            ax.set(title=task, xlabel="canonical agent step", ylabel="fit space")
            ax.legend()
        fig.tight_layout()
        fig.savefig(folder / (target + "-trajectories.png"), dpi=160)
        plt.close(fig)


@exclusive_artifact("difficulty-ridge", "experiment_id")
def run_probe(cfg, experiment_id, *, mode="fit"):
    assert_frozen(cfg)
    if mode not in {"fit", "dry-run", "check-data"}:
        raise ValueError("Unknown probe mode")
    settings_raw = cfg["difficulty"].get("ridge_probe")
    if settings_raw is None:
        raise ValueError("Configure difficulty.ridge_probe in the unique config.yaml")
    settings = RidgeProbe.model_validate(settings_raw)
    output = artifact_dir(cfg, "difficulty", experiment_id, "checkpoints")
    if output.exists():
        raise FileExistsError(f"Experiment ID already exists (including partial): {output}")
    if mode == "fit":
        import importlib.util

        if importlib.util.find_spec("matplotlib") is None:
            raise ValueError("Ridge fit/report requires the analysis environment (matplotlib missing)")
    root = Path(cfg["project_root"])
    before_code = code_evidence(root)
    rollouts, samples, evidence = load_selection(cfg, settings)
    split = make_split(rollouts, samples, settings.validation_tasks, settings.split_seed)
    split["dataset_receipt_sha256"] = settings.dataset_receipt_sha256
    roles = {task["rollout_id"]: task["split"] for task in split["tasks"]}
    split["distributions"] = {}
    for role in ("train", "validation"):
        tasks = [t for t in split["tasks"] if t["split"] == role]
        rows = [s for s in samples if roles[s["rollout_id"]] == role]
        split["distributions"][role] = {
            "repo_tasks": dict(Counter(t["repo"] for t in tasks)),
            "source_tasks": dict(Counter(t["source_kind"] for t in tasks)),
            "source_samples": dict(Counter(s["source_kind"] for s in rows)),
            "canonical_steps": {r["rollout_id"]: r["canonical_valid_assistant_steps"] for r in rollouts
                                if roles[r["rollout_id"]] == role},
        }
    split_sha = hash_json(split)
    experiment = {"schema_version": 1, "mode": mode, "created_at": now(), "fit_scope": "train",
                  "config": cfg, "settings": settings.model_dump(), "input": evidence,
                  "split_sha256": split_sha, "code": before_code,
                  "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
                  "r2_interpretation": "selection validation score, not independent test R²"}
    output.mkdir(parents=True)
    write_json(output / "experiment.json", experiment)
    write_json(output / "split.json", split)
    if mode == "dry-run":
        finish(output, experiment, root, before_code)
        return {"path": str(output), "mode": mode, "task_counts": split["task_counts"],
                "sample_counts": split["sample_counts"]}
    cache = prepare_cache(cfg, settings, samples)
    write_json(output / "data-check.json", {**evidence, "cache": str(cache),
               "cache_manifest_sha256": sha256(cache / "cache.json"),
               "verified_shape": [len(samples), 32, 4096], "sites": list(range(32)),
               "labels": "common submitted, uncensored samples", "checked_at": now()})
    if mode == "check-data":
        finish(output, experiment, root, before_code)
        return {"path": str(output), "mode": mode, "samples": len(samples), "cache": str(cache)}
    train_indices = np.array([i for i, s in enumerate(samples) if roles[s["rollout_id"]] == "train"])
    val_indices = np.array([i for i, s in enumerate(samples) if roles[s["rollout_id"]] == "validation"])
    validation = [samples[i] for i in val_indices]
    raw = np.column_stack([[s[LABELS[target]] for s in samples] for target in settings.targets])
    y = np.column_stack([transform_target(raw[:, i], target) for i, target in enumerate(settings.targets)])
    yt, yv = y[train_indices], y[val_indices]
    trials = []
    for layer in settings.layers:
        matrix = np.load(cache / f"layer-{layer:02d}.f32.npy", mmap_mode="r", allow_pickle=False)
        xt, xv = matrix[train_indices].astype(np.float64), matrix[val_indices].astype(np.float64)
        del matrix
        grid = RidgeGrid(xt, yt)
        zt, zv = grid.scaler.transform(xt), grid.scaler.transform(xv)
        for alpha in settings.alphas:
            coef, intercept = grid.coefficients(alpha)
            pt, pv = zt @ coef + intercept, zv @ coef + intercept
            for i, target in enumerate(settings.targets):
                trial = {"target": target, "layer": layer, "alpha": alpha,
                         "train": report(yt[:, i], pt[:, i]), "validation": report(yv[:, i], pv[:, i]),
                         "status": "complete", "fit_scope": "train"}
                if not np.isfinite(coef[:, i]).all() or not np.isfinite(intercept[i]) or any(
                    trial[role][metric] is None for role in ("train", "validation") for metric in ("mae", "rmse")
                ):
                    trial.update(status="nonfinite_failure")
                else:
                    model = {"coef": coef[:, i], "intercept": intercept[i],
                             "feature_mean": grid.scaler.mean, "feature_scale": grid.scaler.scale}
                    relative = f"models/{target}/site-{layer:02d}/alpha-{alpha:.17g}.npz"
                    save_model(output / relative, model, {"target": target, "layer": layer, "alpha": alpha,
                               "native_site": layer if layer < 31 else 32, "split_sha256": split_sha,
                               "dataset_receipt_sha256": settings.dataset_receipt_sha256,
                               "train_sample_ids_sha256": hash_json(split["sample_ids"]["train"]),
                               "fit_scope": "train", "loss": "sum_squared_error; intercept_unpenalized"})
                    trial.update(model=relative, model_sha256=sha256(output / relative))
                trials.append(trial)
        write_jsonl(output / "trials.jsonl", trials)
        print(f"ridge grid: layer {layer}, {len(trials)} candidates recorded", flush=True)
        del grid, xt, xv, zt, zv
    expected = len(settings.layers) * len(settings.alphas) * len(settings.targets)
    if len(trials) != expected:
        raise ValueError("Grid incomplete")
    winners = {target: select_best([t for t in trials if t["target"] == target], settings.tie_tolerance)
               for target in settings.targets}
    write_json(output / "selected.json", {"selection_metric": settings.selection_metric,
               "tie_tolerance": settings.tie_tolerance, "tie_order": "larger alpha, smaller public layer",
               "targets": winners})
    metrics = {}
    predictions = []
    champions = {}
    for i, target in enumerate(settings.targets):
        winner = winners[target]
        model = load_model(output / winner["model"])
        matrix = np.load(cache / f"layer-{winner['layer']:02d}.f32.npy", mmap_mode="r", allow_pickle=False)
        prediction = predict_model(model, matrix[val_indices].astype(float))
        del matrix
        reloaded_report = report(yv[:, i], prediction)
        if not np.isclose(reloaded_report["r2"], winner["validation"]["r2"], rtol=1e-9, atol=1e-9):
            raise ValueError("Saved model validation metrics did not round-trip")
        with np.errstate(over="ignore", invalid="ignore"):
            prediction_raw = inverse_target(prediction, target)
        metrics[target] = {"best_validation": reloaded_report, "selected": winner,
             "baseline_train_mean": report(yv[:, i], np.full(len(yv), yt[:, i].mean())),
             "validation_raw_units": report(raw[val_indices, i], prediction_raw),
             "raw_prediction_nonfinite_count": int((~np.isfinite(prediction_raw)).sum()),
             "groups_transformed": grouped_reports(validation, yv[:, i], prediction)}
        for j, sample in enumerate(validation):
            predictions.append({"sample_id": sample["sample_id"], "rollout_id": sample["rollout_id"],
                 "task_group": sample["task_group"], "target": target, "split": "validation",
                 "target_raw": float(raw[val_indices[j], i]), "target_transformed": float(yv[j, i]),
                 "prediction_raw": float(prediction_raw[j]) if np.isfinite(prediction_raw[j]) else None,
                 "prediction_transformed": float(prediction[j]), "layer": winner["layer"], "alpha": winner["alpha"]})
        champions[target] = validation, yv[:, i], prediction
    write_jsonl(output / "validation-predictions.jsonl", predictions)
    write_json(output / "metrics.json", metrics)
    draw_plots(output, settings, trials, champions)
    report_lines = ["# Submission remainder Ridge grid", "", "Validation is used for layer/alpha selection; these are not independent test scores.",
                    "", f"Tasks: {split['task_counts']}; samples: {split['sample_counts']}.", "",
                    "| target | public layer | alpha | train R² | best validation R² |",
                    "|---|---:|---:|---:|---:|"]
    report_lines.extend(f"| {target} | {t['layer']} | {t['alpha']:g} | {t['train']['r2']} | {t['validation']['r2']} |"
                        for target, t in winners.items())
    (output / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    finish(output, experiment, root, before_code)
    return {"path": str(output), "mode": mode, "models": expected, "selected": winners}


def finish(output, experiment, root, before_code):
    if code_evidence(root) != before_code:
        raise ValueError("Implementation changed during experiment")
    write_json(output / "receipt.json", {"schema_version": 1, "state": "complete", "mode": experiment["mode"],
               "completed_at": now(), "dataset_receipt_sha256": experiment["input"]["receipt_sha256"],
               "split_sha256": experiment["split_sha256"], "code": before_code,
               "files": {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
                         if p.is_file() and p.name != "receipt.json"}})

"""Project-level action artifacts and immutable, parent-bound plot views."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from core.io import artifact_dir, hash_json, identifier, now, owned_path, read_json, sha256, write_json
from core.locking import file_lock
from difficulty.receipts import verify_receipt


def analysis_directory(root, reference):
    reference = str(reference)
    if "/" not in reference and "\\" not in reference:
        return artifact_dir({"project_root": str(root)}, "action", reference)
    path = owned_path(root, reference)
    parts = path.relative_to(Path(root).resolve()).parts
    if not (len(parts) == 3 and parts[:2] == ("action", "outputs") or
            len(parts) == 5 and parts[0] == "runs" and parts[2:4] == ("action", "outputs")):
        raise ValueError("Use an analysis ID or explicit action/outputs or runs/RUN/action/outputs directory")
    return path


def _plot_manifest(directory, analysis, parent, *, parent_receipt_sha256=None):
    manifest = read_json(directory / "plots.json")
    if manifest["analysis_id"] != analysis["analysis_id"]:
        raise ValueError("Plot analysis identity mismatch")
    expected = {"labels_sha256": parent["files"]["labels.jsonl"], "summary_sha256": parent["files"]["summary.json"]}
    if manifest["input"] != expected:
        raise ValueError("Plot inputs disagree with frozen analysis")
    settings = manifest["settings"]
    from action.evaluation.visualize import PLOT_NAMES
    if (settings["vmin"] != 0 or settings["vmax"] != 1 or settings["colormap"] != "viridis"
            or settings["aggregation"] not in {"micro", "macro"} or not settings["names"]
            or set(settings["names"]) - set(PLOT_NAMES)):
        raise ValueError("Invalid plot settings")
    expected_files = {f"{name}.{fmt}" for name in settings["names"] for fmt in settings["formats"]}
    if set(manifest["files"]) != expected_files:
        raise ValueError("Incomplete plot output")
    for name, digest in manifest["files"].items():
        if sha256(owned_path(directory, name)) != digest:
            raise ValueError(f"Action plot changed: {name}")
    if parent_receipt_sha256 is not None:
        receipt = read_json(directory / "receipt.json")
        if (receipt.get("parent_receipt_sha256") != parent_receipt_sha256
                or set(receipt["files"]) != expected_files | {"plots.json"}):
            raise ValueError("Plot view parent or file set changed")
    return manifest


def load_local(root, reference):
    """Validate local artifact content only; drawing must never read source runs."""
    directory = analysis_directory(root, reference)
    receipt = read_json(directory / "receipt.json")
    if receipt.get("schema_version") != 2:
        raise ValueError("Plotting requires a schema-2 analysis; create a new analysis for legacy inputs")
    try:
        receipt, _ = verify_receipt(root, directory)
    except ValueError as error:
        if "Artifact checksum mismatch" in str(error):
            raise ValueError(f"Action artifact changed: {error}") from error
        raise
    analysis, selection = read_json(directory / "analysis.json"), read_json(directory / "selection.json")
    aid = directory.name
    if (analysis.get("schema_version") != 2 or analysis.get("analysis_id") != aid or analysis.get("state") != "complete"
            or receipt.get("analysis_id") != aid or receipt.get("input_identity") != analysis.get("input_identity")):
        raise ValueError("Action analysis identity mismatch")
    if hash_json(selection) != analysis["selection_sha256"] or analysis["rollouts"] != selection["sources"]:
        raise ValueError("Action selection changed")
    if analysis["source_runs"] != selection["source_runs"]:
        raise ValueError("Action source list changed")
    fixed = {"analysis.json", "selection.json", "resolved-config.json", "labels.jsonl", "summary.json", "transition.json", "examples.json"}
    required = fixed | {f"code/{name}" for name in analysis["code"]}
    required |= {source["snapshot_path"] for source in analysis["source_runs"]}
    plots = _plot_manifest(directory / "plots", analysis, receipt)
    required |= {"plots/plots.json"} | {f"plots/{name}" for name in plots["files"]}
    if set(receipt["files"]) != required:
        raise ValueError("Incomplete action receipt")
    for name, digest in analysis["code"].items():
        if sha256(owned_path(directory, "code/" + name)) != digest:
            raise ValueError(f"Action implementation snapshot changed: {name}")
    for source in analysis["source_runs"]:
        if sha256(owned_path(directory, source["snapshot_path"])) != source["snapshot_sha256"]:
            raise ValueError("Action source evidence snapshot changed")
    rows = [json.loads(line) for line in (directory / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    from action.evaluation.distributions import summarize
    summary = read_json(directory / "summary.json")
    if summary != summarize(rows, selection["sources"]) or read_json(directory / "transition.json") != summary["transition"]:
        raise ValueError("Action summary disagrees with labels")
    if read_json(directory / "resolved-config.json")["config_hash"] != analysis["config_hash"]:
        raise ValueError("Action config identity mismatch")
    return directory, analysis, selection, rows, summary, receipt


def verify(root, scope, artifact_id, **_):
    from action.analyzers.trajectories import check_analysis
    directory = analysis_directory(root, artifact_id)
    result = check_analysis({"project_root": str(root)}, artifact_id)
    return {**result, "scope": scope, "artifact_id": directory.name, "path": str(directory),
            "receipt_sha256": sha256(directory / "receipt.json")}


def show(root, scope, artifact_id, **kwargs):
    result = verify(root, scope, artifact_id, **kwargs)
    directory = Path(result["path"])
    return {**result, "files": [{"path": str(p.relative_to(directory)), "bytes": p.stat().st_size}
                               for p in sorted(directory.rglob("*")) if p.is_file()]}


def verify_views(root, directory, analysis, parent):
    views = directory / "plots/views"
    found = []
    if views.exists():
        for view in sorted(views.iterdir()):
            if view.name.startswith("."):
                continue
            identifier(view.name)
            verify_receipt(root, view)
            _plot_manifest(view, analysis, parent, parent_receipt_sha256=sha256(directory / "receipt.json"))
            found.append(view.name)
    return found


def plot_analysis(cfg, analysis_id, *, plot_id=None, names=(), aggregation="micro", formats=(), dpi=200):
    """Return default plots or create an immutable view without mutating the parent."""
    from action.evaluation.visualize import PLOT_NAMES, render_plots
    root = Path(cfg["project_root"])
    directory, analysis, _, _, summary, parent = load_local(root, analysis_id)
    names, formats = tuple(names or PLOT_NAMES), tuple(formats or ("png",))
    if not plot_id:
        if (aggregation != "micro" or formats != ("png",) or dpi != 200 or set(names) - set(PLOT_NAMES)
                or len(names) != len(set(names))):
            raise ValueError("Use --plot-id NEW_ID for a new immutable plot configuration")
        return {"analysis_id": analysis["analysis_id"], "reused": True,
                "plots": [str(directory / "plots" / f"{name}.png") for name in names]}
    plot_id = identifier(plot_id)
    out = directory / "plots/views" / plot_id
    with file_lock(root / "state/artifact-locks/action-plot" / f"{analysis['analysis_id']}.{plot_id}.lock", nonblocking=True):
        if out.exists():
            raise FileExistsError(f"Use a new action plot ID: {out}")
        parent_sha = sha256(directory / "receipt.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".action-view-", dir=out.parent) as temporary:
            stage = Path(temporary)
            plots = render_plots(stage, summary, analysis_id=analysis["analysis_id"],
                                 labels_sha256=parent["files"]["labels.jsonl"], summary_sha256=parent["files"]["summary.json"],
                                 names=names, aggregation=aggregation, formats=formats, dpi=dpi)
            if sha256(directory / "receipt.json") != parent_sha:
                raise ValueError("Parent analysis changed during plotting")
            write_json(stage / "receipt.json", {"schema_version": 1, "state": "complete", "created_at": now(),
                       "analysis_id": analysis["analysis_id"], "plot_id": plot_id, "parent_receipt_sha256": parent_sha,
                       "files": {p.name: sha256(p) for p in stage.iterdir() if p.is_file()}})
            stage.rename(out)
    return {"analysis_id": analysis["analysis_id"], "plot_id": plot_id, "path": str(out),
            "plots": [str(out / name) for name in plots["files"]]}

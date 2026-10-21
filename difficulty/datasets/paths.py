"""Resolve shared dataset IDs or explicit legacy dataset paths, without fallback."""
from pathlib import Path

from core.io import artifact_dir, owned_path


def dataset_directory(cfg, reference):
    reference = str(reference)
    if "/" not in reference and "\\" not in reference:
        return artifact_dir(cfg, "difficulty", reference, "datasets")
    path = owned_path(cfg["project_root"], reference)
    relative = path.relative_to(Path(cfg["project_root"]).resolve()).parts
    shared = len(relative) == 3 and relative[:2] == ("difficulty", "datasets")
    legacy = len(relative) == 5 and relative[0] == "runs" and relative[2:4] == ("difficulty", "datasets")
    if not shared and not legacy:
        raise ValueError("Dataset reference must name a shared dataset or an explicit legacy dataset directory")
    return path


def source_directory(cfg, index, index_directory, module):
    key, source_id = ("label_root", "dataset_id") if module == "enrichment" else ("capture_root", "capture_id")
    if index.get(key):
        return owned_path(cfg["project_root"], index[key])
    relative = Path(index_directory).relative_to(Path(cfg["project_root"]).resolve()).parts
    if relative[0] == "runs":
        category = "offline" if module == "enrichment" else f"outputs/{index['capture_id']}"
        return owned_path(cfg["project_root"], f"runs/{relative[1]}/{module}/{category}")
    # Legacy shared indexes were produced from project-level enrichment/hidden.
    return owned_path(cfg["project_root"], f"{module}/outputs/{index[source_id]}")

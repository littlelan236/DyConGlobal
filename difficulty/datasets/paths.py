"""Resolve shared dataset IDs or explicit legacy dataset paths, without fallback."""
from pathlib import Path

from core.io import artifact_roots, identifier, owned_path


def dataset_directory(cfg, reference):
    """Resolve datasets only within the explicit difficulty_dataset root or legacy project."""
    reference = str(reference)
    project = Path(cfg['project_root']).resolve()
    from difficulty.storage import active_root
    scoped = active_root(project)
    preferred = cfg.get('paths', {}).get('artifact_roots', {}).get('difficulty_dataset') or scoped
    roots = [project]
    if preferred:
        physical = Path(preferred).resolve() if scoped and preferred == scoped else artifact_roots(cfg, 'difficulty_dataset')[0]
        roots = list(dict.fromkeys([physical, project]))
    if '/' not in reference and '\\' not in reference:
        name = identifier(reference)
        candidates = [owned_path(root, f'difficulty/datasets/{name}') for root in roots]
        matches = list(dict.fromkeys(path for path in candidates if path.exists()))
        if len(matches) > 1:
            raise ValueError('Ambiguous dataset ID across configured roots: ' + name)
        return matches[0] if matches else candidates[0]
    raw = Path(reference)
    absolute = raw.resolve() if raw.is_absolute() else (project/raw).resolve()
    for root in roots:
        if absolute.is_relative_to(root):
            rel = absolute.relative_to(root).parts
            if len(rel) == 3 and rel[:2] == ('difficulty', 'datasets'):
                identifier(rel[2])
                return owned_path(root, absolute)
    path = owned_path(project, reference)
    relative = path.relative_to(project).parts
    legacy = len(relative) == 5 and relative[0] == 'runs' and relative[2:4] == ('difficulty', 'datasets')
    if not legacy:
        raise ValueError('Dataset reference must name a configured dataset or explicit legacy dataset directory')
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

"""Explicit, scoped difficulty artifact roots; never relax project source ownership."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

_active = ContextVar('difficulty_storage', default=None)


def active_root(project):
    value = _active.get()
    project = Path(project).resolve()
    return value[1] if value and value[0] == project else None


@contextmanager
def storage_scope(project, storage_root=None):
    project = Path(project).resolve()
    if storage_root is None:
        yield
        return
    raw = Path(storage_root).expanduser()
    if not raw.is_absolute() or raw == Path('/') or '..' in raw.parts:
        raise ValueError('Difficulty storage root must be absolute and non-root')
    token = _active.set((project, raw.resolve()))
    try:
        yield
    finally:
        _active.reset(token)


def scoped_storage(function):
    @wraps(function)
    def wrapped(cfg, *args, **kwargs):
        preferred = cfg.get('paths', {}).get('artifact_roots', {}).get('difficulty')
        with storage_scope(cfg['project_root'], preferred):
            return function(cfg, *args, **kwargs)
    return wrapped


def difficulty_path(project, category, artifact_id):
    from core.io import identifier, owned_path
    project = Path(project).resolve()
    physical = active_root(project)
    if category not in {'datasets', 'checkpoints', 'evaluations', 'predictions'}:
        raise ValueError('Unsupported difficulty storage category')
    rel = f'difficulty/{category}/{identifier(artifact_id)}'
    candidates = [owned_path(root, rel) for root in dict.fromkeys([physical or project, project])]
    existing = [p for p in candidates if p.exists()]
    if len(existing) > 1:
        raise ValueError('Ambiguous difficulty artifact across storage roots: ' + artifact_id)
    return existing[0] if existing else candidates[0]


def receipt_parent(project, reference):
    from core.io import owned_path
    physical = active_root(project)
    raw = Path(reference)
    if raw.is_absolute() and physical and raw.resolve().is_relative_to(physical):
        relative = raw.resolve().relative_to(physical).parts
        valid = (len(relative) == 3 and relative[0] == 'difficulty' and
                 relative[1] in {'datasets', 'checkpoints', 'evaluations', 'predictions'})
        valid_split = (len(relative) == 5 and relative[:2] == ('difficulty', 'datasets')
                       and relative[3] == 'splits')
        if valid or valid_split:
            return owned_path(physical, raw)
        raise ValueError('Invalid difficulty receipt parent path')
    return owned_path(project, reference)

"""Unified configuration entry; delegates to the existing immutable attempt services."""
from datetime import datetime, timezone
from pathlib import Path
import uuid

from core.config import assert_frozen
from core.io import identifier
from difficulty.training.config import read_config, resolve_settings
from difficulty.training.visualization import Visualization


def run(cfg, training_config='difficulty/training/config.yaml'):
    assert_frozen(cfg)
    root = Path(cfg['project_root'])
    path = (root / training_config).resolve()
    raw, config = read_config(path)
    if config is None:
        from difficulty.training.mlp_probe import run_probe
        return run_probe(cfg, 'mlp-' + uuid.uuid4().hex[:16], training_config_path=path)
    return run_config(cfg, config, raw=raw, path=path)


def run_config(cfg, config, *, raw, path):
    """The same training service for file-based CLI and typed workflow callers."""
    assert_frozen(cfg)
    root = Path(cfg['project_root'])
    attempt_id = config.run.attempt_id
    if attempt_id == 'auto':
        attempt_id = (config.model.kind.replace('_', '-') + '-' +
                      datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    identifier(attempt_id)
    output = root / 'difficulty/checkpoints' / attempt_id
    if output.exists():
        raise FileExistsError(f'Attempt ID already exists (including partial): {output}')
    settings = resolve_settings(root, config)
    visual = Visualization(cfg, attempt_id, config.tensorboard)
    if config.run.mode == 'fit':
        visual.preflight()
    print(f'Training attempt: {attempt_id}; model={config.model.kind}; mode={config.run.mode}', flush=True)
    success = False
    try:
        if config.model.kind == 'ridge':
            import os
            for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS'):
                os.environ[name] = str(settings.blas_threads)
            from difficulty.training.ridge_probe import run_probe
        else:
            from difficulty.training.mlp_probe import run_probe
        result = run_probe(cfg, attempt_id, mode=config.run.mode, settings_override=settings,
                           training_config_path=path, training_config_raw=raw,
                           visualization=visual, **({'kind': config.model.kind} if config.model.kind != 'ridge' else {}))
        success = True
        return result
    finally:
        visual.close(success=success)

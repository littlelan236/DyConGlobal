"""SimPO: independent reference, submitted outputs and newly executed outputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REFERENCE = Path(__file__).resolve().with_name('loss.npz')
REFERENCE_SHA256 = '162e3cd015afa98503a8419c6955523ba7c4401cd50f34e7d8f7e601e8489d2c'
OUTPUT = Path('/root/loss.npz')
ENVIRONMENT = Path('/root/python_info.txt')
REPOSITORY = Path('/root/SimPO')
STAGES = Path('/logs/verifier/simpo-stages.json')


class SimPOVerifierError(RuntimeError):
    """A broken protected reference is an evaluator fault, including absent submissions."""


def load_losses(path, *, reference=False):
    error = SimPOVerifierError if reference else AssertionError
    try:
        with np.load(path, allow_pickle=False) as archive:
            if 'losses' not in archive.files:
                raise ValueError("missing key 'losses'")
            losses = np.array(archive['losses'], copy=True)
        if not np.issubdtype(losses.dtype, np.number) or losses.size == 0 or not np.isfinite(losses).all():
            raise ValueError('losses must be a nonempty finite numeric array')
        return losses
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise error(f"Invalid {'reference' if reference else 'submitted'} loss file: {exc}") from exc


def compare(expected, actual):
    assert expected.shape == actual.shape, f'Shape mismatch: gt={expected.shape}, got={actual.shape}'
    assert np.allclose(expected, actual, rtol=1e-5, atol=1e-6), 'Loss mismatch'


def verify(record):
    # A known reference failure always wins over a missing-output shortcut.
    try:
        digest = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    except OSError as exc:
        raise SimPOVerifierError(f'Protected reference unavailable: {exc}') from exc
    if digest != REFERENCE_SHA256:
        raise SimPOVerifierError('Protected reference hash mismatch')
    gt = load_losses(REFERENCE, reference=True)
    record.append({'phase':'reference', 'status':'completed', 'sha256':digest})
    assert OUTPUT.is_file(), f'Missing output: {OUTPUT}'
    assert ENVIRONMENT.is_file(), f'Missing environment info: {ENVIRONMENT}'
    env_info = ENVIRONMENT.read_text()
    assert 'python 3.10' in env_info.lower(), f"Environment info does not contain 'python 3.10': {env_info}"
    compare(gt, load_losses(OUTPUT))
    record.append({'phase':'submission', 'status':'completed', 'sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest()})
    OUTPUT.unlink()
    started_ns = time.time_ns()
    env = {**os.environ, 'PYTHONPATH':str(REPOSITORY)}
    proc = subprocess.run([sys.executable,str(REPOSITORY/'unit_test/unit_test_1.py')],
        cwd=REPOSITORY, env=env, capture_output=True, text=True, timeout=600)
    record.append({'phase':'rerun','return_code':proc.returncode,'started_ns':started_ns,
                   'stdout_tail':proc.stdout[-4096:],'stderr_tail':proc.stderr[-4096:]})
    assert proc.returncode == 0, f'unit_test_1.py failed.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}'
    assert OUTPUT.is_file(), f'Missing freshly generated output: {OUTPUT}'
    # Deletion before the call establishes generation; no stale open archive survives.
    compare(gt, load_losses(OUTPUT))
    record.append({'phase':'rerun_comparison','status':'completed',
                   'sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),'mtime_ns':OUTPUT.stat().st_mtime_ns})


def test_outputs():
    record=[]
    result={'schema_version':1,'policy':'simpo-independent-reference-v1','stages':record}
    try:
        verify(record)
        result['outcome']='passed'
    except SimPOVerifierError as exc:
        result.update(outcome='verifier_error',error=str(exc))
        raise
    except (AssertionError, subprocess.TimeoutExpired) as exc:
        result.update(outcome='agent_failure',error=str(exc))
        raise
    finally:
        STAGES.parent.mkdir(parents=True,exist_ok=True)
        STAGES.write_text(json.dumps(result,indent=2))

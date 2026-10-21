"""Validate/freeze one extraction run and launch a single vLLM TP worker."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import time
from pathlib import Path

from catalog.store import Catalog
from core.config import resolve
from core.io import read_json, sha256, write_json
from core.locking import file_lock
from enrichment.scripts.run_extraction_queue import BATCH


def recover_jobs(jobs):
    recovered = []
    for job in jobs:
        error = job.get('error') or ''
        if job['status'] != 'failed' or 'extra_forbidden' not in error or 'config.py' not in error:
            continue
        job.setdefault('failure_history', []).append({
            'error': error, 'gpu': job.get('gpu'), 'pid': job.get('pid'),
            'at': job.get('finished'), 'scope': 'system', 'recovered_at': time.time(),
        })
        job.update(status='pending', error=None)
        recovered.append(job['rollout_id'])
    return recovered


def verify_completed(root, jobs):
    count = 0
    for job in jobs:
        if job['status'] != 'completed':
            continue
        name = BATCH + '-' + job['rollout_id']
        source = root / 'enrichment/outputs' / name / 'receipt.json'
        target = root / 'hidden/outputs' / name / 'receipt.json'
        enrichment, hidden = read_json(source), read_json(target)
        row = enrichment['selection']['rollouts'][job['rollout_id']]
        if row['trajectory_sha256'] != job['trajectory_sha256']:
            raise ValueError('Completed trajectory binding mismatch')
        if hidden['source_receipt_sha256'] != sha256(source):
            raise ValueError('Completed hidden source binding mismatch')
        for receipt, directory in [(enrichment, source.parent), (hidden, target.parent)]:
            for file, digest in receipt['files'].items():
                if sha256(directory / file) != digest:
                    raise ValueError('Completed artifact changed: ' + str(directory / file))
        count += 1
    return count


def _launch_locked(root):
    directory = root / 'state' / BATCH
    # Hold the configured TP group during preflight; duplicate workers are rejected.
    from contextlib import ExitStack
    with ExitStack() as locks:
        gpus = [0, 1, 2, 3]
        for gpu in gpus:
            locks.enter_context(file_lock(directory / f'gpu-{gpu}.lock', nonblocking=True))
        old = read_json(directory / 'launch.json')
        for record in old:
            proc = Path('/proc') / str(record['pid']) / 'cmdline'
            if proc.exists() and b'run_extraction_queue.py' in proc.read_bytes():
                raise RuntimeError('Old extraction worker still alive: ' + str(record['pid']))
        catalog = Catalog(root)
        print('CATALOG', catalog.verify(), flush=True)
        current = catalog.read()
        models = read_json(root / 'state/extraction-models.json')
        jobs = read_json(directory / 'queue.json')
        if any(j['status'] == 'running' for j in jobs):
            raise RuntimeError('Recover stale running jobs explicitly before launch')
        for job in jobs:
            row = current['rollouts'][job['rollout_id']]
            if row['trajectory_sha256'] != job['trajectory_sha256'] or row['model_fingerprint'] != models[job['model']]['fingerprint']:
                raise ValueError('Queue/catalog checkpoint or trajectory binding mismatch')
        print('COMPLETED_ARTIFACTS_VERIFIED', verify_completed(root, jobs), flush=True)
        configs = {}
        for name, model in models.items():
            metadata = read_json(Path(model['path']) / 'config.json')
            text = metadata.get('text_config', metadata)
            layers = list(range(text['num_hidden_layers']))
            for anchors in [['think_open'], ['think_open', 'think_close']]:
                overrides = [
                    'model.path=' + model['path'], 'model.fingerprint=' + model['fingerprint'],
                    'model.served_name=' + name, 'benchmark.partition=train', 'benchmark.purposes=[training]',
                    'enrichment.anchors=' + str(anchors), 'hidden.device=cpu', 'hidden.chunk_size=1024',
                    'hidden.layers=' + str(layers), 'hidden.adapter=vllm_eager', 'hidden.vllm.gpus=' + str(gpus),
                    'hidden.vllm.tensor_parallel_size=' + str(len(gpus)), 'capture.enabled=false',
                ]
                configs[name + '|' + ','.join(anchors)] = resolve(root / 'config.yaml', overrides)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        snapshot = directory / ('config-snapshot-' + stamp + '.json')
        write_json(snapshot, {'models': models, 'configs': configs, 'created_at': stamp})
        snapshot.chmod(0o444)
        # Archive the original queue before resetting only schema-validation failures.
        with file_lock(directory / 'queue.lock'):
            jobs = read_json(directory / 'queue.json')
            write_json(directory / ('queue-before-recovery-' + stamp + '.json'), jobs)
            recovered = recover_jobs(jobs)
            write_json(directory / 'queue.json', jobs)
        with file_lock(root / 'state/product-manifests.lock'):
            for owner in ['enrichment', 'hidden']:
                path = root / owner / 'manifest.json'
                manifest = read_json(path)
                for rid in recovered:
                    entry = manifest['extraction_jobs'][rid]
                    entry.update(status='pending', error=None, recovery_at=stamp,
                                 config_snapshot=str(snapshot.relative_to(root)))
                batch = manifest['extraction_batches'][BATCH]
                batch.update(config_snapshot=str(snapshot.relative_to(root)), restart_at=stamp,
                             recovered_system_failures=len(recovered), startup_verified=False)
                write_json(path, manifest)
        print('RECOVERED', len(recovered), 'CONFIGS_VALIDATED', len(configs), flush=True)
    # One vLLM process owns the full TP group and keeps APC warm across jobs.
    records = []
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false',
               PYTHONUNBUFFERED='1', PYTHONPATH=str(root))
    logfile = directory / 'vllm-tp.log'
    with logfile.open('a') as log:
        proc = subprocess.Popen([
            str(root / '.venvs/latent-vllm/bin/python'), '-u',
            str(root / 'enrichment/scripts/run_extraction_queue.py'),
            '--root', str(root), '--snapshot', str(snapshot), '--gpus', *map(str, gpus),
        ], cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log,
           stderr=subprocess.STDOUT, start_new_session=True)
    records.append({'gpus': gpus, 'pid': proc.pid, 'log': str(logfile), 'snapshot': str(snapshot)})
    write_json(directory / 'launch.json', records)
    print(json.dumps(records), flush=True)


def launch(root):
    with file_lock(root / 'state' / BATCH / 'launcher.lock', nonblocking=True):
        _launch_locked(root)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/home/wangtianyi/DyConGlobal'))
    a = p.parse_args()
    launch(a.root.resolve())

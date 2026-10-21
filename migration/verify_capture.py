"""Verify selected imported NAS shards and every finalized tensor member."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from application.execution.artifact import refresh
from application.execution.run_plan import load_plan
from core.io import now, owned_path, read_json, sha256, write_json


def verify_shard(job):
    nas, shard, rows = job
    path = owned_path(nas, shard)
    if {r['tensor_locator']['shard_sha256'] for r in rows} != {sha256(path)}:
        raise ValueError('NAS shard checksum mismatch: ' + shard)
    samples = 0
    with tarfile.open(path, 'r:') as archive:
        for row in rows:
            locator = row['tensor_locator']
            member = archive.getmember(locator['member'])
            if not member.isfile():
                raise ValueError('Selected tensor is not a regular member')
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != locator['member_sha256']:
                raise ValueError('Tensor member hash mismatch')
            if locator['member_sha256'] != row['sha256']:
                raise ValueError('Event and member hashes disagree')
            with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
                shapes = row['storage_shapes']
                if set(arrays.files) != set(shapes):
                    raise ValueError('Native sites differ from recorded capture')
                for name in arrays.files:
                    value = arrays[name]
                    if list(value.shape) != shapes[name] or not np.isfinite(value).all():
                        raise ValueError('Tensor shape or finite-value check failed')
                    if str(value.dtype) != row['storage_dtypes'][name]:
                        raise ValueError('Tensor storage dtype differs from receipt')
            samples += 1
    return {'shard': shard, 'sha256': rows[0]['tensor_locator']['shard_sha256'],
            'bytes': path.stat().st_size, 'samples': samples}


def verify_capture(root, run_id, workers=4):
    root = Path(root).resolve()
    run = load_plan(root, run_id)['run']
    trace = run / 'enrichment/online'
    receipt = read_json(trace / 'receipt.json')
    for name, digest in receipt['files'].items():
        if sha256(trace / name) != digest:
            raise ValueError('Finalized evidence changed')
    by_shard = defaultdict(list)
    rows = read_json(trace / 'index.json')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample identity')
    for row in rows:
        by_shard[row['tensor_locator']['shard']].append(row)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(verify_shard, (receipt['source_root'], shard, members))
                   for shard, members in by_shard.items()]
        for number, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            write_json(run / 'reports/tensor-audit-progress.json', {
                'shards': number, 'total_shards': len(futures),
                'samples': sum(r['samples'] for r in results), 'updated_at': now()})
    report = {'schema_version': 1, 'run_id': run_id, 'verified': True,
        'source_root': receipt['source_root'], 'trace_receipt_sha256': sha256(trace / 'receipt.json'),
        'samples': len(rows), 'shards': len(results), 'bytes': sum(r['bytes'] for r in results),
        'checks': ['full shard SHA', 'member SHA', 'native sites', 'shape', 'storage dtype', 'all finite'],
        'results': sorted(results, key=lambda r: r['shard']), 'finished_at': now()}
    write_json(run / 'reports/tensor-audit.json', report)
    write_json(run / 'reports/tensor-audit-receipt.json', {'schema_version': 1,
        'files': {'tensor-audit.json': sha256(run / 'reports/tensor-audit.json')}})
    refresh(root, run_id)
    return {k: v for k, v in report.items() if k != 'results'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='.')
    p.add_argument('--run', required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    print(json.dumps(verify_capture(a.root, a.run, a.workers), indent=2))


if __name__ == '__main__':
    main()

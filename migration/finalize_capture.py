"""Bounded token/hidden finalization for an explicitly imported capture run.

Uses the shared finalizer and original hash-bound canonical request ledgers.
Only selected published sessions enter the output; NAS data remain immutable.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from application.execution.artifact import refresh
from application.execution.run_plan import load_plan
from application.execution.evidence import RunEvidence
from core.io import identifier, now, owned_path, read_json, sha256, write_json
from core.locking import file_lock
from enrichment.online_trace import finalize_token_events
from hidden.artifacts import event_records, response_records


def _finalize_session(job):
    root, run_id, nas, output, sid, session, row, events, open_ids, close_ids = job
    rid = session['rollout_id']
    if row['run_id'] != run_id or row['trajectory_sha256'] != session['trajectory_sha256']:
        raise ValueError('Migrated canonical identity changed')
    trajectory = owned_path(root, row['trajectory_path'])
    if sha256(trajectory) != row['trajectory_sha256']:
        raise ValueError('Canonical trajectory checksum mismatch')
    for ref in session['files'].values():
        if sha256(ref['nas_locator']) != ref['sha256']:
            raise ValueError('Source NAS session metadata changed')
    final = read_json(session['files']['finalization.json']['nas_locator'])
    accepted = set(final['accepted_native_request_ids'])
    evidence = []
    seen = set()
    for _, message, response in response_records(read_json(trajectory)):
        native = response.get('dycon_native_request_id')
        if native in accepted:
            if native in seen:
                raise ValueError('Duplicate canonical native request identity')
            seen.add(native)
            evidence.append({'ledger_ref': response['ledger_ref'],
                             'ledger_sha256': response['ledger_sha256']})
    if seen != accepted:
        raise ValueError('Canonical response/accepted request cardinality differs')
    target = output / 'sessions' / identifier(sid)
    receipt_path = target / 'receipt.json'
    if receipt_path.exists():
        proof = read_json(receipt_path)
        for name, digest in proof['files'].items():
            if sha256(target / name) != digest:
                raise ValueError('Previously finalized session output changed')
        for name, digest in proof['source_files'].items():
            if sha256(owned_path(nas, name)) != digest:
                raise ValueError('Previously finalized source changed')
    else:
        if target.exists():
            # Preserve partial output for diagnosis; explicit retry starts a clean directory.
            target.rename(target.with_name(target.name + '-partial-' + str(time.time_ns())))
        finalize_token_events(nas, target, open_ids=open_ids, close_ids=close_ids,
            request_evidence=evidence, event_rows=events,
            trajectory_paths={rid: str(trajectory)})
    return target


def finalize(root, run_id, workers=4):
    root = Path(root).resolve()
    run_id = identifier(run_id)
    loaded = load_plan(root, run_id)
    run = loaded['run']
    publication = read_json(run / 'hidden/publication.json')
    nas = Path(publication['nas_locator'])
    inventory = read_json(nas / 'inventory.json')
    open_ids, close_ids = inventory.get('open_token_ids'), inventory.get('close_token_ids')
    if not open_ids or not close_ids:
        raise ValueError('Recorded capture inventory must declare reasoning marker token IDs')
    sessions = {s['session_id']: s for s in publication['sessions']}
    output = run / 'enrichment/online'
    with file_lock(root / 'state' / f'finalize-migration-{run_id}.lock', nonblocking=True):
        if (output / 'receipt.json').exists():
            raise FileExistsError('Finalized output exists; create a fresh analysis run to change it')
        catalog = RunEvidence(root, run_id).read()
        events = defaultdict(list)
        for path, event in event_records(nas):
            if event['session_id'] in sessions:
                events[event['session_id']].append((path, event))
        combined, excluded, requests, session_proofs = [], [], {}, {}
        jobs = [(root, run_id, nas, output, sid, session, catalog['rollouts'][session['rollout_id']],
                 events[sid], open_ids, close_ids) for sid, session in sorted(sessions.items())]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = [pool.submit(_finalize_session, job) for job in jobs]
            for number, future in enumerate(as_completed(results), 1):
                target = future.result()
                receipt_path = target / 'receipt.json'
                combined.extend(read_json(target / 'index.json'))
                excluded.extend(read_json(target / 'excluded.json'))
                for key, request in read_json(target / 'requests.json').items():
                    if key in requests:
                        raise ValueError('Native request reused across sessions')
                    requests[key] = request
                session_proofs[str(receipt_path.relative_to(output))] = sha256(receipt_path)
                write_json(run / 'reports/finalization-progress.json', {
                    'completed_sessions': number, 'total_sessions': len(sessions),
                    'events': len(combined), 'requests': len(requests), 'updated_at': now()})
                print(f'finalized {number}/{len(sessions)} sessions; {len(combined)} events', flush=True)
        combined.sort(key=lambda r: (r['task_id'], r['rollout_id'], r['step_id'], r['absolute_position']))
        write_json(output / 'index.json', combined)
        write_json(output / 'excluded.json', excluded)
        write_json(output / 'requests.json', requests)
        write_json(output / 'coverage.json', {'sessions': len(sessions), 'events': len(combined),
            'missing': [], 'unexpected': [], 'session_receipts': session_proofs})
        files = {name: sha256(output / name) for name in
                 ('index.json', 'excluded.json', 'requests.json', 'coverage.json')}
        write_json(output / 'receipt.json', {'schema_version': 1, 'origin': 'migrated',
            'source_root': str(nas), 'source_run_id': publication['source_run_id'],
            'source_publication_sha256': sha256(run / 'hidden/publication.json'),
            'events': len(combined), 'requests': len(requests), 'sessions': len(sessions),
            'files': {**files, **session_proofs}, 'finished_at': now(),
            'labels': 'exact observed trajectory remainder; successful remainder requires resolved evaluation'})
        refresh(root, run_id)
        return {'sessions': len(sessions), 'events': len(combined), 'requests': len(requests)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='.')
    p.add_argument('--run', required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    print(json.dumps(finalize(a.root, a.run, a.workers), indent=2))


if __name__ == '__main__':
    main()

"""Native OpenHands condenser settings and immutable offline publication updates."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import shutil
import zipfile
from pathlib import Path

from core.io import hash_json, identifier, read_json, sha256, write_json

POLICY = 'configurable_condenser_events_v1'
MEMBER = 'openhands_cli/stores/agent_store.py'
SEAM = '        threshold = os.environ.get("OPENHANDS_CONDENSER_MAX_TOKENS")'
EVENT_SETTINGS = '''        updates = {}
        for name, field in (("OPENHANDS_CONDENSER_MAX_SIZE", "max_size"),
                            ("OPENHANDS_CONDENSER_KEEP_FIRST", "keep_first")):
            raw = os.environ.get(name)
            if raw is not None:
                updates[field] = int(raw)
        if updates:
            # model_copy(update=...) skips Pydantic validation. Reconstruct the
            # native condenser to validate both bounds and their relationship.
            condenser = type(condenser)(**{
                **{field: getattr(condenser, field) for field in type(condenser).model_fields},
                **updates})
'''


def validate_settings(max_size, keep_first):
    if (type(max_size) is not int or type(keep_first) is not int
            or max_size <= 0 or keep_first < 0
            or max_size // 2 - keep_first - 1 <= 0):
        raise ValueError('Invalid condenser settings: keep_first must be nonnegative '
                         'and less than max_size // 2 - 1')
    return {'max_size': max_size, 'keep_first': keep_first}


def adapt_source(source):
    if source.count(SEAM) != 1 or 'OPENHANDS_CONDENSER_MAX_SIZE' in source:
        raise ValueError('Pinned CLI condenser settings seam changed')
    return source.replace(SEAM, EVENT_SETTINGS + SEAM)


def _contents(wheel):
    with zipfile.ZipFile(wheel) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _adapted_contents(wheel):
    contents = _contents(wheel)
    contents[MEMBER] = adapt_source(contents[MEMBER].decode()).encode()
    record_name = next(name for name in contents if name.endswith('.dist-info/RECORD'))
    record = io.StringIO()
    writer = csv.writer(record, lineterminator='\n')
    for name, data in sorted(contents.items()):
        if name == record_name:
            writer.writerow([name, '', ''])
        else:
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
            writer.writerow([name, 'sha256=' + digest, str(len(data))])
    contents[record_name] = record.getvalue().encode()
    return contents


def verify_derivation(source, target):
    """Allow qualification transfer only for the exact known condenser adaptation."""
    from runtime.agents.dependencies import verify_snapshot
    source, target = Path(source).resolve(), Path(target).resolve()
    old, new = verify_snapshot(source), verify_snapshot(target)
    if new.get('condenser_parent_receipt_sha256') != old['receipt_sha256']:
        raise ValueError('Condenser publication does not bind this parent')
    wheel = next(source.glob('wheels/openhands-*.whl')).relative_to(source).as_posix()
    before = {r['path']: r for r in old['files']}
    after = {r['path']: r for r in new['files']}
    if before.keys() != after.keys() or any(before[p] != after[p] for p in before
                                           if p not in {wheel, 'requirements.txt'}):
        raise ValueError('Condenser publication changed unrelated dependency files')
    if _contents(target / wheel) != _adapted_contents(source / wheel):
        raise ValueError('Condenser publication contains unrelated CLI changes')
    expected_lock = (source / 'requirements.txt').read_text().replace(
        sha256(source / wheel), sha256(target / wheel))
    if (target / 'requirements.txt').read_text() != expected_lock:
        raise ValueError('Condenser publication changed unrelated requirement locks')
    # Every parent identity field survives, including interpreter and source.
    body = {k: v for k, v in new.items() if k not in {
        'files', 'receipt_sha256', 'source_adaptations', 'condenser_parent_receipt_sha256'}}
    parent_body = {k: v for k, v in old.items() if k not in {
        'files', 'receipt_sha256', 'source_adaptations'}}
    if body != parent_body or new['source_adaptations'][:-1] != old['source_adaptations']:
        raise ValueError('Condenser publication changed parent identity')
    adaptation = new['source_adaptations'][-1]
    if (adaptation.get('policy') != POLICY
            or adaptation.get('before_sha256') != sha256(source / wheel)
            or adaptation.get('after_sha256') != sha256(target / wheel)):
        raise ValueError('Condenser publication lacks exact adaptation evidence')
    return {'source_receipt': old['receipt_sha256'], 'target_receipt': new['receipt_sha256'],
            'policy': POLICY, 'verification': 'exact_cli_condenser_delta_and_lock_only'}


def derive_snapshot(cfg, snapshot_id, source):
    """Derive one CLI adaptation; retain every SDK/tools/interpreter byte."""
    from runtime.agents.dependencies import verify_snapshot
    root = Path(cfg['project_root']).resolve()
    source = Path(source).resolve()
    if not source.is_relative_to(root):
        raise ValueError('Source dependency snapshot must be project-owned')
    parent = verify_snapshot(source)
    if POLICY in {a['policy'] for a in parent.get('source_adaptations', [])}:
        raise ValueError('Source publication already supports condenser event settings')
    target = root / 'runtime/dependency-snapshots' / identifier(snapshot_id)
    if target.exists():
        verify_derivation(source, target)
        return {'path': str(target), 'manifest': read_json(target / 'manifest.json')}
    staging = root / 'temp/skillsbench-dependencies' / identifier(snapshot_id) / 'publication'
    if staging.exists():
        raise FileExistsError('Interrupted dependency build: use a new snapshot ID')
    shutil.copytree(source, staging)
    wheel = next(staging.glob('wheels/openhands-*.whl'))
    old_hash = sha256(wheel)
    contents = _adapted_contents(wheel)
    with zipfile.ZipFile(wheel, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    lock = staging / 'requirements.txt'
    old_lock = lock.read_text()
    if old_hash not in old_lock:
        raise ValueError('Published requirements do not bind the CLI wheel hash')
    lock.write_text(old_lock.replace(old_hash, sha256(wheel)))
    manifest = {**parent, 'condenser_parent_receipt_sha256': parent['receipt_sha256'],
        'source_adaptations': [*parent['source_adaptations'], {
            'path': wheel.relative_to(staging).as_posix() + ':' + MEMBER,
            'before_sha256': old_hash, 'after_sha256': sha256(wheel), 'policy': POLICY,
            'reason': 'Apply validated frozen event-count settings to the native condenser'}]}
    manifest['files'] = [{'path': p.relative_to(staging).as_posix(), 'size': p.stat().st_size,
                          'sha256': sha256(p)} for p in sorted(staging.rglob('*'))
                         if p.is_file() and p != staging / 'manifest.json']
    manifest['receipt_sha256'] = hash_json({k: v for k, v in manifest.items() if k != 'receipt_sha256'})
    write_json(staging / 'manifest.json', manifest)
    verify_derivation(source, staging)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(target)
    return {'path': str(target), 'manifest': manifest}

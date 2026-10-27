"""Hash-verified NAS archival of an inactive run's local capture spool."""
from pathlib import Path
from collections import Counter
import os
import shutil
import stat
import subprocess

from core.config import load_snapshot
from core.disk_budget import DiskBudget
from core.io import identifier, now, owned_path, read_json, sha256, write_json
from core.locking import exclusive_run


def inventory(path, *, size_only=False):
    path = Path(path)
    files, allocated = {}, 0
    if path.is_symlink():
        raise ValueError('Inventory requires the physical source directory')
    for base, dirs, names in os.walk(path, followlinks=False):
        dirs.sort()
        for name in dirs:
            if (Path(base) / name).is_symlink():
                raise ValueError('Nested symlink in capture spool')
        for name in sorted(names):
            p = Path(base) / name
            st = p.lstat()
            if not stat.S_ISREG(st.st_mode):
                raise ValueError('Non-regular capture file: ' + str(p))
            value = {'bytes': st.st_size}
            if not size_only:
                value['sha256'] = sha256(p)
            files[str(p.relative_to(path))] = value
            allocated += st.st_blocks * 512
    return files, allocated


def verify_tree(path, expected, *, size_only=False):
    observed, _ = inventory(path, size_only=size_only)
    if size_only:
        expected = {name: {'bytes': value['bytes']} for name, value in expected.items()}
    if observed != expected:
        raise ValueError('Capture archive/source checksum or file set mismatch: ' + str(path))
    return len(observed)


def assert_inactive(run_id, source):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) in {os.getpid(), os.getppid()}:
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(bytes([0]))
        except (OSError, PermissionError):
            continue
        parts = [arg.decode(errors='replace') for arg in args if arg]
        if run_id in parts or str(source) in parts:
            raise ValueError('Run/capture process still active: ' + proc.name)


def copy_tree(source, target, log):
    if not shutil.which('rsync'):
        raise RuntimeError('Capture archival requires rsync')
    target.mkdir(parents=True, exist_ok=True)
    with Path(log).open('ab') as stream:
        # NAS may map local UID/GID and reject chown/chgrp/chmod. Archive
        # identity is the frozen byte-size/SHA manifest, not local ownership.
        subprocess.run(['rsync', '-r', '--checksum', '--fsync', '--', str(source) + '/', str(target) + '/'],
                       stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, check=True)


def release_reservations(root, source, options):
    budget = DiskBudget(root, source.parent, options['spool_max_bytes'])
    prefix = 'capture:' + str(source) + '/'
    rows = {key: value for key, value in budget.allocations().items() if key.startswith(prefix)}
    size = sum(value['bytes'] for value in rows.values())
    for key in rows:
        budget.release(key)
    if any(key.startswith(prefix) for key in budget.allocations()):
        raise ValueError('Archived run still has durable capture reservations')
    return {'keys': len(rows), 'bytes': size, 'remaining_run_keys': 0}


@exclusive_run
def archive_online(cfg, run_id, *, archive_id, size_only_cleanup=False):
    root = Path(cfg['project_root']).resolve()
    run_id, archive_id = map(identifier, (run_id, archive_id))
    frozen = load_snapshot(root, run_id)
    options = frozen['capture']['export']
    source = root / options['spool_root'] / run_id
    source = Path(os.path.abspath(source))
    if not source.is_relative_to(root) or source == root:
        raise ValueError('Capture source must belong to this project')
    assert_inactive(run_id, source)
    nas = Path(frozen['paths']['nas_root']).resolve()
    target = nas / 'archives/capture-spool' / run_id / archive_id
    payload = target / 'spool'
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('Capture archive overlaps source')
    out = root / 'runs' / run_id / 'hidden/capture-archives' / archive_id
    out.mkdir(parents=True, exist_ok=True)
    signature = {'run_id': run_id, 'archive_id': archive_id, 'source': str(source),
                 'destination': str(target), 'snapshot_sha256': sha256(root / 'runs' / run_id / 'resolved-config.json')}
    selection = out / 'selection.json'
    if selection.exists() and read_json(selection) != signature:
        raise ValueError('Archive ID already binds different source/destination')
    write_json(selection, signature)
    retired = source.parent.parent / 'capture-archive-retired' / (run_id + '--' + archive_id)
    if source.is_symlink():
        if source.resolve() != payload.resolve():
            raise ValueError('Source pointer belongs to another archive')
        manifest = read_json(target / 'manifest.json')
        receipt = read_json(target / 'receipt.json')
        if receipt['selection'] != signature or sha256(target / 'manifest.json') != receipt['files']['manifest.json']:
            raise ValueError('Existing NAS archive receipt mismatch')
        verify_tree(payload, manifest['files'], size_only=size_only_cleanup)
        if retired.exists():
            verify_tree(retired, manifest['files'], size_only=size_only_cleanup)
            shutil.rmtree(retired)
        released = release_reservations(root, source, options)
        result = {**receipt['summary'], 'status': 'complete', 'source_pointer': str(source),
                  'source_pointer_target': str(payload), 'reservation_release': released,
                  'completed_at': now(), 'archive_receipt_sha256': sha256(target / 'receipt.json')}
        previous = out / 'result.json'
        if previous.exists():
            result['reservation_release'] = read_json(previous)['reservation_release']
        write_json(previous, result)
        return result
    if not source.is_dir():
        raise FileNotFoundError(source)
    if retired.exists():
        raise FileExistsError('Inspect interrupted local retirement before proceeding: ' + str(retired))
    manifest_path = out / 'manifest.json'
    if size_only_cleanup:
        if not manifest_path.exists():
            raise ValueError('Size-only cleanup requires an existing frozen archive manifest')
        manifest = read_json(manifest_path)
        if manifest['selection'] != signature:
            raise ValueError('Frozen archive manifest belongs to another selection')
        files = manifest['files']
        _, allocated = inventory(source, size_only=True)
    else:
        print('capture archive: hashing source', flush=True)
        files, allocated = inventory(source)
    if not files:
        raise ValueError('Capture spool is empty')
    manifest = {'schema_version': 1, 'selection': signature, 'files': files}
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError('Source changed after archive inventory; use a new archive ID')
    write_json(manifest_path, manifest)
    if not size_only_cleanup:
        write_json(out / 'progress.json', {'phase': 'copying', 'files': len(files), 'updated_at': now()})
        print(f'capture archive: copying {len(files)} files to {target}', flush=True)
        copy_tree(source, payload, out / 'rsync.log')
    print('capture archive: checking file set and sizes' if size_only_cleanup else 'capture archive: verifying full NAS copy', flush=True)
    verify_tree(payload, files, size_only=size_only_cleanup)
    verify_tree(source, files, size_only=size_only_cleanup)
    target.mkdir(parents=True, exist_ok=True)
    if (target / 'manifest.json').exists() and read_json(target / 'manifest.json') != manifest:
        raise ValueError('Immutable NAS archive manifest conflict')
    write_json(target / 'manifest.json', manifest)
    summary = {'run_id': run_id, 'archive_id': archive_id, 'archive_path': str(target),
               'archived_files': len(files), 'archived_bytes': sum(v['bytes'] for v in files.values()),
               'released_allocated_file_bytes': allocated,
               'verification': 'file_set_and_size_only_per_user' if size_only_cleanup else 'all_file_sha256_and_exact_file_set',
               'top_level_files': dict(Counter(Path(p).parts[0] for p in files))}
    receipt_path = target / 'receipt.json'
    receipt = {'schema_version': 1, 'state': 'complete', 'selection': signature,
               'summary': summary, 'files': {'manifest.json': sha256(target / 'manifest.json')}}
    if receipt_path.exists() and read_json(receipt_path) != receipt:
        raise ValueError('Immutable NAS archive receipt conflict')
    write_json(receipt_path, receipt)
    if sha256(target / 'manifest.json') != sha256(manifest_path):
        raise ValueError('NAS/local manifest mismatch')
    for p in (target / 'manifest.json', receipt_path):
        with p.open('rb') as handle:
            os.fsync(handle.fileno())
    assert_inactive(run_id, source)
    write_json(out / 'progress.json', {'phase': 'retiring_local', 'updated_at': now()})
    retired.parent.mkdir(parents=True, exist_ok=True)
    source.rename(retired)
    try:
        source.symlink_to(payload, target_is_directory=True)
    except BaseException:
        retired.rename(source)
        raise
    if source.resolve() != payload.resolve():
        raise ValueError('Archived evidence path does not resolve to verified NAS payload')
    # All bytes have a checksum-verified NAS copy and sealed manifest/receipt.
    # Keep original absolute evidence paths readable through the NAS pointer.
    shutil.rmtree(retired)
    released = release_reservations(root, source, options)
    result = {**summary, 'status': 'complete', 'source_pointer': str(source),
              'source_pointer_target': str(payload), 'reservation_release': released,
              'completed_at': now(), 'archive_receipt_sha256': sha256(receipt_path)}
    write_json(out / 'result.json', result)
    write_json(out / 'progress.json', {'phase': 'complete', 'updated_at': now()})
    print('capture archive: local duplicate removed and run reservations released', flush=True)
    return result

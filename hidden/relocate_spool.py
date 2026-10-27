"""Offline relocation of local capture storage, preserving frozen logical paths."""
from contextlib import ExitStack, closing
from pathlib import Path
import os
import shutil
import sqlite3
import stat
import subprocess

from core.disk_budget import DiskBudget
from core.io import identifier, now, read_json, sha256, write_json
from core.locking import file_lock
from runtime.serving.spool import local_filesystem


def inventory(source):
    """Hash regular bytes and preserve symlink identities without traversing NAS."""
    entries, allocated = {}, 0
    for base, directories, files in os.walk(source, followlinks=False):
        for name in sorted([*directories, *files]):
            path = Path(base) / name
            info = path.lstat()
            relative = str(path.relative_to(source))
            if stat.S_ISLNK(info.st_mode):
                entries[relative] = {"type": "symlink", "target": os.readlink(path)}
            elif stat.S_ISREG(info.st_mode):
                entries[relative] = {"type": "file", "bytes": info.st_size,
                                     "sha256": sha256(path)}
                allocated += info.st_blocks * 512
            elif stat.S_ISDIR(info.st_mode):
                entries[relative] = {"type": "directory"}
            else:
                raise ValueError("Unsupported capture file: " + str(path))
        directories[:] = sorted(d for d in directories if not (Path(base) / d).is_symlink())
    return entries, allocated


def assert_no_writers(root, run_ids):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) in {os.getpid(), os.getppid()}:
            continue
        try:
            argv = [v.decode(errors='replace') for v in
                    (proc / 'cmdline').read_bytes().split(b'\0') if v]
            env = dict(v.split(b'=', 1) for v in
                       (proc / 'environ').read_bytes().split(b'\0') if b'=' in v)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        producer = env.get(b'DYCON_CAPTURE_PROJECT', b'').decode() == str(root)
        if producer or any(name in argv for name in run_ids):
            raise RuntimeError('Capture/run process still active: ' + proc.name)


def rebind_rows(db, budget, source, device):
    """Use the shared ledger's accounting primitives in the caller's transaction."""
    prefix = 'capture:' + str(source) + '/'
    rows = db.execute('SELECT key,bytes,kind,device,path,pid FROM reservations').fetchall()
    moved, size = 0, 0
    for key, amount, kind, old_device, path, pid in rows:
        if not key.startswith(prefix):
            continue
        if kind != 'capture':
            raise ValueError('Unexpected durable reservation kind: ' + key)
        budget._put(db, key, {'bytes': amount, 'kind': kind, 'device': device,
                              'path': path, 'pid': pid})
        moved += 1
        size += amount
    return {'keys': moved, 'bytes': size, 'destination_device': device}


def relocate_spool(cfg, *, destination, migration_id, apply=False):
    root = Path(cfg['project_root']).resolve()
    source = Path(os.path.abspath(root / cfg['capture']['export']['spool_root']))
    target = Path(destination).absolute()
    migration_id = identifier(migration_id)
    if not source.is_relative_to(root) or source == root:
        raise ValueError('Source must be a project-local logical capture directory')
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('Source/destination overlap')
    if target.is_symlink() or (target.exists() and target.resolve() != target):
        raise ValueError('Destination must be a physical local directory')
    parent = target.parent
    while not parent.exists():
        parent = parent.parent
    local_filesystem(parent)
    signature = {'source': str(source), 'destination': str(target),
                 'migration_id': migration_id}
    out = root / 'state/capture-storage-relocations' / migration_id
    previous = out / 'selection.json'
    if previous.exists() and read_json(previous) != signature:
        raise ValueError('Migration ID already binds another location')
    if source.is_symlink():
        if source.resolve() != target.resolve() or not (out / 'result.json').exists():
            raise ValueError('Source is already redirected without this relocation receipt')
        result = read_json(out / 'result.json')
        if not result.get('source_backup_removed'):
            raise RuntimeError('Inspect interrupted relocation before resuming producers')
        return result
    if not source.is_dir():
        raise FileNotFoundError(source)
    run_ids = sorted(p.name for p in source.iterdir() if p.is_dir())
    retired = source.with_name(source.name + '.retired-' + migration_id)
    if retired.exists() or retired.is_symlink():
        raise RuntimeError('An interrupted source retirement requires explicit recovery')
    if target.exists() and not previous.exists():
        raise ValueError('Refusing to merge an existing unowned destination')
    with ExitStack() as locks:
        locks.enter_context(file_lock(root / 'state/capture-storage.lock', nonblocking=True))
        for name in run_ids:
            locks.enter_context(file_lock(root / 'state/run-locks' / (name + '.lock'), nonblocking=True))
            locks.enter_context(file_lock(root / 'runs' / name / 'execution/active-supervisor.lock',
                                          nonblocking=True))
        assert_no_writers(root, run_ids)
        entries, allocated = inventory(source)
        needed = sum(v.get('bytes', 0) for v in entries.values())
        if shutil.disk_usage(parent).free < needed + cfg['images']['reserve_bytes']:
            raise RuntimeError('Destination has insufficient free space')
        summary = {**signature, 'regular_files': sum(v['type'] == 'file' for v in entries.values()),
                   'symlinks': sum(v['type'] == 'symlink' for v in entries.values()),
                   'bytes': needed, 'source_allocated_bytes': allocated,
                   'source_device': source.stat().st_dev, 'planned_at': now()}
        if not apply:
            return {**summary, 'apply': False}
        if not shutil.which('rsync'):
            raise RuntimeError('Relocation requires rsync')
        out.mkdir(parents=True, exist_ok=True)
        write_json(previous, signature)
        write_json(out / 'manifest.json', {'selection': signature, 'entries': entries})
        write_json(out / 'progress.json', {'phase': 'copying', 'updated_at': now()})
        target.mkdir(parents=True, exist_ok=True)
        with (out / 'rsync.log').open('ab') as log:
            subprocess.run(['rsync', '-a', '--checksum', '--fsync', '--',
                            str(source) + '/', str(target) + '/'], check=True,
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        observed, _ = inventory(target)
        if observed != entries:
            raise ValueError('Destination file set, symlink or SHA-256 mismatch')
        assert_no_writers(root, run_ids)
        # Reclaim only the existing ledger's exited transient reservations.
        budget = DiskBudget(root, source, cfg['capture']['export']['spool_max_bytes'])
        device = target.stat().st_dev
        write_json(out / 'progress.json', {'phase': 'switching', 'updated_at': now()})
        with file_lock(budget.lock), closing(sqlite3.connect(budget.state, timeout=30)) as db:
            db.execute('BEGIN IMMEDIATE')
            switched = False
            try:
                source.rename(retired)
                source.symlink_to(target, target_is_directory=True)
                switched = True
                moved = rebind_rows(db, budget, source, device)
                db.commit()
            except BaseException:
                db.rollback()
                if switched:
                    source.unlink()
                if retired.exists() and not source.exists():
                    retired.rename(source)
                raise
        with sqlite3.connect(budget.state) as db:
            prefix = 'capture:' + str(source) + '/'
            bad = [r for r in db.execute('SELECT key,device FROM reservations')
                   if r[0].startswith(prefix) and r[1] != device]
            if bad:
                raise RuntimeError('Relocated reservation still references the old device')
        if source.resolve() != target.resolve() or source.stat().st_dev != device:
            raise RuntimeError('Logical capture path did not switch to destination')
        result = {**summary, 'apply': True, 'destination_device': device,
                  'reservation_rebind': moved,
                  'verification': 'all_regular_file_sha256_exact_file_set_and_symlink_targets',
                  'manifest_sha256': sha256(out / 'manifest.json'),
                  'source_backup_removed': False, 'switched_at': now()}
        write_json(out / 'result.json', result)
        # Both the new bytes and ledger are verified before removing the old copy.
        if retired.parent != source.parent or retired.is_symlink():
            raise RuntimeError('Unexpected retired source path')
        shutil.rmtree(retired)
        result.update(source_backup_removed=True, completed_at=now())
        write_json(out / 'result.json', result)
        write_json(out / 'progress.json', {'phase': 'complete', 'updated_at': now()})
        return result

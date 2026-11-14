import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from core.io import sha256
from runtime.agents.submission_restore import archive_inventory, RESTORE_CODE


def archive(tmp_path, members):
    path = tmp_path / 'subject.tar'
    with tarfile.open(path, 'w') as stream:
        for name, value in members:
            info = tarfile.TarInfo(name)
            if isinstance(value, tuple):
                info.type = tarfile.SYMTYPE
                info.linkname = value[0]
                stream.addfile(info)
            else:
                data = value.encode()
                info.size, info.mode = len(data), 0o640
                stream.addfile(info, io.BytesIO(data))
    return path


@pytest.mark.parametrize('name,target', [('../escape', None), ('/absolute', None),
    ('app/link', '../../outside'), ('app/link', '/etc/passwd')])
def test_archive_cannot_escape_declared_output(tmp_path, name, target):
    path = archive(tmp_path, [(name, (target,) if target else 'data')])
    with pytest.raises(ValueError):
        archive_inventory(path, sha256(path), '/app')


def test_archive_cannot_write_through_internal_symlink(tmp_path):
    path = archive(tmp_path, [('app/link', ('other',)), ('app/link/file', 'data')])
    with pytest.raises(ValueError, match='link parent'):
        archive_inventory(path, sha256(path), '/app')


def test_archive_hash_tampering_is_rejected(tmp_path):
    path = archive(tmp_path, [('app/source.py', 'original')])
    expected = sha256(path)
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='Changed subject archive'):
        archive_inventory(path, expected, '/app')


def test_restoration_replaces_old_tree_and_checks_bytes_and_internal_links(tmp_path):
    source = tmp_path / 'container' / 'app'
    source.mkdir(parents=True)
    (source / 'initial-file').write_text('must not fill in unexported submission')
    path = archive(tmp_path, [('app/source.py', 'saved code'), ('app/copy.py', ('source.py',))])
    request = archive_inventory(path, sha256(path), str(source))
    request.update(kind='directory_archive', uploaded_path=str(path), allowed_paths=[str(source)])
    request_path = tmp_path / 'request.json'
    request_path.write_text(json.dumps(request))
    result = subprocess.run([sys.executable, '-c', RESTORE_CODE, str(request_path)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['verified']
    assert not (source / 'initial-file').exists()
    assert (source / 'source.py').read_text() == 'saved code'
    assert (source / 'copy.py').is_symlink()
    assert (source / 'copy.py').read_text() == 'saved code'


def test_missing_output_is_restored_as_missing_without_filling_image_default(tmp_path):
    source = tmp_path / 'output.txt'
    source.write_text('initial image default')
    request = tmp_path / 'request.json'
    request.write_text(json.dumps({'kind': 'missing', 'source_path': str(source), 'allowed_paths': [str(source)]}))
    result = subprocess.run([sys.executable, '-c', RESTORE_CODE, str(request)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert not source.exists()


def test_replay_plan_rejects_original_run_as_output_or_invalid_workers(monkeypatch):
    from runtime.agents.submission_regrading import regrade
    for source, workers in [('same', 3), ('other', 0), ('other', 5)]:
        with pytest.raises(ValueError, match='distinct frozen target'):
            regrade('.', 'same', source, 'audit.jsonl', 'output', workers=workers)

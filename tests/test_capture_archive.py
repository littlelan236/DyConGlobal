from pathlib import Path
import shutil
import pytest

from core.io import read_json, sha256, write_json
from core.disk_budget import DiskBudget
from hidden.archive_online import archive_online


def fixture(tmp_path, monkeypatch):
    import hidden.archive_online as archive
    source = tmp_path / 'spool/run'
    (source / 'requests').mkdir(parents=True)
    (source / 'requests/native.json').write_text('native evidence')
    (source / 'index.json').write_text('original index')
    cfg = {'project_root': str(tmp_path), 'paths': {'nas_root': str(tmp_path / 'nas')},
           'capture': {'export': {'spool_root': 'spool', 'spool_max_bytes': 1000000}}}
    write_json(tmp_path / 'runs/run/resolved-config.json', cfg)
    monkeypatch.setattr(archive, 'load_snapshot', lambda *a: cfg)
    monkeypatch.setattr(archive, 'assert_inactive', lambda *a: None)
    monkeypatch.setattr(archive, 'copy_tree', lambda src, dst, log: shutil.copytree(src, dst, dirs_exist_ok=True))
    budget = DiskBudget(tmp_path, source.parent, 1000000)
    budget.acquire('capture:' + str(source / 'requests/native.json'), 100)
    budget.acquire('capture:' + str(source / 'index.json'), 200)
    budget.acquire('capture:' + str(source.parent / 'other/file'), 333)
    return cfg, source, budget


def test_archive_preserves_evidence_and_releases_only_selected_run(tmp_path, monkeypatch):
    cfg, source, budget = fixture(tmp_path, monkeypatch)
    expected = sha256(source / 'requests/native.json')
    result = archive_online(cfg, 'run', archive_id='archive')
    assert source.is_symlink()
    assert sha256(source / 'requests/native.json') == expected
    assert source.resolve().is_relative_to(tmp_path / 'nas')
    assert result['archived_files'] == 2
    assert result['reservation_release']['bytes'] == 300
    assert budget.usage() == 333
    assert not list((tmp_path / 'capture-archive-retired').glob('*'))
    again = archive_online(cfg, 'run', archive_id='archive')
    assert again['reservation_release']['bytes'] == 300
    assert budget.usage() == 333


def test_archive_custom_destination_with_symlinked_spool(tmp_path, monkeypatch):
    cfg, source, budget = fixture(tmp_path, monkeypatch)
    physical = tmp_path / 'other-device' / 'spool'
    physical.parent.mkdir()
    source.parent.rename(physical)
    (tmp_path / 'spool').symlink_to(physical, target_is_directory=True)
    destination = tmp_path / 'archive-destination'
    expected = sha256(source / 'requests/native.json')
    result = archive_online(cfg, 'run', archive_id='archive', destination=destination)
    assert source.is_symlink()
    assert source.resolve().is_relative_to(destination)
    assert sha256(source / 'requests/native.json') == expected
    assert result['reservation_release']['bytes'] == 300
    assert budget.usage() == 333
    assert archive_online(cfg, 'run', archive_id='archive', destination=destination)['status'] == 'complete'


def test_bad_nas_copy_never_removes_local_bytes_or_reservations(tmp_path, monkeypatch):
    import hidden.archive_online as archive
    cfg, source, budget = fixture(tmp_path, monkeypatch)
    def corrupt(src, dst, log):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        (dst / 'index.json').write_text('corrupt')
    monkeypatch.setattr(archive, 'copy_tree', corrupt)
    with pytest.raises(ValueError, match='checksum'):
        archive_online(cfg, 'run', archive_id='archive')
    assert source.is_dir() and not source.is_symlink()
    assert budget.usage() == 633


def test_conflicting_archive_and_corruption_refuse_reuse(tmp_path, monkeypatch):
    cfg, source, budget = fixture(tmp_path, monkeypatch)
    archive_online(cfg, 'run', archive_id='archive')
    with pytest.raises(ValueError, match='another archive'):
        archive_online(cfg, 'run', archive_id='different')
    (source / 'index.json').write_text('corrupt')
    with pytest.raises(ValueError, match='checksum'):
        archive_online(cfg, 'run', archive_id='archive')


def test_archived_spool_cannot_start_capture(tmp_path):
    from runtime.serving.spool import CaptureSpool
    spool = tmp_path / 'spool'
    spool.mkdir()
    target = tmp_path / 'nas/archive'
    target.mkdir(parents=True)
    (spool / 'run').symlink_to(target, target_is_directory=True)
    (tmp_path / 'runs/run/hidden/capture-archives').mkdir(parents=True)
    with pytest.raises(ValueError, match='archived or redirected'):
        CaptureSpool(tmp_path, {'spool_root': 'spool'}, 'run', str(tmp_path / 'nas'))
    assert not (tmp_path / 'state/disk-budget.sqlite3').exists()


def test_rsync_copy_transfers_content_without_local_owner_group(tmp_path):
    from hidden.archive_online import copy_tree
    if not shutil.which('rsync'):
        pytest.skip('rsync is unavailable')
    source, target = tmp_path / 'source', tmp_path / 'target'
    (source / 'requests').mkdir(parents=True)
    (source / 'requests/evidence.json').write_text('native evidence')
    copy_tree(source, target, tmp_path / 'rsync.log')
    assert sha256(source / 'requests/evidence.json') == sha256(target / 'requests/evidence.json')
    copy_tree(source, target, tmp_path / 'rsync.log')
    assert len(list(target.rglob('*'))) == 2


def test_explicit_size_only_cleanup_of_already_copied_data(tmp_path, monkeypatch):
    import hidden.archive_online as archive
    cfg, source, budget = fixture(tmp_path, monkeypatch)
    def corrupt(src, dst, log):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        (dst / 'index.json').write_text('same-size-data')
    monkeypatch.setattr(archive, 'copy_tree', corrupt)
    with pytest.raises(ValueError, match='checksum'):
        archive_online(cfg, 'run', archive_id='archive')
    result = archive_online(cfg, 'run', archive_id='archive', size_only_cleanup=True)
    assert result['verification'] == 'file_set_and_size_only_per_user'
    assert source.is_symlink() and budget.usage() == 333


def test_size_only_cleanup_still_rejects_missing_files(tmp_path, monkeypatch):
    import hidden.archive_online as archive
    cfg, source, budget = fixture(tmp_path, monkeypatch)
    def incomplete(src, dst, log):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        (dst / 'index.json').unlink()
    monkeypatch.setattr(archive, 'copy_tree', incomplete)
    with pytest.raises(ValueError, match='checksum'):
        archive_online(cfg, 'run', archive_id='archive')
    with pytest.raises(ValueError, match='checksum'):
        archive_online(cfg, 'run', archive_id='archive', size_only_cleanup=True)
    assert not source.is_symlink() and budget.usage() == 633

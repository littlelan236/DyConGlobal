from pathlib import Path
from contextlib import closing
import sqlite3

import pytest

from core.disk_budget import DiskBudget
from hidden.relocate_spool import inventory, rebind_rows, relocate_spool


def config(root):
    return {'project_root': str(root), 'capture': {'export': {
        'spool_root': 'hidden/spool', 'spool_max_bytes': 1024 ** 3}},
        'images': {'reserve_bytes': 0}}


def test_inventory_preserves_symlink_without_traversal(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    outside = tmp_path / 'archive'
    outside.mkdir()
    (outside / 'keep').write_text('archive bytes')
    (source / 'historical').symlink_to(outside, target_is_directory=True)
    (source / 'point').write_bytes(b'original')
    rows, _ = inventory(source)
    assert rows['historical'] == {'type': 'symlink', 'target': str(outside)}
    assert 'historical/keep' not in rows
    assert rows['point']['bytes'] == 8


def test_device_rebind_keeps_keys_and_unrelated_image_budget(tmp_path):
    source = tmp_path / 'hidden/spool'
    budget = DiskBudget(tmp_path, source, 1024 ** 3)
    key = 'capture:' + str(source / 'run/point')
    budget.acquire(key, 100)
    budget.acquire('image:separate', 200, kind='image')
    rows = budget.allocations()
    new_device = source.stat().st_dev + 999
    with closing(sqlite3.connect(budget.state)) as db, db:
        result = rebind_rows(db, budget, source, new_device)
    moved = budget.allocations()
    assert moved[key]['device'] == new_device
    assert moved['image:separate'] == rows['image:separate']
    assert budget.usage() == 100
    assert result['keys'] == 1


def test_relocate_preserves_bytes_pointer_and_accounting(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    source = root / 'hidden/spool'
    (source / 'run').mkdir(parents=True)
    (source / 'run/point').write_bytes(b'exact hidden bytes')
    budget = DiskBudget(root, source, 1024 ** 3)
    key = 'capture:' + str(source / 'run/point')
    budget.acquire(key, 18)
    monkeypatch.setattr('hidden.relocate_spool.assert_no_writers', lambda *a: None)
    target = tmp_path / 'data/spool'
    result = relocate_spool(config(root), destination=target, migration_id='move-test', apply=True)
    assert result['source_backup_removed']
    assert source.is_symlink() and source.resolve() == target
    assert (source / 'run/point').read_bytes() == b'exact hidden bytes'
    assert budget.allocations()[key]['device'] == target.stat().st_dev
    assert budget.usage() == 18
    assert relocate_spool(config(root), destination=target, migration_id='move-test', apply=True) == result


def test_active_writer_refuses_before_copy(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    source = root / 'hidden/spool/run'
    source.mkdir(parents=True)
    (source / 'point').write_bytes(b'keep')
    def active(*args):
        raise RuntimeError('writer active')
    monkeypatch.setattr('hidden.relocate_spool.assert_no_writers', active)
    target = tmp_path / 'data/spool'
    with pytest.raises(RuntimeError, match='writer active'):
        relocate_spool(config(root), destination=target, migration_id='blocked', apply=True)
    assert (source / 'point').read_bytes() == b'keep'
    assert not target.exists()


def test_existing_destination_requires_owned_receipt(tmp_path):
    root = tmp_path / 'project'
    (root / 'hidden/spool').mkdir(parents=True)
    target = tmp_path / 'existing'
    target.mkdir()
    with pytest.raises(ValueError, match='unowned'):
        relocate_spool(config(root), destination=target, migration_id='blocked', apply=True)


def test_cli_storage_is_project_scoped_not_a_run(tmp_path, monkeypatch, capsys):
    import json
    from core.cli import main
    root = tmp_path / 'project'
    (root / 'hidden/spool/run').mkdir(parents=True)
    (root / 'hidden/spool/run/point').write_bytes(b'keep')
    monkeypatch.setattr('core.cli.resolve', lambda *args: config(root))
    monkeypatch.setattr('hidden.relocate_spool.assert_no_writers', lambda *args: None)
    main(['capture', 'relocate-spool', '--destination', str(tmp_path / 'target'),
          '--migration-id', 'dry-run'])
    result = json.loads(capsys.readouterr().out)
    assert result['apply'] is False
    assert result['regular_files'] == 1
    assert not (tmp_path / 'target').exists()

import copy
import pytest
from core.io import hash_json, sha256, write_json
from runtime.agents.dependencies import verify_snapshot, install_command
from runtime.agents.openhands import CLI_REVISION


def publication(tmp_path):
    (tmp_path/'requirements.txt').write_text('example==1\n')
    row={'path':'requirements.txt','size':(tmp_path/'requirements.txt').stat().st_size,
         'sha256':sha256(tmp_path/'requirements.txt')}
    manifest={'schema_version':1,'cli_revision':CLI_REVISION,'files':[row]}
    manifest['receipt_sha256']=hash_json(manifest)
    write_json(tmp_path/'manifest.json',manifest)
    return manifest


def test_dependency_snapshot_detects_tampered_manifest_and_file(tmp_path):
    manifest=publication(tmp_path)
    assert verify_snapshot(tmp_path)==manifest
    (tmp_path/'requirements.txt').write_text('example==2\n')
    with pytest.raises(ValueError,match='Changed dependency'):
        verify_snapshot(tmp_path)
    changed=copy.deepcopy(manifest)
    changed['files'][0]['sha256']=sha256(tmp_path/'requirements.txt')
    write_json(tmp_path/'manifest.json',changed)
    with pytest.raises(ValueError,match='receipt changed'):
        verify_snapshot(tmp_path)


def test_dependency_snapshot_rejects_unregistered_wheels(tmp_path):
    publication(tmp_path)
    (tmp_path/'extra.whl').write_bytes(b'injected')
    with pytest.raises(ValueError,match='unregistered'):
        verify_snapshot(tmp_path)


def test_offline_install_requires_hashes_and_private_runtime():
    command=install_command()
    assert '--no-index' in command and '--require-hashes' in command
    assert '/opt/dycon-openhands/bin/python' in command
    assert 'pip install' not in command


@pytest.mark.parametrize('capture',['collecting','partial','complete','disabled'])
def test_official_reward_survives_independent_capture_failure(capture):
    pytest.importorskip('benchflow')
    from runtime.agents.skillsbench import _map_evaluation
    result=_map_evaluation({}, {'completed':True,'rewards':{'reward':1.0}},capture)
    assert result[1] is True


def test_verifier_failure_is_unknown_even_with_complete_capture():
    pytest.importorskip('benchflow')
    from runtime.agents.skillsbench import _map_evaluation
    assert _map_evaluation({}, {'completed':False}, 'complete')[1] is None

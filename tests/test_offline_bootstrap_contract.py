"""Offline requirements must fail before container creation when inputs are absent."""
import copy

import pytest

from runtime.containers.policy import EnvironmentPolicy


def test_repo_pins_require_a_nonempty_wheelhouse(cfg):
    cfg['environment']['constraints'] = []
    cfg['environment']['repo_pins'] = {'astropy/astropy': ['pytest==4.6.11']}
    with pytest.raises(ValueError, match='Offline wheelhouse is empty'):
        EnvironmentPolicy(cfg)


def test_an_empty_pin_list_does_not_require_wheels(cfg):
    cfg['environment']['repo_pins'] = {'org/repo': []}
    assert EnvironmentPolicy(cfg).specs('org/repo') == []


def policy(cfg, tmp_path):
    wh = tmp_path / 'wheels'
    wh.mkdir()
    # Hash-only fixtures; real wheel compatibility belongs to container preflight.
    (wh / 'fixture-1.0-py3-none-any.whl').write_bytes(b'fixture')
    cfg['environment']['wheelhouse'] = str(wh)
    cfg['environment']['constraints'] = ['PyTest == 4.6.11', 'pytest_remotedata==0.3.2']
    cfg['environment']['repo_pins'] = {'astropy/astropy': ['pytest==4.6.11', 'pytest-remotedata==0.3.2']}
    return EnvironmentPolicy(cfg)


def test_duplicate_package_spellings_have_one_requirement(cfg, tmp_path):
    p = policy(cfg, tmp_path)
    expected = ['pytest==4.6.11', 'pytest-remotedata==0.3.2']
    assert p.specs('astropy/astropy') == expected
    identity = p.identity('astropy/astropy')
    assert identity['requirements'] == identity['constraints'] == expected
    assert len(identity['wheel_hashes']) == 1
    bootstrap = p.bootstrap('astropy/astropy')
    assert '--no-index' in bootstrap
    assert 'DYCON_NETWORK_FALLBACK' not in bootstrap


def test_conflicting_exact_pins_fail_with_package_and_versions(cfg, tmp_path):
    p = policy(cfg, tmp_path)
    p.environment = copy.deepcopy(p.environment)
    p.environment['repo_pins']['astropy/astropy'].append('PYTEST==8.0.0')
    with pytest.raises(ValueError, match=r'pytest: 4.6.11 vs 8.0.0'):
        p.specs('astropy/astropy')


@pytest.mark.parametrize('drift', [False, True])
def test_bootstrap_only_audit_repeats_shared_factory_without_running_harness(cfg, tmp_path, monkeypatch, drift):
    from types import SimpleNamespace

    from core.io import read_json, write_json
    from runtime.scripts import gold_image_audit as module

    instances_path = tmp_path / 'instances.json'
    write_json(instances_path, [{'instance_id': 'task', 'repo': 'org/repo', 'nas_image': {'manifest_sha256': 'sha'}}])
    creates = []
    closed = []

    class Store:
        def __init__(self, *args):
            self.image_id = 'image'
            self.record = {}

        def verify(self, **kwargs):
            return []

    class Factory:
        def __init__(self, cfg):
            self.images = {'task': {'image_id': 'image'}}
            self.client = SimpleNamespace(close=lambda: None)
            self.image_manager = SimpleNamespace(_exists=lambda image: False)

        def lock_images(self, instances):
            return self.images

        def create(self, row, context):
            creates.append(context)

            def start():
                write_json(context.output_dir / 'environment.json', {
                    'policy_hash': 'policy', 'packages_hash': context.attempt_id if drift else 'packages',
                    'python': 'fixture', 'constraints': [{'spec': 'pytest==4.6.11', 'satisfied': True}]})
                write_json(context.output_dir / 'bootstrap-receipt.json', {'status': 'ready'})

            return SimpleNamespace(start=start, id=context.attempt_id)

        def close(self, container):
            closed.append(container.id)

        def cleanup_owned(self, run_id):
            pass

    def forbidden(*args):
        raise AssertionError('Bootstrap preflight must not run the gold harness')

    monkeypatch.setattr('runtime.images.nas.NasImageStore', Store)
    monkeypatch.setattr(module, 'ContainerFactory', Factory)
    monkeypatch.setattr(module, 'install_harness_adapter', forbidden)
    summary = module.audit(tmp_path, instances_path, 'bootstrap-fixture', 1,
                           cfg=cfg, bootstrap_only=True, repeat_count=2)
    assert len(creates) == len(closed) == 2
    assert summary['mode'] == 'bootstrap_only'
    assert summary['resolved'] == 0
    assert summary['infra_error'] == int(drift)
    assert summary['results'][0]['environment_verified'] is (not drift)
    assert all(p['receipt_sha256'] for p in summary['results'][0]['probes'])
    frozen = read_json(tmp_path / 'state/gold-image-audit/bootstrap-fixture/instances.json')
    assert frozen == read_json(instances_path)


@pytest.mark.parametrize('failures', [0, 2])
def test_bootstrap_cli_returns_failure_when_preflight_did_not_succeed(tmp_path, monkeypatch, failures):
    import sys

    from runtime.scripts import gold_image_audit as module

    monkeypatch.setattr(sys, 'argv', ['gold_image_audit', '--root', str(tmp_path),
                                    '--instances', str(tmp_path / 'instances.json'), '--audit-id', 'preflight',
                                    '--bootstrap-only', '--repeat-count', '2'])
    monkeypatch.setattr(module, 'audit', lambda *args, **kwargs: {'infra_error': failures})
    if failures:
        with pytest.raises(SystemExit) as error:
            module.main()
        assert error.value.code == 1
    else:
        assert module.main() is None

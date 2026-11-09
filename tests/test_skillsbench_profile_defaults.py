from pathlib import Path

from core.io import read_json


def test_profile_defaults_match_public_oracle_proxy_and_allow_override(cfg, monkeypatch):
    from core import cli

    monkeypatch.setattr(cli, "resolve", lambda *args: cfg)
    base = ["skillsbench", "profile", "default-proxy", "--bridge-advertise", "10.0.0.1",
            "--output", "test-task:/out.txt"]
    cli.main(base)
    root = Path(cfg["project_root"])
    path = root / "runtime/agent-profiles/default-proxy/manifest.json"
    assert read_json(path)["use_host_proxy"] is True

    cli.main([*base[:2], "explicit-off", *base[3:], "--no-use-host-proxy"])
    path = root / "runtime/agent-profiles/explicit-off/manifest.json"
    assert read_json(path)["use_host_proxy"] is False

    cli.main([*base[:2], "offline-network", *base[3:], "--network", "none"])
    path = root / "runtime/agent-profiles/offline-network/manifest.json"
    assert read_json(path)["use_host_proxy"] is False


def test_profile_verifier_python_index_is_validated_and_offline_is_rejected():
    from runtime.agents.openhands import OpenHandsProfile, validate_profile
    base = {'bridge_listen': '127.0.0.1', 'bridge_advertise': '127.0.0.1'}
    profile = validate_profile({**base, 'verifier_python_index': 'https://mirrors.aliyun.com/pypi/simple/'})
    assert profile['verifier_python_index'].startswith('https://')
    import pytest
    with pytest.raises(ValueError, match='credentials'):
        OpenHandsProfile(**base, verifier_python_index='https://u:p@example.test/simple/')
    with pytest.raises(ValueError, match='Offline'):
        OpenHandsProfile(**base, network='none', verifier_python_index='https://example.test/simple/')


def test_copy_profile_preserves_adaptations_clears_qualification_and_is_immutable(cfg, monkeypatch):
    import pytest
    from core import cli
    from core.io import hash_json, sha256, write_json
    from runtime.agents.openhands import validate_profile
    monkeypatch.setattr(cli, 'resolve', lambda *args: cfg)
    root = Path(cfg['project_root'])
    proof = {'executed_tool_call_ids': ['tool'], 'requests': ['request']}
    proof['receipt_sha256'] = hash_json(proof)
    source = validate_profile({'bridge_listen': '10.0.0.1', 'bridge_advertise': '10.0.0.1',
        'network': 'public', 'use_host_proxy': True, 'cpu_quota': 'unenforced_with_receipt',
        'request_roles': 'openhands-1.22.1', 'qualification': proof,
        'output_paths': {'task': ['/root/result']}, 'memory_overrides': {'task': 8192},
        'verifier_timeout_overrides': {'task': 1600}, 'agent_idle_seconds': 321})
    path = root / 'source-profile.json'
    write_json(path, source)
    original = path.read_bytes()
    args = ['skillsbench', 'profile', 'copied', '--from-profile', str(path),
            '--verifier-python-index', 'https://mirror.test/simple/']
    cli.main(args)
    target = root / 'runtime/agent-profiles/copied/manifest.json'
    result = read_json(target)
    assert result['request_roles'] == 'unqualified' and result['qualification'] is None
    for key in ('cpu_quota', 'output_paths', 'memory_overrides', 'verifier_timeout_overrides',
                'agent_idle_seconds', 'use_host_proxy'):
        assert result[key] == source[key]
    assert path.read_bytes() == original
    digest = sha256(target)
    cli.main(args)
    assert sha256(target) == digest
    with pytest.raises(FileExistsError):
        cli.main([*args[:-1], 'https://another.test/simple/'])


def test_legacy_profile_optional_fields_do_not_change_normalized_identity():
    from runtime.agents.openhands import validate_profile
    from core.io import hash_json
    legacy = validate_profile({'bridge_listen': '127.0.0.1', 'bridge_advertise': '127.0.0.1'})
    assert 'verifier_python_index' not in legacy
    assert 'verifier_index_compatibility' not in legacy
    assert hash_json(validate_profile(legacy)) == hash_json(legacy)


def test_reject_malformed_index_urls():
    import pytest
    from runtime.agents.openhands import validate_profile
    base = {'bridge_listen': '127.0.0.1', 'bridge_advertise': '127.0.0.1'}
    for value in ('', ' https://mirror.test/simple/', 'https://mirror.test/\n',
                  'file:///tmp/simple', 'https://mirror.test:bogus/simple/',
                  'https://mirror.test:99999/simple/', 'https://mirror.test/simple/?token=secret',
                  'https://mirror.test/simple/#fragment', 'https://mirror.test\\@other.test/simple/'):
        with pytest.raises(ValueError):
            validate_profile({**base, 'verifier_python_index': value})

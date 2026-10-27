import hashlib
import json
from pathlib import Path

import pytest
import requests

from benchmark_evaluation.metadata import make_cached_get, prefetch
from core.io import hash_json


def response(status, body):
    r = requests.Response()
    r.status_code, r._content = status, body
    return r


def test_http_status_and_sha_survive_offline_cache(tmp_path):
    url = 'https://raw.githubusercontent.com/org/repo/abc/requirements.txt'
    get = make_cached_get(lambda *a, **k: response(200, b'pytest==4.6.11\n'), tmp_path, 1)
    assert get(url).status_code == 200
    cached = make_cached_get(lambda *a, **k: pytest.fail('unexpected network'), tmp_path, 1)
    assert cached(url).status_code == 200
    p = tmp_path / (hash_json(url) + '.txt')
    p.write_bytes(b'changed')
    with pytest.raises(ValueError, match='integrity'):
        cached(url)


def test_retry_alternate_source_keeps_canonical_identity(tmp_path):
    seen = []
    def get(url, **kwargs):
        seen.append(url)
        if 'raw.githubusercontent.com' in url:
            raise requests.ConnectionError('reset')
        return response(200, b'pytest==4.6.11\n')
    url = 'https://raw.githubusercontent.com/org/repo/abc/requirements.txt'
    assert make_cached_get(get, tmp_path, 1)(url).status_code == 200
    receipt = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert receipt['url'] == url
    assert receipt['source_url'].startswith('https://cdn.jsdelivr.net/')
    assert receipt['sha256'] == hashlib.sha256(b'pytest==4.6.11\n').hexdigest()
    assert len(seen) == 2


def test_prefetch_uses_official_recursive_reader_and_setup_commit(tmp_path, monkeypatch):
    from swebench.harness.test_spec import python as official
    instances = tmp_path / 'instances.json'
    instances.write_text(json.dumps([{'instance_id':'task','repo':'org/repo',
        'base_commit':'base','environment_setup_commit':'setup'}]))
    def read_requirements(repo, commit):
        assert commit == 'setup'
        first = official.requests.get(f'https://raw.githubusercontent.com/{repo}/{commit}/requirements.txt')
        nested = official.requests.get(f'https://raw.githubusercontent.com/{repo}/{commit}/tests.txt')
        return first.text + nested.text
    monkeypatch.setattr(official, 'get_requirements_by_commit', read_requirements)
    monkeypatch.setattr(requests.Session, 'get', lambda *a, **k: response(200, b'pytest==4.6.11\n'))
    cfg = {'project_root':str(tmp_path), 'environment':{'requirements_cache':str(tmp_path/'cache')},
        'timeouts':{'metadata_request_seconds':1}}
    output = tmp_path/'receipt.json'
    result = prefetch(cfg, instances, output)
    assert result['status'] == 'complete'
    assert len(result['resources']) == 2
    with pytest.raises(FileExistsError):
        prefetch(cfg, instances, output)


def test_refresh_rejects_changed_pinned_resource_without_replacing_cache(tmp_path):
    url = 'https://raw.githubusercontent.com/org/repo/abc/requirements.txt'
    make_cached_get(lambda *a, **k: response(200, b'original'), tmp_path, 1)(url)
    with pytest.raises(ValueError, match='Immutable'):
        make_cached_get(lambda *a, **k: response(200, b'changed'), tmp_path, 1, refresh=True)(url)
    assert next(tmp_path.glob('*.txt')).read_bytes() == b'original'

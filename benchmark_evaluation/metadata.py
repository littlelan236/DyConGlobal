"""Exact requirements metadata cache with bounded retries and atomic writes."""

from __future__ import annotations

import os
import hashlib
import tempfile
import time
from pathlib import Path

import requests

from core.io import hash_json, now, read_json, sha256, write_json
from core.locking import file_lock


def make_cached_get(original_get, cache: Path, timeout: int, attempts=3, refresh=False):
    cache.mkdir(parents=True, exist_ok=True)

    def get(url, *args, **kwargs):
        if not url.startswith("https://raw.githubusercontent.com/"):
            return original_get(url, *args, **kwargs)
        path = cache / (hash_json(url) + ".txt")
        if path.exists() and not refresh:
            with file_lock(str(path) + '.lock'):
                receipt_path = path.with_suffix('.json')
                proof = read_json(receipt_path) if receipt_path.exists() else None
                if proof and (proof['url'] != url or proof['sha256'] != sha256(path)):
                    raise ValueError(f'Requirements metadata cache integrity error: {path}')
                response = requests.Response()
                response.status_code = proof['status_code'] if proof else 200
                response._content = path.read_bytes()
                response.url = url
                if proof is None:
                    write_json(receipt_path, {'schema_version': 1, 'url': url, 'source_url': None,
                        'status_code': 200, 'sha256': sha256(path), 'recorded_at': now(),
                        'origin': 'legacy-cache-unattributed'})
                return response
        parts = url.removeprefix("https://raw.githubusercontent.com/").split("/", 3)
        candidates = [url]
        if len(parts) == 4:
            owner, repo, revision, resource = parts
            candidates.append(f"https://cdn.jsdelivr.net/gh/{owner}/{repo}@{revision}/{resource}")
        kwargs.setdefault("timeout", timeout)
        error = None
        for attempt in range(attempts):
            for candidate in candidates:
                try:
                    response = original_get(candidate, *args, **kwargs)
                    if response.status_code in (404, 410):
                        # Missing candidates must reach official path fallback;
                        # never persist them as successful requirements content.
                        return response
                    response.raise_for_status()
                except requests.RequestException as exc:
                    error = exc
                    continue
                fd, temporary = tempfile.mkstemp(dir=cache, prefix=".metadata-")
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(response.content)
                        stream.flush()
                        os.fsync(stream.fileno())
                    with file_lock(str(path) + '.lock'):
                        if path.exists() and sha256(path) != sha256(Path(temporary)):
                            raise ValueError(f'Immutable requirements metadata changed: {url}')
                        os.replace(temporary, path)
                        write_json(path.with_suffix('.json'), {'schema_version': 1,
                            'url': url, 'source_url': candidate,
                            'status_code': response.status_code, 'sha256': sha256(path),
                            'recorded_at': now(), 'origin': 'network'})
                finally:
                    Path(temporary).unlink(missing_ok=True)
                return response
            if attempt + 1 < attempts:
                time.sleep(min(0.5 * 2**attempt, 2))
        raise error

    return get


def prefetch(cfg, instances_path: Path, output: Path, proxy=None, refresh=False):
    """Warm the same exact-URL cache via the official recursive requirements reader.

    Runs in its own CLI process. Real missing candidate paths retain their HTTP
    status so replay follows official fallback instead of inventing requirements.
    """
    from swebench.harness.test_spec import python as official

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    instances_path = Path(instances_path)
    instances = read_json(instances_path)
    ids = [row['instance_id'] for row in instances]
    if not instances or len(ids) != len(set(ids)):
        raise ValueError('Expected nonempty unique instance records')
    cache = Path(cfg['environment']['requirements_cache'])
    if not cache.is_absolute():
        cache = Path(cfg['project_root']) / cache
    session = requests.Session()
    session.trust_env = False
    if proxy:
        session.proxies.update(http=proxy, https=proxy)
    getter = make_cached_get(session.get, cache, cfg['timeouts']['metadata_request_seconds'], refresh=refresh)
    fetched, tasks = {}, []

    def tracked(url, *args, **kwargs):
        response = getter(url, *args, **kwargs)
        path = cache / (hash_json(url) + '.txt')
        proof = read_json(path.with_suffix('.json')) if path.with_suffix('.json').exists() else {}
        fetched[url] = {'url': url, 'status_code': response.status_code,
            'sha256': hashlib.sha256(response.content).hexdigest(), 'cache_path': str(path),
            'source_url': proof.get('source_url') or response.url or url,
            'origin': proof.get('origin', 'uncached-http-response')}
        return response

    original = official.requests.get
    started = now()
    official.requests.get = tracked
    try:
        for row in instances:
            commit = row.get('environment_setup_commit') or row['base_commit']
            try:
                value = official.get_requirements_by_commit(row['repo'], commit)
                tasks.append({'instance_id': row['instance_id'], 'repo': row['repo'],
                    'environment_commit': commit, 'status': 'complete',
                    'requirements_sha256': hashlib.sha256(value.encode()).hexdigest()})
            except Exception as exc:
                tasks.append({'instance_id': row['instance_id'], 'repo': row['repo'],
                    'environment_commit': commit, 'status': 'failed', 'error': repr(exc)})
    finally:
        official.requests.get = original
        session.close()
    result = {'schema_version': 1, 'status': 'complete' if all(
        row['status'] == 'complete' for row in tasks) else 'incomplete',
        'started_at': started, 'finished_at': now(), 'instances': str(instances_path),
        'instances_sha256': sha256(instances_path), 'cache': str(cache),
        'proxy_used': bool(proxy), 'refresh': refresh, 'tasks': tasks, 'resources': list(fetched.values())}
    write_json(output, result)
    return result

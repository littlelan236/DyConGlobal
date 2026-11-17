"""Resource identity, isolation, corruption checks and real proxy replay contracts."""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
import pytest
from core.io import sha256
from runtime.backends.harbor.resource_store import Store, request_key
from runtime.backends.harbor import resource_cache
from runtime.backends.harbor.task_dependencies import task_dependency_overlay, resource_cache_setup_command


def test_cache_key_separates_credentials_ranges_and_payloads():
    key = request_key('GET', 'https://example/a', {})
    for method, url, headers, body in [
        ('HEAD', 'https://example/a', {}, b''),
        ('GET', 'https://example/b', {}, b''),
        ('GET', 'https://example/a', {'authorization': 'Bearer secret'}, b''),
        ('GET', 'https://example/a', {'cookie': 'session=secret'}, b''),
        ('GET', 'https://example/a', {'range': 'bytes=0-2'}, b''),
        ('GET', 'https://example/a', {}, b'body')]:
        assert key != request_key(method, url, headers, body)


def test_blob_seal_detects_byte_and_index_mutation(tmp_path):
    import hashlib
    store = Store(tmp_path)
    f = store.writer(); f.write(b'abc')
    digest = hashlib.sha256(b'abc').hexdigest()
    store.publish('key', f, digest, 3, 200, [])
    from core.io import write_json
    write_json(tmp_path / 'record.json', {'mode': 'record'})
    resource_cache.seal(tmp_path)
    assert resource_cache.verify(tmp_path)['entries'] == 1
    (tmp_path / 'blobs' / digest).write_bytes(b'bad')
    with pytest.raises(ValueError, match='bytes changed'):
        resource_cache.verify(tmp_path)
    (tmp_path / 'blobs' / digest).write_bytes(b'abc')
    with store.connect() as db:
        db.execute('DELETE FROM entries')
    with pytest.raises(ValueError, match='index differs'):
        resource_cache.verify(tmp_path)


def test_overlay_is_public_only_and_private_key_is_not_mounted(tmp_path):
    ca = tmp_path / 'ca.pem'; ca.write_text('ca')
    cache = dict(ca_bundle=str(ca), ca_sha256=sha256(ca), endpoint='http://user:token@host:1234')
    mounts, env = task_dependency_overlay({'task_resource_cache': cache}, 'public')
    assert [x['source'] for x in mounts] == [str(ca)]
    assert all(x['read_only'] for x in mounts)
    assert env['https_proxy'] == env['HTTPS_PROXY'] == cache['endpoint']
    assert env['UV_NATIVE_TLS'] == 'true'
    assert '0.0.0.0' in env['NO_PROXY']
    assert env['SSL_CERT_FILE'] == mounts[0]['target']
    assert 'Acquire::https::Proxy' in resource_cache_setup_command(cache)
    assert task_dependency_overlay({'task_resource_cache': cache}, 'no_network') == ([], {})
    ca.write_text('changed')
    with pytest.raises(ValueError, match='trust bundle changed'):
        task_dependency_overlay({'task_resource_cache': cache}, 'public')


def test_real_http_record_and_strict_replay_without_upstream(tmp_path):
    root = Path.cwd()
    runtime = root / 'runtime/resource-cache-runtime/venv/bin/mitmdump'
    if not runtime.exists():
        pytest.skip('Requires optional isolated cache runtime')
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    class Upstream(BaseHTTPRequestHandler):
        requests = 0
        retry_requests = 0
        def do_GET(self):
            type(self).requests += 1
            if self.path == '/optional':
                self.send_error(404); return
            if self.path == '/conditional' and self.headers.get('If-Modified-Since'):
                self.send_response(304); self.end_headers(); return
            if self.path == '/retry':
                type(self).retry_requests += 1
                if type(self).retry_requests == 1:
                    self.send_error(503); return
            data = (str(type(self).requests).encode() if '/dists/' in self.path
                    else b'cached-download' * 100000)
            self.send_response(200); self.send_header('Content-Length', str(len(data)))
            self.send_header('Vary', 'Origin,Access-Control-Request-Method,Access-Control-Request-Headers')
            self.end_headers(); self.wfile.write(data)
        def do_HEAD(self):
            type(self).requests += 1
            self.send_response(200)
            self.send_header('Content-Length', str(len(b'cached-download' * 100000)))
            self.end_headers()
        def log_message(self, *args):
            pass
    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    directory = tmp_path / 'cache'
    resource_cache.prepare(root, directory, f'http://127.0.0.1:{port}')
    item = resource_cache.descriptor(directory)[1]
    def fetch(path, extra=()):
        return subprocess.run(['curl', '--noproxy', '', '-fsS', '--max-time', '10',
            '--proxy', item['endpoint'], *extra, f'http://127.0.0.1:{upstream.server_port}/{path}'],
            capture_output=True, timeout=15)
    try:
        resource_cache.start(root, directory, 'record')
        first = fetch('resource'); assert first.returncode == 0, first.stderr
        second = fetch('resource'); assert second.stdout == first.stdout
        assert Upstream.requests == 1
        head_args = ['curl', '--silent', '--show-error', '--noproxy', '', '--proxy',
                     resource_cache.descriptor(directory)[1]['endpoint'], '--head',
                     f'http://127.0.0.1:{upstream.server_port}/head']
        first_head = subprocess.run(head_args, capture_output=True, timeout=10)
        second_head = subprocess.run(head_args, capture_output=True, timeout=10)
        assert first_head.returncode == second_head.returncode == 0
        assert b'content-length: 1500000' in second_head.stdout.lower()
        assert Upstream.requests == 2
        # Mutable APT indexes must refresh during recording, then freeze for replay.
        old_index = fetch('dists/stable/InRelease')
        new_index = fetch('dists/stable/InRelease')
        assert old_index.stdout != new_index.stdout
        assert fetch('retry').returncode == 22
        conditional = fetch('conditional', ('--header', 'If-Modified-Since: Wed, 21 Oct 2026 05:15:44 GMT'))
        assert conditional.returncode == 0 and conditional.stdout == b''
        warmed = resource_cache.warm(directory, workers=1, max_time=10, attempts=2)
        assert warmed['completed'] == 2 and warmed['uncovered'] == 0
        assert fetch('optional').returncode == 22
        resource_cache.stop(directory)
        upstream.shutdown(); upstream.server_close()
        # HTTPS cache hits must work before DNS or an upstream TLS connection exists.
        import hashlib
        tls_body = b'HTTPS replay without upstream'
        store = Store(directory)
        tls_url = 'https://dycon-upstream-unreachable.invalid/resource'
        tls_headers = {'user-agent': 'dycon-replay-test', 'accept': '*/*'}
        file = store.writer(); file.write(tls_body)
        store.publish(request_key('GET', tls_url, tls_headers), file,
                      hashlib.sha256(tls_body).hexdigest(), len(tls_body), 200,
                      [('content-length', str(len(tls_body)))])
        resource_cache.seal(directory)
        resource_cache.start(root, directory, 'replay')
        replay = fetch('resource'); assert replay.returncode == 0, replay.stderr
        assert replay.stdout == first.stdout
        derived_head = fetch('resource', ('--head',))
        assert derived_head.returncode == 0
        assert b'content-length: 1500000' in derived_head.stdout.lower()
        ranged = fetch('resource', ('--range', '2-8'))
        assert ranged.returncode == 0 and ranged.stdout == first.stdout[2:9]
        suffix = fetch('resource', ('--range', '-5'))
        assert suffix.returncode == 0 and suffix.stdout == first.stdout[-5:]
        mismatched = fetch('resource', ('--range', '2-8', '--header', 'If-Range: unknown'))
        assert mismatched.returncode == 0 and mismatched.stdout == first.stdout
        assert fetch('resource', ('--range', '999999999-')).returncode == 22
        miss = fetch('absent'); assert miss.returncode == 22
        # Replay keeps the exact failed request for later warming without changing its seal.
        misses = Store(directory).missing_requests()
        assert any(req['url'].endswith('/absent') for _, req in misses)
        target = Store(tmp_path / 'next-cache')
        assert target.import_missing_requests(directory) == 1
        assert target.import_missing_requests(directory) == 0
        assert len(target.missing_requests()) == 1
        assert target.missing_requests()[0][1]['url'].endswith('/absent')
        assert resource_cache.verify(directory)['entries'] > 0
        with pytest.raises(ValueError, match='sealed'):
            Store(directory).import_missing_requests(target.root)
        with pytest.raises(ValueError, match='sealed'):
            resource_cache.warm(directory)
        counts = resource_cache.status(directory)['events']
        assert counts['hit'] >= 3 and counts['denied'] == 1
        assert fetch('dists/stable/InRelease').stdout == new_index.stdout
        assert fetch('retry').returncode == 0
        assert fetch('conditional').stdout == first.stdout
        assert fetch('optional').returncode == 22
        assert resource_cache.status(directory)['events']['denied'] == 1
        tls = subprocess.run(['curl', '--noproxy', '', '-fsS', '--max-time', '10',
            '--user-agent', 'dycon-replay-test', '--proxy', item['endpoint'],
            '--cacert', item['ca_bundle'], tls_url], capture_output=True, timeout=15)
        assert tls.returncode == 0, tls.stderr
        assert tls.stdout == tls_body
    finally:
        resource_cache.stop(directory)
        upstream.server_close()


def test_new_optional_variants_preserve_old_cache_keys():
    import hashlib
    headers = ('accept', 'accept-encoding', 'accept-language', 'range', 'authorization', 'cookie',
               'user-agent', 'x-pjax', 'x-pjax-container', 'turbo-visit', 'turbo-frame',
               'x-requested-with', 'x-github-client-version', 'sec-fetch-site')
    values = dict.fromkeys(headers, '')
    old = hashlib.sha256(json.dumps(['GET', 'https://example/a', values,
                                    hashlib.sha256(b'').hexdigest()], sort_keys=True).encode()).hexdigest()
    assert request_key('GET', 'https://example/a', {}) == old
    assert request_key('GET', 'https://example/a', {'origin': 'https://a'}) != old


def test_mirror_configuration_is_frozen_and_requires_stopped_record_service(tmp_path):
    from core.io import write_json
    write_json(tmp_path / 'record.json', {'mode': 'record'})
    resource_cache.configure_upstreams(tmp_path, ['huggingface.co=hf-mirror.com'])
    assert json.loads((tmp_path / 'record.json').read_text())['upstream_hosts'] == {'huggingface.co': 'hf-mirror.com'}
    (tmp_path / 'seal.json').write_text('{}')
    with pytest.raises(ValueError, match='Stop the mutable'):
        resource_cache.configure_upstreams(tmp_path, [])


def test_admit_verifies_bytes_and_preserves_recorded_request_identity(tmp_path):
    import hashlib
    directory = tmp_path / 'cache'
    store = Store(directory)
    headers = {'user-agent': 'Wget/1.21.4', 'accept': '*/*'}
    url = 'https://example.invalid/dataset.tar.gz'
    key = request_key('GET', url, headers)
    store.record_request(key, 'GET', url, headers)
    source = tmp_path / 'download'; source.write_bytes(b'verified bytes')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        resource_cache.admit(directory, key, source, '0' * 64)
    assert store.get(key) is None and not list((directory / 'partial').iterdir())
    receipt = resource_cache.admit(directory, key, source, digest)
    assert receipt['sha256'] == store.get(key)['sha256'] == digest
    assert store.get(request_key('GET', url, {'user-agent': 'different'})) is None
    (directory / 'seal.json').write_text('{}')
    with pytest.raises(ValueError, match='sealed'):
        resource_cache.admit(directory, key, source, digest)


def test_hf_session_telemetry_does_not_prevent_fresh_container_replay(tmp_path):
    import hashlib
    url = 'https://huggingface.co/org/model/resolve/main/config.json'
    first = {'user-agent': 'unknown/None; hf_hub/0.36.2; transformers/4.56.0; session_id/' + 'a' * 32}
    second = {'user-agent': first['user-agent'].replace('a' * 32, 'b' * 32)}
    assert request_key('GET', url, first) == request_key('GET', url, second)
    cdn = 'https://cas-bridge.xethub.hf.co/xet-bridge-us/model?Policy=frozen'
    assert request_key('GET', cdn, first) == request_key('GET', cdn, second)
    assert request_key('GET', cdn + '&X-Amz-Signature=abc', first) != request_key('GET', cdn + '&X-Amz-Signature=def', second)
    assert request_key('GET', 'https://example.com/file', first) != request_key('GET', 'https://example.com/file', second)
    changed = {'user-agent': second['user-agent'].replace('4.56.0', '4.48.3')}
    assert request_key('GET', url, first) != request_key('GET', url, changed)
    store = Store(tmp_path)
    store.record_request('old-key', 'GET', url, first)
    writer = store.writer(); writer.write(b'config')
    digest = hashlib.sha256(b'config').hexdigest()
    store.publish('old-key', writer, digest, 6, 200, [])
    assert store.normalize_requests() == 1
    assert store.get(request_key('GET', url, second))['sha256'] == digest
    assert store.normalize_requests() == 0


def test_debian_archive_route_is_limited_to_retired_bullseye_security():
    from runtime.backends.harbor.resource_store import archived_debian_resource
    prefix = 'http://deb.debian.org/debian-security/'
    assert archived_debian_resource(prefix + 'dists/bullseye-security/InRelease')
    assert archived_debian_resource(prefix + 'pool/updates/main/c/curl/curl_7.74.0-1.3%2bdeb11u16_amd64.deb')
    assert not archived_debian_resource(prefix + 'dists/bookworm-security/InRelease')
    assert not archived_debian_resource(prefix + 'pool/updates/main/c/curl/curl_7.88.1-10+deb12u15_amd64.deb')


def test_forbidden_package_index_is_negative_probe_only():
    from runtime.backends.harbor.resource_store import public_package_index_absence
    url = 'https://download.pytorch.org/whl/cpu/pip/'
    assert public_package_index_absence('GET', url, {}, 403)
    assert public_package_index_absence('HEAD', url, {}, 403)
    assert public_package_index_absence('GET', url, {'cookie': 'cloudfront=x'}, 403)
    for method, changed, headers, status in [
        ('POST', url, {}, 403), ('GET', url, {}, 503),
        ('GET', url + '?token=x', {}, 403),
        ('GET', url, {'authorization': 'Bearer x'}, 403),
        ('GET', url.replace('download.pytorch.org', 'example.com'), {}, 403),
        ('GET', url.replace('pip/', 'torch.whl'), {}, 403)]:
        assert not public_package_index_absence(method, changed, headers, status)


@pytest.mark.parametrize('method,extra', [('HEAD', {}), ('GET', {'range': 'bytes=2-8'}), ('GET', {'range': 'bytes=-4'})])
def test_verified_admission_derives_head_and_ranges_without_crossing_variants(tmp_path, method, extra):
    import hashlib
    store = Store(tmp_path / 'cache')
    url = 'https://example.invalid/wheel.whl'
    headers = {'user-agent': 'uv/test', 'accept-encoding': 'identity', **extra}
    key = request_key(method, url, headers)
    store.record_request(key, method, url, headers)
    source = tmp_path / 'wheel'; source.write_bytes(b'0123456789')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    resource_cache.admit(store.root, key, source, digest)
    resolved, entry, byte_range = store.replay_entry(method, url, headers)
    assert entry['sha256'] == digest and entry['size'] == 10
    assert byte_range == extra.get('range')
    for name, value in [('authorization', 'Bearer private'), ('cookie', 'session=x'),
                        ('user-agent', 'pip/other'), ('accept-encoding', 'gzip')]:
        assert store.replay_entry(method, url, {**headers, name: value})[1] is None
    # mitmproxy Headers retain original case during iteration.
    from collections import UserDict
    class Headers(UserDict):
        def get(self, name, default=None):
            return next((v for k, v in self.data.items() if k.lower() == name.lower()), default)
    upper = Headers({k.title(): v for k, v in headers.items()})
    assert store.replay_entry(method, url, upper)[1]['sha256'] == digest


def test_head_and_if_range_preserve_representation_and_validators(tmp_path):
    import hashlib
    store = Store(tmp_path)
    url = 'https://example.invalid/resource'
    writer = store.writer(); writer.write(b'0123456789')
    key = request_key('GET', url, {})
    store.publish(key, writer, hashlib.sha256(b'0123456789').hexdigest(), 10, 200,
                  [('etag', '"v1"'), ('content-length', '10')])
    assert store.replay_entry('HEAD', url, {})[1]['size'] == 10
    assert store.replay_entry('GET', url, {'range': 'bytes=1-3', 'if-range': '"v1"'})[2] == 'bytes=1-3'
    assert store.replay_entry('GET', url, {'range': 'bytes=1-3', 'if-range': '"v2"'})[2] is None
    assert store.replay_entry('GET', url, {'range': 'bytes=1-3', 'if-range': 'W/"v1"'})[2] is None
    assert store.replay_entry('GET', url, {'range': 'bytes=1-3,6-8'})[1] is None

"""Loaded only by the isolated pinned mitmdump, never by the runner."""
from __future__ import annotations
import hashlib
import json
import os
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from mitmproxy import http
# mitmproxy reloads addons while retaining imported modules. Refresh the policy
# module before resolving its public functions, so a live development reload
# cannot unload the cache and silently become a forwarding proxy.
import importlib.util
_snapshot_store = Path(__file__).with_name('resource_store.py')
if _snapshot_store.exists() and 'service-code' in _snapshot_store.parts:
    _spec = importlib.util.spec_from_file_location('dycon_cache_service_store', _snapshot_store)
    _store_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_store_module)
else:
    import importlib
    from runtime.backends.harbor import resource_store as _store_module
    importlib.reload(_store_module)
Store = _store_module.Store
VARIANT_HEADERS = _store_module.VARIANT_HEADERS
HOP_HEADERS = _store_module.HOP_HEADERS
request_key = _store_module.request_key
archived_debian_resource = _store_module.archived_debian_resource
public_package_index_absence = _store_module.public_package_index_absence
readonly_api_post = _store_module.readonly_api_post
public_stateless_response = _store_module.public_stateless_response



def cacheable_request(request):
    return (request.method in ('GET', 'HEAD') or
            (request.method == 'POST' and request.path.split('?', 1)[0].endswith('/git-upload-pack') and
             request.headers.get('content-type', '').split(';')[0] == 'application/x-git-upload-pack-request') or
            readonly_api_post(request.method, request.pretty_url, request.headers, request.raw_content or b''))


def mutable_repository_index(url):
    path = urlsplit(url).path
    return '/dists/' in path and '/by-hash/' not in path


class DownloadCache:
    def __init__(self):
        self.store = Store(os.environ['DYCON_RESOURCE_CACHE'])
        self.offline = os.environ['DYCON_RESOURCE_CACHE_MODE'] == 'replay'
        descriptor = json.loads((self.store.root / 'record.json').read_text())
        self.upstream_hosts = descriptor.get('upstream_hosts', {})
        self.pending = {}
        self.token = secrets.token_urlsafe(32)
        owner = self

        class Files(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                parts = self.path.strip('/').split('/')
                entry = owner.store.get(parts[-1]) if len(parts) == 2 and secrets.compare_digest(parts[0], owner.token) else None
                if not entry:
                    self.send_error(404)
                    return
                start, end = 0, entry['size'] - 1
                byte_range = self.headers.get('Range')
                if byte_range:
                    match = re.fullmatch(r'bytes=(\d*)-(\d*)', byte_range)
                    left, right = match.groups() if match else ('', '')
                    if left:
                        start = int(left)
                        end = min(int(right), end) if right else end
                    elif right and int(right):
                        start = max(0, entry['size'] - int(right))
                    else:
                        start = entry['size']
                    if start > end:
                        self.send_response_only(416)
                        self.send_header('Content-Range', f'bytes */{entry["size"]}')
                        self.send_header('Content-Length', '0')
                        self.end_headers()
                        return
                self.send_response_only(206 if byte_range else entry['status'])
                for name, value in entry['headers']:
                    if byte_range and name.lower() in ('content-length', 'content-range', 'accept-ranges'):
                        continue
                    self.send_header(name, value)
                if byte_range:
                    self.send_header('Content-Length', str(end - start + 1))
                    self.send_header('Content-Range', f'bytes {start}-{end}/{entry["size"]}')
                    self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Connection', 'close')
                self.end_headers()
                if self.command == 'HEAD':
                    return
                with (owner.store.root / 'blobs' / entry['sha256']).open('rb') as source:
                    source.seek(start)
                    remaining = end - start + 1
                    while remaining and (chunk := source.read(min(1024 * 1024, remaining))):
                        self.wfile.write(chunk)
                        remaining -= len(chunk)

            do_HEAD = do_GET

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Files)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        from core.io import write_json, sha256
        write_json(self.store.root / 'addon-ready.json', {
            'pid': os.getpid(), 'mode': 'replay' if self.offline else 'record',
            'script_sha256': sha256(Path(__file__)), 'script': str(Path(__file__).resolve())})

    def request(self, flow):
        request = flow.request
        key = request_key(request.method, request.pretty_url, request.headers, request.raw_content or b'')
        flow.metadata.update(cache_key=key, cache_host=request.pretty_host)
        eligible = cacheable_request(request)
        refresh = not self.offline and mutable_repository_index(request.pretty_url)
        flow.metadata['refresh'] = refresh
        if eligible:
            self.store.record_request(key, request.method, request.pretty_url, request.headers, request.raw_content or b'')
        entry = self.store.get(key) if eligible and not refresh else None
        archive = archived_debian_resource(request.pretty_url)
        if archive and not self.offline and entry and entry['status'] in (404, 410):
            entry = None
            flow.metadata['refresh'] = True
        blob_key, byte_range = key, None
        if eligible and not entry and not refresh:
            blob_key, entry, byte_range = self.store.replay_entry(
                request.method, request.pretty_url, request.headers, request.raw_content or b'')
        if entry:
            self.store.event('hit', request.pretty_host, key)
            flow.metadata['cache_hit'] = True
            # Stream from a private loopback file server, avoiding large in-memory bodies.
            request.url = f'http://127.0.0.1:{self.server.server_port}/{self.token}/{blob_key}'
            request.headers.clear()
            if byte_range:
                request.headers['Range'] = byte_range
            if request.method == 'POST':
                request.method = 'GET'
                request.raw_content = b''
            # Cached POST bodies are already part of the key; local blob service is GET-only.
            return
        flow.metadata['cache_eligible'] = eligible
        if refresh:
            # Record a full signed index even when the image has a local APT copy.
            request.headers.pop('if-modified-since', None)
            request.headers.pop('if-none-match', None)
        if not self.offline and (archive or request.host in self.upstream_hosts):
            request.host = 'archive.debian.org' if archive else self.upstream_hosts[request.host]
            request.port = 443
            request.scheme = 'https'
            request.headers['host'] = request.host
        self.store.event('denied' if self.offline else 'miss', request.pretty_host, key)
        if self.offline:
            flow.response = http.Response.make(504, b'DyCon resource cache miss; upstream disabled\n')
            flow.metadata['cache_hit'] = True

    def responseheaders(self, flow):
        if flow.metadata.get('cache_hit'):
            flow.response.stream = True
            return
        response = flow.response
        vary = {x.strip().lower() for x in response.headers.get('vary', '').split(',') if x.strip()}
        allowed_status = response.status_code in (200, 206, 301, 302, 303, 307, 308, 404, 410)
        allowed_status = allowed_status or public_package_index_absence(
            flow.request.method, flow.request.pretty_url, flow.request.headers, response.status_code)
        strip_cookie = bool(response.headers.get('set-cookie')) and public_stateless_response(
            flow.request.method, flow.request.pretty_url, flow.request.headers, response.status_code)
        eligible = (flow.metadata.get('cache_eligible', cacheable_request(flow.request)) and allowed_status
                    and (not response.headers.get('set-cookie') or strip_cookie) and not vary - set(VARIANT_HEADERS))
        if not eligible:
            response.stream = True
            self.store.event('uncacheable', flow.metadata['cache_host'], flow.metadata['cache_key'])
            return
        file = self.store.writer()
        headers = [(k, v) for k, v in response.headers.items(multi=True) if k.lower() not in HOP_HEADERS | {'content-length'} | ({'set-cookie'} if strip_cookie else set())]
        state = dict(file=file, hash=hashlib.sha256(), size=0, headers=headers,
                     expected=int(response.headers['content-length']) if 'content-length' in response.headers and flow.request.method != 'HEAD' else None)
        if strip_cookie:
            state['sanitization'] = {'key': flow.metadata['cache_key'], 'removed': ['set-cookie'],
                'original_headers_sha256': hashlib.sha256(json.dumps(list(response.headers.items(multi=True))).encode()).hexdigest()}
        self.pending[flow.id] = state

        def stream(chunk):
            file.write(chunk)
            state['hash'].update(chunk)
            state['size'] += len(chunk)
            return chunk
        response.stream = stream

    def response(self, flow):
        state = self.pending.pop(flow.id, None)
        if not state:
            return
        # Empty-body responses do not invoke the stream callback.
        if flow.response.raw_content:
            state['file'].write(flow.response.raw_content)
            state['hash'].update(flow.response.raw_content)
            state['size'] += len(flow.response.raw_content)
        expected = state['expected']
        if expected is not None and state['size'] != expected:
            self._discard(state)
            self.store.event('incomplete', flow.metadata['cache_host'], flow.metadata['cache_key'])
            return
        length = flow.response.headers.get('content-length', '0') if flow.request.method == 'HEAD' else str(state['size'])
        state['headers'].append(('content-length', length))
        self.store.publish(flow.metadata['cache_key'], state['file'], state['hash'].hexdigest(),
                           state['size'], flow.response.status_code, state['headers'],
                           refresh=flow.metadata.get('refresh', False))
        if state.get('sanitization'):
            with (self.store.root / 'response-sanitization.jsonl').open('a') as receipt:
                receipt.write(json.dumps(state['sanitization']) + '\n')
            self.store.event('public_cookie_stripped', flow.metadata['cache_host'], flow.metadata['cache_key'])
        self.store.event('stored', flow.metadata['cache_host'], flow.metadata['cache_key'])

    @staticmethod
    def _discard(state):
        state['file'].close()
        Path(state['file'].name).unlink(missing_ok=True)

    def error(self, flow):
        state = self.pending.pop(flow.id, None)
        if state:
            self._discard(state)
        if 'cache_key' in flow.metadata:
            self.store.event('error', flow.metadata['cache_host'], flow.metadata['cache_key'])

    def done(self):
        for state in self.pending.values():
            self._discard(state)
        self.server.shutdown()

addons = [DownloadCache()]

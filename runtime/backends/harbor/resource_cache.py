"""Managed shared download cache for public-network Harbor task containers."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
import secrets
import shutil
import signal
import ssl
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit
from core.io import read_json, write_json, sha256
from runtime.backends.harbor.resource_store import Store


def register(sub):
    parser = sub.add_parser('resource-cache', help='Shared curl/uv/HTTP download cache')
    parser.add_argument('operation', choices=['prepare', 'start', 'status', 'stop', 'seal', 'verify', 'prefetch', 'scan', 'publish', 'runtime-prepare', 'configure-upstreams', 'warm', 'admit'])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--destination', type=Path, help='Publish a sealed copy with independent service identity')
    parser.add_argument('--base-directory', type=Path, help='Preserve every response in a sealed baseline; add only new request keys (publish only)')
    parser.add_argument('--endpoint', help='Advertised HTTP proxy URL (prepare only)')
    parser.add_argument('--mode', choices=['record', 'replay'], default='record')
    parser.add_argument('--url', action='append', default=[])
    parser.add_argument('--url-file', type=Path)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--uv', type=Path)
    parser.add_argument('--upstream-map', action='append', default=[], metavar='ORIGINAL=MIRROR')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-time', type=int, default=900)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--requests-from', type=Path, help='Import missing request variants from a previous replay journal (warm only)')
    parser.add_argument('--request-key', help='Exact recorded request for a verified external download')
    parser.add_argument('--source-file', type=Path)
    parser.add_argument('--sha256', dest='expected_sha256')
    parser.add_argument('--content-type', default='application/octet-stream')


def _alive(state):
    try:
        fields = Path(f'/proc/{state["pid"]}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == state['start_ticks']
    except (FileNotFoundError, KeyError):
        return False


def descriptor(directory, mode='record'):
    directory = Path(directory).resolve()
    path = directory / ('replay.json' if mode == 'replay' else 'record.json')
    return path, read_json(path)


def validate_descriptor(path):
    item = read_json(path)
    if item['schema_version'] != 1 or sha256(item['ca_bundle']) != item['ca_sha256']:
        raise ValueError('Resource cache CA identity changed')
    state = read_json(Path(item['directory']) / 'service.json')
    if not _alive(state) or state['descriptor_sha256'] != sha256(path):
        raise ValueError('Configured resource cache service is absent or has another identity/mode')
    return item


def prepare(root, directory, endpoint):
    directory = Path(directory).resolve()
    parsed = urlsplit(endpoint or '')
    if (parsed.scheme != 'http' or not parsed.hostname or not parsed.port or parsed.username
            or parsed.path or parsed.query or parsed.fragment):
        raise ValueError('Specify an advertised http://HOST:PORT endpoint without credentials/path')
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'record.json').exists():
        raise FileExistsError('Cache already prepared')
    os.chmod(directory, 0o700)
    Store(directory)
    ca_dir = directory / 'private-ca'
    ca_dir.mkdir(mode=0o700)
    runtime = Path(root) / 'runtime/resource-cache-runtime/venv/bin'
    if not (runtime / 'mitmdump').is_file():
        raise ValueError('Pinned resource-cache runtime must be installed first')
    probe = subprocess.check_output([str(runtime / 'python'), '-c',
        'from mitmproxy.certs import CertStore; import sys; CertStore.from_store(sys.argv[1], "mitmproxy", 2048)',
        str(ca_dir)], timeout=30, stderr=subprocess.STDOUT)
    public = directory / 'public'
    public.mkdir(mode=0o755)
    bundle = public / 'ca-bundle.pem'
    system_ca = Path(ssl.get_default_verify_paths().cafile or '/etc/ssl/certs/ca-certificates.crt')
    bundle.write_bytes(system_ca.read_bytes() + b'\n' +
                      (ca_dir / 'mitmproxy-ca-cert.pem').read_bytes())
    token = secrets.token_urlsafe(32)
    proxy = f'http://dycon:{token}@{parsed.hostname}:{parsed.port}'
    value = dict(schema_version=1, directory=str(directory), endpoint=proxy, mode='record',
                 ca_bundle=str(bundle), ca_sha256=sha256(bundle), runtime=str(runtime),
                 runtime_version='mitmproxy==11.0.2', listen_host='0.0.0.0', port=parsed.port)
    write_json(directory / 'record.json', value)
    os.chmod(directory / 'record.json', 0o600)
    return dict(prepared=True, descriptor=str(directory / 'record.json'), endpoint=endpoint)


def _link_cached_blob(source, target):
    if not target.exists():
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)


def _record_store(directory):
    seal_path = directory / 'seal.json'
    if not seal_path.is_file():
        return directory
    target = directory / 'record-store'
    marker = 'base-seal-sha256'
    identity = sha256(seal_path)
    if target.exists():
        if (target / marker).read_text().strip() != identity:
            raise ValueError('Record store belongs to another sealed cache')
        return target

    verify(directory)
    entries = read_json(seal_path)['entries']
    with tempfile.TemporaryDirectory(prefix='.record-store-', dir=directory) as staging:
        store = Store(Path(staging) / 'store')
        with store.connect() as db:
            for key, entry in entries.items():
                blob = Path('blobs') / entry['sha256']
                _link_cached_blob(directory / blob, store.root / blob)
                db.execute('INSERT INTO entries VALUES(?,?)', (key, json.dumps(entry)))
        (store.root / marker).write_text(identity + '\n')
        # A failed seed can be retried without exposing a partially copied index.
        store.root.rename(target)
    return target


def start(root, directory, mode):
    directory = directory.resolve()
    path, item = descriptor(directory, mode)
    with (directory / '.start.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _start(root, directory, mode, path, item)


def _start(root, directory, mode, path, item):
    state_path = directory / 'service.json'
    if state_path.exists() and _alive(read_json(state_path)):
        raise ValueError('Cache service already running; stop before switching mode')
    store_directory = directory
    if mode == 'record':
        store_directory = _record_store(directory)
    else:
        verify(directory)
    # mitmproxy watches scripts for changes. Execute an immutable copy so that
    # editing project code cannot unload a live cache or interrupt a transfer.
    code_files = ['resource_proxy.py', 'resource_store.py']
    from core.io import hash_json
    code_identity = {name: sha256(Path(root) / 'runtime/backends/harbor' / name) for name in code_files}
    code_directory = directory / 'service-code' / hash_json(code_identity)
    code_directory.mkdir(parents=True, exist_ok=True)
    for name in code_files:
        target = code_directory / name
        if not target.exists():
            shutil.copyfile(Path(root) / 'runtime/backends/harbor' / name, target)
        if sha256(target) != code_identity[name]:
            raise ValueError('Cache service code snapshot changed')
    endpoint = urlsplit(item['endpoint'])
    args = [str(Path(item['runtime']) / 'mitmdump'), '--listen-host', item['listen_host'],
            '--listen-port', str(item['port']), '--set', 'connection_strategy=lazy',
            '--set', 'upstream_cert=false', '--set', 'ssl_insecure=false', '--set', 'flow_detail=0',
            '--set', 'confdir=' + str(directory / 'private-ca'),
            '--proxyauth', endpoint.username + ':' + endpoint.password,
            '-s', str(code_directory / 'resource_proxy.py')]
    env = dict(os.environ, DYCON_RESOURCE_CACHE=str(directory),
               DYCON_RESOURCE_CACHE_STORE=str(store_directory), DYCON_RESOURCE_CACHE_MODE=mode,
               PYTHONPATH=str(root))
    (directory / 'addon-ready.json').unlink(missing_ok=True)
    with (directory / 'service.log').open('ab') as log:
        process = subprocess.Popen(args, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=log, start_new_session=True)
    fields = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()
    state = dict(pid=process.pid, start_ticks=fields[19], descriptor_sha256=sha256(path), mode=mode, code_identity=code_identity, code_directory=str(code_directory))
    write_json(state_path, state)
    time.sleep(2)
    ready = directory / 'addon-ready.json'
    marker = read_json(ready) if ready.is_file() else {}
    if (not _alive(state) or marker.get('pid') != process.pid or marker.get('mode') != mode
            or marker.get('script_sha256') != code_identity['resource_proxy.py']):
        stop(directory)
        raise RuntimeError('Cache addon failed readiness validation; service stopped')
    return dict(started=True, pid=process.pid, mode=mode, store_directory=str(store_directory))


def stop(directory):
    state_path = directory / 'service.json'
    if not state_path.exists() or not _alive(read_json(state_path)):
        return dict(stopped=True, already_stopped=True)
    state = read_json(state_path)
    os.kill(state['pid'], signal.SIGTERM)
    for _ in range(100):
        if not _alive(state):
            return dict(stopped=True)
        time.sleep(.1)
    raise RuntimeError('Cache service has not exited')


def status(directory):
    state_path = directory / 'service.json'
    state = read_json(state_path) if state_path.exists() else {}
    store_directory = directory / 'record-store'
    if state.get('mode') != 'record' or not store_directory.is_dir():
        store_directory = directory
    store = Store(store_directory)
    with store.connect() as db:
        counts = dict(db.execute('SELECT kind, count(*) FROM events GROUP BY kind'))
        entries = db.execute('SELECT count(*) FROM entries').fetchone()[0]
        blobs = db.execute('SELECT metadata FROM entries').fetchall()
    return dict(running=_alive(state), mode=state.get('mode'), entries=entries,
                store_directory=str(store_directory),
                bytes=sum(json.loads(v[0])['size'] for v in blobs), events=counts,
                sealed=(directory / 'seal.json').is_file())


def seal(directory):
    if status(directory)['running']:
        raise ValueError('Stop service before sealing')
    if (directory / 'seal.json').exists():
        raise FileExistsError('Cache seal is immutable')
    store = Store(directory)
    with store.connect() as db:
        rows = db.execute('SELECT key, metadata FROM entries ORDER BY key').fetchall()
    entries = {k: json.loads(v) for k, v in rows}
    receipts = {name: sha256(directory / name) for name in
                ('admissions.jsonl', 'response-sanitization.jsonl', 'publication.json') if (directory / name).is_file()}
    value = dict(schema_version=1, entries=entries)
    if receipts:
        value['receipts'] = receipts
    write_json(directory / 'seal.json', value)
    path, item = descriptor(directory)
    item.update(mode='replay', seal_sha256=sha256(directory / 'seal.json'))
    write_json(directory / 'replay.json', item)
    os.chmod(directory / 'replay.json', 0o600)
    return dict(sealed=True, entries=len(entries), seal_sha256=item['seal_sha256'])


def verify(directory):
    _, item = descriptor(directory, 'replay')
    if sha256(directory / 'seal.json') != item['seal_sha256']:
        raise ValueError('Cache seal identity changed')
    sealed = read_json(directory / 'seal.json')
    entries = sealed['entries']
    for name, expected in sealed.get('receipts', {}).items():
        if name not in ('admissions.jsonl', 'response-sanitization.jsonl', 'publication.json') or sha256(directory / name) != expected:
            raise ValueError('Cache provenance receipt changed')
    store = Store(directory)
    with store.connect() as db:
        actual = {k: json.loads(v) for k, v in db.execute('SELECT key, metadata FROM entries')}
    if actual != entries:
        raise ValueError('Cache index differs from seal')
    for value in entries.values():
        blob = directory / 'blobs' / value['sha256']
        if blob.stat().st_size != value['size'] or sha256(blob) != value['sha256']:
            raise ValueError('Cached resource bytes changed')
    return dict(verified=True, entries=len(entries))


def prefetch(directory, urls):
    from concurrent.futures import ThreadPoolExecutor
    _, item = descriptor(directory)
    validate_descriptor(directory / 'record.json')
    endpoint = urlsplit(item['endpoint'])
    proxy = f'http://{endpoint.hostname}:{endpoint.port}'
    def fetch(url):
        # curl handles HTTPS CONNECT proxy authentication consistently across versions.
        result = subprocess.run(['curl', '--silent', '--show-error', '--location', '--fail',
            '--max-time', '45', '--proxy', proxy, '--proxy-user', endpoint.username + ':' + endpoint.password,
            '--cacert', item['ca_bundle'], '--output', '/dev/null', url], capture_output=True, timeout=50)
        return dict(host=urlsplit(url).hostname, url_sha256=hashlib.sha256(url.encode()).hexdigest(),
                    return_code=result.returncode, error=result.stderr.decode(errors='replace')[-300:])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, urls))
    write_json(directory / 'prefetch-last.json', results)
    return dict(total=len(results), completed=sum(r['return_code'] == 0 for r in results),
                failed=sum(r['return_code'] != 0 for r in results), receipt=str(directory / 'prefetch-last.json'))


def configure_upstreams(directory, mappings):
    """Freeze explicit mirror routing in the descriptor used by new runs."""
    import re
    if status(directory)['running'] or (directory / 'seal.json').exists():
        raise ValueError('Stop the mutable record service before changing mirror routes')
    path, item = descriptor(directory)
    hosts = {}
    for mapping in mappings:
        original, mirror = mapping.split('=', 1)
        if not all(re.fullmatch(r'[A-Za-z0-9.-]+', host) for host in (original, mirror)):
            raise ValueError('Mirror mapping requires two host names')
        hosts[original] = mirror
    item['upstream_hosts'] = hosts
    write_json(path, item)
    return dict(configured=True, upstream_hosts=hosts, descriptor_sha256=sha256(path))


def warm(directory, *, workers=4, max_time=900, attempts=3, requests_from=None, request_key_filter=None):
    """Retry incomplete downloads with original curl/uv/git request variants."""
    import base64
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    if not 1 <= workers <= 16 or not 1 <= attempts <= 10 or not 1 <= max_time <= 3600:
        raise ValueError('Invalid bounded warm concurrency/timeout/attempts')
    if (directory / 'seal.json').exists():
        raise ValueError('Cannot warm a sealed cache')
    path, item = descriptor(directory)
    validate_descriptor(path)
    store = Store(directory)
    imported = store.import_missing_requests(requests_from) if requests_from else 0
    normalized = store.normalize_requests()
    requests = store.missing_requests()
    if request_key_filter:
        requests = [p for p in requests if p[0] == request_key_filter]
        if not requests:
            raise ValueError('No missing recorded request matches --request-key')
    def fetch(pair):
        key, request = pair
        cmd = ['curl', '--silent', '--show-error', '--fail', '--noproxy', '',
               '--connect-timeout', '30', '--max-time', str(max_time),
               '--speed-limit', '1', '--speed-time', str(max_time if request['method'] == 'POST' else 45),
               '--proxy', item['endpoint'], '--cacert', item['ca_bundle'], '--output', '/dev/null',
               '--write-out', '%{http_code}',
               '--header', 'Accept:', '--user-agent', '']
        for name, value in request['headers'].items():
            if name.lower() not in ('host', 'connection', 'proxy-authorization', 'content-length',
                                    'if-modified-since', 'if-none-match'):
                cmd += ['--header', f'{name}: {value}']
        body = base64.b64decode(request['body'])
        with tempfile.NamedTemporaryFile(dir=directory / 'partial') as payload:
            if request['method'] == 'HEAD':
                cmd += ['--head']
            elif request['method'] == 'POST':
                payload.write(body); payload.flush()
                cmd += ['--request', 'POST', '--data-binary', '@' + payload.name]
            cmd += [request['url']]
            code = None
            http_status = None
            for attempt in range(attempts):
                if store.replay_entry(request['method'], request['url'], request['headers'], body)[1]:
                    break
                result = subprocess.run(cmd, capture_output=True, timeout=max_time + 15)
                code = result.returncode
                text = result.stdout.decode(errors='replace').strip()
                http_status = int(text[-3:]) if text[-3:].isdigit() else None
                # Negative index responses retain their original status for resolver fallback.
        return dict(key=key, host=urlsplit(request['url']).hostname,
                    cached=store.replay_entry(request['method'], request['url'], request['headers'], body)[1] is not None, return_code=code, http_status=http_status)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(fetch, requests))
    write_json(directory / 'warm-last.json', results)
    return dict(total=len(results), imported_requests=imported, normalized_requests=normalized, completed=sum(r['cached'] for r in results),
                uncovered=sum(not r['cached'] for r in results), receipt=str(directory / 'warm-last.json'))


def admit(directory, key, source, expected_sha256, content_type='application/octet-stream'):
    """Admit an independently verified complete file for one recorded request."""
    import re
    if (directory / 'seal.json').exists():
        raise ValueError('Cannot admit resources into a sealed cache')
    if not expected_sha256 or not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
        raise ValueError('A SHA256 digest is required')
    if not content_type or any(c in content_type for c in '\r\n'):
        raise ValueError('Invalid content type')
    store = Store(directory)
    with store.connect() as db:
        row = db.execute('SELECT request FROM requests WHERE key=?', (key,)).fetchone()
    if not row:
        raise ValueError('Select an existing recorded request key')
    request = json.loads(row[0])
    if (request['method'] not in ('GET', 'HEAD')
            or request['headers'].get('accept-encoding', '').lower() not in ('', 'identity')):
        raise ValueError('Only identity-encoded GET/HEAD resources may be admitted')
    selected_key = key
    # Explicit admission maps verified bytes only to the selected request variant.
    # HEAD/Range callers receive protocol-correct responses from this complete GET.
    headers = dict(request['headers'])
    headers.pop('range', None)
    headers.pop('if-range', None)
    import base64
    from runtime.backends.harbor.resource_store import request_key
    body = base64.b64decode(request['body'])
    if body:
        raise ValueError('File admission does not accept a request body')
    key = request_key('GET', request['url'], headers, body)
    store.record_request(key, 'GET', request['url'], headers, body)
    source = Path(source)
    writer = store.writer()
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                writer.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError('External resource SHA256 mismatch')
        existing = store.get(key)
        if existing and existing['sha256'] != expected_sha256:
            raise ValueError('A different completed response already exists for this request')
        store.publish(key, writer, expected_sha256, size, 200,
                      [('content-length', str(size)), ('content-type', content_type)])
        if store.get(key)['sha256'] != expected_sha256:
            raise ValueError('A concurrent response differs from the admitted resource')
    finally:
        writer.close()
        Path(writer.name).unlink(missing_ok=True)
    receipt = dict(key=key, selected_request_key=selected_key, sha256=expected_sha256, size=size, source=str(source.resolve()))
    with (directory / 'admissions.jsonl').open('a') as log:
        log.write(json.dumps(receipt) + '\n')
    store.event('admitted', urlsplit(request['url']).hostname, key)
    return dict(admitted=True, **receipt)


def prepare_runtime(root, uv):
    import shutil
    uv = str(uv or shutil.which('uv') or Path.home() / '.local/bin/uv')
    root = Path(root)
    runtime = root / 'runtime/resource-cache-runtime/venv'
    lock = root / 'runtime/backends/harbor/resource-cache-requirements.lock'
    if not runtime.exists():
        subprocess.run([uv, 'venv', '--python', str(root / '.venvs/runner/bin/python'), str(runtime)],
                       check=True, timeout=60, capture_output=True)
    subprocess.run([uv, 'pip', 'sync', '--python', str(runtime / 'bin/python'),
                    '--require-hashes', '--index-url', 'https://mirrors.aliyun.com/pypi/simple/', str(lock)],
                   check=True, timeout=600, capture_output=True)
    write_json(runtime.parent / 'manifest.json', dict(lock_sha256=sha256(lock), version='mitmproxy==11.0.2'))
    return dict(prepared=True, runtime=str(runtime), lock_sha256=sha256(lock))


def scan(directory, snapshot):
    import re
    snapshot = Path(snapshot).resolve()
    packages = snapshot.parent / 'packages' if snapshot.is_file() else snapshot / 'packages'
    if not packages.is_dir():
        raise ValueError('Expected an imported Harbor snapshot with packages/')
    found = {}
    tasks = 0
    for task in sorted(packages.iterdir()):
        if not task.is_dir():
            continue
        tasks += 1
        for section in ('solution', 'tests', 'environment'):
            for file in (task / section).rglob('*'):
                if not file.is_file() or file.stat().st_size > 2_000_000:
                    continue
                text = file.read_text(errors='replace')
                for url in re.findall(r"https?://[^\s\"'<>`]+", text):
                    url = url.rstrip(');,]')
                    if any(c in url for c in '$}{\\') or urlsplit(url).hostname in ('localhost', '127.0.0.1', 'example.com'):
                        continue
                    found.setdefault(url, set()).add(task.name)
    def downloadable(url):
        path = urlsplit(url).path
        return ('/resolve/' in path or '/releases/download/' in path or
                path.endswith(('.sh', '.tar.gz', '.tar.xz', '.tgz', '.zip', '.whl', '.pt', '.pth', '.onnx', '.jsonl.gz')))
    selected = sorted(u for u in found if downloadable(u))
    write_json(directory / 'inventory.json', dict(snapshot=str(snapshot), tasks=tasks,
        urls=[dict(url=u, tasks=sorted(t), prefetch=u in selected) for u,t in sorted(found.items())]))
    (directory / 'prefetch-urls.txt').write_text('\n'.join(selected) + '\n')
    return dict(tasks=tasks, distinct_urls=len(found), static_download_urls=len(selected),
                inventory=str(directory / 'inventory.json'), urls_file=str(directory / 'prefetch-urls.txt'))


def publish(root, directory, destination, endpoint, *, base_directory=None):
    """Publish complete resources, optionally retaining a validated frozen baseline."""
    from core.io import hash_json
    base = Path(base_directory).resolve() if base_directory else None
    if base is not None:
        if not (base / 'seal.json').is_file():
            raise ValueError('Publication baseline must be a sealed cache')
        verify(base)
    destination = Path(destination).resolve()
    prepare(root, destination, endpoint)
    source = Store(directory)
    source.normalize_requests()
    sources = [Store(base), source] if base is not None else [source]
    selected = {}
    retained_conflicts = []
    source_indices = []
    for store in sources:
        with store.connect() as db:
            rows = db.execute('SELECT key, metadata FROM entries ORDER BY key').fetchall()
        source_indices.append({'directory': str(store.root),
                               'entries_sha256': hash_json({k: json.loads(v) for k, v in rows})})
        for key, raw in rows:
            item = json.loads(raw)
            if key in selected:
                previous = json.loads(selected[key][1])
                if previous != item:
                    retained_conflicts.append(key)
                continue
            selected[key] = (store, raw)
    target = Store(destination)
    with target.connect() as db:
        for key, (store, raw) in sorted(selected.items()):
            item = json.loads(raw)
            blob = store.root / 'blobs' / item['sha256']
            if sha256(blob) != item['sha256'] or blob.stat().st_size != item['size']:
                raise ValueError('Source cached resource failed integrity check')
            dest = target.root / 'blobs' / item['sha256']
            _link_cached_blob(blob, dest)
            db.execute('INSERT INTO entries VALUES(?,?)', (key, raw))
    for name in ('admissions.jsonl', 'response-sanitization.jsonl'):
        receipts = [store.root / name for store in sources if (store.root / name).is_file()]
        if receipts:
            with (destination / name).open('wb') as stream:
                for receipt in receipts:
                    data = receipt.read_bytes()
                    stream.write(data)
                    if data and not data.endswith(b'\n'):
                        stream.write(b'\n')
    if base is not None:
        write_json(destination / 'publication.json', {
            'schema_version': 1, 'policy': 'sealed_baseline_then_new_request_keys',
            'base_directory': str(base), 'base_seal_sha256': sha256(base / 'seal.json'),
            'source_indices': source_indices, 'retained_conflict_keys': retained_conflicts,
            'entries': len(selected)})
    result = seal(destination)
    return dict(**result, descriptor=str(destination / 'replay.json'))


def dispatch(root, args, cfg=None):
    directory = args.directory.resolve()
    if args.operation == 'publish':
        if args.destination is None:
            raise ValueError('publish requires --destination and --endpoint')
        return publish(root, directory, args.destination, args.endpoint, base_directory=args.base_directory)
    if args.operation == 'configure-upstreams':
        return configure_upstreams(directory, args.upstream_map)
    if args.operation == 'warm':
        return warm(directory, workers=args.workers, max_time=args.max_time, attempts=args.attempts,
                    requests_from=args.requests_from, request_key_filter=args.request_key)
    if args.operation == 'admit':
        if args.source_file is None or not args.request_key:
            raise ValueError('admit requires --request-key, --source-file, and --sha256')
        return admit(directory, args.request_key, args.source_file, args.expected_sha256, args.content_type)
    if args.operation == 'runtime-prepare':
        return prepare_runtime(root, args.uv)
    if args.operation == 'scan':
        snapshot = args.snapshot or Path(root) / cfg['benchmark']['snapshot_ref']
        return scan(directory, snapshot)
    if args.operation == 'prepare':
        return prepare(root, directory, args.endpoint)
    if args.operation == 'start':
        return start(root, directory, args.mode)
    if args.operation == 'prefetch':
        urls = args.url + (args.url_file.read_text().splitlines() if args.url_file else [])
        return prefetch(directory, [u for u in urls if u and not u.startswith('#')])
    return globals()[args.operation](directory)

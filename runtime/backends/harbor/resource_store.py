"""Content-addressed HTTP download cache; no task or verifier code is stored here."""
from __future__ import annotations
import base64
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

VARIANT_HEADERS = ('accept', 'accept-encoding', 'accept-language', 'range', 'authorization', 'cookie', 'user-agent', 'x-pjax', 'x-pjax-container',
                   'turbo-visit', 'turbo-frame', 'x-requested-with', 'x-github-client-version', 'sec-fetch-site', 'git-protocol', 'content-type', 'origin', 'sec-fetch-mode', 'sec-fetch-dest', 'if-range', 'access-control-request-method', 'access-control-request-headers')
HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
               'te', 'trailer', 'transfer-encoding', 'upgrade'}


def request_key(method, url, headers, body=b''):
    values = {name: headers.get(name, '') for name in VARIANT_HEADERS
              if name not in ('git-protocol', 'content-type', 'origin', 'sec-fetch-mode', 'sec-fetch-dest', 'if-range', 'access-control-request-method', 'access-control-request-headers') or headers.get(name)}
    # Transformers adds a random process ID to telemetry; it does not select a model.
    host = urlsplit(url).hostname or ''
    if ((host in ('huggingface.co', 'hf-mirror.com')
         or host.endswith(('.huggingface.co', '.hf.co')))
            and 'hf_hub/' in values.get('user-agent', '')):
        values['user-agent'] = re.sub(r'; session_id/[0-9a-fA-F]{32}(?=;|$)', '', values['user-agent'])
    return hashlib.sha256(json.dumps([method, url, values, hashlib.sha256(body).hexdigest()],
                                    sort_keys=True).encode()).hexdigest()


def archived_debian_resource(url):
    parsed = urlsplit(url)
    return (parsed.hostname in ('deb.debian.org', 'security.debian.org') and
            (parsed.path.startswith('/debian-security/dists/bullseye-security/') or
             (parsed.path.startswith('/debian-security/pool/') and
              re.search(r'deb11(?:u\d+)?(?:_|%|\+)', parsed.path) is not None)))


def public_package_index_absence(method, url, headers, status):
    """PyTorch's S3 index returns 403 for packages absent from its extra index."""
    parsed = urlsplit(url)
    return (status == 403 and method in ('GET', 'HEAD')
            and parsed.hostname == 'download.pytorch.org'
            and re.fullmatch(r'/whl/(?:cpu|cu\d+)/[a-zA-Z0-9_.-]+/', parsed.path) is not None
            and not parsed.query and not headers.get('authorization'))


def readonly_api_post(method, url, headers, body):
    """RCSB GraphQL query documents only; no mutations, batches or other hosts."""
    parsed = urlsplit(url)
    if (method != 'POST' or parsed.scheme != 'https' or parsed.hostname != 'data.rcsb.org'
            or parsed.port not in (None, 443) or parsed.path != '/graphql' or parsed.query
            or headers.get('content-type', '').split(';')[0].lower() != 'application/json'
            or len(body) > 2 * 1024 * 1024):
        return False
    try:
        from graphql import parse
        from graphql.language.ast import OperationDefinitionNode, FragmentDefinitionNode
        data = json.loads(body)
        if not isinstance(data, dict) or set(data) - {'query', 'variables', 'operationName'}:
            return False
        if not isinstance(data.get('query'), str) or not data['query'].strip():
            return False
        document = parse(data['query'])
        operations = [n for n in document.definitions if isinstance(n, OperationDefinitionNode)]
        return bool(operations) and all(n.operation.value == 'query' for n in operations) and all(
            isinstance(n, (OperationDefinitionNode, FragmentDefinitionNode)) for n in document.definitions)
    except Exception:
        # graphql syntax errors are intentionally fail-closed too.
        return False


def public_stateless_response(method, url, headers, status):
    """Explicit public lookup/artifact endpoints may set incidental CDN cookies."""
    parsed = urlsplit(url)
    if (method not in ('GET', 'HEAD') or status != 200 or parsed.scheme != 'https'
            or parsed.port not in (None, 443) or parsed.query
            or headers.get('authorization') or headers.get('cookie')):
        return False
    if parsed.hostname == 'pubchem.ncbi.nlm.nih.gov':
        return method == 'GET' and re.fullmatch(
            r'/rest/pug/compound/(?:smiles/[^/]+/cids|cid/[0-9]+/synonyms)/TXT', parsed.path) is not None
    # Public immutable wheel artifacts/PEP 658 metadata only. Never package
    # indexes, arbitrary Cloudflare sites, or authenticated request variants.
    return (parsed.hostname in ('download.pytorch.org', 'download-r2.pytorch.org')
            and re.fullmatch(r'/whl/(?:[A-Za-z0-9_.+-]+/)*[^/]+\.whl(?:\.metadata)?', parsed.path) is not None)


class Store:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / 'blobs').mkdir(parents=True, exist_ok=True)
        (self.root / 'partial').mkdir(exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, metadata TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS requests (key TEXT PRIMARY KEY, request TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS events (kind TEXT, host TEXT, key TEXT, at REAL DEFAULT (unixepoch()))')

    def record_request(self, key, method, url, headers, body=b''):
        # Private journal retains exact variants so prewarming helps unchanged clients.
        values = {k.lower(): v for k, v in headers.items()
                  if k.lower() not in HOP_HEADERS | {'host', 'content-length'}}
        item = dict(method=method, url=url, headers=values,
                    body=base64.b64encode(body).decode())
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO requests VALUES(?,?)', (key, json.dumps(item)))

    def missing_requests(self):
        with self.connect() as db:
            rows = db.execute('SELECT r.key,r.request FROM requests r LEFT JOIN entries e ON r.key=e.key '
                              'WHERE e.key IS NULL ORDER BY r.key').fetchall()
        return [(key, json.loads(raw)) for key, raw in rows]

    def normalize_requests(self):
        """Add canonical aliases for recorded HF telemetry without replacing bytes."""
        with self.connect() as db:
            rows = db.execute('SELECT r.key,r.request,e.metadata FROM requests r '
                              'LEFT JOIN entries e ON r.key=e.key').fetchall()
            aliases = 0
            for old, raw, metadata in rows:
                item = json.loads(raw)
                key = request_key(item['method'], item['url'], item['headers'],
                                  base64.b64decode(item['body']))
                if key == old:
                    continue
                db.execute('INSERT OR IGNORE INTO requests VALUES(?,?)', (key, raw))
                if metadata:
                    db.execute('INSERT OR IGNORE INTO entries VALUES(?,?)', (key, metadata))
                db.execute('DELETE FROM requests WHERE key=?', (old,))
                aliases += 1
        return aliases

    def import_missing_requests(self, source):
        """Copy replay misses into a mutable journal; sealed blobs remain unchanged."""
        if (self.root / 'seal.json').exists():
            raise ValueError('Cannot import requests into a sealed cache')
        path = Path(source).resolve() / 'index.sqlite3'
        with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=30) as db:
            rows = db.execute('SELECT r.request FROM requests r LEFT JOIN entries e ON r.key=e.key '
                              "WHERE e.key IS NULL AND EXISTS (SELECT 1 FROM events v "
                              "WHERE v.key=r.key AND v.kind='denied')").fetchall()
        imported = 0
        with self.connect() as db:
            for (raw,) in rows:
                item = json.loads(raw)
                key = request_key(item['method'], item['url'], item['headers'],
                                  base64.b64decode(item['body']))
                imported += db.execute('INSERT OR IGNORE INTO requests VALUES(?,?)', (key, raw)).rowcount
        return imported

    def connect(self):
        return sqlite3.connect(self.root / 'index.sqlite3', timeout=30)

    def event(self, kind, host, key):
        with self.connect() as db:
            db.execute('INSERT INTO events(kind,host,key) VALUES(?,?,?)', (kind, host, key))

    def get(self, key):
        with self.connect() as db:
            row = db.execute('SELECT metadata FROM entries WHERE key=?', (key,)).fetchone()
        if row:
            value = json.loads(row[0])
            path = self.root / 'blobs' / value['sha256']
            if path.is_file() and path.stat().st_size == value['size']:
                return value
        return None

    def replay_entry(self, method, url, headers, body=b''):
        """Resolve exact responses, or HEAD/single Range from the same GET variant."""
        key = request_key(method, url, headers, body)
        exact = self.get(key)
        if exact:
            return key, exact, None
        requested_range = headers.get('range', '')
        if method != 'HEAD' and not (method == 'GET' and re.fullmatch(r'bytes=\d*-\d*', requested_range)):
            return key, None, None
        # All representation, authentication and cookie fields remain in the key.
        full_headers = {k.lower(): v for k, v in headers.items()}
        full_headers.pop('range', None)
        full_headers.pop('if-range', None)
        full_key = request_key('GET', url, full_headers, body)
        full = self.get(full_key)
        if not full or full['status'] != 200:
            return key, None, None
        response = {k.lower(): v for k, v in full['headers']}
        if method == 'HEAD':
            return full_key, full, None
        if response.get('content-encoding', 'identity') != 'identity':
            return key, None, None
        validator = headers.get('if-range')
        matches = not validator or (not validator.startswith('W/') and validator in
                                    (response.get('etag'), response.get('last-modified')))
        return full_key, full, requested_range if matches else None

    def writer(self):
        return tempfile.NamedTemporaryFile(dir=self.root / 'partial', delete=False)

    def publish(self, key, file, digest, size, status, headers, *, refresh=False):
        file.close()
        target = self.root / 'blobs' / digest
        os.replace(file.name, target)
        metadata = dict(sha256=digest, size=size, status=status, headers=headers)
        with self.connect() as db:
            # Only mutable repository indexes may replace a completed entry in record mode.
            verb = 'REPLACE' if refresh else 'IGNORE'
            db.execute(f'INSERT OR {verb} INTO entries VALUES(?,?)', (key, json.dumps(metadata)))

    def inventory(self):
        with self.connect() as db:
            entries = dict(db.execute('SELECT key,metadata FROM entries ORDER BY key'))
            counts = dict(db.execute('SELECT kind,count(*) FROM events GROUP BY kind'))
            misses = list(db.execute("SELECT host,count(*) FROM events WHERE kind IN ('miss','error','denied') GROUP BY host ORDER BY count(*) DESC LIMIT 15"))
        return entries, counts, misses

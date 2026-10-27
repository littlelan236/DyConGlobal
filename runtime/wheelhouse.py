"""Immutable, content-checked shared wheelhouse publication for task containers."""
from __future__ import annotations

import json
import os
import shutil
import urllib.request
import zipfile
from email.parser import BytesParser
from pathlib import Path

from packaging.utils import canonicalize_name, parse_wheel_filename

from core.io import hash_json, identifier, now, read_json, sha256, write_json
from core.locking import file_lock

# Keep multiple Python-compatible versions; pip applies Requires-Python per image.
# Python 3.6 is still used by pinned benchmark images.
BUILD_TOOLS = (
    ('setuptools', '50.3.2'), ('wheel', '0.34.2'),
    ('packaging', '20.9'), ('pyparsing', '2.4.7'),
    ('setuptools', '59.6.0'), ('setuptools', '75.1.0'),
    ('wheel', '0.37.1'), ('wheel', '0.45.1'),
    ('packaging', '21.3'), ('packaging', '24.2'),
    ('pyparsing', '3.0.7'), ('tomli', '1.2.3'), ('tomli', '2.2.1'),
    # Sphinx 6.x editable installs declare flit_core>=3.7 in build-system.
    # 3.9.0 remains compatible with the Python 3.6/3.9 task images.
    ('flit_core', '3.9.0'),
)


def discover_sources(cfg):
    from runtime.containers.repairs import environment_overrides

    root = Path(cfg['project_root'])
    candidates = set()
    if cfg['environment'].get('wheelhouse'):
        candidates.add(Path(cfg['environment']['wheelhouse']))
    for path in (root / 'runtime/cache').glob('*'):
        if path.is_dir() and (path.name.endswith('-wheels') or any(path.glob('*.whl'))):
            candidates.add(path)
    tasks = read_json(root / 'catalog/manifest.json')['tasks']
    for task in tasks.values():
        wheelhouse = environment_overrides(cfg, task['instance']).get('wheelhouse')
        if wheelhouse:
            path = Path(wheelhouse)
            candidates.add(path if path.is_absolute() else root / path)
    return sorted(candidates)


def wheel_metadata(path):
    name, version, _, tags = parse_wheel_filename(path.name)
    with zipfile.ZipFile(path) as wheel:
        members = [n for n in wheel.namelist()
                   if n.endswith('.dist-info/METADATA') and n.count('/') == 1]
        if len(members) != 1:
            raise ValueError(f'Expected one wheel METADATA: {path}')
        metadata = BytesParser().parsebytes(wheel.read(members[0]))
    if canonicalize_name(metadata['Name']) != name or metadata['Version'] != str(version):
        raise ValueError(f'Wheel filename/metadata mismatch: {path}')
    return {'name': name, 'version': str(version), 'requires_python': metadata.get('Requires-Python'),
            'requires_dist': metadata.get_all('Requires-Dist', []),
            'tags': sorted(map(str, tags)), 'size': path.stat().st_size, 'sha256': sha256(path)}


def download_build_tools(destination):
    downloaded = {}
    for package, version in BUILD_TOOLS:
        with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json', timeout=45) as response:
            release = json.load(response)
        files = [f for f in release['urls'] if f['packagetype'] == 'bdist_wheel']
        if not files:
            raise ValueError(f'No published wheel for {package}=={version}')
        for info in files:
            path = destination / info['filename']
            with urllib.request.urlopen(info['url'], timeout=90) as response, path.open('wb') as output:
                shutil.copyfileobj(response, output)
            if sha256(path) != info['digests']['sha256']:
                raise ValueError(f'PyPI SHA-256 mismatch: {path.name}')
            downloaded[path.name] = {'release': f'{package}=={version}', 'url': info['url'],
                                     'expected_sha256': info['digests']['sha256']}
    return downloaded


def build(cfg, publication_id, extra_sources=(), download=False):
    root = Path(cfg['project_root'])
    publication_id = identifier(publication_id)
    destination = root / 'runtime/cache' / publication_id
    staging = root / 'temp/wheelhouse' / publication_id
    with file_lock(root / 'state/wheelhouse' / (publication_id + '.lock'), nonblocking=True):
        if destination.exists() or staging.exists():
            raise FileExistsError(f'Use a new immutable wheelhouse ID: {publication_id}')
        staging.mkdir(parents=True)
        entries = {}
        origins = discover_sources(cfg) + [Path(p).resolve() for p in extra_sources]
        missing = []
        for source in origins:
            if not source.is_dir():
                missing.append(str(source))
                continue
            for path in sorted(source.glob('*.whl')):
                metadata = wheel_metadata(path)
                if path.name in entries:
                    if entries[path.name]['sha256'] != metadata['sha256']:
                        raise ValueError(f'Same wheel filename has conflicting content: {path.name}')
                    entries[path.name]['sources'].append(str(path))
                else:
                    shutil.copy2(path, staging / path.name)
                    entries[path.name] = {**metadata, 'sources': [str(path)]}
        downloaded = {}
        if download:
            download_dir = staging / 'downloads'
            download_dir.mkdir()
            downloaded = download_build_tools(download_dir)
            for path in download_dir.glob('*.whl'):
                metadata = wheel_metadata(path)
                if path.name in entries and entries[path.name]['sha256'] != metadata['sha256']:
                    raise ValueError(f'Published/local wheel content conflict: {path.name}')
                if path.name not in entries:
                    shutil.copy2(path, staging / path.name)
                    entries[path.name] = {**metadata, 'sources': []}
                entries[path.name]['pypi'] = downloaded[path.name]
            shutil.rmtree(download_dir)
        if not entries:
            raise ValueError('Cannot publish an empty wheelhouse')
        manifest = {'schema_version': 1, 'publication_id': publication_id, 'created_at': now(),
                    'sources': list(map(str, origins)), 'missing_sources': missing,
                    'wheel_count': len(entries), 'bytes': sum(e['size'] for e in entries.values()),
                    'downloaded_build_tools': downloaded, 'wheels': dict(sorted(entries.items()))}
        manifest['manifest_sha256'] = hash_json(manifest)
        write_json(staging / 'manifest.json', manifest)
        for path in staging.iterdir():
            path.chmod(0o444)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, destination)
        receipt = {'publication_id': publication_id, 'path': str(destination),
                   'manifest_sha256': sha256(destination / 'manifest.json'),
                   'wheel_count': len(entries), 'bytes': manifest['bytes'], 'created_at': now()}
        write_json(root / 'state/wheelhouse' / (publication_id + '.json'), receipt)
        return receipt


def verify(path):
    path = Path(path)
    manifest = read_json(path / 'manifest.json')
    expected = manifest.pop('manifest_sha256')
    if hash_json(manifest) != expected:
        raise ValueError('Wheelhouse manifest changed')
    names = {p.name for p in path.glob('*.whl')}
    if names != set(manifest['wheels']):
        raise ValueError('Wheelhouse file selection changed')
    for name, record in manifest['wheels'].items():
        if sha256(path / name) != record['sha256']:
            raise ValueError(f'Wheelhouse content changed: {name}')
    return {'status': 'verified', 'path': str(path), 'wheel_count': len(names),
            'manifest_sha256': sha256(path / 'manifest.json')}

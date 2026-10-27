"""Prepare pinned SkillsBench task images; publish verified graphs to the NAS.

This is image preparation only: no agents, oracle, verifier or catalog migration.
The existing OCI validator, NAS loader and Docker archive importer own their
respective formats. Native daemon BuildKit is used after source images are cached through the NAS;
this never changes daemon proxy settings or restarts the active Docker daemon.
"""
from __future__ import annotations

import concurrent.futures
import copy
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from urllib.parse import urlparse

import docker
import yaml

from core.io import hash_json, identifier, now, read_json, sha256, write_json
from core.locking import file_lock
from runtime.docker_storage import assert_data_root, storage_lease
from runtime.images.nas import ImageLeaseManager, NasImageStore
from runtime.images.oci import blob_path, descriptor_graph, platform_manifest
from runtime.images.registry import image_metadata, transfer_blob
from runtime.scripts.export_task_images_to_nas import import_archive


def _command(cfg, args, *, log=None, timeout=None):
    command = [cfg['environment']['docker_executable'], '--host',
               cfg['environment']['docker_host'], *args]
    context = cfg['skillsbench_images'].get('docker_context')
    if args and args[0] == 'buildx' and context:
        command = [cfg['environment']['docker_executable'], '--context', context, *args]
    env = os.environ.copy()
    proxy = cfg['skillsbench_images'].get('host_proxy')
    if proxy:
        for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
            env[name] = proxy
        env.pop('ALL_PROXY', None)
        env.pop('all_proxy', None)
    if log is None:
        return subprocess.check_output(command, text=True, timeout=timeout, env=env).strip()
    with Path(log).open('a', encoding='utf-8') as stream:
        stream.write('\nCOMMAND ' + json.dumps(command) + '\n')
        stream.flush()
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                       timeout=timeout, check=True, env=env)


def _context_hash(path):
    rows = []
    for p in sorted(path.rglob('*')):
        rel = p.relative_to(path).as_posix()
        if p.is_symlink():
            rows.append([rel, 'link', os.readlink(p)])
        elif p.is_file():
            rows.append([rel, p.stat().st_mode & 0o777, sha256(p)])
    return hash_json(rows)


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Build path escapes the pinned task package')
    return path


@contextmanager
def build_proxy(cfg, state):
    """Own a byte-transparent socat forwarder only during image preparation."""
    options = cfg['skillsbench_images']
    if not options.get('manage_build_proxy'):
        yield
        return
    executable = shutil.which('socat')
    if not executable:
        raise RuntimeError('Managed build proxy requires the installed socat utility')
    listen, upstream = urlparse(options['build_proxy']), urlparse(options['host_proxy'])
    if listen.scheme != 'http' or upstream.scheme != 'http' or not listen.port or not upstream.port:
        raise ValueError('Managed proxy requires explicit HTTP host:port endpoints')
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((listen.hostname, listen.port))
        except OSError as exc:
            raise RuntimeError('Build proxy port is already owned; refusing to replace it') from exc
    state.mkdir(parents=True, exist_ok=True)
    args = [executable, f'TCP4-LISTEN:{listen.port},bind={listen.hostname},reuseaddr,backlog=1024,fork',
            f'TCP4:{upstream.hostname}:{upstream.port}']
    with (state / 'proxy.log').open('a') as log:
        process = subprocess.Popen(args, stdout=log, stderr=log, start_new_session=True)
        receipt = {'pid': process.pid, 'owner_pid': os.getpid(), 'args': args,
                   'listen': options['build_proxy'], 'upstream': options['host_proxy'],
                   'started_at': now(), 'active': True}
        try:
            for _ in range(50):
                if process.poll() is not None:
                    raise RuntimeError('Managed build proxy exited before readiness')
                try:
                    with socket.create_connection((listen.hostname, listen.port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                raise TimeoutError('Managed build proxy did not become ready')
            write_json(state / 'proxy.json', receipt)
            yield
        finally:
            # start_new_session creates an owned process group containing only
            # this socat and its connection children. Never touch port 7897.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            write_json(state / 'proxy.json', {**receipt, 'active': False, 'finished_at': now()})


def inventory(cfg):
    options = cfg['skillsbench_images']
    source = (Path(cfg['project_root']) / options['source_root']).resolve()
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != options['revision']:
        raise ValueError('SkillsBench checkout differs from pinned revision')
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--', 'tasks'], text=True)
    if dirty:
        raise ValueError('Pinned tasks contain local modifications: ' + dirty[:1000])
    tasks, units = [], []
    for task in sorted((source / 'tasks').iterdir()):
        if not task.is_dir() or not (task / 'task.md').is_file():
            continue
        name = identifier(task.name)
        environment = task / 'environment'
        if not (environment / 'Dockerfile').is_file():
            raise ValueError('Missing task Dockerfile: ' + name)
        # Read the native frontmatter without reinterpreting task instructions.
        text = (task / 'task.md').read_text()
        metadata = yaml.safe_load(text.split('---', 2)[1])
        services = {'main': {'build': {'context': '.', 'dockerfile': 'Dockerfile'}}}
        compose = environment / 'docker-compose.yaml'
        if compose.exists():
            services = yaml.safe_load(compose.read_text())['services']
        context_hash = _context_hash(environment)
        task_units = []
        for service, spec in services.items():
            service = identifier(service)
            build = spec.get('build')
            if build:
                build = {'context': build} if isinstance(build, str) else build
                relative = build.get('context', '.')
                relative = '.' if relative == '${CONTEXT_DIR}' else relative
                context = _inside(environment, relative)
                dockerfile = _inside(context, build.get('dockerfile', 'Dockerfile'))
                if not dockerfile.is_file():
                    raise FileNotFoundError(dockerfile)
                if any(k in build for k in ('additional_contexts', 'secrets', 'ssh', 'dockerfile_inline')):
                    raise ValueError('Unsupported build contract needs explicit integration: ' + name)
                unit = {'kind': 'build', 'context': str(context.relative_to(source)),
                        'dockerfile': str(dockerfile.relative_to(source)),
                        'build_args': build.get('args', {}), 'target': build.get('target')}
            else:
                if not spec.get('image') or '${' in spec['image']:
                    raise ValueError('Unresolved image for service: ' + name + '/' + service)
                unit = {'kind': 'pull', 'source_image': spec['image']}
            key = identifier(name + '--' + service)
            unit.update(key=key, task_id=name, service=service, context_sha256=context_hash,
                        task_sha256=sha256(task / 'task.md'),
                        tag=f'dycon/skillsbench-{name}-{service}:{revision[:12]}')
            units.append(unit)
            task_units.append(key)
        tasks.append({'task_id': name, 'task_sha256': sha256(task / 'task.md'),
                      'environment_sha256': context_hash, 'units': task_units,
                      'sandbox': metadata.get('sandbox', metadata.get('environment', {}))})
    if not tasks:
        raise ValueError('No official tasks selected')
    return {'schema_version': 1, 'repository': 'https://github.com/benchflow-ai/skillsbench',
            'revision': revision, 'scope': 'tasks/', 'excluded': ['tasks-extra/'],
            'source_root': str(source), 'tasks': tasks, 'units': units}


def _paths(cfg):
    options = cfg['skillsbench_images']
    store_name = identifier(options['store'])
    nas = Path(cfg['paths']['nas_root']).resolve()
    if not any(p != Path('/') and os.path.ismount(p) for p in [nas, *nas.parents]):
        raise ValueError('Configured NAS is not mounted')
    store = nas / store_name
    state = Path(cfg['project_root']) / 'state/skillsbench-images' / store_name
    return store, state


def _record(cfg, store, key, tag, image, provenance):
    descriptor = image.attrs.get('Descriptor')
    if not descriptor:
        path = blob_path(store / 'blobs/sha256', image.id)
        obj = read_json(path)
        descriptor = {'digest': image.id, 'size': path.stat().st_size, 'mediaType': obj['mediaType']}
    blobs = store / 'blobs/sha256'
    graph = descriptor_graph(blobs, descriptor, deep=True)
    runtime, manifest = platform_manifest(blobs, descriptor, cfg['environment']['platform'])
    record = {'schema_version': 2, 'requested': tag, 'manifest_digest': descriptor['digest'],
              'manifest_size': descriptor['size'], 'root_media_type': descriptor['mediaType'],
              'runtime_manifest_digest': runtime['digest'], 'config': manifest['config'],
              'layers': manifest['layers'], 'compressed_bytes': sum(x['size'] for x in manifest['layers']),
              'repo_digests': image.attrs.get('RepoDigests') or [], 'platform': cfg['environment']['platform'],
              'descriptor_count': len(graph), 'deep_verified': True, 'provenance': provenance}
    path = store / 'images' / (key + '.json')
    if path.exists() and read_json(path) != record:
        raise ValueError('An immutable image publication already exists with different content: ' + key)
    write_json(path, record)
    return {'store': store.name, 'manifest': 'images/' + key + '.json',
            'manifest_sha256': sha256(path), 'image_id': descriptor['digest']}


def docker_instructions(text):
    """Yield Docker instruction spans while treating heredoc bodies as data."""
    lines, cursor = text.splitlines(keepends=True), 0
    keywords = {'FROM', 'RUN', 'ARG', 'ENV', 'COPY', 'ADD', 'USER', 'WORKDIR', 'CMD',
                'ENTRYPOINT', 'LABEL', 'EXPOSE', 'SHELL', 'VOLUME', 'HEALTHCHECK', 'STOPSIGNAL', 'ONBUILD'}
    while cursor < len(lines):
        start = cursor
        match = re.match(r'\s*([A-Za-z]+)\s+(.*)', lines[cursor])
        cursor += 1
        if not match or match.group(1).upper() not in keywords:
            continue
        header = lines[start]
        while header.rstrip().endswith('\\') and cursor < len(lines):
            header = header.rstrip()[:-1] + ' ' + lines[cursor]
            cursor += 1
        delimiters = re.findall(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", header)
        for delimiter in delimiters:
            while cursor < len(lines):
                body = lines[cursor]
                cursor += 1
                if body.rstrip('\r\n').lstrip('\t') == delimiter:
                    break
            else:
                raise ValueError('Unterminated Dockerfile heredoc: ' + delimiter)
        yield start, cursor, match.group(1).upper(), header


def source_images(text, build_args=None):
    """Resolve FROM defaults/stage aliases and external Dockerfile frontends."""
    variables, stages, images = dict(build_args or {}), set(), []
    syntax = re.match(r'\s*#\s*syntax\s*=\s*(\S+)', text)
    if syntax:
        images.append(syntax.group(1))
    for _, _, keyword, line in docker_instructions(text):
        argument = re.match(r'\s*ARG\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)', line, re.I)
        if keyword == 'ARG' and argument:
            variables.setdefault(argument.group(1), argument.group(2).strip())
        match = re.match(r'\s*FROM\s+(?:--platform=\S+\s+)?(\S+)(?:\s+AS\s+(\S+))?', line, re.I)
        if keyword == 'FROM' and match:
            image = re.sub(r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)',
                           lambda m: str(variables.get(m.group(1) or m.group(2), m.group(0))), match.group(1))
            if '$' in image:
                raise ValueError('Unresolved FROM argument: ' + image)
            if image.lower() not in stages and image.lower() != 'scratch':
                images.append(image)
            if match.group(2):
                stages.add(match.group(2).lower())
    return list(dict.fromkeys(images))


def effective_dockerfile(text, options):
    """Apply a recorded download-only profile; keep upstream files untouched."""
    compatibility = options.get('compatibility') or {}
    if set(compatibility) - {'replacements', 'python_bootstrap', 'note'}:
        raise ValueError('Unknown task compatibility option')
    for change in compatibility.get('replacements', []):
        before, after = change['before'], change['after']
        if '{build_proxy_jvm_args}' in after:
            proxy = urlparse(options['build_proxy'])
            if not proxy.hostname or not proxy.port:
                raise ValueError('JVM proxy requires a configured host and port')
            flags = [f'-J-D{scheme}.proxy{field}={value}' for scheme in ('http', 'https')
                     for field, value in [('Host', proxy.hostname), ('Port', str(proxy.port))]]
            after = after.replace('{build_proxy_jvm_args}', ' '.join(map(shlex.quote, flags)))
        if not before or text.count(before) != 1:
            raise ValueError('Task compatibility replacement does not match exactly once')
        text = text.replace(before, after)
    ubuntu = options.get('ubuntu_mirror')
    debian = options.get('debian_mirror')
    if not ubuntu and not debian and not options.get('python_index') and not options.get('mount_verified_wheels'):
        return text
    substitutions = []
    for original, replacement in [
        ('http://archive.ubuntu.com/ubuntu', ubuntu),
        ('http://security.ubuntu.com/ubuntu', ubuntu),
        ('https://archive.ubuntu.com/ubuntu', ubuntu),
        ('https://security.ubuntu.com/ubuntu', ubuntu),
        ('http://deb.debian.org/debian', debian),
        ('https://deb.debian.org/debian', debian),
    ]:
        if replacement:
            if not re.fullmatch(r'https?://[A-Za-z0-9./_-]+', replacement):
                raise ValueError('Invalid apt mirror URL')
            substitutions.extend(['-e', f"'s|{original}|{replacement}|g'"])
    setup = ("RUN if [ $(id -u) = 0 ] && [ -d /etc/apt ]; then "
        "find /etc/apt -type f '(' -name '*.list' -o -name '*.sources' ')' "
        "-exec sed -i " + ' '.join(substitutions) + " {} +; "
        "echo 'Acquire::Retries 5;' > /etc/apt/apt.conf.d/80dycon-retries; fi\n")
    if not ubuntu and not debian:
        setup = ''
    lines, python_args_added = [], False
    source_lines = text.splitlines(keepends=True)
    instructions = {start: (end, keyword, header) for start, end, keyword, header in docker_instructions(text)}
    stage_ends = {end - 1 for _, end, keyword, _ in docker_instructions(text) if keyword == 'FROM'}
    for index, line in enumerate(source_lines):
        instruction = instructions.get(index)
        if options.get('python_index') and not python_args_added and instruction and instruction[1] == 'RUN':
            probe = ''.join(source_lines[index:instruction[0]])
            if re.search(r'\b(?:pip3?|uv)\s+(?:install|pip|sync|run)\b|python3?\s+-m\s+pip', probe):
                # Declare only before Python installation, preserving cache
                # keys for already completed operating-system setup layers.
                lines.append('ARG PIP_INDEX_URL\nARG UV_INDEX_URL\nARG PIP_DEFAULT_TIMEOUT\n')
                if options.get('mount_verified_wheels'):
                    lines.append('ARG PIP_CONSTRAINT\n')
                bootstrap = compatibility.get('python_bootstrap', [])
                if bootstrap:
                    from packaging.requirements import Requirement
                    for value in bootstrap:
                        requirement = Requirement(value)
                        if requirement.url or len(list(requirement.specifier)) != 1 or next(iter(requirement.specifier)).operator != '==':
                            raise ValueError('Python bootstrap dependencies must have one exact version pin')
                    mount = ('--mount=type=bind,from=dycon_verified_wheels,target=/opt/dycon-verified-wheels,readonly '
                             if options.get('mount_verified_wheels') else '')
                    lines.append('RUN ' + mount + 'python3 -m pip install --no-cache-dir ' +
                                 ' '.join(shlex.quote(x) for x in bootstrap) + '\n')
                python_args_added = True
        if instruction and instruction[1] == 'RUN' and python_args_added and options.get('mount_verified_wheels'):
            line = re.sub(r'^(\s*RUN\s+)', r'\1--mount=type=bind,from=dycon_verified_wheels,target=/opt/dycon-verified-wheels,readonly ', line, count=1, flags=re.I)
        lines.append(line)
        if index in stage_ends:
            lines.append(setup)
            python_args_added = False
    return ''.join(lines)


def verified_wheel_context(cfg, store, unit):
    """Mount only explicitly configured, official-hash-verified wheel bytes."""
    from packaging.utils import canonicalize_name, parse_wheel_filename
    artifacts = cfg['skillsbench_images'].get('verified_wheels', {}).get(unit['task_id'], [])
    if not artifacts:
        return None, []
    nas = Path(cfg['paths']['nas_root']).resolve()
    context = store / 'wheel-contexts' / unit['key']
    context.mkdir(parents=True, exist_ok=True)
    receipts, constraints = [], []
    for artifact in artifacts:
        source = (nas / artifact['path']).resolve()
        if not source.is_relative_to(nas) or source.name != artifact['filename']:
            raise ValueError('Verified wheel path escapes NAS or changes filename')
        if not re.fullmatch(r'[A-Za-z0-9_.+-]+\.whl', source.name):
            raise ValueError('Invalid wheel filename')
        package, version, _, _ = parse_wheel_filename(source.name)
        if canonicalize_name(artifact['package']) != package or str(version) != artifact['version']:
            raise ValueError('Verified wheel package/version mismatch')
        if sha256(source) != artifact['sha256']:
            raise ValueError('Verified wheel checksum mismatch; refusing to bypass integrity')
        metadata = read_json(source.with_name(source.name + '.source.json'))
        for field in ('package', 'version', 'filename', 'sha256'):
            if metadata[field] != artifact[field]:
                raise ValueError('Wheel disagrees with source metadata')
        target = context / source.name
        if not target.exists():
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        if sha256(target) != artifact['sha256']:
            raise ValueError('Wheel build-context checksum mismatch')
        constraints.append(f'{package} @ file:///opt/dycon-verified-wheels/{source.name}')
        receipts.append({**artifact, 'official_url': metadata['url'], 'size': source.stat().st_size})
    (context / 'constraints.txt').write_text('\n'.join(constraints) + '\n')
    return context, receipts


def _cache_source(cfg, store, state, source):
    """Pull a source image through NAS without changing daemon proxy settings."""
    host = 'docker.io'
    repository = source
    first = source.split('/')[0]
    if '/' in source and ('.' in first or ':' in first or first == 'localhost'):
        host, repository = source.split('/', 1)
    if '/' not in repository:
        repository = 'library/' + repository
    if '@' in repository:
        repository, reference = repository.rsplit('@', 1)
    elif ':' in repository.rsplit('/', 1)[-1]:
        repository, reference = repository.rsplit(':', 1)
    else:
        reference = 'latest'
    key = '_source-' + hash_json(source)[:20]
    path = store / 'images' / (key + '.json')
    blobs = store / 'blobs/sha256'
    with file_lock(state / (key + '.lock')):
        if not path.exists():
            kwargs = {} if host == 'docker.io' else {'registry': host}
            root, descriptors = image_metadata(repository, blobs, reference=reference,
                proxy=cfg['skillsbench_images']['host_proxy'], **kwargs)
            current = root
            while True:
                obj = read_json(blob_path(blobs, current['digest']))
                if 'manifests' not in obj:
                    break
                current = next(x for x in obj['manifests']
                               if x.get('platform', {}).get('os') == 'linux'
                               and x.get('platform', {}).get('architecture') == 'amd64')
            lookup = {x['digest']: x for x in descriptors}
            def pull(item):
                with file_lock(state / 'blob-locks' / (item['digest'][7:] + '.lock')):
                    return transfer_blob(lookup[item['digest']], blobs,
                                         proxy=cfg['skillsbench_images']['host_proxy'])
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(pull, [obj['config'], *obj['layers']]))
            descriptor_graph(blobs, current, deep=True)
            write_json(path, {'schema_version': 2, 'requested': source,
                'manifest_digest': current['digest'], 'manifest_size': current['size'],
                'root_media_type': current['mediaType'], 'runtime_manifest_digest': current['digest'],
                'config': obj['config'], 'layers': obj['layers'],
                'compressed_bytes': sum(x['size'] for x in obj['layers']),
                'repo_digests': [host + '/' + repository + '@' + current['digest']],
                'source_index': root, 'deep_verified': True})
            print('SOURCE_CACHED ' + source, flush=True)
        return {'store': store.name, 'manifest': 'images/' + key + '.json',
                'manifest_sha256': sha256(path), 'image_id': read_json(path)['manifest_digest']}


def prepare(cfg, *, only=()):
    options = cfg['skillsbench_images']
    module_sha = sha256(Path(__file__))
    store, state = _paths(cfg)
    for directory in (store / 'blobs/sha256', store / 'images', state / 'logs'):
        directory.mkdir(parents=True, exist_ok=True)
    with storage_lease(cfg), file_lock(state / 'prepare.lock', nonblocking=True):
        plan = inventory(cfg)
        if (store / 'plan.json').exists():
            if read_json(store / 'plan.json') != plan:
                raise ValueError('Published source selection changed; use a new store')
        else:
            write_json(store / 'plan.json', plan)
            write_json(state / 'resolved-config.json', cfg)
        chosen = set(only)
        if chosen - {x['task_id'] for x in plan['tasks']}:
            raise ValueError('Unknown task IDs')
        client = docker.DockerClient(base_url=cfg['environment']['docker_host'], timeout=120)
        # Native daemon BuildKit handles this host's rootless cgroups correctly.
        # Source images are loaded via the existing NAS boundary first, so its
        # registry resolver never needs a daemon-wide proxy change.
        base_cfg = copy.deepcopy(cfg)
        base_cfg['images']['cleanup'] = False
        base_manager = ImageLeaseManager(base_cfg, client)
        lock = threading.Lock()
        resources = ExitStack()
        results = read_json(state / 'progress.json') if (state / 'progress.json').exists() else {}
        if results:
            history = state / 'history'
            history.mkdir(exist_ok=True)
            write_json(history / (str(results.get('pid', 'unknown')) + '.json'), results)
        for field in ('error', 'finished_at', 'complete_images', 'complete_tasks'):
            results.pop(field, None)
        for entry in results.get('images', {}).values():
            if entry.get('status') in ('building', 'exporting'):
                entry['status'] = 'pending'
        write_json(state / 'attempt-configs' / (str(os.getpid()) + '.json'), cfg)
        results.update(schema_version=1, revision=plan['revision'], total_tasks=len(plan['tasks']),
                       total_images=len(plan['units']), status='running', pid=os.getpid(), started_at=now())
        results.setdefault('images', {})
        for unit in plan['units']:
            if (not chosen or unit['task_id'] in chosen) and results['images'].get(unit['key'], {}).get('status') != 'complete':
                results['images'][unit['key']] = {'status': 'pending'}
        def save():
            complete = {k for k, v in results['images'].items() if v['status'] == 'complete'}
            results['complete_images'] = len(complete)
            results['complete_tasks'] = sum(set(t['units']) <= complete for t in plan['tasks'])
            results['updated_at'] = now()
            write_json(state / 'progress.json', results)
        def worker(unit):
            key, log = unit['key'], state / 'logs' / (unit['key'] + '.log')
            try:
                path = store / 'images' / (key + '.json')
                if path.exists():
                    record = read_json(path)
                    if record['provenance']['unit'] != unit:
                        raise ValueError('Image belongs to different source context')
                    descriptor_graph(store / 'blobs/sha256', {'digest': record['manifest_digest'],
                        'size': record['manifest_size'], 'mediaType': record['root_media_type']}, deep=True)
                    reference = {'store': store.name, 'manifest': 'images/' + key + '.json',
                                 'manifest_sha256': sha256(path), 'image_id': record['manifest_digest']}
                else:
                    if shutil.disk_usage(client.info()['DockerRootDir']).free < options['reserve_bytes']:
                        raise RuntimeError('Insufficient /data1 free space; no global prune is permitted')
                    with lock:
                        results['images'][key] = {'status': 'building', 'started_at': now(), 'log': str(log)}
                        save()
                    if unit['kind'] != 'build':
                        raise ValueError('External service image needs registry integration: ' + unit['source_image'])
                    source = Path(plan['source_root'])
                    upstream_text = (source / unit['dockerfile']).read_text()
                    wheel_context, wheel_receipts = verified_wheel_context(cfg, store, unit)
                    compatibility = options.get('task_compatibility', {}).get(unit['task_id'], {})
                    build_text = effective_dockerfile(upstream_text, {**options,
                        'mount_verified_wheels': bool(wheel_context), 'compatibility': compatibility})
                    build_file = state / 'build-inputs' / key / 'Dockerfile'
                    build_file.parent.mkdir(parents=True, exist_ok=True)
                    build_file.write_text(build_text)
                    command = ['buildx', 'build', '--builder', options['builder_name'], '--load',
                               '--provenance=false', '--platform', cfg['environment']['platform'],
                               '--tag', unit['tag'], '--file', str(build_file),
                               '--label', 'org.dycon.skillsbench.revision=' + plan['revision'],
                               '--label', 'org.dycon.skillsbench.task=' + unit['task_id']]
                    if wheel_context:
                        command.extend(['--build-context', 'dycon_verified_wheels=' + str(wheel_context),
                            '--build-arg', 'PIP_CONSTRAINT=/opt/dycon-verified-wheels/constraints.txt'])
                    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
                        command.extend(['--build-arg', name + '=' + options['build_proxy']])
                    if options.get('mirror_no_proxy'):
                        for name in ('NO_PROXY', 'no_proxy'):
                            command.extend(['--build-arg', name + '=' + options['mirror_no_proxy']])
                    if options.get('python_index'):
                        for name in ('PIP_INDEX_URL', 'UV_INDEX_URL'):
                            command.extend(['--build-arg', name + '=' + options['python_index']])
                        command.extend(['--build-arg', 'PIP_DEFAULT_TIMEOUT=' + str(options.get('python_read_seconds', 60))])
                    args = unit.get('build_args') or {}
                    entries = args.items() if isinstance(args, dict) else (x.split('=', 1) for x in args)
                    for name, value in entries:
                        if value is None or '${' in str(value):
                            raise ValueError('Unresolved build argument')
                        command.extend(['--build-arg', name + '=' + str(value)])
                    if unit.get('target'):
                        command.extend(['--target', unit['target']])
                    command.append(str(source / unit['context']))
                    source_receipts = []
                    with ExitStack() as source_leases:
                        for source_image in source_images((source / unit['dockerfile']).read_text(),
                                                         unit.get('build_args')):
                            ref = _cache_source(cfg, store, state, source_image)
                            original_ref = ref
                            override = options.get('source_overrides', {}).get(source_image)
                            if override:
                                from runtime.images.ownership import validate_repair
                                validate_repair(cfg, original_ref, override)
                                ref = override
                            cached = NasImageStore(cfg, ref)
                            if override:
                                cached.verify(deep=True)
                            source_receipts.append({'requested': source_image, 'original': original_ref,
                                'effective': ref, 'ownership_adapted': bool(override),
                                'oracle_validation': 'not_run'})
                            source_leases.enter_context(base_manager.lease(cached))
                            native_image = client.images.get(cached.image_id)
                            # Docker SDK accepts the complete source reference
                            # as repository, including its tag.
                            if '@' not in source_image:
                                if ':' in source_image.rsplit('/', 1)[-1]:
                                    tag_repo, tag_version = source_image.rsplit(':', 1)
                                else:
                                    tag_repo, tag_version = source_image, 'latest'
                                native_image.tag(tag_repo, tag_version)
                        _command(cfg, command, log=log, timeout=options['build_seconds'])
                    image = client.images.get(unit['tag'])
                    with lock:
                        results['images'][key]['status'] = 'exporting'
                        save()
                    import_archive(cfg, [unit['tag']], store / 'blobs/sha256')
                    reference = _record(cfg, store, key, unit['tag'], image,
                                        {'revision': plan['revision'], 'unit': unit,
                                         'config_hash': cfg['config_hash'],
                                         'module_sha256': module_sha,
                                         'source_images': source_receipts,
                                         'verified_wheels': wheel_receipts,
                                         'compatibility_profile': compatibility,
                                         'effective_dockerfile_sha256': sha256(build_file),
                                         'download_profile': {k: options.get(k) for k in (
                                             'ubuntu_mirror', 'debian_mirror', 'mirror_no_proxy',
                                             'python_index', 'python_read_seconds')},
                                         'build_log_sha256': sha256(log)})
                outcome = {'status': 'complete', 'nas_image': reference, 'finished_at': now(), 'log': str(log)}
                print('COMPLETE ' + key, flush=True)
            except Exception as exc:
                outcome = {'status': 'failed', 'error': repr(exc), 'finished_at': now(), 'log': str(log)}
                print('FAILED ' + key + ': ' + repr(exc), flush=True)
            with lock:
                results['images'][key] = outcome
                save()
        try:
            assert_data_root(cfg, client.info())
            context = options.get('docker_context')
            if context:
                endpoint = subprocess.check_output([cfg['environment']['docker_executable'],
                    'context', 'inspect', context, '--format', '{{.Endpoints.docker.Host}}'], text=True).strip()
                if endpoint != cfg['environment']['docker_host']:
                    raise ValueError('Build context points to another Docker daemon')
            save()
            resources.enter_context(build_proxy(cfg, state))
            _command(cfg, ['buildx', 'inspect', options['builder_name']],
                     log=state / 'builder.log', timeout=options['build_seconds'])
            units = [u for u in plan['units'] if not chosen or u['task_id'] in chosen]
            # Start with a small real artifact task so the first publication is
            # observable before the long-running toolchain builds finish.
            units.sort(key=lambda u: (u['task_id'] != 'offer-letter-generator', u['key']))
            with concurrent.futures.ThreadPoolExecutor(max_workers=options['workers']) as pool:
                list(pool.map(worker, units))
            complete = {k for k, v in results['images'].items() if v['status'] == 'complete'}
            results['complete_images'] = len(complete)
            results['complete_tasks'] = sum(set(t['units']) <= complete for t in plan['tasks'])
            results['status'] = 'complete' if len(complete) == len(plan['units']) else 'incomplete'
            results['finished_at'] = now()
            save()
            write_json(store / 'receipt.json', results)
            return results
        except BaseException as exc:
            results.update(status='failed', error=repr(exc), finished_at=now())
            save()
            raise
        finally:
            resources.close()
            client.close()


def status(cfg):
    store, state = _paths(cfg)
    result = read_json(state / 'progress.json') if (state / 'progress.json').exists() else {'status': 'not_started'}
    return {**result, 'store': str(store), 'state': str(state)}


def verify(cfg):
    store, state = _paths(cfg)
    plan = read_json(store / 'plan.json')
    inventory_now = inventory(cfg)
    if plan != inventory_now:
        raise ValueError('Current source does not match frozen image plan')
    images, missing = {}, []
    for unit in plan['units']:
        path = store / 'images' / (unit['key'] + '.json')
        if not path.exists():
            missing.append(unit['key'])
            continue
        row = read_json(path)
        if row['provenance']['unit'] != unit:
            raise ValueError('Image source binding changed')
        descriptor_graph(store / 'blobs/sha256', {'digest': row['manifest_digest'],
            'size': row['manifest_size'], 'mediaType': row['root_media_type']}, deep=True)
        images[unit['key']] = {'store': store.name, 'manifest': 'images/' + unit['key'] + '.json',
                              'manifest_sha256': sha256(path), 'image_id': row['manifest_digest']}
    result = {'schema_version': 1, 'revision': plan['revision'], 'verified_at': now(),
              'total_tasks': len(plan['tasks']), 'total_images': len(plan['units']),
              'verified_images': len(images), 'complete_tasks': sum(all(k in images for k in t['units'])
                for t in plan['tasks']), 'missing': missing, 'deep_verified': not missing, 'images': images}
    write_json(state / 'verification.json', result)
    if not missing:
        write_json(store / 'images-manifest.json', result)
    return result

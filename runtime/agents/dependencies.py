"""Immutable, platform-specific offline OpenHands runtime publications."""
from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

from core.io import hash_json, identifier, now, read_json, sha256, write_json
from runtime.agents.openhands import CLI_REVISION


def verify_snapshot(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / 'manifest.json')
    if manifest.get('schema_version') != 1 or manifest.get('cli_revision') != CLI_REVISION:
        raise ValueError('Unsupported OpenHands dependency snapshot')
    # New publications declare their interpreter ABI and must carry the
    # platform-specific blake3 wheel. Legacy minimal receipts remain readable
    # so they can be diagnosed and migrated rather than rejected by schema tightening.
    if 'python' in manifest:
        if manifest.get('python') != '3.12':
            raise ValueError('OpenHands dependency snapshot requires Python 3.12')
        wheel_paths = {row.get('path', '') for row in manifest.get('files', [])}
        if not any(path.startswith('wheels/blake3-1.0.11-cp312-cp312-') for path in wheel_paths):
            raise ValueError('OpenHands dependency snapshot lacks blake3==1.0.11 cp312 wheel')
    body = {k: v for k, v in manifest.items() if k != 'receipt_sha256'}
    if hash_json(body) != manifest.get('receipt_sha256'):
        raise ValueError('Dependency manifest receipt changed')
    for row in manifest['files']:
        path = directory / row['path']
        if (path.is_symlink() or not path.resolve().is_relative_to(directory)
                or not path.is_file() or path.stat().st_size != row['size']
                or sha256(path) != row['sha256']):
            raise ValueError('Changed dependency file: ' + row['path'])
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    expected = {r['path'] for r in manifest['files']} | {'manifest.json'}
    if actual != expected:
        raise ValueError('Dependency snapshot contains unregistered files')
    return manifest



def adapt_sdk_transport(wheels):
    import base64, csv, hashlib, io
    wheel = next(Path(wheels).glob('openhands_sdk-1.22.1-*.whl'))
    before_hash = sha256(wheel)
    with zipfile.ZipFile(wheel) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    member = 'openhands/sdk/llm/llm.py'
    text = contents[member].decode()
    seam = '                ret = litellm_completion('
    if text.count(seam) != 1: raise ValueError('Pinned SDK completion transport seam changed')
    contents[member] = text.replace(seam,
        '                from openhands.sdk.llm.utils.dycon_transport import completion\n'
        '                ret = completion(litellm_completion, self,').encode()
    contents['openhands/sdk/llm/utils/dycon_transport.py'] = Path(__file__).with_name('openhands_transport.py').read_bytes()
    record_name = next(n for n in contents if n.endswith('.dist-info/RECORD'))
    record = io.StringIO(); writer = csv.writer(record, lineterminator='\n')
    for name, data in sorted(contents.items()):
        if name == record_name: writer.writerow([name, '', ''])
        else:
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
            writer.writerow([name, 'sha256=' + digest, str(len(data))])
    contents[record_name] = record.getvalue().encode()
    with zipfile.ZipFile(wheel, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items(): archive.writestr(name, data)
    return {'path': 'wheels/' + wheel.name + ':' + member,
        'before_sha256': before_hash, 'after_sha256': sha256(wheel),
        'policy': 'dycon_sdk_transport_and_native_retry_zero_v1',
        'reason': 'Propagate explicit zero retries into OpenAI; observe actual HTTP timeout and phase without changing the model budget'}


def build_snapshot(cfg, snapshot_id, wheelhouse, cli_archive, uv, python_runtime=None):
    """Build pinned CLI source, resolve wheels, and atomically publish a closure."""
    root = Path(cfg['project_root'])
    target = root / 'runtime/dependency-snapshots' / identifier(snapshot_id)
    if target.exists():
        manifest = verify_snapshot(target)
        if manifest['source_archive_sha256'] != sha256(cli_archive):
            raise ValueError('Dependency snapshot ID already binds another source')
        if python_runtime is not None and manifest.get('agent_runtime', {}).get('source') != str(Path(python_runtime).resolve()):
            raise ValueError('Dependency snapshot ID already binds another interpreter')
        return {'path': str(target), 'manifest': manifest}
    work = root / 'temp/skillsbench-dependencies' / identifier(snapshot_id)
    work.mkdir(parents=True, exist_ok=True)
    staging = work / 'publication'
    if staging.exists():
        raise FileExistsError('Interrupted dependency build: use a new snapshot ID')
    wheels = staging / 'wheels'
    wheels.mkdir(parents=True)
    for file in sorted(Path(wheelhouse).glob('*.whl')):
        if file.name.startswith('openhands-'):
            continue  # PyPI wheel differs from the required Git revision.
        shutil.copy2(file, wheels / file.name)
    shutil.copy2(uv, staging / 'uv')
    (staging / 'uv').chmod(0o755)
    source = work / 'source'
    source.mkdir()
    with tarfile.open(cli_archive) as archive:
        names = archive.getnames()
        expected = 'OpenHands-CLI-' + CLI_REVISION
        if any(name.split('/')[0] != expected for name in names):
            raise ValueError('CLI archive does not name the pinned revision')
        archive.extractall(source, filter='data')
    project = source / expected
    # The stock CLI implicitly imports mutable personal and public skills.
    # BenchFlow is the sole skill authority for this benchmark. Record this
    # small source adaptation, leaving the ACP/agent/tool loop unchanged.
    context_file = project / 'openhands_cli/stores/agent_store.py'
    original = context_file.read_text()
    old = 'load_user_skills=True,\n            load_public_skills=True,'
    if original.count(old) != 1:
        raise ValueError('Pinned CLI skill-discovery seam changed')
    before_hash = sha256(context_file)
    context_file.write_text(original.replace(old,
        'load_user_skills=False,\n            load_public_skills=False,'))
    adaptations = [{'path': 'openhands_cli/stores/agent_store.py',
        'before_sha256': before_hash, 'after_sha256': sha256(context_file),
        'policy': 'benchflow_only_skills_v1',
        'reason': 'Disable implicit personal/public skills; retain BenchFlow project skills'}]
    before_context_hash=sha256(context_file)
    old_context='''        if not (
            agent.condenser and isinstance(agent.condenser, LLMSummarizingCondenser)
        ):
            return None

        condenser_llm = self._with_llm_metadata(
            agent.condenser.llm, session_id=session_id, llm_type="condenser"
        )

        return agent.condenser.model_copy(update={"llm": condenser_llm})'''
    new_context='''        import os
        condenser = agent.condenser
        if condenser is None:
            condenser = LLMSummarizingCondenser(
                llm=agent.llm.model_copy(update={"usage_id": "condenser"}),
                max_size=80, keep_first=4)
        if not isinstance(condenser, LLMSummarizingCondenser):
            return None
        threshold = os.environ.get("OPENHANDS_CONDENSER_MAX_TOKENS")
        if threshold:
            value = int(threshold)
            if value <= 0:
                raise ValueError("OPENHANDS_CONDENSER_MAX_TOKENS must be positive")
            condenser = condenser.model_copy(update={"max_tokens": value})
        condenser_llm = self._with_llm_metadata(
            condenser.llm, session_id=session_id, llm_type="condenser")
        return condenser.model_copy(update={"llm": condenser_llm})'''
    text=context_file.read_text()
    if text.count(old_context)!=1:raise ValueError('Pinned CLI condenser seam changed')
    context_file.write_text(text.replace(old_context,new_context))
    adaptations.append({'path':'openhands_cli/stores/agent_store.py',
        'before_sha256':before_context_hash,'after_sha256':sha256(context_file),
        'policy':'stock_condenser_for_minimal_BenchFlow_settings_v1',
        'reason':'Restore native default condenser for minimal BenchFlow settings and allow an explicit token threshold'})
    before_runtime_hash = sha256(context_file)
    text = context_file.read_text()
    seam = '        updated_tools = self._resolve_tools(session_id)'
    adapted = '''        updated_tools = self._resolve_tools(session_id)
        if os.environ.get("DYCON_OPENHANDS_TERMINAL") == "subprocess":
            updated_tools = [tool.model_copy(update={"params": {
                **tool.params, "terminal_type": "subprocess"}})
                if tool.name == "terminal" else tool for tool in updated_tools]'''
    if text.count(seam) != 1:
        raise ValueError('Pinned CLI terminal selection seam changed')
    text = text.replace(seam, adapted)
    seam = '        if not should_set_litellm_extra_body(llm.model, llm.base_url):'
    adapted = '''        seconds = os.environ.get("DYCON_OPENHANDS_MODEL_SECONDS")
        if seconds is not None:
            value = int(seconds)
            if value <= 0:
                raise ValueError("DYCON_OPENHANDS_MODEL_SECONDS must be positive")
            llm = llm.model_copy(update={"timeout": value, "num_retries": 0})
        if not should_set_litellm_extra_body(llm.model, llm.base_url):'''
    if text.count(seam) != 1:
        raise ValueError('Pinned CLI LLM timeout seam changed')
    context_file.write_text(text.replace(seam, adapted))
    adaptations.append({'path': 'openhands_cli/stores/agent_store.py',
        'before_sha256': before_runtime_hash, 'after_sha256': sha256(context_file),
        'policy': 'dycon_bounded_request_subprocess_terminal_v1',
        'reason': 'Select native subprocess terminal and bounded SDK timeout; disable opaque SDK retries'})
    # The native reset path previously discarded an explicitly selected backend
    # and re-ran tmux auto-detection. Adapt the published wheel, never site-packages.
    import base64
    import csv
    import hashlib
    import io
    tool_wheel = next(wheels.glob('openhands_tools-1.22.1-*.whl'))
    before_tools_hash = sha256(tool_wheel)
    member = 'openhands/tools/terminal/impl.py'
    with zipfile.ZipFile(tool_wheel) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    text = contents[member].decode()
    if text.count('terminal_type=None,') != 1:
        raise ValueError('Pinned tools terminal reset seam changed')
    contents[member] = text.replace('terminal_type=None,', 'terminal_type=self._terminal_type,').encode()
    record_path = next(name for name in contents if name.endswith('.dist-info/RECORD'))
    record = io.StringIO()
    writer = csv.writer(record, lineterminator='\n')
    for name, data in sorted(contents.items()):
        if name == record_path:
            writer.writerow([name, '', ''])
        else:
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
            writer.writerow([name, 'sha256=' + digest, str(len(data))])
    contents[record_path] = record.getvalue().encode()
    with zipfile.ZipFile(tool_wheel, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    adaptations.append({'path': 'wheels/' + tool_wheel.name + ':' + member,
        'before_sha256': before_tools_hash, 'after_sha256': sha256(tool_wheel),
        'policy': 'preserve_explicit_terminal_backend_on_reset_v1',
        'reason': 'Keep the native subprocess backend on reset; wheel RECORD rebuilt and hash-bound'})
    adaptations.append(adapt_sdk_transport(wheels))
    env = dict(os.environ, UV_FIND_LINKS=str(wheels), UV_NO_INDEX='1', UV_PYTHON_DOWNLOADS='never',
               UV_PYTHON=str(root / '.venvs/skillsbench/bin/python'))

    def run(args):
        result = subprocess.run([str(staging / 'uv'), *args], env=env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        with (work / 'build.log').open('a') as log:
            log.write(result.stdout)
        if result.returncode:
            raise RuntimeError('Dependency closure build failed: ' + result.stdout[-4000:])

    run(['build', '--wheel', '--no-sources', '--out-dir', str(wheels), str(project)])
    requirements = staging / 'requirements.in'
    requirements.write_text('openhands==1.16.0\nopenhands-sdk==1.22.1\n'
        'openhands-tools==1.22.1\nhttpx==0.28.1\nhttpcore==1.0.9\n', encoding='utf-8')
    overrides = staging / 'overrides.txt'
    overrides.write_text('openhands-sdk==1.22.1\nopenhands-tools==1.22.1\n', encoding='utf-8')
    run(['pip', 'compile', '--no-index', '--find-links', str(wheels),
         '--python-version', '3.12', '--python-platform', 'x86_64-manylinux_2_28',
         '--override', str(overrides), '--generate-hashes', '--no-header', '--no-annotate',
         '--output-file', str(staging / 'requirements.txt'), str(requirements)])
    check = work / 'install-check'
    run(['venv', '--python', str(root / '.venvs/skillsbench/bin/python'), str(check)])
    run(['pip', 'sync', '--python', str(check / 'bin/python'), '--no-index',
         '--find-links', str(wheels), '--require-hashes', str(staging / 'requirements.txt')])
    cli = next(wheels.glob('openhands-*.whl'))
    checked = []
    with zipfile.ZipFile(cli) as wheel:
        for name in wheel.namelist():
            if name.startswith('openhands_cli/') and not name.endswith('/'):
                local = project / name
                if not local.is_file() or local.read_bytes() != wheel.read(name):
                    raise ValueError('Built CLI differs from pinned source: ' + name)
                checked.append(name)
    agent_runtime = None
    if python_runtime is not None:
        source_runtime = Path(python_runtime).resolve()
        interpreter = source_runtime / 'bin/python3.12'
        probe = subprocess.run([str(interpreter), '-c',
            'import json,sys,sysconfig; print(json.dumps({"version":sys.version,"abi":sys.implementation.cache_tag,"platform":sysconfig.get_platform()}))'],
            text=True, capture_output=True, check=True, timeout=30)
        import json
        runtime_identity = json.loads(probe.stdout)
        if runtime_identity['abi'] != 'cpython-312' or runtime_identity['platform'] != 'linux-x86_64':
            raise ValueError('Agent interpreter must be CPython 3.12 linux-x86_64')
        shutil.copytree(source_runtime, staging / 'python', symlinks=False,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (staging / 'python').chmod(0o755)
        agent_runtime = {'path': 'python', 'source': str(source_runtime), **runtime_identity,
                         'network_download': False, 'scope': 'private OpenHands interpreter; task Python unchanged'}
    files = [{'path': f.relative_to(staging).as_posix(), 'size': f.stat().st_size, 'sha256': sha256(f)}
             for f in sorted(staging.rglob('*')) if f.is_file()]
    manifest = {'schema_version': 1, 'cli_revision': CLI_REVISION,
        'source_url': 'https://codeload.github.com/OpenHands/OpenHands-CLI/tar.gz/' + CLI_REVISION,
        'source_archive_sha256': sha256(cli_archive), 'cli_source_files_verified': len(checked),
        'source_adaptations': adaptations,
        'python': '3.12', 'platform': 'linux/amd64', 'sdk_version': '1.22.1',
        'tools_version': '1.22.1', 'files': files, 'network_install': False}
    if agent_runtime is not None:
        manifest['agent_runtime'] = agent_runtime
    manifest['receipt_sha256'] = hash_json(manifest)
    write_json(staging / 'manifest.json', manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(target)
    verify_snapshot(target)
    return {'path': str(target), 'manifest': manifest}


def install_command(wheelhouse="/opt/dycon-wheels"):
    """Install the pinned private OpenHands runtime from a mounted snapshot."""
    import shlex
    root = shlex.quote(wheelhouse)
    probe = ('import importlib.metadata as m, json; '
        'expected={"openhands":"1.16.0","openhands-sdk":"1.22.1",'
        '"openhands-tools":"1.22.1","httpx":"0.28.1","httpcore":"1.0.9"}; '
        'actual={k:m.version(k) for k in expected}; assert actual==expected, actual; '
        'import openhands.sdk, openhands.tools, openhands_cli.acp_impl; print(json.dumps(actual))')
    return ('export LITELLM_LOCAL_MODEL_COST_MAP=true OPENHANDS_SUPPRESS_BANNER=1 && '
        'export UV_PYTHON_DOWNLOADS=never UV_OFFLINE=1 UV_NO_CACHE=1; '
        f'PYTHON_BIN=""; for candidate in {root}/python/bin/python3.12 '
        '$(command -v python3.12 2>/dev/null) $(command -v python3 2>/dev/null); do '
        'if "$candidate" -c "import sys; assert sys.version_info[:2] == (3,12)" >/dev/null 2>&1; then '
        'PYTHON_BIN="$candidate"; break; fi; done; '
        'test -n "$PYTHON_BIN" || '
        '{ echo "No executable CPython 3.12 in the task image or bound runtime publication" >&2; exit 86; }; '
        f'{root}/uv venv --python $PYTHON_BIN --no-managed-python /opt/dycon-openhands && '
        f'{root}/uv pip sync --python /opt/dycon-openhands/bin/python '
        f'--no-index --find-links {root}/wheels --require-hashes '
        f'{root}/requirements.txt && '
        'ln -sf /opt/dycon-openhands/bin/openhands /usr/local/bin/openhands && '
        '/opt/dycon-openhands/bin/python -c ' + shlex.quote(probe) + ' && openhands --help >/dev/null')


async def install_runtime(env, cfg, output, ownership):
    """The same private offline installation hook for startup audits and attempts."""
    output = Path(output)
    profile = cfg['execution_context']['profile']
    result = await env.exec(install_command(), user='root', timeout_sec=900)
    write_json(output / 'dependency-install.json', {
        'schema_version': 2, 'return_code': result.return_code, 'stdout': result.stdout,
        'stderr': getattr(result, 'stderr', ''), 'network_install': False, 'ownership': ownership,
        'configuration_sha256': cfg['config_hash'], 'profile_sha256': cfg['execution_context']['profile_sha256'],
        'dependency_snapshot_sha256': sha256(Path(profile['dependency_snapshot']) / 'manifest.json'),
        'policy': 'private-cpython312-offline-v2', 'at': now()})
    if result.return_code:
        from types import SimpleNamespace
        from runtime.containers.dependencies import setup_error
        context = SimpleNamespace(run_id=ownership['run'], instance_id=ownership['task'],
                                  attempt_id=ownership['attempt'], role='skillsbench-agent', output_dir=output)
        raise setup_error(cfg, context, (result.stdout or '') + (getattr(result, 'stderr', '') or ''),
                          stage='skillsbench_agent_install', returncode=result.return_code,
                          reason='python_or_platform_incompatible' if result.return_code == 86 else 'dependency_install_failed',
                          log_path=output / 'dependency-install.json', wheelhouse=profile['dependency_snapshot'])


async def probe_runtime(env, cfg, output, ownership, workspace):
    import shlex
    output = Path(output)
    source = Path(__file__).with_name('sandbox_probe.py').read_text()
    seconds = cfg['timeouts']['model_seconds']
    command = 'export HOME=/home/agent LITELLM_LOCAL_MODEL_COST_MAP=true OPENHANDS_SUPPRESS_BANNER=1; '
    command += 'cd /home/agent && /opt/dycon-openhands/bin/python -c ' + shlex.quote(source)
    # -c cannot run the __main__ block with arguments without sys.argv[0].
    command += ' ' + shlex.quote(workspace) + ' ' + str(seconds)
    result = await env.exec(command, user=cfg['execution_context']['profile']['sandbox_user'], timeout_sec=120)
    record = {'schema_version': 1, 'ownership': ownership, 'return_code': result.return_code,
              'stdout': result.stdout, 'stderr': getattr(result, 'stderr', ''),
              'probe_sha256': sha256(Path(__file__).with_name('sandbox_probe.py')), 'at': now(), 'verified': False}
    if result.return_code == 0:
        try:
            record['probe'] = __import__('json').loads(result.stdout.splitlines()[-1])
            record['verified'] = record['probe'].get('verified') is True
        except (ValueError, IndexError):
            pass
    write_json(output / 'agent-environment.json', record)
    if not record['verified']:
        raise ValueError('Non-root OpenHands environment/tool initialization failed; see agent-environment.json')
    return record

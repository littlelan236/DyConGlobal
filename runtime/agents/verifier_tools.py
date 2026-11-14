"""Shared sandbox-local tools for official online and offline verifiers."""


def resource_inventory_command():
    """Read bounded dependency declarations from the actual leased task image."""
    return "/opt/dycon-wheels/python/bin/python3.12 - <<'DYCON_RESOURCE_INVENTORY'\n" + r'''
import json, pathlib, platform, shutil, subprocess, sys
roots = [pathlib.Path('/home/github/build/failed/AgentOps-AI/agentops'),
         pathlib.Path('/opt/syzkaller'), pathlib.Path('/root/SimPO'),
         *[pathlib.Path('/app') / name for name in ['arrow','ujson','black','ipython','minisgl']]]
files = {}
for root in roots:
    for name in ['tox.ini','pyproject.toml','setup.py','setup.cfg','requirements.txt','requirements-dev.txt','go.mod','go.sum','.python-version']:
        p = root / name
        if p.is_file() and p.stat().st_size <= 50000:
            files[str(p)] = p.read_text(errors='replace')
print(json.dumps({'python':sys.version, 'executable':sys.executable,
    'platform':platform.platform(), 'files':files,
    'tools':{n:shutil.which(n) for n in ['go','make','gcc','curl','pdftotext','bc','python3.7','python3.10','python3.12']},
    'browsers':[str(p) for p in pathlib.Path('/opt/ms-playwright').glob('*')],
    'interpreters':[str(p) for p in pathlib.Path('/opt/hostedtoolcache/Python').glob('*/x64/bin/python3.*') if not p.name.endswith('config')]}))
DYCON_RESOURCE_INVENTORY
'''


async def capture_resource_sources(env, task_id, output, ownership):
    """Export image-owned build inputs through the existing sandbox transport."""
    import shlex
    from pathlib import Path
    from core.io import sha256, write_json
    paths = {
        'syzkaller-ppdev-syzlang':['/opt/syzkaller'],
        'setup-fuzzing-py':['/app/arrow','/app/ujson','/app/black','/app/ipython','/app/minisgl'],
        'fix-build-agentops':['/home/github/build/failed/AgentOps-AI/agentops','/opt/hostedtoolcache/Python'],
        'latex-formula-extraction':['/opt/ms-playwright'],
    }.get(task_id)
    if not paths:
        return
    dest=Path(output)/'diagnostics/resource-sources.tar.gz'
    remote='/tmp/dycon-resource-sources.tar.gz'
    try:
        packed=await env.exec('tar -czf '+shlex.quote(remote)+' -C / '+
            ' '.join(shlex.quote(p.lstrip('/')) for p in paths),user='root',timeout_sec=600)
        if packed.return_code:
            raise RuntimeError('Could not archive image resource sources: '+str(packed.stderr)[-1000:])
        await env.download_file(remote,dest)
        write_json(dest.with_suffix('.json'),{'schema_version':1,'ownership':ownership,
            'paths':paths,'sha256':sha256(dest),'size':dest.stat().st_size,
            'source':'leased frozen task image before agent execution'})
    finally:
        await env.exec('rm -f '+shlex.quote(remote),user='root',timeout_sec=30)


def verifier_tool_command():
    # BenchFlow deliberately excludes answer workspaces from its trusted PATH.
    # Keep its invariants: install only the published curl shim in the existing
    # trusted /usr/local/bin, and let official scripts source the local uv env.
    return r'''set -eu
test ! -L /root/.local
test ! -L /root/.local/bin
mkdir -p /root/.local/bin /usr/local/bin
uv_source=
for source in /opt/dycon-wheels/uv /opt/dycon-skillsbench-tools/uv; do
    if test -x "$source"; then uv_source=$source; break; fi
done
if test -z "$uv_source"; then printf 'uv unavailable\n'; exit 0; fi
rm -f /root/.local/bin/uv /root/.local/bin/uvx /root/.local/bin/env
cp "$uv_source" /root/.local/bin/uv
printf '%s\n' '#!/bin/sh' 'exec /root/.local/bin/uv tool run "$@"' > /root/.local/bin/uvx
printf '%s\n' 'export PATH=/root/.local/bin:/opt/dycon-wheels:$PATH' > /root/.local/bin/env
chmod 755 /root/.local/bin/uv /root/.local/bin/uvx
chmod 644 /root/.local/bin/env
if test -x /opt/dycon-skillsbench-tools/curl; then
    rm -f /usr/local/bin/curl
    cp /opt/dycon-skillsbench-tools/curl /usr/local/bin/curl
    chmod 755 /usr/local/bin/curl
    sha256sum /usr/local/bin/curl
fi
sha256sum /root/.local/bin/uv /root/.local/bin/uvx /root/.local/bin/env
/root/.local/bin/uv --version
'''


def verifier_wheel_links_command():
    """Expose immutable wheels with Requires-Python metadata for legacy pip."""
    return r'''set -eu
if ! test -d /opt/dycon-wheels/wheels; then exit 0; fi
test ! -L /opt/dycon-verifier-links
mkdir -p /opt/dycon-verifier-links
test ! -L /opt/dycon-verifier-links/index.html
wheel_python=/opt/dycon-wheels/python/bin/python3.12
if ! test -x "$wheel_python"; then wheel_python=python3; fi
"$wheel_python" - <<'DYCON_WHEEL_LINKS'
from pathlib import Path
from email.parser import BytesParser
import html
import hashlib
import os
import zipfile
from urllib.parse import quote

sources = [Path(p) for p in os.environ.get('DYCON_VERIFIER_FIND_LINKS','/opt/dycon-wheels/wheels').split()]
rows = []
flat=Path('/opt/dycon-verifier-links/wheels')
if flat.is_symlink():raise ValueError('Unsafe flat wheel directory')
flat.mkdir(exist_ok=True)
for wheel in sorted({p for source in sources for p in source.glob('*.whl')}):
    link=flat/wheel.name
    if link.exists():
        if hashlib.sha256(link.read_bytes()).digest()!=hashlib.sha256(wheel.read_bytes()).digest():
            raise ValueError('Conflicting immutable wheel filename: '+wheel.name)
    else:link.symlink_to(wheel)
    with zipfile.ZipFile(wheel) as archive:
        metadata = [n for n in archive.namelist()
                    if n.endswith('.dist-info/METADATA') and n.count('/') == 1]
        if len(metadata) != 1:
            raise ValueError('Wheel metadata must be unique: ' + wheel.name)
        with archive.open(metadata[0]) as stream:
            raw = stream.read(1048577)
        if len(raw) > 1048576:
            raise ValueError('Wheel metadata exceeds diagnostic bound')
        requires = BytesParser().parsebytes(raw, headersonly=True).get('Requires-Python')
    attribute = (' data-requires-python="' + html.escape(requires, quote=True) + '"') if requires else ''
    rows.append('<a href="file://' + quote(str(wheel)) + '"' + attribute + '>' + html.escape(wheel.name) + '</a>')
target = Path('/opt/dycon-verifier-links/index.html')
target.write_text('<!doctype html>\n' + '\n'.join(rows) + '\n', encoding='utf-8')
print('Immutable verifier wheel links:', len(rows))
DYCON_WHEEL_LINKS
sha256sum /opt/dycon-verifier-links/index.html
'''


def agentops_interpreter_command():
    """Expose existing image toolcache interpreters to the unchanged tox matrix."""
    return r'''set -eu
for minor in 7 8 9 10 11 12; do
    name=python3.$minor
    if command -v "$name" >/dev/null 2>&1; then "$name" --version; continue; fi
    found=0
    for candidate in /opt/hostedtoolcache/Python/3.$minor.*/x64/bin/"$name"; do
        if test -x "$candidate" && "$candidate" --version; then
            test ! -e /usr/local/bin/"$name"
            test ! -L /usr/local/bin/"$name"
            ln -s "$candidate" /usr/local/bin/"$name"
            found=1
            break
        fi
    done
    if test "$found" = 0; then printf 'Image interpreter unavailable: %s\n' "$name"; fi
done
'''


async def prepare_verifier_tools(env, task_id=None, variables=None):
    import shlex
    selected = variables or {}
    prefix = ''.join('export '+key+'='+shlex.quote(str(selected[key]))+'\n'
        for key in ('DYCON_VERIFIER_FIND_LINKS','GOROOT') if key in selected)
    extra = ''
    if selected.get('GOROOT'):
        extra = r'''
test -x "$GOROOT/bin/go"
ln -sf "$GOROOT/bin/go" /usr/local/bin/go
ln -sf "$GOROOT/bin/gofmt" /usr/local/bin/gofmt
mkdir -p /opt/dycon-go-modules /tmp/dycon-go-build
cp -aL /opt/dycon-wheels/verifier/assets/go-modules/cache/. /opt/dycon-go-modules/
GOTOOLCHAIN=local "$GOROOT/bin/go" version
'''
    result = await env.exec(prefix + verifier_tool_command() + '\n' + verifier_wheel_links_command() + extra,
                            user='root', timeout_sec=300 if selected.get('GOROOT') else 30)
    if result.return_code:
        raise RuntimeError('Could not prepare trusted writable SkillsBench verifier tools')
    if task_id == 'fix-build-agentops':
        interpreters = await env.exec(agentops_interpreter_command(), user='root', timeout_sec=30)
        if interpreters.return_code:
            raise RuntimeError('Could not expose image-owned AgentOps tox interpreters')
        result.stdout = (result.stdout or '') + '\n' + (interpreters.stdout or '')
    return result

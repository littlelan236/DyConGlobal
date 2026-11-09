"""Bounded real dependency-source probe inside an owned task environment."""
from __future__ import annotations

import shlex
import re
from urllib.parse import urlsplit

from core.io import hash_json, now, write_json


async def probe_verifier_index(env, variables, output, ownership):
    """Exercise pip, uv pip and uvx with fresh candidates and bounded downloads."""
    index = variables['PIP_INDEX_URL']
    script = r'''set -eu
probe_root=$(mktemp -d /tmp/dycon-index-probe.XXXXXXXX)
trap 'rm -rf "$probe_root"' EXIT
export UV_NO_CACHE=1 UV_HTTP_TIMEOUT=15 UV_HTTP_RETRIES=0 UV_PYTHON_DOWNLOADS=never
export UV_TOOL_DIR="$probe_root/tools" UV_TOOL_BIN_DIR="$probe_root/bin"
export PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=15 PIP_RETRIES=0
. /root/.local/bin/env
python_bin=/opt/dycon-wheels/python/bin/python3.12
python3 -c 'import sys; print("TASK_PYTHON=" + sys.version.split()[0])' || true
"$python_bin" -c 'import sys; print("AGENT_PYTHON=" + sys.version.split()[0]); assert sys.version_info[:2] == (3,12)'
"$python_bin" -m venv "$probe_root/pip"
"$probe_root/pip/bin/python" -m pip download --verbose --verbose --no-deps --no-cache-dir --dest "$probe_root/download" wheel==0.45.1
printf '\nDYCON_PROBE_PIP_OK\n'
uv pip install --verbose --python "$python_bin" --target "$probe_root/target" --no-deps wheel==0.45.1
printf '\nDYCON_PROBE_UV_PIP_OK\n'
uvx --verbose --python "$python_bin" --from wheel==0.45.1 wheel version
printf '\nDYCON_PROBE_UVX_OK\n'
'''
    probe_variables = {**variables, 'PIP_FIND_LINKS': '', 'UV_FIND_LINKS': ''}
    result = await env.exec('bash -c ' + shlex.quote(script), env=probe_variables, user='root', timeout_sec=150)
    from runtime.agents.skillsbench import _safe_verifier_text
    logs = _safe_verifier_text((result.stdout or '') + '\n' + (getattr(result, 'stderr', '') or ''))
    host = urlsplit(index).hostname
    requests = [line[:1000] for line in logs.splitlines()
                if host and host in line and '.whl' in line
                and ('Sending fresh GET request' in line or re.search(r'"GET .*" 200 ', line))]
    record = {'schema_version': 1, 'ownership': ownership, 'at': now(),
              'selected_index': index, 'network_cache': 'fresh temporary directory; pip no-cache; uv no-cache',
              'local_candidates': 'find-links cleared only for this download probe',
              'return_code': result.return_code, 'policy': 'pip-uv-pip-uvx-real-download-v1',
              'source_confirmed': bool(requests), 'download_request_evidence': requests[-8:],
              'pip_ok': 'DYCON_PROBE_PIP_OK' in logs,
              'uv_pip_ok': 'DYCON_PROBE_UV_PIP_OK' in logs,
              'uvx_ok': 'DYCON_PROBE_UVX_OK' in logs,
              'stdout': _safe_verifier_text(result.stdout or '')[-16000:],
              'stderr': _safe_verifier_text(getattr(result, 'stderr', '') or '')[-24000:]}
    record['verified'] = (record['return_code'] == 0 and record['source_confirmed']
                          and record['pip_ok'] and record['uv_pip_ok'] and record['uvx_ok'])
    record['receipt_sha256'] = hash_json(record)
    write_json(output / 'diagnostics/verifier-index-probe.json', record)
    return record

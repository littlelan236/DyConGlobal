"""Exercise the published resolver binaries against isolated real indexes."""
import contextlib
import http.server
import os
from pathlib import Path
import subprocess
import sys
import threading
import zipfile

import pytest


def wheel_bytes(root, version):
    name = f'dycon_index_probe-{version}-py3-none-any.whl'
    path = root / name
    dist = f'dycon_index_probe-{version}.dist-info'
    with zipfile.ZipFile(path, 'w') as wheel:
        wheel.writestr('dycon_index_probe.py', f'def main():\n    print("PROBE_VERSION={version}")\n')
        wheel.writestr(dist + '/METADATA', f'Metadata-Version: 2.1\nName: dycon-index-probe\nVersion: {version}\n')
        wheel.writestr(dist + '/WHEEL', 'Wheel-Version: 1.0\nGenerator: dycon-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
        wheel.writestr(dist + '/entry_points.txt', '[console_scripts]\ndycon-index-probe=dycon_index_probe:main\n')
        wheel.writestr(dist + '/RECORD', '')
    return name


@contextlib.contextmanager
def index(root, version, *, absent=False):
    root.mkdir()
    filename = wheel_bytes(root, version)
    requests = []
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            requests.append(self.path)
            if absent and self.path.startswith('/simple/'):
                self.send_error(404)
                return
            if self.path == '/simple/dycon-index-probe/':
                content = f'<a href="/{filename}">{filename}</a>'.encode()
                content_type = 'text/html'
            elif self.path == '/' + filename:
                content = (root / filename).read_bytes()
                content_type = 'application/octet-stream'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(content)
        def log_message(self, *args):
            pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/simple/', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_real_pip_uv_pip_and_uvx_use_explicit_index_and_default_precedence(tmp_path):
    from runtime.agents.skillsbench import verifier_environment_policy
    uv = Path(__file__).parents[1] / 'runtime/dependency-snapshots/openhands-sdk-format-v4-20261101/uv'
    if not uv.exists():
        pytest.skip('Published SkillsBench uv binary is not available')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PIP_', 'UV_'))
           and k.lower() not in ('http_proxy', 'https_proxy', 'all_proxy')}
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK='1',
               UV_PYTHON_DOWNLOADS='never', UV_NO_CONFIG='1', UV_NO_CACHE='1',
               UV_TOOL_DIR=str(tmp_path / 'tools'), UV_TOOL_BIN_DIR=str(tmp_path / 'bin'))
    # Runner uses uv and need not contain pip. Bootstrap only a temporary
    # test interpreter from CPython's bundled ensurepip wheels, offline.
    import venv
    venv.EnvBuilder(with_pip=True).create(tmp_path / 'pip-env')
    pip_python = str(tmp_path / 'pip-env/bin/python')
    with index(tmp_path / 'selected', '1.0') as (selected, selected_requests), \
         index(tmp_path / 'decoy', '9.0') as (decoy, decoy_requests):
        configured, _ = verifier_environment_policy(
            {'network': 'public', 'verifier_python_index': selected}, {'UV_DEFAULT_INDEX': 'https://pypi.org/simple'})
        active = {**env, **configured}
        commands = [
            [pip_python, '-m', 'pip', 'install', '--no-cache-dir', '--no-deps', '--target',
             str(tmp_path / 'pip-target'), 'dycon-index-probe==1.0'],
            [str(uv), 'pip', 'install', '--no-deps', '--python', sys.executable, '--target',
             str(tmp_path / 'uv-target'), 'dycon-index-probe==1.0'],
            [str(uv), 'tool', 'run', '--python', sys.executable, '--from',
             'dycon-index-probe==1.0', 'dycon-index-probe'],
        ]
        for command in commands:
            before = len(selected_requests)
            result = subprocess.run(command, env=active, capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, result.stderr
            assert any('.whl' in url for url in selected_requests[before:])
        assert 'PROBE_VERSION=1.0' in result.stdout
        assert not decoy_requests
        # uv 0.13.5 gives UV_DEFAULT_INDEX precedence over UV_INDEX_URL.
        # Verify why the project sets all three selectors to the same URL.
        active['UV_DEFAULT_INDEX'] = decoy
        result = subprocess.run([str(uv), 'tool', 'run', '--python', sys.executable,
            '--from', 'dycon-index-probe==9.0', 'dycon-index-probe'], env=active,
            capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert 'PROBE_VERSION=9.0' in result.stdout
        assert any('.whl' in url for url in decoy_requests)
        # uv's existing first-index semantics can reject a version missing
        # from the extra index. Do not silently change its resolver strategy.
        active['UV_DEFAULT_INDEX'] = selected
        result = subprocess.run([str(uv), 'tool', 'run', '--python', sys.executable,
            '--index-url', decoy, '--extra-index-url', selected, '--from',
            'dycon-index-probe==9.0', 'dycon-index-probe'], env=active,
            capture_output=True, text=True, timeout=30)
        assert result.returncode != 0 and 'first index' in result.stderr
        # v2 makes the dedicated repository the high-priority extra index.
        # Keep first-index resolution and prove the generic copy is ignored.
        active.update(UV_DEFAULT_INDEX='', UV_INDEX_URL=selected)
        result = subprocess.run([str(uv), 'tool', 'run', '--python', sys.executable,
            '--index-url', selected, '--extra-index-url', decoy, '--from',
            'dycon-index-probe==9.0', 'dycon-index-probe'], env=active,
            capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert 'PROBE_VERSION=9.0' in result.stdout
        with index(tmp_path / 'generic-without-special-package', '1.0', absent=True) as (generic, _):
            active.update(UV_DEFAULT_INDEX='', UV_INDEX_URL=generic)
            result = subprocess.run([str(uv), 'tool', 'run', '--python', sys.executable,
                '--index-url', decoy, '--extra-index-url', generic, '--from',
                'dycon-index-probe==9.0', 'dycon-index-probe'], env=active,
                capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, result.stderr
            assert 'PROBE_VERSION=9.0' in result.stdout

"""Real child HTTP processes exercise parallel launch, capture ownership and rollback."""
import json
import socket
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from core.io import read_json
from runtime.serving.manager import ServingManager
from runtime.serving.pool import ServingPool
from tests.test_serving_worker_pool import sharded


def ports():
    with socket.socket() as first, socket.socket() as second:
        first.bind(('127.0.0.1', 0))
        second.bind(('127.0.0.1', 0))
        return [first.getsockname()[1], second.getsockname()[1]]


@pytest.mark.parametrize('capture', [False, True])
@pytest.mark.parametrize('failure', [None, 0, 1])
def test_real_parallel_processes_share_capture_and_rollback(cfg, tmp_path, monkeypatch, capture, failure):
    import runtime.serving.manager as module
    cfg = sharded(cfg)
    cfg['serving'].update(startup_mode='parallel', startup_seconds=5, shutdown_seconds=2)
    allocated_ports = ports()
    cfg['serving']['replica_base_urls'] = [f'http://127.0.0.1:{port}/v1' for port in allocated_ports]
    cfg['serving']['base_url'] = cfg['serving']['replica_base_urls'][0]
    checkpoint = tmp_path / 'model'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text('{}')
    cfg['model']['path'] = str(checkpoint)
    cfg['capture'].update(enabled=capture, layers=[0])
    cfg['agent']['backend'] = 'minisweagent'
    reserved, released, controllers, closed, children = [], [], [], [], []

    @contextmanager
    def reserve(root, python, gpus, **kwargs):
        reserved.append(gpus[0])
        try:
            yield {'device_kind': 'physical', 'lock_paths': []}
        finally:
            released.append(gpus[0])

    class Controller:
        def __init__(self, *args):
            controllers.append(self)
            self.export_root = tmp_path / 'capture'
        def _write(self, *args): pass
        def write_configuration(self): pass

    class Server:
        base_url = 'http://127.0.0.1:1'
        def __init__(self, *args): pass
        def start(self): return self
        def close(self): closed.append(True)

    monkeypatch.setattr(module, 'cuda_visible_devices', lambda devices: ','.join(map(str, devices)))
    monkeypatch.setattr(module, 'reserve_serving_devices', reserve)
    monkeypatch.setattr('hidden.backends.vllm.engine.serving_interpreter', lambda cfg: Path(sys.executable))
    monkeypatch.setattr(module, 'OnlineCaptureController', Controller)
    monkeypatch.setattr(module, 'CaptureControlServer', Server)
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: None)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *args, **kwargs: json.dumps({
        'newline_token_ids': [1], 'newline_token_texts': ['\n'], 'open_token_ids': [2], 'close_token_ids': [3]}))
    real_popen = module.subprocess.Popen
    def popen(*args, **kwargs):
        assert len(reserved) == 2, 'a process spawned before all devices were reserved'
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(module.subprocess, 'Popen', popen)
    monkeypatch.setattr(module, 'sha256', lambda path: 'fixture-runtime')

    def command(self, python, host, port):
        if failure is not None and port == allocated_ports[failure]:
            code = 'import time; time.sleep(0.1); raise SystemExit(3)'
        else:
            code = (
                'import time; from http.server import BaseHTTPRequestHandler, HTTPServer; '
                'time.sleep(0.3); '
                'H=type("H", (BaseHTTPRequestHandler,), {"do_GET": lambda s: '
                '(s.send_response(200), s.send_header("Content-Type", "application/json"), '
                's.end_headers(), s.wfile.write(b\'{"data":[{"id":"fixture"}]}\')), '
                '"log_message": lambda *args: None}); '
                f'HTTPServer(("127.0.0.1", {port}), H).serve_forever()'
            )
        return [python, '-c', code]
    monkeypatch.setattr(ServingManager, 'build_args', command)
    output = tmp_path / 'serving'
    pool = ServingPool(cfg, output, 'parallel-test')
    if failure is None:
        with pool:
            assert pool.capacity == 4
            assert len(children) == 2
            if capture:
                assert pool.workers[0].manager.capture_controller is pool.workers[1].manager.capture_controller
            timings = [worker.manager.startup_timing for worker in pool.workers]
            assert max(t['process_created_at'] for t in timings) < min(t['api_ready_at'] for t in timings)
            assert all(read_json(w.output / 'launch.json')['startup_mode'] == 'parallel' for w in pool.workers)
    else:
        with pytest.raises(RuntimeError, match='server exited'):
            with pool:
                pytest.fail('a failed pool cannot become usable')
    assert len(children) == 2
    assert all(child.poll() is not None for child in children)
    assert sorted(reserved) == sorted(released) == [2, 3]
    assert len(controllers) == len(closed) == (1 if capture else 0)
    receipt = read_json(output / 'worker-pool.json')
    assert receipt['startup_mode'] == 'parallel' and receipt['state'] == 'stopped'
    assert all('cleanup_finished_at' in row['startup_timing'] for row in receipt['workers'])


@pytest.mark.parametrize('mode', ['serial', 'parallel'])
@pytest.mark.parametrize('prepare_failure', [0, 1])
def test_preparation_failure_releases_all_resources(cfg, tmp_path, monkeypatch, mode, prepare_failure):
    instances = []
    class Manager:
        def __init__(self, cfg, output, **kwargs):
            self.index = len(instances)
            self.closed = False
            instances.append(self)
        def prepare(self):
            if self.index == prepare_failure:
                raise ValueError('preparation rejected')
        def __enter__(self):
            self.prepare()
        def __exit__(self, *args):
            self.closed = True
    monkeypatch.setattr('runtime.serving.pool.ServingManager', Manager)
    cfg = sharded(cfg)
    cfg['serving']['startup_mode'] = mode
    with pytest.raises(ValueError, match='preparation rejected'):
        with ServingPool(cfg, tmp_path / 'serving', 'preparation'):
            pytest.fail('preparation failed')
    assert all(manager.closed for manager in instances)

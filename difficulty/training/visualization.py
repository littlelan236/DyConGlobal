"""Owned TensorBoard service lifecycle and synchronous training event sink."""
import importlib.util
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from core.io import identifier, read_json, write_json
from core.locking import exclusive_artifact


def process_identity(pid):
    try:
        # starttime survives PID reuse; proc stat comm may contain spaces/parentheses.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def service_file(root, attempt_id):
    return Path(root) / 'state/difficulty-training' / (identifier(attempt_id) + '.json')


def status(root, attempt_id):
    path = service_file(root, attempt_id)
    if not path.exists():
        return {'attempt_id': attempt_id, 'running': False}
    record = read_json(path)
    running = process_identity(record['pid']) == record['process_start']
    return {**record, 'running': running and record['state'] == 'running'}


@exclusive_artifact('difficulty-tensorboard', 'attempt_id')
def manage(cfg, attempt_id, *, action='status'):
    root = cfg['project_root']
    record = status(root, attempt_id)
    if action == 'stop' and record['running']:
        pid = record['pid']
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if b'tensorboard.main' not in command or str(record['logdir']).encode() not in command:
            raise ValueError('TensorBoard process ownership mismatch; refusing to signal')
        os.kill(pid, signal.SIGTERM)
        record.update(state='stopped', running=False)
        write_json(service_file(root, attempt_id), {k: v for k, v in record.items() if k != 'running'})
    return record


class Visualization:
    def __init__(self, cfg, attempt_id, options):
        self.cfg, self.attempt_id, self.options = cfg, attempt_id, options
        self.root = Path(cfg['project_root'])
        self.process = None
        self.writers = {}
        self.logdir = None

    def preflight(self):
        if not self.options.enabled:
            return
        if importlib.util.find_spec('tensorboard') is None:
            raise ValueError('TensorBoard is missing; install the locked analysis dependencies')
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        with socket.socket() as listener:
            listener.bind((self.options.host, self.options.port))

    def start(self, output):
        if not self.options.enabled or self.process is not None:
            return
        output = Path(output)
        logdir = output / 'events'
        self.logdir = logdir
        logdir.mkdir(parents=True, exist_ok=True)
        path = service_file(self.root, self.attempt_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-m', 'tensorboard.main', '--logdir', str(logdir),
                   '--host', self.options.host, '--port', str(self.options.port),
                   '--reload_interval', str(self.options.reload_interval), '--load_fast', 'false']
        with path.with_suffix('.log').open('ab') as log:
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(attempt_id=self.attempt_id, pid=self.process.pid,
                      process_start=process_identity(self.process.pid), state='running',
                      host=self.options.host, port=self.options.port, logdir=str(logdir), command=command)
        write_json(path, record)
        try:
            deadline = time.monotonic() + 30
            url = f'http://127.0.0.1:{self.options.port}'
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f'TensorBoard exited; see {path.with_suffix(".log")}')
                try:
                    with opener.open(url, timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(0.1)
            else:
                raise TimeoutError('TensorBoard HTTP readiness timeout')
            print(f'TensorBoard ready: {url}; attempt={self.attempt_id}', flush=True)
            print(f'Local PC: ssh -N -L 16006:127.0.0.1:{self.options.port} '
                  'wangtianyi@10.249.187.9 ; browser http://localhost:16006', flush=True)
        except BaseException:
            self.close(success=False)
            raise

    def write(self, target, step, values):
        if not self.options.enabled:
            return
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError('TensorBoard service is not running')
        if target not in self.writers:
            from torch.utils.tensorboard import SummaryWriter
            self.writers[target] = SummaryWriter(
                str(self.logdir / target),
                flush_secs=self.options.flush_secs)
        for key, value in values.items():
            if value is not None:
                self.writers[target].add_scalar(key, value, step)
        # Each completed epoch/trial is immediately readable, even for short runs.
        self.writers[target].flush()

    def close_writers(self):
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()

    def close(self, *, success):
        self.close_writers()
        if self.process is not None and (not success or not self.options.keep_after_training):
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            path = service_file(self.root, self.attempt_id)
            record = read_json(path)
            record['state'] = 'stopped'
            write_json(path, record)

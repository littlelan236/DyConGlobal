"""Execution-scoped cancellation, durable requests and filesystem event wakes."""
from __future__ import annotations

import os
import threading
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

from core.io import now, read_json, write_json
from core.locking import directory_changes

DEFAULT_POLICY = {'control_request_seconds': 3.0, 'cooperative_grace_seconds': 10.0,
                  'term_grace_seconds': 30.0, 'kill_verify_seconds': 5.0,
                  'cleanup_seconds': 15.0}
CURRENT = ContextVar('dycon_cancellation', default=None)


class Cancelled(RuntimeError):
    """A user stop; never a model/format/infra/loop retry trigger."""


def policy(cfg):
    return {**DEFAULT_POLICY, **cfg.get('execution', {}).get('cancellation', {})}


def request_for(run, attempt_id=None):
    path = Path(run) / 'execution/cancel-request.json'
    if not path.exists():
        return None
    value = read_json(path)
    target = value.get('execution_attempt_id')
    if target is not None and attempt_id is not None and target != attempt_id:
        return None
    return value


class CancellationContext:
    def __init__(self, run=None, attempt_id=None):
        self.run = Path(run) if run is not None else None
        self.attempt_id = attempt_id or os.environ.get('DYCON_EXECUTION_ATTEMPT')
        self.event = threading.Event()
        self.request = None
        self._lock = threading.Lock()
        self._callbacks = []
        self._closed = threading.Event()
        self._pipe = os.pipe()
        self._watcher = None

    def add_callback(self, callback):
        with self._lock:
            self._callbacks.append(callback)
            cancelled = self.event.is_set()
        if cancelled:
            callback()
        def remove():
            with self._lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)
        return remove

    def cancel(self, request=None):
        with self._lock:
            if self.event.is_set():
                return
            self.request = request or {'reason': 'user cancellation'}
            self.event.set()
            callbacks = list(self._callbacks)
        if self.run is not None and self.attempt_id is not None:
            write_json(self.run / 'execution/attempts' / self.attempt_id / 'cancel-runner-ack.json',
                       {**self.request, 'acknowledged_at': now(), 'pid': os.getpid()})
        # Wakes are deliberately tiny; slow network control runs in its own callback thread.
        for callback in callbacks:
            try:
                callback()
            except Exception:
                import logging
                logging.getLogger(__name__).exception('Cancellation callback failed')

    def check(self):
        if self.event.is_set():
            raise Cancelled((self.request or {}).get('reason', 'Run cancelled'))

    def start(self):
        if self.run is None or self._watcher is not None:
            return self
        directory = self.run / 'execution'
        directory.mkdir(parents=True, exist_ok=True)

        def watch():
            with directory_changes(directory) as changed:
                while not self._closed.is_set():
                    request = request_for(self.run, self.attempt_id)
                    if request is not None:
                        self.cancel(request)
                        return
                    changed(1.0, (self._pipe[0],))  # Event-driven; periodic rescan covers overflow.

        self._watcher = threading.Thread(target=watch, name='dycon-cancel-watch', daemon=True)
        self._watcher.start()
        return self

    def close(self):
        self._closed.set()
        os.write(self._pipe[1], b'x')
        if self._watcher is not None:
            self._watcher.join(timeout=2)
        for fd in self._pipe:
            os.close(fd)


class CompletionWake:
    """Wake the scheduler for completed futures OR cancellation, without polling."""
    def __init__(self, cancellation):
        self.condition = threading.Condition()
        self._remove = cancellation.add_callback(self.notify)

    def close(self):
        self._remove()

    def notify(self, *_):
        with self.condition:
            self.condition.notify_all()

    def wait(self, futures, cancellation):
        with self.condition:
            self.condition.wait_for(lambda: cancellation.event.is_set() or any(f.done() for f in futures))
        return {future for future in futures if future.done()}


def cancellable_run(function):
    @wraps(function)
    def wrapped(cfg, run_id, *args, **kwargs):
        run = Path(cfg['project_root']) / 'runs' / run_id
        cancellation = CancellationContext(run)
        if run.exists():
            cancellation.start()
        token = CURRENT.set(cancellation)
        try:
            return function(cfg, run_id, *args, **kwargs)
        finally:
            CURRENT.reset(token)
            cancellation.close()
    return wrapped

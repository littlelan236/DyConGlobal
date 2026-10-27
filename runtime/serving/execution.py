"""Bounded, context-preserving offload lanes owned by one serving plugin.

Admission waits must never consume the workers needed to finish requests.
Cancellation of an asyncio waiter does not release its permit until the
underlying concurrent future has actually finished.
"""
from __future__ import annotations

import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor


class ExecutionLane:
    def __init__(self, name, workers, capacity):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name)
        self.capacity = capacity
        self._lock = threading.Lock()
        self._outstanding = 0
        self._running = 0
        self.closed = False

    async def run(self, function, *args, **kwargs):
        with self._lock:
            if self.closed:
                raise RuntimeError('Serving execution lane closed')
            if self._outstanding >= self.capacity:
                raise RuntimeError('Serving execution lane queue full')
            self._outstanding += 1
        context = contextvars.copy_context()
        def execute():
            with self._lock:
                self._running += 1
            try:
                return context.run(function, *args, **kwargs)
            finally:
                with self._lock:
                    self._running -= 1
        try:
            future = self.pool.submit(execute)
        except BaseException:
            with self._lock:
                self._outstanding -= 1
            raise

        def completed(_):
            with self._lock:
                self._outstanding -= 1

        future.add_done_callback(completed)
        return await asyncio.wrap_future(future)

    def status(self):
        with self._lock:
            return {'outstanding': self._outstanding, 'capacity': self.capacity,
                    'running': self._running, 'queued': self._outstanding - self._running, 'closed': self.closed}

    def close(self):
        with self._lock:
            self.closed = True
        self.pool.shutdown(wait=False, cancel_futures=True)

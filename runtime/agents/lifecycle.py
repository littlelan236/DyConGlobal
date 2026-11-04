"""Cancellation bridge for a single BenchFlow-owned rollout lifecycle."""
from __future__ import annotations

import asyncio
import contextlib
import time

from core.io import now, write_json


async def run_cancellable(rollout, cancellation, deadline, cleanup_seconds, *, evidence_path=None,
                          binding=None):
    """Wake during tools/verifier as well as during a model request."""
    if cancellation is not None:
        cancellation.check()
    loop = asyncio.get_running_loop()
    stop = loop.create_future()

    def notify():
        def wake():
            if not stop.done():
                stop.set_result(True)
        loop.call_soon_threadsafe(wake)

    remove = cancellation.add_callback(notify) if cancellation is not None else lambda: None
    started_at = now()
    started = time.monotonic()

    def evidence(event, **details):
        if evidence_path is not None:
            try:
                write_json(evidence_path, {"schema_version": 1, "binding": binding,
                    "started_at": started_at, "elapsed_seconds": time.monotonic() - started,
                    "event": event, **details, "observed_at": now()})
            except OSError:
                pass

    task = asyncio.create_task(rollout.run())
    try:
        done, _ = await asyncio.wait({task, stop}, timeout=deadline.remaining(float('inf')),
                                     return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            result = await task
            evidence("rollout_completed", task_cancelled=False)
            if cancellation is not None:
                cancellation.check()
            return result
        reason = "run_cancelled" if stop.done() and cancellation is not None else "attempt_deadline"
        evidence(reason, cancellation_requested=True, cancellation_action="asyncio_task_cancel",
                 operating_system_signal_sent=None, signal_observation="not_available_from_asyncio_task")
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, TimeoutError):
            await asyncio.wait_for(task, timeout=cleanup_seconds)
        if cancellation is not None:
            cancellation.check()
        if reason == "run_cancelled":
            cancellation.check()
        raise TimeoutError('SkillsBench attempt wall-clock deadline exceeded')
    finally:
        remove()
        stop.cancel()
        if not task.done():
            task.cancel()

"""Cancellation bridge for a single BenchFlow-owned rollout lifecycle."""
from __future__ import annotations

import asyncio
import contextlib

from core.cancellation import Cancelled


async def run_cancellable(rollout, cancellation, deadline, cleanup_seconds):
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
    task = asyncio.create_task(rollout.run())
    try:
        done, _ = await asyncio.wait({task, stop}, timeout=deadline.remaining(float('inf')),
                                     return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            result = await task
            if cancellation is not None:
                cancellation.check()
            return result
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, TimeoutError):
            await asyncio.wait_for(task, timeout=cleanup_seconds)
        if cancellation is not None:
            cancellation.check()
        raise TimeoutError('SkillsBench attempt wall-clock deadline exceeded')
    finally:
        remove()
        stop.cancel()
        if not task.done():
            task.cancel()

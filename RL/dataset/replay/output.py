"""Bounded single writer; GPU coordination never performs shard I/O."""
from queue import Empty, Full, Queue
from threading import Thread
import time


class AsyncWriter:
    def __init__(self, writer, capacity):
        self.writer = writer; self.finished = writer.finished
        self.queue = Queue(capacity); self.error = None; self.pending = set()
        self.peak = 0; self.seconds = 0.0; self.closed = False
        self.thread = Thread(target=self._run, name="replay-result-writer", daemon=True)
        self.thread.start()

    def check(self):
        if self.error: raise RuntimeError("Replay writer failed") from self.error

    def put(self, row, output, values, **binding):
        self.check(); bid = row["branch_sample_id"]
        if bid in self.finished or bid in self.pending: raise ValueError("Duplicate replay result")
        # Only compact bindings/tokens/values enter the queue; prompt payloads are
        # retained by group ownership rather than duplicated in writer backlog.
        from .artifacts import BINDINGS
        record = {k: row[k] for k in BINDINGS}
        record.update(completion_token_ids=row["completion_token_ids"], audit_enabled=row.get("audit_enabled", False))
        self.pending.add(bid)
        while True:
            self.check()
            try: self.queue.put((record, output, values, binding), timeout=.1); break
            except Full: continue
        self.peak = max(self.peak, self.queue.qsize())

    def _run(self):
        try:
            while True:
                try: item = self.queue.get(timeout=.1)
                except Empty:
                    if self.writer.committed_count != len(self.writer.records) and time.monotonic() - self.writer.last_commit >= self.writer.commit_seconds:
                        self.writer.checkpoint("replaying")
                    continue
                if item is None: break
                start = time.monotonic()
                row, output, values, binding = item
                self.writer.put(row, output, values, **binding)
                self.seconds += time.monotonic() - start
        except BaseException as error: self.error = error

    def verify_completed(self, *args, **kwargs): return self.writer.verify_completed(*args, **kwargs)

    def close(self):
        if self.closed: self.check(); return
        self.closed = True
        while self.thread.is_alive():
            self.check()
            try: self.queue.put(None, timeout=.1); break
            except Full: continue
        self.thread.join(timeout=60)
        if self.thread.is_alive(): raise RuntimeError("Replay writer did not stop")
        self.check(); self.writer.checkpoint("interrupted")

    def finish(self, *args):
        self.close()
        return self.writer.finish(*args)

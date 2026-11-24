"""Per-rank trace progress, with timings at explicit training phase boundaries."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path


class TraceLogger:
    def __init__(self, directory, rank, memory=None):
        self.directory, self.rank, self.memory = Path(directory), rank, memory
        self.phase, self.started, self.fields = None, None, {}
        self.directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, phase, **fields):
        import torch
        # Synchronize only at coarse phase transitions, never per layer/token.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.monotonic()
        self.fields.update(fields)
        row = {"observed_at": datetime.now(timezone.utc).isoformat(), "rank": self.rank,
               "phase": phase, "previous_phase": self.phase,
               "previous_phase_seconds": None if self.started is None else now - self.started,
               **self.fields}
        if self.memory is not None:
            row["memory"] = self.memory.logger.memory()
            if self.memory.offloader is not None:
                row["offload"] = self.memory.offloader.metrics()
            self.memory.phase(phase, **fields)
        with (self.directory / "trace-events.jsonl").open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        from core.io import write_json
        write_json(self.directory / "trace-progress.json", row)
        self.phase, self.started = phase, now

"""A shared wall-clock deadline; budgets are never silently extended."""

import time


class Deadline:
    def __init__(self, seconds=0):
        self.end = time.monotonic() + seconds if seconds else None

    def remaining(self, maximum):
        if self.end is None:
            return maximum
        remaining = self.end - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Pipeline deadline expired")
        return min(maximum, remaining)

    def check(self):
        self.remaining(float("inf"))

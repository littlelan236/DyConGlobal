"""Exact cross-step periodicity shared by both supported agent adapters."""
from collections import deque


def repeated_suffix(history, repeats, max_period, step, *, match_policy):
    if repeats is None:
        return None
    history = list(history)
    for period in range(1, max_period + 1):
        size = repeats * period
        if len(history) >= size and history[-size:] == history[-period:] * repeats:
            return {'period': period, 'repeats': repeats, 'start_step': step - size + 1,
                    'end_step': step, 'action_executed': False, 'match_policy': match_policy}
    return None


class StepLoopDetector:
    def __init__(self, repeats, max_period):
        self.repeats, self.max_period = repeats, max_period
        self.history = deque(maxlen=(repeats or 0) * max_period)
        self.steps = 0

    def feed(self, fingerprint, *, match_policy):
        step = self.steps
        self.steps += 1
        self.history.append(fingerprint)
        return repeated_suffix(self.history, self.repeats, self.max_period, step, match_policy=match_policy)

    def clear(self):
        self.history.clear()

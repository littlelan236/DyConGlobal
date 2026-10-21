"""Bounded CPU detection of exact periodic suffixes in generated reasoning."""

from core.io import hash_json


class ReasoningLoopDetector:
    VERSION = "exact_token_period_v1"
    _BASE = 1000003
    _MASK = (1 << 64) - 1

    def __init__(self, options, close_ids):
        self.options = options
        self.close_ids = set(close_ids)
        self.count = 0
        self.closed = False
        self.hit = None
        self.capacity = max(options["repeat_count"] * options["max_period_tokens"],
                            options["min_repeated_tokens"] + options["max_period_tokens"] - 1)
        self.tokens = [0] * self.capacity
        self.prefix = [0] * (self.capacity + 1)
        self.powers = [1]
        for _ in range(self.capacity):
            self.powers.append((self.powers[-1] * self._BASE) & self._MASK)
        self._hash = 0

    def _span_hash(self, start, end):
        width = self.capacity + 1
        return (self.prefix[end % width] - self.prefix[start % width] * self.powers[end - start]) & self._MASK

    def _check(self):
        opts, end = self.options, self.count
        for period in range(1, min(opts["max_period_tokens"], end // opts["repeat_count"]) + 1):
            repeats = max(opts["repeat_count"], (opts["min_repeated_tokens"] + period - 1) // period)
            size = repeats * period
            if size > end or self.tokens[(end - 1) % self.capacity] != self.tokens[(end - 1 - period) % self.capacity]:
                continue
            start = end - size
            if self._span_hash(start, end - period) != self._span_hash(start + period, end):
                continue
            # Hashes only prefilter; native IDs prove equality even on collision.
            if all(self.tokens[i % self.capacity] == self.tokens[(i + period) % self.capacity]
                   for i in range(start, end - period)):
                return {"rule": self.VERSION, "config_sha256": hash_json(opts),
                        "period_tokens": period, "repeats": repeats, "repeated_tokens": size,
                        "start_output_token": start, "end_output_token_exclusive": end,
                        "detected_at_output_tokens": end}
        return None

    def feed(self, new_ids):
        if self.closed or self.hit is not None:
            return self.hit
        for token in new_ids:
            if token in self.close_ids:
                self.closed = True
                break
            self.tokens[self.count % self.capacity] = token
            self.count += 1
            self._hash = (self._hash * self._BASE + token + 1) & self._MASK
            self.prefix[self.count % (self.capacity + 1)] = self._hash
            if (self.count >= self.options["min_generated_tokens"]
                    and self.count % self.options["check_interval_tokens"] == 0):
                self.hit = self._check()
                if self.hit is not None:
                    return self.hit
        return None

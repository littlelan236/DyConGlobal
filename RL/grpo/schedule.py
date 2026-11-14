"""Torch-free, versioned group ordering shared by preflight and preparation."""
import random


LEGACY_SHUFFLE = "stable_random_v1"
LENGTH_SHUFFLE = "prefix_length_pool128_v1"
POOL_GROUPS = 128


def shuffle_metadata(algorithm):
    if algorithm == LEGACY_SHUFFLE:
        return {"shuffle_algorithm": algorithm}
    if algorithm == LENGTH_SHUFFLE:
        return {"shuffle_algorithm": algorithm, "shuffle_pool_size": POOL_GROUPS}
    raise ValueError("Unsupported GRPO shuffle algorithm")


def order_groups(items, *, seed, batch_size, algorithm=LENGTH_SHUFFLE, prompt_length=None):
    """Shuffle compact items, length-sort candidate pools, then shuffle full windows.

    Sorting is stable: equal lengths retain their seeded random order. A pool's
    incomplete window is carried into the next pool; the single final partial
    window stays last so a flat schedule never merges it into another window.
    """
    shuffle_metadata(algorithm)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("GRPO group batch size must be positive")
    result = list(items)
    rng = random.Random(seed)
    rng.shuffle(result)
    if algorithm == LEGACY_SHUFFLE:
        return result
    if batch_size > POOL_GROUPS or prompt_length is None:
        raise ValueError("Length shuffle requires batch size <=128 and prompt lengths")
    windows = []
    pending = []
    offset = 0
    while offset < len(result):
        count = min(POOL_GROUPS - len(pending), len(result) - offset)
        pending.extend(result[offset:offset + count])
        offset += count
        pending.sort(key=prompt_length)
        full = len(pending) // batch_size * batch_size
        windows.extend(pending[i:i + batch_size] for i in range(0, full, batch_size))
        pending = pending[full:]
    rng.shuffle(windows)
    return [item for window in windows for item in window] + pending

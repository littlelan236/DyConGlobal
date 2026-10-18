"""Opt-in bound on the native APC lookup; native cache ownership is unchanged."""

from functools import wraps


def install():
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    original = KVCacheManager.get_computed_blocks
    if getattr(original, "_dycon_boundary", False):
        return

    @wraps(original)
    def bounded(manager, request):
        params = request.sampling_params
        extra = getattr(params, "extra_args", None) or {}
        boundary = extra.get("dycon_capture_cache_boundary")
        if boundary is None:
            return original(manager, request)
        if type(boundary) is not int or boundary < 0:
            raise ValueError("Invalid DyCon capture cache boundary")
        # Native lookup accepts the maximum hit length already; constrain that
        # argument before lookup so hybrid KV/GDN restoration can choose a valid
        # shorter hit. Never truncate already-returned blocks or recurrent state.
        lookup = manager.coordinator.find_longest_cache_hit

        def limited(block_hashes, max_cache_hit_length):
            return lookup(block_hashes, min(max_cache_hit_length, boundary))

        # Scheduler's cache lookup is synchronous and has no await/forward. The
        # instance method is restored even if native cache admission raises.
        manager.coordinator.find_longest_cache_hit = limited
        try:
            return original(manager, request)
        finally:
            manager.coordinator.find_longest_cache_hit = lookup

    bounded._dycon_boundary = True
    KVCacheManager.get_computed_blocks = bounded

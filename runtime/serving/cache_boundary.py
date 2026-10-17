"""Opt-in bound on the native APC lookup; native cache ownership is unchanged."""

from functools import wraps


def install():
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    dispatch = CudaGraphManager.dispatch
    if not getattr(dispatch, "_dycon_uniform_prompt", False):
        @wraps(dispatch)
        def uniform_prompt(manager, num_reqs, num_tokens, uniform_token_count,
                           num_active_loras, max_query_len=None):
            # A prompt chunk of one input token per request has the same
            # geometry as uniform decode. GDN's native builder classifies it
            # as decode already; supply the corresponding dispatch hint too.
            if uniform_token_count is None and num_tokens == num_reqs and max_query_len == 1:
                uniform_token_count = 1
            return dispatch(manager, num_reqs, num_tokens, uniform_token_count,
                            num_active_loras, max_query_len)

        uniform_prompt._dycon_uniform_prompt = True
        CudaGraphManager.dispatch = uniform_prompt

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

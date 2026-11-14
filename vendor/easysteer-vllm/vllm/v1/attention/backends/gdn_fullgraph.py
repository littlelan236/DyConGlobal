# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stable metadata for raw FP16 GDN full graphs, including mixed prefills.

The existing convolution and FLA chunk kernels already accept device sequence
boundaries. Keep their launch grids and metadata addresses stable across replay;
unused convolution programs use PAD_SLOT_ID and unused FLA chunks lie beyond
the last nonempty sequence. No padded program writes tokens or live state.
"""

import torch

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor


def supports_raw_full_graph(config):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        _resolve_gdn_prefill_backend,
    )

    parallel = config.parallel_config
    return (
        config.additional_config.get("raw_gdn_fullgraph", False) is True
        and config.model_config.enable_trace_replay
        and config.model_config.dtype == torch.float16
        and config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL
        and config.speculative_config is None
        and config.lora_config is None
        and parallel.tensor_parallel_size == 1
        and parallel.pipeline_parallel_size == 1
        and parallel.decode_context_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and config.cache_config.mamba_cache_mode == "align"
        and _resolve_gdn_prefill_backend(config)[1] == "triton"
    )


class RawGDNGraphBuffers:
    def __init__(self, config, kv_cache_spec, device):
        self.config = config
        self.kv_cache_spec = kv_cache_spec
        self.max_tokens = config.compilation_config.max_cudagraph_capture_size or config.scheduler_config.max_num_batched_tokens
        n = config.scheduler_config.max_num_seqs
        self.query = torch.empty(n + 1, dtype=torch.int32, device=device)
        self.states = torch.empty(n, dtype=torch.int32, device=device)
        self.initial = torch.empty(n, dtype=torch.bool, device=device)
        self.chunk_offsets = torch.empty(n + 1, dtype=torch.int32, device=device)
        self.chunks = torch.empty(((self.max_tokens + 63) // 64 + n, 2), dtype=torch.int32, device=device)
        self.conv_batch = torch.empty((self.max_tokens + 7) // 8 + n, dtype=torch.int32, device=device)
        self.conv_offset = torch.empty_like(self.conv_batch)

    def build(self, common):
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

        n, tokens = common.num_reqs, common.num_actual_tokens
        if tokens > self.max_tokens:
            raise ValueError("GDN graph metadata exceeds the frozen capture bound")
        query_cpu = common.query_start_loc_cpu[:n + 1]
        lengths = query_cpu.diff().tolist()
        if len(lengths) != n or any(length < 0 for length in lengths):
            raise ValueError("Invalid GDN graph request boundaries")
        chunk_capacity = (tokens + 63) // 64 + n
        conv_capacity = (tokens + 7) // 8 + n
        chunks, conv_batch, conv_offset, offsets = [], [], [], [0]
        last_seq = last_chunks = 0
        for seq, length in enumerate(lengths):
            count = (length + 63) // 64
            chunks.extend((seq, chunk) for chunk in range(count))
            offsets.append(offsets[-1] + count)
            conv_count = (length + 7) // 8
            conv_batch.extend([seq] * conv_count)
            conv_offset.extend(range(conv_count))
            if length:
                last_seq, last_chunks = seq, count
        if len(chunks) > chunk_capacity or len(conv_batch) > conv_capacity:
            raise ValueError("GDN graph launch bound underestimated")
        chunks.extend([(last_seq, last_chunks)] * (chunk_capacity - len(chunks)))
        conv_batch.extend([-1] * (conv_capacity - len(conv_batch)))
        conv_offset.extend([0] * (conv_capacity - len(conv_offset)))
        def copy_cpu(target, values):
            target.copy_(torch.tensor(values, dtype=target.dtype, pin_memory=True), non_blocking=True)
            return target
        query = self.query[:n + 1]
        query.copy_(common.query_start_loc[:n + 1], non_blocking=True)
        states = self.states[:n]
        states.copy_(mamba_get_block_table_tensor(
            common.block_table_tensor, common.seq_lens, self.kv_cache_spec,
            self.config.cache_config.mamba_cache_mode,
        )[:n, 0], non_blocking=True)
        initial = self.initial[:n]
        initial.copy_(common.compute_num_computed_tokens()[:n] > 0, non_blocking=True)
        chunk_offsets = copy_cpu(self.chunk_offsets[:n + 1], offsets)
        chunk_indices = copy_cpu(self.chunks[:chunk_capacity], chunks)
        batch_ptr = copy_cpu(self.conv_batch[:conv_capacity], conv_batch)
        token_chunk_offset_ptr = copy_cpu(self.conv_offset[:conv_capacity], conv_offset)
        nums = {8: {"tot": conv_capacity, "mlist": None, "mlist_len": 0,
                    "offsetlist": None, "batch_ptr": batch_ptr,
                    "token_chunk_offset_ptr": token_chunk_offset_ptr}}
        # One generic FLA path handles single-token decodes, new prefills and
        # resumed chunked prefills. These host fields depend only on graph shape.
        return GDNAttentionMetadata(
            num_prefills=1, num_prefill_tokens=tokens, num_decodes=0,
            num_decode_tokens=0, num_spec_decodes=0, num_spec_decode_tokens=0,
            num_actual_tokens=tokens, has_initial_state=initial,
            non_spec_query_start_loc=query, non_spec_state_indices_tensor=states,
            prefill_query_start_loc=query, prefill_state_indices=states,
            prefill_has_initial_state=initial, chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets, nums_dict=nums, batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )

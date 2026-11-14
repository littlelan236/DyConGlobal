# SPDX-License-Identifier: Apache-2.0
"""Opt-in, pre-grammar target-only observation for DyCon TP=1 generation.

Only the current step's opted-in rows survive sampling. No probability tensor
with a vocabulary axis is constructed, and the observer never writes logits.
"""
import numpy as np
import torch

from vllm.v1.outputs import LogprobsTensors
from vllm.triton_utils import tl, triton

RAW_T1_KEY = "dycon_raw_t1_sampled_token_v1"


@triton.jit
def _partial_lse(logits, stride, maxima, sums, V: tl.constexpr,
                 PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    columns = part * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(logits + row * stride + columns, columns < V,
                     other=-float("inf")).to(tl.float32)
    maximum = tl.max(values, 0)
    total = tl.sum(tl.exp(values - maximum), 0)
    tl.store(maxima + row * PARTS + part, maximum)
    tl.store(sums + row * PARTS + part, total)


@triton.jit
def _finish_lse(logits, stride, maxima, sums, targets, result,
                PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    parts = tl.arange(0, BLOCK)
    maximums = tl.load(maxima + row * PARTS + parts, parts < PARTS,
                       other=-float("inf"))
    totals = tl.load(sums + row * PARTS + parts, parts < PARTS, other=0.)
    maximum = tl.max(maximums, 0)
    total = tl.sum(totals * tl.exp(maximums - maximum), 0)
    target = tl.load(targets + row).to(tl.int64)
    value = tl.load(logits + row * stride + target).to(tl.float32)
    tl.store(result + row, value - maximum - tl.log(total))


def compute_raw_target_logprobs(logits, targets):
    """Parallel FP32 vocabulary reduction with O(batch*vocab/4096) scratch."""
    rows, vocab = logits.shape
    parts = triton.cdiv(vocab, 4096)
    maxima = torch.empty((rows, parts), dtype=torch.float32, device=logits.device)
    sums = torch.empty_like(maxima)
    result = torch.empty((rows, 1), dtype=torch.float32, device=logits.device)
    _partial_lse[(rows, parts)](logits, logits.stride(0), maxima, sums,
                                V=vocab, PARTS=parts, BLOCK=4096)
    _finish_lse[(rows,)](logits, logits.stride(0), maxima, sums, targets, result,
                        PARTS=parts, BLOCK=triton.next_power_of_2(parts))
    return result


class RawT1Observer:
    def __init__(self, max_num_reqs):
        self.enabled = np.zeros(max_num_reqs, dtype=bool)

    def add_request(self, req_idx, params):
        enabled = (params.extra_args or {}).get(RAW_T1_KEY, False)
        if type(enabled) is not bool:
            raise ValueError("DyCon raw T=1 enablement must be boolean")
        if enabled and (params.logprobs != 0 or params.prompt_logprobs is not None
                        or params.logprob_token_ids or params.n != 1):
            raise ValueError("DyCon raw T=1 requires target-only logprobs=0, n=1")
        self.enabled[req_idx] = enabled

    def prepare(self, logits, batch):
        mask = self.enabled[batch.idx_mapping_np]
        if not np.any(mask):
            return None
        if batch.num_draft_tokens or logits.shape[0] != len(mask):
            raise ValueError("DyCon raw T=1 does not support speculative decoding")
        rows = torch.as_tensor(np.flatnonzero(mask), device=logits.device)
        # index_select owns the bytes even when logits are already FP32.
        return rows, logits.index_select(0, rows)

    def regular_logprobs_requested(self, sampler, mapping):
        regular = mapping[~self.enabled[mapping]]
        return bool(len(regular) and sampler.get_logprobs_dims(regular) is not None)

    @staticmethod
    def finish(state, sampled, existing=None):
        rows, raw = state
        targets = sampled.index_select(0, rows).view(-1, 1)
        values = compute_raw_target_logprobs(raw, targets)
        if existing is None:
            ids = sampled.to(torch.int32).view(-1, 1)
            scores = torch.full(ids.shape, -float("inf"), dtype=torch.float32,
                                device=sampled.device)
            ranks = torch.zeros(len(sampled), dtype=torch.int32, device=sampled.device)
        else:
            ids = existing.logprob_token_ids
            scores = existing.logprobs
            ranks = existing.selected_token_ranks
        scores[rows, 0] = values[:, 0]
        # Rank is not part of the raw target contract; do not reduce vocab again.
        ranks[rows] = 0
        return LogprobsTensors(ids, scores, ranks)

"""Observe the final sampler distribution without changing logits or RNG.

Installed only for key-step source runs, on the pinned native GPU sampler.
Only scalar entropies and sampled IDs survive each reduction. No vocabulary
vector is copied to the CPU or written to disk.
"""

import math

SEMANTICS = "sampler_distribution_entropy_nats_v1"


def entropy(logits, greedy=None):
    import torch

    logq = torch.log_softmax(logits.float(), dim=-1)
    # masked tokens contribute exactly zero, including a one-token grammar.
    h = -torch.where(torch.isfinite(logq), logq.exp() * logq, 0.0).sum(-1)
    if greedy is not None:
        h = torch.where(greedy, 0.0, h)
    return h


class SamplerEntropy:
    def __init__(self, sampler):
        if not hasattr(sampler, "req_states") or not hasattr(sampler, "sample"):
            raise ValueError("Entropy requires the pinned native GPU sampler")
        self.sampler = sampler
        self.pending = []
        self.values = {}
        self.finished = set()
        original = sampler.sample

        def sample(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            return_logprobs=False,
        ):
            if logits.shape[0] != len(idx_mapping_np):
                raise ValueError("Entropy v1 does not support speculative decoding")
            if not (
                sampler.sampling_states.any_explicit_seed(idx_mapping_np)
                or sampler.sampling_states.any_greedy(idx_mapping_np)
            ):
                raise ValueError("Entropy v1 requires native sampling without the fused FlashInfer path")
            sampled, processed = original(
                logits,
                expanded_idx_mapping,
                idx_mapping,
                idx_mapping_np,
                pos,
                input_ids,
                expanded_local_pos,
                return_logprobs=return_logprobs,
            )
            reverse = {v: k for k, v in sampler.req_states.req_id_to_index.items()}
            request_ids = [reverse.get(int(i)) for i in idx_mapping_np]
            greedy = sampler.sampling_states.temperature.gpu[idx_mapping] == 0
            valid = pos >= sampler.req_states.prefill_len.gpu[idx_mapping] - 1
            self.pending.append(
                (request_ids, entropy(processed, greedy).detach(), sampled.detach().clone(), valid.detach())
            )
            return sampled, processed

        sampler.sample = sample

    def take(self, request_id, token_ids):
        # Called on the serialized worker RPC lane after native terminal output.
        for names, entropies, tokens, valid in self.pending:
            for name, h, t, keep in zip(
                names, entropies.cpu().tolist(), tokens.cpu().tolist(), valid.cpu().tolist(), strict=True
            ):
                if keep and name is not None and name not in self.finished:
                    self.values.setdefault(name, []).append((int(t), float(h)))
        self.pending.clear()
        values = self.values.pop(request_id, [])
        self.finished.add(request_id)
        if len(self.finished) > 100000:
            self.finished.clear()
        observed = [t for t, h in values]
        # vLLM may hide trailing stop tokens. Retained IDs must match exactly.
        if observed[: len(token_ids)] != token_ids or len(observed) < len(token_ids):
            raise ValueError(
                f"Entropy/native output token alignment mismatch: request={request_id!r} observed={len(observed)} expected={len(token_ids)} first={observed[:4]} target={token_ids[:4]} buffered={list(self.values)[:8]}"
            )
        selected = [h for t, h in values[: len(token_ids)]]
        if not selected or any(not math.isfinite(h) or h < -1e-5 for h in selected):
            raise ValueError("Empty or invalid generation entropy")
        total = math.fsum(max(0.0, h) for h in selected)
        return {
            "semantics": SEMANTICS,
            "unit": "nats",
            "entropy_sum": total,
            "entropy_count": len(selected),
            "entropy_mean": total / len(selected),
        }

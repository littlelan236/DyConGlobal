"""Explicit HF hidden-state convention: decoder outputs, then final norm.

Each anchor is independently replayed. No cache is cropped or reused across
requests; this also supports hybrid attention/recurrent caches safely.
"""

from __future__ import annotations


class HFDecoderAdapter:
    def __init__(self, model, fingerprint):
        self.model = model.eval()
        self.fingerprint = fingerprint
        cfg = getattr(model.config, "text_config", model.config)
        self.layer_count = cfg.num_hidden_layers
        self.hidden_size = cfg.hidden_size

    def sites(self):
        return [{"kind": "decoder_output", "block_index": i} for i in range(self.layer_count - 1)] + [
            {"kind": "final_norm", "block_index": None}
        ]

    def capture(self, ids, layers, chunk_size):
        import torch

        if not ids or not layers or any(i < 0 or i >= self.layer_count for i in layers):
            raise ValueError("Nonempty prefix and valid configured layer/site indices are required")
        device = next(self.model.parameters()).device
        cache = None
        with torch.inference_mode():
            # Only request hidden states at the final marker token to bound memory.
            for start in range(0, len(ids) - 1, chunk_size):
                end = min(start + chunk_size, len(ids) - 1)
                result = self.model(
                    input_ids=torch.tensor([ids[start:end]], device=device),
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=False,
                )
                cache = result.past_key_values
                del result
            result = self.model(
                input_ids=torch.tensor([ids[-1:]], device=device),
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
            hidden = result.hidden_states
            if hidden is None or len(hidden) != self.layer_count + 1:
                raise ValueError("Model does not expose the audited HF hidden-state convention")
            values = torch.stack([hidden[i + 1][0, -1].float().cpu() for i in layers]).numpy()
        return values


def load_adapter(cfg):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    from core.model_identity import verify_model

    verify_model(cfg)
    m = cfg["model"]
    config = AutoConfig.from_pretrained(
        m["path"], local_files_only=True, trust_remote_code=m["trust_remote_code"]
    )
    cls = (
        AutoModelForImageTextToText
        if "ConditionalGeneration" in " ".join(config.architectures or [])
        else AutoModelForCausalLM
    )
    model = cls.from_pretrained(
        m["path"],
        local_files_only=True,
        trust_remote_code=m["trust_remote_code"],
        dtype=getattr(torch, cfg["hidden"]["dtype"]),
        device_map=cfg["hidden"]["device"],
    )
    # Use text decoder directly for conditional wrappers, as the old audited extractor did.
    candidates = [model, getattr(model, "model", None)]
    candidates += [getattr(x, "language_model", None) for x in list(candidates) if x is not None]
    candidates += [getattr(x, "model", None) for x in list(candidates) if x is not None]
    decoder = next(
        (x for x in candidates if x is not None and hasattr(x, "layers") and hasattr(x, "embed_tokens")),
        model,
    )
    return HFDecoderAdapter(decoder, m["fingerprint"])

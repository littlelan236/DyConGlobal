"""Functional cache boundary for the pinned Qwen3.5 text decoder.

Reuse native decoder/rotary/mask/convolution/GDN operators. Each checkpoint
receives its layer's state as explicit inputs and builds a fresh cache facade;
no mutable cache survives into recomputation. No model parameters are replaced.
"""
from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from RL.memory import text_backbone


class _LayerCache:
    """Only the cache protocol consumed by Qwen3_5DecoderLayer.forward.

    record_past selects native functional convolution, including Q=1. Published
    convolution state is still bounded to kernel_size, with no prefix alias.
    """
    def __init__(self, index, states):
        self.index, self.states = index, states
        self.layers = {index: SimpleNamespace(record_past=True,
            conv_states={0: states[0] if states else None},
            recurrent_states={0: states[1] if states else None})}
        self.outputs = ()

    def has_previous_state(self, layer_idx, state_idx=0):
        assert layer_idx == self.index and state_idx == 0
        return bool(self.states)

    def update(self, keys, values, layer_idx):
        assert layer_idx == self.index
        if self.states:
            keys = torch.cat((self.states[0], keys), dim=-2)
            values = torch.cat((self.states[1], values), dim=-2)
        self.outputs = (keys, values)
        return keys, values

    def update_conv_state(self, values, layer_idx, conv_kernel_size):
        assert layer_idx == self.index
        full = torch.cat((self.states[0], values), dim=-1) if self.states else values
        tail = full[..., -conv_kernel_size:]
        if tail.shape[-1] < conv_kernel_size:
            tail = F.pad(tail, (conv_kernel_size - tail.shape[-1], 0))
        self.outputs = (tail.clone(),)
        return full

    def update_recurrent_state(self, value, layer_idx):
        assert layer_idx == self.index
        self.outputs += (value,)
        return value


def validate_model(model, expected_backend=None):
    import transformers
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    text = text_backbone(model)
    if type(text) is not Qwen3_5TextModel or transformers.__version__ != "5.17.0":
        raise ValueError("boundary_grad requires pinned Transformers 5.17.0 Qwen3_5TextModel")
    if text.config.attention_dropout != 0 or any(
            isinstance(m, torch.nn.Dropout) and m.p != 0 for m in text.modules()):
        raise ValueError("boundary_grad requires all text/LoRA dropout=0")
    if next(text.parameters()).dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("boundary_grad supports FP32/BF16 only")
    for layer in text.layers:
        if layer.block_type not in {"full_attention", "linear_attention"}:
            raise ValueError("Unsupported Qwen boundary layer")
    if getattr(text.config, 'use_kernels', False):
        raise ValueError("boundary_grad optional GDN/conv kernels have not been validated")
    from RL.gdn_backend import active_binding
    binding = active_binding()
    if binding is None:
        raise ValueError("boundary_grad requires an installed GDN backend")
    if expected_backend is not None and binding.execution_contract() != expected_backend:
        raise ValueError("Model boundary GDN backend differs from the execution contract")
    return text


def chunked_causal_sdpa(query, key, value, *, scale, query_tokens):
    """Keep each query block's complete causal key/value prefix in the graph."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.bias import causal_lower_right

    outputs = []
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for start in range(0, query.shape[-2], query_tokens):
            end = min(start + query_tokens, query.shape[-2])
            outputs.append(F.scaled_dot_product_attention(
                query[..., start:end, :], key[..., :end, :], value[..., :end, :],
                attn_mask=causal_lower_right(end - start, end), scale=scale,
                enable_gqa=query.shape[1] != key.shape[1]))
    return torch.cat(outputs, dim=-2)


@contextmanager
def retry_chunked_attention(query_tokens=32768):
    """Reduce the deterministic Flash backward workspace on a replay."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    # Install a Qwen-only registry for the replay, leaving the shared
    # Transformers registry and unrelated attention interfaces untouched.
    original = qwen.ALL_ATTENTION_FUNCTIONS
    retry = type(original)()
    retry.update(original)
    sdpa = original["sdpa"]

    def attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        if (query.is_cuda and attention_mask is None and not dropout
                and kwargs.get("is_causal", getattr(module, "is_causal", True))
                and kwargs.get("position_bias") is None
                and query.shape[-2] == key.shape[-2] > query_tokens):
            output = chunked_causal_sdpa(query, key, value, scale=scaling, query_tokens=query_tokens)
            return output.transpose(1, 2).contiguous(), None
        return sdpa(module, query, key, value, attention_mask,
                    dropout=dropout, scaling=scaling, **kwargs)

    retry["sdpa"] = attention
    qwen.ALL_ATTENTION_FUNCTIONS = retry
    try:
        yield
    finally:
        qwen.ALL_ATTENTION_FUNCTIONS = original


def segment(model, input_ids, states=(), prefix_length=0, metrics=None, offloader=None):
    """Return final hidden and stable layer-major (K,V)/(conv,recurrent) roots."""
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    text = text_backbone(model)
    hidden = text.embed_tokens(input_ids)
    positions = torch.arange(prefix_length, prefix_length + hidden.shape[1],
                             device=hidden.device).view(1, -1)
    rotary = text.rotary_emb(hidden, positions[None].expand(3, -1, -1))
    # Native mask builder accounts for absolute offsets and SDPA/Flash dispatch.
    mask_cache = None
    if states:
        mask_cache = DynamicCache(config=text.config)
        for index, layer in enumerate(text.layers):
            if layer.block_type == "full_attention":
                mask_cache.layers[index].keys, mask_cache.layers[index].values = states[2*index:2*index+2]
                mask_cache.layers[index].is_initialized = True
    if states and text.config._attn_implementation == "sdpa" and hidden.shape[1] > 1:
        # Native SDPA's ordinary cached mask is dense for Q<K, which prevents
        # Flash-only dispatch. PyTorch's symbolic lower-right causal bias has
        # exactly the absolute-position semantics and dispatches to native Flash.
        from torch.nn.attention.bias import causal_lower_right
        mask = causal_lower_right(hidden.shape[1], prefix_length + hidden.shape[1])
    else:
        mask = qwen.create_causal_mask(config=text.config, inputs_embeds=hidden,
            attention_mask=None, past_key_values=mask_cache, position_ids=positions)
    outputs = []
    for index, layer in enumerate(text.layers):
        incoming = tuple(states[2*index:2*index+2]) if states else ()
        # Bind layer/index now: checkpoint runs again after the outer loop ends.
        def run(value, *boundary, layer=layer, index=index):
            if metrics is not None:
                key = "suffix_layer_forward_calls" if states else "prefix_layer_forward_calls"
                metrics[key] = metrics.get(key, 0) + 1
            cache = _LayerCache(index, boundary)
            value = layer.forward(value, position_embeddings=rotary,
                attention_mask=mask if layer.block_type == "full_attention" else None,
                position_ids=positions, past_key_values=cache, use_cache=True)
            if len(cache.outputs) != 2:
                raise RuntimeError("Incomplete Qwen boundary outputs")
            if metrics is not None and states:
                size = sum(t.numel() * t.element_size() for t in cache.outputs)
                metrics["suffix_layer_cache_output_peak_bytes"] = max(
                    size, metrics.get("suffix_layer_cache_output_peak_bytes", 0))
            return (value, *cache.outputs) if not states else (value,)
        if layer.gradient_checkpointing and model.training and torch.is_grad_enabled():
            result = (offloader.checkpoint(run, hidden, *incoming)
                      if offloader is not None else checkpoint(run, hidden, *incoming, use_reentrant=False))
        else:
            result = run(hidden, *incoming)
        hidden = result[0]
        if not states:
            outputs.extend(result[1:])
    return text.norm(hidden), tuple(outputs)

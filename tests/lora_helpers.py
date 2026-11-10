"""Tiny real Qwen3.5+PEFT fixtures. No downloads or full random 9B model."""

import copy

import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from RL.config import Lora, Training
from RL.modeling import inject
from RL.training import seed_everything

LORA = Lora(rank=2, alpha=4, dropout=0.0, target_scope="text_backbone_all_linear")


def raw_model(layers=2):
    seed_everything(123)
    config = Qwen3_5Config(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=layers,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                         linear_num_key_heads=1, linear_num_value_heads=2,
                         linear_key_head_dim=16, linear_value_head_dim=16,
                         layer_types=["linear_attention", "full_attention"] * (layers // 2),
                         max_position_embeddings=128, pad_token_id=0, eos_token_id=1,
                         rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                          "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=48, num_heads=2, out_hidden_size=32))
    config.use_cache = False
    config.text_config.use_cache = False
    model = Qwen3_5ForConditionalGeneration(config)
    return model


def model_and_targets(settings=LORA, adapter=None, expected_targets=None):
    return inject(raw_model(), settings, adapter=adapter, expected_targets=expected_targets)


def settings(max_updates=4):
    return Training(max_updates=max_updates, gradient_accumulation_steps=2, learning_rate=0.003,
                    weight_decay=0.0, grad_clip=1.0, scheduler="linear", checkpoint_every_updates=2)


def batches():
    rows = []
    for index in range(3):
        ids = torch.tensor([[2, 3, 4 + index, 5, 6, 7, 8, 9]])
        mask = torch.ones_like(ids)
        labels = ids.clone()
        labels[:, :2 + index] = -100
        if index:
            mask[:, -index:] = 0
            labels[:, -index:] = -100
        rows.append(dict(input_ids=ids, attention_mask=mask, labels=labels))
    return rows


def stream(count=8, offset=0):
    rows = batches()
    return [copy.deepcopy(rows[i % len(rows)]) for i in range(offset, offset + count)]

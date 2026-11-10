import copy

import pytest
import torch
import torch.nn.functional as F

from RL.modeling import Coverage, adapter_fingerprint, enable_checkpointing, inject, text_targets
from RL.training import optimizer_and_scheduler, train
from RL.validation import causal_ce, validate_batch
from tests.lora_helpers import LORA, batches, model_and_targets, raw_model, settings


def kwargs(model, max_updates=4, accumulation=2):
    return dict(max_updates=max_updates, gradient_accumulation_steps=accumulation,
                grad_clip=1.0, vocab_size=64, max_length=32)


def test_exact_coverage_gradients_and_frozen_base():
    raw = raw_model()
    originals = [(p, p.detach().clone()) for p in raw.parameters()]
    expected = text_targets(raw)
    assert {t["category"] for t in expected} == {"mlp", "linear_attention", "full_attention"}
    assert all("lm_head" not in t["path"] and "visual" not in t["path"] for t in expected)
    model, targets = inject(raw, LORA)
    assert targets == expected
    enable_checkpointing(model)
    before = adapter_fingerprint(model)
    coverage = Coverage(model, targets)
    optimizer, scheduler = optimizer_and_scheduler(model, settings(2))
    state = train(model, batches(), causal_ce, optimizer, scheduler, **kwargs(model, 2),
                  on_gradients=coverage.gradients)
    report = coverage.finish(before, adapter_fingerprint(model))
    coverage.close()
    assert state["global_step"] == 2
    assert state["consumed_microbatches"] == 3  # Final partial accumulation window commits.
    assert all(row["executed"] and row["nonzero_B_gradient"] and row["adapter_changed"] for row in report)
    for parameter, original in originals:
        torch.testing.assert_close(parameter, original, atol=0, rtol=0)
    assert all("lora_" in name for name, p in model.named_parameters() if p.requires_grad)


def test_token_weighted_accumulation_matches_large_batch_with_partial_window():
    model, _ = model_and_targets()
    other, _ = model_and_targets()
    first, first_schedule = optimizer_and_scheduler(model, settings(2))
    second, second_schedule = optimizer_and_scheduler(other, settings(2))
    rows = batches()
    train(model, rows, causal_ce, first, first_schedule, **kwargs(model, 2))
    merged = {key: torch.cat([rows[0][key], rows[1][key]], dim=0) for key in rows[0]}
    train(other, [merged, rows[2]], causal_ce, second, second_schedule, **kwargs(other, 2, 1))
    for (name, p), (other_name, q) in zip(model.named_parameters(), other.named_parameters(), strict=True):
        assert name == other_name
        torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-5)


def test_causal_shift_and_ignored_targets():
    class Fixed(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.logits = logits

        def forward(self, **_):
            from types import SimpleNamespace
            return SimpleNamespace(logits=self.logits)

    logits = torch.tensor([[[2., 0., -1.], [0., 3., -1.], [0., 0., 2.], [1., 2., 0.]]])
    row = dict(input_ids=torch.tensor([[0, 1, 2, 0]]), attention_mask=torch.ones((1, 4), dtype=torch.long),
               labels=torch.tensor([[-100, 1, -100, 0]]))
    loss, count = causal_ce(Fixed(logits), row)
    manual = F.cross_entropy(torch.stack([logits[0, 0], logits[0, 2]]), torch.tensor([1, 0]), reduction="sum")
    torch.testing.assert_close(loss, manual)
    assert count == 2
    changed_logits = logits.clone()
    changed_logits[:, 1] += torch.tensor([20., -20., 0.])  # Its next target is ignored.
    torch.testing.assert_close(causal_ce(Fixed(changed_logits), row)[0], loss)
    changed_labels = copy.deepcopy(row)
    changed_labels["labels"][:, 1] = 0
    assert causal_ce(Fixed(logits), changed_labels)[0] != loss


@pytest.mark.parametrize("defect", ["zero", "overflow", "shape", "padding", "length", "mask"])
def test_invalid_batches(defect):
    batch = batches()[0]
    if defect == "zero":
        batch["labels"].fill_(-100)
    elif defect == "overflow":
        batch["input_ids"][0, 0] = 64
    elif defect == "shape":
        batch["labels"] = batch["labels"][:, :-1]
    elif defect == "padding":
        batch["attention_mask"][0, -1] = 0
    elif defect == "mask":
        batch["attention_mask"][0, -1] = 2
    with pytest.raises(ValueError):
        validate_batch(batch, vocab_size=64, max_length=4 if defect == "length" else 32)


@pytest.mark.parametrize("defect", ["loss", "gradient", "optimizer"])
def test_bad_updates_rejected(defect):
    model, _ = model_and_targets()
    optimizer, scheduler = optimizer_and_scheduler(model, settings(1))
    before = adapter_fingerprint(model)
    if defect == "optimizer":
        optimizer.param_groups[0]["params"].append(next(p for p in model.parameters() if not p.requires_grad))
    elif defect == "gradient":
        next(p for p in model.parameters() if p.requires_grad).register_hook(lambda grad: grad * float("nan"))

    def loss(m, b):
        value, count = causal_ce(m, b)
        return (value * float("nan") if defect == "loss" else value), count

    with pytest.raises((ValueError, RuntimeError)):
        train(model, batches()[:1], loss, optimizer, scheduler, **kwargs(model, 1, 1))
    assert before == adapter_fingerprint(model)

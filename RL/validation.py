"""Engineering-only fixed text fixture and causal CE; no business dataset or RL loss."""

from pathlib import Path

import torch
import torch.nn.functional as F

from core.io import hash_json, read_json, sha256

FIXTURE = Path(__file__).parent / "fixtures/causal-text-smoke-v1.json"


def validate_batch(batch, *, vocab_size, max_length):
    if set(batch) != {"input_ids", "attention_mask", "labels"}:
        raise ValueError("Batch must contain input_ids, attention_mask and labels")
    ids, mask, labels = (batch[key] for key in ("input_ids", "attention_mask", "labels"))
    if ids.ndim != 2 or ids.shape != mask.shape or ids.shape != labels.shape or not ids.numel():
        raise ValueError("Inconsistent/nonempty batch shapes are required")
    if ids.dtype != torch.long or labels.dtype != torch.long:
        raise ValueError("Token IDs and labels must be int64")
    if ids.shape[1] > max_length or ids.shape[1] < 2:
        raise ValueError("Batch sequence length exceeds the limit or has no shifted position")
    if ((ids < 0) | (ids >= vocab_size)).any() or ((labels != -100) & ((labels < 0) | (labels >= vocab_size))).any():
        raise ValueError("Token IDs or labels outside vocabulary")
    if not ((mask == 0) | (mask == 1)).all() or ((mask == 0) & (labels != -100)).any():
        raise ValueError("Invalid attention mask or supervised padding")
    count = int((labels[:, 1:] != -100).sum())
    if count == 0:
        raise ValueError("Batch contains zero supervised causal targets")
    return count


def causal_ce(model, batch):
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
    labels = batch["labels"][:, 1:].contiguous()
    # The model receives no labels; causal shift happens exactly once here.
    loss_sum = F.cross_entropy(logits[:, :-1].contiguous().float().reshape(-1, logits.shape[-1]),
                               labels.reshape(-1), ignore_index=-100, reduction="sum")
    return loss_sum, int((labels != -100).sum())


def fixture_batches(tokenizer, max_length):
    fixture = read_json(FIXTURE)
    batches = []
    records = []
    for index, row in enumerate(fixture["rows"]):
        prompt = tokenizer.encode(row["prompt"], add_special_tokens=False)
        completion = tokenizer.encode(row["completion"], add_special_tokens=False)
        if not prompt or not completion:
            raise ValueError("Fixture encoded to empty prompt/completion")
        # Deterministic repetition explicitly constructs a fixture at the requested length.
        # This is not truncation of user inputs and is not a persisted training dataset.
        remaining = max_length - len(prompt)
        if remaining < 2:
            raise ValueError("Configured length is too short for the fixed prompt")
        tokens = (completion * (remaining // len(completion) + 1))[:remaining]
        ids = prompt + tokens
        labels = [-100] * len(prompt) + tokens
        # Variable supervision and padding lengths exercise token-weighted accumulation.
        padding = min(index * 7, len(tokens) - 1)
        if padding:
            labels[-padding:] = [-100] * padding
            ids[-padding:] = [tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                              else tokenizer.eos_token_id] * padding
        mask = [1] * (len(ids) - padding) + [0] * padding
        batch = {key: torch.tensor([value], dtype=torch.long) for key, value in
                 (("input_ids", ids), ("attention_mask", mask), ("labels", labels))}
        batches.append(batch)
        records.append({key: value.tolist() for key, value in batch.items()})
    return batches, {"fixture_id": fixture["fixture_id"], "fixture_sha256": sha256(FIXTURE),
                     "encoded_batches_sha256": hash_json(records), "batch_count": len(batches),
                     "max_sequence_length": max_length, "evidence_class": "engineering_generated_tokens"}


def probe_logits(model, batch, length=24):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(input_ids=batch["input_ids"][:, :length].to(device),
                       attention_mask=batch["attention_mask"][:, :length].to(device), use_cache=False).logits
    model.train(was_training)
    return logits.detach().float().cpu()

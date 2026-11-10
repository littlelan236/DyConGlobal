"""Canonical raw-target supervision. Importing policy descriptions needs no Torch.

Selection never changes attention or input positions. Prepared labels are private
copies, and the optimizer loop consumes each window's selection exactly once.
"""

from dataclasses import dataclass

from core.io import hash_json
from RL.config import Supervision

ALGORITHMS = {"labels": "causal_labels_v1", "explicit_mask": "raw_target_mask_v1",
              "uniform": "uniform_valid_targets_v1"}
FIXTURE_MASK = "fixture_row_uniform_endpoints_v1"
MODEL_BATCH_KEYS = ("input_ids", "attention_mask", "labels")


def policy_identity(settings=None):
    settings = settings or Supervision()
    return {**settings.model_dump(), "algorithm": ALGORITHMS[settings.mode],
            "coordinate_system": "raw_target_token_v1"}


def describe_policy(settings):
    return {**policy_identity(settings),
            "mask_source": FIXTURE_MASK if settings.mode == "explicit_mask" else None,
            "selection_timing": "before_vocabulary_projection",
            "selected_tokens": None, "selection_sha256": None,
            "note": "Actual counts and content identities require encoded batches; dry-run does not load a tokenizer."}


def uniform_indices(eligible, limit):
    if len(eligible) <= limit:
        return eligible
    if limit == 1:
        return eligible[-1:]
    return [eligible[i * (len(eligible) - 1) // (limit - 1)] for i in range(limit)]


@dataclass(frozen=True)
class Selection:
    batch: dict
    coordinates: tuple
    available_count: int
    selected_count: int
    input_tokens: int
    identity: dict
    sha256: str

    def model_batch(self):
        # Neither loss_mask nor selection metadata belongs in Transformers forward.
        return {key: self.batch[key] for key in MODEL_BATCH_KEYS}


def prepare_batch(batch, *, vocab_size, max_length, supervision=None):
    import torch

    settings = supervision or Supervision()
    keys = set(batch)
    if not set(MODEL_BATCH_KEYS).issubset(keys) or keys - {*MODEL_BATCH_KEYS, "loss_mask"}:
        raise ValueError("Batch must contain input_ids, attention_mask and labels, with optional loss_mask only")
    ids, attention, labels = (batch[key] for key in MODEL_BATCH_KEYS)
    if not all(isinstance(value, torch.Tensor) for value in (ids, attention, labels)):
        raise ValueError("Batch fields must be tensors")
    if ids.ndim != 2 or ids.shape != attention.shape or ids.shape != labels.shape or not ids.numel():
        raise ValueError("Inconsistent/nonempty batch shapes are required")
    if any(value.device != ids.device for value in (attention, labels)):
        raise ValueError("Batch tensors must share a device")
    if ids.dtype != torch.long or labels.dtype != torch.long:
        raise ValueError("Token IDs and labels must be int64")
    if ids.shape[1] > max_length or ids.shape[1] < 2:
        raise ValueError("Batch sequence length exceeds the limit or has no shifted position")
    if ((ids < 0) | (ids >= vocab_size)).any() or ((labels != -100) & ((labels < 0) | (labels >= vocab_size))).any():
        raise ValueError("Token IDs or labels outside vocabulary")
    if not ((attention == 0) | (attention == 1)).all() or ((attention == 0) & (labels != -100)).any():
        raise ValueError("Invalid attention mask or supervised padding")
    eligible = labels != -100
    eligible = eligible.clone()
    eligible[:, 0] = False
    # A target requires both its own input slot and its preceding predictor slot.
    accessible = attention.bool().clone()
    accessible[:, 0] = False
    accessible[:, 1:] &= attention[:, :-1].bool()
    if (eligible & ~accessible).any():
        raise ValueError("Causal target has an inaccessible predecessor or padding")
    external = batch.get("loss_mask")
    if settings.mode == "explicit_mask":
        if external is None:
            raise ValueError("explicit_mask supervision requires loss_mask in every batch")
        if (not isinstance(external, torch.Tensor) or external.dtype != torch.bool
                or external.shape != labels.shape or external.device != ids.device):
            raise ValueError("loss_mask must be a bool tensor with exactly the label shape and device")
        if (external & ~eligible).any():
            raise ValueError("loss_mask selects position zero, ignored labels, padding or inaccessible target")
        selected = external.clone()
    else:
        if "loss_mask" in batch:
            raise ValueError("loss_mask conflicts with labels/uniform supervision")
        selected = eligible.clone()
        if settings.mode == "uniform":
            selected.zero_()
            for index, row in enumerate(eligible):
                positions = row.nonzero(as_tuple=True)[0].tolist()
                selected[index, uniform_indices(positions, settings.max_targets_per_sample)] = True
    if not selected.any(dim=1).all():
        raise ValueError("Every sample must have nonzero selected supervised causal targets")
    effective = labels.clone()
    effective[~selected] = -100
    prepared = dict(batch, labels=effective)
    if external is not None:
        prepared["loss_mask"] = selected
    coordinates = tuple(tuple(pair) for pair in selected.nonzero(as_tuple=False).tolist())
    content = {key: value.detach().cpu().tolist() for key, value in batch.items()}
    identity = {"policy": policy_identity(settings), "input_sha256": hash_json(content),
                "shape": list(ids.shape), "effective_lengths": attention.sum(dim=1).tolist(),
                "target_coordinates": [list(pair) for pair in coordinates],
                "available_supervised_tokens": int(eligible.sum()),
                "supervised_tokens": len(coordinates)}
    return Selection(prepared, coordinates, int(eligible.sum()), len(coordinates),
                     int(attention.sum()), identity, hash_json(identity))


def fixture_loss_mask(batch, row_index):
    """The synthetic input provider owns this versioned mask, in raw coordinates."""
    import torch

    mask = torch.zeros_like(batch["labels"], dtype=torch.bool)
    for index, row in enumerate(batch["labels"]):
        positions = (row[1:] != -100).nonzero(as_tuple=True)[0].add(1).tolist()
        mask[index, uniform_indices(positions, row_index + 1)] = True
    return mask


def input_selection_identity(selections, settings):
    return {"policy": policy_identity(settings),
            "selection_sha256": hash_json([selection.sha256 for selection in selections]),
            "batches": [{**selection.identity, "selection_sha256": selection.sha256} for selection in selections]}


def consumed_selected_tokens(batches, consumed_microbatches, *, vocab_size, max_length, supervision):
    counts = [prepare_batch(batch, vocab_size=vocab_size, max_length=max_length,
                            supervision=supervision).selected_count for batch in batches]
    cycles, remainder = divmod(consumed_microbatches, len(counts))
    return cycles * sum(counts) + sum(counts[:remainder])

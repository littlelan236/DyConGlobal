"""Validation helpers for the versioned offline objective."""

from RL.config import Grpo

OBJECTIVE_VERSION = "offline_grpo_k3_v1"


def validate_objective_config(value):
    """Validate a mapping without importing torch or loading a model."""
    return Grpo.model_validate(value)


def objective_contract(settings, *, dataset_id, dataset_receipt_sha256,
                       model_fingerprint, tokenizer_fingerprint):
    settings = settings if hasattr(settings, "model_dump") else Grpo.model_validate(settings)
    return {
        "objective_version": OBJECTIVE_VERSION,
        "dataset_id": dataset_id,
        "dataset_receipt_sha256": dataset_receipt_sha256,
        "model_fingerprint": model_fingerprint,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "grpo": settings.model_dump(),
    }

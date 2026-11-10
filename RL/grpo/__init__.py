"""Offline GRPO objective components.

The package is deliberately independent of rollout generation.  It consumes
frozen native groups and their historical estimator scores.
"""

from .advantage import AdvantageResult, compute_advantages
from .groups import GroupBatch, iter_group_batches

__all__ = ["AdvantageResult", "compute_advantages", "GroupBatch", "iter_group_batches"]

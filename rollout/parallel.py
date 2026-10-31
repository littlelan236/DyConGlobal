"""Backward-compatible mini adapter for the shared decision coordinator."""
from rollout.adapters.minisweagent import MiniPolicy
from rollout.decision.coordinator import (
    BranchCandidate, DecisionCoordinator, ParallelSelectionError, RANDOM_SELECTION_MODE,
    ServingParallelClient, active, baseline_selection_seed, random_branch_id,
)


class ParallelReasoningCoordinator(DecisionCoordinator):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("error_policy", MiniPolicy())
        super().__init__(*args, **kwargs)

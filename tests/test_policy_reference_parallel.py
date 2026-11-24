"""Four-rank updater agreement with forced frozen KL recomputation."""
from pathlib import Path


def _worker(rank, store, output):
    import torch
    import RL.online_runtime as runtime
    original = runtime.TorchOnlineUpdater
    class RecomputingUpdater(original):
        def __init__(self, *args, **kwargs):
            kwargs["kl_reference_mode"] = "recompute"
            kwargs["reference_scorer"] = lambda tensors: torch.full_like(
                tensors["input_ids"][:, 1:], -2.2, dtype=torch.float)
            super().__init__(*args, **kwargs)
    runtime.TorchOnlineUpdater = RecomputingUpdater
    from tests.test_online_parallel import _run
    _run(rank, store, output, False)


def test_four_rank_recompute_matches_dense_and_empty_ranks(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_worker, args=(str(tmp_path/"store"), str(tmp_path/"result")), nprocs=4, join=True)
    assert (tmp_path/"result").exists()

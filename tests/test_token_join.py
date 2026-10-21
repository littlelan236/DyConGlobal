from application.execution.evidence import RunEvidence
import numpy as np

from difficulty.datasets.token_join import _candidate_indices


def test_token_join_requires_exact_position_and_kind():
    arrays={
        "special_kinds": np.array([1,2,2,3]),
        "step_ids": np.array([0,0,0,0]),
        "absolute_positions": np.array([2,5,7,9]),
        "token_ids": np.array([248068,271,198,248069]),
    }
    meta={"rollout_id":"r"}
    assert _candidate_indices(meta, arrays, {"rollout_id":"r","anchor":"reasoning_newline","step_id":0,"anchor_position":5,"anchor_token_id":271}) == [1]
    assert _candidate_indices(meta, arrays, {"rollout_id":"r","anchor":"reasoning_newline","step_id":0,"anchor_position":6,"anchor_token_id":271}) == []

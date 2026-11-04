import pytest

from hidden.publish_finalized import _check_finalization


def test_partial_branch_capture_requires_explicit_canonical_scope():
    completion = {'rollout_id': 'r1', 'capture_status': 'partial'}
    coverage = {'expected': 26, 'observed': 26, 'missing': []}
    with pytest.raises(ValueError):
        _check_finalization(completion, 'r1', coverage, 'all_branches')
    _check_finalization(completion, 'r1', coverage, 'canonical_native')


@pytest.mark.parametrize('coverage', [
    {'expected': 26, 'observed': 25, 'missing': ['request']},
    {'expected': 26, 'observed': 25, 'missing': []},
    {'expected': 0, 'observed': 0, 'missing': []},
    {'expected': 26, 'observed': 26},
])
def test_canonical_scope_rejects_incomplete_native_coverage(coverage):
    with pytest.raises(ValueError):
        _check_finalization({'rollout_id': 'r1', 'capture_status': 'partial'}, 'r1', coverage, 'canonical_native')


def test_canonical_scope_rejects_rollout_mismatch_and_failed_capture():
    coverage = {'expected': 26, 'observed': 26, 'missing': []}
    with pytest.raises(ValueError):
        _check_finalization({'rollout_id': 'other', 'capture_status': 'complete'}, 'r1', coverage, 'canonical_native')
    with pytest.raises(ValueError):
        _check_finalization({'rollout_id': 'r1', 'capture_status': 'failed'}, 'r1', coverage, 'canonical_native')

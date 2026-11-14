import pytest

from RL.grpo.advantage import compute_advantages


def test_top_k_advantages_are_deterministic_and_zero_sum():
    result = compute_advantages([0.1, 0.2, 0.8, 1.0], top_k=2)
    assert result.order == [0, 1, 2, 3]
    assert result.positive_count == 2
    assert result.negative_count == 2
    assert sum(result.advantages) == pytest.approx(0.0)
    assert result.advantages[0] > result.advantages[1] > 0
    assert result.advantages[2] == result.advantages[3] == -1


def test_boundary_tie_skips_group():
    result = compute_advantages([0.0, 1.0, 1.0], top_k=2, boundary_tie_sigma=1.0)
    assert result.metadata["status"] == "skipped"
    assert result.advantages == []


def test_top_k_policies_are_mutually_exclusive():
    with pytest.raises(ValueError):
        compute_advantages([1, 2], top_k=1, top_k_fraction=.5)


@pytest.mark.parametrize("n,k", [(2,1), (3,1), (9,2), (17,3), (32,4)])
def test_default_scaled_top_k_zero_sum_and_mean(n,k):
    result = compute_advantages(list(range(n)))
    assert result.top_k == k and result.positive_count == k
    assert sum(result.advantages) == pytest.approx(0., abs=1e-12)
    assert sum(x for x in result.advantages if x>0)/k == pytest.approx(1.)
    assert all(x<0 for x in result.advantages[k:])
    assert result == compute_advantages(list(range(n)))


@pytest.mark.parametrize("scores,reason", [([1.,1.],"all_scores_equal"), ([1.,None],"missing_or_nonfinite_score"),
    ([1.,float('nan')],"missing_or_nonfinite_score"), ([1.,float('inf')],"missing_or_nonfinite_score"),
    ([0.,.001,1.],"top_k_boundary_tie")])
def test_invalid_or_ambiguous_scores_skip(scores,reason):
    result = compute_advantages(scores)
    assert result.metadata["skip_reason"] == reason
    assert result.advantages == []


def test_tied_positive_scores_equal_and_fraction_rounds_up():
    result = compute_advantages([0.,0.,.5,1.,2.], top_k=2)
    assert result.advantages[:2] == [1.,1.]
    fractional = compute_advantages(list(range(7)), top_k=None, top_k_fraction=.3)
    assert fractional.top_k == 3


@pytest.mark.parametrize("k", [0,-1,1.5,3])
def test_illegal_k_rejected(k):
    with pytest.raises(ValueError):
        compute_advantages([0.,1.,2.], top_k=k)


@pytest.mark.parametrize("scores", [[0., 1., 1., 2.], [0., 1., 1.001, 2.]])
def test_zero_tie_tolerance_disables_boundary_exclusion(scores):
    assert compute_advantages(scores, top_k=2).metadata["status"] == "skipped"
    result = compute_advantages(scores, top_k=2, boundary_tie_sigma=0.)
    assert result.metadata["status"] == "ok"
    assert result.order == [0, 1, 2, 3]
    assert result.advantages[0] > result.advantages[1] > 0
    assert result.advantages[2:] == [-1., -1.]
    assert sum(result.advantages) == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize("scores,reason", [
    ([1., 1., 1., 1.], "all_scores_equal"),
    ([0., None, 1., 2.], "missing_or_nonfinite_score"),
    ([0., float("nan"), 1., 2.], "missing_or_nonfinite_score"),
])
def test_zero_tie_tolerance_keeps_uninformative_score_exclusions(scores, reason):
    result = compute_advantages(scores, top_k=2, boundary_tie_sigma=0.)
    assert result.metadata["status"] == "skipped"
    assert result.metadata["skip_reason"] == reason
    assert result.advantages == []

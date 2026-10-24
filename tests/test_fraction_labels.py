import pytest
from difficulty.datasets.resolved import trajectory_fraction_labels


def test_trajectory_fraction_uses_full_denominators():
    assert trajectory_fraction_labels(total_steps=4, step=0,
        remaining_reasoning_tokens=9, total_reasoning_tokens=12) == {
        'remaining_steps_fraction': 1.0,
        'remaining_reasoning_tokens_fraction': 0.75}
    assert trajectory_fraction_labels(total_steps=4, step=3,
        remaining_reasoning_tokens=0, total_reasoning_tokens=12) == {
        'remaining_steps_fraction': 0.25,
        'remaining_reasoning_tokens_fraction': 0.0}


def test_trajectory_fraction_rejects_zero_or_invalid_denominator():
    with pytest.raises(ValueError):
        trajectory_fraction_labels(total_steps=4, step=0,
            remaining_reasoning_tokens=0, total_reasoning_tokens=0)
    with pytest.raises(ValueError):
        trajectory_fraction_labels(total_steps=4, step=4,
            remaining_reasoning_tokens=0, total_reasoning_tokens=12)

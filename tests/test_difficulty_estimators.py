import numpy as np
import pytest

from difficulty.models.estimators import (
    FFNMLEstimator,
    RidgeEstimator,
    RidgePlusFFNEstimator,
    transform_target,
)


def data():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(24, 3, 5))
    y = x[:, 0, 0] * 0.7 + x[:, 2, 3] * -0.2 + 2
    return x, y


def test_variable_layer_width_and_ridge():
    x, y = data()
    m = RidgeEstimator(alpha=1.0).fit(x, y)
    assert m.input_dim == 15
    assert np.isfinite(m.predict(x)).all()
    with pytest.raises(ValueError):
        m.predict(np.zeros((2, 2, 5)))


def test_target_transforms():
    from difficulty.models.estimators import inverse_target

    y = np.array([0.0, 1.0, 10.0])
    assert np.allclose(
        inverse_target(transform_target(y, "remaining_steps_log1p"), "remaining_steps_log1p"), y
    )


def test_mlp_and_zero_initialized_residual():
    pytest.importorskip("torch")
    x, y = data()
    m = RidgePlusFFNEstimator(epochs=1, hidden_dim=8)
    m.ridge.fit(x, y)
    base = m.ridge.predict(x)
    m.input_dim = x.reshape(len(x), -1).shape[1]
    m._module = __import__("difficulty.models.estimators", fromlist=["_mlp"])._mlp(
        m.input_dim, 8, 0.0, 42, True
    )
    assert np.allclose(m.predict(x), base, atol=1e-6)
    f = FFNMLEstimator(epochs=2, hidden_dim=8).fit(x, y)
    assert np.isfinite(f.predict(x)).all()

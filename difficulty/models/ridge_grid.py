"""A shared spectral decomposition with the existing sum-loss Ridge convention."""

from pathlib import Path

import numpy as np

from difficulty.models.estimators import Standardizer, flatten_features


class RidgeGrid:
    def __init__(self, x, y):
        x = flatten_features(x)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2 or len(y) != len(x) or not np.isfinite(y).all():
            raise ValueError("Incompatible Ridge targets")
        self.scaler = Standardizer().fit(x)
        z = self.scaler.transform(x)
        # Centering explicitly leaves the intercept unpenalized, including roundoff in z.mean().
        self.x_mean = z.mean(0)
        self.y_mean = y.mean(0)
        z -= self.x_mean
        if len(z) < z.shape[1]:
            # Thin SVD solves the same positive-alpha Ridge problem without truncation.
            u, singular, vt = np.linalg.svd(z, full_matrices=False)
            self.values = singular**2
            self.vectors = vt.T
            self.projected = singular[:, None] * (u.T @ (y - self.y_mean))
        else:
            gram = z.T @ z
            self.values, self.vectors = np.linalg.eigh(gram)
            self.values = np.maximum(self.values, 0.0)
            self.projected = self.vectors.T @ (z.T @ (y - self.y_mean))

    def coefficients(self, alpha):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge grid requires positive finite alpha")
        coef = self.vectors @ (self.projected / (self.values[:, None] + alpha))
        intercept = self.y_mean - self.x_mean @ coef
        return coef, intercept


def predict_model(model, x):
    return ((x - model["feature_mean"]) / model["feature_scale"]) @ model["coef"] + model["intercept"]


def save_model(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **model, **{key: np.asarray(value) for key, value in metadata.items()})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_model(path):
    with np.load(path, allow_pickle=False) as archive:
        model = {name: archive[name] for name in ("coef", "intercept", "feature_mean", "feature_scale")}
    if (model["coef"].ndim != 1 or model["intercept"].shape != ()
            or any(model[name].shape != model["coef"].shape for name in ("feature_mean", "feature_scale"))
            or not all(np.isfinite(value).all() for value in model.values())
            or (model["feature_scale"] <= 0).any()):
        raise ValueError("Invalid saved Ridge parameters")
    return model

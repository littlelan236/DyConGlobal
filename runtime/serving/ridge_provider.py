"""Thin read-only bridge to the existing trained Ridge API.

Training, model parameter validation and prediction remain owned by difficulty.
No alternative estimator/preprocessing formula is maintained here.
"""

from pathlib import Path

import numpy as np


def load(directory, contract):
    from difficulty.models.ridge_grid import load_model, predict_model

    filename = contract["model_file"]
    if filename not in contract["files"]:
        raise ValueError("Ridge payload must be hash-bound by the external contract")
    model = load_model(Path(directory) / filename)
    if len(contract["sites"]) != 1 or len(model["coef"]) != contract["hidden_size"]:
        raise ValueError("Ridge bridge requires one matching public site")

    class Predictor:
        def predict(self, batch):
            batch = np.asarray(batch)
            if batch.ndim != 3 or batch.shape[1:] != (1, contract["hidden_size"]):
                raise ValueError("Expected [N, 1, hidden_size] Ridge inputs")
            return np.asarray(predict_model(model, batch[:, 0, :]), dtype=float)

    return Predictor()

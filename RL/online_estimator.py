"""Read-only predictor bridge for online active-pool linear estimators.

The online trainer owns fitting and writes a small NumPy checkpoint.  This
module only validates the published artifact contract and exposes the same
``predict(batch)`` interface used by the managed parallel scoring broker.
"""
from pathlib import Path

import numpy as np


def publish_online_predictor(cfg, predictor_id, checkpoint, *, target, site, width):
    """Replay only an identical campaign publication, keeping global publish strict."""
    from core.io import read_json, sha256
    from runtime.serving.parallel import artifact_contract
    from runtime.serving.predictors import publish_predictor
    path = Path(cfg["project_root"]) / "runtime/parallel-artifacts" / predictor_id / "manifest.json"
    if not path.exists():
        return publish_predictor(cfg, predictor_id, checkpoint=checkpoint, target=target, fit_scope="train",
            loader="RL.online_estimator:load", sites=[site], hidden_size=width, endpoint="submission")
    contract = artifact_contract(path, model_fingerprint=cfg["model"]["fingerprint"])
    expected = {"loader": "RL.online_estimator:load", "sites": [site], "hidden_size": width,
                "target": target, "endpoint": "submission", "source_checkpoint_sha256": sha256(checkpoint)}
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError("existing online predictor is not the same fit publication")
    return {"manifest": str(path), "manifest_sha256": sha256(path),
            **{k: contract[k] for k in ("loader", "target", "sites", "provenance")}}


def load(directory, contract):
    filename = contract["model_file"]
    if filename not in contract["files"]:
        raise ValueError("Online estimator payload is not hash-bound by the contract")
    path = Path(directory) / filename
    with np.load(path, allow_pickle=False) as archive:
        if "coef" not in archive or "intercept" not in archive:
            raise ValueError("Online estimator checkpoint lacks coef/intercept")
        coef = np.asarray(archive["coef"], dtype=np.float64).reshape(-1)
        intercept = float(np.asarray(archive["intercept"]).item())
        if coef.size != int(contract["hidden_size"]):
            raise ValueError("Online estimator hidden width differs from predictor contract")
        if not np.isfinite(coef).all() or not np.isfinite(intercept):
            raise ValueError("Online estimator checkpoint contains non-finite values")
        feature_mean = np.asarray(archive["feature_mean"], dtype=np.float64) if "feature_mean" in archive else np.zeros_like(coef)
        feature_scale = np.asarray(archive["feature_scale"], dtype=np.float64) if "feature_scale" in archive else np.ones_like(coef)
        if (feature_mean.shape != coef.shape or feature_scale.shape != coef.shape
                or not np.isfinite(feature_mean).all() or not np.isfinite(feature_scale).all()
                or (feature_scale <= 0).any()):
            raise ValueError("invalid online estimator standardizer")
        feature_family = str(np.asarray(archive["feature_family"]).item()) if "feature_family" in archive else None
        base_fingerprint = str(np.asarray(archive["base_fingerprint"]).item()) if "base_fingerprint" in archive else None
    if contract.get("feature_family") is not None and feature_family != contract["feature_family"]:
        raise ValueError("Online estimator feature family differs from predictor contract")
    if contract.get("base_fingerprint") is not None and base_fingerprint != contract["base_fingerprint"]:
        raise ValueError("Online estimator base fingerprint differs from predictor contract")

    class Predictor:
        def predict(self, batch):
            value = np.asarray(batch, dtype=np.float64)
            expected = (len(contract["sites"]), int(contract["hidden_size"]))
            if value.ndim != 3 or value.shape[1:] != expected:
                raise ValueError(f"Expected [N, {expected[0]}, hidden_size] online estimator inputs")
            # The first configured public site is the active-pool feature site.
            from difficulty.models.ridge_grid import predict_model
            return predict_model({"coef": coef, "intercept": intercept,
                                  "feature_mean": feature_mean, "feature_scale": feature_scale}, value[:, 0, :])

    return Predictor()

"""The sole steering payload adapter; no implicit default engine vector."""

from core.io import read_json


def steering_payload(cfg):
    if not cfg["steering"]["enabled"]:
        return False
    payload = read_json(cfg["steering"]["payload"])
    if payload.get("model_fingerprint") != cfg["model"]["fingerprint"]:
        raise ValueError("Steering payload belongs to another checkpoint")
    wire = payload["steering"]
    if not isinstance(wire, dict) or not wire.get("vectors"):
        raise ValueError("Enabled EasySteer payload must contain native vectors")
    return wire

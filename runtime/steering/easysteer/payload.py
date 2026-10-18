"""The sole steering payload adapter; no implicit default engine vector."""

from core.io import read_json, sha256


def steering_payload(cfg):
    if not cfg["steering"]["enabled"]:
        return False
    if (
        cfg["steering"].get("payload_sha256")
        and sha256(cfg["steering"]["payload"]) != cfg["steering"]["payload_sha256"]
    ):
        raise ValueError("Steering payload changed after configuration freeze")
    payload = read_json(cfg["steering"]["payload"])
    if payload.get("model_fingerprint") != cfg["model"]["fingerprint"]:
        raise ValueError("Steering payload belongs to another checkpoint")
    wire = payload["steering"]
    if not isinstance(wire, dict) or not wire.get("vectors"):
        raise ValueError("Enabled EasySteer payload must contain native vectors")
    return wire

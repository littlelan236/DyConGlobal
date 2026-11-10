"""Contract and cache identity helpers for checkpoint/receipt binding."""

from pathlib import Path
import json
import hashlib


def contract_sha256(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_objective_contract(path, contract):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"contract": contract, "sha256": contract_sha256(contract)},
                               sort_keys=True, indent=2) + "\n", encoding="utf-8")


def verify_objective_contract(path, expected):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("contract") != expected or value.get("sha256") != contract_sha256(expected):
        raise ValueError("Objective contract identity mismatch")
    return value

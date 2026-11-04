from pathlib import Path

import yaml

from core.config import resolve
from core.model_identity import model_identity_verification_enabled


def test_model_identity_verification_is_enabled_by_default(cfg):
    cfg["model"].pop("verify_identity", None)
    assert model_identity_verification_enabled(cfg)


def test_model_identity_verification_can_be_disabled(tmp_path, cfg):
    raw = yaml.safe_load(Path(cfg["config_source"]).read_text())
    raw["model"]["verify_identity"] = False
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    resolved = resolve(path)
    assert resolved["model"]["verify_identity"] is False
    assert not model_identity_verification_enabled(resolved)


def test_legacy_config_without_switch_keeps_default_behavior(tmp_path, cfg):
    raw = yaml.safe_load(Path(cfg["config_source"]).read_text())
    raw["model"].pop("verify_identity", None)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    resolved = resolve(path)
    assert "verify_identity" not in resolved["model"]
    assert model_identity_verification_enabled(resolved)

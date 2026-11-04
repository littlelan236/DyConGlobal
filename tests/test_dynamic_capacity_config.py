import pytest

from core.config import Config


@pytest.mark.parametrize("dynamic", [True, False])
def test_capacity_is_derived_from_max_num_seqs(cfg, dynamic):
    cfg["capture"].update(enabled=True, anchors=["think_close"], layers=[3])
    cfg["capture"]["export"]["enabled"] = True
    cfg["serving"].update(management="managed", max_num_seqs=2)
    cfg.setdefault("parallel", {}).update(enabled=True, branch_count=2,
                           dynamic_branch_admission=dynamic, save_sites=[3],
                           estimator_ref="fixture/manifest.json")
    # `cfg` contains resolver metadata outside the strict input schema.
    raw = dict(cfg)
    for key in ("project_root", "config_source", "overrides", "config_source_sha256", "config_hash"):
        raw.pop(key, None)
    assert Config.model_validate(raw).parallel.max_inflight_branches == 2


def test_capacity_cannot_be_manually_configured(cfg):
    cfg["serving"].update(management="managed", max_num_seqs=2)
    cfg.setdefault("parallel", {}).update(enabled=True, branch_count=2,
                           max_inflight_branches=1, save_sites=[3],
                           estimator_ref="fixture/manifest.json")
    raw = dict(cfg)
    for key in ("project_root", "config_source", "overrides", "config_source_sha256", "config_hash"):
        raw.pop(key, None)
    with pytest.raises(ValueError, match="derived from serving.max_num_seqs"):
        Config.model_validate(raw)

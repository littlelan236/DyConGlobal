import pytest

from core.config import Config


@pytest.mark.parametrize("dynamic", [True, False])
def test_capacity_below_group_requires_dynamic_admission(cfg, dynamic):
    cfg["capture"].update(enabled=True, anchors=["think_close"], layers=[3])
    cfg["capture"]["export"]["enabled"] = True
    cfg["serving"].update(management="managed", max_num_seqs=2)
    cfg.setdefault("parallel", {}).update(enabled=True, branch_count=2, max_inflight_branches=1,
                           dynamic_branch_admission=dynamic, save_sites=[3],
                           estimator_ref="fixture/manifest.json")
    # `cfg` contains resolver metadata outside the strict input schema.
    raw = dict(cfg)
    for key in ("project_root", "config_source", "overrides", "config_source_sha256", "config_hash"):
        raw.pop(key, None)
    if dynamic:
        assert Config.model_validate(raw).parallel.max_inflight_branches == 1
    else:
        with pytest.raises(ValueError, match="capacity must accommodate"):
            Config.model_validate(raw)

"""Consensus membership intersects the global best-scoring half, never the representative."""
import copy
import math
from pathlib import Path

import pytest

from core.io import hash_json, read_json
from RL.config import Config, load, schedule_contract
from RL.grpo.advantage import CONSENSUS_ADVANTAGE_VERSION, compute_advantages
from RL.grpo.data import prepare, tensor_groups
from RL.grpo.groups import GroupBatch


SETTINGS = dict(advantage_policy=CONSENSUS_ADVANTAGE_VERSION, top_k=None,
                top_k_fraction=0.5, boundary_tie_sigma=0.)


def consensus_config():
    root = Path(__file__).parents[1]
    config, _ = load(root / "RL/experiments/grpo-offline-k3-smoke.yaml", root)
    return Config.model_validate({**config.model_dump(),
        "grpo": {**config.grpo.model_dump(), **SETTINGS}})


def group(scores, retained):
    rows = []
    for i, (score, flag) in enumerate(zip(scores, retained, strict=True)):
        p, c = [2, 3], [4 + i]
        rows.append(dict(branch_sample_id=f"b{i}", prompt_token_ids=p,
            completion_token_ids=c, prompt_sha256=hash_json(p), completion_sha256=hash_json(c),
            historical_score=score, retained=flag, selected=i == len(scores) - 1,
            action={"command": "true"}))
    return dict(group=dict(group_id="g", expected_branch_count=len(rows), data_ready=True), members=rows)


def test_global_top_16_intersection_and_original_score_weights():
    scores = list(range(32))
    flags = [i in {0, 7, 15, 16, 30} for i in range(32)]
    result = compute_advantages(scores, retained=flags, **SETTINGS)
    assert result.top_k == 16 and result.positive_count == 3 and result.negative_count == 29
    assert [i for i, a in enumerate(result.advantages) if a > 0] == [0, 7, 15]
    weights = [1., .5 + .5 * 8 / 15, .5]
    for i, w in zip([0, 7, 15], weights):
        assert result.advantages[i] == pytest.approx(w / (sum(weights) / 3))
    assert all(a == pytest.approx(-3 / 29) for i, a in enumerate(result.advantages) if i not in {0, 7, 15})
    assert math.fsum(result.advantages) == pytest.approx(0., abs=1e-12)
    assert result.metadata["positive_indices"] == [0, 7, 15]


@pytest.mark.parametrize("n", [2, 3, 9, 17, 32])
def test_full_consensus_still_uses_global_half(n):
    result = compute_advantages(list(range(n)), retained=[True] * n, **SETTINGS)
    k = min(n - 1, math.ceil(n / 2))
    assert result.positive_count == k
    assert all(a > 0 for a in result.advantages[:k])
    assert all(a < 0 for a in result.advantages[k:])
    assert sum(result.advantages[:k]) / k == pytest.approx(1.)
    assert math.fsum(result.advantages) == pytest.approx(0., abs=1e-12)


def test_exact_boundary_tie_uses_native_order_and_not_retained_first():
    result = compute_advantages([0., 1., 1., 2.], retained=[False, True, True, True], **SETTINGS)
    assert result.advantages == [-1 / 3, 1., -1 / 3, -1 / 3]
    assert result.metadata["boundary_gap"] == 0.


@pytest.mark.parametrize("scores,flags,reason", [
    ([0., 1., 2., 3.], [False, False, True, True], "no_consensus_in_top_half"),
    ([0., 1., 2., 3.], [False] * 4, "no_consensus_in_top_half"),
    ([1.] * 4, [True] * 4, "all_scores_equal"),
    ([0., None, 2., 3.], [True] * 4, "missing_or_nonfinite_score"),
    ([0., float("inf"), 2., 3.], [True] * 4, "missing_or_nonfinite_score"),
])
def test_empty_positive_or_invalid_scores_are_explicit_exclusions(scores, flags, reason):
    result = compute_advantages(scores, retained=flags, **SETTINGS)
    assert result.advantages == [] and result.metadata["skip_reason"] == reason


@pytest.mark.parametrize("flags", [None, [], [True], [True, 1, False, False], [True, None, False, False]])
def test_consensus_requires_exact_boolean_membership(flags):
    with pytest.raises(ValueError, match="boolean retained"):
        compute_advantages([0., 1., 2., 3.], retained=flags, **SETTINGS)


def test_group_loader_uses_retained_not_selected_and_keeps_every_branch():
    value = group(list(range(32)), [i in {1, 6, 16} for i in range(32)])
    prepared = GroupBatch.from_dataset_group(value, as_tensors=False, **SETTINGS)
    assert prepared.group_size == 32
    assert [i for i, a in enumerate(prepared.advantages) if a > 0] == [1, 6]
    value["members"][1]["selected"] = True
    assert GroupBatch.from_dataset_group(value, as_tensors=False, **SETTINGS).advantages == prepared.advantages
    del value["members"][0]["retained"]
    with pytest.raises(ValueError, match="boolean retained"):
        GroupBatch.from_dataset_group(value, as_tensors=False, **SETTINGS)


@pytest.mark.parametrize("override", [{"top_k": 4, "top_k_fraction": None},
    {"top_k_fraction": .25}, {"boundary_tie_sigma": .05}, {"advantage_policy": "unknown"}])
def test_config_rejects_conflicting_consensus_settings(override):
    raw = consensus_config().model_dump()
    raw["grpo"].update(override)
    with pytest.raises(ValueError):
        Config.model_validate(raw)


def synthetic_membership(monkeypatch):
    original = read_json(Path(__file__).parents[1] / "RL/fixtures/grpo-native-smoke-v1.json")
    for g in original["groups"]:
        for i, row in enumerate(g["members"]):
            row["retained"] = i != 1
    monkeypatch.setattr("RL.grpo.data.read_json", lambda p: copy.deepcopy(original))
    return original


def test_preflight_freezes_policy_membership_and_advantages(monkeypatch):
    fixture = synthetic_membership(monkeypatch)
    config = consensus_config()
    raw, proof = prepare(config)
    assert proof["advantage_policy"] == CONSENSUS_ADVANTAGE_VERSION
    assert schedule_contract(config)["grpo"]["advantage_policy"] == CONSENSUS_ADVANTAGE_VERSION
    assert all(g["frozen_advantages"] == [1., -.5, -.5] for g in raw)
    assert all(list(g.advantages) == [1., -.5, -.5] for g in tensor_groups(raw))
    # Changing membership outside the ranking cutoff preserves weights but changes input identity.
    fixture["groups"][0]["members"][2]["retained"] = False
    changed_raw, changed = prepare(config)
    assert [g["frozen_advantages"] for g in changed_raw] == [g["frozen_advantages"] for g in raw]
    assert changed["group_order_sha256"] != proof["group_order_sha256"]
    fixture["groups"][0]["members"][0]["retained"] = None
    with pytest.raises(ValueError, match="boolean retained"):
        prepare(config)


def test_preflight_reports_no_positive_consensus_group(monkeypatch):
    fixture = synthetic_membership(monkeypatch)
    fixture["groups"][0]["members"][0]["retained"] = False
    config = consensus_config()
    config.training.max_updates = 1
    _, proof = prepare(config)
    assert proof["group_count"] == 4
    assert proof["exclusion_reasons"] == {"no_consensus_in_top_half": 1}


def test_consensus_tiny_training_resume_and_membership_change_rejection(tmp_path, monkeypatch):
    import test_grpo_lifecycle as lifecycle
    from RL.grpo.contract import validate_resume
    from RL.modeling import adapter_fingerprint
    fixture = synthetic_membership(monkeypatch)
    original_load = lifecycle.load

    def policy_load(*args):
        config, source = original_load(*args)
        raw = config.model_dump()
        raw["grpo"].update(SETTINGS)
        return Config.model_validate(raw), source

    monkeypatch.setattr(lifecycle, "load", policy_load)
    _, full_path, full_model, full = lifecycle.run_tiny(tmp_path, "consensus-full")
    config, paused_path, _, paused = lifecycle.run_tiny(tmp_path, "consensus-paused", stop=1)
    checkpoint = paused_path / "checkpoints/step-000001"
    _, _, resumed_model, resumed = lifecycle.run_tiny(tmp_path, "consensus-resumed", mode="resume", source=checkpoint)
    assert paused["status"] == "paused" and full["status"] == resumed["status"] == "completed"
    assert full["state"] == resumed["state"]
    assert adapter_fingerprint(full_model) == adapter_fingerprint(resumed_model)
    metadata = read_json(checkpoint / "state.json")
    assert metadata["contract"]["input"]["advantage_policy"] == CONSENSUS_ADVANTAGE_VERSION
    fixture["groups"][0]["members"][2]["retained"] = False
    changed = prepare(config)[1]
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_resume(config, changed, metadata)

"""Online pre-cap evidence survives capping without an extra model forward."""
import json
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from RL.config import Grpo, TokenDiagnostics
from RL.grpo.backend import GroupObjective
from RL.grpo.metrics import summarize_update


@pytest.mark.parametrize("extreme_logprob", [-22., -110., -1000.])
def test_online_raw_outlier_survives_cap_without_replay(monkeypatch, tmp_path, extreme_logprob):
    import RL.grpo.backend as backend

    group = SimpleNamespace(group_id="group", branch_ids=["branch"], advantages=[1.],
        prompt_token_ids=[[11, 12]], completion_token_ids=[[21, 22, 23]],
        group_size=1, metadata={"physical_task": {"native_branch_ordinals": [11]}},
        branch_batch=lambda *_: {})
    reference = torch.tensor([-1., -2., -2.])
    model = torch.nn.Linear(1, 1)
    rng = torch.random.get_rng_state().clone()
    runs = []
    for enabled in (False, True):
        current = torch.tensor([-1., extreme_logprob, -2.], requires_grad=True)
        calls = []

        def score(*args, **kwargs):
            calls.append(1)
            return current

        monkeypatch.setattr(backend, "model_target_logprobs", score)
        settings = Grpo(kl_beta=.02, k3_cap=1., token_diagnostics=TokenDiagnostics(
            enabled=enabled, k3_threshold=1., max_tokens_per_branch=8, context_tokens=1,
            compare_full_branch=False, compare_base=False))
        assert not settings.ratio_guard.enabled
        objective = GroupObjective({"group": {"branch": reference}}, settings, 128)
        result = objective._branch_result(model, group, 0, "branch", torch.device("cpu"))
        result.loss.backward()
        assert len(calls) == 1
        assert torch.equal(torch.random.get_rng_state(), rng)
        assert torch.isfinite(result.loss) and torch.isfinite(current.grad).all()
        assert result.k3_penalty.item() == pytest.approx(1 / 3)
        runs.append((result.loss.detach(), current.grad.clone(), result.metrics))

    torch.testing.assert_close(runs[0][0], runs[1][0], rtol=0, atol=0)
    torch.testing.assert_close(runs[0][1], runs[1][1], rtol=0, atol=0)
    assert "_token_diagnostics" not in runs[0][2]
    # Exercise the same public summary and strict JSON boundary as committed updates.
    metrics = {**summarize_update([[runs[1][2]]]), "global_step": 7,
               "evaluated_at_global_step": 6}
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(metrics, allow_nan=False) + "\n", encoding="utf-8")
    saved = json.loads(path.read_text(encoding="utf-8"))
    diagnostics = saved["token_diagnostics"]
    assert diagnostics["threshold_exceeded_tokens"] == 1
    assert diagnostics["truncated_outlier_tokens"] == 0
    token, = diagnostics["outliers"]
    assert token["group_id"] == "group" and token["branch_id"] == "branch"
    assert token["native_branch_ordinal"] == 11
    assert token["target_token_id"] == 22
    assert token["completion_offset"] == 1 and token["sequence_position"] == 3
    assert token["context_token_ids"] == [21, 22, 23]
    assert token["current_logprob"] == extreme_logprob
    assert token["reference_logprob"] == -2.
    delta = -2. - extreme_logprob
    if delta < 700:
        assert token["k3_fp64"] == pytest.approx(math.expm1(delta) - delta)
        assert token["k3_fp64"] > 1.
    else:
        assert token["k3_fp64"] == "inf"
    assert saved["k3_cap_clipped_tokens"] == 1

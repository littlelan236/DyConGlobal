"""Bounded token evidence, finite checks and unchanged training mathematics."""
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from RL.config import Grpo, TokenDiagnostics, load, schedule_contract
from RL.grpo.token_diagnostics import branch_diagnostics, summarize_token_diagnostics


def group():
    return SimpleNamespace(group_id="group", branch_ids=["branch"], advantages=[-1.],
        prompt_token_ids=[[11, 12]], completion_token_ids=[[21, 22, 23]],
        group_size=1, metadata={}, branch_batch=lambda *_: {})


def test_outlier_coordinates_probabilities_and_truncation():
    current = torch.tensor([-1., -22., -20.], requires_grad=True)
    reference = torch.tensor([-1., -2., -2.])
    state = torch.random.get_rng_state().clone()
    row = branch_diagnostics(current, reference, group(), 0,
                             TokenDiagnostics(max_tokens_per_branch=1, context_tokens=1))
    token = row["outliers"][0]
    assert token["completion_offset"] == 1 and token["sequence_position"] == 3
    assert token["target_token_id"] == 22 and token["context_token_ids"] == [21, 22, 23]
    assert token["ratio_fp64"] == pytest.approx(__import__("math").exp(-20))
    assert token["current_probability_fp64"] == pytest.approx(__import__("math").exp(-22))
    assert row["threshold_exceeded_tokens"] == 2 and row["truncated_outlier_tokens"] == 1
    assert not any(row["counts"].values()) and current.grad is None
    assert torch.equal(state, torch.random.get_rng_state())
    json.dumps(row, allow_nan=False)
    merged = summarize_token_diagnostics([[{"_token_diagnostics": row}], [{"_token_diagnostics": row}]])
    assert merged["token_diagnostics"]["token_count"] == 6
    assert len(merged["token_diagnostics"]["outliers"]) == 2


def test_nonfinite_values_and_fp32_underflow_remain_distinguishable():
    row = branch_diagnostics(torch.tensor([-110., -float("inf"), float("nan")]),
        torch.tensor([-1., -2., -3.]), group(), 0, TokenDiagnostics())
    counts = row["counts"]
    assert counts["ratio_zero_fp32_tokens"] == 2
    assert counts["current_probability_zero_fp32_tokens"] == 2
    # Nonfinite target evidence must survive strict JSON serialization.
    json.dumps(row, allow_nan=False)
    assert row["outliers"][0]["current_logprob"] in ("-inf", "nan")


def test_backend_diagnostics_preserve_exact_loss_gradient_and_rng(monkeypatch):
    from RL.grpo.backend import GroupObjective
    import RL.grpo.backend as backend

    results = []
    state = torch.random.get_rng_state().clone()
    for enabled in (False, True):
        theta = torch.tensor([-1., -22., -2.], requires_grad=True)
        monkeypatch.setattr(backend, "model_target_logprobs", lambda *args: theta)
        settings = Grpo(kl_beta=0.0, token_diagnostics=TokenDiagnostics(enabled=enabled))
        objective = GroupObjective({"group": {"branch": torch.tensor([-1., -2., -2.])}}, settings, 128)
        model = torch.nn.Linear(1, 1)
        torch.random.set_rng_state(state)
        result = objective._branch_result(model, group(), 0, "branch", torch.device("cpu"))
        result.loss.backward()
        results.append((result.loss.detach(), theta.grad, result.metrics))
        assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
    assert "_token_diagnostics" not in results[0][2]
    assert len(results[1][2]["_token_diagnostics"]["outliers"]) == 1


def test_raw_k3_overflow_diagnostics_do_not_reject_guarded_finite_objective(monkeypatch):
    import RL.grpo.backend as backend
    from RL.config import RatioGuard
    current = torch.tensor([-1., -1000., -2.], requires_grad=True)
    monkeypatch.setattr(backend, "model_target_logprobs", lambda *args: current)
    settings = Grpo(kl_beta=0.0, ratio_guard=RatioGuard(enabled=True),
                    token_diagnostics=TokenDiagnostics(enabled=True))
    objective = backend.GroupObjective({"group": {"branch": torch.tensor([-1., -2., -2.])}}, settings, 128)
    result = objective._branch_result(torch.nn.Linear(1, 1), group(), 0, "branch", torch.device("cpu"))
    result.loss.backward()
    assert torch.isfinite(result.loss) and torch.isfinite(current.grad).all()
    assert result.metrics["_token_diagnostics"]["counts"]["k3_posinf_tokens"] == 1


def test_enabling_diagnostics_preserves_legacy_schedule_contract():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    config, _ = load(root / "RL/experiments/grpo-offline-k3-smoke.yaml", root)
    original = schedule_contract(config)
    config.grpo.token_diagnostics.enabled = True
    assert schedule_contract(config) == original


def test_full_and_base_controls_reuse_same_native_targets(monkeypatch):
    from contextlib import contextmanager
    from RL.grpo.token_diagnostics import compare_branch_scores
    import RL.grpo.logprobs as projection

    model = torch.nn.Linear(1, 1)
    model.base = False
    @contextmanager
    def disable_adapter():
        model.base = True
        try:
            yield
        finally:
            model.base = False
    model.disable_adapter = disable_adapter
    calls = []
    def score(*args, **kwargs):
        calls.append((model.base, torch.is_grad_enabled()))
        return torch.tensor([-1., -2.1 if model.base else -21.9, -2.])
    monkeypatch.setattr(projection, "model_target_logprobs", score)
    current, reference = torch.tensor([-1., -22., -2.]), torch.tensor([-1., -2., -2.])
    settings = TokenDiagnostics(enabled=True, compare_full_branch=True, compare_base=True)
    row = branch_diagnostics(current, reference, group(), 0, settings)
    compare_branch_scores(model, group(), 0, current, reference, row, settings, 128)
    token = row["outliers"][0]
    assert token["full_current_logprob"] == pytest.approx(-21.9)
    assert token["base_bf16_logprob"] == pytest.approx(-2.1)
    assert calls == [(False, False), (True, False)] and not model.base


def test_alternative_distributions_use_causal_positions_and_preserve_scorer():
    from RL.grpo.token_diagnostics import target_distributions
    from RL.grpo.logprobs import hidden_target_logprobs
    model = torch.nn.Linear(2, 4, bias=False)
    model.get_output_embeddings = lambda: model
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
    model.requires_grad_(False)
    hidden = torch.tensor([[[0., 0.], [2., 1.], [-1., 2.], [0., 0.]]])
    labels = torch.tensor([[-100, -100, 2, 1]])
    state = torch.random.get_rng_state().clone()
    evidence = target_distributions(model, hidden, labels, [0, 1], 2, top_k=4)
    scored = hidden_target_logprobs(model, hidden, labels, 2)
    assert evidence[0]["target_token_id"] == 2 and evidence[0]["target_rank"] == 4
    assert evidence[1]["target_token_id"] == 1 and evidence[1]["target_rank"] == 1
    assert [r["token_id"] for r in evidence[0]["top_tokens"]] == [0, 1, 3, 2]
    for i in range(2):
        assert evidence[i]["target_logprob"] == pytest.approx(float(scored[i]), abs=1e-6)
        assert sum(__import__("math").exp(t["logprob"]) for t in evidence[i]["top_tokens"]) == pytest.approx(1.)
        assert not any(evidence[i][k] for k in ("logits_nan", "logits_posinf", "logits_neginf"))
    assert torch.equal(state, torch.random.get_rng_state())
    with pytest.raises(ValueError, match="positions"):
        target_distributions(model, hidden, labels, [2], 2)

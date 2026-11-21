"""Numerical and fault recovery acceptance of the online logical update."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from RL.online_bootstrap import EstimatorDatasetStore, HiddenPoint, SourceMass


def point(name, *, birth=99, base="base"):
    return HiddenPoint(name, [1., 2.], 3., "S0", "fam", base, "tok", "layer_1", 2, "bf16", birth)


def row(name, source="baseline", task="a", group="g", trajectory="t", step=0, tokens=1, total=1):
    from RL.online_runtime import OnlineRow
    return OnlineRow(name, task, source, group, (1, 2) + (3,) * tokens, (3,) * tokens,
                     (-2.,) * tokens, None, -1., trajectory, total, step, False)


def test_pool_restores_age_and_tie_order(tmp_path):
    store = EstimatorDatasetStore(str(tmp_path), capacity=3, feature_family="fam")
    store.initialize([point("a"), point("b"), point("c")])
    store.ingest([point("d")], step=1)
    expected = store.pool.points
    fresh = EstimatorDatasetStore(str(tmp_path), capacity=3, feature_family="fam")
    fresh.restore()
    assert fresh.pool.points == expected
    fresh.ingest([point("e"), point("f")], step=2)
    assert {p.point_id for p in fresh.pool.points} == {"d", "e", "f"}


@pytest.mark.parametrize("stage", ["archive", "active", "receipt"])
def test_ingest_replays_after_partial_publication(tmp_path, monkeypatch, stage):
    store = EstimatorDatasetStore(str(tmp_path), capacity=2, feature_family="fam")
    store.initialize([point("a"), point("b")])
    atomic = store._atomic_json
    interrupted = False
    def fail(path, value):
        nonlocal interrupted
        target = (path.name == "active.json" if stage in {"archive", "active"}
                  else path.parent.name == "ingests")
        if target and not interrupted:
            interrupted = True
            if stage != "archive":
                atomic(path, value)
            raise RuntimeError("power loss")
        atomic(path, value)
    monkeypatch.setattr(store, "_atomic_json", fail)
    with pytest.raises(RuntimeError, match="power loss"):
        store.ingest([point("c")], step=1)
    fresh = EstimatorDatasetStore(str(tmp_path), capacity=2, feature_family="fam")
    fresh.restore()
    assert "c" in {p.point_id for p in fresh.pool.points}
    assert fresh.ingest([point("c")], step=1)["reused"]
    assert len((tmp_path / "history.jsonl").read_text().splitlines()) == 3
    assert [p.birth_step for p in fresh.pool.points if p.point_id == "c"] == [1]


def test_ingest_validates_before_mutation_and_deduplicates(tmp_path):
    store = EstimatorDatasetStore(str(tmp_path), capacity=1, feature_family="fam")
    store.initialize([point("a")])
    before = (tmp_path / "history.jsonl").read_bytes()
    for points in ([point("b", base="other")], [point("b"), point("c")],
                   [point("a", base="other")]):
        with pytest.raises(ValueError):
            store.ingest(points, step=1)
        assert (tmp_path / "history.jsonl").read_bytes() == before
    assert store.ingest([point("b"), point("b")], step=1)["added"] == 1
    with pytest.raises(ValueError, match="different points"):
        store.ingest([point("c")], step=1)


def test_row_masses_balance_tasks_decisions_and_trajectory_tokens():
    from RL.online_runtime import online_row_masses
    rows = [row("d1", "dycon", group="g1"), row("d2", "dycon", group="g2"),
            row("d3", "dycon", task="b", group="g1"),
            row("b1", total=4), row("b2", step=1, tokens=3, total=4),
            row("b3", trajectory="t2", tokens=2, total=2),
            row("b4", task="b", trajectory="t3")]
    assert online_row_masses(rows, SourceMass()) == pytest.approx(
        [1/8, 1/8, 1/4, 1/32, 3/32, 1/8, 1/4])


def test_provider_accepts_multiple_rows_per_trajectory_and_rejects_missing():
    from RL.online_runtime import OnlineRolloutBatch, SharedVllmRolloutProvider
    def request(payload):
        rows = []
        for item in payload["work_items"]:
            for step in range(2):
                rows.append(row(item["trajectory_id"] + str(step), item["source"],
                                group=str(step), trajectory=item["trajectory_id"], step=step, total=2))
        return OnlineRolloutBatch(0, ("a",), "S0", "E0", tuple(rows),
                                  request_sha256=payload["request_sha256"]).as_dict()
    provider = SharedVllmRolloutProvider(request, m=2)
    assert len(provider(["a"], 0, "S0", "E0").rows) == 6
    def incomplete(payload):
        response = request(payload)
        response["rows"] = response["rows"][:-2]
        return response
    with pytest.raises(ValueError, match="coverage"):
        SharedVllmRolloutProvider(incomplete, m=2)(["a"], 0, "S0", "E0")


def test_sequential_update_matches_dense_gradient_and_preserves_prefix(monkeypatch):
    import copy
    import torch
    from RL.grpo.objective import selected_logprobs
    from RL.online_runtime import (OnlineRolloutBatch, TorchOnlineUpdater,
                                    collate_rows, online_row_masses, online_torch_objective)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.projection = torch.nn.Linear(4, 8)
            self.forward_sizes = []
        def forward(self, input_ids, attention_mask):
            self.forward_sizes.append(input_ids.shape[0])
            return SimpleNamespace(logits=self.projection(self.embedding(input_ids).cumsum(1)))
    def optimizer(model, settings):
        opt = torch.optim.SGD(model.parameters(), lr=.01)
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
    monkeypatch.setattr("RL.training.optimizer_and_scheduler", optimizer)
    torch.manual_seed(12)
    model = Model()
    reference = copy.deepcopy(model)
    rows = (row("d1", "dycon"), row("d2", "dycon"), row("b1", total=3),
            row("b2", step=1, tokens=2, total=3))
    batch = OnlineRolloutBatch(0, ("a",), "S0", "E0", rows)
    updater = TorchOnlineUpdater(model, SimpleNamespace(grad_clip=1e6), source_mass=SourceMass())
    metrics = updater(batch, 0)
    tensors = collate_rows(rows, device=torch.device("cpu"))
    current, mask = selected_logprobs(reference(tensors["input_ids"], tensors["attention_mask"]).logits,
                                      tensors["input_ids"], tensors["labels"], tensors["loss_mask"])
    loss = 0
    for i, mass in enumerate(online_row_masses(rows, SourceMass())):
        source = rows[i].source
        local, _ = online_torch_objective(current[i:i+1], tensors["rollout_logprobs"][i:i+1], None,
                    torch.tensor([-1.]), mask[i:i+1], [source], {source: [[0]]},
                    source_mass=SourceMass(1, 0) if source == "dycon" else SourceMass(0, 1))
        loss = loss + mass * local
    loss.backward()
    # Prefix-only token 1 is absent from all completions yet receives gradient.
    assert reference.embedding.weight.grad[1].abs().sum() > 0
    assert model.forward_sizes == [1, 1, 1, 1]
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=1e-7)
        torch.testing.assert_close(actual, expected - .01 * expected.grad, rtol=2e-5, atol=1e-7)
    assert metrics["loss"] == pytest.approx(float(loss.detach()), rel=1e-6)


def test_row_rejects_missing_prefix_and_nonfinite_advantage():
    from RL.online_runtime import OnlineRow
    raw = row("r").__dict__
    for update in ({"input_token_ids": (3,)}, {"advantage": float("nan")},
                   {"input_token_ids": (-1, 2, 3)}):
        with pytest.raises(ValueError):
            OnlineRow.from_dict({**raw, **update})


def test_barrier_ingests_prefetch_before_refit_once(tmp_path):
    from RL.online_bootstrap import OnlineCoordinator, TaskSchedule
    from RL.online_runtime import OnlineRolloutBatch
    events = []
    schedule = TaskSchedule.build(["a"], n=1, epochs=3)
    def rollout(tasks, step, policy, estimator):
        return OnlineRolloutBatch(step, tuple(tasks), policy, estimator, (row(str(step)),))
    def ingest(points, step):
        events.append(("ingest", step))
        return {}
    def fit(step, result):
        events.append(("fit", step))
        return {}
    OnlineCoordinator(schedule, rollout=rollout, update=lambda *args: {},
        resolved_points=lambda _: (), estimator_ingest=ingest, estimator_refit=fit,
        campaign_id="barrier", state_path=str(tmp_path / "state.json")).run()
    assert events == [("ingest", 0), ("ingest", 1), ("fit", 0),
                      ("ingest", 2), ("fit", 1), ("fit", 2)]


def test_snapshot_ridge_preserves_train_only_scaler(tmp_path):
    import json
    import numpy as np
    from difficulty.training.online import fit_snapshot
    points = [replace(point("a"), hidden=[0., 0.], target=0., split="train"),
              replace(point("b"), hidden=[2., 2.], target=2., split="train"),
              replace(point("v"), hidden=[1000., 1000.], target=10000., split="val")]
    snapshot = tmp_path / "snapshot.jsonl"
    snapshot.write_text("".join(json.dumps(p.__dict__) + "\n" for p in points))
    result = fit_snapshot({"model": {"fingerprint": "base"}}, snapshot, tmp_path / "fit.npz",
                           fit_spec={"alpha": 2., "target": "submission_remaining_steps_linear"})
    with np.load(result["checkpoint"]) as archive:
        np.testing.assert_allclose(archive["feature_mean"], [1., 1.])
        np.testing.assert_allclose(archive["feature_scale"], [1., 1.])
    receipt = json.loads(Path(result["receipt"]).read_text())
    assert receipt["train_count"] == 2 and receipt["diagnostics"]["val"]["count"] == 1

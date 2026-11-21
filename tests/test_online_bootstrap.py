import pytest
from RL.online_bootstrap import (ActiveEstimatorPool, HiddenPoint, MixedRolloutScheduler,
    RolloutWorkItem, SourceMass, TaskSchedule, make_baseline_step,
    validate_feature_family, online_surrogate_loss, OnlineBootstrapEngine, OnlineCoordinator, EstimatorDatasetStore)


def point(i, birth=99):
    return HiddenPoint(str(i), [1., 2., 3., 4.], 1.0, f"model-{i}", "fam", "base", "tok", "site", 4, "bf16", birth)


def test_schedule_shuffles_once_and_repeats_epoch_order():
    s = TaskSchedule.build(["a", "b", "c", "d", "e"], n=2, epochs=3, seed=7)
    assert s.steps_per_epoch == 3
    assert s.total_steps == 9
    assert s.batches[:3] == s.batches[3:6] == s.batches[6:9]
    assert sorted(sum((list(x) for x in s.batches[:3]), [])) == ["a", "b", "c", "d", "e"]


def test_baseline_completion_only_labels_keep_context():
    x = make_baseline_step(trajectory_id="t", task_id="a", repetition=0, agent_step=1,
                           context_token_ids=[1, 2, 3], completion_token_ids=[4, 5],
                           resolved=True, trajectory_token_count=7)
    assert x.input_token_ids == (1, 2, 3, 4, 5)
    assert x.labels == (-100, -100, -100, 4, 5)
    assert x.loss_mask == (0, 0, 0, 1, 1)
    assert x.advantage == 1 and x.trajectory_token_count == 7


def test_source_mass_is_independent_of_group_count():
    c = SourceMass().token_coefficients(task_count=4, dycon_groups=200,
        dycon_candidates=4, dycon_tokens=10, baseline_repetitions=8, baseline_tokens=10)
    assert c["alpha"] == pytest.approx(.5)
    assert c["beta"] == pytest.approx(.5)
    assert c["dycon"] * 200 * 4 * 10 * 4 == pytest.approx(.5)
    assert c["baseline"] * 8 * 10 * 4 == pytest.approx(.5)


def test_scheduler_interleaves_sources_and_starts_longest_first_within_source():
    items = [RolloutWorkItem("d-long", "a", "dycon", 100),
             RolloutWorkItem("d-short", "a", "dycon", 1),
             RolloutWorkItem("b-long", "b", "baseline", 90)]
    out = MixedRolloutScheduler(items).order()
    assert out[0].source == "dycon" and out[0].trajectory_id == "d-long"
    assert {x.source for x in out} == {"dycon", "baseline"}


def test_estimator_pool_replaces_oldest_without_split_filter():
    pool = ActiveEstimatorPool(3, tie_seed=1)
    pool.initialize([point("a"), point("b"), point("c")])
    pool.ingest([point("d")], step=1)
    assert {x.point_id for x in pool.points}.issuperset({"c", "d"}) and len(pool.points) == 3
    pool.ingest([point("e"), point("f")], step=2)
    assert {x.point_id for x in pool.points} == {"d", "e", "f"}


def test_feature_family_allows_model_version_change_but_not_base_change():
    validate_feature_family([point("a")], expected_family="fam", expected_base="base",
                            expected_tokenizer="tok", expected_site="site", expected_width=4)
    with pytest.raises(ValueError):
        validate_feature_family([HiddenPoint("x", [], 0, "new", "fam", "other", "tok", "site", 4, "bf16")],
                                expected_family="fam", expected_base="base", expected_tokenizer="tok",
                                expected_site="site", expected_width=4)




def test_online_objective_uses_rollout_denominator_and_separate_reference():
    x = online_surrogate_loss(logp_current=-1.0, logp_rollout=-1.2, advantage=1.0,
                              logp_kl_reference=-1.3, kl_beta=.02)
    assert x["ratio"] == pytest.approx(__import__("math").exp(.2))  # rollout denominator
    assert x["loss"] == pytest.approx(x["policy"] + .02 * x["k3"])
    with pytest.raises(ValueError):
        online_surrogate_loss(logp_current=-1, logp_rollout=-1, advantage=1, kl_beta=.1)



def test_engine_prefetches_one_batch_and_commits_at_barrier():
    schedule = TaskSchedule.build(["a", "b", "c"], n=2, epochs=1, seed=1)
    events = []
    def rollout(tasks, rollout_step, *_):
        events.append(("rollout", rollout_step, tuple(tasks)))
        return tuple(tasks)
    def update(batch, step):
        events.append(("update", step, batch))
        return step
    def points(batch):
        return ()
    def refit(points_, step):
        events.append(("refit", step))
        return step
    history = OnlineBootstrapEngine(schedule, rollout=rollout, update=update,
                                    resolved_points=points, refit=refit).run()
    assert len(history) == 2
    assert [x[0] for x in events if x[0] == "update"] == ["update", "update"]
    assert [x[1] for x in events if x[0] == "rollout"] == [0, 1]



def test_coordinator_persists_barrier_and_resumes_without_duplicate_updates(tmp_path):
    schedule = TaskSchedule.build(["a", "b", "c"], n=2, epochs=2, seed=3)
    events = []
    def rollout(tasks, step, policy, estimator):
        events.append(("rollout", step, policy, estimator))
        return tuple(tasks)
    def update(batch, step, policy, estimator):
        events.append(("update", step, policy, estimator))
        return step
    def resolved(batch): return ()
    def ingest(points, step): return {"step": step, "count": len(points)}
    def refit(step, ingest_result): return {"version": f"E{step + 1}"}
    state = tmp_path / "state.json"
    first = OnlineCoordinator(schedule, rollout=rollout, update=update, resolved_points=resolved,
                              estimator_ingest=ingest, estimator_refit=refit,
                              campaign_id="c", state_path=str(state))
    assert len(first.run()) == schedule.total_steps
    prefetched = [x for x in events if x[0] == "rollout" and x[1] == 1]
    assert prefetched and prefetched[0][2:] == ("S0", "E0")
    update_count = len([x for x in events if x[0] == "update"])
    resumed = OnlineCoordinator(schedule, rollout=rollout, update=update, resolved_points=resolved,
                                estimator_ingest=ingest, estimator_refit=refit,
                                campaign_id="c", state_path=str(state))
    assert resumed.run() == []
    assert len([x for x in events if x[0] == "update"]) == update_count



def test_estimator_store_archives_history_and_replaces_active_points(tmp_path):
    store = EstimatorDatasetStore(str(tmp_path / "estimator"), capacity=2,
                                  feature_family="fam")
    store.initialize([point("a"), point("b")])
    result = store.ingest([point("c")], step=4)
    assert result["historical_appended"] == 1
    assert "c" in {x.point_id for x in store.pool.points} and len(store.pool.points) == 2
    assert (tmp_path / "estimator" / "history.jsonl").read_text().count("\n") == 3
    manifest = store.manifest()
    assert manifest["active"]["capacity"] == 2
    assert manifest["history_exists"] is True



def test_online_runtime_collates_all_completion_tokens_and_backprops():
    torch = pytest.importorskip("torch")
    from RL.online_runtime import OnlineRow, collate_rows, online_torch_objective
    row = OnlineRow("r", "task", "baseline", "task", (1, 2, 3, 4), (3, 4),
                    (-1.0, -1.0), (-1.0, -1.0), 1.0)
    batch = collate_rows([row], device=torch.device("cpu"))
    assert batch["loss_mask"].tolist() == [[False, False, True, True]]
    current = torch.tensor([[-.5, -.4, -.3]], requires_grad=True)
    loss, metrics = online_torch_objective(
        current, batch["rollout_logprobs"][:, :3], batch["kl_reference_logprobs"][:, :3],
        torch.tensor([1.0]), torch.tensor([[False, False, True]]), ["baseline"],
        {"baseline": [[0]]}, source_mass=SourceMass(), kl_beta=.02)
    loss.backward()
    assert metrics["baseline_rows"] == 1
    assert torch.isfinite(current.grad).all()



def test_shared_vllm_provider_sends_mixed_work_manifest():
    from RL.online_runtime import SharedVllmRolloutProvider
    seen = {}
    def request(payload):
        seen.update(payload)
        rows = [{"row_id": "d", "task_id": "a", "source": "dycon", "group_id": "dg",
                 "input_token_ids": [1, 2], "completion_token_ids": [2],
                 "rollout_logprobs": [-1.0], "advantage": 1.0, "trajectory_id": "a:dycon"}]
        for rep in range(2):
            rows.append({"row_id": f"b{rep}", "task_id": "a", "source": "baseline",
                         "group_id": f"bg{rep}", "input_token_ids": [1, 2],
                         "completion_token_ids": [2], "rollout_logprobs": [-1.0],
                         "advantage": -1.0, "trajectory_id": f"a:baseline:{rep}", "agent_step": 0,
                         "trajectory_token_count": 1, "resolved": False})
        return {"schema": "online_bootstrap_rollout_batch_v1", "step": 0,
                "tasks": ["a"], "policy_version": "S0", "estimator_version": "E0",
                "rows": rows, "request_sha256": payload["request_sha256"]}
    provider = SharedVllmRolloutProvider(request, m=2)
    batch = provider(["a"], 0, "S0", "E0")
    assert len(batch.rows) == 3
    assert len(seen["work_items"]) == 3
    assert len(seen["request_sha256"]) == 64


def test_http_rollout_client_posts_json(monkeypatch):
    import json
    from RL.online_runtime import HttpRolloutClient
    seen = {}
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return json.dumps({"schema": "online_bootstrap_rollout_batch_v1"}).encode()
    def fake_open(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["body"] = json.loads(request.data.decode())
        return Response()
    monkeypatch.setattr("RL.online_runtime.urlopen", fake_open)
    result = HttpRolloutClient("http://127.0.0.1:18080/rollout", timeout_seconds=7)(
        {"schema": "online_bootstrap_rollout_request_v1", "step": 0})
    assert result["schema"] == "online_bootstrap_rollout_batch_v1"
    assert seen["url"].endswith("/rollout") and seen["timeout"] == 7
    assert seen["body"]["step"] == 0


def test_estimator_active_records_include_hidden_payload(tmp_path):
    from RL.online_bootstrap import EstimatorDatasetStore
    store = EstimatorDatasetStore(str(tmp_path / "estimator"), capacity=1, feature_family="fam")
    store.initialize([point("a")])
    records = store.active_records()
    assert records[0]["point_id"] == "a"
    assert records[0]["hidden"] == [1., 2., 3., 4.]


def test_scheduler_assigns_long_tail_to_least_loaded_worker():
    from RL.online_bootstrap import MixedRolloutScheduler, RolloutWorkItem
    items = [RolloutWorkItem("a", "a", "dycon", 10), RolloutWorkItem("b", "b", "baseline", 9),
             RolloutWorkItem("c", "c", "baseline", 2), RolloutWorkItem("d", "d", "baseline", 1)]
    assigned = MixedRolloutScheduler.assign_workers(items, [2, 3])
    assert [worker for _, worker in assigned] == [2, 3, 3, 2]


def test_coordinator_reuses_prefetched_batch_after_update_interrupt(tmp_path):
    from RL.online_runtime import OnlineRolloutBatch, OnlineRow
    schedule = TaskSchedule.build(["a", "b"], n=1, epochs=2, seed=2)
    calls, interrupted = [], {"value": True}

    def rollout(tasks, step, policy, estimator):
        calls.append(("rollout", step))
        task = tasks[0]
        row = OnlineRow(f"r{step}", task, "baseline", f"g{step}", (1, 2), (2,),
                        (-1.0,), None, -1.0, resolved=False)
        return OnlineRolloutBatch(step, tuple(tasks), policy or "S0", estimator or "E0", (row,))

    def update(batch, step, policy, estimator):
        calls.append(("update", step))
        if step == 0 and interrupted["value"]:
            interrupted["value"] = False
            raise RuntimeError("injected interruption")
        return {"checkpoint_path": f"checkpoint-{step}"}

    state = tmp_path / "state.json"
    kwargs = dict(rollout=rollout, update=update, resolved_points=lambda _b: (),
                  estimator_ingest=lambda _p, _s: {}, estimator_refit=lambda _s, _i: {},
                  campaign_id="prefetch", state_path=str(state))
    with pytest.raises(RuntimeError):
        OnlineCoordinator(schedule, **kwargs).run()
    assert (tmp_path / "state.json.batches" / "step-000001.json").is_file()
    OnlineCoordinator(schedule, **kwargs).run()
    assert [x for x in calls if x == ("rollout", 1)] == [("rollout", 1)]


def test_estimator_history_deduplicates_evicted_point(tmp_path):
    store = EstimatorDatasetStore(str(tmp_path / "estimator"), capacity=1, feature_family="fam")
    store.initialize([point("a")])
    assert store.ingest([point("b")], step=1)["historical_appended"] == 1
    assert store.ingest([point("a")], step=2)["historical_appended"] == 0
    assert (tmp_path / "estimator" / "history.jsonl").read_text().count("\n") == 2


def test_online_estimator_predictor_loader_validates_contract(tmp_path):
    import hashlib
    import numpy as np
    from RL.online_estimator import load
    checkpoint = tmp_path / "model.npz"
    np.savez(checkpoint, coef=np.array([1.0, 2.0]), intercept=np.array(3.0),
             feature_family=np.asarray("fam"), base_fingerprint=np.asarray("base"))
    contract = {"model_file": "model.npz",
                "files": {"model.npz": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
                "sites": [0], "hidden_size": 2, "feature_family": "fam",
                "base_fingerprint": "base"}
    predictor = load(tmp_path, contract)
    assert predictor.predict(np.asarray([[[1.0, 2.0]], [[2.0, 3.0]]])).tolist() == [8.0, 11.0]


def test_online_run_work_items_assign_one_dycon_and_m_baseline_per_task():
    from RL.online_runtime import make_online_run_work_items
    selection = {"task_keys": ["a", "b"], "tasks": {
        "a": {"instance_id": "a"}, "b": {"instance_id": "b"}}}
    manifest = make_online_run_work_items(selection, "run", m=3, base_seed=7, branch_count=4)
    by_task = {}
    for item in manifest["items"]:
        by_task.setdefault(item["task_key"], []).append(item)
    assert all(len(items) == 4 for items in by_task.values())
    assert all(items[0]["online_source"] == "dycon" for items in by_task.values())
    assert all(sum(item["online_source"] == "baseline" for item in items) == 3
               for items in by_task.values())
    assert manifest["schema"] == "online_bootstrap_run_work_items_v1"


def test_online_difficulty_cli_fit_contract(tmp_path):
    import json
    import numpy as np
    from difficulty.training.online import fit_snapshot
    base = "base"
    snapshot = tmp_path / "snapshot.jsonl"
    rows = [{"point_id": "a", "hidden": [0.0, 1.0], "target": 1.0,
             "model_fingerprint": "S0", "feature_family": "fam",
             "base_fingerprint": base, "tokenizer_fingerprint": "tok",
             "site": "layer_0", "width": 2, "dtype": "bf16"},
            {"point_id": "b", "hidden": [1.0, 2.0], "target": 2.0,
             "model_fingerprint": "S1", "feature_family": "fam",
             "base_fingerprint": base, "tokenizer_fingerprint": "tok",
             "site": "layer_0", "width": 2, "dtype": "bf16"}]
    snapshot.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "estimator.npz"
    result = fit_snapshot({"model": {"fingerprint": base}}, snapshot, output,
                          expected_feature_family="fam")
    assert result["count"] == 2 and output.is_file()
    with np.load(output, allow_pickle=False) as archive:
        assert archive["coef"].shape == (2,)


def test_online_rollout_server_validates_and_caches_request(tmp_path):
    from RL.online_server import OnlineRolloutService, request_digest
    calls = []
    def backend(payload):
        calls.append(payload["request_sha256"])
        rows = [{"row_id": "d", "task_id": "a", "source": "dycon", "group_id": "dg",
                 "input_token_ids": [1, 2], "completion_token_ids": [2],
                 "rollout_logprobs": [-1.0], "advantage": 1.0, "trajectory_id": "a:dycon"},
                {"row_id": "b", "task_id": "a", "source": "baseline", "group_id": "bg",
                 "input_token_ids": [1, 2], "completion_token_ids": [2],
                 "rollout_logprobs": [-1.0], "advantage": -1.0, "resolved": False, "trajectory_id": "a:baseline:0",
                 "agent_step": 0, "trajectory_token_count": 1}]
        return {"schema": "online_bootstrap_rollout_batch_v1", "step": payload["step"],
                "tasks": payload["tasks"], "policy_version": "S0", "estimator_version": "E0",
                "rows": rows, "request_sha256": payload["request_sha256"]}
    payload = {"schema": "online_bootstrap_rollout_batch_v1", "step": 0, "tasks": ["a"],
               "policy_version": "S0", "estimator_version": "E0", "worker_ids": ["shared"],
               "work_items": [{"trajectory_id": "a:dycon", "task_id": "a", "source": "dycon", "worker_id": "shared"},
                              {"trajectory_id": "a:baseline:0", "task_id": "a", "source": "baseline", "worker_id": "shared"}]}
    payload["request_sha256"] = request_digest(payload)
    service = OnlineRolloutService(backend, tmp_path / "receipts")
    first = service.handle(payload)
    second = service.handle(payload)
    assert first["request_sha256"] == payload["request_sha256"]
    assert second == first and calls == [payload["request_sha256"]]

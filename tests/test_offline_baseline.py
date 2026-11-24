from dataclasses import asdict
import json

import pytest

from core.io import sha256, write_json
from RL.offline_baseline import BaselinePool
from RL.online_runtime import OnlineRow, SourceMass, online_row_masses


def test_task_cap_and_failure_priority_survive_json_resume(tmp_path):
    directory = tmp_path / "trajectories"
    directory.mkdir()
    entries = []
    for index, (task, failure) in enumerate(
        [("a", None), ("a", None), ("a", "reasoning_loop"), ("b", None)]
    ):
        key = str(index)
        row = OnlineRow(key, task, "baseline", task, (1, 2), (2,), (-1.,),
                        None, -3. if failure else -1., trajectory_id=key,
                        trajectory_token_count=1, agent_step=0, resolved=False,
                        reward_reason=failure)
        shard = directory / (key + ".json")
        write_json(shard, [asdict(row)])
        entries.append({"id": key, "task": task, "failure": failure, "sha256": sha256(shard)})
    write_json(tmp_path / "manifest.json",
               {"schema_version": 1, "probability": "raw_t1", "entries": entries})
    pool = BaselinePool(tmp_path, seed=7, n=2, m=1, coverage_window=2,
                        targets={"reasoning_loop": 1, "LimitsExceeded": 1})
    rows, cursor, _ = pool.batch({})
    assert {row.trajectory_id for row in rows} == {"2", "3"}
    assert online_row_masses(rows, SourceMass(0., 1.)) == [.5, .5]
    recovered = json.loads(json.dumps(cursor))
    assert pool.batch(recovered) == pool.batch(cursor)
    next_rows, _, metrics = pool.batch(recovered)
    assert next(row for row in next_rows if row.task_id == "a").trajectory_id in {"0", "1"}
    assert metrics["coverage_shortfall"] == {"reasoning_loop": 0, "LimitsExceeded": 1}


@pytest.mark.parametrize("advantage_mode", ["direct", "task_grpo"])
def test_mixed_updates_match_weighted_oracle_and_resume(tmp_path, advantage_mode):
    import copy
    from types import SimpleNamespace as NS

    import torch
    from RL.config import OfflineBaseline
    from RL.grpo.objective import ObjectiveResult
    from RL.offline_baseline import binding, build_updater, prepare_task_reward_stats
    from RL.online_runtime import collate_rows, online_selected_logprobs, online_torch_objective
    from RL.training import train_grpo

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(5, 4)
            self.head = torch.nn.Linear(4, 5)

        def forward(self, input_ids, attention_mask):
            return NS(logits=self.head(self.embedding(input_ids).cumsum(1)))

    class Objective:
        settings = NS(kl_beta=.2)

        def iter_results(self, model, group):
            loss = model.embedding.weight.square().mean()
            yield ObjectiveResult(loss, .8 * loss, loss, {"policy_loss": float(loss.detach())})

    path = tmp_path / "trajectories"
    path.mkdir()
    row = OnlineRow("row", "task", "baseline", "task", (1, 2, 3), (3,), (-1.5,),
                    None, 1., trajectory_id="traj", trajectory_token_count=1, agent_step=0, resolved=True)
    write_json(path / "traj.json", [asdict(row)])
    entry = {"id": "traj", "task": "task", "failure": None, "model": "model", "tokenizer": "tokenizer"}
    entries = [entry]
    if advantage_mode == "task_grpo":
        from dataclasses import replace
        failed = replace(row, row_id="limit", trajectory_id="limit", advantage=-3.,
                         resolved=False, reward_reason="LimitsExceeded")
        write_json(path / "limit.json", [asdict(failed)])
        entries.append({**entry, "id": "limit", "failure": "LimitsExceeded", "run": "other"})
    write_json(tmp_path / "manifest.json", {"schema_version": 1, "probability": "raw_t1", "entries": entries})
    settings = OfflineBaseline(dataset=str(tmp_path), dycon_weight=.25, baseline_weight=.75, kl_beta=0.,
                               attribution_every=1, trajectories_per_task=1, advantage_mode=advantage_mode)
    if advantage_mode == "task_grpo":
        prepare_task_reward_stats(tmp_path)
    config = NS(offline_baseline=settings, training=NS(), grpo=NS(kl_beta=.2))
    identity = {"model_fingerprint": "model", "tokenizer_fingerprint": "tokenizer"}
    contract = {"grpo": {"reference_identity": identity},
                "offline_baseline": binding(settings, "model", "tokenizer")}
    groups = [NS(group_id=str(i), group_size=1, loss_mask=torch.ones(1, 1, dtype=torch.bool)) for i in range(3)]
    torch.manual_seed(42)
    model = Model()
    oracle = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, .9)
    baseline = build_updater(config, model, tmp_path, contract, optimizer, scheduler)

    metrics = []

    def train(stop, state=None):
        return train_grpo(model, groups, Objective(), optimizer, scheduler, group_batch_size=1,
                          grad_clip=100., max_updates=stop, state=state, baseline=baseline,
                          on_update=lambda _, item: metrics.append(item))

    state = train(1)
    state = train(3, json.loads(json.dumps(state)))
    assert state["global_step"] == 3
    assert scheduler.last_epoch == 3
    assert metrics[0]["offline_baseline"]["gradient_attribution"]["components"]["kl"]["norm"] > 0
    assert metrics[0]["offline_baseline"]["advantage"]["mode"] == advantage_mode
    oracle_opt = torch.optim.SGD(oracle.parameters(), lr=.01)
    oracle_sched = torch.optim.lr_scheduler.StepLR(oracle_opt, 1, .9)
    tensors = collate_rows([row], device=torch.device("cpu"))
    cursor = {}
    oracle_pool = BaselinePool(tmp_path, seed=0, n=1, m=1)
    for _ in range(3):
        oracle_opt.zero_grad()
        selected, cursor, _ = oracle_pool.batch(cursor)
        raw = selected[0].advantage
        advantage = raw if advantage_mode == "direct" else (raw + 1.) / (2. + 1e-8)
        current, mask = online_selected_logprobs(oracle, tensors)
        loss, _ = online_torch_objective(current, tensors["rollout_logprobs"], None,
            torch.tensor([advantage]), mask, ["baseline"], {"baseline": [[0]]},
            source_mass=SourceMass(0., 1.), kl_beta=0.)
        (.25 * oracle.embedding.weight.square().mean() + .75 * loss).backward()
        oracle_opt.step()
        oracle_sched.step()
    for actual, expected in zip(model.parameters(), oracle.parameters()):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)


def test_dataset_task_grpo_reuses_full_snapshot_statistics(tmp_path, monkeypatch):
    import math

    from RL.artifacts import resume_contract_matches
    from RL.config import OfflineBaseline
    from RL.offline_baseline import (_task_grpo_rows, _encode_rows, binding,
                                     prepare_task_reward_stats)
    from RL.online_trace import build_training_traces

    shard_dir = tmp_path / "trajectories"
    shard_dir.mkdir()
    entries = []
    for task, key, reward, steps, schema, metadata, run in [
        ("a", "a1", 1., 2, 1, False, "r1"),
        ("a", "a2", 1., 1, 2, True, "r2"),
        ("a", "a3", -1., 1, 2, False, "r2"),
        ("a", "a4", -3., 3, 1, False, "r3"),
        ("b", "b1", -1., 1, 1, False, "r1"),
        ("c", "c1", -3., 1, 1, False, "r1"),
        ("c", "c2", -3., 1, 2, True, "r2"),
    ]:
        failure = "LimitsExceeded" if reward == -3. else None
        trajectory = tuple(OnlineRow(f"{key}-{i}", task, "baseline", task,
            (1, *range(2, i + 3)), (i + 2,), (-1.,), None, reward,
            trajectory_id=key, trajectory_token_count=steps, agent_step=i,
            resolved=reward > 0, reward_reason=failure) for i in range(steps))
        shard = shard_dir / f"{key}.json"
        write_json(shard, _encode_rows(trajectory, steps) if schema == 2 else [asdict(r) for r in trajectory])
        entry = {"task": task, "id": key, "tokenizer": "tok", "run": run,
                 "failure": failure, "schema_version": schema, "bytes": shard.stat().st_size}
        if metadata:
            entry["raw_reward"] = reward
        entries.append(entry)
    manifest = {"schema_version": 1, "probability": "raw_t1", "entries": entries}
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    settings = OfflineBaseline(dataset=str(tmp_path), advantage_mode="task_grpo")
    with pytest.raises(ValueError, match="rl-prepare-baseline-stats"):
        binding(settings, "base", "tok")
    stats = prepare_task_reward_stats(tmp_path)
    assert stats["tasks"]["a"]["histogram"] == [[-3., 1], [-1., 1], [1., 2]]
    assert stats["tasks"]["a"]["mean"] == -.5
    assert stats["tasks"]["a"]["std"] == pytest.approx(math.sqrt(2.75))
    direct = binding(OfflineBaseline(dataset=str(tmp_path)), "base", "tok")
    assert "advantage" not in direct and "advantage_mode" not in direct["config"]
    normalized = binding(settings, "base", "tok")
    assert resume_contract_matches({"input": normalized}, {"input": normalized})
    assert not resume_contract_matches({"input": direct}, {"input": normalized})

    pool = BaselinePool(tmp_path, seed=0, n=3, m=1, targets={"LimitsExceeded": 3})
    rows, cursor, _ = pool.batch({})
    assert pool.batch(json.loads(json.dumps(cursor))) == pool.batch(cursor)
    all_rows, _, _ = BaselinePool(tmp_path, seed=0, n=3, m=8).batch({})
    selected, metrics = _task_grpo_rows(rows, stats)
    expanded, _ = _task_grpo_rows(all_rows, stats)
    by_row = {row.row_id: row.advantage for row in expanded}
    assert {row.trajectory_id for row in selected} & {"a1", "a2", "a3", "a4"} == {"a4"}
    assert [row.advantage for row in selected if row.trajectory_id == "a4"] == pytest.approx(
        [-2.5 / (math.sqrt(2.75) + 1e-8)] * 3)
    assert all(row.advantage == by_row[row.row_id] for row in selected)
    assert all(row.advantage == 0 for row in expanded if row.task_id in {"b", "c"})
    assert metrics["zero_advantage_trajectories"] == 2
    from types import SimpleNamespace as NS
    from RL.offline_baseline import _active_task_ids, schedule

    eligible = _active_task_ids(stats)
    assert eligible == {"a"}
    filtered, _, _ = BaselinePool(tmp_path, seed=0, n=3, m=1, eligible_tasks=eligible).batch({})
    assert {row.task_id for row in filtered} == {"a"}
    assert schedule(NS(offline_baseline=settings.model_copy(update={"tasks_per_update": 1}),
                       training=NS(max_updates=1))) == {
        "tasks": 1, "updates_per_epoch": 1, "epochs": 1, "max_updates": 1,
        "tasks_skipped_zero_advantage": 2,
    }
    masses = online_row_masses(selected, SourceMass(0., 1.))
    traces, coefficients, _ = build_training_traces(selected, masses, max_completion_tokens=2)
    assert sum(c for t, c in zip(traces, coefficients) if t.trajectory_id == "a4") == pytest.approx(1/3)

    with monkeypatch.context() as patch:
        def forbid_scan(*_):
            pytest.fail("The existing snapshot statistics must not reread trajectory shards")
        patch.setattr("RL.offline_baseline._read_shard", forbid_scan)
        assert prepare_task_reward_stats(tmp_path) == stats
        assert binding(settings, "base", "tok") == normalized

    write_json(manifest_path, {**manifest, "entries": entries[1:]})
    with pytest.raises(ValueError, match="do not match this dataset snapshot"):
        binding(settings, "base", "tok")
    assert prepare_task_reward_stats(tmp_path)["tasks"]["a"]["count"] == 3
    assert not resume_contract_matches({"input": normalized}, {"input": binding(settings, "base", "tok")})


def test_noncanonical_import_only_keeps_two_failure_types(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from RL.offline_baseline import prepare

    run = tmp_path / "run"
    item = {"instance_id": "task", "repetition_id": "rep", "sample_key": "sample"}
    cfg = {"paths": {"nas_root": str(tmp_path)}, "capture": {"export": {"root": "native"}},
           "serving": {}, "model": {"fingerprint": "model"}}
    monkeypatch.setattr("application.execution.run_plan.load_plan",
                        lambda *_: {"config": cfg, "run": run, "work_items": {"items": [item]}})
    data = {"rollouts": {}, "tasks": {"task": {"instance_id": "task", "partition": "train"}},
            "evaluations": {"eval": {"outcomes": {"accepted": {"official_result": {"resolved": True}}}}}}
    for rid, status in [("accepted", "Submitted"), ("limit", "LimitsExceeded"),
                        ("loop", "reasoning_loop"), ("other", "TimeExceeded")]:
        path = run / rid
        path.parent.mkdir(exist_ok=True)
        write_json(path, {"messages": [{"extra": {"response": {"id": rid}}}]})
        data["rollouts"][rid] = {"task_key": "task", "sample_key": "sample", "attempt_id": rid,
                                 "selected_evaluation_id": "eval",
                                 "trajectory_path": str(path), "trajectory_sha256": sha256(path)}
        attempt = run / "rollout/tasks/task/repetitions/rep/attempts" / rid / "attempt.json"
        attempt.parent.mkdir(parents=True)
        write_json(attempt, {"agent_exit_status": status, "rollout_id": rid,
                             "sample_key": "sample", "attempt_id": rid})
    monkeypatch.setattr("application.execution.evidence.RunEvidence", lambda *_: NS(
        verify=lambda: None, read=lambda: data, select=lambda: {
            "rollouts": {"accepted": data["rollouts"]["accepted"]}, "evidence_sha256": "evidence"}))
    monkeypatch.setattr("RL.online_evidence.native_completion", lambda _root, response, *_, **_kwargs: (
        {"request_id": response["id"], "prompt_token_ids": [1], "output_token_ids": [2],
         "tokenizer_fingerprint": "tokenizer"}, (-1.,)))
    monkeypatch.setattr("RL.offline_baseline.default_final_ids", lambda *_, **_kwargs: ["accepted"])
    dest = tmp_path / "dataset"
    manifest = prepare(tmp_path, ["run"], dest, -3.)
    assert {e["rollout"]: e["raw_reward"] for e in manifest["entries"]} == {
        "accepted": 1., "limit": -3., "loop": -3.}
    rows, _, _ = BaselinePool(dest, seed=0, n=1, m=8).batch({})
    assert {r.row_id: r.advantage for r in rows} == {"accepted": 1., "limit": -3., "loop": -3.}


def test_baseline_only_schedule_and_real_optimizer_steps(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    import torch
    from RL.config import OfflineBaseline, Training
    from RL.offline_baseline import schedule
    from RL.offline_baseline_train import execute

    data = tmp_path / "data"
    (data / "trajectories").mkdir(parents=True)
    entries = []
    for i in range(3):
        row = OnlineRow(str(i), str(i), "baseline", str(i), (1, 2, 3), (3,), (-1.5,),
                        None, 1., trajectory_id=str(i), trajectory_token_count=1,
                        agent_step=0, resolved=True)
        write_json(data / "trajectories" / f"{i}.json", [asdict(row)])
        entries.append({"id": str(i), "task": str(i), "failure": None,
                        "model": "historical-checkpoint", "tokenizer": "tokenizer"})
    write_json(data / "manifest.json",
               {"schema_version": 1, "probability": "raw_t1", "entries": entries})
    config = NS(offline_baseline=OfflineBaseline(dataset=str(data), dycon_weight=0,
                baseline_weight=1, kl_beta=0, tasks_per_update=2, epochs=2),
                training=Training(max_updates=4, gradient_accumulation_steps=1,
                    learning_rate=0.001, weight_decay=0, grad_clip=1,
                    scheduler="constant", checkpoint_every_updates=2),
                validation=NS(stop_after_update=None), run=NS(start=NS(mode="fresh"),
                    attempt_id="baseline-test"), grpo=None, lora={})
    assert schedule(config) == {"tasks": 3, "updates_per_epoch": 2, "epochs": 2, "max_updates": 4}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(5, 4)
            self.head = torch.nn.Linear(4, 5)

        def forward(self, input_ids, attention_mask):
            return NS(logits=self.head(self.embedding(input_ids).cumsum(1)))

    checkpoints = []
    def export(_model, directory, _identity):
        directory.mkdir(parents=True)
        (directory / "test-reference").write_text("initial")
    monkeypatch.setattr("RL.checkpoints.export_adapter", export)
    monkeypatch.setattr("RL.checkpoints.save_checkpoint",
                        lambda _m, _o, _s, state, *a, **kw: checkpoints.append(json.loads(json.dumps(state))))
    monkeypatch.setattr("RL.artifacts.commit_receipt", lambda *a, **kw: None)
    model = Model()
    before = model.embedding.weight.detach().clone()
    directory = tmp_path / "run"
    directory.mkdir()
    result = execute(config, directory, {"fingerprint": "model"}, None, {}, model, [], None)
    assert result["state"]["global_step"] == 4
    assert result["state"]["baseline_cursor"]["cycle"] == 2
    assert result["state"]["baseline_cursor"]["offset"] == 0
    assert [s["global_step"] for s in checkpoints] == [2, 4]
    assert not torch.equal(before, model.embedding.weight)
    assert result["state"]["consumed_tokens"] == 6


def test_baseline_only_preflight_never_opens_dycon(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from RL.config import OfflineBaseline
    from RL.service import _preflight
    write_json(tmp_path / "manifest.json", {"entries": [
        {"task": "a"}, {"task": "b"}, {"task": "c"}]})
    config = NS(algorithm=NS(name="offline_baseline_grpo_v1"), tuning=None,
                offline_baseline=OfflineBaseline(dataset=str(tmp_path), dycon_weight=0,
                    baseline_weight=1, tasks_per_update=2, epochs=3),
                training=NS(max_updates=6), model=NS(path="unused"),
                preflight=NS(verify_identity=False), run=NS(start=NS(mode="fresh")))
    monkeypatch.setattr("RL.service.base_identity", lambda *a: {"fingerprint": "base"})
    assert _preflight(config, tmp_path, None) == {"fingerprint": "base"}
    config.training.max_updates = 5
    import pytest
    with pytest.raises(ValueError, match="= 6"):
        _preflight(config, tmp_path, None)


def _task_grpo_rank(rank, store, dataset):
    from pathlib import Path
    from types import SimpleNamespace as NS

    import torch
    import torch.distributed as dist
    from RL.config import OfflineBaseline
    from RL.offline_baseline import binding, build_updater
    from RL.parallel import GroupParallel

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 3)
            self.head = torch.nn.Linear(3, 8)

        def forward(self, input_ids, attention_mask):
            return NS(logits=self.head(self.embed(input_ids).cumsum(1)))

    def compute(context, limit):
        torch.manual_seed(727)
        model = Model()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        settings = OfflineBaseline(dataset=dataset, advantage_mode="task_grpo", dycon_weight=0.,
            baseline_weight=1., kl_beta=0., tasks_per_update=2, trajectories_per_task=1,
            limits_exceeded_target=1, max_trace_completion_tokens=limit)
        identity = {"model_fingerprint": "base", "tokenizer_fingerprint": "tok"}
        contract = {"grpo": {"reference_identity": identity},
                    "offline_baseline": binding(settings, "base", "tok")}
        updater = build_updater(NS(offline_baseline=settings, training=NS(), grpo=None),
                                model, dataset, contract, optimizer, scheduler)
        metrics, cursor, _ = updater({"global_step": 0}, context)
        params = list(model.parameters())
        loss = metrics["loss"]
        if context:
            context.reduce_gradients(params)
            loss = sum(context.gather(loss))
        return [p.grad for p in params], loss, cursor, metrics

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(store).as_uri(), rank=rank, world_size=2)
    try:
        expected = compute(None, 1)
        merged = compute(None, 2)
        actual = compute(GroupParallel(rank, 2, data_group=dist.group.WORLD), 2)
        assert expected[3]["row_count"] == 3 and actual[3]["row_count"] == 2
        assert actual[3]["token_count"] > 0
        assert actual[3]["advantage"]["advantage_max"] < 0
        assert actual[3]["advantage"]["dataset_tasks_skipped_zero_advantage"] == 1
        assert any(gradient.abs().max() > 0 for gradient in expected[0])
        for gradients, loss, cursor, _ in (merged, actual):
            assert loss == pytest.approx(expected[1], abs=1e-7, rel=2e-5)
            assert cursor == expected[2]
            for got, want in zip(gradients, expected[0]):
                torch.testing.assert_close(got, want, rtol=2e-5, atol=1e-7)
    finally:
        dist.destroy_process_group()


def test_dataset_task_grpo_gradients_survive_trace_merging_and_two_ranks(tmp_path):
    import torch.multiprocessing as mp
    from RL.offline_baseline import prepare_task_reward_stats

    shard_dir = tmp_path / "trajectories"
    shard_dir.mkdir()
    entries = []
    for key, task, reward, reason, count in [
        ("limit", "task", -3., "LimitsExceeded", 3),
        ("resolved", "task", 1., None, 1),
        ("skip", "zero", -1., None, 1),
    ]:
        rows = [OnlineRow(f"{key}-{i}", task, "baseline", task,
                          (1, *range(2, i + 3)), (i + 2,), (-5.,), None, reward,
                          trajectory_id=key, trajectory_token_count=count, agent_step=i,
                          resolved=reward > 0, reward_reason=reason)
                for i in range(count)]
        # The skipped task needs no shard read, even if it would fit in the batch.
        if key != "skip":
            write_json(shard_dir / f"{key}.json", [asdict(row) for row in rows])
        entries.append({"id": key, "task": task, "failure": reason, "tokenizer": "tok",
                        "raw_reward": reward})
    write_json(tmp_path / "manifest.json", {"schema_version": 1, "probability": "raw_t1", "entries": entries})
    prepare_task_reward_stats(tmp_path)
    mp.spawn(_task_grpo_rank, args=(str(tmp_path / "store"), str(tmp_path)), nprocs=2, join=True)


def test_trace_progress_logs_phase_and_identity(tmp_path):
    from RL.trace_logging import TraceLogger
    log = TraceLogger(tmp_path, 2)
    log("online_reference_forward", trajectory_id="trace", completion_tokens=12)
    log("online_policy_forward")
    rows = [json.loads(line) for line in (tmp_path / "trace-events.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[1]["previous_phase"] == "online_reference_forward"
    assert rows[1]["previous_phase_seconds"] >= 0
    assert rows[1]["trajectory_id"] == "trace"
    assert rows[1]["rank"] == 2
    assert json.loads((tmp_path / "trace-progress.json").read_text()) == rows[1]


@pytest.mark.parametrize("local_failure", [True, False])
def test_baseline_oom_replays_uncommitted_update(tmp_path, local_failure):
    from types import SimpleNamespace as NS
    import torch
    from RL.config import Training
    from RL.memory import MemoryController
    from RL.offline_baseline_train import _compute_update
    from RL.trace_logging import TraceLogger

    settings = Training(max_updates=2, gradient_accumulation_steps=1, learning_rate=0.1,
                        weight_decay=0, grad_clip=1, scheduler="constant", checkpoint_every_updates=1,
                        memory={"oom_retry": {"enabled": True, "max_retries": 1}})
    config = NS(training=settings, run=NS(attempt_id="retry-test"))
    memory = MemoryController(config, tmp_path)
    trace = TraceLogger(tmp_path, 0, memory)
    # This CPU test has no real activation pool. The normal CUDA wiring always
    # provisions one when oom_retry is enabled.
    memory.retry = lambda *a, **kw: None
    model = torch.nn.Linear(1, 1, bias=False)
    model.register_buffer("marker", torch.zeros(1))
    optimizer = torch.optim.AdamW(model.parameters())
    state = {"global_step": 30, "baseline_cursor": {"cycle": 0, "offset": 60}}
    calls, controls = [], []

    class Peer:
        rank = 0

        def control_step(self, step, attempt, phase, payload):
            controls.append((step, attempt, phase))
            if phase == "baseline_compute_result" and attempt == 0 and not local_failure:
                return [payload, {"state": "CUDA_OOM", "rank": 1, "error": "injected"}]
            return [payload, {"state": "OK"}]

    def accumulate(before, parallel, *, oom_replay):
        assert before == state
        assert model.marker.item() == 0
        assert model.weight.grad is None
        value = torch.rand(())
        calls.append((oom_replay, value))
        model.marker.add_(1)
        (model.weight * (2 if oom_replay else 11)).sum().backward()
        if local_failure and not oom_replay:
            raise torch.cuda.OutOfMemoryError("injected")
        return {"token_count": 4}, {"cycle": 0, "offset": 62}, None

    metrics, cursor, _, retries = _compute_update(
        model, optimizer, accumulate, state, Peer(), memory, trace)
    assert metrics == {"token_count": 4}
    assert cursor == {"cycle": 0, "offset": 62}
    assert state == {"global_step": 30, "baseline_cursor": {"cycle": 0, "offset": 60}}
    assert not optimizer.state
    assert model.weight.grad.item() == 2 and retries == 1
    assert [mode for mode, _ in calls] == [False, True]
    assert torch.equal(calls[0][1], calls[1][1])
    assert controls == [(31, 0, "baseline_compute_result"), (31, 0, "baseline_cleanup"),
                        (31, 1, "baseline_compute_result")]

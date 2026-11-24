"""Native group import, reward isolation and differentiable common history."""

import copy
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from core.io import hash_json, read_json, sha256, write_json
from RL.key_steps import ParentPool, binding, prepare, read_member, scores


def seal(path, value):
    value = {**value, "sha256": hash_json(value)}
    write_json(path, value)
    return value


def native(root, name, prompt, completion, branch=0, step=0, *, model_fingerprint="model"):
    import numpy as np

    root.mkdir(parents=True, exist_ok=True)
    values = np.full(len(completion), -2.0, dtype="<f4")
    values.tofile(root / (name + ".f32"))
    record = {
        "request_id": name,
        "run_id": "run",
        "session_id": "session",
        "decision_id": "decision",
        "branch_id": branch,
        "step_id": step,
        "api_attempt_id": "attempt",
        "prompt_token_ids": prompt,
        "output_token_ids": completion,
        "tokenizer_fingerprint": "tokenizer",
    }
    record["raw_t1_probability"] = {
        "probability_semantics": "raw_t1_sampled_token_v1",
        "normalization_temperature": 1.0,
        "sampling_transforms_applied": False,
        "target_only": True,
        "values_dtype": "float32",
        "status": "sealed",
        "locator": name + ".f32",
        "bytes": len(completion) * 4,
        "count": len(completion),
        "sha256": sha256(root / (name + ".f32")),
        "output_token_ids_sha256": hash_json(completion),
        "prompt_token_ids_sha256": hash_json(prompt),
        "tokenizer_fingerprint": "tokenizer",
        "source_identity": {"model_fingerprint": model_fingerprint},
        "identity": {
            k: record[k]
            for k in (
                "request_id",
                "run_id",
                "session_id",
                "decision_id",
                "branch_id",
                "step_id",
                "api_attempt_id",
            )
        },
        "negative_infinity_count": 0,
        "finite_count": len(completion),
    }
    write_json(root / (name + ".json"), record)
    return {
        "id": name,
        "dycon_native_request_id": name,
        "native_request_id": name,
        "ledger_root": str(root),
        "ledger_ref": name + ".json",
        "ledger_sha256": sha256(root / (name + ".json")),
    }


def collection(tmp_path, parents=2, *, model_fingerprint="model"):
    def make_native(*args, **kwargs):
        return native(*args, model_fingerprint=model_fingerprint, **kwargs)

    groups = []
    for parent in range(parents):
        pid = f"p-{parent}"
        source_dir = tmp_path / pid
        prefix_native = make_native(source_dir, "prefix", [1, 2], [3])
        mother = {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "extra": {"response": prefix_native}},
                {"role": "tool", "content": "obs"},
                {"role": "user", "content": "FUTURE_MUST_NOT_ENTER_CHILD"},
            ]
        }
        write_json(source_dir / "mother.json", mother)
        source = {
            "rollout_id": f"mother-{parent}",
            "trajectory_path": str(source_dir / "mother.json"),
            "trajectory_sha256": sha256(source_dir / "mother.json"),
            "trajectory_bytes": (source_dir / "mother.json").stat().st_size,
        }
        candidates = []
        for b in range(8):
            completion = [5 + b % 3] * (b + 1 if parent == 0 else 2)
            first = make_native(source_dir, f"first-{b}", [1, 2, 3, 4], completion, b, 1)
            candidates.append({**first, "branch_id": b, "response": {"id": first["id"]}})
        decision = {"candidates": candidates}
        write_json(source_dir / "decision.json", decision)
        members = []
        for b, candidate in enumerate(candidates):
            cid = f"c-{parent}-{b}"
            child = {
                "source": source,
                "child_id": cid,
                "parent_id": pid,
                "instance_id": "org__task",
                "step": 1,
                "branch": b,
                "message_index": 4,
                "decision_path": str(source_dir / "decision.json"),
                "decision_sha256": hash_json(decision),
                "ledger_root": str(source_dir),
                "source_native_request_id": candidate["native_request_id"],
            }
            first = read_json(source_dir / candidate["ledger_ref"])
            second = make_native(
                source_dir,
                f"suffix-{b}",
                first["prompt_token_ids"] + first["output_token_ids"] + [9],
                [10, 11],
                0,
                2,
            )
            traj = {
                "continuation": {
                    "child_id": cid,
                    "parent_id": pid,
                    "source_step": 1,
                    "prefix_message_count": 4,
                    "trajectory_sha256": source["trajectory_sha256"],
                },
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "response": {"id": candidate["native_request_id"]},
                            "continuation_source": {
                                "child_id": cid,
                                "native_request_id": candidate["native_request_id"],
                            },
                        },
                    },
                    {"role": "tool", "content": "not supervised"},
                    {"role": "assistant", "extra": {"response": second}},
                ],
            }
            if parent == 0 and b == 7:
                traj["info"] = {"loop_detection": {"reason": "repeated_reasoning_action_loop"}}
            write_json(source_dir / (cid + ".json"), traj)
            write_json(source_dir / (cid + "-result.json"), {"disposition": "accepted"})
            members.append(
                {
                    "child_id": cid,
                    "branch_id": b,
                    "source_ref": child,
                    "trajectory_ref": str(source_dir / (cid + ".json")),
                    "trajectory_sha256": sha256(source_dir / (cid + ".json")),
                    "result_ref": str(source_dir / (cid + "-result.json")),
                    "reward": 1 if b < 3 or parent == 1 else 0,
                    "terminal_status": "LimitsExceeded" if b == 6 else "Submitted",
                }
            )
        groups.append({"parent_id": pid, "members": members})
    return seal(
        tmp_path / "groups.json",
        {
            "schema": "continuation-groups-v1",
            "branch_count": 8,
            "project_root": str(tmp_path),
            "methods": {m: [g["parent_id"] for g in groups] for m in ["adaptive", "random", "entropy"]},
            "groups": groups,
        },
    )


def test_reward_range_linear_resolved_only_ties_and_infrastructure():
    rows = [
        {"outcome": o, "total_generated_tokens": n}
        for o, n in [
            ("resolved", 100),
            ("resolved", 150),
            ("resolved", 300),
            ("loop", 1),
            ("unresolved", 500),
            ("infrastructure", 2),
        ]
    ]
    scored, stats = scores(rows)
    assert [r["reward"] for r in scored] == [3.0, 2.5, 1.0, -3.0, -1.0, None]
    assert not stats["complete"] and all(r["advantage"] is None for r in scored)
    equal, _ = scores([rows[0], rows[0]])
    assert [r["reward"] for r in equal] == [2.0, 2.0]
    single, _ = scores([rows[0], rows[3]])
    assert [r["reward"] for r in single] == [3.0, -3.0]


def test_native_import_groups_original_branches_and_counts_prefix_once(tmp_path):
    source = collection(tmp_path)
    report = prepare(tmp_path / "groups.json", tmp_path / "dataset", "adaptive")
    value = read_json(tmp_path / "dataset/manifest.json")
    first, zero = value["entries"]
    assert report["parents"] == 2 and report["train_eligible"] == 1
    assert [r["branch_id"] for r in first["members"]] == list(range(8))
    assert [r["reward"] for r in first["members"]] == [3.0, 2.0, 1.0, -1.0, -1.0, -1.0, -1.0, -3.0]
    assert first["members"][0]["prefix_generated_tokens"] == 1
    assert first["members"][0]["total_generated_tokens"] == 4  # 1 prefix + 1 candidate + 2 suffix
    assert zero["zero_variance"] and not zero["train_eligible"]
    compressed = read_json(tmp_path / "dataset/trajectories/c-0-0.json")
    assert compressed[0]["prefix_tokens"] == 4
    assert compressed[0]["input_token_ids"] == [5]  # public prompt stored only in parent prefix file
    turns, record = read_member(source["groups"][0]["members"][6], tmp_path)
    assert record["outcome"] == "unresolved"  # generic LimitsExceeded is not a loop
    assert len(turns) == 2 and turns[0].input_token_ids[:4] == (1, 2, 3, 4)


def test_group_pool_never_resamples_branches_and_cursor_is_reproducible(tmp_path):
    from RL.config import KeySteps

    collection(tmp_path)
    prepare(tmp_path / "groups.json", tmp_path / "dataset", "adaptive")
    settings = KeySteps(dataset=str(tmp_path / "dataset"), kl_beta=0.0)
    pool = ParentPool(settings)
    rows, masses, cursor, _ = pool.batch({})
    assert len(rows) == 16 and len({r.trajectory_id for r in rows}) == 8
    assert sum(masses) == pytest.approx(1.0)
    for child in {r.trajectory_id for r in rows}:
        assert sum(m for r, m in zip(rows, masses) if r.trajectory_id == child) == pytest.approx(1 / 8)
    assert cursor == {"cycle": 1, "offset": 0}
    assert pool.batch(cursor) == ParentPool(settings).batch(cursor)
    path = tmp_path / "dataset/trajectories/c-0-0.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="byte count|evidence changed"):
        pool.batch(cursor)


def test_missing_child_excludes_whole_mother_but_retains_audit(tmp_path):
    source = collection(tmp_path)
    Path(source["groups"][0]["members"][0]["trajectory_ref"]).unlink()
    report = prepare(tmp_path / "groups.json", tmp_path / "dataset", "random")
    assert report["excluded"] == 1
    value = read_json(tmp_path / "dataset/manifest.json")
    assert value["failed_mothers"] == ["mother-0"]
    assert all(e["id"] != "p-0" for e in value["entries"])


def test_gradient_matches_per_turn_oracle_prefix_has_no_labels_but_nonzero_gradient(tmp_path):
    import torch

    from RL.config import KeySteps, Training
    from RL.key_steps import build_updater
    from RL.online_runtime import SourceMass, collate_rows, online_selected_logprobs, online_torch_objective
    from RL.online_trace import build_training_traces

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 5)
            self.head = torch.nn.Linear(5, 16)

        def forward(self, input_ids, attention_mask):
            return NS(logits=self.head(self.embedding(input_ids).cumsum(1)))

    collection(tmp_path)
    prepare(tmp_path / "groups.json", tmp_path / "dataset", "adaptive")
    settings = KeySteps(dataset=str(tmp_path / "dataset"), kl_beta=0.0)
    config = NS(
        key_steps=settings,
        training=Training(
            max_updates=1,
            gradient_accumulation_steps=1,
            learning_rate=0.001,
            weight_decay=0.0,
            grad_clip=1.0,
            scheduler="constant",
            checkpoint_every_updates=1,
        ),
        objective=None,
        model=NS(attention_backend="sdpa"),
    )
    torch.manual_seed(20)
    model = Model()
    oracle = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1)
    contract = {
        "grpo": {"reference_identity": {"model_fingerprint": "model", "tokenizer_fingerprint": "tokenizer"}},
        "offline_baseline": binding(settings, "model", "tokenizer"),
    }
    accumulate = build_updater(config, model, tmp_path, contract, opt, sched)
    metrics, cursor, _ = accumulate({"global_step": 0}, None)
    rows, masses, _, _ = ParentPool(settings).batch({})
    loss = 0
    for row, mass in zip(rows, masses):
        tensors = collate_rows([row], device=torch.device("cpu"))
        assert not tensors["loss_mask"][0, :4].any()
        assert (tensors["labels"][0, :4] == -100).all()
        values, mask = online_selected_logprobs(oracle, tensors)
        term, _ = online_torch_objective(
            values,
            tensors["rollout_logprobs"],
            None,
            torch.tensor([row.advantage]),
            mask,
            ["baseline"],
            {"baseline": [[0]]},
            source_mass=SourceMass(0.0, 1.0),
            kl_beta=0.0,
        )
        loss = loss + term * mass
    loss.backward()
    assert metrics["loss"] == pytest.approx(float(loss.detach()), abs=2e-7)
    for p, q in zip(model.parameters(), oracle.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=3e-5, atol=2e-7)
    # Token 1 occurs only in common history, never as a supervised target.
    assert model.embedding.weight.grad[1].norm() > 0
    merged, _, _ = build_training_traces(rows, masses, max_input_tokens=None)
    assert len(merged) == 8 and sum(len(r.completion_token_ids) for r in merged) == sum(
        len(r.completion_token_ids) for r in rows
    )
    for row in merged:
        assert all(p >= 4 for p in row.supervision_positions)


def test_config_formal_algorithm_and_legacy_contract_isolation(tmp_path):
    import yaml

    from RL.config import Config, load, schedule_contract

    collection(tmp_path)
    prepare(tmp_path / "groups.json", tmp_path / "dataset", "adaptive")
    base = yaml.safe_load((Path(__file__).parents[1] / "RL/experiments/lora-smoke.yaml").read_text())
    base["algorithm"] = {"name": "key_step_grpo_v1"}
    base["key_steps"] = {"dataset": str(tmp_path / "dataset"), "epochs": 2}
    base["training"].update(max_updates=None, gradient_accumulation_steps=1)
    config = Config.model_validate(base)
    assert config.training.max_updates == 2
    assert schedule_contract(config)["key_steps"]["epochs"] == 2
    with pytest.raises(ValueError):
        Config.model_validate({**base, "offline_baseline": {"dataset": "bad"}})
    base["key_steps"]["dataset"] = "dataset"
    write_json(tmp_path / "unused.json", {})
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(base))
    loaded, _ = load(tmp_path / "config.yaml", tmp_path)
    assert loaded.key_steps.dataset == str(tmp_path / "dataset")


@pytest.mark.parametrize("beta", [0.0, 0.02])
def test_tiny_qwen_real_updates_checkpoint_and_exact_resume(tmp_path, beta):
    import torch

    from RL.checkpoints import verify_checkpoint
    from RL.config import KeySteps, Training
    from RL.offline_baseline_train import execute
    from tests.lora_helpers import LORA, model_and_targets

    torch.set_num_threads(1)
    collection(tmp_path)
    prepare(tmp_path / "groups.json", tmp_path / "dataset", "adaptive")
    training = Training(
        max_updates=2,
        gradient_accumulation_steps=1,
        learning_rate=0.001,
        weight_decay=0.0,
        grad_clip=1.0,
        scheduler="constant",
        checkpoint_every_updates=1,
    )
    config = NS(
        key_steps=KeySteps(dataset=str(tmp_path / "dataset"), epochs=2, kl_beta=beta),
        offline_baseline=None,
        training=training,
        validation=NS(stop_after_update=None),
        run=NS(start=NS(mode="fresh", extend_baseline_epochs=False), attempt_id="key-step-test"),
        model=NS(attention_backend="sdpa"),
        objective=None,
        lora=LORA,
    )

    def run(model, targets, name, parent=None, stop=None):
        config.validation.stop_after_update = stop
        config.run.start.mode = "resume" if parent else "fresh"
        location = tmp_path / name
        location.mkdir()
        attempt = {"attempt_id": name}
        return execute(config, location, {"fingerprint": "model"}, parent, attempt, model, targets, None)

    model, targets = model_and_targets()
    if os.environ.get("DYCON_KEY_STEP_TEST_GPU") == "1":
        model = model.cuda()
    initial = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}
    result = run(model, targets, "full")
    assert result["state"]["global_step"] == 2
    assert any(not torch.equal(initial[k], v) for k, v in model.named_parameters() if v.requires_grad)
    reference = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}
    first, targets = model_and_targets()
    if os.environ.get("DYCON_KEY_STEP_TEST_GPU") == "1":
        first = first.cuda()
    run(first, targets, "paused", stop=1)
    checkpoint = tmp_path / "paused/checkpoints/step-000001"
    verify_checkpoint(checkpoint)
    restored, targets = model_and_targets(adapter=checkpoint / "adapter")
    if os.environ.get("DYCON_KEY_STEP_TEST_GPU") == "1":
        restored = restored.cuda()
    result = run(restored, targets, "resumed", parent={"path": str(checkpoint)})
    assert result["state"]["baseline_cursor"] == {"cycle": 2, "offset": 0}
    for k, v in restored.named_parameters():
        if v.requires_grad:
            torch.testing.assert_close(v, reference[k], atol=2e-7, rtol=2e-5)


def _key_step_rank(rank, store, data, result):
    import torch
    import torch.distributed as dist

    from RL.config import KeySteps, Training
    from RL.key_steps import build_updater

    torch.set_num_threads(1)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)
            self.head = torch.nn.Linear(4, 16)

        def forward(self, input_ids, attention_mask):
            return NS(logits=self.head(self.embedding(input_ids).cumsum(1)))

    torch.manual_seed(12)
    model = Model()
    settings = KeySteps(dataset=data, kl_beta=0.0)
    config = NS(
        key_steps=settings,
        training=Training(
            max_updates=1,
            gradient_accumulation_steps=1,
            learning_rate=0.001,
            weight_decay=0.0,
            grad_clip=1.0,
            scheduler="constant",
            checkpoint_every_updates=1,
        ),
        objective=None,
        model=NS(attention_backend="sdpa"),
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1)
    contract = {
        "grpo": {"reference_identity": {"model_fingerprint": "model", "tokenizer_fingerprint": "tokenizer"}},
        "offline_baseline": binding(settings, "model", "tokenizer"),
    }
    build_updater(config, model, Path(data).parent, contract, opt, sched)({"global_step": 0}, None)
    expected = [p.grad.clone() for p in model.parameters()]
    opt.zero_grad(set_to_none=True)
    dist.init_process_group("gloo", init_method=Path(store).as_uri(), rank=rank, world_size=2)
    try:
        build_updater(config, model, Path(data).parent, contract, opt, sched)(
            {"global_step": 0}, NS(world_size=2, rank=rank)
        )
        for p, q in zip(model.parameters(), expected):
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad)
            torch.testing.assert_close(p.grad, q, rtol=2e-5, atol=2e-7)
        if rank == 0:
            Path(result).write_text("sum gradients match one-rank oracle")
    finally:
        dist.destroy_process_group()


def test_two_rank_groups_keep_global_branch_and_token_weights(tmp_path):
    import torch.multiprocessing as mp

    collection(tmp_path)
    prepare(tmp_path / "groups.json", tmp_path / "dataset", "adaptive")
    result = tmp_path / "parallel-result.txt"
    mp.spawn(
        _key_step_rank, args=(str(tmp_path / "rendezvous"), str(tmp_path / "dataset"), str(result)), nprocs=2
    )
    assert result.is_file()


def test_public_service_and_artifact_receipt_resume_with_new_schema(tmp_path, monkeypatch):
    import yaml

    from core.locking import file_lock
    from RL.artifacts import verify
    from RL.offline_baseline_train import execute
    from RL.service import run
    from tests.lora_helpers import LORA, model_and_targets

    root = Path(__file__).parents[1]
    collection(tmp_path)
    prepare(tmp_path / "groups.json", tmp_path / "dataset", "adaptive")
    values = yaml.safe_load((root / "RL/experiments/key-step-grpo.yaml").read_text())
    values["run"].update(mode="smoke", attempt_id="key-step-public-paused")
    values["runtime"]["output_root"] = str(tmp_path / "artifacts")
    values["key_steps"].update(dataset=str(tmp_path / "dataset"), epochs=2)
    values["lora"] = LORA.model_dump()
    values["training"]["checkpoint_every_updates"] = 1
    values["validation"]["stop_after_update"] = 1
    monkeypatch.setattr("RL.service.base_identity", lambda *a: {"fingerprint": "model"})
    monkeypatch.setattr(
        "RL.service.file_lock",
        lambda path, **kw: file_lock(
            tmp_path / "isolated-device-locks" / path.name if path.parent.name == "lora-gpu" else path, **kw
        ),
    )

    def train(config, root, directory, base, parent, attempt):
        model, targets = model_and_targets(adapter=parent["adapter"] if parent else None)
        write_json(directory / "environment.json", {"tiny_fixture": True})
        write_json(
            directory / "trainable-parameters.json",
            {
                "targets": targets,
                "target_count": len(targets),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            },
        )
        return execute(config, directory, base, parent, attempt, model, targets, None)

    monkeypatch.setattr("RL.service._train", train)
    config_file = tmp_path / "train.yaml"
    config_file.write_text(yaml.safe_dump(values))
    paused = run({"project_root": str(root)}, config_file)
    assert paused["status"] == "paused"
    assert verify(
        {"project_root": str(root)},
        "lora-attempt",
        paused["attempt_id"],
        storage_root=values["runtime"]["output_root"],
    )["verified"]
    values["run"].update(
        attempt_id="key-step-public-resumed",
        start={"mode": "resume", "checkpoint": str(Path(paused["path"]) / "checkpoints/step-000001")},
    )
    values["validation"]["stop_after_update"] = None
    config_file.write_text(yaml.safe_dump(values))
    resumed = run({"project_root": str(root)}, config_file)
    assert resumed["status"] == "completed"
    assert resumed["state"]["consumed_groups"] == 2 and resumed["state"]["consumed_branches"] == 16
    assert verify(
        {"project_root": str(root)},
        "lora-attempt",
        resumed["attempt_id"],
        storage_root=values["runtime"]["output_root"],
    )["verified"]

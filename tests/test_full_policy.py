"""Full language boundaries and differentiable head against an independent oracle."""
import copy
from pathlib import Path

import pytest
import torch
import yaml

from RL.config import Config, schedule_contract
from RL.full_policy import FullPolicy, parameter_scope, optimizer, microsteps
from RL.grpo.objective import selected_logprobs
from tests.lora_helpers import raw_model, settings
from tests.test_grpo_prefix import group

ROOT = Path(__file__).parents[1]


def full_values():
    value = yaml.safe_load((ROOT / "RL/experiments/grpo-offline-k3-smoke.yaml").read_text())
    value["schema_version"] = 2
    del value["lora"]
    value["tuning"] = {"mode": "full", "scope": "language_model", "freeze_vision": True}
    value["execution"] = {"backend": "fsdp2", "sharding": "full_shard", "topology_policy": "exact"}
    value["optimizer"] = {"name": "adamw", "master_weights_dtype": "float32", "states_dtype": "float32"}
    value["training"]["learning_rate"] = 1e-6
    value["training"]["memory"].update(activation_offload="none", oom_retry={"enabled": False})
    return value


@pytest.mark.parametrize("tied", [False, True])
def test_full_gradients_match_conditional_model_and_freeze_vision(tied):
    torch.set_num_threads(2)
    original = raw_model()
    if tied:
        original.config.tie_word_embeddings = True
        original.config.text_config.tie_word_embeddings = True
        original.tie_weights()
    oracle = copy.deepcopy(original)
    inventory = parameter_scope(original)
    parameter_scope(oracle)
    policy = FullPolicy(original)
    row = group(3, 3)
    batch = row.branch_batch(1, "cpu")
    actual = policy(batch, 1)
    logits = oracle(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
    values, mask = selected_logprobs(logits, batch["input_ids"], batch["labels"], batch["loss_mask"])
    expected = values[mask]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    actual.sum().backward()
    expected.sum().backward()
    for row in inventory["parameters"]:
        p = dict(policy.named_parameters())[row["name"]]
        q = dict(oracle.named_parameters())[row["hf_names"][0]]
        assert p.grad is not None
        torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4)
    frozen = {n: p.clone() for n, p in original.base_model.visual.named_parameters()}
    opt, scheduler = optimizer(policy, inventory, settings())
    opt.step()
    scheduler.step()
    assert all(p.grad is None for p in original.base_model.visual.parameters())
    for n, p in original.base_model.visual.named_parameters():
        torch.testing.assert_close(p, frozen[n], atol=0, rtol=0)
    assert {st["exp_avg"].dtype for st in opt.state.values()} == {torch.float32}
    if tied:
        assert policy.lm_head.weight is policy.language_model.embed_tokens.weight


def test_scope_rejects_unknown_module_ownership():
    model = raw_model()
    model.model.extra = torch.nn.Linear(32, 32)
    with pytest.raises(ValueError, match="ownership"):
        parameter_scope(model)


def test_full_8x16_is_validated_for_eight_ranks_and_uses_frozen_microsteps():
    values = full_values()
    values["runtime"].pop("gpu", None)
    values["runtime"]["gpus"] = list(range(8))
    values["grpo"].update(group_batch_size=8, parallel_branch_chunk_size=16,
        parallel_branch_schedule="distinct_parent_two_pass_v1", prefix_reuse={"mode": "disabled"})
    Config.model_validate(values)
    rows = [type("Group", (), {"group_size": 16, "metadata": {"execution_chunk": {"rank": rank}}})()
            for rank in range(8)]
    table = microsteps(rows, 8)
    assert len(table) == 16
    assert table[0] == tuple((rank, 0) for rank in range(8))
    assert table[-1] == tuple((rank, 15) for rank in range(8))
    short = microsteps(rows[:3], 8)
    assert short[0] == ((0, 0), (1, 0), (2, 0), None, None, None, None, None)
    assert len(short) == 16


def test_microstep_weights_and_zero_contribution_ranks():
    rows = [group(2, 3), group(3, 2)]
    table = microsteps(rows, 4)
    assert table == [((0, 0), (0, 1), (0, 2), (1, 0)), ((1, 1), None, None, None)]
    # FSDP mean cancels the one explicit world-size compensation.
    assert sum(4 / (len(rows) * rows[g].group_size) / 4
               for step in table for task in step if task is not None for g, _ in [task]) == pytest.approx(1)


@pytest.mark.parametrize("mutation", [
    lambda v: v.update(lora={"rank": 2, "alpha": 4, "dropout": 0.0, "target_scope": "text_backbone_all_linear"}),
    lambda v: v["model"].update(gdn_backend="fla_chunk", dtype="float32"),
    lambda v: v["grpo"].update(prefix_reuse={"mode": "boundary_grad"}),
    lambda v: v["runtime"].update(topology_policy="preserve_updates"),
    lambda v: v["training"]["memory"].update(oom_retry={"enabled": True}),
    lambda v: v["training"]["memory"].update(activation_offload="checkpoint_boundary"),
    lambda v: v["runtime"].update(bf16_allow_reduced_precision_reduction=True),
    lambda v: v["runtime"].update(bf16_allow_split_k=False),
])
def test_full_rejects_unaccepted_combinations(mutation):
    values = full_values()
    mutation(values)
    with pytest.raises(ValueError):
        Config.model_validate(values)


def test_legacy_schedule_has_no_new_default_contract_fields():
    values = yaml.safe_load((ROOT / "RL/experiments/grpo-offline-k3-smoke.yaml").read_text())
    contract = schedule_contract(Config.model_validate(values))
    assert not {"execution", "tuning", "optimizer"}.intersection(contract)


@pytest.mark.parametrize("world", [1, 2, 4])
def test_unequal_parent_rounds_all_gradients_and_adam_match_independent_oracle(world, monkeypatch):
    from types import SimpleNamespace
    from RL.full_training import window_update
    from RL.grpo.objective import grpo_objective
    torch.set_num_threads(2)
    conditional = raw_model()
    inventory = parameter_scope(conditional)
    groups = [group(2, 2, "a"), group(3, 3, "b"), group(1, 2, "tail")]
    native = type(groups[0]).branch_batch
    monkeypatch.setattr(type(groups[0]), "branch_batch", lambda self, b, device: native(self, b, "cpu"))
    config = Config.model_validate(full_values())
    refs = {(g.group_id, bid): torch.linspace(-4.8, -3.7, len(g.completion_token_ids[b]))
            for g in groups for b, bid in enumerate(g.branch_ids)}
    oracle = copy.deepcopy(conditional)
    loss = 0.0
    for g in groups:
        for b, bid in enumerate(g.branch_ids):
            batch = g.branch_batch(b, "cpu")
            logits = oracle(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            values, mask = selected_logprobs(logits, batch["input_ids"], batch["labels"], batch["loss_mask"])
            v = values[mask]
            result = grpo_objective(v[None], refs[(g.group_id, bid)][None], [g.advantages[b]],
                torch.ones_like(v[None], dtype=torch.bool), clip_epsilon=config.grpo.clip_epsilon,
                kl_beta=config.grpo.kl_beta, k3_cap=config.grpo.k3_cap,
                ratio_guard_enabled=config.grpo.ratio_guard.enabled,
                ratio_guard_threshold=config.grpo.ratio_guard.threshold)
            loss = loss + result.loss / (len(groups) * g.group_size)
    loss.backward()
    policies, totals = [], []
    for rank in range(world):
        policy = FullPolicy(copy.deepcopy(conditional))
        ctx = SimpleNamespace(rank=rank, world_size=world, gather=lambda obj: [obj])
        value, _, _ = window_update(policy, groups, refs, config, ctx)
        totals.append(value)
        policies.append(policy)
    assert sum(totals) == pytest.approx(float(loss.detach()), abs=2e-6)
    oracle_params = dict(oracle.named_parameters())
    for row in inventory["parameters"]:
        gradients = [dict(p.named_parameters())[row["name"]].grad for p in policies]
        average = torch.stack(gradients).mean(0)
        expected = oracle_params[row["hf_names"][0]].grad
        torch.testing.assert_close(average, expected, atol=3e-6, rtol=3e-4, msg=row["name"])
        for policy in policies:
            dict(policy.named_parameters())[row["name"]].grad.copy_(average)
    expected_policy = FullPolicy(oracle)
    expected_opt, _ = optimizer(expected_policy, inventory, config.training)
    torch.nn.utils.clip_grad_norm_(expected_policy.parameters(), 1.0)
    expected_opt.step()
    for policy in policies:
        opt, _ = optimizer(policy, inventory, config.training)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        for (name, p), q in zip(policy.named_parameters(), expected_policy.parameters(), strict=True):
            torch.testing.assert_close(p, q, atol=3e-6, rtol=3e-4, msg=name)


def test_full_checkpoint_insufficient_disk_never_publishes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from RL.full_checkpoint import save
    import RL.full_checkpoint as module
    policy = FullPolicy(raw_model())
    inventory = parameter_scope(raw_model())
    opt, schedule = optimizer(policy, inventory, settings())
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    ctx = SimpleNamespace(rank=0, world_size=1, primary=True, gather=lambda obj: [obj])
    with pytest.raises(OSError, match="Insufficient disk"):
        save(tmp_path, policy, opt, schedule, {"global_step": 1}, {}, inventory, ctx)
    assert not list(tmp_path.rglob("receipt.json"))
    assert not (tmp_path / "checkpoints/step-000001").exists()


def test_full_checkpoint_write_failure_cannot_publish_receipt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from RL.full_checkpoint import save
    import RL.full_checkpoint as module
    original = raw_model()
    inventory = parameter_scope(original)
    policy = FullPolicy(original)
    opt, schedule = optimizer(policy, inventory, settings())
    for p in policy.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: SimpleNamespace(free=10**12))
    monkeypatch.setattr(module.dist, 'barrier', lambda: None)
    def fail(*args):
        raise OSError('injected shard write failure')
    monkeypatch.setattr(module, 'write_tensor', fail)
    ctx = SimpleNamespace(rank=0, world_size=1, primary=True, gather=lambda obj: [obj])
    with pytest.raises(OSError, match='injected shard'):
        save(tmp_path, policy, opt, schedule, {'global_step': 1}, {}, inventory, ctx)
    assert not list(tmp_path.rglob('receipt.json'))
    assert (tmp_path / 'checkpoints/.step-000001.staging').is_dir()
    assert not (tmp_path / 'checkpoints/step-000001').exists()


@pytest.mark.parametrize("fault", [None, "checkpoint_fingerprint", "tokenizer_fingerprint"])
def test_full_external_anchor_import_identity_and_no_forward(tmp_path, monkeypatch, fault):
    from types import SimpleNamespace
    from core.io import hash_json
    from tests.test_grpo_external_reference import make_source, select, BASE
    from RL.full_artifacts import prepared
    from RL.full_training import _anchor
    config = Config.model_validate(full_values())
    config.runtime.output_root = str(tmp_path)
    config.algorithm.dataset_root = str(tmp_path)
    def mutate(manifest, records, path):
        if fault:
            manifest["scoring_identity"][fault] = "wrong"
            for r in records:
                r["scoring_identity_sha256"] = hash_json(manifest["scoring_identity"])
    make_source(tmp_path, config, mutate=mutate)
    select(config, tmp_path)
    reader = prepared(config, ROOT, None)
    conditional = raw_model()
    inventory = parameter_scope(conditional)
    policy = FullPolicy(conditional)
    def forbidden(*args, **kwargs):
        raise AssertionError("external anchor must not forward the model")
    monkeypatch.setattr(policy, "forward", forbidden)
    monkeypatch.setattr("torch.distributed.barrier", lambda: None)
    ctx = SimpleNamespace(primary=True, rank=0, gather=lambda obj: [obj])
    if fault:
        with pytest.raises(ValueError):
            _anchor(config, reader, policy, BASE, inventory, tmp_path, ctx, None)
    else:
        manifest = _anchor(config, reader, policy, BASE, inventory, tmp_path, ctx, None)
        assert len(manifest["entries"]) == 15
        assert manifest["roles"] == ["ratio_denominator", "k3_reference"]


def test_full_clip_accepts_guarded_zero_and_rejects_nonfinite_or_missing(monkeypatch):
    from RL.full_policy import global_clip
    monkeypatch.setattr("torch.distributed.all_reduce", lambda value, **kw: None)
    module = torch.nn.Linear(3, 2, bias=False)
    module.weight.grad = torch.zeros_like(module.weight)
    assert global_clip(module, 1.0, None) == 0.0
    module.weight.grad.fill_(2.0)
    assert global_clip(module, 1.0, None) == pytest.approx(24**0.5)
    assert module.weight.grad.norm().item() == pytest.approx(1.0, abs=1e-6)
    module.weight.grad.fill_(float('inf'))
    with pytest.raises(ValueError, match='Invalid full gradient'):
        global_clip(module, 1.0, None)
    module.weight.grad = None
    with pytest.raises(ValueError, match='missing'):
        global_clip(module, 1.0, None)


def test_full_runner_dry_run_has_no_torch(tmp_path):
    import subprocess
    values = full_values()
    values["run"]["mode"] = "dry-run"
    values["runtime"]["output_root"] = str(tmp_path)
    values["algorithm"]["dataset_root"] = str(tmp_path)
    model = tmp_path / "minimal-base"
    model.mkdir()
    # Reuse the actual tiny HF definition; dry-run only reads config/identity files.
    raw_model().save_pretrained(model)
    (model / "tokenizer.json").write_text('{}')
    (model / "tokenizer_config.json").write_text('{}')
    values["model"]["path"] = str(model)
    values["preflight"] = {"verify_identity": True}
    source = tmp_path / "full-dry.yaml"
    source.write_text(yaml.safe_dump(values))
    code = '''
import sys
from RL.service import run
result=run({'project_root':sys.argv[1]}, sys.argv[2])
assert result['status']=='validated'
assert 'torch' not in sys.modules and 'peft' not in sys.modules
'''
    subprocess.run([str(ROOT / ".venvs/runner/bin/python"), "-c", code, str(ROOT), str(source)],
                   cwd=ROOT, check=True, capture_output=True, timeout=30)


def test_full_branch_metrics_feed_existing_scalar_and_tensorboard_contract(monkeypatch):
    from types import SimpleNamespace
    from RL.full_training import window_update
    from RL.grpo.metrics import summarize_update
    from RL.observability_tensorboard import scalar_values
    policy = FullPolicy(raw_model())
    g = group(2, 2)
    native = type(g).branch_batch
    monkeypatch.setattr(type(g), 'branch_batch', lambda self, b, device: native(self, b, 'cpu'))
    refs = {(g.group_id, bid): policy(g.branch_batch(b, 'cpu')).detach() for b, bid in enumerate(g.branch_ids)}
    _, rows, _ = window_update(policy, [g], refs, Config.model_validate(full_values()),
        SimpleNamespace(rank=0, world_size=1, gather=lambda obj: [obj]))
    result = summarize_update([[r['metrics'] for r in rows]])
    assert result['policy_active_objective_fraction'] == pytest.approx(1.0)
    values = scalar_values({'global_step': 1, 'loss': 0., 'policy_loss': 0., 'k3_penalty_mean': 0.,
        'global_grad_norm': 2., 'trainable_parameters': 99, 'policy_changed': True, 'microsteps': 2,
        'learning_rate': 1e-6, 'next_learning_rate': 0.0, 'gradient_clip_scale': .5,
        'gradient_clip_threshold': 1., **result})
    assert values['full/global_grad_norm'] == 2.
    assert values['full/policy_changed'] == 1
    assert values['optimizer/learning_rate/next'] == 0.0
    assert values['learning/preference/advantage_alignment'] == result['advantage_logprob_alignment']

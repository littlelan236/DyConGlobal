"""Behavioral sparse CE acceptance on real Qwen3.5 GatedDeltaNet and PEFT."""

import copy
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
import yaml

from core.io import hash_json
from RL.artifacts import verify_receipt
from RL.checkpoints import adapter_contract, restore_checkpoint, save_checkpoint
from RL.config import Config, Memory, Supervision
from RL.memory import (
    LossBackend,
    configure_checkpointing,
    exact_length_batch,
    install_mlp_backend,
    text_backbone,
)
from RL.modeling import adapter_fingerprint, inject
from RL.supervision import consumed_selected_tokens, policy_identity, prepare_batch
from RL.training import capture_rng, optimizer_and_scheduler, train
from RL.validation import causal_ce, fixture_batches, validate_batch
from tests.lora_helpers import LORA, batches, raw_model, settings

EXPLICIT = Supervision(mode="explicit_mask")
ROOT = Path(__file__).parents[1]


def tiny(linear_only=False):
    model, targets = inject(raw_model(linear_only=linear_only, max_positions=256), LORA)
    # Exercise gradients of both A and B, including every linear-attention target.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.03)
    model.train()
    model.enable_input_require_grads()
    return model, targets


def row(length=65, padding=0, targets=None):
    ids = ((torch.arange(length + padding) * 7 + 3) % 61 + 2).view(1, -1)
    attention = torch.ones_like(ids)
    attention[:, length:] = 0
    labels = ids.clone()
    labels[:, :2] = -100
    labels[:, length:] = -100
    # Labels deliberately need not equal input IDs (including the selected EOS).
    labels[:, length - 1] = 1
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, targets if targets is not None else [length - 1]] = True
    return dict(input_ids=ids, attention_mask=attention, labels=labels, loss_mask=mask)


def prepare(batch, policy=EXPLICIT):
    return prepare_batch(batch, vocab_size=64, max_length=256, supervision=policy)


def train_options(updates=2, accumulation=2, policy=EXPLICIT):
    return dict(max_updates=updates, gradient_accumulation_steps=accumulation, grad_clip=1.,
                vocab_size=64, max_length=256, supervision=policy)


@pytest.mark.parametrize("limit,expected", [(1, [7]), (2, [2, 7]), (4, [2, 3, 5, 7]), (6, list(range(2, 8))),
                                           (100, list(range(2, 8)))])
def test_uniform_integer_coordinates_and_per_sample_counts(limit, expected):
    batch = batches()[0]
    original = batch["labels"].clone()
    policy = Supervision(mode="uniform", max_targets_per_sample=limit)
    selection = prepare(batch, policy)
    assert selection.coordinates == tuple((0, position) for position in expected)
    assert selection.sha256 == prepare(batch, policy).sha256
    assert selection.available_count == 6
    assert selection.selected_count == len(expected)
    torch.testing.assert_close(batch["labels"], original, atol=0, rtol=0)
    second = batches()[2]
    merged = {key: torch.cat([batch[key], second[key]]) for key in batch}
    combined = prepare(merged, policy)
    assert sum(b == 0 for b, _ in combined.coordinates) == len(expected)
    assert sum(b == 1 for b, _ in combined.coordinates) == min(2, limit)


@pytest.mark.parametrize("defect", ["dtype", "float_nan", "shape", "zero", "ignored", "padding", "missing",
                                    "conflict_labels", "conflict_uniform", "empty_sample", "predecessor", "four_d"])
def test_invalid_masks_fail_strictly(defect):
    batch = row(8, 2)
    policy = EXPLICIT
    if defect == "dtype":
        batch["loss_mask"] = batch["loss_mask"].long()
    elif defect == "float_nan":
        batch["loss_mask"] = batch["loss_mask"].float().fill_(float("nan"))
    elif defect == "shape":
        batch["loss_mask"] = batch["loss_mask"][:, :-1]
    elif defect == "zero":
        batch["loss_mask"][0, 0] = True
    elif defect == "ignored":
        batch["loss_mask"][0, 1] = True
    elif defect == "padding":
        batch["loss_mask"][0, -1] = True
    elif defect == "missing":
        del batch["loss_mask"]
    elif defect.startswith("conflict"):
        policy = Supervision() if defect.endswith("labels") else Supervision(mode="uniform", max_targets_per_sample=1)
    elif defect == "empty_sample":
        batch = {key: value.repeat(2, 1) for key, value in batch.items()}
        batch["loss_mask"][1].zero_()
    elif defect == "predecessor":
        batch["attention_mask"][0, 6] = 0
        batch["labels"][0, 6] = -100
    elif defect == "four_d":
        batch["attention_mask"] = torch.ones((1, 1, 10, 10))
    with pytest.raises(ValueError):
        prepare(batch, policy)


@pytest.mark.parametrize("value", [0, -1, True, 2.0, "2", None])
def test_invalid_uniform_limit(value):
    with pytest.raises(ValueError):
        Supervision(mode="uniform", max_targets_per_sample=value)


def test_strict_schema_sparse_backend_and_default_compatibility():
    values = yaml.safe_load((ROOT / "RL/experiments/lora-smoke.yaml").read_text())
    assert Config.model_validate(values).training.supervision.mode == "labels"
    for mode in ["explicit_mask", "uniform"]:
        values["training"]["supervision"] = dict(mode=mode, max_targets_per_sample=2 if mode == "uniform" else None)
        with pytest.raises(ValueError, match="loss_chunk_tokens"):
            Config.model_validate(values)
        values["training"]["memory"] = dict(loss_chunk_tokens=3)
        Config.model_validate(values)
        del values["training"]["memory"]
    for invalid in [dict(mode="labels", max_targets_per_sample=2), dict(mode="explicit_mask", unexpected=True)]:
        with pytest.raises(ValueError):
            Supervision.model_validate(invalid)
    for batch in batches():
        selection = prepare(batch, Supervision())
        assert selection.selected_count == int((batch["labels"][:, 1:] != -100).sum())
        assert validate_batch(batch, vocab_size=64, max_length=32) == selection.selected_count


def test_target_coordinates_once_shift_eos_and_no_input_mutation():
    model, _ = tiny()
    batch = row(8, 2, targets=[2, 7])
    selection = prepare(batch)
    hidden = text_backbone(model)(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                  use_cache=False).last_hidden_state
    expected = F.cross_entropy(model.get_output_embeddings()(hidden[0, [1, 6]]),
                               batch["labels"][0, [2, 7]], reduction="sum")
    actual, count = LossBackend(Memory(loss_chunk_tokens=1))(model, selection.model_batch())
    torch.testing.assert_close(actual, expected)
    assert count == 2 and selection.coordinates == ((0, 2), (0, 7))
    assert batch["labels"][0, 3] != -100 and selection.batch["labels"][0, 3] == -100
    with pytest.raises(ValueError, match="Resolve"):
        LossBackend(Memory(loss_chunk_tokens=1))(model, batch)


@pytest.mark.parametrize("linear_only", [False, True])
@pytest.mark.parametrize("length", [63, 64, 65, 127, 128])
@pytest.mark.parametrize("chunk", [1, 2, 100])
def test_sparse_dense_loss_hidden_all_lora_gradients_and_projection_scope(linear_only, length, chunk):
    dense, targets = tiny(linear_only)
    sparse = copy.deepcopy(dense)
    batch = row(length, targets=[length - 1] if length == 65 else [2, length // 2, length - 1])
    selection = prepare(batch)
    effective = selection.model_batch()
    frozen = {name: p.detach().clone() for name, p in sparse.named_parameters() if not p.requires_grad}
    hidden_outputs = []
    def retain(module, args, output):
        output.last_hidden_state.retain_grad()
        hidden_outputs.append(output.last_hidden_state)
        assert output.last_hidden_state.shape[1] == length
        assert output.past_key_values is None
    handles = [text_backbone(model).register_forward_hook(retain) for model in (dense, sparse)]
    expected, count = causal_ce(dense, effective)
    projections = []
    original_linear = F.linear
    head = sparse.get_output_embeddings()
    def observe(inputs, weight, bias=None):
        if weight is head.weight:
            projections.append(inputs.shape[:-1].numel())
        return original_linear(inputs, weight, bias)
    backend = LossBackend(Memory(loss_chunk_tokens=chunk))
    with patch("RL.memory.F.linear", side_effect=observe):
        actual, sparse_count = backend(sparse, effective)
    assert sum(projections) == count == sparse_count == selection.selected_count == backend.tokens
    assert max(projections) <= chunk and len(projections) == backend.blocks
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=3e-6)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(hidden_outputs[0].grad, hidden_outputs[1].grad, atol=4e-7, rtol=4e-5)
    direct = hidden_outputs[1].grad
    selected_predictors = {(b, j - 1) for b, j in selection.coordinates}
    for position in range(length):
        if (0, position) not in selected_predictors:
            assert direct[0, position].count_nonzero() == 0
    categories = set()
    for (name, p), (other_name, q) in zip(dense.named_parameters(), sparse.named_parameters(), strict=True):
        assert name == other_name
        if p.requires_grad:
            assert p.grad is not None and q.grad is not None
            torch.testing.assert_close(p.grad, q.grad, atol=5e-7, rtol=4e-4)
            assert p.grad.norm() > 0  # Nonzero B exercises A as well.
            categories.add("linear_attention" if ".linear_attn." in name else
                           "full_attention" if ".self_attn." in name else "mlp")
        else:
            assert p.grad is q.grad is None
            torch.testing.assert_close(q, frozen[name], atol=0, rtol=0)
    assert categories == {target["category"] for target in targets}
    for handle in handles:
        handle.remove()


def test_pure_linear_history_influences_last_target_with_prefix_gradient():
    model, targets = tiny(True)
    # A random tiny decay can erase 100-token history below fp32 resolution.
    # Use a nondegenerate long-memory fixture, without changing production weights.
    with torch.no_grad():
        for layer in text_backbone(model).layers:
            layer.linear_attn.A_log.fill_(-4.)
    assert {t["category"] for t in targets} == {"mlp", "linear_attention"}
    batch = row(128)
    selection = prepare(batch)
    embeddings = []
    def retain(module, args, output):
        output.retain_grad()
        embeddings.append(output)
    handle = model.get_input_embeddings().register_forward_hook(retain)
    expected, _ = LossBackend(Memory(loss_chunk_tokens=1))(model, selection.model_batch())
    expected.backward()
    assert embeddings[0].grad[0, :64].count_nonzero() > 0
    assert model.get_input_embeddings().weight.grad is None
    changed = copy.deepcopy(batch)
    changed["input_ids"][0, :16] = (changed["input_ids"][0, :16] + 19) % 64
    with torch.no_grad():
        first = text_backbone(model)(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                     use_cache=False).last_hidden_state
        second = text_backbone(model)(input_ids=changed["input_ids"], attention_mask=changed["attention_mask"],
                                      use_cache=False).last_hidden_state
        first_logits = model.get_output_embeddings()(first[:, -2])
        second_logits = model.get_output_embeddings()(second[:, -2])
    assert float((first_logits - second_logits).abs().max()) > 1e-7
    handle.remove()


@pytest.mark.parametrize("exact", [False, True])
def test_supervision_never_changes_attention_lengths_padding_or_recurrent_history(exact):
    model, _ = tiny()
    batch = row(65, 5, targets=[2])
    selection = prepare(batch)
    trimmed = exact_length_batch(selection.batch)
    assert trimmed["loss_mask"].shape == (1, 65)
    assert trimmed["loss_mask"][0, 2] and selection.coordinates == ((0, 2),)
    observed = []
    handles = []
    def observe(module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        mask = kwargs.get("attention_mask")
        observed.append((type(module).__name__, hidden.shape[1], None if mask is None else mask.detach().clone()))
        assert "loss_mask" not in kwargs and kwargs.get("past_key_values") is None
    for layer in text_backbone(model).layers:
        attention = layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn
        handles.append(attention.register_forward_pre_hook(observe, with_kwargs=True))
    backend = LossBackend(Memory(loss_chunk_tokens=2, exact_length=exact))
    backend(model, selection.model_batch())
    first = observed[:]
    observed.clear()
    batch["loss_mask"].zero_()
    batch["loss_mask"][0, 64] = True
    backend(model, prepare(batch).model_batch())
    assert len(first) == len(observed) == 2
    for (kind, length, mask), (other_kind, other_length, other_mask) in zip(first, observed, strict=True):
        assert kind == other_kind and length == other_length == (65 if exact else 70)
        if mask is None:
            assert other_mask is None
        else:
            torch.testing.assert_close(mask, other_mask, atol=0, rtol=0)
    # Repeating a complete forward has no mutable recurrent state left over.
    a = backend(model, prepare(batch).model_batch())[0]
    b = backend(model, prepare(batch).model_batch())[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("policy", ["layer", "selective_mlp", "layer_group"])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("linear_only", [False, True])
def test_cross_block_tail_checkpoint_mlp_copy_hooks_autocast(policy, bf16, linear_only):
    # Tolerances fixed before execution; CPU reference kernels only.
    atol, rtol = (0.005, 0.07) if bf16 else (7e-7, 5e-4)
    dense, _ = tiny(linear_only)
    sparse = copy.deepcopy(dense)
    install_mlp_backend(sparse, 17)
    config = SimpleNamespace(training=SimpleNamespace(checkpoint_policy=policy, checkpoint_group_layers=2,
                                                      memory=Memory(mlp_chunk_tokens=17)))
    configure_checkpointing(sparse, config, 2**30)
    batch = prepare(row(128, 3)).model_batch()
    copied = []
    def pack(tensor):
        copied.append(tuple(tensor.shape))
        return tensor.detach().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        expected, _ = causal_ce(dense, batch)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            actual, _ = LossBackend(Memory(loss_chunk_tokens=1, exact_length=True))(sparse, batch)
    torch.testing.assert_close(actual, expected, atol=0.03 if bf16 else 2e-5, rtol=0.02 if bf16 else 3e-6)
    expected.backward()
    actual.backward()
    assert copied
    for (name, p), (other_name, q) in zip(dense.named_parameters(), sparse.named_parameters(), strict=True):
        assert name == other_name
        if p.requires_grad:
            assert torch.isfinite(q.grad).all()
            torch.testing.assert_close(p.grad, q.grad, atol=atol, rtol=rtol)


def test_selected_weighted_accumulation_merged_reference_and_partial_window():
    sparse, _ = tiny()
    dense = copy.deepcopy(sparse)
    rows = [row(8, targets=[2, 3, 7]), row(6, 2, targets=[5]), row(7, 1, targets=[3, 6])]
    effective = [prepare(batch).model_batch() for batch in rows]
    merged = {key: torch.cat([effective[0][key], effective[1][key]]) for key in effective[0]}
    opt, sched = optimizer_and_scheduler(sparse, settings(2))
    other_opt, other_sched = optimizer_and_scheduler(dense, settings(2))
    metrics = []
    state = train(sparse, rows, LossBackend(Memory(loss_chunk_tokens=2)), opt, sched,
                  **train_options(), on_update=lambda _, values: metrics.append(values))
    expected = train(dense, [merged, effective[2]], causal_ce, other_opt, other_sched,
                     **train_options(accumulation=1, policy=Supervision()))
    assert state["consumed_tokens"] == expected["consumed_tokens"] == 6
    assert state["consumed_microbatches"] == 3 and state["global_step"] == 2
    assert [m["supervised_tokens"] for m in metrics] == [4, 2]
    assert [m["input_tokens"] for m in metrics] == [14, 7]
    assert metrics[0]["supervision_fraction"] == 4 / 10
    for p, q in zip(sparse.parameters(), dense.parameters(), strict=True):
        torch.testing.assert_close(p, q, atol=5e-7, rtol=4e-4)


def test_invalid_later_batch_fails_before_any_forward_backward_or_update():
    model, _ = tiny()
    first, bad = row(8), row(8)
    bad["loss_mask"].zero_()
    calls = []
    opt, sched = optimizer_and_scheduler(model, settings(1))
    before = adapter_fingerprint(model)
    def forbidden(*args):
        calls.append(True)
        raise AssertionError("No forward should have started")
    with pytest.raises(ValueError, match="nonzero"):
        train(model, [first, bad], forbidden, opt, sched, **train_options(updates=1))
    assert not calls and not opt.state and before == adapter_fingerprint(model)
    assert all(p.grad is None for p in model.parameters())


def test_later_exact_length_error_precedes_window_forward():
    model, _ = tiny()
    first = row(8)
    second = {key: value.repeat(2, 1) for key, value in row(8).items()}
    opt, sched = optimizer_and_scheduler(model, settings(1))
    backend = LossBackend(Memory(loss_chunk_tokens=2, exact_length=True))
    with pytest.raises(ValueError, match="microbatch=1"):
        train(model, [first, second], backend, opt, sched, **train_options(updates=1))
    assert backend.tokens == 0 and not opt.state


def test_first_legal_target_uses_hidden_zero_and_no_implicit_eos_selection():
    model, _ = tiny()
    batch = row(8, targets=[1])
    batch["labels"][0, 1] = 17
    selection = prepare(batch)
    assert selection.coordinates == ((0, 1),) and selection.batch["labels"][0, 7] == -100
    hidden = text_backbone(model)(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                  use_cache=False).last_hidden_state
    expected = F.cross_entropy(model.get_output_embeddings()(hidden[:, 0]), torch.tensor([17]), reduction="sum")
    actual, count = LossBackend(Memory(loss_chunk_tokens=3))(model, selection.model_batch())
    torch.testing.assert_close(actual, expected)
    assert count == 1


def test_input_identity_masks_labels_coordinates_and_trim_binding():
    batch = row(65, 4)
    selection = prepare(batch)
    for key in ["input_ids", "labels", "loss_mask"]:
        changed = copy.deepcopy(batch)
        if key == "loss_mask":
            changed[key][0, 2] = True
        else:
            changed[key][0, 3] = 17
        assert prepare(changed).sha256 != selection.sha256
    assert prepare(exact_length_batch(batch)).sha256 != selection.sha256
    assert prepare(exact_length_batch(batch)).coordinates == selection.coordinates
    assert policy_identity(Supervision())["algorithm"] != policy_identity(EXPLICIT)["algorithm"]


def test_fixture_mask_owner_raw_identity_and_selected_cursor():
    class Tokenizer:
        pad_token_id, eos_token_id = 0, 1
        def __len__(self):
            return 64
        def encode(self, text, add_special_tokens=False):
            return [2 + ord(char) % 60 for char in text.split()[0]]
    rows, identity = fixture_batches(Tokenizer(), 65, EXPLICIT)
    assert identity["mask_source"] == "fixture_row_uniform_endpoints_v1"
    assert identity == fixture_batches(Tokenizer(), 65, EXPLICIT)[1]
    assert [prepare(batch).selected_count for batch in rows] == list(range(1, len(rows) + 1))
    count = consumed_selected_tokens(rows, 8, vocab_size=64, max_length=65, supervision=EXPLICIT)
    assert count == sum(prepare(rows[i % len(rows)]).selected_count for i in range(8))
    assert count < sum(int((rows[i % len(rows)]["labels"][:, 1:] != -100).sum()) for i in range(8))


@pytest.mark.parametrize("policy", [EXPLICIT, Supervision(mode="uniform", max_targets_per_sample=2)])
def test_sparse_resume_new_process_contract_rejection_receipt_and_warm_start(tmp_path, policy):
    full, targets = tiny()
    paused = copy.deepcopy(full)
    rows = [row(8, targets=[2, 7]), row(7, 1), row(6, 2, targets=[2, 3, 5])]
    if policy.mode == "uniform":
        rows = [{key: value for key, value in batch.items() if key != "loss_mask"} for batch in rows]
    selections = [prepare(batch, policy) for batch in rows]
    identity = adapter_contract({"fingerprint": "a" * 64}, targets, LORA)
    contract = {"supervision": policy_identity(policy), "input": [s.sha256 for s in selections], "schedule": "4 linear"}
    opt, sched = optimizer_and_scheduler(full, settings(4))
    metrics = []
    final = train(full, [copy.deepcopy(rows[i % 3]) for i in range(8)], LossBackend(Memory(loss_chunk_tokens=2)),
                  opt, sched, **train_options(updates=4, policy=policy),
                  on_update=lambda _, values: metrics.append(values))
    full_rng = capture_rng()
    paused_opt, paused_sched = optimizer_and_scheduler(paused, settings(4))
    state = train(paused, [copy.deepcopy(rows[i % 3]) for i in range(4)], LossBackend(Memory(loss_chunk_tokens=2)),
                  paused_opt, paused_sched, **train_options(updates=2, policy=policy))
    checkpoint = tmp_path / "step-000002"
    save_checkpoint(paused, paused_opt, paused_sched, state, checkpoint, contract, identity)
    torch.save({"rows": rows, "policy": policy.model_dump(), "contract": contract, "identity": identity},
               tmp_path / "inputs.pt")
    code = '''
import sys, torch
from pathlib import Path
from RL.config import Supervision, Memory
from RL.memory import LossBackend
from RL.modeling import inject, adapter_fingerprint
from RL.training import optimizer_and_scheduler, train, capture_rng
from RL.checkpoints import restore_checkpoint
from tests.lora_helpers import raw_model, LORA, settings
from tests.test_lora_supervision import train_options
root=Path(sys.argv[1]); cp=root/'step-000002'
data=torch.load(root/'inputs.pt', weights_only=True)
model,_=inject(raw_model(max_positions=256), LORA, adapter=cp/'adapter')
model.enable_input_require_grads()
opt,sched=optimizer_and_scheduler(model,settings(4))
state=restore_checkpoint(cp,opt,sched,data['contract'],data['identity'])
metrics=[]
state=train(model,[data['rows'][i%3] for i in range(4,8)],LossBackend(Memory(loss_chunk_tokens=2)),
            opt,sched,**train_options(updates=4,policy=Supervision(**data['policy'])),state=state,
            on_update=lambda _,values: metrics.append(values))
torch.save(dict(state=state, metrics=metrics, parameters=model.state_dict(), optimizer=opt.state_dict(),
                scheduler=sched.state_dict(), rng=capture_rng()),root/'resumed.pt')
'''
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=ROOT, check=True, timeout=60,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}, capture_output=True)
    result = torch.load(tmp_path / "resumed.pt", weights_only=True)
    assert result["state"] == final
    for name, parameter in full.state_dict().items():
        torch.testing.assert_close(parameter, result["parameters"][name], atol=0, rtol=0)
    for key in ["loss", "selection_sha256", "consumed_tokens", "learning_rate"]:
        assert [m[key] for m in metrics[2:]] == [m[key] for m in result["metrics"]]
    assert result["scheduler"] == sched.state_dict()
    for index, values in opt.state_dict()["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(value, result["optimizer"]["state"][index][key], atol=0, rtol=0)
    assert result["rng"]["python"] == full_rng["python"]
    assert result["rng"]["numpy"] == full_rng["numpy"]
    torch.testing.assert_close(result["rng"]["torch_cpu"], full_rng["torch_cpu"], atol=0, rtol=0)
    for field in ["mode", "max_targets_per_sample", "algorithm", "mask", "coordinates"]:
        changed = copy.deepcopy(contract)
        if field in ["mask", "coordinates"]:
            changed["input"][0] = hash_json(field)
        else:
            changed["supervision"][field] = "changed"
        with pytest.raises(ValueError, match="contract mismatch"):
            restore_checkpoint(checkpoint, paused_opt, paused_sched, changed, identity)
    # Adapter-only warm start starts a new schedule and selected cursor.
    warm, _ = inject(raw_model(max_positions=256), LORA, adapter=checkpoint / "adapter")
    warm_opt, warm_sched = optimizer_and_scheduler(warm, settings(1))
    assert not warm_opt.state and warm_sched.last_epoch == 0
    warm_state = train(warm, rows[:1], LossBackend(Memory(loss_chunk_tokens=2)), warm_opt, warm_sched,
                       **train_options(updates=1, accumulation=1, policy=policy))
    assert warm_state["consumed_microbatches"] == 1 and warm_state["consumed_tokens"] == selections[0].selected_count
    verify_receipt(checkpoint)
    (checkpoint / "state.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        restore_checkpoint(checkpoint, paused_opt, paused_sched, contract, identity)


def test_runner_real_dry_run_all_modes_without_torch_gpu_or_attempt():
    runner = ROOT / ".venvs/runner/bin/python"
    if not runner.exists():
        pytest.skip("Runner exists only on the server")
    code = '''
import sys, json
from pathlib import Path
from RL.service import run
root=Path.cwd()
for mode in ['labels','explicit','uniform']:
    result=run({'project_root':str(root)},f'RL/experiments/lora-sparse-{mode}.yaml')
    assert result['status']=='validated'
    assert result['supervision']['selected_tokens'] is None
    assert result['supervision']['selection_sha256'] is None
    assert not Path(result['output']).exists()
    assert 'torch' not in sys.modules and 'peft' not in sys.modules
print('all sparse dry-runs passed without Torch or attempts')
'''
    subprocess.run([str(runner), "-c", code], cwd=ROOT, check=True, timeout=180, capture_output=True,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})

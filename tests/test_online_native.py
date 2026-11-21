"""Native policy evidence, CPU gradient equivalence, and publication replay."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
import time

import numpy as np
import pytest

from core.io import hash_json, read_json, sha256, write_json
from RL.online_runtime import OnlineRow, OnlineRolloutBatch, SharedVllmRolloutProvider


def completion(name, source="baseline", trajectory="task:baseline:0"):
    return OnlineRow(name, "task", source, "decision", (1, 2, 3, 4), (3, 4), (-2., -3.),
                     None, 1., trajectory, 2, 0, True)


def test_native_completion_binds_exact_tokens_policy_and_dtype(tmp_path):
    from RL.online_evidence import native_completion
    from runtime.serving.raw_logprobs import RawLogprobWriter
    record = {"request_id": "native", "run_id": "run", "session_id": "session",
              "decision_id": "decision", "branch_id": 0, "step_id": 0, "api_attempt_id": "api",
              "tokenizer_fingerprint": "tok", "prompt_token_ids": [1, 2], "output_token_ids": [3, 4]}
    record.update(request_sha256=hash_json(record["prompt_token_ids"]), output_sha256=hash_json(record["output_token_ids"]))
    writer = RawLogprobWriter(tmp_path, tmp_path / "export", {}, source_identity={
        "model_fingerprint": "base", "serving_dtype": "bfloat16", "adapter": "adapter-sha"})
    try:
        record["raw_t1_probability"] = writer.write(record, [-.2, -.3])
    finally:
        writer.close()
    cfg = {"run_id": "run", "model": {"fingerprint": "base"}, "serving": {"dtype": "bfloat16"}}
    request = {"policy_version": "S1", "deployment": {"adapter_identity_sha256": "adapter-sha"}}
    native, values = native_completion(tmp_path / "export", record, cfg, request)
    assert native["output_token_ids"] == [3, 4]
    assert values == pytest.approx([-.2, -.3])
    with pytest.raises(ValueError, match="adapter"):
        native_completion(tmp_path / "export", record, cfg, {**request, "deployment": {"adapter_identity_sha256": "other"}})
    with pytest.raises(ValueError, match="tokens|token evidence"):
        native_completion(tmp_path / "export", {**record, "output_token_ids": [4, 3]}, cfg, request)


def wire():
    captured = []
    def backend(payload):
        captured.append(payload)
        time.sleep(.04)
        rows = tuple(completion(str(i), item["source"], item["trajectory_id"])
                     for i, item in enumerate(payload["work_items"]))
        return OnlineRolloutBatch(0, ("task",), "S0", "E0", rows, request_sha256=payload["request_sha256"]).as_dict()
    requests = []
    def capture(payload):
        requests.append(payload)
        return backend(payload)
    SharedVllmRolloutProvider(capture, m=1)(["task"], 0, "S0", "E0")
    captured.clear()
    return requests[0], backend, captured


def test_concurrent_duplicate_http_requests_execute_one_backend(tmp_path):
    from RL.online_server import OnlineRolloutService
    payload, backend, calls = wire()
    assert payload["max_inflight"] == 1
    service = OnlineRolloutService(backend, tmp_path)
    with ThreadPoolExecutor(6) as workers:
        replies = list(workers.map(lambda _: service.handle(payload), range(6)))
    assert len(calls) == 1 and all(reply == replies[0] for reply in replies)
    path = tmp_path / "responses" / (payload["request_sha256"] + ".json")
    value = read_json(path)
    value["batch"]["rows"][0]["advantage"] = 99.
    write_json(path, value)
    with pytest.raises(ValueError, match="checksum"):
        service.handle(payload)


def test_http_recovery_after_response_before_receipt(tmp_path, monkeypatch):
    from RL.online_server import OnlineRolloutService
    from RL.online_bootstrap import EstimatorDatasetStore
    payload, backend, calls = wire()
    original = EstimatorDatasetStore._atomic_json
    def fail_receipt(path, value):
        if path.parent.name == "requests":
            raise RuntimeError("receipt interrupted")
        original(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(EstimatorDatasetStore, "_atomic_json", staticmethod(fail_receipt))
        with pytest.raises(RuntimeError, match="interrupted"):
            OnlineRolloutService(backend, tmp_path).handle(payload)
    reply = OnlineRolloutService(backend, tmp_path).handle(payload)
    assert reply["step"] == 0 and len(calls) == 1


def tiny_policy():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                            attention_dropout=0., use_cache=False))
    target = "model.layers.0.self_attn.q_proj"
    model = get_peft_model(model, LoraConfig(r=1, lora_alpha=2, lora_dropout=0.,
                                           target_modules=[target], task_type="CAUSAL_LM"))
    identity = {"base_fingerprint": "tiny", "targets": [{"path": target}],
                "lora": {"rank": 1, "alpha": 2, "dropout": 0.}}
    return model, identity


def test_chunked_native_forward_matches_dense_gradients_with_prefix():
    import copy
    import torch
    from RL.online_runtime import collate_rows, online_selected_logprobs
    from RL.grpo.objective import selected_logprobs
    model, _ = tiny_policy()
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            parameter.data.fill_(.3)
    dense = copy.deepcopy(model)
    embeddings = []
    def retain_embeddings(module, inputs, output):
        output.requires_grad_(True)
        output.retain_grad()
        embeddings.append(output)
    hook = model.get_input_embeddings().register_forward_hook(retain_embeddings)
    tensors = collate_rows([completion("r")], device=torch.device("cpu"))
    actual, mask = online_selected_logprobs(model, tensors, block_size=1)
    output = dense(input_ids=tensors["input_ids"], attention_mask=tensors["attention_mask"])
    expected, _ = selected_logprobs(output.logits, tensors["input_ids"], tensors["labels"], tensors["loss_mask"])
    torch.testing.assert_close(actual[mask], expected[mask])
    actual[mask].sum().backward()
    expected[mask].sum().backward()
    hook.remove()
    assert not tensors["loss_mask"][0, :2].any()
    assert embeddings[0].grad[0, 0].abs().sum() > 0
    for (name, parameter), (_, reference) in zip(model.named_parameters(), dense.named_parameters()):
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, reference.grad, atol=1e-6, rtol=1e-4)


def test_initial_kl_scoring_and_checkpoint_restore_preserve_policy(tmp_path):
    import torch
    from peft.utils.save_and_load import get_peft_model_state_dict
    from RL.online_policy import InitialPolicyScorer, publish_adapter, restore_policy_checkpoint
    from RL.online_runtime import collate_rows, online_selected_logprobs
    from RL.checkpoints import save_checkpoint
    model, identity = tiny_policy()
    tensors = collate_rows([completion("r")], device=torch.device("cpu"))
    initial, mask = online_selected_logprobs(model, tensors)
    destination = tmp_path / "initial/adapter"
    publish_adapter(model, destination, identity)
    scorer = InitialPolicyScorer(model, destination)
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            parameter.data.fill_(.8)
    current = {k: v.clone() for k, v in get_peft_model_state_dict(model).items()}
    torch.testing.assert_close(scorer(tensors)[mask], initial.detach()[mask])
    assert model.training
    for name, value in get_peft_model_state_dict(model).items():
        torch.testing.assert_close(value, current[name])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    contract = {"adapter": identity, "schedule": {}, "objective": {}, "input": {}}
    checkpoint = tmp_path / "checkpoints/step-000001"
    save_checkpoint(model, optimizer, scheduler, {"global_step": 1}, checkpoint, contract, identity)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data.zero_()
    restore_policy_checkpoint(model, checkpoint, optimizer, scheduler, contract, identity)
    for name, value in get_peft_model_state_dict(model).items():
        torch.testing.assert_close(value, current[name])


def test_online_predictor_replay_checks_checkpoint(tmp_path):
    from RL.online_estimator import publish_online_predictor
    checkpoint = tmp_path / "fit.npz"
    np.savez(checkpoint, coef=np.array([1., 2.]), intercept=np.array(0.))
    cfg = {"project_root": str(tmp_path), "model": {"fingerprint": "base"}}
    first = publish_online_predictor(cfg, "fit", checkpoint, target="submission_remaining_steps_linear", site=0, width=2)
    assert publish_online_predictor(cfg, "fit", checkpoint, target="submission_remaining_steps_linear", site=0, width=2) == first
    np.savez(checkpoint, coef=np.array([3., 4.]), intercept=np.array(0.))
    with pytest.raises(ValueError, match="same fit"):
        publish_online_predictor(cfg, "fit", checkpoint, target="submission_remaining_steps_linear", site=0, width=2)


def test_mini_baseline_bypasses_consensus_without_changing_shared_config(tmp_path, monkeypatch):
    import rollout.runner as runner
    cfg = {"project_root": str(tmp_path), "parallel": {"enabled": True, "branch_count": 4}}
    monkeypatch.setattr(runner, "_boundary", lambda *a: None)
    monkeypatch.setattr(runner, "run_task", lambda derived, *args, **kwargs: derived["parallel"])
    assert runner.admitted_task(cfg, "run", "task", {}, None, {"online_source": "baseline"}) == {"enabled": False, "branch_count": 1}
    assert runner.admitted_task(cfg, "run", "task", {}, None, {"online_source": "dycon"}) == {"enabled": True, "branch_count": 4}
    assert cfg["parallel"]["enabled"]


def test_harness_stage_grades_only_final_trajectories_and_reuses(tmp_path, monkeypatch):
    from RL.online_evidence import ensure_evaluated
    data = {"rollouts": {"final": {"task_key": "t", "evaluation_status": "not_run", "resolved": None},
                          "old-retry": {"evaluation_status": "not_run"}},
            "evaluations": {}, "tasks": {"t": {"benchmark": "swebench"}}}
    calls = []
    class Evidence:
        def __init__(self, *args): pass
        def verify(self): pass
        def read(self): return data
        def select(self): return {"rollouts": {"final": data["rollouts"]["final"]}}
    monkeypatch.setattr("application.execution.evidence.RunEvidence", Evidence)
    monkeypatch.setattr("application.execution.run_plan.load_plan", lambda *args: {
        "config": {}, "run": tmp_path, "work_items": {"items": [{}]}})
    monkeypatch.setattr("application.execution.harness.prepare", lambda *args: calls.append("prepare"))
    def run(*args):
        calls.append("run")
        data["rollouts"]["final"].update(evaluation_status="completed", resolved=True, selected_evaluation_id="eval")
        data["evaluations"]["eval"] = {"outcomes": {"final": {"status": "completed", "resolved": True,
            "verifier_audit": {"classification": "ValidOutcome", "effective_resolved": True}}}}
    monkeypatch.setattr("application.execution.harness.run", run)
    assert ensure_evaluated(tmp_path, "run")["state"] == "evaluated"
    assert ensure_evaluated(tmp_path, "run")["state"] == "reused"
    assert calls == ["prepare", "run"]


@pytest.mark.parametrize("classification", ["ValidOutcome", "Review", "VerifierError"])
@pytest.mark.parametrize("completed,resolved", [(True, True), (True, False), (False, False)])
def test_online_reward_preserves_official_raw_boolean(classification, completed, resolved):
    from RL.online_evidence import online_terminal_label
    outcome = {"status": "Review", "resolved": None,
               "raw_outcome": {"status": "infra_error", "resolved": None},
               "official_result": {"completed": completed, "resolved": resolved},
               "verifier_audit": {"classification": classification}}
    data = {"evaluations": {"eval": {"outcomes": {"r": outcome}}}}
    assert online_terminal_label({"selected_evaluation_id": "eval"}, data, "r") is resolved
    assert outcome["resolved"] is None
    assert outcome["verifier_audit"]["classification"] == classification


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_online_reward_does_not_coerce_missing_official_result(value):
    from RL.online_evidence import online_terminal_label
    data = {"evaluations": {"eval": {"outcomes": {"r": {
        "official_result": {"completed": False, "resolved": value},
        "raw_outcome": {"status": "completed", "resolved": False}}}}}}
    assert online_terminal_label({"selected_evaluation_id": "eval"}, data, "r") is None


def test_online_advantage_config_reuses_existing_strategy():
    from RL.config import OnlineBootstrap
    from RL.grpo.advantage import advantage_settings
    config = OnlineBootstrap(tasks=["a"], initial_estimator="initial.json")
    assert advantage_settings(config.dycon_advantage)["advantage_policy"] == "consensus_top_half_binary_v1"
    with pytest.raises(ValueError):
        OnlineBootstrap(tasks=["a"], initial_estimator="initial.json", dycon_advantage={"top_k": 1})
    historical = OnlineBootstrap(tasks=["a"], initial_estimator="initial.json", branch_count=8,
        dycon_advantage={"advantage_policy": "historical_topk_zero_sum_v1", "top_k": 2, "top_k_fraction": None})
    assert advantage_settings(historical.dycon_advantage)["top_k"] == 2


def test_online_reward_preserves_existing_missing_submission_negative():
    from RL.online_evidence import online_terminal_label
    outcome = {"status": "completed", "resolved": False,
               "official_result": {"completed": False, "resolved": None},
               "verifier_audit": {"classification": "ValidOutcome", "reason_code": "missing_patch",
                                  "effective_resolved": False}}
    evidence = {"evaluations": {"e": {"outcomes": {"r": outcome}}}}
    assert online_terminal_label({"selected_evaluation_id": "e"}, evidence, "r") is False


def test_resolved_publication_uses_bound_mini_terminal_receipt(tmp_path):
    from hidden.publish_finalized import _completion_receipt, _check_finalization
    attempt = tmp_path / 'attempts/attempt-0'
    row = {'provenance': {'source': str(attempt / 'raw.traj.json')}, 'attempt_id': 'attempt-0'}
    value = {'rollout_id': 'r', 'attempt_id': 'attempt-0', 'capture_session_id': 's',
             'disposition': 'accepted', 'capture_status': 'complete'}
    write_json(attempt / 'attempt.json', value)
    path, completion = _completion_receipt(tmp_path, row, 'r', 's')
    assert path == attempt / 'attempt.json'
    _check_finalization(completion, 'r', {}, 'canonical_native')
    with pytest.raises(ValueError, match='identity'):
        _completion_receipt(tmp_path, row, 'r', 'another-session')
    write_json(attempt / 'attempt.json', {**value, 'disposition': 'failed'})
    with pytest.raises(ValueError, match='identity'):
        _completion_receipt(tmp_path, row, 'r', 's')


def test_native_assembler_excludes_unscored_decision_before_probability_loading(tmp_path, monkeypatch):
    from RL.online_evidence import NativeRunAssembler
    root = tmp_path
    run = root / "runs/run"
    run.mkdir(parents=True)
    cfg = {"project_root": str(root), "run_id": "run", "paths": {"nas_root": str(root)},
           "capture": {"export": {"root": "native"}}}
    records, items = {}, []
    for source in ("dycon", "baseline"):
        trajectory = run / (source + ".json")
        extra = {"response": {"source": source}}
        if source == "dycon":
            extra["dycon_parallel"] = {"session_id": "session", "decision_id": "decision"}
        write_json(trajectory, {"messages": [{"extra": extra}]})
        records[source] = {"trajectory_path": str(trajectory.relative_to(root)),
                           "trajectory_sha256": sha256(trajectory), "selected_evaluation_id": "eval"}
        items.append({"instance_id": "task", "online_source": source, "online_repetition": 0,
                      "repetition_id": source})
        write_json(run / "rollout/tasks/task/repetitions" / source / "result.json", {"rid": source})
    evidence_path = run / "evidence.json"
    write_json(evidence_path, {})
    class Evidence:
        path = evidence_path
        def __init__(self, *args): pass
        def verify(self): pass
        def read(self): return {"rollouts": records, "evaluations": {"eval": {"outcomes": {
            source: {"official_result": {"completed": False, "resolved": False},
                     "verifier_audit": {"classification": "Review"}} for source in records}}}}
    monkeypatch.setattr("application.execution.evidence.RunEvidence", Evidence)
    monkeypatch.setattr("application.execution.run_plan.load_plan", lambda *args: {
        "config": cfg, "run": run, "work_items": {"items": items}, "plan": {"plan_sha256": "plan"}})
    monkeypatch.setattr("rollout.retry.verify_final", lambda root, rid, item, result, data: result["rid"])
    monkeypatch.setattr("benchmark_evaluation.swebench_settlement.training_label", lambda *args: False)
    session = root / "session"
    write_json(session / "parallel/decisions/decision.json", {})
    monkeypatch.setattr("hidden.token_evidence.load_token_session", lambda *args, **kwargs: ({"base": session}, None))
    monkeypatch.setattr("steering.datasets.sessions.full_decision", lambda *args: ({
        "branch_count": 2, "candidates": [{"branch_id": 0, "score": None}, {"branch_id": 1, "score": 1.}]}, "hash"))
    def completion(root, response, cfg, request):
        assert response["source"] == "baseline"
        return {"request_id": "baseline", "prompt_token_ids": [1, 2], "output_token_ids": [3, 4]}, (-2., -3.)
    monkeypatch.setattr("RL.online_evidence.native_completion", completion)
    request = {"step": 0, "tasks": ["task"], "policy_version": "S0", "estimator_version": "E0",
               "request_sha256": "request", "work_items": [
                   {"task_id": "task", "source": source, "trajectory_id": source} for source in ("dycon", "baseline")]}
    result = NativeRunAssembler(cfg, feature={}, advantage_settings={"top_k": 1, "top_k_fraction": None})(
        request, {"exit_code": 0}, run_id="run")
    assert len(result["rows"]) == 1
    assert result["rows"][0]["advantage"] == -1.
    assert result["excluded_groups"][0]["reason"] == "missing_or_nonfinite_score"


def shared_broker():
    from collections import deque
    import threading
    from runtime.serving.parallel import ScoringBroker
    broker = ScoringBroker.__new__(ScoringBroker)
    broker.cfg = {"request_admission": "round_robin_by_trajectory", "dycon_candidate_portion": 1,
                  "dynamic_branch_admission": True, "max_inflight_branches": 1,
                  "branch_count": 4, "admission_timeout_seconds": 2}
    broker.condition = threading.Condition()
    broker.closed = broker.run_cancelled = False
    broker.groups, broker.cancelled, broker.admission_waiters = {}, {}, {}
    broker.admission_turns = deque()
    broker.admission_portion_used = broker.peak_active = broker.saturated_waiting_peak = 0
    return broker


def test_shared_broker_round_robin_prevents_one_decision_draining_queue():
    broker = shared_broker()
    broker.reserve("dycon", [0, 1, 2])
    broker.reserve("baseline", [0])
    broker.acquire_branch("dycon", 0)
    admitted = []
    def request(group, branch):
        broker.acquire_branch(group, branch)
        admitted.append(group)
        broker.release_branch({"decision_id": group, "branch_id": branch})
    with ThreadPoolExecutor(3) as pool:
        futures = [pool.submit(request, "dycon", branch) for branch in (1, 2)]
        futures.append(pool.submit(request, "baseline", 0))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with broker.condition:
                if sum(map(len, broker.admission_waiters.values())) == 3:
                    break
            time.sleep(.005)
        else:
            pytest.fail("requests did not reach the shared queue")
        broker.release_branch({"decision_id": "dycon", "branch_id": 0})
        for future in futures:
            future.result(timeout=3)
    assert admitted == ["dycon", "baseline", "dycon"]
    assert broker.peak_active == 1


def test_shared_broker_cancellation_clears_waiter_and_unblocks_next_trajectory():
    from core.cancellation import Cancelled
    broker = shared_broker()
    broker.reserve("running", [0])
    broker.acquire_branch("running", 0)
    broker.reserve("cancelled", [0])
    broker.reserve("next", [0])
    with ThreadPoolExecutor(2) as pool:
        cancelled = pool.submit(broker.acquire_branch, "cancelled", 0)
        ready = pool.submit(broker.acquire_branch, "next", 0)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with broker.condition:
                if sum(map(len, broker.admission_waiters.values())) == 2:
                    break
            time.sleep(.005)
        else:
            pytest.fail("requests did not reach the shared queue")
        broker.cancel("cancelled")
        with pytest.raises(Cancelled):
            cancelled.result(timeout=2)
        assert not broker.admission_waiters.get("cancelled")
        broker.release_branch({"decision_id": "running", "branch_id": 0})
        assert ready.result(timeout=2)["acquired"]
    assert broker.peak_active == 1


def test_baseline_releases_shared_reservation_when_generation_fails():
    from rollout.online_admission import online_request_admission
    calls = []
    class Client:
        def call(self, action, payload):
            calls.append((action, payload["decision_id"]))
            return {}
    with pytest.raises(RuntimeError, match="generation"):
        with online_request_admission({"parallel": {"request_admission": "round_robin_by_trajectory"}},
                {"online_source": "baseline", "parallel_client": Client()}):
            raise RuntimeError("generation failed")
    assert [call[0] for call in calls] == ["reserve", "branch-acquire", "release"]
    assert len({call[1] for call in calls}) == 1


@pytest.mark.parametrize("limit,expected", [(1, 1), (4, 4), (20, 12)])
def test_native_backend_freezes_requested_trajectory_cap(tmp_path, monkeypatch, limit, expected):
    from RL.online_server import native_backend, request_digest
    adapter = tmp_path / "adapter"
    write_json(adapter / "identity.json", {})
    digest = sha256(adapter / "identity.json")
    monkeypatch.setattr("RL.checkpoints.verify_adapter", lambda *args, **kwargs: {
        "contract": {"base_fingerprint": "base", "lora": {"rank": 1}}})
    monkeypatch.setattr("runtime.serving.parallel.artifact_contract", lambda *args, **kwargs: {
        "sites": [3], "hidden_size": 2, "target": "target"})
    cfg = {"project_root": str(tmp_path), "model": {"fingerprint": "base"},
           "agent": {"backend": "minisweagent"},
           "capture": {"enabled": True, "export": {"enabled": True}, "logprobs": {"enabled": True}},
           "parallel": {"enabled": True, "mode": "parallel_consensus"},
           "serving": {"gpus": [3]}, "execution": {"rollout_workers": 12}}
    request = {"max_inflight": limit, "worker_ids": [3], "deployment": {
        "adapter": str(adapter), "adapter_identity_sha256": digest, "base_fingerprint": "base",
        "policy_fingerprint": request_digest({"base": "base", "adapter": digest}),
        "estimator_manifest": "predictor", "estimator_manifest_sha256": "sha",
        "feature": {"site": "layer_3", "width": 2, "target": "target"},
        "advantage_settings": {}, "branch_count": 2,
        "request_admission": "round_robin_by_trajectory", "dycon_candidate_portion": 1}}
    frozen = native_backend(cfg).bind_deployment(request)
    assert frozen["execution"]["rollout_workers"] == expected
    assert frozen["execution"]["selection_retry"] == {"enabled": True, "max_retries": 3}
    assert cfg["execution"]["rollout_workers"] == 12
    assert "selection_retry" not in cfg["execution"]

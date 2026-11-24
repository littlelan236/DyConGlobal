"""Cross-stage inventory, reward and actual per-turn objective contracts."""

import copy
from types import SimpleNamespace

import pytest

from core.io import hash_json, read_json, sha256, write_json
from rollout.key_steps import service
from rollout.key_steps.state import append


def test_select_equal_k_common_u_original_eight_members_and_immutable_output(tmp_path, monkeypatch):
    root = tmp_path
    source = {
        "rollout_id": "mother-1",
        "source_run_id": "mother",
        "attempt_dir": str(root / "source"),
        "task_key": "t",
        "instance_id": "org__1",
    }
    values = [0.05, 0.05, 0.05, 0.6, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7, 0.05, 0.05, 0.05, 0.05, 0.05]
    boundaries = [{"step": t} for t in range(len(values))]
    for b in boundaries:
        append(root / "source/replay-boundaries.jsonl", b)
    source["boundary_sha256"] = hash_json(boundaries)
    settings = {"mode": "source"}
    service.publish(
        root / "plan.json",
        {
            "schema": "key-step-plan-v1",
            "source_run": "mother",
            "source_config_hash": "hash",
            "settings": settings,
            "sources": [source],
        },
    )
    records = []
    for t, v in enumerate(values):
        candidates = [
            {
                "branch_id": b,
                "score": v if b % 2 else -v,
                "native_request_id": f"{t}-{b}",
                "response": {"id": f"{t}-{b}"},
                "ledger_ref": "ledger",
                "generation_entropy": {
                    "semantics": "sampler_distribution_entropy_nats_v1",
                    "entropy_mean": float(t),
                    "entropy_count": 3,
                },
            }
            for b in range(8)
        ]
        records.append(
            {
                "step": t,
                "message_index": 2 + t * 2,
                "decision": {"candidates": candidates},
                "decision_path": "decision",
                "decision_sha256": "sha",
                "ledger_root": "ledger-root",
            }
        )
    scan = root / "runs/scan/rollout/tasks/org__1/repetitions/rep-0"
    write_json(scan / "result.json", {"attempt_id": "a0"})
    write_json(
        scan / "attempts/a0/replay-validation.json",
        {"boundaries": [{"step": t, "eligible": t != 9} for t in range(len(values))]},
    )
    scfg = {"config_hash": "hash", "selection": {"tasks": {"t": {"instance": {"repo": "org/repo"}}}}}
    scan_plan = {
        "config": {"key_steps": {"mode": "scan", "inventory": {"source_run": "mother"}}},
        "work_items": {
            "items": [{"instance_id": "org__1", "repetition_id": "rep-0", "continuation": {"source": source}}]
        },
    }
    monkeypatch.setattr(
        "application.execution.run_plan.load_plan",
        lambda root, r: scan_plan if r == "scan" else {"config": scfg},
    )
    monkeypatch.setattr(service, "source_decisions", lambda *args: ({}, copy.deepcopy(records)))
    report = service.select({"project_root": str(root)}, root / "plan.json", "scan", root / "selected")
    inventory = read_json(root / "selected/inventory.json")
    audit = inventory["audits"][0]
    assert audit["K"] == 1 and audit["methods"]["adaptive"] == [3]
    assert 9 not in audit["U"]
    assert all(len(steps) == 1 and set(steps) <= set(audit["U"]) for steps in audit["methods"].values())
    assert len(inventory["children"]) == 8 * len(set().union(*map(set, audit["methods"].values())))
    assert len({x["child_id"] for x in inventory["children"]}) == report["children"]
    service.select({"project_root": str(root)}, root / "plan.json", "scan", root / "selected")
    records[0]["decision"]["candidates"][0]["generation_entropy"]["entropy_mean"] = 1000.0
    with pytest.raises(ValueError, match="Immutable"):
        service.select({"project_root": str(root)}, root / "plan.json", "scan", root / "selected")


def test_seal_terminal_rewards_native_coverage_zero_variance_and_paired_failure(tmp_path, monkeypatch):
    run = tmp_path / "run"
    items = []
    rollouts = {}
    outcomes = {}
    for parent in range(3):
        for branch in range(8):
            key = f"{parent}-{branch}"
            rep = "rep-" + key
            child = {
                "parent_id": f"p{parent}",
                "child_id": key,
                "branch": branch,
                "methods": ["adaptive", "random", "entropy"],
                "ledger_root": "root",
                "decision_path": "decision",
                "source": {"rollout_id": "mother" if parent < 2 else "other"},
            }
            item = {"instance_id": "org__1", "repetition_id": rep, "sample_key": key, "continuation": child}
            items.append(item)
            # p0 has mixed terminal rewards, p1 contains one infra-null member;
            # both are excluded together. p2 remains a valid zero-variance group.
            result = {"attempt_id": "a0", "disposition": "accepted", "exit_status": "Submitted"}
            if parent == 1 and branch == 7:
                result.update(disposition="failed", exit_status="infrastructure_exhausted")
            write_json(run / "rollout/tasks/org__1/repetitions" / rep / "result.json", result)
            rollouts[key] = {
                "sample_key": key,
                "attempt_id": "a0",
                "selected_evaluation_id": "evaluation",
                "task_key": "t",
                "submission_status": "submitted",
                "evaluation_status": "completed",
                "resolved": bool(branch % 2) if parent < 2 else False,
                "trajectory_path": "trajectory",
                "trajectory_sha256": "sha",
            }
            outcomes[key] = {
                "status": "completed",
                "resolved": rollouts[key]["resolved"],
                "verifier_audit": {
                    "classification": "ValidOutcome",
                    "effective_resolved": rollouts[key]["resolved"],
                },
            }
            if parent == 1 and branch == 6:
                outcomes[key]["verifier_audit"]["classification"] = "Review"
    loaded = {
        "run": run,
        "config": {"key_steps": {"mode": "continuation"}, "config_hash": "hash"},
        "work_items": {"items": items},
    }
    monkeypatch.setattr("application.execution.run_plan.load_plan", lambda *a: loaded)
    monkeypatch.setattr(
        "application.execution.evidence.RunEvidence.read",
        lambda self: {
            "rollouts": rollouts,
            "evaluations": {"evaluation": {"outcomes": outcomes}},
            "tasks": {"t": {"benchmark": "swebench"}},
        },
    )
    monkeypatch.setattr("RL.dataset.continuation.member_turns", lambda *a: [{"completion_token_ids": [1, 2]}])
    service.seal({"project_root": str(tmp_path)}, "child", tmp_path / "sealed")
    groups = read_json(tmp_path / "sealed/groups.json")["groups"]
    assert groups[0]["complete"] and not groups[0]["zero_variance"]
    assert not groups[0]["paired_eligible"] and not groups[0]["train_eligible"]
    assert groups[1]["members"][7]["reward"] is None
    assert groups[1]["members"][6]["reward"] is None  # known-invalid harness verdict never becomes reward
    assert groups[2]["zero_variance"] and groups[2]["paired_eligible"] and not groups[2]["train_eligible"]
    assert all(m["advantage"] == 0 for m in groups[2]["members"])


def test_multiturn_reader_preserves_prompts_and_rejects_tampering(tmp_path, monkeypatch):
    from RL.dataset.continuation import member_turns

    decision = {"candidates": [{"branch_id": 0, "ledger_ref": "old", "ledger_sha256": "x"}]}
    write_json(tmp_path / "decision.json", decision)
    native = {
        "old": {
            "request_id": "old-id",
            "prompt_token_ids": [10, 11],
            "output_token_ids": [12],
            "raw_t1_probability": {},
        },
        "new": {
            "request_id": "new-id",
            "prompt_token_ids": [10, 11, 12, 13, 14],
            "output_token_ids": [15, 16],
            "raw_t1_probability": {},
        },
    }
    # Nonempty descriptor required even when the vector loader is mocked here.
    for n in native.values():
        n["raw_t1_probability"] = {"status": "sealed"}
    trajectory = {
        "messages": [
            {
                "role": "assistant",
                "extra": {
                    "continuation_source": {"child_id": "child", "native_request_id": "old-id"},
                    "response": {"id": "old-id"},
                },
            },
            {"role": "tool", "content": "observation"},
            {
                "role": "assistant",
                "extra": {"response": {"id": "new-id", "ledger_ref": "new", "ledger_root": str(tmp_path)}},
            },
        ]
    }
    write_json(tmp_path / "trajectory.json", trajectory)
    member = {
        "child_id": "child",
        "branch_id": 0,
        "trajectory_ref": "trajectory.json",
        "trajectory_sha256": sha256(tmp_path / "trajectory.json"),
        "advantage": 1.0,
        "source_ref": {
            "source_native_request_id": "old-id",
            "decision_path": str(tmp_path / "decision.json"),
            "decision_sha256": hash_json(decision),
            "ledger_root": str(tmp_path),
        },
    }
    monkeypatch.setattr("RL.dataset.continuation.read_request", lambda root, desc: native[desc["ledger_ref"]])
    monkeypatch.setattr("runtime.serving.raw_logprobs.read_values", lambda *a, **k: None)
    turns = member_turns(member, tmp_path)
    assert turns[0]["loss_mask"] == [False, False, True]
    without_advantage = {k: v for k, v in member.items() if k != "advantage"}
    assert (
        member_turns(without_advantage, tmp_path)[0]["advantage"] is None
    )  # seal audits before rewards reduce
    assert turns[1]["prompt_token_ids"] == [10, 11, 12, 13, 14]
    assert turns[1]["loss_mask"] == [False] * 5 + [True] * 2
    write_json(tmp_path / "decision.json", {"candidates": []})
    with pytest.raises(ValueError, match="decision changed"):
        member_turns(member, tmp_path)


def test_multiturn_objective_equal_child_weight_and_gradients():
    torch = pytest.importorskip("torch")
    from RL.dataset.continuation import objective

    theta = [
        [torch.tensor([-0.4] * (i + 1), requires_grad=True), torch.tensor([-0.6], requires_grad=True)]
        for i in range(8)
    ]
    reference = [[a.detach() - 0.02 for a in turns] for turns in theta]
    advantages = [-1.0, 1.0] * 4
    actual = objective(theta, reference, advantages, kl_beta=0.0)
    expected = (
        -sum(
            torch.exp(torch.cat(t) - torch.cat(r)).mean() * adv
            for t, r, adv in zip(theta, reference, advantages)
        )
        / 8
    )
    assert actual.loss.item() == pytest.approx(expected.item())
    actual.loss.backward()
    for i, turns in enumerate(theta):
        for t in turns:
            assert t.grad.tolist() == pytest.approx(
                [-advantages[i] * torch.exp(torch.tensor(0.02)).item() / 8 / (i + 2)] * len(t)
            )


@pytest.mark.parametrize("branch_id", range(8))
def test_restore_injects_each_original_candidate_once_and_never_reads_future(
    cfg, tmp_path, monkeypatch, branch_id
):

    from core.deadline import Deadline
    from rollout.agent import DurableAgent
    from rollout.model import make_model

    model = make_model(cfg, Deadline())
    executed = []
    env = SimpleNamespace(
        get_template_vars=lambda **kw: {},
        serialize=lambda: {},
        execute=lambda action, *a, **kw: (
            executed.append(action["command"]),
            {"output": "ok", "returncode": 0, "exception_info": ""},
        )[1],
    )
    agent = DurableAgent(
        model,
        env,
        system_template="system",
        instance_template="{{task}}",
        step_limit=2,
        output_path=tmp_path / "child.json",
    )
    history = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
    write_json(
        tmp_path / "mother.json", {"messages": history + [{"role": "user", "content": "FUTURE_SECRET"}]}
    )
    source = {
        "attempt_dir": str(tmp_path),
        "boundary_sha256": hash_json([{"step": 1}]),
        "trajectory_path": str(tmp_path / "mother.json"),
        "trajectory_sha256": sha256(tmp_path / "mother.json"),
        "trajectory_bytes": (tmp_path / "mother.json").stat().st_size,
    }
    append(tmp_path / "replay-boundaries.jsonl", {"step": 1})
    response = {
        "id": "original",
        "object": "chat.completion",
        "created": 0,
        "model": "fixture",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command":"echo branch' + str(branch_id) + '"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }
    native = {
        "request_id": "native-" + str(branch_id),
        "branch_id": branch_id,
        "step_id": 1,
        "output_token_ids": [1, 2, 3],
    }
    write_json(tmp_path / "native.json", native)
    candidate = {
        "branch_id": branch_id,
        "native_request_id": native["request_id"],
        "ledger_ref": "native.json",
        "ledger_sha256": sha256(tmp_path / "native.json"),
        "response": response,
        "cost": 0.1,
    }
    decision = {"candidates": [candidate]}
    write_json(tmp_path / "decision.json", decision)
    boundary = {
        "step": 1,
        "message_index": 2,
        "history_sha256": hash_json(model._prepare_messages_for_api(history)),
        "agent": {
            "n_calls": 1,
            "cost": 0.25,
            "format_errors": 0,
            "loop_history": [],
            "loop_step": 1,
            "extra_template_vars": {"task": "task"},
            "elapsed_seconds": 3.0,
            "loop_escape": {"decisions": 1, "slots": {"0": {"count": 1, "action": {}, "reasoning": {}}}},
        },
    }
    monkeypatch.setattr(service, "replay", lambda *a: boundary)
    child = {
        "source": source,
        "step": 1,
        "branch": branch_id,
        "decision_path": str(tmp_path / "decision.json"),
        "decision_sha256": hash_json(decision),
        "ledger_root": str(tmp_path),
        "child_id": "child",
        "parent_id": "parent",
    }
    result = service.continue_agent(agent, {"mode": "continuation"}, child, tmp_path / "continuation")
    assert result["exit_status"] == "LimitsExceeded"
    assert executed == ["echo branch" + str(branch_id)] and agent.n_calls == 2
    assert agent.cost == pytest.approx(0.35)
    assert all("FUTURE_SECRET" not in str(m) for m in agent.messages)
    stored = read_json(tmp_path / "child.json")
    assert stored["continuation"]["prefix_message_count"] == 2
    assert stored["messages"][0]["extra"]["continuation_source"]["native_request_id"] == native["request_id"]
    assert len(stored["messages"]) == 3  # injected response, actual tool result, terminal exit

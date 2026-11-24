from pathlib import Path

import pytest
import yaml

from catalog.store import empty_manifest
from core.config import resolve
from core.io import write_json


@pytest.fixture
def cfg(tmp_path):
    raw = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text())
    raw.pop("parallel", None)  # Unit fixtures do not depend on a GPU acceptance config.
    raw["benchmark"].update(backend="swebench", snapshot_ref=None)
    raw["agent"].update(backend="minisweagent", profile_ref=None)
    raw["agent"].pop("condenser_max_tokens", None)  # Generic fixtures retain legacy profile policy.
    raw["execution"]["sample_completion_token_budget"] = None
    raw["timeouts"]["model_seconds"] = 28800  # Match default parallel TTL in synthetic fixtures.
    raw["generation"].update(thinking_token_budget=None, require_reasoning_close=False)
    raw["paths"]["artifact_roots"] = {}
    raw["paths"]["artifact_scan_roots"] = []
    raw["environment"].pop("docker_data_root", None)  # Host Docker paths are not test fixtures.
    raw["environment"]["wheelhouse"] = None  # Published host caches are not unit fixtures.
    raw["model"].update(path=None, fingerprint="fixture-model", served_name="fixture")
    raw["capture"]["enabled"] = False
    raw["capture"].setdefault("logprobs", {})["enabled"] = False

    if raw.get("harbor"):
        raw["harbor"].pop("terminal_policy", None)
        raw["harbor"].pop("task_python_index", None)
        raw["harbor"].pop("task_apt_mirror", None)
        raw["harbor"].pop("task_resource_cache", None)
        raw["harbor"].pop("task_resource_cache_sha256", None)
        raw["harbor"].pop("task_preparation", None)  # Host Terminal-Bench policy is not a generic SWE fixture.
    raw["hidden"]["adapter"] = "vllm_eager"
    raw["hidden"]["vllm"] = {"gpus": [0], "tensor_parallel_size": 1}
    raw["steering"].update(enabled=False, payload=None)
    raw["serving"].update(
        management="external", gpus=[0], tensor_parallel_size=1,
        trajectory_sharding="none", replica_base_urls=[], max_cpu_workers_per_gpu=1,
    )
    raw["images"].update(
        reserve_bytes=0,
        minimum_load_bytes=1,
        max_reserved_bytes=10000,
        expansion_factor=1.0,
        admission_seconds=2,
    )
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(raw))
    write_json(tmp_path / "catalog/manifest.json", empty_manifest())
    return resolve(tmp_path / "config.yaml")


class ToyTokenizer:
    name_or_path = "fixture"

    def encode(self, text, add_special_tokens=False, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(
            "".join(
                f"{m['role']}:{m.get('content') or ''}:{m.get('reasoning_content') or ''}" for m in messages
            )
            + "assistant:<think>\n"
        )


def task(iid="t-1", partition="train"):
    return {
        "benchmark": "fixture",
        "revision": "v1",
        "instance_id": iid,
        "repo": "org/repo",
        "base_commit": "abc",
        "partition": partition,
        "instance": {
            "instance_id": iid,
            "repo": "org/repo",
            "base_commit": "abc",
            "problem_statement": "Fix issue",
        },
    }


def import_fixture(cfg, *, resolved=None, iid="t-1"):
    from application.execution.evidence import RunEvidence
    from catalog.store import Catalog
    from rollout.artifacts import import_rollout

    root = Path(cfg["project_root"])
    Catalog(root).register_tasks([task(iid)])
    raw = root / (iid + ".traj.json")
    write_json(
        raw,
        {
            "instance_id": iid,
            "info": {"exit_status": "Submitted", "submission": "patch"},
            "messages": [
                {"role": "system", "content": "agent"},
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "reasoning_content": "reason-1",
                    "content": "FUTURE_ACTION_SECRET",
                    "tool_calls": [],
                    "extra": {
                        "response": {
                            "choices": [{"finish_reason": "tool_calls"}],
                            "usage": {"completion_tokens_details": {"reasoning_tokens": 3}},
                        }
                    },
                },
                {"role": "tool", "content": "OBSERVATION_SECRET"},
                {
                    "role": "assistant",
                    "reasoning_content": "reason-2",
                    "content": "submit",
                    "tool_calls": [],
                    "extra": {
                        "response": {
                            "choices": [{"finish_reason": "stop"}],
                            "usage": {"completion_tokens_details": {"reasoning_tokens": 2}},
                        }
                    },
                },
            ],
        },
    )
    rid = import_rollout(
        root,
        {
            "source": str(raw),
            "task": task(iid),
            "run_id": "source",
            "attempt_id": "a0",
            "purposes": ["training", "baseline"],
            "model_fingerprint": "fixture-model",
        },
    )
    if resolved is not None:
        row = RunEvidence(root, "source").read()["rollouts"][rid]
        report = root / f"runs/source/harness/e-{iid}/summary.json"
        outcomes = {rid: {"patch_sha256": row["patch_sha256"], "status": "completed", "resolved": resolved}}
        write_json(report, outcomes)
        RunEvidence(root, "source").record_evaluation("e-" + iid, report, outcomes, {"fixture": True})
    return rid

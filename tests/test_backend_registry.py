import copy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from application.execution.run_plan import create_run, load_plan
from catalog.store import Catalog
from core.deadline import Deadline
from core.io import hash_json, read_json, write_json
from runtime.backends.registry import adapter_for, from_snapshot, resolve
from tests.conftest import task


@pytest.mark.parametrize("benchmark,agent,adapter,mode", [
    ("swebench", "minisweagent", "native_swe", "replay"),
    ("swebench", "openhands", "native_swe_openhands", "replay"),
    ("skillsbench", "openhands", "benchflow", "live"),
    ("webarena_verified", "minisweagent", "native_webarena", "artifact_offline"),
])
def test_legacy_binding_derivation_never_mutates_config(benchmark, agent, adapter, mode):
    cfg = {"benchmark": {"backend": benchmark}, "agent": {"backend": agent}}
    before = copy.deepcopy(cfg)
    digest = hash_json(cfg)
    binding = from_snapshot(cfg)
    assert (binding.adapter, binding.evaluation_mode) == (adapter, mode)
    assert cfg == before and hash_json(cfg) == digest


def test_bindings_reject_unknown_task_families_and_changed_runtime(cfg):
    with pytest.raises(ValueError, match="another benchmark family"):
        resolve(cfg, {"unknown": {"benchmark": "terminalbench_2_1"}})
    cfg["execution_binding"] = {**resolve(cfg).as_dict(), "adapter_revision": "other"}
    with pytest.raises(ValueError, match="incompatible"):
        adapter_for(cfg)


def test_registry_and_facades_import_without_optional_sdks():
    code = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'harbor', 'benchflow', 'docker', 'minisweagent', 'openhands'}:
            raise AssertionError('eager SDK import: ' + fullname)
sys.meta_path.insert(0, Block())
from runtime.backends.registry import adapter_for
for benchmark, agent in [('skillsbench', 'openhands'), ('swebench', 'minisweagent'), ('swebench', 'openhands'), ('webarena_verified', 'minisweagent')]:
    adapter_for({'benchmark': {'backend': benchmark}, 'agent': {'backend': agent}})
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_new_run_freezes_binding_and_load_does_not_consult_yaml(cfg):
    Catalog(cfg["project_root"]).register_tasks([task()])
    run = create_run(cfg, "binding", operation="rollout")
    assert read_json(run / "resolved-config.json")["execution_binding"] == resolve(cfg).as_dict()
    before = {p.name: p.read_bytes() for p in (run / "resolved-config.json", run / "selection.json", run / "plan.json")}
    Path(cfg["config_source"]).write_text("agent: changed\nbenchmark: changed\n")
    loaded = load_plan(cfg["project_root"], "binding")
    assert loaded["config"]["execution_binding"]["adapter"] == "native_swe"
    assert before == {name: (run / name).read_bytes() for name in before}


def test_cleanup_is_idempotent_but_a_failed_cleanup_can_retry(cfg):
    adapter = adapter_for(cfg)
    calls = []

    def cleanup(run_id):
        calls.append(run_id)
        if len(calls) == 1:
            raise OSError("temporary cleanup failure")

    context = adapter.open_run(cfg, "cleanup", SimpleNamespace(cleanup_owned=cleanup))
    with pytest.raises(OSError):
        adapter.close_run(context)
    assert not context.closed
    assert adapter.close_run(context).outcome == "completed"
    assert adapter.close_run(context).outcome == "already_closed"
    assert calls == ["cleanup", "cleanup"]


def test_resume_selects_factory_from_frozen_config_after_root_pair_drift(cfg, monkeypatch):
    pytest.importorskip("minisweagent")
    import core.model_identity
    import rollout.agent
    import rollout.model
    import runtime.containers.rollout_adapter
    import runtime.serving.manager
    import runtime.serving.startup
    from rollout.runner import run_manifest
    from runtime.backends.native.swe import NativeSWEAdapter

    Catalog(cfg["project_root"]).register_tasks([task()])
    run = create_run(cfg, "drift", operation="rollout")
    frozen_bytes = (run / "resolved-config.json").read_bytes()
    caller = copy.deepcopy(cfg)
    caller["benchmark"]["backend"] = "skillsbench"
    caller["agent"]["backend"] = "openhands"
    observed = []

    class Agent:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, problem):
            assert problem == "Fix issue"
            return {"exit_status": "Submitted", "submission": "patch"}

        def save(self, path, extra):
            write_json(path, {**extra, "messages": [{"role": "assistant", "content": "done"}]})

    class Env:
        def __init__(self, *args):
            pass

        def close(self):
            pass

    class Service:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def make_factory(self, frozen, run_id):
        observed.append((frozen["benchmark"]["backend"], frozen["agent"]["backend"]))
        return SimpleNamespace(deadline=Deadline(), lock_images=lambda _: None,
                               cleanup_owned=lambda _: None)

    monkeypatch.setattr(core.model_identity, "verify_model", lambda _: None)
    monkeypatch.setattr(NativeSWEAdapter, "make_factory", make_factory)
    monkeypatch.setattr(rollout.agent, "DurableAgent", Agent)
    monkeypatch.setattr(rollout.model, "make_model", lambda *args: None)
    monkeypatch.setattr(runtime.containers.rollout_adapter, "RolloutEnvironment", Env)
    monkeypatch.setattr(runtime.serving.manager, "ServingManager", Service)
    monkeypatch.setattr(runtime.serving.startup, "check_tool_chain", lambda *args: None)
    result = run_manifest(caller, "drift", resume=True)
    assert observed == [("swebench", "minisweagent")]
    assert len(result) == 1
    assert next(iter(result.values()))["agent_exit_status"] == "Submitted"
    assert frozen_bytes == (run / "resolved-config.json").read_bytes()

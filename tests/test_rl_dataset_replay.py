"""Offline replay contracts with schema-real synthetic captures and bounded fake engines."""
import copy
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

from core.io import hash_json, read_json, sha256, write_json
from RL.dataset.artifacts import publication, seal, verify_local
from RL.dataset.config import ReplayConfig, load
from RL.dataset.loader import collate_group, iter_groups
from RL.dataset.materialize import materialize
from RL.dataset.replay.artifacts import ReplayWriter, ReplayReader, check_replay, write_replay
from RL.dataset.replay.service import dry_run, replay_logprobs, request_policy
from RL.dataset.replay.validation import effective_policy, raw_contract, raw_policy, root_sampling, target_logprobs
from RL.dataset.replay.worker import execute_batches, sampling_options
from test_rl_dataset import token_source as original_token_source, build
from RL.dataset.build import build_dataset, check_dataset


def token_source(tmp_path, **kwargs):
    fixture = original_token_source(tmp_path, **kwargs)
    source_file = tmp_path / "runs" / fixture[0]["run"] / "resolved-config.json"
    source = read_json(source_file); source["serving"]["dtype"] = "float16"
    write_json(source_file, source)
    return fixture


ROOT = {"generation": {"temperature": .6, "top_p": .95, "top_k": 20}, "config_source_sha256": "root-sha"}


def replay_config(ds, tmp_path):
    return {"schema_version": 2, "probability_semantics": "raw_t1_output_tokens_v1", "replay_id": "unit-replay", "dataset": {"id": ds.name, "root": str(ds.parents[2])},
            "runtime": {"gpus": [0], "max_model_len": 32}, "storage": {"output_root": str(tmp_path/"replayed"), "max_shard_bytes": 1024}}


def test_raw_contract_is_independent_of_root_and_history():
    assert raw_contract()["temperature"] == 1.0
    assert raw_contract()["top_p"] == 1.0 and raw_contract()["top_k"] == -1
    assert raw_contract()["semantics"] == "raw_t1_output_tokens_v1"
    assert raw_policy()["policy_sha256"] == raw_policy()["policy_sha256"]
    cfg = ReplayConfig.model_validate(replay_config(Path("/tmp/RL/datasets/dataset"), Path("/tmp")))
    assert cfg.sampling_semantics == "raw_t1_output_tokens_v1"
    with pytest.raises(ValueError, match="read-only"):
        ReplayConfig.model_validate({**replay_config(Path("/tmp/RL/datasets/dataset"), Path("/tmp")), "schema_version": 1})


def test_raw_score_target_value_is_selected_by_native_token():
    logits = np.array([2.0, 1.0, -3.0], dtype=np.float64)
    expected = logits[1] - np.log(np.exp(logits).sum())
    output = NS(finished=True, outputs=[NS(token_ids=[1], logprobs=[{1:NS(logprob=expected), 2:NS(logprob=-5)}])])
    assert target_logprobs(output, [1]) == pytest.approx([expected], abs=0, rel=0)


def outputs(ds, zero=False):
    return {r["branch_sample_id"]: ({"token_ids": r["completion_token_ids"]}, [-1/3, -float("inf") if zero else -.234567891234, -1.23456789])
            for g in iter_groups(ds) for r in g["members"]}


def test_float32_roundtrip_loader_and_receipt_binding(tmp_path):
    ds = build(token_source(tmp_path)); data = outputs(ds)
    result = write_replay(ds, "unit-replay", ROOT, data, tmp_path/"replayed", max_shard_bytes=1024)
    path = Path(result["path"])
    assert check_replay(path, dataset_path=ds, with_values=True)["tokens"] == 18
    with pytest.raises(ValueError, match="estimates require explicit"):
        list(iter_groups(ds, replay=path))
    groups = list(iter_groups(ds, replay=path, allow_replay_estimates=True))
    np.testing.assert_array_equal(groups[0]["members"][0]["behavior_logprobs"], np.asarray([-1/3, -.234567891234, -1.23456789], dtype=np.float32))
    assert not read_json(path/"replay.json")["historical_behavior_verified"]
    shard = next(path.glob("results/*/shards/*"))
    with shard.open("r+b") as stream: stream.write(b"bad!")
    with pytest.raises(ValueError, match="inventory/hash"):
        check_replay(path, with_values=True)


def test_zero_support_is_stored_but_entire_group_is_unavailable(tmp_path):
    ds = build(token_source(tmp_path))
    with pytest.raises(ValueError, match="Zero-support"):
        write_replay(ds, "zero", ROOT, outputs(ds, zero=True), tmp_path/"replayed")
    assert not (tmp_path/"replayed/RL/replays/zero/receipt.json").exists()


@pytest.mark.parametrize("bad", ["token", "short", "nan", "positive", "extra"])
def test_invalid_replay_never_publishes_receipt(tmp_path, bad):
    ds = build(token_source(tmp_path)); data = outputs(ds); bid = next(iter(data))
    output, values = data[bid]
    if bad == "token": output["token_ids"] = [99]*3
    if bad == "short": values.pop()
    if bad == "nan": values[0] = float("nan")
    if bad == "positive": values[0] = .2
    if bad == "extra": data["unbound"] = output, values
    with pytest.raises(ValueError): write_replay(ds, "bad", ROOT, data, tmp_path/"replayed")
    assert not (tmp_path/"replayed/RL/replays/bad/receipt.json").exists()


def test_target_gather_uses_token_key_not_dict_insertion_order():
    obj = NS(finished=True, outputs=[NS(token_ids=[3, 4], logprobs=[{99:NS(logprob=-9),3:NS(logprob=-.4)}, {4:NS(logprob=-.8)}])])
    assert target_logprobs(obj, [3,4]) == [-.4, -.8]
    with pytest.raises(ValueError, match="token IDs"):
        target_logprobs(obj, [3,5])
    obj.outputs[0].logprobs[1] = {99:NS(logprob=-.2)}
    with pytest.raises(ValueError, match="target token missing"):
        target_logprobs(obj, [3,4])


@pytest.mark.parametrize("field,value", [("thinking_token_budget",12),("require_reasoning_close",True),("structured_outputs",{"grammar":"x"}),("bad_words",["a"]),("stop",["end"]),("n",2)])
def test_constraints_refused_before_engine(field, value):
    source = {"generation": {field:value}, "model": {}}
    with pytest.raises(ValueError, match="Unsupported historical"):
        effective_policy(source, root_sampling(ROOT))


def test_historical_model_and_root_sampling_preflight(tmp_path):
    ds = build(token_source(tmp_path)); cfg = ReplayConfig.model_validate(replay_config(ds, tmp_path))
    result = dry_run(ds, ROOT, cfg)
    assert result["branches"] == 6 and result["completion_tokens"] == 18
    assert result["historical_models"][0]["model"]["fingerprint"] == "unit-model"
    assert result["model_load"] is False and result["receipt"] is False
    with pytest.raises(ValueError, match="sampling conflict"):
        effective_policy({"generation": {"temperature": .8}}, root_sampling(ROOT))
    source = {"generation": {"temperature": .1}}
    assert effective_policy(source, root_sampling(ROOT), {"temperature":.6})["parameters"]["temperature"] == .6


def test_audit_off_does_not_replace_actual_branch_prompts_in_loader_or_cache(tmp_path):
    fixture = token_source(tmp_path, bad_prompt=True); f, file, config = fixture
    config["audit"]["enabled"] = False; file.write_text(yaml.safe_dump(config))
    ds = build(fixture)
    direct = list(iter_groups(ds))[0]
    assert direct["members"][0]["prompt_token_ids"] == [10,11,12]
    assert direct["members"][1]["prompt_token_ids"] == [10,11,13]
    cache = materialize(ds, tmp_path/"cache")["path"]
    assert collate_group(list(iter_groups(ds, cache=cache))[0], as_tensors=False) == collate_group(direct, as_tensors=False)


def test_original_completion_first_last_and_right_padding(tmp_path):
    ds = build(token_source(tmp_path)); group = next(iter_groups(ds))
    group["members"][1]["completion_token_ids"].pop()
    row = group["members"][1]; row["completion_sha256"] = hash_json(row["completion_token_ids"])
    batch = collate_group(group, as_tensors=False)
    assert batch["labels"][0][3] == 20 and batch["labels"][0][-1] == 42
    assert batch["labels"][1][-1] == -100
    assert batch["attention_mask"][1][-1] == 0 and not batch["loss_mask"][1][-1]


def test_worker_bounded_batches_and_native_sampling_contract(tmp_path):
    ds = build(token_source(tmp_path)); rows = [r for g in iter_groups(ds) for r in g["members"]]
    policy = effective_policy({"generation":{}}, root_sampling(ROOT)); sizes=[]; saved=[]
    class Engine:
        def generate(self, prompts, params, use_tqdm):
            sizes.append(len(prompts)); result=[]
            for p in params:
                assert p["logprobs"] == 0 and p["prompt_logprobs"] is None and p["detokenize"] is False
                assert p["max_tokens"] == len(p["trace_decode_token_ids"])
                tokens = p["trace_decode_token_ids"]
                result.append(NS(finished=True, outputs=[NS(token_ids=tokens, logprobs=[{t:NS(logprob=-.5)} for t in tokens])]))
            return result
    class Writer:
        def put(self,row,output,values,**kw): saved.append(row["branch_sample_id"])
    execute_batches(Engine(), lambda **k:k, rows, 2, Writer(), lambda r:policy, "model-key")
    assert sizes == [2,2,2] and saved == [r["branch_sample_id"] for r in rows]


def test_interrupted_replay_recovers_only_sealed_branches(tmp_path):
    ds = build(token_source(tmp_path)); rows=[r for g in iter_groups(ds) for r in g["members"]]
    identity={"replay_id":"recover", "dataset_receipt_sha256":sha256(ds/"receipt.json"), "raw_contract_sha256":hash_json(raw_contract()), "selection_sha256":"test"}
    root=tmp_path/"replayed"
    with pytest.raises(RuntimeError, match="interrupted"):
        with publication(root,"rl-replay","recover",signature=identity) as (stage,_):
            writer=ReplayWriter(stage,ds,identity,1024,commit_every_branches=1)
            writer.put(rows[0],{"token_ids":rows[0]["completion_token_ids"]},[-.1]*3,policy_sha256="p",model_sha256="m")
            raise RuntimeError("interrupted")
    with publication(root,"rl-replay","recover",signature=identity,recover=True) as (stage,dest):
        writer=ReplayWriter(stage,ds,identity,1024,recover=True)
        assert writer.finished == {rows[0]["branch_sample_id"]}
        for row in rows[1:]: writer.put(row,{"token_ids":row["completion_token_ids"]},[-.1]*3,policy_sha256="p",model_sha256="m")
        writer.finish(root_sampling(ROOT),{}, {}, {})
        seal(stage,"rl-replay","recover")
    assert check_replay(dest,with_values=True)["branches"]==6


def test_formal_dispatcher_dry_run_and_cpu_end_to_end(tmp_path, monkeypatch):
    ds=build(token_source(tmp_path)); value=replay_config(ds,tmp_path); config_file=tmp_path/"replay.yaml"
    config_file.write_text(yaml.safe_dump(value))
    cfg={**ROOT,"project_root":str(tmp_path)}
    assert replay_logprobs(cfg,config_file,dry=True)["branches"]==6
    def dispatch(python,job_path,log,seconds,root):
        job=read_json(job_path); writer=ReplayWriter(job["stage"],ds,job["identity"],1024)
        for g in iter_groups(ds):
            for row in g["members"]: writer.put(row,{"token_ids":row["completion_token_ids"]},[-.1]*3,policy_sha256=next(iter(job['plan']['policies'])),model_sha256=next(iter(job['plan']['models'])))
        writer.finish(job["plan"]["sampling"],job["plan"]["policies"],job["plan"]["models"],{"status":"fake_engine_test"})
        Path(log).write_text("worker closed\n")
    monkeypatch.setattr("RL.dataset.replay.service.dispatch",dispatch)
    monkeypatch.setattr("hidden.backends.vllm.engine.gpu_reservation",lambda *a,**k:nullcontext())
    result=replay_logprobs(cfg,config_file)
    assert check_replay(result["path"],dataset_path=ds,with_values=True)["branches"]==6
    with pytest.raises(FileExistsError): replay_logprobs(cfg,config_file)


def test_runtime_interpreter_keeps_venv_symlink(tmp_path):
    ds=tmp_path/"out/RL/datasets/unit-dataset";value=replay_config(ds,tmp_path)
    interpreter=tmp_path/".venvs/latent-vllm/bin/python";interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to('/usr/bin/python3')
    value["runtime"]["python"]=str(interpreter)
    file=tmp_path/"replay.yaml";file.write_text(yaml.safe_dump(value))
    config,_=load(file,tmp_path,replay=True)
    assert config.runtime.python==str(interpreter)


@pytest.mark.parametrize("when",["partial_rollout","before_seal"])
def test_dataset_recovery_freezes_selection_and_discards_uncommitted_tail(tmp_path,monkeypatch,when):
    fixture=token_source(tmp_path);f,file,_=fixture
    import RL.dataset.build as module
    original_line=module.json_line;original_seal=module.seal
    def interrupted_line(stream,value):
        original_line(stream,value)
        if stream.name.endswith('branches.jsonl'): raise RuntimeError('interrupted')
    def interrupted_seal(*a,**k): raise RuntimeError('interrupted')
    if when=='partial_rollout': monkeypatch.setattr(module,'json_line',interrupted_line)
    else: monkeypatch.setattr(module,'seal',interrupted_seal)
    with pytest.raises(RuntimeError,match='interrupted'): build_dataset(f['cfg'],file)
    monkeypatch.setattr(module,'json_line',original_line);monkeypatch.setattr(module,'seal',original_seal)
    def no_reselect(*a,**k): raise AssertionError('Recovery must use frozen selection')
    monkeypatch.setattr(module,'select_sources',no_reselect)
    result=build_dataset(f['cfg'],file,recover=True)
    assert check_dataset(result['path'])['branches']==6
    assert len(list(iter_groups(result['path'])))==1


def test_formal_recovery_uses_frozen_root_sampling(tmp_path,monkeypatch):
    ds=build(token_source(tmp_path));file=tmp_path/'replay.yaml';file.write_text(yaml.safe_dump(replay_config(ds,tmp_path)))
    cfg={**ROOT,'project_root':str(tmp_path)}; calls=[]
    def dispatch(python,job_file,log,seconds,root):
        job=read_json(job_file); calls.append(job['plan']['sampling']['temperature'])
        writer=ReplayWriter(job['stage'],ds,job['identity'],1024,recover=job['recover'])
        for group in iter_groups(ds):
            for row in group['members']:
                if row['branch_sample_id'] in writer.finished:continue
                writer.put(row,{'token_ids':row['completion_token_ids']},[-.1]*3,policy_sha256=next(iter(job['plan']['policies'])),model_sha256=next(iter(job['plan']['models'])))
                if not job['recover']:raise RuntimeError('interrupted')
        writer.finish(job['plan']['sampling'],job['plan']['policies'],job['plan']['models'],{})
    monkeypatch.setattr('RL.dataset.replay.service.dispatch',dispatch)
    monkeypatch.setattr('hidden.backends.vllm.engine.gpu_reservation',lambda *a,**k:nullcontext())
    with pytest.raises(RuntimeError,match='interrupted'):replay_logprobs(cfg,file)
    changed={**cfg,'generation':{**ROOT['generation'],'temperature':.3},'config_source_sha256':'edited-root'}
    result=replay_logprobs(changed,file,recover=True)
    assert calls==[1.0,1.0]
    assert 'root_config_sha256' not in read_json(Path(result['path'])/'root-sampling.json')
    assert check_replay(result['path'],dataset_path=ds,with_values=True)['branches']==6


def test_recovery_rejects_tampered_committed_array(tmp_path):
    ds=build(token_source(tmp_path));row=next(iter_groups(ds))['members'][0]
    identity={'replay_id':'broken','dataset_receipt_sha256':sha256(ds/'receipt.json'), 'raw_contract_sha256':hash_json(raw_contract()), 'selection_sha256':'test'}
    root=tmp_path/'replayed'
    with pytest.raises(RuntimeError):
        with publication(root,'rl-replay','broken',signature=identity) as (stage,_):
            writer=ReplayWriter(stage,ds,identity,1024,commit_every_branches=1)
            writer.put(row,{'token_ids':row['completion_token_ids']},[-.2]*3,policy_sha256='p',model_sha256='m')
            raise RuntimeError('interrupted')
    shard=next(stage.glob('results/*/shards/*'))
    shard.write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        with publication(root,'rl-replay','broken',signature=identity,recover=True) as (stage,_):
            ReplayWriter(stage,ds,identity,1024,recover=True)


def test_unified_artifact_cli_for_dataset_cache_replay(tmp_path,capsys):
    import core.cli as cli
    ds=build(token_source(tmp_path));path=write_replay(ds,'cli-replay',ROOT,outputs(ds),tmp_path/'replayed')['path']
    cli.main(['artifact','verify','cli-replay','--scope','rl-replay','--storage-root',str(tmp_path/'replayed')])
    assert '"scope": "rl-replay"' in capsys.readouterr().out
    cli.main(['artifact','verify',ds.name,'--scope','rl-dataset','--storage-root',str(ds.parents[2])])
    assert '"branches": 6' in capsys.readouterr().out


def test_missing_full_decision_keeps_all_slots_as_unavailable(tmp_path):
    fixture=token_source(tmp_path);f,file,config=fixture
    config['audit']['enabled']=False;file.write_text(yaml.safe_dump(config))
    (f['session']/'parallel/decisions/dec-unit.json').unlink()
    ds=build(fixture)
    from RL.dataset.build import iter_rows
    rows=list(iter_rows(ds/'branches.jsonl'))
    assert len(rows)==6 and all(r['exclusion_reason']=='full_decision_unavailable' for r in rows)
    assert not list(iter_groups(ds))


def test_dispatch_timeout_terminates_worker_group(tmp_path,monkeypatch):
    from RL.dataset.replay.service import dispatch
    events=[]
    class Child:
        pid=43210
        def wait(self,timeout):
            events.append(('wait',timeout))
            if len([x for x in events if x[0]=='wait'])==1:raise subprocess.TimeoutExpired('worker',timeout)
            return 0
    monkeypatch.setattr('RL.dataset.replay.service.subprocess.Popen',lambda *a,**k:Child())
    monkeypatch.setattr('RL.dataset.replay.service.os.killpg',lambda pid,sig:events.append(('kill',sig)))
    with pytest.raises(subprocess.TimeoutExpired):dispatch('python',tmp_path/'job.json',tmp_path/'worker.log',1,tmp_path)
    import signal
    assert ('kill',signal.SIGTERM) in events and ('kill',signal.SIGKILL) in events


@pytest.mark.parametrize("field, value, blocked", [
    ("verify_identity", False, False),
    ("fingerprint", "different-checkpoint", True),
    ("path", "/different/checkpoint", True),
    ("trust_remote_code", True, True),
    ("dtype", "bfloat16", True),
    ("tokenizer", "different-tokenizer", True),
])
def test_replay_cohort_ignores_only_verification_control(tmp_path, monkeypatch, field, value, blocked):
    import RL.dataset.replay.service as service
    ds = build(token_source(tmp_path))
    group = next(iter_groups(ds))
    original_run = group["members"][0]["source_run_id"]
    source = read_json(ds / "source-evidence" / original_run / "resolved-config.json")
    other = copy.deepcopy(source)
    if field == "dtype":
        other["serving"]["dtype"] = value
    elif field != "tokenizer":
        other["model"][field] = value
    original_model = copy.deepcopy(source["model"])
    cloned = copy.deepcopy(group)
    for member in cloned["members"]:
        member["source_run_id"] = "other-run"
        if field == "tokenizer":
            member["tokenizer_fingerprint"] = value
    real_read = service.read_json
    def read(path):
        if str(path).endswith("source-evidence/other-run/resolved-config.json"):
            return other
        return real_read(path)
    monkeypatch.setattr(service, "read_json", read)
    monkeypatch.setattr(service, "selected_groups", lambda path, config, **kwargs: iter([group, cloned]))
    cfg = ReplayConfig.model_validate(replay_config(ds, tmp_path))
    plan = service.prepare(ds, ROOT, cfg)
    assert bool(plan["blocked"]) is blocked
    assert source["model"] == original_model
    if field == "verify_identity":
        assert len(plan["models"]) == 1
        assert "verify_identity" not in next(iter(plan["models"].values()))["model"]
        assert other["model"]["verify_identity"] is False


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_preflight_reads_native_tokens_only_when_enabled(tmp_path, monkeypatch, enabled):
    import RL.dataset.replay.service as service
    ds = build(token_source(tmp_path))
    value = replay_config(ds, tmp_path)
    if enabled:
        value["preflight"] = {"enabled": True}
    cfg = ReplayConfig.model_validate(value)
    assert cfg.preflight.enabled is enabled
    real_groups = service.selected_groups
    requested = []
    def groups(path, config, **kwargs):
        requested.append(kwargs["with_tokens"])
        return real_groups(path, config, **kwargs)
    def forbidden(*args):
        raise AssertionError("Default metadata preparation must not read request ledgers")
    monkeypatch.setattr(service, "selected_groups", groups)
    if not enabled:
        monkeypatch.setattr(service, "request_policy", forbidden)
    plan = service.prepare(ds, ROOT, cfg)
    assert requested == [enabled]
    assert plan["counts"]["branches"] == 6
    assert not plan["blocked"]

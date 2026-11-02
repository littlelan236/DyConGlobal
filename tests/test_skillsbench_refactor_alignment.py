"""Contracts retained from checkpoint 9810087 before 8c3be1e refactor."""
import pytest
from runtime.agents.openhands import driver_environment
from runtime.backends.registry import adapter_for

@pytest.mark.parametrize('remaining,condenser',[(28800,None),(28799,49152),(0.1,49152)])
def test_shared_driver_matches_pre_refactor_inline_environment(cfg,remaining,condenser):
    class Deadline:
        def remaining(self,budget): return min(budget,remaining)
    cfg['execution_context']={'profile':{'condenser_max_tokens':condenser}}
    model='openai/'+cfg['model']['served_name']
    expected={'DYCON_OPENHANDS_TERMINAL':'subprocess','BENCHFLOW_PROVIDER_BASE_URL':'http://bridge/v1',
       'LITELLM_LOCAL_MODEL_COST_MAP':'true','OPENHANDS_SUPPRESS_BANNER':'1',
       'BENCHFLOW_PROVIDER_API_KEY':'test','BENCHFLOW_PROVIDER_MODEL':model,
       'LLM_MODEL':model,'MODEL':model,
       'DYCON_OPENHANDS_MODEL_SECONDS':str(max(1,int(min(remaining,cfg['timeouts']['model_seconds']))))}
    if condenser: expected['OPENHANDS_CONDENSER_MAX_TOKENS']=str(condenser)
    assert driver_environment(cfg,'http://bridge/v1','test',Deadline())==expected

def test_benchflow_adapter_forwards_identical_attempt_objects(cfg,monkeypatch,tmp_path):
    from runtime.backends.benchflow.adapter import BenchFlowAdapter
    from runtime.backends.contracts import AttemptBinding
    from types import SimpleNamespace
    from runtime.agents import skillsbench
    factory=object();task={'benchmark':'skillsbench'};runtime={'serving_worker_endpoint':'http://worker/v1'}
    sample=object();identity={'attempt_id':'rep-001-attempt-001','effective_seed':43}
    ctx=SimpleNamespace(cfg=cfg,run_id='run',factory=factory)
    seen=[]
    monkeypatch.setattr(skillsbench,'execute_attempt',lambda *args:seen.append(args) or 'ok')
    assert BenchFlowAdapter(None).execute_attempt(ctx,AttemptBinding(task,sample,identity,tmp_path,runtime))=='ok'
    assert seen==[(cfg,'run',task,runtime,sample,identity,tmp_path,factory)]


import pytest

from core.io import read_json, write_json
from runtime.agents.openhands import OpenHandsDriverProfile, validate_profile
from runtime.agents.qualification import qualify_profile


@pytest.fixture
def probe(tmp_path, monkeypatch):
    import application.execution.run_plan as plans
    import application.execution.evidence as evidence
    import rollout.adapters.openhands as adapter
    run = tmp_path / 'runs/probe'
    base = OpenHandsDriverProfile(bridge_listen='127.0.0.1', bridge_advertise='127.0.0.1').model_dump()
    context = dict(benchmark='terminalbench_2_1', agent='openhands', profile=base,
                   profile_sha256='source', context_sha256='context', worker_protocol=1, runtime_sha256='runtime')
    monkeypatch.setattr(plans, 'load_plan', lambda *a: {'config': {'execution_context': context}, 'run': run})
    monkeypatch.setattr(evidence.RunEvidence, 'verify', lambda self: None)
    monkeypatch.setattr(adapter, 'parse_response', lambda response, request: response)
    request = run / 'rollout/tasks/task/requests/req-1.json'
    row = dict(role='agent_decision', binding={'profile_sha256': 'source', 'backend_family': 'harbor'},
               request_id='request', step_id=0, status='completed',
               request={'tools': [{'function': {'name': 'terminal'}}],
                        'messages': [{'role': 'tool', 'tool_call_id': 'call'}]},
               selected_response={'tool_calls': [{'id': 'call'}]})
    write_json(request, row)
    write_json(run / 'rollout/tasks/task/dependency-install.json', {'return_code': 0})
    write_json(run / 'rollout/tasks/task/openhands/acp-events.json', {'events': []})
    return tmp_path, context, request, row


@pytest.mark.parametrize('benchmark', ['terminalbench_2_1', 'skillsbench'])
def test_qualification_publishes_the_benchmark_profile_contract(probe, benchmark):
    root, context, _, _ = probe
    context['benchmark'] = benchmark
    if benchmark == 'skillsbench':
        context['profile'] = validate_profile(context['profile'])
    result = qualify_profile(root, 'probe', 'qualified')
    profile = read_json(root / result['profile_ref'])
    assert result['tool_result_pairs'] == 1
    assert result['qualified_requests'] == 1
    assert profile['request_roles'] == 'openhands-1.22.1'
    if benchmark == 'terminalbench_2_1':
        assert OpenHandsDriverProfile.model_validate(profile)
        assert 'benchflow_version' not in profile
        assert profile['qualification']['backend_family'] == 'harbor'
    else:
        assert profile == validate_profile(profile)
        assert profile['benchflow_version'] == '0.6.3'


@pytest.mark.parametrize('defect,expected', [
    ('role', 'Unqualified model request'),
    ('binding', 'another agent profile'),
    ('no_result', 'executed tool/result pairs'),
])
def test_terminal_qualification_rejects_invalid_actual_evidence(probe, defect, expected):
    root, _, path, row = probe
    if defect == 'role':
        row['role'] = 'unknown'
    elif defect == 'binding':
        row['binding']['profile_sha256'] = 'other'
    else:
        row['request']['messages'] = []
    write_json(path, row)
    with pytest.raises(ValueError, match=expected):
        qualify_profile(root, 'probe', 'rejected')
    assert not (root / 'runtime/agent-profiles/rejected').exists()


def test_harbor_qualification_rejects_native_transport_evidence(probe):
    root, _, path, row = probe
    row['binding']['backend_family'] = 'native'
    write_json(path, row)
    with pytest.raises(ValueError, match='actual Harbor request bindings'):
        qualify_profile(root, 'probe', 'rejected')

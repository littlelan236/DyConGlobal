import json

import pytest

from core.cli import main
from core.io import write_json
from runtime.agents import accepted_reward_audit


@pytest.mark.parametrize('complete,exit_code', [(True, None), (False, 3)])
def test_real_cli_dispatch_reports_completeness(tmp_path, monkeypatch, capsys, complete, exit_code):
    called = []
    def audit(root, run_id, output):
        called.append((root, run_id, output))
        return {'complete': complete, 'candidate_count': 1}
    monkeypatch.setattr(accepted_reward_audit, 'audit_accepted_rewards', audit)
    args = ['--config', str(tmp_path/'missing.yaml'), 'skillsbench', 'audit-rewards',
            'run', '--output', str(tmp_path/'report')]
    if exit_code:
        with pytest.raises(SystemExit) as error:
            main(args)
        assert error.value.code == exit_code
    else:
        main(args)
    assert json.loads(capsys.readouterr().out)['complete'] is complete
    assert called == [(tmp_path, 'run', tmp_path/'report')]


def test_real_cli_dispatch_rejects_invalid_run(tmp_path, monkeypatch, capsys):
    def fail(*args):
        raise ValueError('invalid frozen run')
    monkeypatch.setattr(accepted_reward_audit, 'audit_accepted_rewards', fail)
    with pytest.raises(SystemExit) as error:
        main(['--config', str(tmp_path/'missing.yaml'), 'skillsbench', 'audit-rewards',
              'run', '--output', str(tmp_path/'report')])
    assert error.value.code == 1
    assert 'invalid frozen run' in capsys.readouterr().err


def test_real_cli_dispatch_rejects_missing_output():
    with pytest.raises(SystemExit) as error:
        main(['skillsbench', 'audit-rewards', 'run'])
    assert error.value.code == 2


def setup_service(tmp_path, monkeypatch, *, rewards=(0, .5), benchmark='skillsbench'):
    run = tmp_path/'runs/run'
    run.mkdir(parents=True)
    rows = {}
    for i, reward in enumerate(rewards):
        row = {'sample_key': f's-{i}', 'task_key': 't', 'instance_id': 'fixture',
               'repetition_id': f'rep-{i:03d}', 'attempt_id': 'attempt-000',
               'disposition': 'accepted', 'exit_status': 'AgentCompleted', 'raw_reward': {'reward': reward}}
        rows[row['sample_key']] = row
        attempt = run/'rollout/tasks/fixture/repetitions'/row['repetition_id']/'attempts/attempt-000'
        write_json(attempt/'evaluation-raw.json', {'completed': True, 'rewards': {'reward': reward}})
    loaded = {'run': run, 'config': {'execution_context': {'benchmark': benchmark},
                                   'selection': {'tasks': {'t': {}}}}}
    monkeypatch.setattr(accepted_reward_audit, 'load_plan', lambda *args: loaded)
    monkeypatch.setattr(accepted_reward_audit, '_sample_status_projection', lambda *args: (rows, {}))
    return run, rows


def test_real_service_isolates_bad_candidate_and_writes_four_reports(tmp_path, monkeypatch):
    run, rows = setup_service(tmp_path, monkeypatch)
    raw = run/'rollout/tasks/fixture/repetitions/rep-000/attempts/attempt-000/evaluation-raw.json'
    raw.write_text('{invalid')
    result = accepted_reward_audit.audit_accepted_rewards(tmp_path, 'run', tmp_path/'audit')
    assert result['candidate_count'] == 2
    assert result['processed_count'] == result['unprocessed_count'] == 1
    assert not result['complete']
    assert {p.name for p in (tmp_path/'audit').iterdir()} == {
        'summary.json', 'samples.jsonl', 'samples.csv', 'report.md'}
    assert raw.read_text() == '{invalid'


@pytest.mark.parametrize('kind', ['existing', 'evidence', 'non-skillsbench'])
def test_real_service_rejects_invalid_output_or_benchmark(tmp_path, monkeypatch, kind):
    run, _ = setup_service(tmp_path, monkeypatch, benchmark='other' if kind == 'non-skillsbench' else 'skillsbench')
    output = tmp_path/'audit'
    if kind == 'existing':
        output.mkdir()
    if kind == 'evidence':
        output = run/'rollout/unsafe'
    with pytest.raises((ValueError, FileExistsError)):
        accepted_reward_audit.audit_accepted_rewards(tmp_path, 'run', output)


def test_real_service_detects_projection_changes_without_progress_change(tmp_path, monkeypatch):
    _, rows = setup_service(tmp_path, monkeypatch)
    calls = []
    def project(*args):
        calls.append(True)
        if len(calls) > 1:
            return {}, {}
        return rows, {}
    monkeypatch.setattr(accepted_reward_audit, '_sample_status_projection', project)
    result = accepted_reward_audit.audit_accepted_rewards(tmp_path, 'run', tmp_path/'audit')
    assert result['unprocessed_count'] == 2 and not result['complete']
    summary = json.loads((tmp_path/'audit/summary.json').read_text())
    assert summary['source_changed']


def test_real_service_detects_change_to_already_consumed_source(tmp_path, monkeypatch):
    run, _ = setup_service(tmp_path, monkeypatch)
    original = accepted_reward_audit._audit_candidate
    def candidate(*args):
        result = original(*args)
        if args[2]['sample_key'] == 's-1':
            raw = run/'rollout/tasks/fixture/repetitions/rep-000/attempts/attempt-000/evaluation-raw.json'
            raw.write_text('{"changed":true}')
        return result
    monkeypatch.setattr(accepted_reward_audit, '_audit_candidate', candidate)
    result = accepted_reward_audit.audit_accepted_rewards(tmp_path, 'run', tmp_path/'audit')
    assert result['unprocessed_count'] == 1 and not result['complete']

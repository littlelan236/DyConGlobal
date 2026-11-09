"""Run-end publication exercises real token/tensor/receipt services on CPU."""
from pathlib import Path
from types import SimpleNamespace
import copy
import os

import pytest

from core.io import read_json, sha256, write_json
from application.execution.capture_publication import finalize_capture_publication, freeze_selection
from test_partial_publication import sequence


def setup_finalizer(tmp_path, monkeypatch, gap='hidden'):
    import application.execution.capture_publication as service
    cfg, evidence = sequence(tmp_path, monkeypatch, gap)
    cfg['capture'].update(enabled=True, publication_seconds=60)
    cfg['capture']['export']['enabled'] = True
    write_json(tmp_path / 'runs/run/resolved-config.json', cfg)
    row = evidence['rollouts']['rollout']
    row.update(repetition_id='rep-000', termination_reason='Submitted')
    item = {'sample_key': 's-a', 'instance_id': 'task', 'task_key': 'task-key', 'repetition_id': 'rep-000'}
    result = {**item, 'attempt_id': 'attempt', 'rollout_id': 'rollout', 'exit_status': 'Submitted',
              'disposition': 'accepted', 'capture_status': 'partial' if gap else 'complete'}
    path = tmp_path / 'runs/run/rollout/tasks/task/repetitions/rep-000/result.json'
    write_json(path, result)
    monkeypatch.setattr(service, 'RunEvidence', lambda *a: SimpleNamespace(read=lambda: evidence))
    monkeypatch.setattr(service, 'assert_inactive', lambda *a: None)
    return cfg, evidence, item, path


@pytest.mark.parametrize('gap,points,missing,status', [(None, 3, 0, 'complete'), ('hidden', 2, 1, 'partial'), ('native', 1, 0, 'partial')])
def test_automatic_publication_preserves_native_coverage_and_sources(tmp_path, monkeypatch, gap, points, missing, status):
    cfg, evidence, item, resultpath = setup_finalizer(tmp_path, monkeypatch, gap)
    original = sha256(resultpath)
    legacy = tmp_path / 'runs/run/enrichment/online/sentinel'
    legacy.parent.mkdir(parents=True)
    legacy.write_text('legacy')
    summary = finalize_capture_publication(cfg, 'run', items=[item])
    assert summary['status'] == status, summary
    assert summary['published_points'] == points and summary['missing'] == missing
    output = tmp_path / 'runs/run/enrichment/online-publications' / summary['publication_id']
    receipt = read_json(output / 'receipt.json')
    assert receipt['publication_signature']['rollout_ids'] == ['rollout']
    assert receipt['publication_signature']['selection_sha256'] == summary['selection_sha256']
    for name, digest in receipt['files'].items():
        assert sha256(output / name) == digest
    for path, digest in receipt['publication_sources'].items():
        assert sha256(path) == digest
    assert sha256(resultpath) == original and legacy.read_text() == 'legacy'
    # A durable stable ID is consumable by the official online-index command.
    assert '--publication-id ' + summary['publication_id'] in summary['index_command']
    again = finalize_capture_publication(cfg, 'run', items=[item])
    assert again['status'] == status and again['publication_id'] == summary['publication_id']
    assert sha256(output / 'receipt.json') == summary['publication_receipt_sha256']
    if os.environ.get('DYCON_CAPTURE_TEST_REPORT_ROOT'):
        example = Path(os.environ['DYCON_CAPTURE_TEST_REPORT_ROOT']) / 'examples' / ('partial-' + gap if gap else 'complete')
        write_json(example / 'capture-publication.json', summary)
        for name in ('publication.json', 'coverage.json', 'receipt.json', 'index.json', 'point-exclusions.json'):
            write_json(example / name, read_json(output / name))


def test_final_selection_excludes_replaced_attempt_and_accepts_unresolved(tmp_path, monkeypatch):
    cfg, evidence, item, path = setup_finalizer(tmp_path, monkeypatch)
    evidence['rollouts']['old-rollout'] = {**evidence['rollouts']['rollout'], 'attempt_id': 'old'}
    evidence['rollouts']['rollout']['resolved'] = False
    reportpath = tmp_path / 'runs/run/report.json'
    report = read_json(reportpath)
    report['outcomes']['rollout']['resolved'] = False
    write_json(reportpath, report)
    evidence['evaluations']['eval']['report_sha256'] = sha256(reportpath)
    evidence['evaluations']['eval']['outcomes']['rollout']['resolved'] = False
    selection = freeze_selection(cfg, 'run', [item], evidence)
    assert [r['rollout_id'] for r in selection['rollouts']] == ['rollout']
    assert {'rollout_id': 'old-rollout', 'reason': 'not_selected_canonical_result'} in selection['excluded']
    summary = finalize_capture_publication(cfg, 'run', items=[item])
    assert summary['status'] == 'partial', summary
    assert summary['published_points'] == 2
    assert evidence['rollouts']['rollout']['resolved'] is False


@pytest.mark.parametrize('failure', ['producer', 'spool', 'publication', 'integrity'])
def test_failures_do_not_change_business_outcome(tmp_path, monkeypatch, failure):
    import application.execution.capture_publication as service
    cfg, evidence, item, path = setup_finalizer(tmp_path, monkeypatch)
    before = sha256(path), copy.deepcopy(evidence)
    if failure == 'producer':
        monkeypatch.setattr(service, 'assert_inactive', lambda *a: (_ for _ in ()).throw(ValueError('producer active')))
    elif failure == 'spool':
        monkeypatch.setattr(service, 'recover_spool', lambda *a: (_ for _ in ()).throw(service.PendingRecovery('upload timeout')))
    elif failure == 'publication':
        monkeypatch.setattr(service, 'publish_online_service', lambda *a, **kw: (_ for _ in ()).throw(OSError('disk failed')))
    else:
        (tmp_path / 'spool/run/raw/0.npz').write_bytes(b'corrupt')
    result = finalize_capture_publication(cfg, 'run', items=[item])
    assert result['status'] == ('pending_recovery' if failure in {'producer', 'spool'} else 'failed'), result
    assert (sha256(path), evidence) == before
    assert not (tmp_path / 'runs/run/enrichment/online-publications').exists()


@pytest.mark.parametrize('unsupported', ['anchor'])
def test_unsupported_is_explicit(tmp_path, monkeypatch, unsupported):
    cfg, evidence, item, path = setup_finalizer(tmp_path, monkeypatch)
    if unsupported == 'anchor':
        cfg['capture']['anchors'] = ['think_open']
    else:
        cfg['parallel'] = {'enabled': True, 'branch_count': 2}
    assert finalize_capture_publication(cfg, 'run', items=[item])['status'] == 'unsupported'


def test_crash_during_publication_resumes_without_overwriting_stage(tmp_path, monkeypatch):
    import hidden.publish_online as publisher
    cfg, evidence, item, path = setup_finalizer(tmp_path, monkeypatch)
    original = publisher.finalize_token_events
    calls = []
    def crash(*a, **kw):
        calls.append(a[1])
        a[1].mkdir(parents=True)
        write_json(a[1] / 'incomplete.json', {'state': 'interrupted'})
        raise OSError('interrupted')
    monkeypatch.setattr(publisher, 'finalize_token_events', crash)
    first = finalize_capture_publication(cfg, 'run', items=[item])
    assert first['status'] == 'failed'
    monkeypatch.setattr(publisher, 'finalize_token_events', original)
    second = finalize_capture_publication(cfg, 'run', items=[item])
    assert second['status'] == 'partial', second
    assert first['publication_id'] == second['publication_id']
    assert list(calls[0].parent.glob(calls[0].name + '.interrupted-*/incomplete.json'))


def test_publication_rejects_changed_receipt_on_resume(tmp_path, monkeypatch):
    cfg, evidence, item, path = setup_finalizer(tmp_path, monkeypatch)
    first = finalize_capture_publication(cfg, 'run', items=[item])
    out = tmp_path / 'runs/run/enrichment/online-publications' / first['publication_id']
    (out / 'index.json').write_text('corrupt')
    second = finalize_capture_publication(cfg, 'run', items=[item])
    assert second['status'] == 'failed' and second['error_type'] == 'ValueError'
    assert second['failed_stage'] == 'publication'


def test_deadline_leaves_retryable_summary(tmp_path, monkeypatch):
    import application.execution.capture_publication as service
    cfg, evidence, item, path = setup_finalizer(tmp_path, monkeypatch)
    monkeypatch.setattr(service, 'recover_spool', lambda *a: cfg['capture'].update(publication_seconds=-1))
    assert finalize_capture_publication(cfg, 'run', items=[item])['status'] == 'pending_recovery'


def test_local_writer_must_exit_before_pointwise_audit(tmp_path, monkeypatch):
    cfg, evidence, item, path = setup_finalizer(tmp_path, monkeypatch)
    producer = SimpleNamespace(queue=SimpleNamespace(_worker=SimpleNamespace(is_alive=lambda: True)))
    result = finalize_capture_publication(cfg, 'run', items=[item], producers=[producer])
    assert result['status'] == 'pending_recovery'
    assert 'writer' in result['reason']


def test_online_index_retries_and_preserves_failed_stage(tmp_path, monkeypatch):
    import core.config
    import difficulty.datasets.online as online
    cfg, evidence, item, path = setup_finalizer(tmp_path, monkeypatch)
    summary = finalize_capture_publication(cfg, 'run', items=[item])
    monkeypatch.setattr(core.config, 'load_snapshot', lambda *a: cfg)
    monkeypatch.setattr(online, 'RunEvidence', lambda *a: SimpleNamespace(read=lambda: evidence))
    monkeypatch.setattr(online, 'select_run', lambda *a: {'rollout_ids': ['rollout'], 'evidence_sha256': 'evidence'})
    stage = tmp_path / 'difficulty/datasets/.index.stage'
    write_json(stage / 'incomplete.json', {'phase': 'interrupted'})
    out = online.build_published_index(cfg, 'run', 'index', publication_id=summary['publication_id'])
    digest = sha256(out / 'receipt.json')
    assert online.build_published_index(cfg, 'run', 'index', publication_id=summary['publication_id']) == out
    assert sha256(out / 'receipt.json') == digest
    assert list(stage.parent.glob(stage.name + '.interrupted-*/incomplete.json'))
    assert read_json(out / 'index.json')['coverage']['native_hidden_missing'] == 1


def test_skillsbench_workspace_subject_is_not_a_patch_submission(tmp_path, monkeypatch):
    cfg, evidence, item, resultpath = setup_finalizer(tmp_path, monkeypatch)
    row = evidence['rollouts']['rollout']
    trajectorypath = tmp_path / row['trajectory_path']
    trajectory = read_json(trajectorypath)
    trajectory['info']['exit_status'] = 'AgentCompleted'
    write_json(trajectorypath, trajectory)
    subjectpath = tmp_path / 'runs/run/subject.json'
    write_json(subjectpath, {'schema_version': 1, 'kind': 'workspace', 'complete': True, 'outputs': []})
    row.update(submission_status='not_applicable', evaluation_subject={'kind': 'workspace', 'schema_version': 1},
               subject_path='runs/run/subject.json', subject_sha256=sha256(subjectpath),
               termination_reason='AgentCompleted', trajectory_sha256=sha256(trajectorypath))
    result = read_json(resultpath)
    result['agent_exit_status'] = 'AgentCompleted'
    write_json(resultpath, result)
    outcome = {'status': 'completed', 'resolved': True, 'subject_sha256': sha256(subjectpath)}
    reportpath = tmp_path / 'runs/run/report.json'
    write_json(reportpath, {'outcomes': {'rollout': outcome}})
    evidence['evaluations']['eval'].update(report_sha256=sha256(reportpath), outcomes={'rollout': outcome})
    summary = finalize_capture_publication(cfg, 'run', items=[item])
    assert summary['status'] == 'partial' and summary['published_points'] == 2, summary


def test_recovered_complete_keeps_original_partial_capture_fact(tmp_path, monkeypatch):
    cfg, evidence, item, resultpath = setup_finalizer(tmp_path, monkeypatch, gap=None)
    result = read_json(resultpath)
    result['capture_status'] = 'partial'
    write_json(resultpath, result)
    summary = finalize_capture_publication(cfg, 'run', items=[item])
    assert summary['status'] == 'complete'
    publication = read_json(tmp_path / 'runs/run/enrichment/online-publications' / summary['publication_id'] / 'publication.json')
    assert publication['sessions'][0]['source_capture_status'] == 'partial'
    assert publication['sessions'][0]['source_coverage_status'] == 'complete'
    assert read_json(resultpath)['capture_status'] == 'partial'

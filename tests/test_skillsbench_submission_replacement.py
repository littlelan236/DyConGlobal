import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.io import hash_json, read_json, sha256, write_json
from runtime.agents.submission_replacement import eligible, prepare_changes
from runtime.agents.pddl_regrading import _transaction


@pytest.fixture
def scenario(tmp_path):
    run, output = tmp_path / 'source', tmp_path / 'target/validation/regrade'
    old = run / 'rollout/tasks/task/repetitions/rep-000/attempts/rep-000-attempt-000'
    folder = output / 'samples/sample'
    write_json(old / 'subject/subject.json', {'original': True})
    subject = sha256(old / 'subject/subject.json')
    raw = {'completed': True, 'rewards': {'reward': 0}, 'subject_sha256': subject}
    write_json(old / 'evaluation-raw.json', raw)
    report = {'raw_rewards': raw['rewards'], 'resolved': False,
              'report_sha256': sha256(old / 'evaluation-raw.json')}
    write_json(old / 'evaluation-report.json', report)
    write_json(old / 'diagnostics/verifier-zero-audit.json', {'classification': 'Review'})
    value = {'sample_key': 'sample', 'attempt_id': 'rep-000-attempt-000', 'raw_reward': raw['rewards'],
             'rollout_id': 'rollout', 'evaluation_id': 'eval', 'subject_sha256': subject,
             'evaluation_report_sha256': sha256(old / 'evaluation-report.json'),
             'capture_status': 'complete', 'resolved': False,
             'termination_evidence': {'evidence_files': [{'path': 'evaluation-raw.json',
                                     'sha256': sha256(old / 'evaluation-raw.json')}]}}
    write_json(old / 'attempt.json', value)
    write_json(old.parent.parent / 'result.json', value)
    for n in ['03-artifacts_committed.json', '05-settled.json']:
        write_json(old / 'finalization' / n, {'details': {}, 'binding': {'sample': 'sample'}})
    evidence = {'revision': 1, 'rollouts': {'rollout': {'selected_evaluation_id': 'eval',
                'subject_sha256': subject, 'resolved': False}}, 'evaluations': {'eval': {
                'report_sha256': sha256(old / 'evaluation-report.json'),
                'outcomes': {'rollout': {'subject_sha256': subject}}, 'provenance': {}}}}
    write_json(run / 'evidence.json', evidence)
    write_json(run / 'progress.json', {'samples': {'sample': value, 'unrelated': {'raw_reward': 0.7}}})
    write_json(tmp_path / 'target/resolved-config.json', {'new': True})
    row = {'sample_key': 'sample', 'attempt_id': 'rep-000-attempt-000',
           'attempt_path': str(old.relative_to(run)), 'raw_reward': raw['rewards']}
    item = {'source': row, 'subject_sha256': subject, 'subject_path': str(old / 'subject/subject.json')}
    record = {'sample_key': 'sample', 'classification': 'ValidOutcome', 'new_rewards': {'reward': 1.0},
              'verifier_executed': True, 'cleanup_confirmed': True, 'replay_scope': 'declared outputs',
              'task_package_sha256': 'new-package', 'dependency_snapshot_sha256': 'new-cache',
              'finished_at': '2026-11-11', 'reason': 'verified'}
    write_json(folder / 'result.json', record)
    write_json(folder / 'evaluation-raw.json', {'completed': True, 'rewards': record['new_rewards'],
                                             'subject_sha256': 'fresh-export'})
    write_json(folder / 'verifier-audit.json', {'classification': 'NotCandidate'})
    return SimpleNamespace(run=run, output=output, old=old, folder=folder,
        source={'run': run}, target={'run': tmp_path / 'target', 'plan': {'run_id': 'target'}},
        selected=[item], record=record)


@pytest.mark.parametrize('reward', [0, 0.83, 1.0])
def test_canonical_scores_identity_chain_and_unrelated_data(scenario, reward):
    s = scenario
    s.record['new_rewards'] = {'reward': reward}
    write_json(s.folder / 'result.json', s.record)
    raw = read_json(s.folder / 'evaluation-raw.json')
    raw['rewards'] = s.record['new_rewards']
    write_json(s.folder / 'evaluation-raw.json', raw)
    changes, applied, skipped = prepare_changes(s.source, s.target, s.selected, [s.record], s.output)
    assert len(applied) == 1 and not skipped
    original = (s.old / 'attempt.json').read_bytes()
    _transaction(s.run, s.output, changes)
    attempt = read_json(s.old / 'attempt.json')
    report = read_json(s.old / 'evaluation-report.json')
    raw = read_json(s.old / 'evaluation-raw.json')
    assert attempt['raw_reward'] == {'reward': reward}
    assert attempt['resolved'] is (reward == 1)
    assert raw['subject_sha256'] == s.selected[0]['subject_sha256']
    assert raw['replay_subject_sha256'] == 'fresh-export'
    assert attempt['evaluation_report_sha256'] == sha256(s.old / 'evaluation-report.json')
    assert report['report_sha256'] == sha256(s.old / 'evaluation-raw.json')
    assert read_json(s.run / 'evidence.json')['evaluations']['eval']['report_sha256'] == attempt['evaluation_report_sha256']
    assert read_json(s.run / 'progress.json')['samples']['unrelated'] == {'raw_reward': 0.7}
    assert attempt['capture_status'] == 'complete'
    assert any(Path(r['backup']).read_bytes() == original for r in read_json(s.output / 'replacement-manifest.json')['files'] if r['backup'])
    for n in ['03-artifacts_committed.json', '05-settled.json']:
        d = read_json(s.old / 'finalization' / n)
        digest = d.pop('receipt_sha256')
        assert digest == hash_json(d)


@pytest.mark.parametrize('classification', ['VerifierError', 'Review', 'RegradeUnavailable', 'RegradeExecutionError'])
def test_untrusted_results_do_not_replace_original(scenario, classification):
    s = scenario
    s.record['classification'] = classification
    write_json(s.folder / 'result.json', s.record)
    changes, applied, skipped = prepare_changes(s.source, s.target, s.selected, [s.record], s.output)
    assert not changes and not applied and len(skipped) == 1
    assert skipped[0]['historical_outcome_preserved']


@pytest.mark.parametrize('reward', [float('nan'), float('inf'), -1, 2, True, '1', None])
def test_invalid_reward_never_eligible(scenario, reward):
    record = copy.deepcopy(scenario.record)
    record['new_rewards']['reward'] = reward
    assert not eligible(record)


def test_superseded_attempt_rejected_before_writes(scenario):
    s = scenario
    attempt = read_json(s.old / 'attempt.json')
    attempt['raw_reward'] = {'reward': 0.5}
    write_json(s.old / 'attempt.json', attempt)
    with pytest.raises(ValueError, match='Historical attempt changed'):
        prepare_changes(s.source, s.target, s.selected, [s.record], s.output)
    assert not (s.output / 'replacement-manifest.json').exists()


def test_partial_transaction_failure_rolls_back(scenario):
    s = scenario
    changes, _, _ = prepare_changes(s.source, s.target, s.selected, [s.record], s.output)
    original = {p: p.read_bytes() if p.exists() else None for p in changes}
    with pytest.raises(RuntimeError, match='injected'):
        _transaction(s.run, s.output, changes, fail_after=4)
    assert all((p.read_bytes() if p.exists() else None) == data for p, data in original.items())
    assert read_json(s.output / 'replacement-state.json')['phase'] == 'rolled_back'

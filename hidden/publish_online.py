"""Publish qualified repaired captures through the existing exact-token finalizer.

Original captures, repair receipts and run snapshots remain immutable. Only
complete native evidence is promoted; rejected trajectories are reported.
"""
from collections import Counter
from pathlib import Path
import copy

from application.execution.evidence import RunEvidence
from core.config import load_snapshot
from core.io import hash_json, identifier, now, owned_path, read_json, sha256, write_json
from core.locking import exclusive_run
from enrichment.online_trace import finalize_token_events, publication_directory
from hidden.artifacts import read_repaired_session, read_request
from hidden.repair_points import _bindings, _canonical, _classification
from runtime.serving.spool import publish_files


def complete_native(coverage):
    return (coverage.get('status') == 'complete'
            and coverage.get('expected_count_complete') is True
            and not coverage.get('unbound_responses')
            and not coverage.get('missing_known')
            and not coverage.get('canonical_error'))


def native_complete_suffix(bound, responses, steps):
    """Keep requests after the last unknown canonical response, including errors.

    Missing hidden is harmless for token labels. Missing native response tokens
    invalidate earlier remainders, but not the fully observed suffix after it.
    """
    observed = {b['message_index'] for b in bound.values()}
    unknown = ({i for i, _, _ in responses} | set(steps)) - observed
    cutoff = max(unknown, default=-1)
    return {key: value for key, value in bound.items()
            if value['message_index'] > cutoff}, cutoff, sorted(unknown)


@exclusive_run
def publish_online(cfg, run_id, *, repair_id, publication_id, resolved_only=False,
                   coverage_policy='complete_native_sessions_v1'):
    root = Path(cfg['project_root']).resolve()
    run_id, repair_id, publication_id = map(identifier, (run_id, repair_id, publication_id))
    policies = {'complete_native_sessions_v1', 'available_native_suffix_v1'}
    if coverage_policy not in policies:
        raise ValueError('Unknown capture publication coverage policy')
    partial = coverage_policy == 'available_native_suffix_v1'
    frozen = load_snapshot(root, run_id)
    if frozen['capture']['anchors'] != ['think_close']:
        raise ValueError('Repaired publication requires close-only capture')
    run = root / 'runs' / run_id
    nas = Path(frozen['paths']['nas_root']) / frozen['capture']['export']['root'] / run_id
    repaired = nas / 'capture-repairs' / repair_id
    manifest = read_json(repaired / 'receipt.json')
    if manifest['run_id'] != run_id or manifest['repair_id'] != repair_id:
        raise ValueError('Repair manifest belongs to another run/repair')
    source_files = {str(repaired / 'receipt.json'): sha256(repaired / 'receipt.json'),
                    str(run / 'resolved-config.json'): sha256(run / 'resolved-config.json')}
    for relative, digest in {**manifest['files'], **manifest['session_receipts']}.items():
        path = owned_path(nas, relative)
        if sha256(path) != digest:
            raise ValueError('Repair manifest checksum mismatch: ' + relative)
        source_files[str(path)] = digest
    signature = {'run_id': run_id, 'repair_id': repair_id, 'publication_id': publication_id,
                 'repair_receipt_sha256': sha256(repaired / 'receipt.json'),
                 'resolved_only': resolved_only, 'policy': coverage_policy}
    output = publication_directory(root, run_id, publication_id if partial else None)
    if output.exists():
        receipt = read_json(output / 'receipt.json')
        if receipt.get('publication_signature') != signature:
            raise FileExistsError('Existing online trace has a different publication identity')
        for name, digest in receipt['files'].items():
            if sha256(output / name) != digest:
                raise ValueError('Published online trace checksum mismatch')
        for path, digest in receipt.get('publication_sources', {}).items():
            if sha256(path) != digest:
                raise ValueError('Published source checksum mismatch')
        for path, digest in receipt.get('source_files', {}).items():
            if sha256(owned_path(nas, path)) != digest:
                raise ValueError('Published native source checksum mismatch')
        return read_json(output / 'publication.json')
    audit = run / 'hidden/capture-publications' / publication_id
    audit.mkdir(parents=True, exist_ok=True)
    selection_path = audit / 'selection.json'
    if selection_path.exists() and read_json(selection_path) != signature:
        raise ValueError('Publication ID already binds different inputs')
    write_json(selection_path, signature)
    evidence = RunEvidence(root, run_id).read()
    proofs, finalizations, requests, events, accepted, excluded = {}, {}, [], [], [], []
    point_exclusions = []
    marker_pairs = set()
    report = read_json(repaired / 'report.json')
    records = {r['rollout_id']: r for r in report['sessions'] if r.get('rollout_id')}
    candidates = [(rid, row) for rid, row in sorted(evidence['rollouts'].items())
                  if not row.get('provenance', {}).get('quarantined')
                  and (not resolved_only or (row.get('evaluation_status') == 'completed'
                                            and row.get('resolved') is True))]
    for number, (rid, row) in enumerate(candidates, 1):
        record = records.get(rid)
        if row.get('submission_status') != 'submitted':
            excluded.append({'rollout_id': rid, 'reason': 'not_submitted'})
            continue
        if not record or (not partial and not complete_native(record['coverage'])):
            excluded.append({'rollout_id': rid, 'reason': 'no_complete_native_session',
                             'coverage': record.get('coverage') if record else None})
            continue
        sid = record['session_id']
        view = read_repaired_session(nas, repair_id, sid)
        if not partial and not complete_native(view['coverage']):
            raise ValueError('Repair report/coverage disagreement')
        if partial and view['coverage'].get('canonical_error'):
            excluded.append({'rollout_id': rid, 'reason': 'canonical_binding_error'})
            continue
        _, _, task, responses = _canonical(root, run_id, sid, [(rid, row)], evidence, proofs)
        trajectory = read_json(owned_path(root, row['trajectory_path']))
        if trajectory.get('info', {}).get('exit_status') != 'Submitted':
            raise ValueError('Resolved canonical trajectory is not Submitted')
        response_indices = {i for i, _, _ in responses}
        steps = [i for i, m in enumerate(trajectory['messages']) if m.get('role') == 'assistant']
        if not partial and not set(steps) <= response_indices:
            excluded.append({'rollout_id': rid, 'reason': 'action_missing_native_response'})
            continue
        receipt_path = repaired / 'sessions' / sid / 'receipt.json'
        native = []
        for relative, digest in view['receipt']['dependencies'].items():
            path = owned_path(nas, relative)
            if path.parent.name == 'requests' and path.suffix == '.json':
                req = read_request(nas, {'ledger_ref': relative, 'ledger_sha256': digest})
                if hash_json(req['prompt_token_ids']) != req['request_sha256'] or hash_json(req['output_token_ids']) != req['output_sha256']:
                    raise ValueError('Native token hash mismatch')
                native.append(req)
            source_files[str(path)] = digest
        all_bound, expected, unbound, identities = _bindings(sid, task, run_id, responses, native)
        if not partial and (unbound or len(all_bound) != len(responses)):
            raise ValueError('Complete repair lacks complete canonical native bindings')
        bound, cutoff, unknown = native_complete_suffix(all_bound, responses, steps)
        if not partial:
            bound, cutoff, unknown = all_bound, -1, []
        expected = {(key, position, anchor) for key, binding in bound.items()
                    for position, anchor in binding['targets']}
        if not bound:
            excluded.append({'rollout_id': rid, 'reason': 'no_native_complete_suffix',
                             'unknown_response_indices': unknown})
            continue
        seen = set()
        session_events = []
        point_path = repaired / 'sessions' / sid / 'point-index.json'
        for original in view['points']:
            classification, index = _classification(original, sid, run_id, task, row['model_fingerprint'], all_bound, identities)
            if classification != 'native_exact':
                if not partial:
                    raise ValueError('Complete publication contains a non-exact point')
                point_exclusions.append({'sample_id': original['sample_id'], 'rollout_id': rid,
                                         'reason': 'not_native_exact', 'binding': classification})
                continue
            if original['recovery']['canonical_rollout_id'] != rid or original['recovery']['trajectory_sha256'] != row['trajectory_sha256']:
                raise ValueError('Repaired point canonical identity changed')
            if original['request_id'] not in bound:
                point_exclusions.append({'sample_id': original['sample_id'], 'rollout_id': rid,
                                         'reason': 'unknown_future_reasoning_tokens',
                                         'last_unknown_response_index': cutoff})
                continue
            seen.add((original['request_id'], original['absolute_position'], original['anchor']))
            event = copy.deepcopy(original)
            event['message_index'] = index
            locator = event.get('tensor_locator')
            index_path = point_path
            if locator:
                key = Path(locator['shard']).stem
                proof_path = nas / 'publications' / (key + '.json')
                if proof_path.exists():
                    proof = read_json(proof_path)
                    indexes = [name for name in proof['files'] if name.startswith('index/')]
                    if len(indexes) != 1:
                        raise ValueError('Shard publication must bind one source index')
                    index_path = owned_path(nas, indexes[0])
                    if sha256(index_path) != proof['files'][indexes[0]]:
                        raise ValueError('Original shard index checksum mismatch')
                    source_files[str(index_path)] = proof['files'][indexes[0]]
                    source_files[str(proof_path)] = sha256(proof_path)
            session_events.append((index_path, event))
        if seen - expected or (not partial and seen != expected):
            raise ValueError('Repaired point coverage differs from native target inventory')
        if not session_events:
            excluded.append({'rollout_id': rid, 'reason': 'no_exact_hidden_in_qualified_suffix'})
            continue
        if resolved_only:
            evaluation = evidence['evaluations'][row['selected_evaluation_id']]
            path = owned_path(root, evaluation['report_path'])
            if sha256(path) != evaluation['report_sha256']:
                raise ValueError('Harness report checksum mismatch')
            outcome = read_json(path)['outcomes'][rid]
            if outcome.get('resolved') is not True or outcome.get('status') != 'completed' or outcome['patch_sha256'] != row['patch_sha256']:
                raise ValueError('Harness outcome does not bind the successful exact patch')
            source_files[str(path)] = evaluation['report_sha256']
        for binding in bound.values():
            req = binding['request']
            marker_pairs.add((tuple(req.get('open_token_ids', [248068])), tuple(req['close_token_ids'])))
            requests.append({'ledger_ref': req['ledger_ref'], 'ledger_sha256': req['ledger_sha256']})
        relative = f'capture-finalizations/{publication_id}/sessions/{sid}/finalization.json'
        derived = {'schema_version': 2, 'run_id': run_id, 'session_id': sid, 'rollout_id': rid,
                   'trajectory_path': str(owned_path(root, row['trajectory_path'])),
                   'trajectory_sha256': row['trajectory_sha256'], 'model_fingerprint': row['model_fingerprint'],
                   'accepted_native_request_ids': list(bound),
                   'responses': {key: {'message_index': value['message_index'],
                                      'is_action_response': value['message_index'] in steps,
                                      'agent_step_index': steps.index(value['message_index']) if value['message_index'] in steps else None}
                                 for key, value in bound.items()},
                   'source_repair_receipt': str(receipt_path),
                   'source_repair_receipt_sha256': sha256(receipt_path)}
        local = audit / 'payload' / relative
        if local.exists() and read_json(local) != derived:
            raise ValueError('Immutable derived finalization conflict')
        write_json(local, derived)
        finalizations[sid] = relative
        events.extend(session_events)
        accepted.append({'rollout_id': rid, 'session_id': sid, 'points': len(session_events),
                         'source_coverage_status': view['coverage']['status'],
                         'last_unknown_response_index': cutoff,
                         'unknown_response_indices': unknown,
                         'native_requests_in_qualified_suffix': len(bound),
                         'known_missing_hidden_in_suffix': len(expected - seen)})
        print(f'capture publish: qualified {len(accepted)}; inspected {number}/{len(candidates)}', flush=True)
    if not accepted or len(marker_pairs) != 1:
        raise ValueError('Publication needs nonempty qualified sessions with one marker inventory')
    files = {p: sha256(audit / 'payload' / p) for p in finalizations.values()}
    for relative, digest in files.items():
        target = nas / relative
        if target.exists() and sha256(target) != digest:
            raise ValueError('Immutable NAS finalization conflict')
    publish_files(audit / 'payload', nas, files, frozen['capture']['export']['upload_max_elapsed_seconds'])
    stage = run / 'enrichment' / ('.online-' + publication_id)
    if stage.exists():
        raise FileExistsError('Incomplete publication stage needs inspection: ' + str(stage))
    open_ids, close_ids = next(iter(marker_pairs))
    # The repaired request ledger is now promoted into the immutable online
    # session namespace. finalize_token_events validates these finalizations
    # and derives the exact remaining-reasoning labels.
    finalize_token_events(nas, stage, open_ids=list(open_ids), close_ids=list(close_ids),
                          request_evidence=requests, event_rows=events, finalization_paths=finalizations,
                          allow_missing=partial)
    summary = {'run_id': run_id, 'repair_id': repair_id, 'publication_id': publication_id,
               'policy': signature['policy'], 'resolved_only': resolved_only,
               'candidate_rollouts': len(candidates), 'published_rollouts': len(accepted),
               'published_points': len(events), 'excluded_rollouts': len(excluded),
               'exclusion_reasons': dict(Counter(r['reason'] for r in excluded)),
               'published_partial_rollouts': sum(r['source_coverage_status'] != 'complete' for r in accepted),
               'point_exclusion_counts': dict(Counter(r['reason'] for r in point_exclusions)),
               'known_missing_hidden_in_qualified_suffixes': sum(r['known_missing_hidden_in_suffix'] for r in accepted),
               'sessions': accepted, 'excluded': excluded, 'created_at': now(), 'path': str(output)}
    write_json(stage / 'point-exclusions.json', point_exclusions)
    write_json(stage / 'publication.json', summary)
    receipt = read_json(stage / 'receipt.json')
    receipt.update(publication_signature=signature, publication_sources={**source_files, **proofs})
    receipt['files'].update({'publication.json': sha256(stage / 'publication.json'),
                             'coverage.json': sha256(stage / 'coverage.json'),
                             'point-exclusions.json': sha256(stage / 'point-exclusions.json')})
    write_json(stage / 'receipt.json', receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(output)
    write_json(audit / 'publication.json', summary)
    return summary

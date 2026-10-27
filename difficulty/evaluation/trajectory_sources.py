"""Prediction eligibility independent of success-label availability.

Read immutable pointwise repair publications through the shared capture readers.
Native-exact and canonical-identity-only points are distinguished in exports.
Neither class grants training eligibility or creates missing token labels.
"""
from collections import Counter, defaultdict
import csv
from pathlib import Path

import numpy as np

from action.datasets.trajectories import read_actions
from core.config import load_snapshot
from core.io import artifact_dir, hash_json, identifier, owned_path, read_json, sha256
from difficulty.datasets.submission import iter_features
from difficulty.models.ridge_grid import predict_model
from enrichment.online_trace import publication_directory, reasoning_blocks, annotate_remaining_workloads
from hidden.artifacts import read_repaired_session, read_request, event_records
from hidden.repair_points import _bindings, _canonical, _classification
from hidden.publish_online import native_complete_suffix
from difficulty.receipts import verify_receipt


def publication_truth(root, run_id, dataset_dir, catalog, actions, source_files):
    """Read exact, uncensored future labels even when the current hidden is absent."""
    evidence = read_json(dataset_dir / 'source-evidence.json')
    index_path = Path(evidence['source_index'])
    if not index_path.is_absolute():
        index_path = owned_path(root, str(index_path))
    if sha256(index_path) != evidence['source_index_sha256']:
        raise ValueError('Dataset source index changed')
    index = read_json(index_path)
    publication = publication_directory(root, run_id, index.get('publication_id'))
    receipt_path = publication / 'receipt.json'
    if sha256(receipt_path) != index['source_receipt_sha256']:
        raise ValueError('Truth publication receipt changed')
    receipt = read_json(receipt_path)
    for name, digest in receipt['files'].items():
        path = owned_path(publication, name)
        if sha256(path) != digest:
            raise ValueError('Truth publication changed: ' + name)
        source_files[str(path)] = digest
    source_files[str(receipt_path)] = sha256(receipt_path)
    result = {}
    for request in read_json(publication / 'requests.json').values():
        rid = request['rollout_id']
        if rid not in actions or request.get('is_action_response') is not True:
            continue
        if (request.get('run_id') != run_id
                or request['trajectory_sha256'] != catalog['rollouts'][rid]['trajectory_sha256']):
            raise ValueError('Truth request/canonical trajectory mismatch')
        step = request['agent_step_index']
        if not 0 <= step < len(actions[rid]) or actions[rid][step]['message_index'] != request['message_index']:
            raise ValueError('Truth request/canonical step mismatch')
        # The existing finalizer already calculated future_reasoning_tokens and
        # remaining_censored from exact native responses in its qualified suffix.
        blocks = [b for b in request['blocks'] if b.get('close') is not None]
        if len(blocks) != 1 or blocks[0].get('remaining_censored', True):
            continue
        value = blocks[0]['future_reasoning_tokens']
        if type(value) is not int or value < 0:
            raise ValueError('Invalid native future reasoning label')
        key = (rid, step)
        if key in result and result[key] != value:
            raise ValueError('Conflicting truth labels at one canonical close')
        result[key] = value
    return result, index.get('publication_id')


def canonical_point(event, *, sid, run_id, rid, row, task, actions, bound, identities, require_recovery=True):
    """Return a canonical prediction binding without requiring future token labels."""
    recovery = event.get('recovery', {})
    if require_recovery and (recovery.get('canonical_rollout_id') != rid
            or recovery.get('trajectory_sha256') != row['trajectory_sha256']):
        raise ValueError('Repaired point canonical identity changed')
    classification, message_index = _classification(
        event, sid, run_id, task, row['model_fingerprint'], bound, identities)
    if classification not in {'native_exact', 'canonical_identity_only'}:
        return None, classification
    if require_recovery and (recovery.get('binding') != classification or recovery.get('canonical_message_index') != message_index):
        raise ValueError('Recovered point classification changed')
    if event.get('anchor') != 'think_close':
        return None, 'not_action_close'
    by_message = {a['message_index']: i for i, a in enumerate(actions)}
    if message_index not in by_message:
        return None, 'not_action_response'
    if event.get('status') != 'persisted':
        return None, 'not_persisted'
    step = by_message[message_index]
    return {**event, 'rollout_id': rid, 'trajectory_sha256': row['trajectory_sha256'],
            'agent_step': step, 'message_index': message_index,
            'hidden_sha256': event['sha256'], 'hidden_path': event.get('path', event['tensor']),
            'prediction_binding': classification}, None


def predict_repaired(cfg, run_id, repair_id, catalog, model, layer, *, cache_id=None, model_path=None, target=None):
    """Score every unambiguous action-close repair point, with no GT filtering."""
    root = Path(cfg['project_root'])
    if cache_id:
        directory = artifact_dir(cfg, 'difficulty', identifier(cache_id), 'evaluations')
        receipt, _ = verify_receipt(root, directory)
        summary = read_json(directory / 'summary.json')
        if (summary['run_id'] != run_id or summary['repair_id'] != repair_id
                or summary['target'] != target
                or receipt['source_files'].get(summary['model']) != sha256(model_path)):
            raise ValueError('Prediction cache run/repair/model/target mismatch')
        actions = {rid: read_actions(root, dict(row, rollout_id=rid))
                   for rid, row in catalog['rollouts'].items() if row.get('resolved') is True}
        predictions = {}
        with (directory / 'curves.csv').open(encoding='utf-8-sig', newline='') as stream:
            for point in csv.DictReader(stream):
                if not point['prediction']:
                    continue
                rid, step = point['rollout_id'], int(point['step_id'])
                row = catalog['rollouts'][rid]
                key = (rid, step)
                if key in predictions:
                    raise ValueError('Duplicate cached prediction')
                sample = dict(run_id=run_id, rollout_id=rid, agent_step=step,
                              sample_id=point['sample_id'], trajectory_sha256=row['trajectory_sha256'],
                              message_index=actions[rid][step]['message_index'],
                              prediction_binding=point['prediction_binding'],
                              capture_source=point.get('capture_source') or 'pointwise_repair')
                predictions[key] = (sample, float(point['prediction_transformed']))
        if len(predictions) != summary['prediction_points']:
            raise ValueError('Prediction cache count mismatch')
        sources = dict(receipt['source_files'])
        for name in ('receipt.json', 'summary.json', 'curves.csv', 'coverage-audit.json'):
            sources[str(directory / name)] = sha256(directory / name)
        audit = read_json(directory / 'coverage-audit.json')
        audit['reused_predictions_id'] = cache_id
        return predictions, sources, audit, actions
    frozen = load_snapshot(root, run_id)
    nas = Path(frozen['paths']['nas_root']) / frozen['capture']['export']['root'] / run_id
    base = nas / 'capture-repairs' / identifier(repair_id)
    manifest_path = base / 'receipt.json'
    manifest = read_json(manifest_path)
    if manifest.get('run_id') != run_id or manifest.get('repair_id') != repair_id:
        raise ValueError('Repair publication run/ID mismatch')
    source_files = {str(manifest_path): sha256(manifest_path)}
    for relative, digest in {**manifest['files'], **manifest['session_receipts']}.items():
        path = owned_path(nas, relative)
        if sha256(path) != digest:
            raise ValueError('Repair publication checksum mismatch: ' + relative)
        source_files[str(path)] = digest
    report = read_json(base / 'report.json')
    records = [r for r in report['sessions'] if r.get('published') and r.get('rollout_id')]
    resolved = {rid: row for rid, row in catalog['rollouts'].items()
                if row.get('resolved') is True and row.get('evaluation_status') == 'completed'
                and not row.get('provenance', {}).get('quarantined')}
    actions = {rid: read_actions(root, dict(row, rollout_id=rid)) for rid, row in resolved.items()}
    samples, exclusions, seen = [], [], {}
    proofs = {}
    for record in records:
        rid = record['rollout_id']
        if rid not in resolved:
            continue
        row = resolved[rid]
        sid = record['session_id']
        # Verify publication metadata now; shared iter_features verifies actual
        # full shard SHA, member SHA, shapes and finite feature values below.
        view = read_repaired_session(nas, repair_id, sid, verify_dependencies=False)
        receipt = view['receipt']
        if receipt.get('canonical_rollout_id') != rid:
            raise ValueError('Repair receipt/canonical rollout mismatch')
        if view['coverage'].get('canonical_error'):
            exclusions.append(dict(rollout_id=rid, reason='canonical_binding_error'))
            continue
        _, _, task, responses = _canonical(root, run_id, sid, [(rid, row)], catalog, proofs)
        native = []
        for relative, digest in receipt['dependencies'].items():
            path = owned_path(nas, relative)
            if path.parent.name == 'requests' and path.suffix == '.json':
                req = read_request(nas, dict(ledger_ref=relative, ledger_sha256=digest))
                if (hash_json(req['prompt_token_ids']) != req['request_sha256']
                        or hash_json(req['output_token_ids']) != req['output_sha256']):
                    raise ValueError('Native request token hash mismatch')
                native.append(req)
                source_files[str(path)] = digest
                if req.get('inventory_ref'):
                    inv = owned_path(nas, req['inventory_ref'])
                    source_files[str(inv)] = sha256(inv)
        bound, _, _, identities = _bindings(sid, task, run_id, responses, native)
        for event in view['points']:
            sample, reason = canonical_point(event, sid=sid, run_id=run_id, rid=rid, row=row,
                                             task=task, actions=actions[rid], bound=bound, identities=identities)
            if sample is None:
                exclusions.append(dict(rollout_id=rid, sample_id=event['sample_id'], reason=reason))
                continue
            key = (rid, sample['agent_step'])
            if key in seen:
                previous = seen[key]
                if previous['hidden_sha256'] != sample['hidden_sha256']:
                    raise ValueError('Ambiguous distinct hidden points for one canonical action-close')
                exclusions.append(dict(rollout_id=rid, sample_id=event['sample_id'], reason='duplicate_hidden_alias'))
                continue
            sample['hidden_root'] = str(nas)
            seen[key] = sample
            samples.append(sample)
    source_files.update(proofs)
    predictions = {}
    batch, positions = [], []
    def score_batch():
        if not batch:
            return
        scores = np.asarray(predict_model(model, np.stack(batch).astype(float)), dtype=float)
        if scores.shape != (len(batch),) or not np.isfinite(scores).all():
            raise ValueError('Invalid repaired hidden predictions')
        for i, score in zip(positions, scores, strict=True):
            sample = samples[i]
            predictions[(sample['rollout_id'], sample['agent_step'])] = (sample, float(score))
        batch.clear()
        positions.clear()
    for number, (i, features) in enumerate(iter_features(samples), 1):
        batch.append(features[layer])
        positions.append(i)
        if len(batch) >= 256:
            score_batch()
        if number % 1000 == 0:
            print(f'prediction coverage: {number}/{len(samples)} verified hidden points', flush=True)
    score_batch()
    if len(predictions) != len(samples):
        raise ValueError('Incomplete repaired hidden predictions')
    # Tensor proofs are bound in the new report, not just checked in memory.
    for sample in samples:
        locator = sample.get('tensor_locator')
        path = owned_path(nas, locator['shard'] if locator else sample['hidden_path'])
        digest = locator['shard_sha256'] if locator else sample['hidden_sha256']
        source_files[str(path)] = digest
    return predictions, source_files, dict(
        repair_id=repair_id, repaired_prediction_points=len(predictions),
        binding_counts=dict(Counter(s['prediction_binding'] for s in samples)),
        exclusions=exclusions, exclusion_counts=dict(Counter(e['reason'] for e in exclusions))), actions


def extend_original_publications(cfg, run_id, catalog, predictions, actions, true_map, model, layer, source_files):
    """Union original NAS shard publications with repair points, without duplicates."""
    root = Path(cfg['project_root'])
    frozen = load_snapshot(root, run_id)
    nas = Path(frozen['paths']['nas_root']) / frozen['capture']['export']['root'] / run_id
    sessions, responses_by_sid, rows_by_sid = {}, {}, {}
    proofs, session_exclusions = {}, []
    # Use the same canonical reader/join used by repair publication.
    from action.datasets.trajectories import read_messages
    for rid in actions:
        row = catalog['rollouts'][rid]
        messages = read_messages(root, dict(row, rollout_id=rid))
        session_ids = {(m.get('extra', {}).get('request') or {}).get('dycon_capture', {}).get('session_id')
                       for m in messages}
        session_ids.discard(None)
        if len(session_ids) != 1:
            session_exclusions.append(dict(rollout_id=rid, reason='missing_or_ambiguous_capture_session'))
            continue
        sid = next(iter(session_ids))
        _, _, task, responses = _canonical(root, run_id, sid, [(rid, row)], catalog, proofs)
        sessions[sid] = (rid, task)
        responses_by_sid[sid] = responses
        rows_by_sid[sid] = row
    grouped = defaultdict(list)
    indices_seen = {}
    total_original = 0
    for path, event in event_records(nas):
        total_original += 1
        sid = event.get('session_id')
        if sid in sessions:
            grouped[sid].append((path, event))
    added_samples, truth, exclusions = [], {}, list(session_exclusions)
    for number, (sid, (rid, task)) in enumerate(sessions.items(), 1):
        rows = grouped[sid]
        known_samples = {s['sample_id'] for (r, _), (s, _) in predictions.items() if r == rid}
        need_prediction = any(e['sample_id'] not in known_samples for _, e in rows)
        need_truth = any((rid, step) not in true_map for step in range(len(actions[rid])))
        if not need_prediction and not need_truth:
            continue
        row = rows_by_sid[sid]
        responses = responses_by_sid[sid]
        request_ids = {e['request_id'] for _, e in rows}
        for _, _, response in responses:
            request_ids.update(v for v in (response.get('id'), response.get('dycon_native_request_id')) if v)
        native = []
        for request_id in request_ids:
            key = hash_json(request_id)
            candidates = [nas / 'requests' / (key + '.json'), nas / 'sessions' / sid / 'requests' / (key + '.json')]
            path = next((p for p in candidates if p.exists()), None)
            if path is None:
                continue
            digest = sha256(path)
            req = read_request(nas, dict(ledger_ref=str(path.relative_to(nas)), ledger_sha256=digest))
            if req.get('session_id') != sid:
                raise ValueError('Original request/session mismatch')
            if (hash_json(req['prompt_token_ids']) != req['request_sha256']
                    or hash_json(req['output_token_ids']) != req['output_sha256']):
                raise ValueError('Original native tokens changed')
            native.append(req)
            source_files[str(path)] = digest
            if req.get('inventory_ref'):
                inventory = owned_path(nas, req['inventory_ref'])
                source_files[str(inventory)] = sha256(inventory)
        bound, _, _, identities = _bindings(sid, task, run_id, responses, native)
        final_path = nas / 'sessions' / sid / 'finalization.json'
        if final_path.exists():
            final = read_json(final_path)
            if (final['rollout_id'] != rid or final['trajectory_sha256'] != row['trajectory_sha256']
                    or final['model_fingerprint'] != row['model_fingerprint']):
                raise ValueError('Original finalization/canonical identity mismatch')
            source_files[str(final_path)] = sha256(final_path)
        for path, event in rows:
            if event['sample_id'] in known_samples:
                continue
            locator = event.get('tensor_locator')
            if not locator:
                exclusions.append(dict(sample_id=event['sample_id'], reason='original_missing_publication_locator'))
                continue
            publication = nas / 'publications' / (Path(locator['shard']).stem + '.json')
            proof = read_json(publication)
            relative = str(path.relative_to(nas))
            if relative not in indices_seen:
                indices_seen[relative] = sha256(path)
            if (proof['files'].get(relative) != indices_seen[relative]
                    or proof['files'].get(locator['shard']) != locator['shard_sha256']
                    or locator['member_sha256'] != event['sha256']):
                raise ValueError('Original shard/index publication mismatch')
            source_files[str(path)] = indices_seen[relative]
            source_files[str(publication)] = sha256(publication)
            sample, reason = canonical_point(event, sid=sid, run_id=run_id, rid=rid, row=row,
                                            task=task, actions=actions[rid], bound=bound,
                                            identities=identities, require_recovery=False)
            if sample is None:
                exclusions.append(dict(sample_id=event['sample_id'], rollout_id=rid, reason=reason))
                continue
            identity = (rid, sample['agent_step'])
            if identity in predictions or any((s['rollout_id'], s['agent_step']) == identity for s in added_samples):
                raise ValueError('Original publication has a distinct extra close at one canonical step')
            sample.update(hidden_root=str(nas), capture_source='original_publication')
            added_samples.append(sample)
        suffix, _, _ = native_complete_suffix(bound, responses, [a['message_index'] for a in actions[rid]])
        requests = []
        for binding in suffix.values():
            req = dict(binding['request'])
            req['message_index'] = binding['message_index']
            req['is_action_response'] = any(a['message_index'] == binding['message_index'] for a in actions[rid])
            req['blocks'] = reasoning_blocks(req['prompt_token_ids'], req['output_token_ids'],
                                             req.get('open_token_ids', [248068]), req['close_token_ids'])
            requests.append(req)
        if requests:
            annotate_remaining_workloads({rid: requests})
            message_steps = {a['message_index']: step for step, a in enumerate(actions[rid])}
            for req in requests:
                blocks = [b for b in req['blocks'] if b.get('close') is not None]
                if (req['is_action_response'] and len(blocks) == 1 and not blocks[0]['remaining_censored']):
                    truth[(rid, message_steps[req['message_index']])] = blocks[0]['future_reasoning_tokens']
        if number % 25 == 0:
            print(f'original publications: inspected {number}/{len(sessions)} canonical sessions', flush=True)
    for i, features in iter_features(added_samples):
        sample = added_samples[i]
        score = float(np.asarray(predict_model(model, features[layer:layer+1].astype(float)))[0])
        if not np.isfinite(score):
            raise ValueError('Invalid original hidden prediction')
        predictions[(sample['rollout_id'], sample['agent_step'])] = (sample, score)
        locator = sample['tensor_locator']
        source_files[str(owned_path(nas, locator['shard']))] = locator['shard_sha256']
    source_files.update(proofs)
    return predictions, truth, dict(original_index_points_inspected=total_original,
        added_prediction_points=len(added_samples), native_truth_points=len(truth),
        exclusions=exclusions, exclusion_counts=dict(Counter(e['reason'] for e in exclusions)))

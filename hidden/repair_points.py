"""Lossless pointwise publication of retained capture bytes, without model replay.

Called by the locked capture repair CLI. Original evidence is never overwritten;
native-exact, identity-only, and orphan points are explicitly distinguished.
"""
from collections import Counter, defaultdict
from pathlib import Path
import copy
import hashlib
import io
import json

import numpy as np

from application.execution.evidence import RunEvidence
from core.io import hash_json, identifier, now, owned_path, read_json, sha256, write_json
from enrichment.online_trace import reasoning_targets
from hidden.artifacts import artifact_bytes, matching_request_receipts, read_request, response_records
from runtime.serving.spool import publish_files


def _digest(path, cache):
    key = str(path.resolve())
    if key not in cache:
        cache[key] = sha256(path)
    return cache[key]


def _native(roots, cohort, issues, proofs):
    requests, sources, conflicts = {}, defaultdict(list), set()
    for base in roots:
        paths = sorted({*(base / 'requests').glob('*.json'), *(base / 'sessions').glob('*/requests/*.json')})
        for number, path in enumerate(paths):
            try:
                # Original ledger and inventory validators own the token contract.
                digest = sha256(path)
                relative = str(path.relative_to(base))
                value = read_request(base, {'ledger_ref': relative, 'ledger_sha256': digest})
                if value.get('session_id') not in cohort:
                    continue
                if value.get('request_sha256') != hash_json(value['prompt_token_ids']):
                    raise ValueError('Native prompt token hash mismatch')
                if value.get('output_sha256') != hash_json(value['output_token_ids']):
                    raise ValueError('Native output token hash mismatch')
                rid = value['request_id']
                semantic = {k: v for k, v in value.items() if k not in {'ledger_ref', 'ledger_sha256'}}
                previous = requests.get(rid)
                if previous is not None and semantic != {k: v for k, v in previous.items() if k not in {'ledger_ref', 'ledger_sha256'}}:
                    conflicts.add(rid)
                    raise ValueError('Conflicting native request receipts: ' + rid)
                requests[rid] = value
                sources[rid].append((base, relative, digest))
                proofs[str(path)] = digest
                if value.get('inventory_ref'):
                    inventory = owned_path(base, value['inventory_ref'])
                    proofs[str(inventory)] = sha256(inventory)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                issues.append({'stage': 'native', 'path': str(path), 'reason': str(exc)})
            if number % 5000 == 0:
                print('point recovery: native', str(base), number, 'accepted', len(requests), flush=True)
    for rid in conflicts:
        requests.pop(rid, None)
    return requests, sources


def _event_groups(roots, cohort, issues, proofs):
    groups = defaultdict(lambda: defaultdict(list))
    for base in roots:
        for path in sorted((base / 'index').glob('*.json')):
            try:
                value = read_json(path)
                rows = value if isinstance(value, list) else [value]
                for row in rows:
                    sid = row.get('session_id')
                    if sid in cohort:
                        groups[sid][row['sample_id']].append((base, row))
                proofs[str(path)] = sha256(path)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                issues.append({'stage': 'index', 'path': str(path), 'reason': str(exc)})
    return groups


def _tensor(variants, nas, cfg, cache):
    identity = ('sample_id', 'session_id', 'run_id', 'task_id', 'request_id',
                'query_id', 'api_attempt_id', 'absolute_position', 'anchor', 'token_id')
    first = variants[0][1]
    if any(any(e.get(k) != first.get(k) for k in identity) for _, e in variants):
        raise ValueError('Contradictory duplicate point identity')
    if len({e['sha256'] for _, e in variants if e.get('sha256')}) > 1:
        raise ValueError('Contradictory duplicate tensor checksums')
    failures = []
    for base, original in sorted(variants, key=lambda x: x[0] != nas):
        event = copy.deepcopy(original)
        try:
            locator = event.get('tensor_locator')
            relative = locator['shard'] if locator else event.get('path', event.get('tensor'))
            path = owned_path(base, relative)
            digest = _digest(path, cache)
            if locator and digest != locator['shard_sha256']:
                raise ValueError('Tensor shard checksum mismatch')
            data = artifact_bytes(base, event)
            with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
            if len(arrays) != len(cfg['capture']['layers']):
                raise ValueError('Incomplete tensor layer count')
            if set(arrays) != set(event.get('storage_shapes', {})) or set(arrays) != set(event.get('storage_dtypes', {})):
                raise ValueError('Missing or inconsistent tensor layer metadata')
            for name, array in arrays.items():
                if not array.size or not np.isfinite(array).all():
                    raise ValueError('Empty or nonfinite tensor')
                if list(array.shape) != event['storage_shapes'][name] or str(array.dtype) != event['storage_dtypes'][name]:
                    raise ValueError('Tensor shape/dtype disagreement')
            if event.get('tensor_content_sha256'):
                content = hash_json({name: hashlib.sha256(array.tobytes()).hexdigest() for name, array in arrays.items()})
                if content != event['tensor_content_sha256']:
                    raise ValueError('Tensor content hash mismatch')
            member_digest = hashlib.sha256(data).hexdigest()
            if not locator and not event.get('sha256') and not event.get('tensor_content_sha256'):
                raise ValueError('No original integrity proof for loose tensor')
            if locator and locator.get('member_sha256') != member_digest:
                raise ValueError('Tensor member checksum mismatch')
            # Persist derived digest only in the independent repaired point row.
            event['sha256'] = member_digest
            return base, relative, digest, event
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(type(exc).__name__ + ': ' + str(exc))
    raise ValueError('No valid retained tensor: ' + ' | '.join(failures))


def _canonical(root, run_id, sid, rows, evidence, proofs):
    if len(rows) != 1:
        raise ValueError('Missing or ambiguous canonical trajectory: ' + str(len(rows)))
    rid, row = rows[0]
    path = owned_path(root, row['trajectory_path'])
    if not path.is_relative_to(root / 'runs' / run_id) or row['run_id'] != run_id:
        raise ValueError('Canonical trajectory belongs to another run')
    digest = sha256(path)
    if digest != row['trajectory_sha256']:
        raise ValueError('Canonical trajectory checksum mismatch')
    proofs[str(path)] = digest
    task = evidence['tasks'][row['task_key']]['instance_id']
    responses = list(response_records(read_json(path)))
    for _, message, _ in responses:
        md = (message.get('extra', {}).get('request') or {}).get('dycon_capture', {})
        if (md.get('session_id'), md.get('run_id'), md.get('task_id')) != (sid, run_id, task):
            raise ValueError('Canonical response capture identity mismatch')
    return rid, row, task, responses


def _bindings(sid, task, run_id, responses, requests):
    bound, expected, unbound = {}, set(), []
    identity = defaultdict(list)
    for index, message, response in responses:
        md = (message.get('extra', {}).get('request') or {}).get('dycon_capture', {})
        identity[(md.get('query_id'), md.get('api_attempt_id'))].append((index, response, md))
        candidates = matching_request_receipts(sid, message, response, requests)
        if len(candidates) != 1:
            unbound.append({'message_index': index, 'reason': 'native_receipt_missing_or_ambiguous', 'candidates': len(candidates)})
            continue
        req = candidates[0]
        try:
            if req.get('run_id') != run_id or req.get('task_id') != task or req.get('anchors') != ['think_close']:
                raise ValueError('Native receipt task/run/anchor conflict')
            if req['request_id'] in bound:
                raise ValueError('Reused native request binding')
            if response.get('dycon_native_request_id') not in {None, req['request_id']}:
                raise ValueError('Native request ID disagreement')
            if response.get('prompt_token_ids') is not None and response['prompt_token_ids'] != req['prompt_token_ids']:
                raise ValueError('Native prompt tokens disagree with canonical response')
            choice = (response.get('choices') or [{}])[0]
            output = choice.get('token_ids', (choice.get('provider_specific_fields') or {}).get('token_ids'))
            if output is not None and output != req['output_token_ids']:
                raise ValueError('Native output tokens disagree with canonical response')
            if not req.get('close_token_ids'):
                raise ValueError('Missing native close token inventory')
            targets, _ = reasoning_targets(req, req.get('open_token_ids', [248068]), req['close_token_ids'])
            expected.update((req['request_id'], pos, anchor) for pos, anchor in targets)
            bound[req['request_id']] = {'message_index': index, 'request': req, 'targets': set(targets)}
        except (ValueError, KeyError, TypeError) as exc:
            unbound.append({'message_index': index, 'reason': str(exc)})
    return bound, expected, unbound, identity


def _classification(event, sid, run_id, task, fingerprint, bound, identity):
    if event.get('session_id') != sid or event.get('run_id') != run_id:
        raise ValueError('Captured point belongs to another run/session')
    if task is not None and event.get('task_id') != task:
        raise ValueError('Captured point task identity conflict')
    if fingerprint is not None and event.get('model_fingerprint') != fingerprint:
        raise ValueError('Captured point model identity conflict')
    request_id = event.get('request_id')
    binding = bound.get(request_id)
    if binding is not None:
        req = binding['request']; position = event['absolute_position']
        if (position, event.get('anchor')) not in binding['targets']:
            return 'diagnostic_non_target', None
        tokens = req['prompt_token_ids'] + req['output_token_ids']
        if not 0 <= position < len(tokens) - 1 or tokens[position] != event['token_id']:
            raise ValueError('Captured point token coordinate conflict')
        for key in ('query_id', 'api_attempt_id', 'task_id', 'run_id', 'request_sha256'):
            if event.get(key) != req.get(key):
                raise ValueError('Captured point/native receipt disagreement: ' + key)
        return 'native_exact', binding['message_index']
    candidates = identity.get((event.get('query_id'), event.get('api_attempt_id')), [])
    if len(candidates) == 1:
        index, response, md = candidates[0]
        rid = response.get('id')
        if (rid and request_id and (request_id == rid or request_id.startswith(rid + '-'))
                and event.get('query_id') and event.get('api_attempt_id')):
            return 'canonical_identity_only', index
    return 'diagnostic_orphan', None


def _publish_dependencies(dependencies, nas, cache, deadline):
    by_source = defaultdict(dict)
    for base, relative, digest in dependencies:
        path = owned_path(nas, relative)
        if path.exists():
            if _digest(path, cache) != digest:
                raise ValueError('NAS dependency content conflict: ' + relative)
        else:
            by_source[base][relative] = digest
    for base, files in by_source.items():
        publish_files(base, nas, files, deadline)
        for relative, digest in files.items():
            cache[str(owned_path(nas, relative).resolve())] = digest


def recover_points(cfg, run_id, *, rollout_ids=(), session_ids=(), apply=False, repair_id=None):
    """Recover every integrity-checked point; caller owns the exclusive run lock."""
    root = Path(cfg['project_root']).resolve(); run = root / 'runs' / run_id
    if apply and not repair_id:
        raise ValueError('Pointwise apply requires --repair-id')
    if cfg.get('parallel', {}).get('enabled') and cfg['parallel'].get('branch_count', 1) > 1:
        raise ValueError('Pointwise repair currently supports nonparallel capture only')
    if cfg['capture']['anchors'] != ['think_close']:
        raise ValueError('Pointwise repair requires close-only capture')
    evidence = RunEvidence(root, run_id).read()
    canonical = defaultdict(list)
    for rid, row in evidence['rollouts'].items():
        key = row.get('sample_key') or row.get('provenance', {}).get('sample_key')
        if key and row.get('attempt_id'):
            sid = identifier('cap-' + key + '-' + row['attempt_id'])
            canonical[sid].append((rid, row))
    cohort = set(session_ids)
    if rollout_ids:
        cohort.update(sid for sid, rows in canonical.items() if any(rid in rollout_ids for rid, _ in rows))
    if not cohort:
        cohort = set(canonical)
    spool = root / cfg['capture']['export']['spool_root'] / run_id
    nas = Path(cfg['paths']['nas_root']) / cfg['capture']['export']['root'] / run_id
    roots = [p for p in (spool, nas) if p.is_dir()]
    out = run / 'hidden/capture-repairs' / identifier(repair_id or 'pointwise-audit')
    signature = {'run_id': run_id, 'repair_id': repair_id, 'sessions': sorted(cohort), 'config_sha256': sha256(run/'resolved-config.json')}
    if apply:
        out.mkdir(parents=True, exist_ok=True)
        if (out/'selection.json').exists() and read_json(out/'selection.json') != signature:
            raise ValueError('Pointwise recovery namespace selection conflict')
        write_json(out/'selection.json', signature)
    issues, proofs, cache = [], {}, {}
    if apply and (out/'source-proofs.json').exists():
        proofs.update(read_json(out/'source-proofs.json'))
    if apply and (nas/'capture-repairs'/repair_id/'receipt.json').exists():
        manifest = read_json(nas/'capture-repairs'/repair_id/'receipt.json')
        if manifest['run_id'] != run_id or manifest['repair_id'] != repair_id:
            raise ValueError('Published repair manifest identity mismatch')
        for relative, digest in {**manifest['files'], **manifest['session_receipts']}.items():
            if _digest(owned_path(nas,relative),cache) != digest:
                raise ValueError('Published repair manifest checksum mismatch')
        for relative in manifest['session_receipts']:
            receipt = read_json(owned_path(nas,relative))
            for dependency,digest in {**receipt['files'],**receipt['dependencies']}.items():
                if _digest(owned_path(nas,dependency),cache) != digest:
                    raise ValueError('Published repaired dependency checksum mismatch')
        return read_json(nas/'capture-repairs'/repair_id/'report.json')
    if apply: write_json(out/'progress.json', {'phase':'native_receipts', 'total':len(cohort), 'finished':0,'updated_at':now()})
    requests, sources = _native(roots, cohort, issues, proofs)
    request_groups = defaultdict(list)
    for req in requests.values(): request_groups[req['session_id']].append(req)
    print('point recovery: loading event metadata', flush=True)
    groups = _event_groups(roots, cohort, issues, proofs)
    if apply: write_json(out/'source-proofs.json', proofs)
    reports=[]; prefix='capture-repairs/'+identifier(repair_id or 'pointwise-audit')
    payload=out/'payload'; deadline=cfg['capture']['export']['upload_max_elapsed_seconds']
    for number,sid in enumerate(sorted(cohort),1):
        session_report=out/'session-reports'/(sid+'.json')
        if apply and session_report.exists():
            prior=read_json(session_report)
            if prior.get('published') and sha256(nas/prior['receipt'])==prior['receipt_sha256']:
                reports.append(prior); continue
        errors=[]; points=[]; dependencies=[]; bound={}; expected=set(); identity={}; unbound=[]
        rid=row=task=None; responses=[]; fingerprint=None; canonical_error=None
        try:
            rid,row,task,responses=_canonical(root,run_id,sid,canonical.get(sid,[]),evidence,proofs)
            fingerprint=row['model_fingerprint']
            bound,expected,unbound,identity=_bindings(sid,task,run_id,responses,request_groups[sid])
        except (OSError,ValueError,KeyError,TypeError) as exc:
            canonical_error=str(exc)
            # Bytes may be preserved diagnostically, never promoted to canonical.
            rid=row=task=fingerprint=None; bound={}; identity={}
        session_metadata=[]
        for base in roots:
            path=base/'sessions'/sid/'session.json'
            if path.exists():
                try:
                    meta=read_json(path)
                    if meta['session_id']!=sid or meta['metadata']['run_id']!=run_id:
                        raise ValueError('Session metadata run/id mismatch')
                    session_metadata.append(meta)
                    proofs[str(path)]=sha256(path)
                except (ValueError,KeyError,OSError) as exc: errors.append({'stage':'session_metadata','reason':str(exc)})
        metadata_conflict=bool(session_metadata and any(m!=session_metadata[0] for m in session_metadata))
        if metadata_conflict: errors.append({'stage':'session_metadata','reason':'conflicting session records'})
        observed=set(); counts=Counter()
        for sample_id,variants in groups.get(sid,{}).items():
            try:
                base,relative,digest,event=_tensor(variants,nas,cfg,cache)
                classification,index=_classification(event,sid,run_id,task,fingerprint,bound,identity)
                if metadata_conflict: classification='diagnostic_session_conflict'; index=None
                if classification=='native_exact': observed.add((event['request_id'],event['absolute_position'],event['anchor']))
                event['recovery']={'repair_id':repair_id,'binding':classification,'canonical_rollout_id':rid,
                    'canonical_message_index':index,'trajectory_sha256':row['trajectory_sha256'] if row else None,
                    'eligible_for_training':False}
                points.append(event); counts[classification]+=1
                dependencies.append((base,relative,digest))
                proofs[str(owned_path(base,relative))]=digest
            except (OSError,ValueError,KeyError,TypeError) as exc:
                errors.append({'stage':'point','sample_id':sample_id,'reason':str(exc)})
        native_ids=set(bound)|{p.get('request_id') for p in points}
        for req_id in native_ids:
            if req_id in requests and sources.get(req_id):
                source=sources[req_id][0]; dependencies.append(source)
                inv=requests[req_id].get('inventory_ref')
                if inv: dependencies.append((source[0],inv,proofs[str(owned_path(source[0],inv))]))
        missing=sorted(expected-observed)
        complete=bool(responses) and canonical_error is None and not unbound and not missing and not metadata_conflict
        status='complete' if complete else 'partial' if points else 'unrecoverable'
        coverage={'expected_known':len(expected),'observed_native_exact':len(observed),'missing_known':missing,
            'expected_count_complete':canonical_error is None and not unbound, 'unbound_responses':unbound,
            'canonical_error':canonical_error,'retained_points':len(points),'binding_counts':dict(counts),
            'eligible_for_training':False,'status':status}
        record={'session_id':sid,'rollout_id':rid,'status':status,'source_sample_ids':len(groups.get(sid,{})),
            'recovered_points':len(points),'binding_counts':dict(counts),'coverage':coverage,'errors':errors,'published':False}
        if apply:
            base_rel=prefix+'/sessions/'+sid
            index_rel=base_rel+'/point-index.json'; coverage_rel=base_rel+'/coverage.json'
            write_json(payload/index_rel,{'schema_version':1,'run_id':run_id,'repair_id':repair_id,'session_id':sid,'tensor_root':str(nas),'points':points})
            write_json(payload/coverage_rel,coverage)
            try:
                _publish_dependencies(dependencies,nas,cache,deadline)
                files={index_rel:sha256(payload/index_rel),coverage_rel:sha256(payload/coverage_rel)}
                publish_files(payload,nas,files,deadline)
                receipt_rel=base_rel+'/receipt.json'
                receipt={'schema_version':1,'run_id':run_id,'repair_id':repair_id,'session_id':sid,
                    'files':files,'dependencies':{relative:digest for _,relative,digest in dependencies},
                    'points':len(points),'status':status,'canonical_rollout_id':rid,'created_at':now()}
                # Never overwrite a previously acknowledged receipt on a retry.
                if (nas/receipt_rel).exists():
                    previous=read_json(nas/receipt_rel)
                    if {k:v for k,v in previous.items() if k!='created_at'} != {k:v for k,v in receipt.items() if k!='created_at'}:
                        raise ValueError('Repair receipt content conflict')
                    receipt=previous
                write_json(payload/receipt_rel,receipt)
                receipt_digest=sha256(payload/receipt_rel)
                publish_files(payload,nas,{receipt_rel:receipt_digest},deadline)
                record.update(published=True,receipt=receipt_rel,receipt_sha256=receipt_digest)
            except (OSError,ValueError,TimeoutError) as exc:
                record['publication_error']=str(exc)
            write_json(session_report,record)
        reports.append(record)
        totals=Counter()
        for r in reports: totals.update(r['binding_counts'])
        progress={'phase':'publishing','total':len(cohort),'finished':number,
            'published_sessions':sum(r['published'] for r in reports),'points':sum(r['recovered_points'] for r in reports),
            'binding_counts':dict(totals),'last_session':sid,'updated_at':now()}
        if apply: write_json(out/'progress.json',progress)
        print('point recovery:',number,'/',len(cohort),sid,status,len(points),'published',record['published'],flush=True)
    totals=Counter()
    for record in reports: totals.update(record['binding_counts'])
    report={'schema_version':1,'mode':'pointwise','run_id':run_id,'repair_id':repair_id,'created_at':now(),
        'summary':{'sessions':len(reports),'states':dict(Counter(r['status'] for r in reports)),
            'published_sessions':sum(r['published'] for r in reports),'points':sum(r['recovered_points'] for r in reports),
            'published_points':sum(r['recovered_points'] for r in reports if r['published']),
            'binding_counts':dict(totals),'known_missing_points':sum(len(r['coverage']['missing_known']) for r in reports),
            'unbound_responses':sum(len(r['coverage']['unbound_responses']) for r in reports),
            'rejected_points':sum(sum(e['stage']=='point' for e in r['errors']) for r in reports)},
        'sessions':reports,'input_issues':issues,'nas_root':str(nas),'source_proofs_sha256':None}
    if apply:
        write_json(out/'source-proofs.json',proofs)
        report['source_proofs_sha256']=sha256(out/'source-proofs.json')
        write_json(out/'report.json',report)
        write_json(payload/(prefix+'/report.json'),report)
        write_json(payload/(prefix+'/source-proofs.json'),proofs)
        files={prefix+'/report.json':sha256(payload/(prefix+'/report.json')),prefix+'/source-proofs.json':sha256(payload/(prefix+'/source-proofs.json'))}
        publish_files(payload,nas,files,deadline)
        report_receipt={'schema_version':1,'run_id':run_id,'repair_id':repair_id,'files':files,
            'session_receipts':{r['receipt']:r['receipt_sha256'] for r in reports if r['published']}}
        rel=prefix+'/receipt.json'; write_json(payload/rel,report_receipt)
        publish_files(payload,nas,{rel:sha256(payload/rel)},deadline)
        write_json(out/'progress.json',{'phase':'completed','updated_at':now(),**report['summary']})
    return report

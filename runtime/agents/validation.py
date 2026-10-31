"""Read-only checks of live OpenHands execution, native capture and branch evidence."""
from pathlib import Path
import numpy as np
from core.io import hash_json, owned_path, read_json, sha256, write_json
from rollout.decision.coordinator import RANDOM_SELECTION_MODE


def audit_step_coverage(requests, points):
    """Use every canonical assistant response, including the final answer."""
    from collections import Counter
    counts = Counter(r['request_id'] for r in requests)
    steps, missing, duplicate, unfinished = [], [], [], []
    for index, request in enumerate(requests):
        rid = request['request_id']
        output = request['output_token_ids']
        closes = set(request.get('close_token_ids') or [])
        positions = [i for i, token in enumerate(output) if token in closes]
        matches = [p for p in points if p['request_id'] == rid]
        errors = []
        if not positions:
            errors.append('explicit_close_missing')
        elif positions[0] + 1 >= len(output):
            errors.append('no_forward_after_close')
            unfinished.append(rid)
        if len(matches) != 1:
            errors.append('hidden_missing' if not matches else 'duplicate_hidden')
        if len(matches) > 1 or counts[rid] != 1:
            duplicate.append(rid)
        if positions and len(matches) == 1:
            point = matches[0]
            position = len(request['prompt_token_ids']) + positions[0]
            if point.get('token_id') != output[positions[0]] or point.get('absolute_position') != position:
                errors.append('hidden_token_position_mismatch')
            if point.get('graph_execution') != 'full':
                errors.append('hidden_not_full_graph')
        if errors:
            missing.append(rid)
        steps.append({'step_id': index, 'request_id': rid, 'explicit_close_positions': positions,
                      'hidden_sample_ids': [p['sample_id'] for p in matches], 'errors': errors,
                      'verified': not errors})
    unexpected = sorted({p['request_id'] for p in points} - set(counts))
    return {'schema_version': 1, 'denominator': 'all_canonical_assistant_responses_including_final',
            'expected_steps': len(requests), 'verified_steps': sum(s['verified'] for s in steps),
            'missing': missing, 'duplicate': sorted(set(duplicate)), 'unexpected': unexpected,
            'unfinished': unfinished, 'steps': steps,
            'verified': bool(requests) and not (missing or duplicate or unexpected or unfinished)}


def audit_terminal_decision(decision):
    """A terminal decision has candidates but no selected/executed winner."""
    from rollout.loop_escape import validate_loop_escape_evidence
    validate_loop_escape_evidence(decision)
    candidates = decision.get('candidates') or []
    if (decision.get('selected_branch_id') is not None or not candidates
            or any(c.get('selected') or c.get('status') == 'eligible' for c in candidates)):
        raise ValueError('Terminal parallel decision contains an eligible or selected winner')
    return {'decision_id': decision['decision_id'], 'candidate_count': len(candidates),
            'candidate_reasons': [c.get('reason') for c in candidates], 'action_executed': False}


def audit_api_terminals(attempt):
    """Report every API outcome separately from accepted canonical steps."""
    from collections import Counter
    requests = [read_json(p) for p in (Path(attempt) / 'requests').glob('req-*.json')]
    missing, unfinished, failed, native_ids = [], [], [], []
    for request in requests:
        path = Path(attempt) / 'terminal-receipts' / (request['request_id'] + '.json')
        if not path.exists():
            missing.append(request['request_id'])
        else:
            terminal = read_json(path)
            if (terminal['request_id'] != request['request_id'] or terminal['binding'] != request['binding']
                    or terminal['status'] != request.get('status')):
                raise ValueError('API terminal receipt binding or status differs from its request')
        if request.get('status') not in {'completed', 'failed', 'cancelled'} or not request.get('finished_at'):
            unfinished.append(request['request_id'])
        if request.get('status') in {'failed', 'cancelled'}:
            failed.append({'request_id': request['request_id'], 'role': request.get('role'),
                           'step_id': request.get('step_id'), 'status': request['status'], 'error': request.get('error')})
        for call in request.get('model_calls', []):
            reference = call.get('native_terminal')
            if reference:
                path = Path(reference['path'])
                if sha256(path) != reference['sha256']:
                    raise ValueError('Physical model call native terminal checksum mismatch')
                native = read_json(path)
                if native['owner']['api_attempt_id'] != call['api_attempt_id']:
                    raise ValueError('Physical model call native API identity mismatch')
                native_ids.append(native['native_request_id'])
    duplicate = sorted(k for k, count in Counter(native_ids).items() if count > 1)
    return {'physical_api_attempts': len(requests), 'model_calls': sum(len(r.get('model_calls', [])) for r in requests),
            'failed_or_cancelled_api_attempts': failed, 'missing': missing, 'duplicate': duplicate,
            'unfinished': unfinished, 'verified': not (missing or duplicate or unfinished)}


def audit_candidate(root, candidate, branch, expected, contract, predictor, num_layers,
                    require_score=True, capture_layers=None):
    """Verify both winning and losing native evidence with the published loader."""
    from hidden.artifacts import read_request, read_arrays
    ledger = read_request(root, candidate)
    if not candidate.get('ledger_ref') or any(ledger.get(k) != v for k, v in expected.items()):
        raise ValueError('Candidate native ledger identity mismatch')
    if (ledger['request_id'] != candidate['native_request_id']
            or hash_json(ledger['prompt_token_ids']) != ledger['request_sha256']
            or hash_json(ledger['output_token_ids']) != candidate['output_sha256']
            or len(ledger['output_token_ids']) != candidate['completion_tokens']):
        raise ValueError('Candidate native tokens or cost mismatch')
    close = next((e for e in branch['close_events']
                  if e['sample_id'] == candidate.get('close_sample_id')), None)
    score = candidate.get('score')
    if not isinstance(score, (int, float)) or not np.isfinite(score):
        if candidate.get('status') == 'eligible' and require_score:
            raise ValueError('Eligible candidate has no finite predictor score')
        return None
    if close is None or close['request_id'] != ledger['request_id']:
        raise ValueError('Scored candidate has no bound native close tensor')
    arrays = read_arrays(root, close)
    layers = {key: value for key, value in arrays.items() if key.startswith('layer_')}
    if ((capture_layers is not None and len(layers) != len(capture_layers))
            or any(value.shape != (contract['hidden_size'],) or not np.isfinite(value).all()
                   for value in layers.values())):
        raise ValueError('Candidate native layer tensors are incomplete or invalid')
    features = np.stack([arrays[f'layer_{num_layers if s == num_layers - 1 else s}']
                         for s in contract['sites']])[None]
    if features.shape != (1, len(contract['sites']), contract['hidden_size']) or not np.isfinite(features).all():
        raise ValueError('Candidate predictor features are invalid')
    offline = float(predictor.predict(features)[0])
    if not np.isfinite(offline) or not np.isclose(score, offline, rtol=1e-10, atol=1e-10):
        raise ValueError('Published predictor disagrees with candidate online score')
    return abs(score - offline)


def audit_run(root, run_id, require_resolved=False, require_parallel=False, rollout_ids=()):
    from application.execution.run_plan import load_plan
    from application.execution.evidence import RunEvidence
    from hidden.artifacts import read_request, read_arrays, event_records, response_records, read_branch_index
    loaded=load_plan(root,run_id); cfg=loaded['config'];run=loaded['run']
    stage=read_json(run/'status.json').get('stage')
    if stage!='complete' and (not rollout_ids or stage not in {'failed','cancelled','rollout'}):
        raise ValueError('Audit requires terminal evidence; failed runs need explicit rollout scope')
    evidence=RunEvidence(root,run_id);evidence.verify();data=evidence.read()
    wanted=set(rollout_ids) if rollout_ids else set(data['rollouts'])
    if not wanted or not wanted<=set(data['rollouts']):raise ValueError('Unknown or empty audit rollout scope')
    nas=Path(cfg['paths']['nas_root'])/cfg['capture']['export']['root']/run_id
    events=[row for _,row in event_records(nas)] if cfg['capture']['enabled'] else []
    outcomes=[]
    checked_shards = {}
    for rid,row in data['rollouts'].items():
        if rid not in wanted:continue
        if stage=='rollout' and row.get('evaluation_status')!='completed':
            raise ValueError('Live audit scope requires a finished official evaluation')
        if require_resolved and row.get('resolved') is not True:
            raise ValueError('Official verifier did not resolve '+rid)
        trajectory=read_json(owned_path(root,row['trajectory_path']))
        responses=list(response_records(trajectory));native=[];points=[];parallel=[];native_rows=[]
        for message_index,message,response in responses:
            if cfg['capture']['enabled']:
                request=read_request(response['ledger_root'],response)
                if (hash_json(request['prompt_token_ids'])!=request['request_sha256']
                        or hash_json(request['output_token_ids']) != request.get('output_sha256')
                        or request['run_id']!=run_id):
                    raise ValueError('Native prompt identity mismatch')
                native.append(request['request_id'])
                native_rows.append(request)
            summary=(message.get('extra') or {}).get('dycon_parallel')
            if summary:parallel.append(summary)
        if len(native)!=len(set(native)):raise ValueError('A native request mapped to two decisions')
        for point in events:
            if point.get('request_id') not in set(native) or point.get('anchor')!='think_close':continue
            native_row = next(r for r in native_rows if r['request_id'] == point['request_id'])
            if (point.get('model_fingerprint') != cfg['model']['fingerprint']
                    or any(point.get(k) != native_row.get(k) for k in
                           ('run_id', 'session_id', 'task_id', 'step_id', 'api_attempt_id', 'tokenizer_fingerprint'))):
                raise ValueError('Close hidden model/tokenizer/request identity mismatch')
            locator = point.get('tensor_locator') or {}
            if locator:
                shard = owned_path(nas, locator['shard'])
                if str(shard) not in checked_shards:
                    checked_shards[str(shard)] = sha256(shard)
                digest = checked_shards[str(shard)]
                if digest != locator['shard_sha256']:
                    raise ValueError('NAS close hidden shard checksum mismatch')
            arrays=read_arrays(nas,point)
            layers={k:v for k,v in arrays.items() if k.startswith('layer_')}
            if len(layers)!=len(cfg['capture']['layers']):raise ValueError('Incomplete captured layers')
            if any(v.shape[-1]!=4096 or not np.isfinite(v).all() for v in layers.values()):
                raise ValueError('Invalid captured feature shape/values')
            points.append({'request_id':point['request_id'],'sample_id':point['sample_id'],
                'token_id':point.get('token_id'),'absolute_position':point['absolute_position'],
                'tensor_sha256':point['sha256'],'layers':len(layers),
                'graph_execution':point.get('graph_execution'),
                'shard_sha256':locator.get('shard_sha256'),'member_sha256':locator.get('member_sha256'),
                'tensor_locator':locator})
        step_coverage = audit_step_coverage(native_rows, points) if cfg['capture']['enabled'] else None
        if step_coverage is not None:
            write_json(run / 'validation' / ('skillsbench-step-coverage-' + rid + '.json'), {
                'run_id': run_id, 'rollout_id': rid, 'config_hash': cfg['config_hash'],
                'trajectory_sha256': row['trajectory_sha256'], **step_coverage})
        if cfg['capture']['enabled']:
            if not native or not points:raise ValueError('No exact native close points')
            completed=read_json(owned_path(root,row['provenance']['source']).parent.parent/'capture-finalization.json')
            if completed.get('capture_status')!='complete':raise ValueError('Attempt capture is not finalized')
            session=read_request(responses[0][2]['ledger_root'],responses[0][2])['session_id']
            final=read_json(nas/'sessions'/session/'finalization.json')
            if final['accepted_native_request_ids']!=native:raise ValueError('Native finalization mapping changed')
            coverage=read_json(nas/'sessions'/session/'coverage.json')
            if coverage['missing']:raise ValueError('Missing native close features')
            representatives={native[0],native[len(native)//2],native[-1]}
            if not representatives<={p['request_id'] for p in points}:
                raise ValueError('First/middle/last native close point is unavailable')
        if require_parallel and not parallel:raise ValueError('No real branch decisions')
        selected_ids={c['id'] for _,m,_ in responses for c in m.get('tool_calls') or []}
        observations={m['tool_call_id'] for _,message,_ in responses
            for m in (message.get('extra',{}).get('request',{}).get('messages') or [])
            if m.get('role')=='tool' and m.get('tool_call_id')}
        acp_results={event.get('event',event).get('tool_call_id')
            for event in trajectory.get('raw_acp_events',[])
            if event.get('event',event).get('type')=='tool_call'
            and event.get('event',event).get('status') in {'completed','failed'}}
        acp_results.discard(None)
        observations.update(acp_results)
        required_results={c['id'] for _,m,_ in responses for c in m.get('tool_calls') or []
                          if c.get('function',{}).get('name') not in {'finish','think'}}
        if required_results and not required_results&observations:raise ValueError('No actual selected tool execution was observed')
        if not acp_results<=selected_ids:raise ValueError('Unbound tool execution appeared in the OpenHands event stream')
        candidate_count=sum(s['branch_count'] for s in parallel)
        if parallel:
            import importlib
            from runtime.serving.parallel import artifact_contract
            from runtime.serving.parallel_validation import validate_selection
            contract = artifact_contract(cfg['parallel']['estimator_ref'], cfg['parallel']['estimator_sha256'],
                                         cfg['model']['fingerprint'])
            module, name = contract['loader'].split(':')
            predictor = getattr(importlib.import_module(module), name)(Path(cfg['parallel']['estimator_ref']).parent, contract)
            model_config = read_json(Path(cfg['model']['path']) / 'config.json')
            num_layers = model_config.get('text_config', model_config)['num_hidden_layers']
            branch=read_branch_index(nas,session)
            losing_calls=set()
            decisions=[read_json(p) for p in (nas/'sessions'/session/'parallel/decisions').glob('*.json')]
            canonical_ids = {s['decision_id'] for s in parallel}
            decision_ids = [d['decision_id'] for d in decisions]
            if len(decision_ids) != len(set(decision_ids)) or not canonical_ids <= set(decision_ids):
                raise ValueError('Missing or duplicate decision receipts')
            terminal_decisions = []
            for decision in decisions:
                if decision['decision_id'] not in canonical_ids:
                    terminal_decisions.append(audit_terminal_decision(decision))
            all_candidate_count = sum(d['branch_count'] for d in decisions)
            if len(branch['samples']) != all_candidate_count:
                raise ValueError('Missing all-candidate native evidence, including terminal decisions')
            branch_rows = {(b['decision_id'], b['branch_id']): b for b in branch['samples']}
            distinct=0;candidate_tokens=0;score_errors=[];candidate_native_ids=set()
            for decision in decisions:
                if decision['selected_branch_id'] is not None:
                    validate_selection(decision, cfg['generation']['seed'], cfg['parallel'])
                hashes={c.get('output_sha256') for c in decision['candidates'] if c.get('output_sha256')}
                distinct+=int(len(hashes)>1)
                candidate_tokens+=sum(c.get('completion_tokens',0) for c in decision['candidates'])
                winner=next((c for c in decision['candidates'] if c['branch_id']==decision['selected_branch_id']), None)
                if (winner is not None and decision.get('selection_mode') != RANDOM_SELECTION_MODE
                        and (not isinstance(winner.get('score'),(int,float)) or not np.isfinite(winner['score']))):
                    raise ValueError('Winner has no finite real predictor score')
                if winner is not None and decision.get('selection_mode')=='minimum_score':
                    eligible=[c for c in decision['candidates'] if c.get('status')=='eligible'
                              and isinstance(c.get('score'),(int,float)) and np.isfinite(c['score'])]
                    if not eligible or min(eligible,key=lambda c:(c['score'],c['branch_id']))['branch_id']!=winner['branch_id']:
                        raise ValueError('Winner differs from the real predictor minimum')
                for item in decision['candidates']:
                    expected = {'run_id': run_id, 'session_id': session, 'decision_id': decision['decision_id'],
                                'branch_id': item['branch_id'], 'api_attempt_id': item['api_attempt_id']}
                    error = audit_candidate(nas, item, branch_rows[(decision['decision_id'], item['branch_id'])],
                                            expected, contract, predictor, num_layers,
                                            decision.get('selection_mode') != RANDOM_SELECTION_MODE, cfg['capture']['layers'])
                    if item['native_request_id'] in candidate_native_ids:
                        raise ValueError('Two candidates share a native request')
                    candidate_native_ids.add(item['native_request_id'])
                    if error is not None:score_errors.append(error)
                    if item.get('selected'):continue
                    for choice in (item.get('response') or {}).get('choices') or []:
                        losing_calls.update(c['id'] for c in (choice.get('message') or {}).get('tool_calls') or [])
            if (losing_calls-selected_ids)&observations:raise ValueError('A losing tool call was executed')
            if not distinct:raise ValueError('No decision exercised distinct native candidate outputs')
        outcomes.append({'rollout_id':rid,'resolved':row.get('resolved'),
            'canonical_decisions':len(responses),'native_requests':len(native),'exact_close_points':len(points),
            'first_middle_last_native':([native[0],native[len(native)//2],native[-1]] if native else []),
            'tool_results_matched':len(selected_ids&observations),'parallel_decisions':len(parallel),
            'unexecuted_or_sdk_rejected_tool_ids':sorted(required_results-observations),
            'candidate_count':candidate_count,'points':points})
        attempt = owned_path(root, row['provenance']['source']).parent.parent
        outcomes[-1]['api_terminal_coverage'] = audit_api_terminals(attempt)
        if cfg['capture']['enabled']:
            outcomes[-1]['step_coverage'] = step_coverage
        if parallel:
            outcomes[-1].update(distinct_candidate_decisions=distinct,all_candidate_completion_tokens=candidate_tokens,
                                all_candidate_count=all_candidate_count,
                                canonical_decision_candidate_count=candidate_count, terminal_decisions=terminal_decisions,
                                losing_tool_calls_not_executed=len(losing_calls-selected_ids),
                                verified_candidate_native_requests=len(candidate_native_ids),
                                independently_recomputed_scores=len(score_errors),
                                max_score_absolute_error=max(score_errors) if score_errors else None)
    scope={rid:{'rollout':data['rollouts'][rid],
        'evaluation': (data['evaluations'].get(data['rollouts'][rid].get('selected_evaluation_id'))
                       if data['rollouts'][rid].get('selected_evaluation_id') else None)} for rid in wanted}
    proof={'schema_version':4,'run_id':run_id,'config_hash':cfg['config_hash'],
        'evidence_scope_sha256':hash_json(scope),'scope_rollout_ids':sorted(wanted),
        'whole_run_acceptance':not bool(rollout_ids),
        'outcomes':outcomes,'verified':all(r.get('step_coverage', {}).get('verified', True)
            and r['api_terminal_coverage']['verified'] for r in outcomes)}
    proof['receipt_sha256']=hash_json(proof)
    suffix='-'+hash_json(sorted(wanted))[:10] if rollout_ids else ''
    target=run/('validation/skillsbench-audit-v4'+suffix+'.json')
    if target.exists() and read_json(target)!=proof:raise ValueError('Audit evidence changed')
    if not target.exists():write_json(target,proof)
    return proof

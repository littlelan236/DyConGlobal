"""Immutable parallel capture views; no dataset or training eligibility."""
from collections import Counter
import copy
import math
from pathlib import Path
from uuid import uuid4

from application.execution.evidence import RunEvidence
from action.datasets.trajectories import read_actions
from core.io import hash_json, identifier, now, owned_path, read_json, sha256, write_json
from enrichment.online_trace import publication_directory, reasoning_blocks, annotate_remaining_workloads
from hidden.artifacts import read_repaired_session, read_request
from hidden.parallel_proof import decision_inventory, validate_canonical_decisions
from hidden.publish_online import native_complete_suffix
from hidden.repair_points import _canonical, _bindings


def read_visualization_publication(cfg, run_id, publication_id, *, repair_id=None):
    root = Path(cfg['project_root']).resolve()
    output = publication_directory(root, run_id, identifier(publication_id))
    receipt = read_json(output / 'receipt.json')
    signature = receipt['publication_signature']
    if (signature.get('run_id') != run_id or signature.get('publication_id') != publication_id
            or signature.get('purpose') != 'visualization' or signature.get('schema_version') != 1):
        raise ValueError('Visualization publication identity/schema mismatch')
    if repair_id is not None and signature['repair_id'] != repair_id:
        raise ValueError('Publication and repair ID disagree')
    for name,digest in receipt['files'].items():
        if sha256(owned_path(output,name)) != digest:
            raise ValueError('Visualization publication checksum mismatch: ' + name)
    for source,digest in receipt['publication_sources'].items():
        if sha256(Path(source)) != digest:
            raise ValueError('Visualization source checksum mismatch: ' + source)
    return {'directory':output,'receipt':receipt,'signature':signature,
            **{name:read_json(output / (name+'.json')) for name in
               ('publication','canonical-point-index','canonical-truth','branch-point-index','decisions','coverage')}}


def publish_visualization(cfg, run_id, *, repair_id, publication_id, selection, selection_sha256,
                          checkpoint=lambda:None):
    """Caller owns run lock. Publish all proven hidden and independent safe truth."""
    root = Path(cfg['project_root']).resolve()
    run = root / 'runs' / identifier(run_id)
    nas = Path(cfg['paths']['nas_root']) / cfg['capture']['export']['root'] / run_id
    repair = nas / 'capture-repairs' / identifier(repair_id)
    repair_receipt = read_json(repair / 'receipt.json')
    if repair_receipt.get('selection_sha256') != selection_sha256:
        raise ValueError('Visualization repair selection mismatch')
    signature = dict(schema_version=1, purpose='visualization', run_id=run_id,
        publication_id=identifier(publication_id), repair_id=repair_id, selection_sha256=selection_sha256,
        repair_receipt_sha256=sha256(repair/'receipt.json'), policy='parallel_visualization_v2')
    output = publication_directory(root,run_id,publication_id)
    if output.exists():
        view = read_visualization_publication(cfg,run_id,publication_id,repair_id=repair_id)
        if view['signature'] != signature:
            raise ValueError('Immutable visualization publication selection conflict')
        return view['publication']
    sources = {str(repair/'receipt.json'):sha256(repair/'receipt.json'),
               str(run/'resolved-config.json'):sha256(run/'resolved-config.json')}
    audit = run/'hidden/capture-repairs'/repair_id
    for name in ('publication-selection.json','publication-selection-receipt.json'):
        sources[str(audit/name)]=sha256(audit/name)
    frozen_selection=read_json(audit/'publication-selection.json')
    selection_receipt=read_json(audit/'publication-selection-receipt.json')
    if (frozen_selection.get('selection')!=selection or hash_json(selection)!=selection_sha256
            or frozen_selection.get('selection_sha256')!=selection_sha256
            or selection_receipt.get('selection_sha256')!=selection_sha256
            or selection_receipt.get('files',{}).get('publication-selection.json')!=sha256(audit/'publication-selection.json')):
        raise ValueError('Visualization selection receipt mismatch')
    for selected in selection['rollouts']:
        for path,digest in selected.get('source_files',{}).items():
            if sha256(Path(path))!=digest: raise ValueError('Visualization selection source checksum mismatch')
            sources[path]=digest
    for relative,digest in {**repair_receipt['files'],**repair_receipt['session_receipts']}.items():
        path=owned_path(nas,relative)
        if sha256(path)!=digest: raise ValueError('Repair manifest checksum mismatch')
        sources[str(path)]=digest
    for path,digest in read_json(repair/'source-proofs.json').items():
        if sha256(Path(path))!=digest: raise ValueError('Visualization repair source checksum mismatch')
        sources[path]=digest
    catalog = RunEvidence(root,run_id).read()
    canonical,truth,branches,decision_views,exclusions,scopes = [],[],[],[],[],[]
    report=read_json(repair/'report.json')
    exclusions.extend(report.get('input_issues',[]))
    for selected_row in selection['rollouts']:
        checkpoint()
        rid,sid=selected_row['rollout_id'],selected_row['session_id']
        row=catalog['rollouts'][rid]
        # Dependencies are collected and hashed once below, including shared shards.
        session=read_repaired_session(nas,repair_id,sid,verify_dependencies=False)
        for relative,digest in {**session['receipt']['files'],**session['receipt']['dependencies']}.items():
            sources[str(owned_path(nas,relative))]=digest
        proofs={}
        _,_,task,responses=_canonical(root,run_id,sid,[(rid,row)],catalog,proofs)
        sources.update(proofs)
        decisions,proofs,receipt_bound=decision_inventory([nas],sid,rid,row['trajectory_sha256'])
        sources.update(proofs)
        selected_proofs=validate_canonical_decisions(responses,decisions)
        native=[]
        for relative,digest in session['receipt']['dependencies'].items():
            path=owned_path(nas,relative)
            if path.parent.name=='requests' and path.suffix=='.json':
                native.append(read_request(nas,dict(ledger_ref=relative,ledger_sha256=digest)))
        bound,_,unbound,_=_bindings(sid,task,run_id,responses,native)
        unavailable_messages={entry['message_index'] for entry in session['coverage'].get('unbound_responses',[])}
        bound={key:value for key,value in bound.items() if value['message_index'] not in unavailable_messages}
        trajectory=read_json(owned_path(root,row['trajectory_path']))
        actions=read_actions(root,dict(row,rollout_id=rid))
        steps=[action['message_index'] for action in actions]
        message_steps={index:step for step,index in enumerate(steps)}
        suffix,cutoff,unknown=native_complete_suffix(bound,responses,steps)
        requests=[]
        for binding in suffix.values():
            req=copy.deepcopy(binding['request'])
            req.update(message_index=binding['message_index'],step_id=binding['message_index'],
                       is_action_response=binding['message_index'] in message_steps)
            req['blocks']=reasoning_blocks(req['prompt_token_ids'],req['output_token_ids'],
                                           req.get('open_token_ids',[248068]),req['close_token_ids'])
            requests.append(req)
        if requests: annotate_remaining_workloads({rid:requests})
        by_message={req['message_index']:req for req in requests}
        final={'session_id':sid,'rollout_id':rid,'trajectory_sha256':row['trajectory_sha256'],
               'model_fingerprint':row['model_fingerprint'],'accepted_native_request_ids':list(bound),
               'responses':{key:dict(message_index=value['message_index'],
                   is_action_response=value['message_index'] in message_steps,
                   agent_step_index=message_steps.get(value['message_index'])) for key,value in bound.items()}}
        final['decision_mappings']={key:dict(message_index=proof['message_index'],
            is_action_response=proof['message_index'] in message_steps,
            agent_step_index=message_steps.get(proof['message_index']),
            binding='canonical_decision_identity',native_available=key in bound)
            for key,proof in selected_proofs.items()}
        for step,index in enumerate(steps):
            req=by_message.get(index)
            blocks=[b for b in req['blocks'] if b.get('close') is not None] if req else []
            available=len(blocks)==1 and not blocks[0]['remaining_censored']
            truth.append(dict(rollout_id=rid,session_id=sid,step_id=step,message_index=index,
                remaining_steps=len(steps)-step,token_truth_available=available,
                remaining_reasoning_tokens=blocks[0]['future_reasoning_tokens'] if available else None,
                unknown_response_indices=unknown,last_unknown_response_index=cutoff,
                truth_endpoint='observed_agent_terminal',censored=not available))
        for event in session['points']:
            recovery=event['recovery']
            binding=recovery['binding']
            if binding not in {'native_exact','native_exact_candidate'}:
                exclusions.append(dict(rollout_id=rid,sample_id=event['sample_id'],reason='not_native_exact',binding=binding))
                continue
            point=copy.deepcopy(event)
            point.update(rollout_id=rid,session_id=sid,hidden_root=str(nas),
                         eligible_for_training=False,purpose='visualization')
            point['recovery']['eligible_for_training']=False
            index=recovery.get('canonical_message_index')
            if binding=='native_exact':
                point.update(message_index=index,step_id=message_steps.get(index),actual_path=True,
                             prediction_binding='native_exact_selected' if recovery.get('decision_id') else 'native_exact')
                canonical.append(point)
            else:
                point.update(actual_path=False,prediction_binding='native_exact_candidate')
            if recovery.get('decision_id'): branches.append(point)
        for decision in decisions.values():
            decision_views.append(dict(rollout_id=rid,session_id=sid,decision=decision,finalization=final,
                                       score_receipt_bound=decision['decision_id'] in receipt_bound))
        scopes.append(dict(rollout_id=rid,session_id=sid,source_capture_status=selected_row.get('capture_status'),**session['coverage']))
        records=report['sessions']
        exclusions.extend(dict(rollout_id=rid,**error) for record in records
                          if record.get('session_id')==sid for error in record.get('errors',[]))
    counts=Counter((point['rollout_id'],point['step_id']) for point in canonical if point['step_id'] is not None)
    ambiguous={key for key,count in counts.items() if count>1}
    if ambiguous:
        for point in canonical:
            if (point['rollout_id'],point['step_id']) in ambiguous:
                exclusions.append(dict(rollout_id=point['rollout_id'],sample_id=point['sample_id'],
                                       reason='ambiguous_canonical_action_close'))
        canonical=[point for point in canonical if (point['rollout_id'],point['step_id']) not in ambiguous]
        for scope in scopes:
            if any(rid==scope['rollout_id'] for rid,_ in ambiguous): scope['canonical_coverage']['status']='partial'
    canonical_complete=bool(scopes) and all(s['canonical_coverage']['status']=='complete' for s in scopes)
    all_complete=bool(scopes) and all(s['all_candidate_coverage']['status']=='complete' for s in scopes)
    reasons=Counter(branch['close_capture'].get('reason') or 'available'
                    for scope in scopes for branch in scope.get('branches',[]))
    score_available=sum(entry['score_receipt_bound'] and type(candidate.get('score')) in (int,float)
        and math.isfinite(candidate['score'])
        for entry in decision_views for candidate in entry['decision']['candidates'])
    candidate_count=sum(len(entry['decision']['candidates']) for entry in decision_views)
    coverage=dict(canonical_coverage=dict(status='complete' if canonical_complete else 'partial',
        expected_known=sum(s['expected_known'] for s in scopes),observed=sum(s['observed_native_exact'] for s in scopes),
        missing_count=sum(len(s['missing_known']) for s in scopes),
        expected_count_complete=bool(scopes) and all(s['expected_count_complete'] for s in scopes)),
        all_candidate_coverage=dict(status='complete' if all_complete else 'partial',candidate_count=candidate_count,
        expected_count_complete=bool(scopes) and all(s['all_candidate_coverage']['expected_count_complete'] for s in scopes),
        expected=sum(s['all_candidate_coverage']['expected'] for s in scopes),
        observed=sum(s['all_candidate_coverage']['observed'] for s in scopes),
        missing_count=sum(len(s['all_candidate_coverage']['missing']) for s in scopes),
        close_reason_counts=dict(reasons),stored_score_available=score_available,
        stored_score_missing=candidate_count-score_available),sessions=scopes)
    summary=dict(**signature,publication_state='published',eligible_for_training=False,
        coverage_status='complete' if canonical_complete and all_complete and not exclusions else 'partial',
        published_points=len(canonical),branch_points=len(branches),truth_available=sum(t['token_truth_available'] for t in truth),
        truth_censored=sum(not t['token_truth_available'] for t in truth),excluded_rollouts=0,
        excluded_points=len(exclusions),path=str(output),created_at=now(),**coverage)
    stage=output.parent/('.visualization-'+publication_id)
    if stage.exists(): stage.rename(stage.with_name(stage.name+'.interrupted-'+uuid4().hex))
    payloads={'publication':summary,'canonical-point-index':canonical,'canonical-truth':truth,
              'branch-point-index':branches,'decisions':decision_views,'coverage':coverage,
              'excluded':selection['excluded'],'point-exclusions':exclusions}
    for name,value in payloads.items(): write_json(stage/(name+'.json'),value)
    # Check immutable sources once more before the atomic completion marker.
    for path,digest in sources.items():
        if sha256(Path(path))!=digest: raise ValueError('Visualization source changed during publication')
    write_json(stage/'receipt.json',dict(schema_version=1,purpose='visualization',eligible_for_training=False,
        publication_signature=signature,publication_sources=sources,
        files={name+'.json':sha256(stage/(name+'.json')) for name in payloads}))
    stage.rename(output)
    return summary

"""Publish sealed successful sessions independently of other run workers."""
from pathlib import Path
from core.io import hash_json, identifier, owned_path, read_json, sha256, write_json
from core.locking import file_lock
from application.execution.evidence import RunEvidence
from enrichment.online_trace import finalize_token_events, publication_directory
from hidden.artifacts import read_request, response_records, event_records


def _check_finalization(completion, rollout_id, coverage, coverage_scope):
    if coverage_scope not in {'all_branches', 'canonical_native'}:
        raise ValueError('Unknown capture coverage scope')
    if completion.get('rollout_id') != rollout_id:
        raise ValueError('Capture finalization rollout identity differs')
    if completion.get('capture_status') == 'complete':
        return
    if (coverage_scope == 'canonical_native' and completion.get('capture_status') == 'partial'
            and type(coverage.get('expected')) is int and coverage['expected'] > 0
            and coverage.get('observed') == coverage['expected'] and coverage.get('missing') == []):
        return
    raise ValueError('Attempt has no successful immutable capture finalization for requested scope')


def publish_finalized(cfg,run_id,publication_id,session_ids,*,eligibility="resolved_only",coverage_scope="all_branches"):
    from difficulty.datasets.eligibility import eligible_evaluation, validate_policy
    validate_policy(eligibility)
    if coverage_scope not in {'all_branches', 'canonical_native'}:
        raise ValueError('Unknown capture coverage scope')
    from core.config import load_snapshot
    root=Path(cfg['project_root']).resolve();frozen=load_snapshot(root,run_id)
    publication_id=identifier(publication_id);sessions=sorted({identifier(s) for s in session_ids})
    if not sessions:raise ValueError('Select explicit finalized sessions')
    nas=Path(frozen['paths']['nas_root'])/frozen['capture']['export']['root']/run_id
    output=publication_directory(root,run_id,publication_id)
    signature={'run_id':run_id,'publication_id':publication_id,'session_ids':sessions,
               'config_hash':frozen['config_hash'],'policy':('sealed_resolved_sessions_v1' if eligibility == 'resolved_only' else 'sealed_positive_reward_sessions_v1')}
    if coverage_scope != 'all_branches':
        signature['coverage_scope'] = coverage_scope
    with file_lock(root/'state/capture-trace-locks'/run_id/(publication_id+'.lock')):
        if output.exists():
            receipt=read_json(output/'receipt.json')
            if receipt.get('publication_signature')!=signature:raise ValueError('Trace publication identity differs')
            for name,digest in receipt['files'].items():
                if sha256(output/name)!=digest:raise ValueError('Published trace changed')
            return read_json(output/'publication.json')
        evidence=RunEvidence(root,run_id);evidence.verify();data=evidence.read()
        requests=[];events=[];paths={};finals={};proofs={};scope={}
        for sid in sessions:
            base=nas/'sessions'/sid;receipt=read_json(base/'receipt.json')
            coverage=read_json(base/'coverage.json');final=read_json(base/'finalization.json')
            if receipt.get('status') not in {'stopped','complete'} or coverage.get('missing'):
                raise ValueError('Session is not stopped with complete native coverage: '+sid)
            rid=final['rollout_id'];row=data['rollouts'][rid]
            if not eligible_evaluation(row, data, rid, eligibility):
                raise ValueError('Session does not satisfy official evaluation eligibility: '+eligibility)
            attempt=owned_path(root,row['provenance']['source']).parent.parent
            final_receipt=attempt/'capture-finalization.json';completion=read_json(final_receipt)
            _check_finalization(completion, rid, coverage, coverage_scope)
            proofs[str(final_receipt)]=sha256(final_receipt)
            trajectory=owned_path(root,row['trajectory_path'])
            if sha256(trajectory)!=final['trajectory_sha256']:raise ValueError('Session/canonical identity differs')
            native=[]
            for _,_,response in response_records(read_json(trajectory)):
                req=read_request(nas,response)
                if req['session_id']!=sid:raise ValueError('Cross-session native request')
                native.append(req['request_id'])
                requests.append({k:response[k] for k in ['ledger_ref','ledger_sha256']})
            if native!=final['accepted_native_request_ids']:raise ValueError('Native request order differs')
            for path,event in event_records(nas):
                if event.get('session_id')==sid and event['request_id'] in set(native):events.append((path,event))
            finals[sid]=str((base/'finalization.json').relative_to(nas));paths[rid]=str(trajectory)
            evaluation=data['evaluations'][row['selected_evaluation_id']]
            scope[rid]={'rollout':row,'evaluation':evaluation}
            for p in [base/'receipt.json',base/'coverage.json',base/'finalization.json',
                      trajectory,owned_path(root,evaluation['report_path'])]:proofs[str(p)]=sha256(p)
        inventory=read_json(nas/'inventory.json')
        stage=output.parent/('.sealed-'+publication_id)
        if stage.exists():raise FileExistsError('Inspect incomplete publication stage: '+str(stage))
        finalize_token_events(nas,stage,open_ids=inventory['open_token_ids'],close_ids=inventory['close_token_ids'],
            request_evidence=requests,event_rows=events,trajectory_paths=paths,finalization_paths=finals)
        after=evidence.read()
        for rid,source in scope.items():
            row=after['rollouts'][rid]
            if source!={'rollout':row,'evaluation':after['evaluations'][row['selected_evaluation_id']]}:
                raise ValueError('Selected source evidence changed during publication')
        summary={**signature,'source_rollout_ids':sorted(scope),'native_requests':len(requests),
                 'published_points':len(events),'scope_evidence_sha256':hash_json(scope),'whole_run_acceptance':False}
        write_json(stage/'publication.json',summary)
        receipt=read_json(stage/'receipt.json');receipt.update(publication_signature=signature,publication_sources=proofs)
        receipt['files']['publication.json']=sha256(stage/'publication.json');write_json(stage/'receipt.json',receipt)
        stage.rename(output)
        return {**summary,'path':str(output)}

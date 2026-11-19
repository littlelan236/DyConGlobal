"""Verify run-local receipts, canonical bytes and external capture references."""
from pathlib import Path
from application.execution.evidence import RunEvidence
from core.io import artifact_reference, hash_json, owned_path, read_json, sha256, write_json
from .run_plan import load_plan
from application.execution.subject import artifact_kinds


def capture_publication(root, run_id):
    loaded=load_plan(root,run_id); run=loaded['run'];cfg=loaded['config']
    nas=Path(cfg['paths']['nas_root'])/cfg['capture']['export']['root']/run_id
    sessions=[]
    if not nas.exists():
        return {'run_id':run_id,'nas_locator':str(nas),'sessions':[],'state':'absent'}
    for path in sorted((nas/'sessions').glob('*/finalization.json')):
        final=read_json(path)
        if final.get('run_id',run_id)!=run_id:raise ValueError('Capture belongs to another run')
        refs={}
        for name in ['session.json','finalization.json','coverage.json','receipt.json',
                     'parallel/branch-index.json','parallel/branch-coverage.json','parallel/branch-receipt.json']:
            source=path.parent/name
            if source.exists():
                refs[name]={'nas_locator':str(source),'sha256':sha256(source)}
        sessions.append({'session_id':path.parent.name,'rollout_id':final.get('rollout_id'),
                         'trajectory_sha256':final.get('trajectory_sha256'),'files':refs})
    value={'schema_version':1,'run_id':run_id,'nas_locator':str(nas),'sessions':sessions,
           'state':'published' if sessions else 'no_finalized_sessions'}
    write_json(run/'hidden/publication.json',value)
    return value


def refresh(root,run_id):
    loaded=load_plan(root,run_id);run=loaded['run'];rows=[]
    evidence = RunEvidence(root, run_id)
    if evidence.path.exists():
        rows.append({'kind':'run-evidence', 'artifact_id':'evidence',
                     'local_locator':artifact_reference(root,evidence.path),
                     'content_sha256':sha256(evidence.path)})
    for path in sorted(run.rglob('*receipt.json')):
        if path.is_file():
            rows.append({'kind':'receipt','artifact_id':str(path.relative_to(run)),
                         'local_locator':artifact_reference(root,path), 'content_sha256':sha256(path)})
    data=RunEvidence(root,run_id).read()
    for rid,row in data['rollouts'].items():
        if row['run_id']==run_id:
            for kind in artifact_kinds(row):
                rows.append({'kind':kind,'artifact_id':rid+':'+kind,
                             'local_locator':row[kind+'_path'],'content_sha256':row[kind+'_sha256']})
    write_json(run/'artifacts.json',{'schema_version':1,'run_id':run_id,'artifacts':rows})
    return rows


def show(root,run_id):
    loaded=load_plan(root,run_id);run=loaded['run']
    return {'run_id':run_id,'plan':loaded['plan'], 'files':[
        {'path':str(p.relative_to(run)), 'bytes':p.stat().st_size} for p in sorted(run.rglob('*'))
        if p.is_file() and not p.name.startswith('.write-')]}


def verify(root,run_id):
    loaded=load_plan(root,run_id);run=loaded['run'];checked=0;receipts=0;external=0
    digests = {}
    def cached_digest(path):
        key = str(Path(path).resolve())
        if key not in digests:
            digests[key] = sha256(path)
        return digests[key]
    RunEvidence(root,run_id).verify()
    from rollout.consensus import consensus_options, validate_consensus_evidence

    for row in RunEvidence(root, run_id).read()['rollouts'].values():
        if row.get('run_id') != run_id:
            continue
        trajectory = read_json(owned_path(root, row['trajectory_path']))
        for message in trajectory.get('messages', []):
            summary = (message.get('extra') or {}).get('dycon_parallel')
            if not isinstance(summary, dict):
                continue
            if (summary.get('selection_mode') == 'parallel_enthophy' or
                    'representative_policy' in (summary.get('consensus_config') or {})):
                validate_consensus_evidence(summary, consensus_options(loaded['config']['parallel']))
                proof = loaded['config'].get('parallel_policy')
                if proof and (summary.get('policy_binding') != {
                        key: proof[key] for key in ('input_sha256', 'effective_sha256')}):
                    raise ValueError('Parallel decision policy binding differs from frozen run')
                checked += 1
    for row in read_json(run/'artifacts.json').get('artifacts',[]):
        if sha256(owned_path(root,row['local_locator']))!=row['content_sha256']:
            raise ValueError('Artifact checksum mismatch: '+row['artifact_id'])
        checked+=1
    for path in sorted(run.rglob('*receipt.json')):
        value=read_json(path)
        if not isinstance(value,dict):raise ValueError('Invalid receipt '+str(path))
        files_root, dependencies_root = receipt_roots(run, path, value, loaded['config'])
        for relative,digest in value.get('files',{}).items():
            if sha256(owned_path(files_root,relative))!=digest:
                raise ValueError('Receipt checksum mismatch: '+str(path)+' '+relative)
            checked+=1
        if dependencies_root is not None:
            for relative, expected in value.get('session_receipts', {}).items():
                if sha256(owned_path(files_root, relative)) != expected:
                    raise ValueError('Repair session receipt checksum mismatch: ' + relative)
                checked += 1
            for relative, expected in value.get('dependencies', {}).items():
                if cached_digest(owned_path(dependencies_root, relative)) != expected:
                    raise ValueError('Repair dependency checksum mismatch: ' + relative)
                external += 1
        if value.get('source_root') and value.get('source_files'):
            for relative,digest in value['source_files'].items():
                if sha256(owned_path(value['source_root'],relative))!=digest:
                    raise ValueError('External source checksum mismatch: '+relative)
                external+=1
        receipts+=1
    publication=run/'hidden/publication.json'
    if publication.exists():
        for session in read_json(publication).get('sessions',[]):
            for row in session['files'].values():
                if sha256(row['nas_locator'])!=row['sha256']:
                    raise ValueError('NAS metadata checksum mismatch')
                external+=1
    # Frozen harness selection is also an input artifact for historical verification runs.
    selection_path=run/'harness/selection.json'
    if selection_path.exists():
        selection=read_json(selection_path)
        proof=read_json(run/'harness/selection-receipt.json')
        if hash_json(selection)!=proof['selection_sha256']:raise ValueError('Harness selection changed')
        for row in selection['rollouts'].values():
            if row.get('evaluation_subject', {}).get('kind') == 'web_trace':
                from application.execution.subject import verify_subject
                verify_subject(root, row)
            for kind in artifact_kinds(row):
                if sha256(owned_path(root,row[kind+'_path']))!=row[kind+'_sha256']:
                    raise ValueError('Selected canonical artifact changed')
                checked+=1
    return {'run_id':run_id,'verified':True,'artifacts_checked':checked,'receipts_checked':receipts,
            'external_metadata_checked':external}


def receipt_roots(run, path, value, cfg):
    """Repair payload receipts use capture-run-relative paths, not receipt.parent.

    Infer legacy scope only from its exact owned staging namespace and identity;
    never select a root by trying whichever path happens to exist.
    """
    relative = path.relative_to(run)
    parts = relative.parts
    if (len(parts) >= 7 and parts[:2] == ('hidden', 'capture-repairs')
            and parts[3:5] == ('payload', 'capture-repairs')):
        repair_id = parts[2]
        if (parts[5] != repair_id or value.get('repair_id') != repair_id
                or value.get('run_id') != run.name):
            raise ValueError('Repair receipt namespace identity mismatch')
        payload = run.joinpath(*parts[:4])
        nas = Path(cfg['paths']['nas_root']) / cfg['capture']['export']['root'] / run.name
        return payload, nas
    return path.parent, None

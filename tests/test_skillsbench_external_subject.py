import pytest
from core.config import snapshot
from core.io import hash_json, owned_path, sha256, write_json
from application.execution.subject import verify_subject


def test_external_subject_is_bound_to_frozen_storage_root(tmp_path):
    project=tmp_path/'project'; project.mkdir()
    physical=tmp_path/'data1'
    cfg={'project_root':str(project),'paths':{'artifact_roots':{'run':str(physical)},
         'artifact_scan_roots':[str(project),str(physical)]}}
    cfg['config_hash']=hash_json(cfg)
    run=snapshot(cfg,'external',{'tasks':{}})
    output=run/'subject'/'document.docx'; output.parent.mkdir(); output.write_bytes(b'binary\x00\xff')
    subject=run/'subject'/'subject.json'
    write_json(subject,{'schema_version':1,'kind':'workspace','outputs':[
        {'path':output.name,'sha256':sha256(output)}]})
    trajectory=run/'trajectory.json'; write_json(trajectory,{'messages':[]})
    row={'evaluation_subject':{'kind':'workspace'},'subject_path':str(subject),
         'subject_sha256':sha256(subject),'trajectory_path':str(trajectory),
         'trajectory_sha256':sha256(trajectory)}
    assert verify_subject(project,row)
    output.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='Changed subject output'):verify_subject(project,row)
    snap=run/'resolved-config.json'
    cfg=__import__('json').loads(snap.read_text());cfg['run_storage_root']=str(tmp_path/'other')
    write_json(snap,cfg)
    with pytest.raises(ValueError,match='escapes project'):owned_path(project,subject)


def test_arbitrary_external_symlink_is_not_an_artifact_root(tmp_path):
    project=tmp_path/'project';project.mkdir()
    other=tmp_path/'other';other.mkdir();(other/'secret').write_text('private')
    (project/'runs').mkdir();(project/'runs'/'fake').symlink_to(other,target_is_directory=True)
    with pytest.raises(ValueError,match='escapes project'):owned_path(project,'runs/fake/secret')


def test_external_rollout_import_and_live_evaluation_preserve_binary_subject(tmp_path):
    from catalog.skillsbench import package_manifest
    from catalog.store import Task
    from rollout.artifacts import import_rollout
    from application.execution.evidence import RunEvidence
    project=tmp_path/'project';project.mkdir()
    cfg={'project_root':str(project),'paths':{'artifact_roots':{'run':str(tmp_path/'data1')}}}
    cfg['config_hash']=hash_json(cfg)
    run=snapshot(cfg,'external',{'tasks':{}})
    package=project/'task';package.mkdir();(package/'task.md').write_text('official')
    files=package_manifest(package)
    task=Task.model_validate({'benchmark':'skillsbench','revision':'a'*40,
        'instance_id':'task','partition':'train','instance':{
        'instance_id':'task','task_path':str(package),'files':files,
        'package_sha256':hash_json(files),'metadata':{},'images':[]}}).model_dump()
    source=run/'input.json';write_json(source,{'instance_id':'task','messages':[],
                                             'info':{'exit_status':'AgentCompleted'}})
    output=run/'subject'/'binary.docx';output.parent.mkdir();output.write_bytes(b'PK\x00\xff')
    subject=output.parent/'subject.json'
    write_json(subject,{'schema_version':1,'kind':'workspace','outputs':[
        {'path':'binary.docx','sha256':sha256(output)}]})
    report=run/'report.json';write_json(report,{'reward':1})
    rid=import_rollout(project,{'source':str(source),'task':task,'run_id':'external','attempt_id':'a',
        'subject':str(subject),'subject_sha256':sha256(subject),'agent':'openhands','model_fingerprint':'model',
        'evaluation_id':'e1','evaluation_report':str(report),'evaluation':{
        'status':'completed','resolved':True,'success_policy':'binary_v1'}})
    evidence=RunEvidence(project,'external')
    assert evidence.verify()['verified']
    row=evidence.read()['rollouts'][rid]
    assert row['resolved'] is True and row['subject_path'].startswith('runs/external/')
    assert 'patch_path' not in row

import copy
from pathlib import Path
import pytest
from catalog.skillsbench import package_manifest,verify_package
from catalog import verifier_repairs
from core.io import hash_json,read_json,sha256,write_json


@pytest.fixture
def source(tmp_path,monkeypatch):
    task_root=tmp_path/'upstream/task';(task_root/'verifier').mkdir(parents=True)
    (task_root/'verifier/test.sh').write_text('original test\n')
    (task_root/'task.md').write_text('Original prompt\n')
    files=package_manifest(task_root)
    task={'benchmark':'skillsbench','revision':'a'*40,'instance_id':'task','partition':'train','instance':{
        'instance_id':'task','task_path':str(task_root),'files':files,'package_sha256':hash_json(files),
        'images':[],'metadata':{}}}
    snapshot=tmp_path/'source.json';write_json(snapshot,{'benchmark':'skillsbench','tasks':{'key':task}})
    patch=tmp_path/'patches/test.sh';patch.parent.mkdir();patch.write_text('corrected test\n')
    rules=patch.with_name('rules.json');write_json(rules,{'tasks':{'task':{'source_package_sha256':hash_json(files),
        'files':{'verifier/test.sh':{'source_sha256':sha256(task_root/'verifier/test.sh'),
            'patch':'test.sh','effective_sha256':sha256(patch)}}}}})
    monkeypatch.setattr(verifier_repairs,'RULES',rules)
    write_json(tmp_path/'runtime/agents/verifier_contracts.json',{'schema_version':1,'contracts':[
        {'instance_id':'task','package_sha256':hash_json(files),'files':{'task.md':sha256(task_root/'task.md'),
         'verifier/test.sh':sha256(task_root/'verifier/test.sh')},'score_mode':'pytest_exit_binary','capabilities':[]}]})
    return tmp_path,snapshot,task


def test_formal_publication_and_contract_registration_preserve_source(source):
    root,snapshot,original=source;before=sha256(snapshot)
    result=verifier_repairs.publish_variant({'project_root':str(root)},'fixed',snapshot)
    task=read_json(root/result['snapshot_ref'])['tasks']['key']
    verify_package(original);verify_package(task)
    # Formal run creation assigns train/test membership independently of content.
    frozen=copy.deepcopy(task);frozen['partition']='test'
    verify_package(frozen)
    frozen['revision']='b'*40
    with pytest.raises(ValueError):verify_package(frozen)
    assert before==sha256(snapshot)
    assert task['instance']['package_sha256']!=original['instance']['package_sha256']
    contracts=read_json(root/'runtime/agents/verifier_contracts.json')['contracts']
    assert {r['package_sha256'] for r in contracts}=={original['instance']['package_sha256'],task['instance']['package_sha256']}
    with pytest.raises(FileExistsError):verifier_repairs.publish_variant({'project_root':str(root)},'fixed',snapshot)


def test_task_metadata_and_patch_tampering_are_rejected(source):
    root,snapshot,_=source;result=verifier_repairs.publish_variant({'project_root':str(root)},'fixed',snapshot)
    task=read_json(root/result['snapshot_ref'])['tasks']['key'];changed=copy.deepcopy(task)
    changed['instance']['metadata']['relaxed']=True
    with pytest.raises(ValueError):verify_package(changed)
    (Path(task['instance']['task_path'])/'verifier/test.sh').write_text('unapproved')
    with pytest.raises(ValueError):verify_package(task)


def test_changed_upstream_fails_before_publication(source):
    root,snapshot,original=source
    (Path(original['instance']['task_path'])/'task.md').write_text('changed')
    with pytest.raises(ValueError):verifier_repairs.publish_variant({'project_root':str(root)},'refused',snapshot)
    assert not (root/'catalog/task-variants/refused').exists()


def test_published_variant_uses_immutable_historical_rules(source):
    root,snapshot,_=source
    result=verifier_repairs.publish_variant({'project_root':str(root)},'fixed',snapshot)
    task=read_json(root/result['snapshot_ref'])['tasks']['key']
    rules=verifier_repairs.RULES
    archive=rules.parent/'history'/(sha256(rules)+'.json')
    archive.parent.mkdir();archive.write_bytes(rules.read_bytes())
    rule=read_json(rules);rule['tasks']['task']['files']['verifier/test.sh']['effective_sha256']='f'*64
    write_json(rules,rule)
    (rules.parent/'test.sh').write_text('next proposed repair')
    verify_package(task)
    archive.write_text('{}')
    with pytest.raises(ValueError):verify_package(task)

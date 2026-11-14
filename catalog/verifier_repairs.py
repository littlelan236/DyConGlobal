"""Exact content-bound task repairs; original task packages remain immutable."""
from __future__ import annotations

import copy
from pathlib import Path
import shutil

from core.io import hash_json,identifier,read_json,sha256,write_json

POLICY='skillsbench-verifier-repair-v1'
RULES=Path(__file__).parent/'verifier_patches/repairs.json'


def approved_files(task, rules_path=None):
    rules_path=RULES if rules_path is None else rules_path
    rule=read_json(rules_path)['tasks'].get(task['instance_id'])
    if rule is None or task['instance']['package_sha256']!=rule['source_package_sha256']:
        raise ValueError('Verifier repair requires the exact approved source package')
    result={}
    root=rules_path.parent
    for name,row in rule['files'].items():
        source=Path(task['instance']['task_path'])/name
        patch=(root/row['patch']).resolve()
        if (not patch.is_relative_to(root.resolve()) or patch.is_symlink()
                or (sha256(source) if source.exists() else None)!=row['source_sha256'] or sha256(patch)!=row['effective_sha256']):
            raise ValueError('Verifier repair source or approved patch changed: '+name)
        result[name]=patch
    return result


def verify_adaptation(task):
    from catalog.skillsbench import verify_package
    payload=task['instance'];adapt=payload['verifier_adaptation'];path=Path(adapt['receipt_path'])
    if sha256(path)!=adapt['receipt_sha256']:
        raise ValueError('Verifier repair receipt changed')
    receipt=read_json(path);original=receipt['upstream_task']
    if (adapt['policy']!=POLICY or receipt['policy']!=POLICY
            or original['instance'].get('verifier_adaptation')
            or any(task.get(k)!=original.get(k) for k in set(task)|set(original)
                   if k not in {'instance','partition'})):
        raise ValueError('Invalid verifier repair identity')
    verify_package(original)
    rules_path=RULES
    if sha256(rules_path)!=receipt['rules_sha256']:
        rules_path=RULES.parent/'history'/(receipt['rules_sha256']+'.json')
    if sha256(rules_path)!=receipt['rules_sha256']:
        raise ValueError('Historical verifier repair rules changed')
    rule=read_json(rules_path)['tasks'][original['instance_id']]
    if rule['source_package_sha256']!=original['instance']['package_sha256']:
        raise ValueError('Historical verifier repair source identity changed')
    expected=copy.deepcopy(original['instance']);expected.pop('verifier_index_adaptation',None)
    for name,row in rule['files'].items():
        member=Path(payload['task_path'])/name
        if (sha256(member)!=row['effective_sha256']
                or (sha256(Path(original['instance']['task_path'])/name) if row['source_sha256'] else None)!=row['source_sha256']):
            raise ValueError('Published verifier differs from its approved repair')
        if row['source_sha256'] is None:
            expected['files'].append({'path':name,'sha256':row['effective_sha256'],'mode':row['mode']})
        else:
            next(item for item in expected['files'] if item['path']==name)['sha256']=row['effective_sha256']
    expected['files'].sort(key=lambda item:Path(item['path']))
    expected.update(task_path=payload['task_path'],package_sha256=hash_json(expected['files']),verifier_adaptation=adapt)
    if payload!=expected or receipt['effective_task_path']!=payload['task_path']:
        raise ValueError('Verifier repair modified content outside the approved delta')


def publish_variant(cfg,snapshot_id,source_ref,*,rules_path=None):
    rules_path=RULES if rules_path is None else rules_path
    from catalog.skillsbench import package_manifest,verify_package
    root=Path(cfg['project_root']).resolve();source_path=(root/source_ref).resolve()
    if not source_path.is_relative_to(root):raise ValueError('Source snapshot must be project-owned')
    source_hash=sha256(source_path);source=read_json(source_path)
    if source.get('benchmark')!='skillsbench':raise ValueError('Expected SkillsBench source snapshot')
    rules=read_json(rules_path)['tasks'];selected=[]
    source=copy.deepcopy(source)
    if rules_path!=RULES:
        # Revise only the selected task, deriving directly from its authenticated
        # original package so historical regrading retains the same ancestry.
        for task in source['tasks'].values():
            if task['instance_id'] in rules and task['instance'].get('verifier_adaptation'):
                verify_package(task)
                receipt=read_json(task['instance']['verifier_adaptation']['receipt_path'])
                original=receipt['upstream_task']
                task.clear();task.update(copy.deepcopy(original))
    for key,task in source['tasks'].items():
        verify_package(task)
        if task['instance_id'] in rules:
            selected.append((key,approved_files(task,rules_path)))
    if len(selected)!=len(rules):raise ValueError('Source does not contain the complete approved repair scope')
    publication=root/'catalog/task-variants'/identifier(snapshot_id)
    target=root/'catalog/snapshots'/snapshot_id/'manifest.json'
    if publication.exists() or target.parent.exists():raise FileExistsError('Verifier variant ID is immutable')
    publication.mkdir(parents=True,exist_ok=False)
    if rules_path!=RULES:
        history=RULES.parent/'history'/(sha256(rules_path)+'.json')
        history.parent.mkdir(exist_ok=True)
        if history.exists() and sha256(history)!=sha256(rules_path):raise ValueError('Rule history collision')
        if not history.exists():history.write_bytes(rules_path.read_bytes())
    result=copy.deepcopy(source)
    registry_path=root/'runtime/agents/verifier_contracts.json';registry=read_json(registry_path)
    registered={r['package_sha256']:r for r in registry['contracts']}
    new_contracts=[]
    for key,patches in selected:
        original=source['tasks'][key];task=result['tasks'][key];iid=task['instance_id']
        effective=publication/'tasks'/iid
        shutil.copytree(original['instance']['task_path'],effective)
        for name,patch in patches.items():
            member=effective/name;member.parent.mkdir(parents=True,exist_ok=True)
            member.write_bytes(patch.read_bytes())
            if rules[iid]['files'][name]['source_sha256'] is None:member.chmod(rules[iid]['files'][name]['mode'])
        files=package_manifest(effective);receipt_path=publication/'receipts'/(iid+'.json')
        write_json(receipt_path,{'schema_version':1,'policy':POLICY,'upstream_task':original,
            'source_snapshot':{'path':str(source_path),'sha256':source_hash},
            'effective_task_path':str(effective),'rules_sha256':sha256(rules_path)})
        task['instance'].pop('verifier_index_adaptation',None)
        task['instance'].update(task_path=str(effective),files=files,package_sha256=hash_json(files),
            verifier_adaptation={'policy':POLICY,'receipt_path':str(receipt_path),'receipt_sha256':sha256(receipt_path)})
        verify_package(task)
        old=registered[original['instance']['package_sha256']]
        contract=copy.deepcopy(old);contract.update(package_sha256=task['instance']['package_sha256'],
            files={r['path']:r['sha256'] for r in files if r['path'] in old['files'] or r['path'] in rules[iid]['files'] or r['path']=='verifier/loss.npz'})
        if iid=='simpo-code-reproduction':contract['capabilities']=['simpo_independent_reference']
        contract.update(rules[iid].get('contract',{}))
        new_contracts.append(contract)
    if sha256(source_path)!=source_hash:raise ValueError('Source snapshot changed during publication')
    # Registry entries are append-only and keyed by full package content identity.
    for contract in new_contracts:
        previous=registered.get(contract['package_sha256'])
        if previous is not None and previous!=contract:raise ValueError('Frozen contract identity collision')
        if previous is None:registry['contracts'].append(contract)
    write_json(registry_path,registry)
    result['verifier_correction']={'policy':POLICY,'source_snapshot':str(source_path),'source_snapshot_sha256':source_hash}
    write_json(target,result)
    return {'snapshot_ref':str(target.relative_to(root)),'snapshot_sha256':sha256(target),
            'policy':POLICY,'tasks':len(result['tasks']),'adapted_tasks':len(selected),
            'registered_contracts':len(new_contracts)}

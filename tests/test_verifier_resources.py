import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from core.io import hash_json,sha256,write_json
from runtime.agents.dependencies import verify_snapshot
from runtime.agents.openhands import CLI_REVISION
from runtime.agents.verifier_dependencies import POLICY,validate_resources,verify_derivation


def manifest(root, files, **extra):
    root.mkdir(exist_ok=True)
    for name,content in files.items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
    result={'schema_version':1,'cli_revision':CLI_REVISION,
            'files':[{'path':name,'size':(root/name).stat().st_size,'sha256':sha256(root/name)} for name in sorted(files)],**extra}
    result['receipt_sha256']=hash_json(result);write_json(root/'manifest.json',result)
    return result


@pytest.fixture
def extension(tmp_path):
    parent=tmp_path/'parent';child=tmp_path/'child'
    old=manifest(parent,{'requirements.txt':'private-agent==1 --hash=sha256:unchanged\n'})
    files={'requirements.txt':(parent/'requirements.txt').read_text(),
           'verifier/runtimes/3.10/bin/python3.10':'interpreter',
           'verifier/environments/test/requirements.txt':'pytest==8.4.1\n',
           'verifier/environments/test/wheels/pytest.whl':'wheel'}
    info={'schema_version':1,'policy':POLICY,'parent_manifest_sha256':sha256(parent/'manifest.json'),
          'parent_receipt_sha256':old['receipt_sha256'],
          'runtimes':{'3.10':{'abi':'cpython-310','platform':'linux-x86_64',
              'executable':'verifier/runtimes/3.10/bin/python3.10'}},
          'environments':{'test':{'runtime':'3.10','lock':'verifier/environments/test/requirements.txt',
              'lock_sha256':__import__('hashlib').sha256(files['verifier/environments/test/requirements.txt'].encode()).hexdigest(),
              'network_install':False,'offline_install_verified':True,
              'wheels':['verifier/environments/test/wheels/pytest.whl']}},
          'tasks':{'task':{'environments':['test']}}}
    new=manifest(child,files,verifier_resources=info)
    return parent,child,old,new


def test_parent_agent_bytes_and_lock_remain_identical(extension):
    parent,child,old,new=extension
    proof=verify_derivation(parent,child)
    assert proof['inherited_files']==1 and proof['added_files']==3
    assert (parent/'requirements.txt').read_bytes()==(child/'requirements.txt').read_bytes()
    assert verify_snapshot(parent)==old


@pytest.mark.parametrize('damage',['abi','platform','runtime','lock','wheel','network','install'])
def test_invalid_consumer_binding_fails(extension,damage):
    _,_,_,new=extension;new=copy.deepcopy(new);r=new['verifier_resources'];e=r['environments']['test']
    if damage=='abi':r['runtimes']['3.10']['abi']='cpython-312'
    elif damage=='platform':r['runtimes']['3.10']['platform']='darwin-arm64'
    elif damage=='runtime':e['runtime']='3.7'
    elif damage=='lock':e['lock_sha256']='0'*64
    elif damage=='wheel':e['wheels']=['verifier/missing.whl']
    elif damage=='network':e['network_install']=True
    else:e['offline_install_verified']=False
    with pytest.raises(ValueError):validate_resources(new)


def test_corrupt_wheel_and_unregistered_partial_build_fail(extension):
    _,child,_,_=extension
    (child/'verifier/environments/test/wheels/pytest.whl').write_text('broken')
    with pytest.raises(ValueError,match='Changed dependency'):verify_snapshot(child)


def test_concurrent_readers_do_not_write_readonly_snapshot(extension):
    _,child,_,new=extension
    before={p.relative_to(child):sha256(p) for p in child.rglob('*') if p.is_file()}
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(v==new for v in pool.map(verify_snapshot,[child]*8))
    assert before=={p.relative_to(child):sha256(p) for p in child.rglob('*') if p.is_file()}


def test_parent_agent_modification_cannot_inherit_qualification(extension):
    parent,child,_,new=extension
    path=child/'requirements.txt';path.write_text('changed agent')
    row=next(r for r in new['files'] if r['path']=='requirements.txt')
    row.update(size=path.stat().st_size,sha256=sha256(path));new.pop('receipt_sha256');new['receipt_sha256']=hash_json(new)
    write_json(child/'manifest.json',new)
    with pytest.raises(ValueError,match='inherited agent'):verify_derivation(parent,child)


def test_bad_asset_is_quarantined_and_cannot_publish(extension,tmp_path):
    from runtime.agents.verifier_dependencies import build
    parent,_,_,_=extension;asset=tmp_path/'asset';asset.write_bytes(b'broken asset')
    spec=tmp_path/'input.json';write_json(spec,{'schema_version':1,'policy':POLICY,
        'runtimes':{},'environments':{},'tasks':{},'assets':[{'id':'asset','source':str(asset),'sha256':'0'*64}]})
    with pytest.raises(ValueError,match='Asset source hash'):
        build({'project_root':str(tmp_path)},'failed',parent,spec)
    assert not (tmp_path/'runtime/dependency-snapshots/failed').exists()
    assert (tmp_path/'temp/skillsbench-dependencies/failed/publication').exists()
    with pytest.raises(FileExistsError):build({'project_root':str(tmp_path)},'failed',parent,spec)

"""Verifier resources in the same immutable publication as the private agent runtime."""
from __future__ import annotations

import copy
import base64
import hashlib
import json
import os
import re
import shutil
import shlex
import subprocess
import tarfile
import tomllib
import urllib.request
import zipfile
from email.parser import BytesParser
from pathlib import Path

from core.io import hash_json, identifier, now, read_json, sha256, write_json

POLICY = 'skillsbench-verifier-resources-v1'


async def probe_resources(env, profile, task_id, output, ownership):
    """Exercise each bound ABI closure in a fresh sandbox-local cache and venv."""
    snapshot=Path(profile['dependency_snapshot'])
    manifest=read_json(snapshot/'manifest.json')
    if sha256(snapshot/'manifest.json')!=profile['dependency_snapshot_sha256']:
        raise ValueError('Verifier probe manifest changed')
    validate_resources(manifest)
    resources=manifest['verifier_resources'];binding=resources['tasks'].get(task_id)
    rows=[]
    for name in binding['environments'] if binding else []:
        item=resources['environments'][name]
        runtime=resources['runtimes'][item['runtime']]
        mounted=Path('/opt/dycon-wheels');target='/tmp/dycon-resource-probe-'+name
        prefix='/usr/bin/env -u UV_CONSTRAINT -u PIP_CONSTRAINT -u UV_BUILD_CONSTRAINT -u PIP_BUILD_CONSTRAINT '
        prefix+='UV_OFFLINE=1 UV_NO_INDEX=1 PIP_NO_INDEX=1 UV_PYTHON_DOWNLOADS=never '
        prefix+='PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR='+target+'/cache /opt/dycon-wheels/uv '
        command='set -eu\ntest ! -e '+target+'\n'
        command+=prefix+'venv --no-project --python '+shlex.quote(str(mounted/runtime['executable']))+' '+target+'/venv\n'
        command+=prefix+'pip sync --offline --no-index --require-hashes --python '+target+'/venv/bin/python --find-links '
        command+=shlex.quote(str(mounted/Path(item['lock']).parent/'wheels'))+' '+shlex.quote(str(mounted/item['lock']))+'\n'
        command+=target+'/venv/bin/python -B -c '+shlex.quote(item.get('probe') or 'import sys; print(sys.version)')+'\n'
        result=await env.exec(command,user='root',timeout_sec=600)
        rows.append({'environment':name,'runtime':item['runtime'],'lock_sha256':item['lock_sha256'],
            'command':command,'return_code':result.return_code,'stdout':result.stdout,'stderr':result.stderr})
    proof={'schema_version':1,'ownership':ownership,'task_id':task_id,
        'dependency_snapshot_sha256':profile['dependency_snapshot_sha256'],'empty_cache':True,
        'offline_resolution':True,'environments':rows,
        'blocked_environments':binding.get('blocked_environments',[]) if binding else [],
        'verified':all(r['return_code']==0 for r in rows),'at':now()}
    proof['receipt_sha256']=hash_json(proof)
    write_json(Path(output)/'verifier-resource-probe.json',proof)
    return proof


def task_environment(profile, task_id):
    """Bind public/proxied/none verifiers to the same local dependency resources."""
    if profile.get('verifier_resource_policy') != POLICY:
        raise ValueError('Unsupported verifier resource policy')
    root=Path(profile['dependency_snapshot'])
    if sha256(root/'manifest.json')!=profile['dependency_snapshot_sha256']:
        raise ValueError('Frozen verifier dependency manifest changed')
    manifest=read_json(root/'manifest.json');validate_resources(manifest)
    resources=manifest.get('verifier_resources',{})
    if resources.get('policy')!=POLICY:
        raise ValueError('Profile requires a verifier resource publication')
    binding=resources['tasks'].get(task_id)
    if binding is None:return {}
    environments=[resources['environments'][n] for n in binding['environments']]
    primary=environments[0];runtime=resources['runtimes'][primary['runtime']]
    mounted=Path('/opt/dycon-wheels')
    links=' '.join(str(mounted/Path(e['lock']).parent/'wheels') for e in environments)
    for directory in binding.get('extra_wheel_directories',[]):
        links+=' '+str(mounted/directory)
    result={'UV_PYTHON':str(mounted/runtime['executable']),'UV_PYTHON_DOWNLOADS':'never',
            'UV_OFFLINE':'1','UV_NO_INDEX':'1','PIP_NO_INDEX':'1',
            'UV_FIND_LINKS':'/opt/dycon-verifier-links/wheels','PIP_FIND_LINKS':'/opt/dycon-verifier-links/index.html',
            'DYCON_VERIFIER_FIND_LINKS':links,
            'UV_CONSTRAINT':str(mounted/primary.get('constraints',primary['lock'])),
            'PIP_CONSTRAINT':str(mounted/primary.get('constraints',primary['lock']))}
    result['DYCON_VERIFIER_DEPENDENCY_SET_SHA256']=hash_json([e['lock_sha256'] for e in environments])
    result['DYCON_VERIFIER_INTERPRETER_SHA256']=hash_json([
        r for r in manifest['files'] if r['path'].startswith('verifier/runtimes/'+primary['runtime']+'/')])
    if task_id=='syzkaller-ppdev-syzlang':
        result.update(GOROOT='/opt/dycon-wheels/verifier/assets/go1244/go',GOTOOLCHAIN='local',
                      GOPROXY='off',GOSUMDB='off',GOMODCACHE='/opt/dycon-go-modules',
                      GOCACHE='/tmp/dycon-go-build')
    if task_id=='latex-formula-extraction':
        result['DYCON_MATHJAX_ROOT']='/opt/dycon-wheels/verifier/assets/mathjax/package/es5'
        result['PLAYWRIGHT_BROWSERS_PATH']='/opt/dycon-wheels/verifier/assets/chromium/opt/ms-playwright'
    if task_id=='fix-build-agentops':
        result.update(VIRTUALENV_EXTRA_SEARCH_DIR='/opt/dycon-wheels/verifier/environments/build310/wheels',
                      VIRTUALENV_DOWNLOAD='0',VIRTUALENV_NO_PERIODIC_UPDATE='1',
                      VIRTUALENV_PIP='23.3.2',VIRTUALENV_SETUPTOOLS='67.8.0',VIRTUALENV_WHEEL='0.42.0')
    if task_id=='setup-fuzzing-py':
        # The verifier requests the local package explicitly with --with .;
        # unselected project doc/dev extras are outside the driver contract.
        result['UV_NO_PROJECT']='1'
    return result


def _run(args, work, *, env=None, seconds=900, cwd=None):
    with (work / 'build.log').open('ab') as log:
        result = subprocess.run(list(map(str, args)), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, env=env, timeout=seconds,cwd=cwd)
    if result.returncode:
        raise RuntimeError('Verifier resource build failed; see ' + str(work / 'build.log'))


def validate_resources(manifest):
    resources = manifest.get('verifier_resources')
    if resources is None:
        return
    if resources.get('schema_version') != 1 or resources.get('policy') != POLICY:
        raise ValueError('Unsupported verifier resource schema/policy')
    files = {r['path']: r for r in manifest['files']}
    for name, runtime in resources['runtimes'].items():
        identifier(name)
        if (runtime['abi'] != 'cpython-' + name.replace('.', '')
                or runtime['platform'] != 'linux-x86_64'
                or runtime['executable'] not in files):
            raise ValueError('Verifier interpreter ABI/platform/path mismatch')
    for name, environment in resources['environments'].items():
        identifier(name)
        if environment['runtime'] not in resources['runtimes']:
            raise ValueError('Verifier environment refers to an unknown interpreter')
        lock = files.get(environment['lock'])
        if not lock or lock['sha256'] != environment['lock_sha256']:
            raise ValueError('Verifier environment lock is missing or changed')
        if environment['network_install'] is not False or not environment['offline_install_verified']:
            raise ValueError('Verifier environment has no successful offline installation')
        if any(w not in files for w in environment['wheels']):
            raise ValueError('Verifier environment wheel is missing')
    for task, binding in resources['tasks'].items():
        identifier(task)
        if any(name not in resources['environments'] for name in binding['environments']):
            raise ValueError('Task refers to an unknown verifier environment')


def verify_derivation(parent, child):
    """Qualification may transfer only when every original agent byte is identical."""
    from runtime.agents.dependencies import verify_snapshot
    parent, child = Path(parent).resolve(), Path(child).resolve()
    old, new = verify_snapshot(parent), verify_snapshot(child)
    info = new.get('verifier_resources', {})
    if (info.get('parent_manifest_sha256') != sha256(parent / 'manifest.json')
            or info.get('parent_receipt_sha256') != old['receipt_sha256']):
        raise ValueError('Verifier resource parent binding changed')
    added = {r['path']: r for r in new['files']}
    for row in old['files']:
        if added.pop(row['path'], None) != row:
            raise ValueError('Verifier resource extension changed an inherited agent file')
    if any(not name.startswith('verifier/') for name in added):
        raise ValueError('Verifier extension added content outside its resource namespace')
    for key in set(old) | set(new):
        if key not in {'files','receipt_sha256','verifier_resources'} and old.get(key) != new.get(key):
            raise ValueError('Verifier extension changed inherited agent metadata')
    return {'policy': POLICY, 'source_receipt': old['receipt_sha256'],
            'target_receipt': new['receipt_sha256'], 'verification': 'all_parent_files_and_agent_metadata_identical',
            'inherited_files': len(old['files']), 'added_files': len(added)}


def build(cfg, snapshot_id, parent, specification):
    from runtime.agents.dependencies import verify_snapshot
    root = Path(cfg['project_root']).resolve()
    parent = (root / parent).resolve()
    old = verify_snapshot(parent)
    spec = read_json(specification)
    if spec.get('schema_version') != 1 or spec.get('policy') != POLICY:
        raise ValueError('Expected a versioned verifier resource specification')
    # Resource closures can be several GiB; use the configured project artifact
    # volume while retaining task-scoped diagnostics under the project temp entry.
    storage=Path(cfg.get('paths',{}).get('artifact_roots',{}).get('run') or root).resolve()
    target = storage / 'runtime/dependency-snapshots' / identifier(snapshot_id)
    work = root / 'temp/skillsbench-dependencies' / identifier(snapshot_id)
    if storage!=root and not work.exists():
        stored_work=storage/'temp/skillsbench-dependencies'/identifier(snapshot_id)
        stored_work.mkdir(parents=True,exist_ok=False)
        work.parent.mkdir(parents=True,exist_ok=True)
        work.symlink_to(stored_work,target_is_directory=True)
    work.mkdir(parents=True, exist_ok=True)
    if target.exists() or (work / 'publication').exists():
        raise FileExistsError('Publication/build ID is immutable; inspect the previous build or use a new ID')
    # Copy bytes, never writable hardlinks into an existing publication.
    staging = work / 'publication'
    shutil.copytree(parent, staging)
    (staging / 'manifest.json').unlink()
    resource = staging / 'verifier'
    resource.mkdir(exist_ok=False)
    shutil.copy2(specification, resource / 'build-input.json')
    info = {'schema_version': 1, 'policy': POLICY,
            'parent_path': str(parent), 'parent_manifest_sha256': sha256(parent/'manifest.json'),
            'parent_receipt_sha256': old['receipt_sha256'], 'specification_sha256': sha256(specification),
            'runtimes': {}, 'environments': {}, 'tasks': spec['tasks'], 'assets': [],
            'blocked_environments':spec.get('blocked_environments',{})}
    uv = staging / 'uv'
    environment = {**os.environ, 'UV_PYTHON_DOWNLOADS': 'never', 'PYTHONDONTWRITEBYTECODE': '1',
                   'UV_HTTP_TIMEOUT':'60'}
    proxy = cfg.get('skillsbench_images',{}).get('host_proxy')
    if proxy:
        environment.update(HTTP_PROXY=proxy, HTTPS_PROXY=proxy, http_proxy=proxy, https_proxy=proxy)
    for minor, runtime in spec['runtimes'].items():
        identifier(minor)
        if 'source' in runtime:
            source = Path(runtime['source']).resolve()
        else:
            download = work / 'runtime-downloads'
            _run([uv,'python','install',runtime['version'],'--install-dir',download],work,
                 env={**environment,'UV_PYTHON_DOWNLOADS':'automatic'})
            candidates = list(download.glob('cpython-'+runtime['version']+'-linux-x86_64-gnu'))
            if len(candidates) != 1:
                raise ValueError('Pinned portable runtime download has no unique matching prefix')
            source = candidates[0]
        executable = source / 'bin' / ('python' + minor)
        probe = subprocess.run([str(executable), '-I', '-B', '-c',
            'import json,sys,sysconfig; print(json.dumps({"version":sys.version.split()[0],'
            '"abi":sys.implementation.cache_tag,"platform":sysconfig.get_platform()}))'],
            capture_output=True, text=True, check=True, timeout=30)
        identity = json.loads(probe.stdout)
        if (identity['version'] != runtime['version'] or identity['abi'] != 'cpython-'+minor.replace('.','')
                or identity['platform'] != 'linux-x86_64'):
            raise ValueError('Runtime version/ABI/platform does not match the declared input')
        dest = resource / 'runtimes' / minor
        shutil.copytree(source, dest, symlinks=False, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        info['runtimes'][minor] = {**identity, 'source':str(source),
            'origin':runtime['origin'], 'license':runtime['license'],
            'executable':(dest/'bin'/('python'+minor)).relative_to(staging).as_posix()}
    reused_publications={}
    for name, entry in spec['environments'].items():
        identifier(name)
        minor = entry['runtime']
        python = staging / info['runtimes'][minor]['executable']
        dest = resource / 'environments' / name
        if entry.get('reuse'):
            reuse=entry['reuse'];other=Path(reuse['snapshot']).resolve()
            if other not in reused_publications:
                reused_publications[other]=verify_snapshot(other)
            verified=reused_publications[other]
            if sha256(other/'manifest.json')!=reuse['manifest_sha256']:
                raise ValueError('Reused environment manifest changed')
            previous=verified['verifier_resources']['environments'][reuse['environment']]
            source_env=other/Path(previous['lock']).parent
            if ((source_env/'requirements.in').read_text().splitlines()!=entry['requirements']
                    or previous['runtime']!=minor):
                raise ValueError('Reused environment does not match the new build input')
            shutil.copytree(source_env,dest)
            info['environments'][name]={**previous,
                'lock':(dest/'requirements.txt').relative_to(staging).as_posix(),
                'wheels':[p.relative_to(staging).as_posix() for p in sorted((dest/'wheels').glob('*.whl'))],
                'reused_manifest_sha256':reuse['manifest_sha256']}
            continue
        dest.mkdir(parents=True)
        requirements=list(entry['requirements'])
        if entry.get('project_source'):
            source=entry['project_source'];archive_path=Path(source['archive'])
            if sha256(archive_path)!=source['sha256']:raise ValueError('Frozen project source archive changed')
            extracted=work/'sources'/source['sha256']
            if not extracted.exists():
                extracted.mkdir(parents=True)
                with tarfile.open(archive_path) as archive:archive.extractall(extracted,filter='data')
            project=extracted/source['member']
            metadata=tomllib.loads((project/'pyproject.toml').read_text())
            requirements.extend(metadata['build-system']['requires'])
            if (project/'uv.lock').is_file():
                exported=dest/'project-requirements.txt'
                _run([uv,'export','--project',project,'--python',python,'--frozen',
                      '--no-emit-project','--no-hashes','--no-header','--no-annotate',
                      '--output-file',exported],work,env=environment)
                requirements.extend(line for line in exported.read_text().splitlines()
                                    if line.strip() and not line.lstrip().startswith('#'))
            else:
                requirements.extend(metadata['project'].get('dependencies',[]))
        (dest/'requirements.in').write_text('\n'.join(requirements)+'\n')
        lock, wheels = dest/'requirements.txt', dest/'wheels'
        wheels.mkdir()
        args = [uv,'pip','compile','--python',python,'--generate-hashes','--no-header',
                '--no-annotate','--no-cache','--output-file',lock,dest/'requirements.in']
        for index in entry.get('indexes',[]): args.extend(['--index', index])
        _run(args, work, env=environment)
        # pip runs in the target ABI, including special CPU torch build tags.
        _run([python,'-m','pip','download','--disable-pip-version-check','--only-binary=:all:',
              '--dest',wheels,'--require-hashes','-r',lock,
              '--index-url',entry.get('pip_index','https://pypi.org/simple'),
              *sum((['--extra-index-url',i] for i in entry.get('pip_extra_indexes',[])),[])],work,env=environment)
        check = work / ('check-'+name)
        offline_env = {**environment, 'UV_OFFLINE':'1','UV_NO_INDEX':'1',
                       'UV_CACHE_DIR':str(work/('empty-cache-'+name)), 'PIP_NO_INDEX':'1'}
        _run([uv,'venv','--no-project','--python',python,check],work,env=offline_env)
        _run([uv,'pip','sync','--python',check/'bin/python','--offline','--no-index',
              '--find-links',wheels,'--require-hashes',lock],work,env=offline_env)
        _run([check/'bin/python','-c',entry.get('probe','import pytest; print(pytest.__version__)')],work,env=offline_env)
        info['environments'][name] = {'runtime':minor,'lock':lock.relative_to(staging).as_posix(),
            'lock_sha256':sha256(lock),'wheels':[p.relative_to(staging).as_posix() for p in sorted(wheels.glob('*.whl'))],
            'network_install':False,'offline_install_verified':True,'platform':'linux/amd64',
            'indexes':entry.get('indexes',[]),'probe':entry.get('probe')}
    # Constraints keep every resolver pinned; hash locks remain the authoritative
    # offline installation input. pip constraints must not implicitly enable hash mode.
    for name, entry in info['environments'].items():
        dest=staging/Path(entry['lock']).parent
        pins=[]
        for line in (dest/'requirements.txt').read_text().splitlines():
            if re.match(r'^[A-Za-z0-9][A-Za-z0-9_.-]*==',line):
                pins.append(line.rstrip().removesuffix('\\').rstrip())
        (dest/'constraints.txt').write_text('\n'.join(pins)+'\n')
        entry['constraints']=(dest/'constraints.txt').relative_to(staging).as_posix()
        entry['constraints_sha256']=sha256(dest/'constraints.txt')
    matrix=[]
    for name, entry in info['environments'].items():
        if name.startswith('agentops-py'):
            pins=(staging/entry['constraints']).read_text().splitlines()
            matrix.extend(pin+'; python_version == "'+entry['runtime']+'"' for pin in pins)
    if matrix:(resource/'agentops-matrix-constraints.txt').write_text('\n'.join(matrix)+'\n')
    for asset in spec.get('assets',[]):
        name=identifier(asset['id'])
        if 'url' in asset:
            source=work/(name+'.download')
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy}) if proxy
                                               else urllib.request.ProxyHandler())
            with opener.open(asset['url'],timeout=90) as response, source.open('wb') as stream:
                shutil.copyfileobj(response,stream)
        else:
            source=Path(asset['source']).resolve()
        if not source.is_file():
            raise ValueError('Asset source hash mismatch: '+name)
        if asset.get('sha256'):
            valid=sha256(source)==asset['sha256']
        elif asset.get('integrity','').startswith('sha512-'):
            valid=hashlib.sha512(source.read_bytes()).digest()==base64.b64decode(asset['integrity'][7:])
        else:
            valid=False
        if not valid:raise ValueError('Asset source hash mismatch: '+name)
        dest=resource/'assets'/name
        if asset.get('archive')=='tar':
            dest.mkdir(parents=True)
            with tarfile.open(source) as archive:
                # No escaping paths, special files or mutable links in a shared publication.
                for member in archive.getmembers():
                    p=Path(member.name)
                    if p.is_absolute() or '..' in p.parts or member.issym() or member.islnk() or member.isdev():
                        raise ValueError('Unsafe verifier resource archive')
                archive.extractall(dest,filter='data')
        else:
            filename=asset.get('filename',source.name)
            if Path(filename).name!=filename or filename in {'.','..'}:
                raise ValueError('Unsafe asset filename')
            dest.mkdir(parents=True);shutil.copy2(source,dest/filename)
        info['assets'].append({k:v for k,v in asset.items() if k!='source'} |
                             {'path':dest.relative_to(staging).as_posix(),'sha256':sha256(source)})
    if spec.get('go'):
        go=spec['go'];archive_path=Path(go['source_archive'])
        if sha256(archive_path)!=go['source_sha256']:raise ValueError('Go source image archive changed')
        source=work/'go-source';source.mkdir()
        with tarfile.open(archive_path) as archive:archive.extractall(source,filter='data')
        project=source/go['member'];toolchain=resource/'assets/go1244/go';cache=resource/'assets/go-modules/cache'
        if go.get('cache_source'):
            seed=Path(go['cache_source']).resolve()
            tree=[{'path':p.relative_to(seed).as_posix(),'sha256':sha256(p)}
                  for p in sorted(seed.rglob('*')) if p.is_file()]
            if hash_json(tree)!=go['cache_tree_sha256']:
                raise ValueError('Go cache seed changed')
            shutil.copytree(seed,cache,symlinks=False)
        else:
            cache.mkdir(parents=True)
        go_env={**environment,'GOROOT':str(toolchain),'GOTOOLCHAIN':'local','GOMODCACHE':str(cache),
                'GOMAXPROCS':'4',
                'GOPATH':str(work/'go-path'),'GOCACHE':str(work/'go-build-cache'),
                'GOPROXY':'https://proxy.golang.org','GOSUMDB':'sum.golang.org'}
        version=subprocess.check_output([str(toolchain/'bin/go'),'version'],env=go_env,text=True,timeout=20).strip()
        if version!='go version go1.24.4 linux/amd64':raise ValueError('Unexpected Go toolchain identity')
        # Proxy interruptions can affect a subset of a large module closure.
        # Go rechecks cached zip hashes on retry; keep checksum verification on.
        for attempt in range(3):
            try:
                _run([toolchain/'bin/go','mod','download','all'],work,env=go_env,seconds=3600,
                     cwd=project)
                break
            except RuntimeError:
                if attempt == 2:
                    raise
        info['go']={'version':version,'source_archive_sha256':go['source_sha256'],
                    'go_mod_sha256':sha256(project/'go.mod'),'go_sum_sha256':sha256(project/'go.sum'),
                    'module_cache':'verifier/assets/go-modules/cache','network_execution':'off',
                    'source_member':go['member']}
    wheel_metadata=[]
    for wheel in sorted(resource.rglob('*.whl')):
        with zipfile.ZipFile(wheel) as archive:
            members=[n for n in archive.namelist() if n.endswith('.dist-info/METADATA') and n.count('/')==1]
            if len(members)!=1:raise ValueError('Verifier wheel metadata must be unique')
            raw=archive.read(members[0]);metadata=BytesParser().parsebytes(raw,headersonly=True)
        wheel_metadata.append({'path':wheel.relative_to(staging).as_posix(),'sha256':sha256(wheel),
            'metadata_sha256':hashlib.sha256(raw).hexdigest(),'name':str(metadata['Name']),'version':str(metadata['Version']),
            'requires_python':str(metadata['Requires-Python']) if metadata.get('Requires-Python') else None,
            'license':str(metadata['License']) if metadata.get('License') else None,
            'license_expression':str(metadata['License-Expression']) if metadata.get('License-Expression') else None,
            'license_classifiers':[str(v) for v in metadata.get_all('Classifier',[]) if str(v).startswith('License ::')]})
    write_json(resource/'wheel-metadata.json',wheel_metadata)
    new=copy.deepcopy(old)
    new.pop('receipt_sha256',None)
    new['verifier_resources']=info
    new['files']=[{'path':p.relative_to(staging).as_posix(),'size':p.stat().st_size,'sha256':sha256(p)}
                  for p in sorted(staging.rglob('*')) if p.is_file()]
    new['receipt_sha256']=hash_json(new)
    write_json(staging/'manifest.json',new)
    verify_snapshot(staging)
    verify_derivation(parent,staging)
    target.parent.mkdir(parents=True,exist_ok=True)
    staging.rename(target)
    receipt={'schema_version':1,'finished_at':now(),'path':str(target),
             'manifest_sha256':sha256(target/'manifest.json'),'files':len(new['files']),
             'deep_verified':True,'parent_manifest_sha256':info['parent_manifest_sha256']}
    write_json(work/'receipt.json',receipt)
    return {'path':str(target),'manifest':new}

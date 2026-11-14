"""Controlled submissions through the existing SkillsBench verifier lifecycle."""
from __future__ import annotations

from pathlib import Path
import shlex

from core.io import sha256,write_json

CASES={'simpo-correct','simpo-wrong-output','simpo-missing-output','simpo-missing-env','simpo-wrong-rerun'}
CASES.update({'latex-correct','latex-wrong-output','latex-missing-resource'})
CASES.add('latex-compatibility-correct')
AGENTOPS_CASES={'agentops-correct','agentops-wrong','agentops-missing-dependency',
    'agentops-missing-interpreter'}
BUSINESS_CASES={**{case:'fix-build-agentops' for case in AGENTOPS_CASES},'google-auto-correct':'fix-build-google-auto','agentops-upstream-blocked':'fix-build-agentops'}


def task_for_case(case):
    if case in BUSINESS_CASES:return BUSINESS_CASES[case]
    if case=='fuzz-drivers':return 'setup-fuzzing-py'
    if case not in CASES:raise ValueError('Unsupported controlled submission')
    return 'latex-formula-extraction' if case.startswith('latex-') else 'simpo-code-reproduction'


async def prepare(env,task,profile,output,case):
    if task['instance_id']!=task_for_case(case):
        raise ValueError('Unsupported controlled submission')
    if task['instance'].get('verifier_adaptation',{}).get('policy')!='skillsbench-verifier-repair-v1':
        raise ValueError('Controls require the exact repaired task package')
    if case in AGENTOPS_CASES:
        return await prepare_agentops(env,task,profile,output,case)
    if case in BUSINESS_CASES:
        import json
        metadata=task['instance']['metadata'];settings=metadata['oracle']['env']
        root='/home/github/build' if case.startswith('agentops') else '/home/travis/build'
        code='''import hashlib,json,pathlib,subprocess,urllib.request
settings=SETTINGS
root=pathlib.Path(ROOT)
url='http://www.api.bugswarm.org/v1/diffs/'+settings['bugswarm_image_tag']
raw=urllib.request.urlopen(url,timeout=60).read();data=json.loads(raw)
repo=root/'failed'/settings['REPO_ID']
(root/'failed/failed_reasons.txt').write_text('Apply the official BugSwarm golden patch; preserve verifier checks.\\n')
patch_results=[]
for i,patch in enumerate(data['patches']):
 p=repo/('patch_'+str(i)+'.diff');p.write_text(patch['content'])
 r=subprocess.run(['git','apply',str(p)],cwd=repo,check=False)
 patch_results.append({'patch':p.name,'return_code':r.returncode,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
print(json.dumps({'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'patch_count':len(data['patches']),'patch_results':patch_results}))
'''.replace('SETTINGS',repr(settings)).replace('ROOT',repr(root))
        result=await env.exec('/opt/dycon-wheels/python/bin/python3.12 -c '+shlex.quote(code),user='root',timeout_sec=120)
        write_json(Path(output)/'controlled-submission.json',{'schema_version':1,'case':case,
            'source':'same official BugSwarm get_diff API and git apply; stdlib client avoids oracle-only installer',
            'oracle_sha256':sha256(Path(task['instance']['task_path'])/'oracle/solve.sh'),
            'task_package_sha256':task['instance']['package_sha256'],
            'return_code':result.return_code,'stdout':result.stdout,'stderr':result.stderr})
        if result.return_code:raise RuntimeError('Official BugSwarm patch fixture failed')
        return
    snapshot=Path(profile['dependency_snapshot'])
    from core.io import read_json
    manifest=read_json(snapshot/'manifest.json')
    resources=manifest['verifier_resources'];environment=resources['environments']['simpo310']
    runtime=resources['runtimes'][environment['runtime']]
    mounted=Path('/opt/dycon-wheels');venv='/opt/dycon-verifier-control'
    source=Path(task['instance']['task_path'])/'oracle/solve.sh'
    oracle=source.read_text()
    if case=='fuzz-drivers':
        oracle=oracle.replace('uv sync --python 3.12','''export UV_CONSTRAINT=/opt/dycon-wheels/verifier/environments/fuzz-$(basename "$PWD")312/constraints.txt
export UV_BUILD_CONSTRAINT="$UV_CONSTRAINT"
uv venv --python "$UV_PYTHON"
uv pip sync --offline --no-index --require-hashes --find-links /opt/dycon-wheels/verifier/environments/fuzz-$(basename "$PWD")312/wheels /opt/dycon-wheels/verifier/environments/fuzz-$(basename "$PWD")312/requirements.txt''')
        oracle=oracle.replace('uv pip install .','uv pip install --no-deps .')
        oracle=oracle.replace('uv run fuzz.py','uv run --no-project --with . --with atheris==3.0.0 fuzz.py')
        result=await env.exec('bash -c '+shlex.quote(oracle),user='root',timeout_sec=600)
        write_json(Path(output)/'controlled-submission.json',{'schema_version':1,'case':case,
            'source':'exact official five driver sources; per-library offline bootstrap replaces uv sync project discovery',
            'oracle_sha256':sha256(source),'task_package_sha256':task['instance']['package_sha256'],
            'return_code':result.return_code,'stdout':result.stdout,'stderr':result.stderr})
        # Official drivers deliberately include real negative examples; run every
        # verifier suite even when one driver exits nonzero.
        return
    if case.startswith('latex-'):
        command=oracle
        if case=='latex-wrong-output':
            command+="\nprintf '%s\\n' '$$x=1$$' > /root/latex_formula_extraction.md\n"
        result=await env.exec('bash -c '+shlex.quote(command),user='root',timeout_sec=60)
        write_json(Path(output)/'controlled-submission.json',{'schema_version':1,'case':case,
            'source':'exact official oracle plus controlled mutation','oracle_sha256':sha256(source),
            'task_package_sha256':task['instance']['package_sha256'],
            'return_code':result.return_code,'stdout':result.stdout,'stderr':result.stderr})
        if result.return_code:raise RuntimeError('Controlled submission setup failed')
        if case=='latex-compatibility-correct':
            await compare_renderers(env,task,profile,output)
        return
    # Execute the exact official method patch, then the official unit-test input.
    start=oracle.index('export TARGET=');end=oracle.index('\nEOF',start)+4
    patch=oracle[start:end].replace('/opt/py310/bin/python',venv+'/bin/python')
    command='set -eu\nexport PYTHONDONTWRITEBYTECODE=1 UV_OFFLINE=1 UV_PYTHON_DOWNLOADS=never\n'
    command+=f'{mounted}/uv venv --python {shlex.quote(str(mounted/runtime["executable"]))} {venv}\n'
    command+=f'{mounted}/uv pip sync --python {venv}/bin/python --offline --no-index --require-hashes --find-links '+\
        shlex.quote(str(mounted/Path(environment['lock']).parent/'wheels'))+' '+shlex.quote(str(mounted/environment['lock']))+'\n'
    command+=patch+'\n'
    command+=f'cd /root/SimPO\nPYTHONPATH=/root/SimPO {venv}/bin/python /root/SimPO/unit_test/unit_test_1.py\n'
    command+=f'{venv}/bin/python -VV > /root/python_info.txt\n'
    if case=='simpo-missing-output':command+='rm /root/loss.npz\n'
    elif case=='simpo-missing-env':command+='rm /root/python_info.txt\n'
    elif case=='simpo-wrong-output':
        command+=venv+'''/bin/python - <<'CONTROL'
import numpy as np
with np.load('/root/loss.npz') as data: losses=np.array(data['losses'],copy=True)
np.savez('/root/loss.npz',losses=losses+1.)
CONTROL
'''
    elif case=='simpo-wrong-rerun':
        command+=venv+'''/bin/python - <<'CONTROL'
from pathlib import Path
p=Path('/root/SimPO/scripts/simpo_trainer.py');text=p.read_text()
old='pi_logratios = policy_chosen_logps - policy_rejected_logps'
assert text.count(old)==1
p.write_text(text.replace(old,old+' + 1.0'))
CONTROL
'''
    result=await env.exec(command,user='root',timeout_sec=600)
    write_json(Path(output)/'controlled-submission.json',{'schema_version':1,'case':case,
        'source':'exact official oracle method patch plus controlled mutation',
        'oracle_sha256':sha256(source),'task_package_sha256':task['instance']['package_sha256'],
        'dependency_snapshot_sha256':profile['dependency_snapshot_sha256'],
        'return_code':result.return_code,'stdout':result.stdout,'stderr':result.stderr})
    if result.return_code:raise RuntimeError('Controlled submission setup failed; see controlled-submission.json')


async def compare_renderers(env,task,profile,output):
    """Execute the original renderer and repaired renderer against identical bytes."""
    import base64,json
    from core.io import read_json
    original=read_json(task['instance']['verifier_adaptation']['receipt_path'])['upstream_task']
    old=Path(original['instance']['task_path'])/'verifier/test_outputs.py'
    new=Path(task['instance']['task_path'])/'verifier/test_outputs.py'
    manifest=read_json(Path(profile['dependency_snapshot'])/'manifest.json')
    resources=manifest['verifier_resources'];item=resources['environments']['latex312']
    runtime=resources['runtimes'][item['runtime']];root='/tmp/dycon-render-compatibility'
    script=r'''import importlib.util,sys,hashlib,json
from pathlib import Path
from urllib.parse import unquote,urlsplit
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
 sys.modules[name]=m;spec.loader.exec_module(m);return m
old=load('legacy_renderer',ROOT+'/legacy.py');new=load('local_renderer',ROOT+'/local.py')
resource=Path('/opt/dycon-wheels/verifier/assets/mathjax/package/es5').resolve()
real_sync=old.sync_playwright
class Browser:
 def __init__(self,real):self.real=real
 def __getattr__(self,name):return getattr(self.real,name)
 def new_page(self,*a,**kw):
  page=self.real.new_page(*a,**kw)
  def serve(route):
   url=urlsplit(route.request.url)
   p=(resource/unquote(url.path.split('/es5/',1)[-1])).resolve()
   if url.hostname!='cdn.jsdelivr.net' or not p.is_relative_to(resource) or not p.is_file():
    raise RuntimeError('Unexpected baseline resource request: '+route.request.url)
   route.fulfill(path=str(p),content_type='application/javascript')
  page.route('**/*',serve);return page
class Chromium:
 def __init__(self,real):self.real=real
 def launch(self,*a,**kw):return Browser(self.real.launch(*a,**kw))
class Started:
 def __init__(self,real):self.real=real;self.chromium=Chromium(real.chromium)
 def stop(self):self.real.stop()
class Manager:
 def start(self):return Started(real_sync().start())
old.sync_playwright=Manager
a=old.MathJaxRenderer();legacy=[];rows=[]
try:
 for formula in old.EXPECTED_FORMULAS:
  latex=old._strip_dollars(formula)
  legacy.append((formula,latex,a.latex_to_png(latex)))
finally:a.close()
b=new.MathJaxRenderer()
try:
 for formula,latex,x in legacy:
  y=b.latex_to_png(latex)
  assert x==y,'Renderer PNG bytes differ'
  rows.append({'formula':formula,'legacy_png_sha256':hashlib.sha256(x).hexdigest(),
    'local_png_sha256':hashlib.sha256(y).hexdigest(),'identical':True})
finally:b.close()
print(json.dumps({'verified':True,'baseline':'original frozen Python renderer; CDN requests intercepted to pinned MathJax 3.2.2 bytes',
 'comparisons':rows,'external_requests':0}))
'''.replace('ROOT',repr(root))
    files={'legacy.py':old.read_bytes(),'local.py':new.read_bytes(),'probe.py':script.encode()}
    writer='from pathlib import Path;import base64;root=Path('+repr(root)+');root.mkdir()\n'
    for name,data in files.items():writer+='(root/'+repr(name)+').write_bytes(base64.b64decode('+repr(base64.b64encode(data).decode())+'))\n'
    command='set -eu\n/opt/dycon-wheels/python/bin/python3.12 -c '+shlex.quote(writer)+'\n'
    command+='/opt/dycon-wheels/uv venv --python /opt/dycon-wheels/'+runtime['executable']+' '+root+'/venv\n'
    command+='/opt/dycon-wheels/uv pip sync --offline --no-index --require-hashes --python '+root+'/venv/bin/python --find-links /opt/dycon-wheels/'+str(Path(item['lock']).parent/'wheels')+' /opt/dycon-wheels/'+item['lock']+'\n'
    command+=root+'/venv/bin/python '+root+'/probe.py\n'
    result=await env.exec(command,user='root',timeout_sec=180)
    write_json(Path(output)/'mathjax-rendering-comparison.json',{'legacy_source_sha256':sha256(old),
        'local_source_sha256':sha256(new),'return_code':result.return_code,'stdout':result.stdout,'stderr':result.stderr})
    if result.return_code:raise RuntimeError('Original/local MathJax comparison failed')


async def prepare_agentops(env,task,profile,output,case):
    """Use authenticated archived BugSwarm patches, with explicit negative controls."""
    import base64,json
    from core.io import read_json
    from runtime.agents.verifier_evidence import frozen_contract
    contract=frozen_contract(task)
    if 'agentops_matrix_v1' not in contract['capabilities']:
        raise ValueError('AgentOps revision control requires its registered contract')
    fixture=Path(__file__).with_name('agentops_reference.json')
    if sha256(fixture)!='18cd07efd123b574376b9f60823e7e68888b1642f367e7f86403b3a9f25818d3':
        raise ValueError('Authenticated AgentOps reference bytes changed')
    data=read_json(fixture)
    if data['failed_sha']!='b0f4a481f7483f3b18cfdc5821d38fda576fa951' or data['passed_sha']!='5a7727997dd9a942ff1889969db793c55b268b70':
        raise ValueError('Unexpected AgentOps reference identity')
    payload=base64.b64encode(fixture.read_bytes()).decode()
    script=r"""import base64,json,pathlib,subprocess
data=json.loads(base64.b64decode(PAYLOAD))
repo=pathlib.Path('/home/github/build/failed/AgentOps-AI/agentops')
pathlib.Path('/home/github/build/failed/failed_reasons.txt').write_text('Frozen official BugSwarm reference / controlled negative.\n')
rows=[]
for i,patch in enumerate(data['patches']):
 if patch['content'].startswith('Binary file '):
  rows.append({'file':patch['new_file'],'status':'binary_metadata_only_no_applicable_diff'})
  continue
 p=repo/('patch_'+str(i)+'.diff');p.write_text(patch['content'])
 if CASE!='agentops-wrong':
  subprocess.run(['git','apply',str(p)],cwd=repo,check=True)
 rows.append(p.name)
if CASE=='agentops-missing-interpreter':
 pathlib.Path('/opt/dycon-wheels/verifier/runtimes/3.8/bin/python3.8')
 # Force the required interpreter resolution to a missing path in both discovery locations.
 for name in ['/usr/local/bin/python3.8','/opt/hostedtoolcache/Python/3.8.18/x64/bin/python3.8']:
  p=pathlib.Path(name)
  if p.exists() or p.is_symlink():p.unlink()
 p=repo/'tox.ini';p.write_text(p.read_text()+'\n[testenv:py38]\nbasepython = /missing-required-python3.8\n')
if CASE=='agentops-missing-dependency':
 p=repo/'tox.ini';p.write_text(p.read_text()+'\n[testenv:py38]\ndeps = dycon-controlled-missing-package==0.0.0\n')
print(json.dumps({'failed_sha':data['failed_sha'],'passed_sha':data['passed_sha'],'patches':rows,'case':CASE}))
""".replace('PAYLOAD',repr(payload)).replace('CASE',repr(case))
    result=await env.exec('/opt/dycon-wheels/python/bin/python3.12 -c '+shlex.quote(script),user='root',timeout_sec=90)
    write_json(Path(output)/'controlled-submission.json',{'schema_version':1,'case':case,
        'source':'archived authenticated official BugSwarm diff; explicit fixture mutation only',
        'fixture_sha256':sha256(fixture),'task_package_sha256':task['instance']['package_sha256'],
        'return_code':result.return_code,'stdout':result.stdout,'stderr':result.stderr})
    if result.return_code:raise RuntimeError('AgentOps controlled fixture setup failed')

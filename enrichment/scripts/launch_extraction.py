import os,sys,subprocess,json
from pathlib import Path
r=Path('/home/wangtianyi/DyConGlobal');d=r/'state/anchors-20261017'
if not (r/'state/extraction-models.json').exists():raise RuntimeError('Model registration not complete')
launch=[]
for gpu in range(4):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTHONPATH=str(r),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
 f=open(d/f'gpu-{gpu}.log','a')
 p=subprocess.Popen([str(r/'.venvs/analysis/bin/python'),'-u',str(r/'enrichment/scripts/run_extraction_queue.py'),'--root',str(r),'--gpu',str(gpu)],cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 launch.append({'gpu':gpu,'pid':p.pid,'log':str(d/f'gpu-{gpu}.log')})
(d/'launch.json').write_text(json.dumps(launch,indent=2));print(launch)

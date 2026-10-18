"""Four-GPU resumable extraction queue; products bind exact rollout hashes."""
from __future__ import annotations
import argparse,json,os,time,traceback,gc,copy
from pathlib import Path
from baseline.catalog import Catalog
from core.io import read_json,write_json,hash_json
from core.locking import file_lock
from core.model_identity import verify_model
from enrichment.trajectory import enrich
from hidden.extract import extract_hidden

BATCH='anchors-20261017'

def frozen_config(bundle, model, anchors):
    """Never re-open editable config.yaml from a long-lived worker."""
    cfg = copy.deepcopy(bundle['configs'][model + '|' + ','.join(anchors)])
    expected = cfg.pop('config_hash')
    if hash_json(cfg) != expected:
        raise ValueError('Frozen extraction config checksum mismatch')
    cfg['config_hash'] = expected
    return cfg


def failure_scope(exc):
    # Only recognized trajectory-specific reconstruction failures may advance the queue.
    row_errors = ('Prompt count mismatch:', 'Actual prompt IDs differ',
                  'No recoverable anchors', 'Terminal message inside a model request')
    return 'trajectory' if isinstance(exc, ValueError) and any(
        str(exc).startswith(message) for message in row_errors
    ) else 'system'

def update(root,owner,key,value):
 with file_lock(root/'state/product-manifests.lock'):
  path=root/owner/'manifest.json'
  m=read_json(path)
  m.setdefault('extraction_jobs',{})[key]=value
  write_json(path,m)
def worker(root,gpu,snapshot):
 import core.model_identity as identity_module
 original_select = Catalog.select
 original_verify = identity_module.verify_model
 try:
  return _worker(root,gpu,snapshot)
 finally:
  Catalog.select = original_select
  identity_module.verify_model = original_verify

def _worker(root,gpu,snapshot):
 import torch
 from transformers import AutoTokenizer
 import hidden.adapters.hf as hf
 queue=root/'state'/BATCH/'queue.json'
 bundle=read_json(snapshot)
 models=bundle['models']
 # Preflight every frozen config before claiming any job.
 for key in bundle['configs']:
  name, anchors = key.split('|')
  frozen_config(bundle,name,anchors.split(','))
 print(json.dumps({'event':'worker_ready','gpu':gpu,'snapshot':str(snapshot),'pid':os.getpid()}),flush=True)
 catalog=Catalog(root); original=Catalog.select
 active=set()
 def select(self,**kwargs):
  x=original(self,**kwargs)
  x['rollout_ids']=[rid for rid in x['rollout_ids'] if rid in active]
  x['rollouts']={rid:x['rollouts'][rid] for rid in x['rollout_ids']}
  return x
 Catalog.select=select
 cached=None; adapter=None; tok=None; verified=set()
 while True:
  with file_lock(root/'state'/BATCH/'queue.lock'):
   q=read_json(queue)
   jobs=[j for j in q if j['status']=='pending']
   if not jobs: break
   same=[j for j in jobs if j['model']==cached]
   job=(same or jobs)[0]
   job.update(status='running',gpu=gpu,pid=os.getpid(),started=time.time())
   write_json(queue,q)
   job=dict(job)
  rid=job['rollout_id']; model=job['model']; active.clear();active.add(rid)
  dataset=BATCH+'-'+rid; capture=dataset
  base={'batch':BATCH,'rollout_id':rid,'task_id':job['task_id'],'run_id':job['run_id'],
    'trajectory_sha256':job['trajectory_sha256'],'anchors':[{'name':a,'text':'<think>' if a=='think_open' else '</think>'} for a in job['anchors']],
    'model':models[model],'gpu':gpu,'dataset_id':dataset,'capture_id':capture,'status':'running','config_snapshot':str(snapshot.relative_to(root))}
  for owner in ['enrichment','hidden']:update(root,owner,rid,base)
  print(json.dumps({'event':'start','gpu':gpu,'rollout':rid,'model':model,'anchors':job['anchors']}),flush=True)
  try:
   cfg=frozen_config(bundle,model,job['anchors'])
   if model not in verified:
    verify_model(cfg);verified.add(model)
   # A verified checkpoint is immutable for this worker lifetime. Avoid hashing 200GB per rollout.
   import core.model_identity as mi
   identity=read_json(root/'runtime/model-identities'/(models[model]['fingerprint']+'.json'))
   def already_verified(candidate):
    if candidate['model']['fingerprint'] != identity['fingerprint']:
     raise ValueError('Adapter requested an unverified checkpoint')
    return identity
   mi.verify_model=already_verified
   if cached!=model:
    adapter=None;tok=None;gc.collect();torch.cuda.empty_cache()
    tok=AutoTokenizer.from_pretrained(models[model]['path'],local_files_only=True)
    cached=model
   enrich(cfg,dataset,tok)
   update(root,'enrichment',rid,{**base,'status':'completed','receipt':'enrichment/outputs/'+dataset+'/receipt.json'})
   if adapter is None:
    print(json.dumps({'event':'loading_model','gpu':gpu,'model':model}),flush=True)
    adapter=hf.load_adapter(cfg)
   extract_hidden(cfg,dataset,capture,adapter)
   update(root,'hidden',rid,{**base,'status':'completed','receipt':'hidden/outputs/'+capture+'/receipt.json'})
   status='completed';error=None;fatal=False
   print(json.dumps({'event':'completed','gpu':gpu,'rollout':rid}),flush=True)
  except Exception as exc:
   error=traceback.format_exc();print(error,flush=True)
   fatal=failure_scope(exc)=='system'
   status='pending' if fatal else 'failed'
   update(root,'hidden',rid,{**base,'status':status,'error':error})
   ep=root/'enrichment/outputs'/dataset/'receipt.json'
   if not ep.exists(): update(root,'enrichment',rid,{**base,'status':status,'error':error})
   gc.collect();torch.cuda.empty_cache()
  with file_lock(root/'state'/BATCH/'queue.lock'):
   q=read_json(queue)
   target=next(j for j in q if j['rollout_id']==rid)
   target.update(status=status,error=error,finished=time.time())
   if error:
    target.setdefault('failure_history',[]).append({'error':error,'gpu':gpu,'pid':os.getpid(),'at':time.time(),'scope':'system' if fatal else 'trajectory'})
   write_json(queue,q)
  if fatal:
   print(json.dumps({'event':'worker_blocked','gpu':gpu,'rollout':rid,'reason':error.splitlines()[-1]}),flush=True)
   raise RuntimeError('System error; current job retained as pending. Worker stopped before claiming more jobs.')
 print(json.dumps({'event':'worker_finished','gpu':gpu}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
 p.add_argument('--snapshot',type=Path,required=True)
 a=p.parse_args()
 with file_lock(a.root/'state'/BATCH/f'gpu-{a.gpu}.lock',nonblocking=True):
  worker(a.root,a.gpu,a.snapshot)

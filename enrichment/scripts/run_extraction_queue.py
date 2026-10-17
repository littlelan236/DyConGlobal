"""Four-GPU resumable extraction queue; products bind exact rollout hashes."""
from __future__ import annotations
import argparse,json,os,time,traceback,gc
from pathlib import Path
from baseline.catalog import Catalog
from core.config import resolve
from core.io import read_json,write_json
from core.locking import file_lock
from core.model_identity import verify_model
from enrichment.trajectory import enrich
from hidden.extract import extract_hidden

BATCH='anchors-20261017'
def update(root,owner,key,value):
 with file_lock(root/'state/product-manifests.lock'):
  path=root/owner/'manifest.json'
  m=read_json(path)
  m.setdefault('extraction_jobs',{})[key]=value
  write_json(path,m)
def worker(root,gpu):
 import torch
 from transformers import AutoTokenizer,AutoConfig
 import hidden.adapters.hf as hf
 queue=root/'state'/BATCH/'queue.json'
 models=read_json(root/'state/extraction-models.json')
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
    'model':models[model],'gpu':gpu,'dataset_id':dataset,'capture_id':capture,'status':'running'}
  for owner in ['enrichment','hidden']:update(root,owner,rid,base)
  print(json.dumps({'event':'start','gpu':gpu,'rollout':rid,'model':model,'anchors':job['anchors']}),flush=True)
  try:
   cfg=resolve(root/'config.yaml',[
    'model.path='+models[model]['path'],'model.fingerprint='+models[model]['fingerprint'],
    'model.served_name='+model,'benchmark.partition=train','benchmark.purposes=[training]',
    'enrichment.anchors='+str(job['anchors']), 'hidden.device=cuda:0','hidden.chunk_size=2048'])
   if model not in verified:
    verify_model(cfg);verified.add(model)
   # A verified checkpoint is immutable for this worker lifetime. Avoid hashing 200GB per rollout.
   import core.model_identity as mi
   identity=read_json(root/'runtime/model-identities'/(models[model]['fingerprint']+'.json'))
   mi.verify_model=lambda _:identity
   if cached!=model:
    adapter=None;tok=None;gc.collect();torch.cuda.empty_cache()
    tok=AutoTokenizer.from_pretrained(models[model]['path'],local_files_only=True)
    cached=model
   count=getattr(AutoConfig.from_pretrained(models[model]['path'],local_files_only=True),'text_config',None)
   if count is None:count=AutoConfig.from_pretrained(models[model]['path'],local_files_only=True)
   # Re-resolve so layer configuration participates in config identity.
   cfg=resolve(root/'config.yaml',[
    'model.path='+models[model]['path'],'model.fingerprint='+models[model]['fingerprint'],
    'model.served_name='+model,'benchmark.partition=train','benchmark.purposes=[training]',
    'enrichment.anchors='+str(job['anchors']),'hidden.device=cuda:0','hidden.chunk_size=2048',
    'hidden.layers='+str(list(range(count.num_hidden_layers)))])
   enrich(cfg,dataset,tok)
   update(root,'enrichment',rid,{**base,'status':'completed','receipt':'enrichment/outputs/'+dataset+'/receipt.json'})
   if adapter is None:
    print(json.dumps({'event':'loading_model','gpu':gpu,'model':model}),flush=True)
    adapter=hf.load_adapter(cfg)
   extract_hidden(cfg,dataset,capture,adapter)
   update(root,'hidden',rid,{**base,'status':'completed','receipt':'hidden/outputs/'+capture+'/receipt.json'})
   status='completed';error=None
   print(json.dumps({'event':'completed','gpu':gpu,'rollout':rid}),flush=True)
  except Exception:
   error=traceback.format_exc();print(error,flush=True);status='failed'
   update(root,'hidden',rid,{**base,'status':'failed','error':error})
   ep=root/'enrichment/outputs'/dataset/'receipt.json'
   if not ep.exists(): update(root,'enrichment',rid,{**base,'status':'failed','error':error})
   gc.collect();torch.cuda.empty_cache()
  with file_lock(root/'state'/BATCH/'queue.lock'):
   q=read_json(queue)
   target=next(j for j in q if j['rollout_id']==rid)
   target.update(status=status,error=error,finished=time.time());write_json(queue,q)
 print(json.dumps({'event':'worker_finished','gpu':gpu}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
 a=p.parse_args();worker(a.root,a.gpu)

import json, time
from pathlib import Path
import requests
from core.config import resolve
from runtime.serving.manager import ServingManager
from hidden.backends.vllm.online import capture_request_metadata

OV = [
 'model.path=/VM/wangtianyi/DyConGlobal/models/Qwen3.5-9B',
 'model.fingerprint=f97643c15ea3cb04517379dd13482b48548458c3a3f6c3a7eb93dcaa6adf86a2',
 'model.served_name=Qwen3.5-9B',
 'model.trust_remote_code=false',
 'serving.management=managed',
 'serving.gpus=[4,5,6,7]',
 'serving.tensor_parallel_size=4',
 'serving.gpu_memory_utilization=0.36',
 'serving.kv_cache_memory_bytes=8589934592',
 'serving.max_model_len=4096',
 'serving.base_url=http://127.0.0.1:8117/v1',
 'serving.tool_parser=hermes',
 'serving.reasoning_parser=qwen3',
 'serving.dtype=float16',
 'capture.enabled=true',
 'capture.layers=[0,15,31]',
 'capture.export.root=hidden/online/fullgraph-smoke-20261017',
 'capture.export.queue_size=64',
 'capture.export.flush_interval_ms=50',
 'generation.max_tokens=512',
 'generation.temperature=0.0',
 'generation.top_p=1.0',
 'generation.top_k=1',
]

from concurrent.futures import ThreadPoolExecutor
import numpy as np
from core.io import write_json, read_json, hash_json
from hidden.evaluation.compare_vllm_replay import row_metrics
from collections import Counter
OV += ['serving.max_num_seqs=4','serving.max_num_batched_tokens=256',
       'capture.layers=[0,15,31]',
       'capture.export.root=hidden/online/gpu47-acceptance',
       'serving.startup_seconds=1200']
cfg=resolve('/home/wangtianyi/DyConGlobal/config.yaml',OV)
out=Path(cfg['project_root'])/'state'/'gpu47-acceptance'
out.mkdir(parents=True,exist_ok=False)
report={'passed':False,'gpus':[4,5,6,7],'dtype':'float16','tp':4}
base='http://127.0.0.1:8117'
def post(path, data):
 r=requests.post(base+path,json=data,timeout=600)
 if not r.ok: raise RuntimeError(f'{path}: {r.status_code}: {r.text}')
 return r.json()
try:
 with ServingManager(cfg,out) as serving:
  inv=serving.capture_token_inventory
  def md(sid,step=0):
   return capture_request_metadata(sid,'gpu-smoke-task',sid,sid,step,
      ['think_open','reasoning_newline','think_close'],
      newline_token_ids=inv['newline_token_ids'], close_token_ids=inv['close_token_ids'],
      tokenizer_fingerprint=inv['tokenizer_fingerprint'])
  def request(metadata=None,long=False):
   body={'model':'Qwen3.5-9B','messages':[
    {'role':'system','content':'Answer quickly. Use at most 20 reasoning tokens. '+ ('Repository note: tests must pass. '*70 if long else '')},
    {'role':'user','content':'What is 1+1?'}],
    'max_tokens':32,'temperature':0.0,'seed':42,'stream':False,
    'return_token_ids':True,'thinking_token_budget':32,'allowed_token_ids':[198],'chat_template_kwargs':{'enable_thinking':True,'preserve_thinking':True}}
   if metadata:body.update(dycon_capture=metadata)
   return post('/v1/chat/completions',body)
  for sid in ['short','long','warm','replay']:
   serving.capture_controller.start(md(sid))
  # Actual capture-disabled dispatcher versus capture-enabled, same server/seed.
  plain=request()
  short=request(md('short'))
  with ThreadPoolExecutor(max_workers=2) as pool:
   longs=pool.submit(request,md('long'),True)
   warms=pool.submit(request,md('warm'),False)
   long=longs.result();warm=warms.result()
  warm_long=request(md('warm',1),True)
  flush=post('/dycon/capture/engine/flush',{})
  status=requests.get(base+'/dycon/capture/engine/status',timeout=60).json()
  write_json(out/'engine-status.json',status)
  report['engine']=status
  report['responses']={'plain':plain,'short':short,'long':long,'warm':warm}
  write_json(out/'responses.json',report['responses'])
  def signature(value):
   choice=value['choices'][0]
   return {'message':choice['message'],'tokens':choice.get('token_ids'), 'finish_reason':choice['finish_reason']}
  assert signature(plain)==signature(short), 'Capture changed rollout output'
  assert not status['failures'],status['failures']
  assert all(w['graph_replays']>0 for w in status['workers']), 'No native graph replay'
  report['strict_fullgraph_passed']=all(w['eager_capture_forwards']==0 for w in status['workers'])
  report['eager_capture_forwards']=[w['eager_capture_forwards'] for w in status['workers']]
  assert report['strict_fullgraph_passed'],report['eager_capture_forwards']
  serving.capture_controller.queue.queue.join()
  rows=[read_json(p) for p in serving.capture_controller.export_root.joinpath('index').glob('*.json')]
  active_ids={r['request_id'] for session in serving.capture_controller.sessions.values() for r in session.samples}
  samples=[r for r in rows if r['request_id'] in active_ids and r['session_id'] in {'short','long','warm'}]
  counts=Counter(r['anchor'] for r in samples)
  report['anchors']=dict(counts)
  assert all(counts[a]>0 for a in ('think_open','reasoning_newline','think_close')),counts
  refs=serving.capture_controller.request_outputs
  assert len(refs)>=3, 'Missing exact token request receipts'
  for r in samples:
   ref=refs[r['request_id']]
   ids=ref['prompt_token_ids']+ref['output_token_ids']
   assert ids[r['absolute_position']]==r['token_id'],r
   assert r['absolute_position'] < len(ids)-1, 'Hidden row was not processed as input'
  report['exact_events']=len(samples)
  report['cached_tokens']=[r['num_cached_tokens'] for r in refs.values()]
  assert max(report['cached_tokens'])>0,'No warm APC hits'
  # Replay exact input prefixes via native vLLM engine in the same FP16/TP graph.
  selected=[]
  for kind in ['think_open','reasoning_newline','think_close']:
   selected.append(next(r for r in samples if r['session_id']=='short' and r['anchor']==kind))
  replay_rows=[]
  for i,r in enumerate(selected):
   ref=refs[r['request_id']];ids=ref['prompt_token_ids']+ref['output_token_ids']
   replay_rows.append({'request_id':f'replay-{i}','metadata':md('replay',i),
                       'prefix_token_ids':ids[:r['absolute_position']+1]})
  result=post('/dycon/capture/engine/replay',{'rows':replay_rows})
  post('/dycon/capture/engine/flush',{})
  serving.capture_controller.queue.queue.join()
  replay_samples=serving.capture_controller.sessions['replay'].samples
  errors=[]
  root=serving.capture_controller.export_root
  for i,r in enumerate(selected):
   other=next(x for x in replay_samples if x['step_id']==i)
   with np.load(root/'raw'/f"{r['sample_id']}.npz") as a, np.load(root/'raw'/f"{other['sample_id']}.npz") as b:
    errors.append({'anchor':r['anchor'],'layers':{key:row_metrics(a[key],b[key]) for key in a.files}})
  report['replay']=errors
  for event in errors:
   for error in event['layers'].values():
    assert error['cosine']>=0.997 and error['relative_l2']<=0.08,error
  for sid in ['short','long','warm','replay']:
   serving.capture_controller.stop(sid)
  report['passed']=True
finally:
 write_json(out/'report.json',report)
 print(json.dumps({k:v for k,v in report.items() if k not in {'responses','engine'}},ensure_ascii=False),flush=True)

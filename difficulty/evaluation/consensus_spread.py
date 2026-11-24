import csv, math, statistics
from pathlib import Path
from tempfile import TemporaryDirectory
from core.io import artifact_dir, identifier, now, sha256, write_json
from core.locking import exclusive_artifact
from difficulty.evaluation.parallel import collect_parallel_scores, list_parallel_trajectories, ParallelEvidenceUnavailable
from action.evaluation.branch_curves import interpolate_at

def _finite(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def plot_consensus_spread(cfg, run_id, plot_id, *, rollout_ids=None, task_id=None, bins=101):
 identifier(run_id); identifier(plot_id)
 from core.config import load_snapshot
 import json
 root=Path(cfg['project_root']); snap=load_snapshot(root,run_id)
 published=Path(snap['paths']['nas_root']) / snap['capture']['export']['root'] / run_id
 if rollout_ids is None:
  catalog=__import__('application.execution.evidence',fromlist=['RunEvidence']).RunEvidence(root,run_id).read()
  rollout_ids=[rid for rid,r in catalog['rollouts'].items() if not r.get('provenance',{}).get('quarantined')]
  if task_id: rollout_ids=[rid for rid in rollout_ids if catalog['tasks'][catalog['rollouts'][rid]['task_key']]['instance_id']==task_id]
 trajectories=[]; unavailable=[]
 for rid in rollout_ids:
  sessions=[]
  for fin in published.glob('sessions/*/finalization.json'):
   try:
    val=json.loads(fin.read_text())
    if val.get('rollout_id')==rid: sessions.append((fin.parent,val))
   except Exception: pass
  if not sessions: unavailable.append({'rollout_id':rid,'reason':'no finalization'}); continue
  session,finalization=sessions[0]
  base=session/'parallel/branch-index.json'
  if not base.exists(): unavailable.append({'rollout_id':rid,'reason':'no branch-index'}); continue
  try: rows=json.loads(base.read_text())
  except Exception: unavailable.append({'rollout_id':rid,'reason':'unreadable branch-index'}); continue
  action_steps=[mapping.get('agent_step_index') for mapping in finalization.get('responses',{}).values()
                if mapping.get('is_action_response') and isinstance(mapping.get('agent_step_index'),int)]
  trajectory_length=max(action_steps)+1 if action_steps else 0
  if trajectory_length == 0:
   unavailable.append({'rollout_id':rid,'reason':'no action-step bindings'}); continue
  groups={}
  for row in rows:
   score=row.get('score')
   events=row.get('close_events') or []
   steps={event.get('step_id') for event in events if isinstance(event.get('step_id'),int)}
   if not _finite(score) or len(steps)!=1: continue
   step=next(iter(steps))
   groups.setdefault(step,[]).append(float(score))
  points={k:statistics.pstdev(v) for k,v in groups.items() if v}
  progress_points=sorted((step/max(trajectory_length-1,1),value) for step,value in points.items()
                         if 0 <= step < trajectory_length)
  if progress_points: trajectories.append((rid,progress_points))
  else: unavailable.append({'rollout_id':rid,'reason':'no finite branch scores'})
 if not trajectories: raise ValueError('No trajectories with finite branch scores')
 aggregate=[]
 for i in range(bins):
  target=i/(bins-1)
  vals=[value for _,points in trajectories if (value:=interpolate_at(points,target)) is not None]
  aggregate.append({'progress':i/(bins-1),'score_std_mean':statistics.fmean(vals) if vals else None,'trajectory_count':len(vals)})
 out=artifact_dir(cfg,'difficulty',plot_id,'evaluations')
 if out.exists(): raise FileExistsError(f'Use a new plot ID: {out}')
 summary={'schema_version':1,'plot_id':plot_id,'run_id':run_id,'created_at':now(),'metric':'population standard deviation of finite branch scores','trajectory_count':len(trajectories),'unavailable_trajectory_count':len(unavailable),'bin_count':bins,'aggregation':'equal trajectory weight; direct branch-index read; validation skipped'}
 out.parent.mkdir(parents=True,exist_ok=True)
 with TemporaryDirectory(prefix='.consensus-spread-',dir=out.parent) as td:
  stage=Path(td); stage.mkdir(exist_ok=True)
  with (stage/'curve.csv').open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=list(aggregate[0])); w.writeheader(); w.writerows(aggregate)
  from matplotlib.backends.backend_agg import FigureCanvasAgg
  from matplotlib.figure import Figure
  fig=Figure(figsize=(8,4.8)); FigureCanvasAgg(fig); ax=fig.subplots(); ax.plot([r['progress']*100 for r in aggregate],[r['score_std_mean'] if r['score_std_mean'] is not None else float('nan') for r in aggregate],color='#2563eb',lw=2,label='Mean branch-score std'); ax.set(xlabel='Normalized trajectory progress (%)',ylabel='Score standard deviation',title='Mean consensus branch-score spread',xlim=(0,100)); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(stage/'consensus_spread.png',dpi=180); fig.savefig(stage/'consensus_spread.svg')
  write_json(stage/'summary.json',summary); write_json(stage/'selection.json',{'requested_rollout_ids':rollout_ids,'included_rollout_ids':[r for r,_ in trajectories],'unavailable':unavailable}); write_json(stage/'receipt.json',{'schema_version':1,'state':'complete','created_at':summary['created_at'],'source_hashes':{},'code':{'module':sha256(Path(__file__))},'files':{p.name:sha256(p) for p in stage.iterdir() if p.is_file()}}); stage.rename(out)
 return {'output':str(out),'plot':str(out/'consensus_spread.png'),**summary}

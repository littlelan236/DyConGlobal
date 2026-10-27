"""Immutable review-length analyses over frozen submission and action parents."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from importlib.metadata import version
from action.analyzers.review_phases import VERSION, SETTINGS, analyze, diff_paths
from action.analyzers.taxonomy import extract_command
from action.artifacts import load_local
from action.datasets.trajectories import read_messages
from action.evaluation.review_statistics import summarize
from core.io import identifier, now, owned_path, read_json, sha256, write_json
from core.locking import exclusive_artifact

CODE_FILES = ['action/analyzers/review_length.py','action/analyzers/review_phases.py',
              'action/analyzers/review_effects.py','action/evaluation/review_statistics.py',
              'action/analyzers/taxonomy.py','action/datasets/trajectories.py','core/cli.py']
CSV_FIELDS = ['source_run_id','rollout_id','instance_id','assistant_steps','status',
              'review_steps','review_steps_permissive','review_span','review_step_share',
              'final_review_start_step','terminal_submit_step','last_candidate_change_step',
              'episode_count','rework_after_review','evaluation_group','unknown_reasons','warnings']


def verify_files(directory, receipt_name):
    receipt = read_json(directory / receipt_name)
    for relative, expected in receipt['files'].items():
        if sha256(owned_path(directory,relative)) != expected:
            raise ValueError(f'Parent/output checksum mismatch: {relative}')
    return receipt


def load_audit(directory):
    receipt = verify_files(directory,'audit-receipt.json')
    if not {'audit.json','summary.json','scope.json','trajectories.jsonl'} <= set(receipt['files']):
        raise ValueError('Incomplete submission-audit receipt')
    rows = [json.loads(x) for x in (directory/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()]
    if len({(r['source_run_id'],r['rollout_id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate submitted trajectory identity')
    return rows


def evaluation_group(row):
    if row.get('latest_patch_failure'):return row['latest_patch_failure']
    evaluation = row.get('latest_evaluation')
    if evaluation is None:return 'not_evaluated'
    outcome = evaluation['outcome']
    if outcome.get('status') == 'completed':
        return 'resolved' if outcome.get('resolved') is True else 'unresolved'
    return 'other_evaluation_failure'


def make_steps(messages, labels):
    assistant = [(i,m) for i,m in enumerate(messages) if m.get('role')=='assistant']
    labels = sorted(labels,key=lambda r:r['step_id'])
    if [r['step_id'] for r in labels] != list(range(len(assistant))):
        raise ValueError('Action/assistant step coverage mismatch')
    steps = []
    for sid,(index,message) in enumerate(assistant):
        label = labels[sid]
        if label['message_index'] != index or label.get('command') != extract_command(message):
            raise ValueError('Action label does not match canonical message/tool identity')
        end = assistant[sid+1][0] if sid+1<len(assistant) else len(messages)
        observations = [{'role':o['role'],'content':str(o.get('content') or ''),
                         'returncode':(o.get('extra') or {}).get('returncode')}
                        for o in messages[index+1:end] if o.get('role') in ('tool','exit')]
        steps.append({'step_id':sid,'message_index':index,'content':message.get('content'),
                      'reasoning':message.get('reasoning_content') or (message.get('provider_specific_fields') or {}).get('reasoning'),
                      'action':label,'observations':observations})
    return steps


def compact_step(source, step):
    def excerpt(value,limit):
        raw=str(value or '')
        return {'excerpt':raw[:limit],'characters':len(raw),
                'sha256':hashlib.sha256(raw.encode()).hexdigest(),'truncated':len(raw)>limit}
    return {'source_run_id':source['source_run_id'],'rollout_id':source['rollout_id'],
            **{k:v for k,v in step.items() if k not in ('content','reasoning','observations','action')},
            'action':{k:v for k,v in step['action'].items() if k in ('label','components','command','rules','uncertainties')},
            'content':excerpt(step.get('content'),800),'reasoning':excerpt(step.get('reasoning'),1600),
            'observations':[{**{k:v for k,v in o.items() if k!='content'},'content':excerpt(o['content'],2000)} for o in step['observations']]}


def report_markdown(analysis_id, summary):
    lines=['# 复查长度运行结果','',f'Analysis ID: {analysis_id}','',
           f'冻结来源：{len(summary["by_run"])}个run、{summary["pooled"]["submitted"]}条submitted；单位为canonical assistant step。',
           '以下为静态规则估计；unknown不填0，均值/分位数只以可估计轨迹为分母。','',
           '| Run | Submitted | 可估计 | Unknown | 净步数均值 | 中位数 | P90 | 跨度均值 | 中段复查后晚变更 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(value):return 'N/A' if value is None else f'{value:.2f}'
    for run,v in [*summary['by_run'].items(),('TOTAL',summary['pooled'])]:
        d=v['review_steps']
        lines.append(f'| {run} | {v["submitted"]} | {v["estimated"]} | {v["unknown"]} | {fmt(d["mean"])} | {fmt(d["median"])} | {fmt(d["p90"])} | {fmt(v["review_span"]["mean"])} | {v["mid_review_before_unresolved_late_mutation"]}/{v["unresolved_late_mutation_trajectories"]} |')
    lines += ['', '严格净步数只计VERIFY/VERIFY_SETUP；打包、清理和提交重试可占跨度，但不计净步数。',
              '再次修改候选解或恢复工作区状态会分隔episode；混合修改+验证step不计严格复查。',
              'unknown原因、警告、无警告子集及评测分组见summary.json，逐步证据见steps.jsonl。',
              '多个run使用相同80题，此处仅作描述性比较，不构成因果或显著性结论。','']
    return '\n'.join(lines)


@exclusive_artifact('action','analysis_id')
def run_review_length(cfg, analysis_id, *, audit_id, action_id):
    root=Path(cfg['project_root']).resolve();out=root/'action/outputs'/identifier(analysis_id)
    if out.exists():raise FileExistsError(f'Use a new analysis ID: {out}')
    audit=root/'action/outputs'/identifier(audit_id);audit_rows=load_audit(audit)
    action,parent,selection,labels,_,_=load_local(root,action_id)
    if sha256(root/'action/analyzers/taxonomy.py') != parent['code']['action/analyzers/taxonomy.py']:
        raise ValueError('Current shell taxonomy differs from parent action snapshot')
    sources={(s['source_run_id'],s['rollout_id']):s for s in selection['sources']}
    requested={(r['source_run_id'],r['rollout_id']) for r in audit_rows}
    if set(sources)!=requested or selection.get('duplicate_trajectory_aliases'):
        raise ValueError('Action parent must cover exactly the audit trajectories, without aliases')
    groups={key:[] for key in requested}
    for label in labels:groups[label['source_run_id'],label['rollout_id']].append(label)
    before={name:(root/name).read_bytes() for name in CODE_FILES}
    parents={'audit':{'analysis_id':audit_id,'receipt_path':str((audit/'audit-receipt.json').relative_to(root)),
                      'sha256':sha256(audit/'audit-receipt.json')},
             'action':{'analysis_id':action_id,'receipt_path':str((action/'receipt.json').relative_to(root)),
                       'sha256':sha256(action/'receipt.json')}}
    out.parent.mkdir(parents=True,exist_ok=True);results=[]
    with TemporaryDirectory(prefix='.review-length-',dir=out.parent) as temporary:
        stage=Path(temporary)
        with (stage/'steps.jsonl').open('w',encoding='utf-8') as stream:
            for row in audit_rows:
                key=row['source_run_id'],row['rollout_id'];source=sources[key]
                if source['trajectory_sha256']!=row['trajectory_sha256']:raise ValueError('Parent/audit trajectory SHA mismatch')
                messages=read_messages(root,source);patch=owned_path(root,row['patch_path']).read_bytes()
                if hashlib.sha256(patch).hexdigest()!=row['patch_sha256']:raise ValueError('Source patch SHA mismatch')
                result=analyze({'source':source,'steps':make_steps(messages,groups[key]),
                                'delivery_paths':diff_paths(patch.decode('utf-8','replace'))})
                for step in result.pop('step_features'):
                    stream.write(json.dumps(compact_step(source,step),ensure_ascii=False,allow_nan=False)+'\n')
                result.update({'trajectory_path':source['trajectory_path'],'trajectory_sha256':source['trajectory_sha256'],
                               'patch_path':row['patch_path'],'patch_sha256':row['patch_sha256'],
                               'evaluation_group':evaluation_group(row)})
                results.append(result)
        summary=summarize(results)
        metadata={'schema_version':1,'analysis_id':analysis_id,'artifact_type':'action-review-length',
                  'state':'complete','created_at':now(),'method':VERSION,'settings':SETTINGS,'parents':parents,
                  'source_runs':sorted({r['source_run_id'] for r in results}),
                  'code':{name:hashlib.sha256(raw).hexdigest() for name,raw in before.items()},
                  'packages':{name:version(name) for name in ('tree-sitter','tree-sitter-bash')},
                  'scope_note':'Retrospective bounded static rule estimates, not human-validated psychological intent',
                  'percentile_method':'linear interpolation at (n-1)*p; unknown excluded, valid zero included'}
        selection_rows=[{k:r[k] for k in ('source_run_id','rollout_id','trajectory_path','trajectory_sha256','patch_path','patch_sha256')} for r in results]
        for filename,value in [('analysis.json',metadata),('summary.json',summary),('selection.json',selection_rows)]:write_json(stage/filename,value)
        (stage/'trajectories.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in results),encoding='utf-8')
        with (stage/'trajectories.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=CSV_FIELDS);writer.writeheader()
            writer.writerows({k:r[k] for k in CSV_FIELDS} for r in results)
        (stage/'REPORT.md').write_text(report_markdown(analysis_id,summary),encoding='utf-8')
        for name,raw in before.items():
            path=stage/'code'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            if (root/name).read_bytes()!=raw:raise ValueError('Review implementation changed during analysis')
        for p in parents.values():
            if sha256(owned_path(root,p['receipt_path']))!=p['sha256']:raise ValueError('Parent receipt changed during analysis')
        for row in results:
            for kind in ('trajectory','patch'):
                if sha256(owned_path(root,row[kind+'_path']))!=row[kind+'_sha256']:raise ValueError('Source bytes changed during review analysis')
        write_json(stage/'receipt.json',{'schema_version':1,'analysis_id':analysis_id,'method':VERSION,
                   'files':{str(p.relative_to(stage)):sha256(p) for p in sorted(stage.rglob('*')) if p.is_file()}})
        stage.rename(out)
    return {'output':str(out),'summary':summary,'check':check_review_length(root,analysis_id)}


def check_review_length(root,analysis_id):
    root=Path(root).resolve();out=root/'action/outputs'/identifier(analysis_id)
    receipt=verify_files(out,'receipt.json');metadata=read_json(out/'analysis.json')
    if metadata.get('artifact_type')!='action-review-length' or metadata['analysis_id']!=analysis_id or receipt['analysis_id']!=analysis_id:
        raise ValueError('Review artifact identity mismatch')
    required={'analysis.json','summary.json','selection.json','steps.jsonl','trajectories.jsonl','trajectories.csv','REPORT.md'}
    required|={'code/'+name for name in metadata['code']}
    if set(receipt['files'])!=required:raise ValueError('Incomplete review artifact receipt')
    for p in metadata['parents'].values():
        if sha256(owned_path(root,p['receipt_path']))!=p['sha256']:raise ValueError('Frozen parent receipt changed')
    load_audit(root/'action/outputs'/metadata['parents']['audit']['analysis_id'])
    load_local(root,metadata['parents']['action']['analysis_id'])
    rows=[json.loads(x) for x in (out/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()]
    keys={(r['source_run_id'],r['rollout_id']) for r in rows}
    if len(keys)!=len(rows) or summarize(rows)!=read_json(out/'summary.json'):raise ValueError('Review aggregate/identity mismatch')
    steps={}
    with (out/'steps.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            s=json.loads(line);steps.setdefault((s['source_run_id'],s['rollout_id']),[]).append(s)
    if set(steps)!=keys:raise ValueError('Review step/trajectory coverage mismatch')
    for row in rows:
        unit=steps[row['source_run_id'],row['rollout_id']]
        if [s['step_id'] for s in unit]!=list(range(row['assistant_steps'])):raise ValueError('Review step order or coverage mismatch')
        if row['review_steps'] is not None:
            start=row['final_review_start_step']
            expected=[s for s in unit if start is not None and start<=s['step_id']<row['terminal_submit_step'] and s['phase'] in ('VERIFY','VERIFY_SETUP')]
            if len(expected)!=row['review_steps']:raise ValueError('Review count disagrees with step labels')
        for kind in ('trajectory','patch'):
            if sha256(owned_path(root,row[kind+'_path']))!=row[kind+'_sha256']:raise ValueError('Canonical source content changed')
    return {'verified':True,'analysis_id':analysis_id,'trajectories':len(rows),
            'assistant_steps':sum(r['assistant_steps'] for r in rows),
            'current_implementation_matches_snapshot':all((root/name).is_file() and sha256(root/name)==value for name,value in metadata['code'].items())}

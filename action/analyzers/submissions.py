"""Submitted patch audit; source runs and evidence remain read-only.

Run through ``dycon action-submission-audit``. Trajectory parsing is shared
with action-classify. This audit has its own receipt, not an action label receipt.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from action.datasets.trajectories import read_actions, read_messages
from application.execution.evidence import RunEvidence
from core.io import identifier, owned_path
from core.locking import exclusive_artifact

VERSION = 'submitted-patch-audit-v1'
REFERENCE_RUN = 'verified80-parallel-baseline-20261021'
CODE = ['action/analyzers/submissions.py', 'action/datasets/trajectories.py',
        'application/execution/evidence.py', 'core/cli.py']
PATCH_MARKERS = ('patch apply failed', 'only garbage was found in the patch input',
                 'failed to apply patch', 'patch failed', 'missingpatch')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def eligible(data):
    return {rid: row for rid, row in data.get('rollouts', {}).items()
            if not row.get('provenance', {}).get('quarantined')
            and data['tasks'][row['task_key']]['partition'] != 'excluded'}


def inventory(root, task_count=80, min_submitted=30, reference_run=REFERENCE_RUN):
    root = Path(root)
    reference = json.loads((root/'runs'/identifier(reference_run)/'tasks.json').read_bytes())
    reference_ids = {t['instance_id'] for t in reference['tasks']}
    result = []
    for directory in sorted((root/'runs').iterdir()):
        if not (directory/'plan.json').exists():
            continue
        plan = json.loads((directory/'plan.json').read_bytes())
        tasks = json.loads((directory/'tasks.json').read_bytes())['tasks']
        if plan['operation'] != 'rollout' or len(tasks) != task_count:
            continue
        evidence = RunEvidence(root, directory.name).read()
        rows = eligible(evidence)
        submitted = sum(r['submission_status'] == 'submitted' for r in rows.values())
        same = {t['instance_id'] for t in tasks} == reference_ids
        result.append({'source_run_id': directory.name, 'task_count': len(tasks),
                       'same_task_ids_as_reference': same,
                       'registered_rollouts': len(evidence['rollouts']),
                       'eligible_rollouts': len(rows), 'submitted': submitted,
                       'include': same and submitted > min_submitted,
                       'reason': 'included' if same and submitted > min_submitted else
                       ('different_task_ids' if not same else 'submitted_not_strictly_above_threshold')})
    return result


def failure_kind(outcome, empty):
    """No generic infra_error inference; require explicit cause or empty bytes."""
    status = str(outcome.get('status', '')).lower().replace('_', '')
    failure = str(outcome.get('failure_type', '')).lower().replace('_', '')
    detail = '\n'.join(str(outcome.get(k) or '') for k in ('reason', 'failure_detail', 'error')).lower()
    patch = failure == 'invalidpatch' or status == 'invalidpatch' or any(x in detail for x in PATCH_MARKERS)
    # Empty artifact alone is insufficient when no evaluation outcome exists.
    if empty and (patch or (status == 'infraerror' and not failure and not detail.strip())):
        return 'empty_patch'
    return 'invalid_nonempty_patch' if patch else None


def ratio(numerator, denominator):
    return numerator/denominator if denominator else None


def summarize(rows):
    counts = Counter()
    for row in rows:
        counts['submitted'] += 1
        counts['evaluated'] += row['latest_evaluation'] is not None
        counts['unevaluated'] += row['latest_evaluation'] is None
        counts['empty_artifacts'] += row['patch_empty']
        counts['latest_patch_failures'] += row['latest_patch_failure'] is not None
        counts['latest_empty_patch_failures'] += row['latest_patch_failure'] == 'empty_patch'
        counts['latest_invalid_nonempty_patch_failures'] += row['latest_patch_failure'] == 'invalid_nonempty_patch'
        counts['ever_patch_failures'] += row['ever_patch_failure']
        counts['selected_patch_failures'] += row['selected_patch_failure'] is not None
        counts['latest_other_infra'] += bool(row['latest_evaluation'] and
            row['latest_evaluation']['outcome']['status'] in ('infra_error', 'ConnectionError')
            and not row['latest_patch_failure'])
    keys = ['submitted','evaluated','unevaluated','empty_artifacts','latest_patch_failures',
            'latest_empty_patch_failures','latest_invalid_nonempty_patch_failures',
            'ever_patch_failures','selected_patch_failures','latest_other_infra']
    return {**{k: counts[k] for k in keys},
            'latest_failure_over_submitted': ratio(counts['latest_patch_failures'], counts['submitted']),
            'latest_failure_over_evaluated': ratio(counts['latest_patch_failures'], counts['evaluated']),
            'ever_failure_over_submitted': ratio(counts['ever_patch_failures'], counts['submitted'])}


def check_audit(cfg, analysis_id):
    root = Path(cfg['project_root'])
    out = root/'action/outputs'/identifier(analysis_id)
    receipt = json.loads((out/'audit-receipt.json').read_bytes())
    for name, expected in receipt['files'].items():
        if digest(owned_path(out, name).read_bytes()) != expected:
            raise ValueError(f'Audit output checksum changed: {name}')
    audit = json.loads((out/'audit.json').read_bytes())
    for name, expected in audit['input_files'].items():
        if digest(owned_path(root, name).read_bytes()) != expected:
            raise ValueError(f'Frozen source input changed: {name}; use a new analysis ID')
    rows = [json.loads(line) for line in (out/'trajectories.jsonl').read_text().splitlines()]
    if len({(r['source_run_id'],r['rollout_id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate trajectory observation')
    summary = json.loads((out/'summary.json').read_bytes())
    if summarize(rows) != summary['pooled']:
        raise ValueError('Pooled summary disagrees with observations')
    for run, expected in summary['by_run'].items():
        if summarize([r for r in rows if r['source_run_id'] == run]) != expected:
            raise ValueError(f'Run summary disagrees: {run}')
    return {'verified': True, 'analysis_id': analysis_id, 'sources': len(summary['by_run']),
            'submitted_trajectories': len(rows), 'files': len(receipt['files'])}


@exclusive_artifact('action', 'analysis_id')
def audit_submissions(cfg, analysis_id, *, task_count=80, min_submitted=30,
                      reference_run=REFERENCE_RUN):
    root = Path(cfg['project_root']).resolve()
    out = root/'action/outputs'/identifier(analysis_id)
    if out.exists():
        raise FileExistsError(f'Use a new analysis ID: {out}')
    frozen = {}

    def freeze(path):
        path = owned_path(root, path)
        key = str(path.relative_to(root))
        raw = path.read_bytes()
        if key in frozen and frozen[key] != raw:
            raise ValueError(f'Input changed during audit: {key}')
        frozen[key] = raw
        return json.loads(raw)

    candidates = inventory(root, task_count, min_submitted, reference_run)
    runs = [r['source_run_id'] for r in candidates if r['include']]
    if not runs:
        raise ValueError('No rollout runs satisfy the cohort and submitted threshold')
    # Include plan/tasks/evidence for the entire 80-task candidate inventory.
    for entry in candidates:
        for filename in ('plan.json','tasks.json','evidence.json'):
            p = root/'runs'/entry['source_run_id']/filename
            if p.exists(): freeze(p)
    evidence = {run: json.loads(frozen[f'runs/{run}/evidence.json']) for run in runs}
    for run in runs:
        RunEvidence(root, run).validate(evidence[run])

    evaluations = {run: {} for run in runs}
    excluded = []
    for directory in sorted((root/'runs').iterdir()):
        ep = directory/'evidence.json'
        if not ep.exists(): continue
        # Only source-owned evaluations and explicitly linked re-evaluation runs.
        raw = ep.read_bytes()
        data = json.loads(raw)
        for eid, evaluation in data.get('evaluations', {}).items():
            source_run = directory.name if directory.name in runs else evaluation.get('provenance', {}).get('source_run_id')
            if source_run not in runs: continue
            freeze(ep)
            report = freeze(evaluation['report_path'])
            if digest(frozen[evaluation['report_path']]) != evaluation['report_sha256']:
                raise ValueError(f'Evaluation report checksum mismatch: {directory.name}/{eid}')
            original = None
            if report.get('source_report_sha256'):
                original_path = Path(evaluation['report_path']).parent/'source-report.json'
                original = freeze(original_path)
                if digest(frozen[str(original_path)]) != report['source_report_sha256']:
                    raise ValueError('Imported source report checksum mismatch')
            meta = original or report
            if meta.get('gold') is not False:
                excluded.append({'owner_run': directory.name, 'evaluation_id': eid, 'reason': 'gold_or_unknown_gold'})
                continue
            finished = meta.get('finished_at')
            if not finished:
                raise ValueError(f'No explicit evaluation timestamp: {directory.name}/{eid}')
            timestamp = datetime.fromisoformat(finished)
            if timestamp.tzinfo is None:
                raise ValueError('Evaluation timestamp must have timezone')
            for rid, outcome in evaluation['outcomes'].items():
                row = evidence[source_run]['rollouts'].get(rid)
                owner_row = data['rollouts'].get(rid)
                if not row or not owner_row: continue
                if any(row[k] != owner_row[k] for k in ('trajectory_sha256','patch_sha256')):
                    raise ValueError(f'Re-evaluation identity mismatch: {rid}')
                source_task = evidence[source_run]['tasks'][row['task_key']]
                owner_task = data['tasks'][owner_row['task_key']]
                if any(source_task[k] != owner_task[k] for k in ('instance_id','repo','base_commit')):
                    raise ValueError(f'Re-evaluation task binding mismatch: {rid}')
                if outcome['patch_sha256'] != row['patch_sha256']:
                    raise ValueError('Evaluation patch SHA mismatch')
                # Migration intentionally reduced fields; restore diagnostics only
                # from the content-addressed original report for this exact outcome.
                enriched = dict(outcome)
                if original is not None:
                    old = original['outcomes'][rid]
                    if any(old.get(k) != outcome.get(k) for k in ('patch_sha256','status','resolved')):
                        raise ValueError('Imported outcome disagrees with original')
                    enriched = {**old, **outcome}
                report_outcome = report['outcomes'][rid]
                if report_outcome != outcome:
                    raise ValueError('Evidence outcome disagrees with registered report')
                item = {'owner_run': directory.name, 'evaluation_id': eid,
                        'finished_at': finished, 'report_path': evaluation['report_path'],
                        'diagnostics_path': str(original_path) if original is not None else evaluation['report_path'],
                        'outcome': enriched}
                evaluations[source_run].setdefault(rid, []).append(item)

    rows, aliases, seen = [], [], {}
    for run in runs:
        selected_rows = eligible(evidence[run])
        for rid, row in sorted(selected_rows.items()):
            if row['submission_status'] != 'submitted': continue
            task = evidence[run]['tasks'][row['task_key']]
            patch = owned_path(root, row['patch_path']).read_bytes()
            if digest(patch) != row['patch_sha256']: raise ValueError('Patch checksum mismatch')
            frozen[row['patch_path']] = patch
            # Reuse the canonical action reader and its exact SHA validation.
            actions = read_actions(root, {'rollout_id': rid, **row})
            raw = owned_path(root, row['trajectory_path']).read_bytes()
            if digest(raw) != row['trajectory_sha256']: raise ValueError('Trajectory checksum mismatch')
            frozen[row['trajectory_path']] = raw
            identity = (task['instance_id'], row['trajectory_sha256'])
            if identity in seen:
                aliases.append({'source_run_id':run,'rollout_id':rid,'same_bytes_as':seen[identity]})
            else: seen[identity] = {'source_run_id':run,'rollout_id':rid}
            history = sorted(evaluations[run].get(rid, []),
                             key=lambda e: (datetime.fromisoformat(e['finished_at']),e['owner_run'],e['evaluation_id']))
            if len(history)>1 and history[-1]['finished_at']==history[-2]['finished_at'] and history[-1]['outcome']!=history[-2]['outcome']:
                raise ValueError('Conflicting evaluations share latest timestamp')
            empty = not patch.strip()
            latest = history[-1] if history else None
            selected = next((e for e in history if e['owner_run']==run and e['evaluation_id']==row.get('selected_evaluation_id')),None)
            rows.append({'source_run_id':run,'rollout_id':rid,'instance_id':task['instance_id'],
                         'task_key':row['task_key'],'attempt_id':row['attempt_id'],
                         'repetition_id':row.get('repetition_id'),'trajectory_path':row['trajectory_path'],
                         'trajectory_sha256':row['trajectory_sha256'],'patch_path':row['patch_path'],
                         'patch_sha256':row['patch_sha256'],'patch_bytes':len(patch),'patch_empty':empty,
                         'assistant_steps':len(actions),'latest_evaluation':latest,
                         'latest_patch_failure':failure_kind(latest['outcome'],empty) if latest else None,
                         'ever_patch_failure':any(failure_kind(e['outcome'],empty) is not None for e in history),
                         'selected_patch_failure':failure_kind(selected['outcome'],empty) if selected else None,
                         'evaluation_history':history})
    if aliases:
        raise ValueError(f'Identical trajectory bytes would be counted twice: {aliases}')

    summary = {'pooled':summarize(rows),'by_run':{run:summarize([r for r in rows if r['source_run_id']==run]) for run in runs}}
    samples = []
    # Stable examples cover every run plus review/test-edit interleaving and bad submissions.
    preferred = ['r-d2bd1a35cb0f6baee93935b84b1b7c6d','r-fcb0568e4597df8cf1fa20e3aec6e1c8',
                 'r-1ba2c626e15e0f11579184c7a35c14dc','r-b5bea03acff6604ef1dc227df049f8b2',
                 'r-ec6db837637430b0bf33afcbc590af51','r-3ad340e10f12890f369e33ad854533ae',
                 'r-3ebc34567aaffe8886c2a8403a3fd7a5','r-77edf194284e3f8dc003292c0047f73c']
    for row in rows:
        if row['rollout_id'] not in preferred: continue
        messages = read_messages(root,row)
        steps=[]
        for mi,m in enumerate(messages):
            if m.get('role')!='assistant':continue
            observations=[]
            for following in messages[mi+1:]:
                if following.get('role')=='assistant':break
                if following.get('role') in ('tool','exit'):observations.append(str(following.get('content') or '')[:1000])
            # Canonical action parser supplies calls; extra.actions is a verified
            # human-readable supplement, never a separately counted action.
            steps.append({'step_id':len(steps),'message_index':mi,
                          'content':m.get('content'),'reasoning':m.get('reasoning_content') or
                          (m.get('provider_specific_fields') or {}).get('reasoning'),
                          'tool_calls':m.get('tool_calls') or [],
                          'commands':[a.get('command') for a in m.get('extra',{}).get('actions',[])],
                          'observations_excerpt':observations})
        samples.append({**{k:row[k] for k in ('source_run_id','rollout_id','instance_id','trajectory_sha256','assistant_steps','latest_patch_failure')},'steps':steps})

    out.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='.submission-audit-',dir=out.parent) as temporary:
        stage=Path(temporary)
        def save(name,value):
            p=stage/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        audit={'schema_version':1,'audit_type':VERSION,'analysis_id':analysis_id,
               'created_at':datetime.now(timezone.utc).isoformat(),
               'task_count':task_count,'submitted_threshold_exclusive':min_submitted,
               'reference_run':reference_run,'source_runs':runs,
               'observation_unit':'unique registered eligible submitted canonical trajectory; all attempts, not only final task samples',
               'evaluation_policy':'latest timezone-aware finished_at, explicit source link, non-gold, exact patch/trajectory/task identity',
               'primary_denominator':'all submitted; separately report evaluated submitted; unknown is not success',
               'input_files':{k:digest(v) for k,v in frozen.items()},'excluded_evaluations':excluded,
               'review_status':'proposal only; no automatic review-boundary labels or population length estimates'}
        save('audit.json',audit);save('scope.json',candidates);save('summary.json',summary)
        for name,values in [('trajectories.jsonl',rows),('review-samples.jsonl',samples)]:
            (stage/name).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in values),encoding='utf-8')
        columns=['source_run_id','rollout_id','instance_id','assistant_steps','patch_bytes','patch_empty','latest_patch_failure','ever_patch_failure']
        with (stage/'trajectories.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows({k:r[k] for k in columns} for r in rows)
        for path,raw in frozen.items():
            if path.endswith('/evidence.json') or path.endswith('/summary.json') or path.endswith('/source-report.json') or path.endswith('/plan.json') or path.endswith('/tasks.json'):
                target=stage/'inputs'/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        for name in CODE:
            p=stage/'code'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((root/name).read_bytes())
        report=['# Submitted 空/无效 patch 统计','',f'生成时间（UTC）：{audit["created_at"]}',
                '',f'范围：同一组 {task_count} tasks、rollout operation、eligible submitted > {min_submitted}；共 {len(runs)} runs。',
                '主分母为全部 submitted；未评测保留未知，并另报已评测分母。重评只更新证据，不增加轨迹。',
                '', '| Run | Submitted | 已评测 | 空 patch | 非空无效 patch | 最新失败/全部 submitted | 最新失败/已评测 | 历史任一次 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
        def fmt(value):return 'N/A' if value is None else f'{value:.2%}'
        for run,s in [*summary['by_run'].items(),('TOTAL',summary['pooled'])]:
            report.append(f'| {run} | {s["submitted"]} | {s["evaluated"]} | {s["latest_empty_patch_failures"]} | {s["latest_invalid_nonempty_patch_failures"]} | {s["latest_patch_failures"]}/{s["submitted"]} ({fmt(s["latest_failure_over_submitted"])}) | {fmt(s["latest_failure_over_evaluated"])} | {s["ever_patch_failures"]} |')
        report+=['','## 口径',
                 '- 兼容旧 infra_error、显式 invalid_patch/invalidpatch，以及新版 completed + failure_type=invalid_patch。',
                 '- 原因优先使用 SHA 校验后的原评测报告；迁移删减诊断字段时读取同目录 content-addressed source-report.json。',
                 '- 不把普通 infra_error/network/dependency/timeout 全部归为空/无效 patch；无评测的空 artifact 也不充当已评测失败。',
                 '- 包括已登记 submitted 的所有 attempt；不同 attempt 不是同一 task 的重复评测。相同轨迹字节重复计数会拒绝。',
                 '- invalid_nonempty_patch 表示评测记录的 patch 应用失败，不等价于“代码逻辑错误”或已证明 diff 格式错误。',
                 '- 最新评测用明确 finished_at 排序；不以文件 mtime 或 selected_evaluation_id 冒充最新。',
                 '- review-samples.jsonl 是抽样阅读材料，step_id 从 0 开始；尚未执行总体复查长度自动统计。','']
        (stage/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
        # Reject a mixed-time snapshot if any frozen evidence changed during work.
        for path,raw in frozen.items():
            if owned_path(root,path).read_bytes()!=raw:raise ValueError(f'Input changed during audit: {path}')
        save('audit-receipt.json',{'schema_version':1,'audit_type':VERSION,
                                  'files':{str(p.relative_to(stage)):digest(p.read_bytes()) for p in sorted(stage.rglob('*')) if p.is_file()}})
        stage.rename(out)
    return {'output':str(out),'source_runs':runs,'summary':summary,'check':check_audit(cfg,analysis_id)}

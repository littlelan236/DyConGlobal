"""Versioned, retrospective review episodes; counts are rule estimates."""
from __future__ import annotations
import fnmatch
import posixpath
import re
import shlex
from action.analyzers.review_effects import Effects

VERSION = 'review-episodes-v1'
SETTINGS = {'unit':'canonical assistant message', 'temporal_policy':'retrospective candidate-change episodes',
            'minimum_evidence_channels':2,'mixed_step_policy':'exclude_from_strict',
            'no_edit_policy':'separate_unknown','repeat_policy':'count_actual_steps'}
VERIFY = re.compile(r'\b(verif\w*|review|double.check|regression|edge cases?|robust|make sure|ensure|reproduc\w*|run\w* (?:the |existing |all |some )?tests?)\b',re.I)
READY = re.compile(r'\b(fix|changes?|implementation|patch)\s+(?:is |has been |was )?(?:applied|works|working|correct|complete|looks)',re.I)
TEST = re.compile(r'\b(pytest|unittest|runtests|tox|nox|assert|test_\w*|repro\w*)\b',re.I)
ALREADY = re.compile(r'(fix (?:has |is |was )?already|already (?:been )?(?:applied|fixed|implemented)|nothing to (?:change|do))',re.I)
CRITICAL = {'shell_parse_error','inplace_target_unknown',
            'python_dynamic_write_path','dynamic_shell_path','python_parse_error'}


def diff_paths(text):
    result=[]
    for line in text.splitlines():
        if not line.startswith('diff --git '):continue
        try:words=shlex.split(line)
        except ValueError:continue
        if len(words)>=4:
            p=words[3] if words[3]!='b/dev/null' else words[2]
            if p.startswith(('a/','b/')):result.append(p[2:])
    return sorted(set(result))


def is_worktree_diff(command):
    """Reject historical comparisons; a diff header alone is insufficient."""
    if not re.search(r'\bgit(?:\s+-C\s+\S+)?\s+diff\b',command):return False
    tail=re.split(r'\bdiff\b',command,maxsplit=1)[-1]
    if re.search(r'\b[0-9a-f]{6,40}\b|HEAD[~^]|\.\.',tail):return False
    return len(re.findall(r'\bHEAD\b',tail))<=1


def _observed_rc(step):
    for obs in step.get('observations',[]):
        rc=obs.get('returncode')
        if isinstance(rc,int):return rc
        match=re.search(r'<returncode>(-?\d+)</returncode>',obs.get('content',''))
        if match:return int(match.group(1))
    return None


def analyze(item):
    source=item['source'];steps=item['steps'];cwd=source.get('classifier_cwd') or '/testbed'
    cwd=posixpath.normpath(cwd);engine=Effects(cwd);features=[]
    for s in steps:
        action=s['action'];command=action.get('command');effects=engine.parse(command)
        text=(s.get('reasoning') or '')+'\n'+(s.get('content') or '')
        features.append({**s,'effects':effects,'review_language':bool(VERIFY.search(text) or READY.search(text)),
                         'test_behavior':bool(TEST.search(command or '')),
                         'already_fixed_claim':bool(ALREADY.search(text)),'returncode':_observed_rc(s)})
    terminal=None
    if features:
        last=features[-1]
        actual_exit=any(o.get('role')=='exit' for o in last.get('observations',[]))
        marker='protocol_marker_emission' in last['action'].get('rules',[])
        if source.get('termination_reason')=='Submitted' and (actual_exit or marker):terminal=last['step_id']
    end=terminal if terminal is not None else len(features)
    paths=list(item.get('delivery_paths') or []);path_source='final_patch_diff' if paths else 'unavailable'
    if not paths:
        for f in features:
            cmd=f['action'].get('command') or ''
            if is_worktree_diff(cmd):
                found=diff_paths('\n'.join(o.get('content','') for o in f.get('observations',[])))
                if found:paths=found;path_source='observed_worktree_diff'
    finals={posixpath.normpath(posixpath.join(cwd,p)) for p in paths}
    prefixes={p.split('/')[0] for p in paths if '/' in p and p.split('/')[0] not in ('tests','testing','test')}
    created={p for f in features if 'explicit_file_construction' in f['action'].get('rules',[]) for p in f['effects']['writes']}
    removed=[p for f in features for p in f['effects']['deletes']]
    def temporary(p):
        if p in finals:return False
        if p.startswith('/tmp/'):return True
        if any(c in p for c in '*?['):
            matches=[c for c in created if fnmatch.fnmatch(c,p)]
            if matches and not any(fnmatch.fnmatch(f,p) for f in finals):
                return all(temporary(c) for c in matches)
        rel=p.removeprefix(cwd+'/')
        # Root test scripts are probable support, not a proof of untracked status.
        if p in created and '/' not in rel and re.match(r'(test_|repro|fix(?:_|[0-9]*\.))',rel):return True
        return (p in created and ('/' not in rel or rel.split('/')[0] in ('tests','testing','test')) and
                any(fnmatch.fnmatch(p,d) or p.startswith(d.rstrip('/')+'/') for d in removed))
    def target(p):
        if p in finals:return True
        if not p.startswith(cwd+'/'):return False
        rel=p[len(cwd)+1:]
        return not temporary(p) and (rel.split('/')[0] in prefixes or bool(prefixes) and '/' not in rel and not rel.endswith(('.txt','.diff','.patch','.log','.png','.fits','.bak')))
    for f in features:
        effect=f['effects'];command=f['action'].get('command') or '';label=f['action']['label']
        writes=effect['writes']+effect['deletes']
        f['target_write_paths']=[p for p in writes if target(p)]
        f['temporary_write_paths']=[p for p in effect['writes'] if temporary(p)]
        f['candidate_change']=bool(f['target_write_paths'] or effect['state_changes'])
        # Only a simple, explicitly failed command can be ruled out; a compound
        # command may have modified files before its later subcommand failed.
        obs='\n'.join(o.get('content','') for o in f.get('observations',[]))
        failed=(f['returncode'] is not None and f['returncode']!=0 and
                not re.search(r'&&|;|\n|\|\||\|',command) and
                re.search(r'No such file|can.t read|command not found|SyntaxError',obs,re.I))
        f['failed_write_ignored']=bool(f['candidate_change'] and failed)
        if failed:f['candidate_change']=False
        f['unresolved_edit']=bool('EDIT' in f['action'].get('components',[]) and
            not writes and not effect['state_changes'] or CRITICAL.intersection(effect['warnings']))
        f['target_inspection']=bool(label=='INSPECT' and any(p in command or posixpath.basename(p) in command for p in paths))
        f['worktree_inspection']=bool(label=='INSPECT' and re.search(r'\bgit\b.*\b(diff|status)\b',command) and
                                           (is_worktree_diff(command) or re.search(r'\bstatus\b',command)))
        f['behavior_channel']=bool(label in ('INSPECT','RUN') or f['temporary_write_paths'] or label=='ENV')
        f['object_channel']=bool(f['test_behavior'] or f['target_inspection'] or f['worktree_inspection'])
        f['evidence_channels']=sum((f['behavior_channel'],f['object_channel'],f['review_language']))
        f['supports_review']=bool(f['evidence_channels']>=SETTINGS['minimum_evidence_channels'] and
            f['behavior_channel'] and label not in ('DELIVER','CLEAN','OTHER') and not f['unresolved_edit'])
        f['phase']='DEVELOP_OR_DIAGNOSE';f['episode_id']=None
    candidate=False;active=None;episodes=[];change_steps=[];rework=0
    for i,f in enumerate(features):
        sid=f['step_id'];label=f['action']['label'];rules=f['action'].get('rules',[])
        if sid==terminal:f['phase']='SUBMIT';continue
        if f['candidate_change']:
            change_steps.append(sid)
            if active is not None:episodes[active]['end_step_exclusive']=sid;rework+=1
            active=None;candidate=True;f['phase']='CANDIDATE_CHANGE';continue
        if f['unresolved_edit']:
            f['phase']='UNKNOWN';
            if active is not None:
                f['episode_id']=active
                episodes[active]['uncertain_steps'].append(sid)
            continue
        if not candidate:continue
        if 'protocol_marker_emission' in rules:f['phase']='SUBMIT_RETRY'
        elif label=='DELIVER':f['phase']='PACKAGE'
        elif label=='CLEAN':f['phase']='CLEANUP'
        else:
            if active is None and f['supports_review']:
                active=len(episodes);episodes.append({'episode_id':active,'start_step':sid,
                    'candidate_change_step':change_steps[-1],'end_step_exclusive':end,'review_steps':[],'uncertain_steps':[]})
            if active is not None:
                if f['supports_review']:f['phase']='VERIFY_SETUP' if f['temporary_write_paths'] or label=='ENV' else 'VERIFY'
                else:f['phase']='REVIEW_UNCERTAIN'
        if active is not None:
            f['episode_id']=active
            if f['phase'] in ('VERIFY','VERIFY_SETUP'):episodes[active]['review_steps'].append(sid)
            if f['phase'] in ('UNKNOWN','REVIEW_UNCERTAIN'):episodes[active]['uncertain_steps'].append(sid)
    last_change=change_steps[-1] if change_steps else None
    final=next((e for e in reversed(episodes) if e['candidate_change_step']==last_change),None)
    reasons=[]
    if terminal is None:reasons.append('terminal_submission_not_observed')
    elif features[terminal]['candidate_change']:reasons.append('submission_mutates_candidate')
    if last_change is None:reasons.append('no_candidate_change_detected')
    late_unknown=[f['step_id'] for f in features[:end] if f['unresolved_edit'] and (last_change is None or f['step_id']>last_change)]
    if late_unknown:reasons.append('unresolved_late_mutation')
    finite=terminal is not None and last_change is not None and not late_unknown and 'submission_mutates_candidate' not in reasons
    if finite and final is None and any(f['phase'] not in ('CLEANUP','PACKAGE','SUBMIT_RETRY') for f in features[last_change+1:end]):
        reasons.append('no_supported_review_anchor');finite=False
    net=len(final['review_steps']) if final else 0
    span=end-final['start_step'] if final else 0
    loose=net+len(final['uncertain_steps']) if final else 0
    warnings=[]
    if path_source!='final_patch_diff':warnings.append('delivery_paths_'+path_source)
    relevant=features[(last_change+1) if last_change is not None else 0:end]
    if any(f['effects']['warnings'] for f in relevant):warnings.append('unresolved_static_effects')
    if final and final['uncertain_steps']:warnings.append('uncertain_review_steps')
    no_edit=[f['step_id'] for f in features[:end] if last_change is None and f['supports_review']]
    return {'source_run_id':source['source_run_id'],'rollout_id':source['rollout_id'],'instance_id':source['instance_id'],
            'assistant_steps':len(features),'terminal_submit_step':terminal,'last_candidate_change_step':last_change,
            'candidate_stable_step':last_change+1 if last_change is not None else None,
            'final_review_start_step':final['start_step'] if final and finite else None,
            'review_steps':net if finite else None,'review_steps_permissive':loose if finite else None,
            'review_span':span if finite else None,'review_step_share':net/len(features) if finite and features else None,
            'all_episode_review_steps':sum(len(e['review_steps']) for e in episodes),
            'episode_count':len(episodes),'rework_after_review':rework,'candidate_change_steps':change_steps,
            'status':'estimated_with_warnings' if finite and warnings else 'estimated' if finite else 'unknown',
            'unknown_reasons':reasons,'warnings':warnings,'late_unknown_steps':late_unknown,
            'no_edit_review_candidate_steps':no_edit,'delivery_paths':paths,'delivery_path_source':path_source,
            'episodes':episodes,'step_features':features}

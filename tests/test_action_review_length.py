"""Rule and artifact contracts; semantic accuracy still requires human labels."""
import hashlib
import json
from pathlib import Path
import pytest
from action.analyzers.review_effects import Effects, python_writes
from action.analyzers.review_phases import analyze, is_worktree_diff, diff_paths
from action.analyzers.taxonomy import Classifier
from action.evaluation.review_statistics import distribution, summarize
from action.analyzers import review_length as service

MARKER = 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
EDIT = "sed -i 's/old/new/' /testbed/pkg/a.py"
DIFF = 'diff --git a/pkg/a.py b/pkg/a.py\n--- a/pkg/a.py\n+++ b/pkg/a.py\n'


def item(commands, reasons=None, observations=None, paths=None):
    classifier=Classifier('/testbed');steps=[]
    for i,command in enumerate(commands):
        action={'command':command,**classifier.classify(command)}
        obs=(observations or {}).get(i,[])
        steps.append({'step_id':i,'message_index':2*i,'action':action,
                      'reasoning':(reasons or {}).get(i,''),'content':'','observations':obs})
    return {'source':{'source_run_id':'source','rollout_id':'r-one','instance_id':'task-1',
                      'classifier_cwd':'/testbed','termination_reason':'Submitted'},
            'steps':steps,'delivery_paths':['pkg/a.py'] if paths is None else paths}


def test_clean_episode_excludes_package_and_submit():
    r=analyze(item([EDIT,'cat /testbed/pkg/a.py','python -m pytest tests/test_a.py',
                    'git diff > patch.txt',MARKER]))
    assert (r['review_steps'],r['review_span'],r['final_review_start_step'])==(2,3,1)
    assert r['episode_count']==1 and r['rework_after_review']==0


def test_temporary_test_write_does_not_reset_episode():
    command="cat > /testbed/test_fix.py <<'PY'\nassert 1 == 1\nPY\npython /testbed/test_fix.py"
    r=analyze(item([EDIT,'cat /testbed/pkg/a.py',command,MARKER],{2:'Verify the fix with edge cases'}))
    assert r['candidate_change_steps']==[0]
    assert r['review_steps']==2
    assert r['step_features'][2]['phase']=='VERIFY_SETUP'


def test_helper_in_tmp_resolves_real_production_write():
    command="cat > /tmp/fix.py <<'PY'\nwith open('/testbed/pkg/a.py','w') as f:f.write('new')\nPY\npython /tmp/fix.py"
    r=analyze(item([EDIT,'cat /testbed/pkg/a.py',command,'cat /testbed/pkg/a.py',MARKER]))
    assert r['candidate_change_steps']==[0,2]
    assert r['review_steps']==1 and r['rework_after_review']==1
    assert r['all_episode_review_steps']==2


def test_git_restore_restarts_final_review():
    r=analyze(item([EDIT,'cat /testbed/pkg/a.py','git restore pkg/b.py',
                    'python -m pytest tests/test_a.py',MARKER]))
    assert r['last_candidate_change_step']==2 and r['final_review_start_step']==3
    assert r['review_steps']==1


def test_no_edit_is_unknown_even_with_many_tests():
    r=analyze(item(['cat /testbed/pkg/a.py','python -m pytest tests/test_a.py',MARKER]))
    assert r['review_steps'] is None and r['review_span'] is None
    assert 'no_candidate_change_detected' in r['unknown_reasons']
    assert r['no_edit_review_candidate_steps']==[0,1]


def test_known_absence_of_review_is_zero():
    r=analyze(item([EDIT,'git diff > patch.txt',MARKER]))
    assert r['review_steps']==0 and r['review_span']==0
    assert r['status']=='estimated'


def test_cleanup_of_root_helper_does_not_reset_target_state():
    helper="cat > /testbed/update_docstring.py <<'PY'\nopen('/testbed/pkg/a.py','w').write('new')\nPY\npython /testbed/update_docstring.py"
    r=analyze(item([helper,'cat /testbed/pkg/a.py','rm /testbed/update_docstring.py',MARKER]))
    assert r['candidate_change_steps']==[0]
    assert r['review_steps']==1 and r['review_span']==2


def test_cleanup_glob_resolves_constructed_reproduction_scripts():
    helper="cat > /testbed/repro_case.py <<'PY'\nassert True\nPY\npython /testbed/repro_case.py"
    r=analyze(item([EDIT,helper,'cat /testbed/pkg/a.py','rm /testbed/repro*.py',MARKER],
                   {1:'Verify the fix using a reproducer'}))
    assert r['candidate_change_steps']==[0]
    assert r['review_steps']==2 and r['review_span']==3


def test_failed_submission_is_not_terminal_boundary():
    r=analyze(item([EDIT,'cat /testbed/pkg/a.py',MARKER+' && cat missing.patch',
                    'git diff > patch.txt',MARKER],observations={2:[{'role':'tool','content':'<returncode>1</returncode>\nNo such file'}]}))
    assert r['terminal_submit_step']==4 and r['review_span']==3
    assert r['review_steps']==1 and r['step_features'][2]['phase']=='SUBMIT_RETRY'


def test_literal_marker_in_comment_is_not_submission():
    r=analyze(item([EDIT,'cat /testbed/pkg/a.py # COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT']))
    assert r['review_steps'] is None
    assert 'terminal_submission_not_observed' in r['unknown_reasons']


def test_canonical_exit_confirms_terminal_assistant():
    r=analyze(item([EDIT,'cat /testbed/pkg/a.py','cat patch.txt'],
                   observations={2:[{'role':'exit','content':DIFF}]}))
    assert r['terminal_submit_step']==2 and r['review_steps']==1


def test_mixed_target_edit_and_test_is_not_strict_review():
    r=analyze(item([EDIT+' && python -m pytest tests/test_a.py',
                    'cat /testbed/pkg/a.py',MARKER]))
    assert r['step_features'][0]['phase']=='CANDIDATE_CHANGE'
    assert r['review_steps']==1


def test_dynamic_late_write_is_unknown():
    command='python -c "import os; f=open(os.environ[\'TARGET\'],\'w\'); f.write(\'x\')"'
    r=analyze(item([EDIT,'cat /testbed/pkg/a.py',command,MARKER]))
    assert r['review_steps'] is None and 'unresolved_late_mutation' in r['unknown_reasons']


def test_simple_failed_write_does_not_reset_candidate():
    r=analyze(item([EDIT,'cat /testbed/pkg/a.py',
                    "sed -i 's/old/new/' /testbed/pkg/missing.py",'cat /testbed/pkg/a.py',MARKER],
                   observations={2:[{'role':'tool','content':'<returncode>2</returncode>\nsed: can\'t read: No such file'}]}))
    assert r['candidate_change_steps']==[0]
    assert r['step_features'][2]['failed_write_ignored']


def test_diff_observation_recovers_missing_patch_paths():
    r=analyze(item([EDIT,'git diff pkg/a.py',MARKER],paths=[],
                   observations={1:[{'role':'tool','content':DIFF}]}))
    assert r['review_steps']==1 and r['delivery_path_source']=='observed_worktree_diff'


@pytest.mark.parametrize('command,expected',[
    ('git diff pkg/a.py',True),('git -C /testbed diff HEAD -- pkg/a.py',True),
    ('git diff HEAD 1234abc -- pkg/a.py',False),('git diff HEAD~2 -- pkg/a.py',False),
    ('git show abc123 -- pkg/a.py',False)])
def test_historical_diff_rejected(command,expected):
    assert is_worktree_diff(command)==expected


def test_python_helper_lifecycle():
    e=Effects();e.parse("cat > /tmp/fix.py <<'PY'\nopen('/testbed/pkg/a.py','w').write('x')\nPY")
    assert '/testbed/pkg/a.py' in e.parse('python /tmp/fix.py')['writes']
    e.parse('rm /tmp/fix.py')
    assert 'script_effects_unavailable' in e.parse('python /tmp/fix.py')['warnings']


def test_pathlib_literal_write_and_dynamic_execution():
    assert python_writes("from pathlib import Path\np='/testbed/pkg/a.py'\nPath(p).write_text('x')")[0]==['/testbed/pkg/a.py']
    assert 'nested_execution_not_resolved' in python_writes("import subprocess\nsubprocess.run(['python','x.py'])")[1]


def test_no_command_is_executed(tmp_path):
    marker=tmp_path/'never-created'
    Effects().parse(f'echo x > {marker}')
    assert not marker.exists()


def test_statistics_exclude_unknown_and_include_real_zero():
    unknown=analyze(item(['cat /testbed/pkg/a.py',MARKER]))
    zero=analyze(item([EDIT,MARKER]))
    one=analyze(item([EDIT,'cat /testbed/pkg/a.py',MARKER]))
    for r in [unknown,zero,one]:r['evaluation_group']='resolved'
    summary=summarize([unknown,zero,one])['pooled']
    assert summary['submitted']==3 and summary['estimated']==2 and summary['unknown']==1
    assert summary['review_steps']['mean']==.5 and summary['review_steps']['sum']==1
    assert distribution([])['mean'] is None


def test_adapter_message_identity_and_observations():
    messages=[{'role':'assistant','tool_calls':[{'function':{'name':'bash','arguments':'{"command":"cat a.py"}'}}]},
              {'role':'tool','content':'ok','extra':{'returncode':0}}]
    labels=[{'step_id':0,'message_index':0,'command':'cat a.py'}]
    assert service.make_steps(messages,labels)[0]['observations'][0]['returncode']==0
    with pytest.raises(ValueError,match='coverage'):
        service.make_steps(messages,[])


def artifact_fixture(tmp_path,monkeypatch):
    root=tmp_path/'project';root.mkdir()
    actual=Path(service.__file__).resolve().parents[2]
    for name in service.CODE_FILES:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((actual/name).read_bytes())
    commands=[EDIT,'cat /testbed/pkg/a.py',MARKER];classifier=Classifier('/testbed');messages=[];labels=[]
    for i,cmd in enumerate(commands):
        messages.append({'role':'assistant','reasoning_content':'verify the fix',
                         'tool_calls':[{'function':{'name':'bash','arguments':json.dumps({'command':cmd})}}]})
        labels.append({'step_id':i,'message_index':2*i,'command':cmd,'source_run_id':'source','rollout_id':'r-one',**classifier.classify(cmd)})
        messages.append({'role':'exit' if i==2 else 'tool','content':DIFF if i==2 else 'ok'})
    raw=json.dumps({'messages':messages}).encode();trajectory=root/'runs/source/raw.json';trajectory.parent.mkdir(parents=True);trajectory.write_bytes(raw)
    patch=root/'runs/source/patch.diff';patch.write_text(DIFF)
    source={'source_run_id':'source','rollout_id':'r-one','instance_id':'task-1','classifier_cwd':'/testbed',
            'termination_reason':'Submitted','trajectory_path':'runs/source/raw.json','trajectory_sha256':hashlib.sha256(raw).hexdigest()}
    row={**source,'patch_path':'runs/source/patch.diff','patch_sha256':service.sha256(patch),'latest_evaluation':None}
    audit=root/'action/outputs/audit';audit.mkdir(parents=True)
    for filename,value in [('audit.json',{}),('summary.json',{}),('scope.json',[])]:
        (audit/filename).write_text(json.dumps(value))
    (audit/'trajectories.jsonl').write_text(json.dumps(row)+'\n')
    (audit/'audit-receipt.json').write_text(json.dumps({'files':{p.name:service.sha256(p) for p in audit.iterdir()}}))
    action=root/'action/outputs/actions';action.mkdir(parents=True);(action/'receipt.json').write_text('{}')
    parent={'code':{'action/analyzers/taxonomy.py':service.sha256(root/'action/analyzers/taxonomy.py')}}
    packet=(action,parent,{'sources':[source],'duplicate_trajectory_aliases':[]},labels,{}, {})
    monkeypatch.setattr(service,'load_local',lambda *_:packet)
    return root


def test_end_to_end_atomic_artifact_and_refuse_overwrite(tmp_path,monkeypatch):
    root=artifact_fixture(tmp_path,monkeypatch)
    result=service.run_review_length({'project_root':str(root)},'review',audit_id='audit',action_id='actions')
    assert result['check']['verified']
    assert result['summary']['pooled']['review_steps']['mean']==1
    out=root/'action/outputs/review'
    assert len((out/'steps.jsonl').read_text().splitlines())==3
    with pytest.raises(FileExistsError):
        service.run_review_length({'project_root':str(root)},'review',audit_id='audit',action_id='actions')
    with (out/'steps.jsonl').open('a') as stream:stream.write(' ')
    with pytest.raises(ValueError,match='checksum'):
        service.check_review_length(root,'review')


def test_canonical_patch_change_is_rejected(tmp_path,monkeypatch):
    root=artifact_fixture(tmp_path,monkeypatch)
    (root/'runs/source/patch.diff').write_text('changed')
    with pytest.raises(ValueError,match='patch SHA'):
        service.run_review_length({'project_root':str(root)},'review',audit_id='audit',action_id='actions')
    assert not (root/'action/outputs/review').exists()

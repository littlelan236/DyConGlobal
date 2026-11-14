import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture
def verifier(tmp_path, monkeypatch):
    source=Path(__file__).parents[1]/'catalog/verifier_patches/simpo_test_outputs.py'
    spec=importlib.util.spec_from_file_location('fixed_simpo_verifier',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reference=tmp_path/'protected/loss.npz';reference.parent.mkdir();np.savez(reference,losses=np.array([1.,2.,3.]))
    output=tmp_path/'submission/loss.npz';output.parent.mkdir();np.savez(output,losses=np.array([1.,2.,3.]))
    environment=output.with_name('python_info.txt');environment.write_text('Python 3.10.22')
    repository=tmp_path/'SimPO';(repository/'unit_test').mkdir(parents=True)
    stage=tmp_path/'logs/simpo-stages.json'
    for key,value in {'REFERENCE':reference,'REFERENCE_SHA256':hashlib.sha256(reference.read_bytes()).hexdigest(),
                      'OUTPUT':output,'ENVIRONMENT':environment,'REPOSITORY':repository,'STAGES':stage}.items():
        monkeypatch.setattr(module,key,value)
    def script(mode='correct'):
        values='[9.,9.,9.]' if mode=='wrong' else '[1.,2.,3.]'
        text=(f'import numpy as np; np.savez({str(output)!r},losses=np.array({values}))' if mode in {'correct','wrong'}
              else 'raise SystemExit(3)' if mode=='failed' else 'pass')
        (repository/'unit_test/unit_test_1.py').write_text(text)
    script()
    return module,script


def test_correct_original_and_regenerated_outputs_pass(verifier):
    m,_=verifier;m.test_outputs();r=json.loads(m.STAGES.read_text())
    assert r['outcome']=='passed'
    assert [s['phase'] for s in r['stages']]==['reference','submission','rerun','rerun_comparison']
    assert r['stages'][0]['sha256']==m.REFERENCE_SHA256


@pytest.mark.parametrize('mode',['missing_output','missing_environment','wrong_environment','missing_key','shape','nan','wrong_values','corrupt_output'])
def test_invalid_original_submission_fails_without_rerun(verifier,mode):
    m,_=verifier
    if mode=='missing_output':m.OUTPUT.unlink()
    elif mode=='missing_environment':m.ENVIRONMENT.unlink()
    elif mode=='wrong_environment':m.ENVIRONMENT.write_text('Python 3.12')
    elif mode=='corrupt_output':m.OUTPUT.write_bytes(b'bad zip')
    elif mode=='missing_key':np.savez(m.OUTPUT,other=[1,2,3])
    else:np.savez(m.OUTPUT,losses={'shape':[1,2],'nan':[1,np.nan,3],'wrong_values':[9,9,9]}[mode])
    with pytest.raises(AssertionError):m.test_outputs()
    r=json.loads(m.STAGES.read_text());assert r['outcome']=='agent_failure'
    assert all(s['phase']!='rerun' for s in r['stages'])


@pytest.mark.parametrize('mode',['wrong','failed','no_output'])
def test_rerun_is_reopened_and_checked(verifier,mode):
    m,script=verifier;script(mode)
    with pytest.raises(AssertionError):m.test_outputs()
    r=json.loads(m.STAGES.read_text());assert r['outcome']=='agent_failure'
    assert 'rerun_comparison' not in [s['phase'] for s in r['stages']]


@pytest.mark.parametrize('mode',['missing','hash','key','nan'])
@pytest.mark.parametrize('missing_submission',[False,True])
def test_broken_reference_is_always_verifier_fault(verifier,mode,missing_submission):
    m,_=verifier
    if missing_submission:m.OUTPUT.unlink()
    if mode=='missing':m.REFERENCE.unlink()
    elif mode=='hash':m.REFERENCE.write_bytes(b'corrupt')
    else:
        np.savez(m.REFERENCE,**({'other':[1,2,3]} if mode=='key' else {'losses':[1,np.nan,3]}))
        m.REFERENCE_SHA256=hashlib.sha256(m.REFERENCE.read_bytes()).hexdigest()
    with pytest.raises(m.SimPOVerifierError):m.test_outputs()
    assert json.loads(m.STAGES.read_text())['outcome']=='verifier_error'


def test_cwd_cannot_replace_protected_reference(verifier,tmp_path,monkeypatch):
    m,_=verifier;rogue=tmp_path/'rogue';rogue.mkdir();np.savez(rogue/'loss.npz',losses=[9.,9.,9.]);monkeypatch.chdir(rogue)
    np.savez(m.OUTPUT,losses=[9.,9.,9.])
    with pytest.raises(AssertionError,match='Loss mismatch'):m.test_outputs()

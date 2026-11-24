"""Real 9B bounded A/B via formal CLI, with a sealed engineering fixture.

This is a short engineering workload, not the historical production sequence.
Every training invocation has a new immutable attempt and a fresh process.
"""
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import pytest
import torch
import yaml

pytestmark = pytest.mark.skipif(os.environ.get('DYCON_HOST_OFFLOAD_9B') != '1',
                               reason='explicit real 9B GPU acceptance')


def test_formal_9b_six_updates_ab_and_exact_resume(tmp_path):
    from core.io import hash_json, read_json, write_json
    from RL.artifacts import base_identity
    from RL.dataset.artifacts import seal
    from RL.dataset.schema import spans
    from tests.test_grpo_prepared import ROOT, dataset
    from tests.test_grpo_lifecycle import assert_tree_close
    evidence = Path(os.environ.get('DYCON_HOST_OFFLOAD_EVIDENCE',str(tmp_path))).resolve()
    evidence.mkdir(parents=True,exist_ok=True)
    raw = yaml.safe_load((ROOT/'RL/experiments/grpo-offline-k3-smoke.yaml').read_text())
    base = base_identity(raw['model']['path'],ROOT,False)
    native = dataset(evidence,'host-offload-6',range(6))
    meta=read_json(native/'dataset.json')
    meta['cohort']['model']=base
    meta['description']='Generated engineering token fixture; not historical production data'
    write_json(native/'dataset.json',meta)
    rows=[json.loads(x) for x in (native/'branches.jsonl').read_text().splitlines()]
    lengths=[16,128,16,192,16,128]
    for row in rows:
        ordinal=int(row['group_id'][1:])
        prompt=[2+i%40 for i in range(lengths[ordinal])]
        ledger=Path(row['native_evidence']['root'])/row['native_evidence']['ledger_ref']
        payload=read_json(ledger)
        payload['prompt_token_ids']=prompt
        write_json(ledger,payload)
        row['prompt_count']=len(prompt)
        row['prompt_sha256']=hash_json(prompt)
        row['mask_spans']=spans(len(prompt),row['completion_count'])
    (native/'branches.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    seal(native,'rl-dataset','host-offload-6')
    raw['run'].update(mode='smoke',seed=42)
    raw['runtime'].update(gpu=int(os.environ.get('DYCON_HOST_OFFLOAD_GPUS','4,5').split(',')[0]),
        output_root=str(evidence/'artifacts'),bf16_allow_split_k=True)
    raw['model']['gdn_backend']='fla_chunk'
    raw['algorithm'].update(dataset_root=str(evidence/'sources'),dataset_id='host-offload-6')
    raw['grpo'].update(group_batch_size=1,top_k=1,prefix_reuse={'mode':'boundary_grad'})
    raw['training'].update(max_updates=6,checkpoint_every_updates=3)
    raw['training']['memory'].update(mlp_chunk_tokens=64,loss_chunk_tokens=16,
        attention_policy='flash_only',host_budget_gib=0.25,
        auto_offload={'prompt_tokens_threshold':64,'sequence_tokens_threshold':96,'device_headroom_gib':8.0},
        oom_retry={'enabled':False,'max_retries':1})
    raw['validation']['max_sequence_length']=256
    raw['preflight']={'check_resources':True,'verify_checkpoint':True}
    modes={
        'A':{'activation_offload':'auto'},
        'B':{'activation_offload':'auto','host_offload':{'enabled':True,'allocator_budget_gib':0.25,
            'cache_limit_gib':0.0,'reclaim_policy':'after_use','reserve_gib':8.0,'reserve_fraction':0.03}},
        'C':{'activation_offload':'memory_pressure','host_offload':{'enabled':True,'allocator_budget_gib':0.25,
            'cache_limit_gib':0.0,'reclaim_policy':'after_use','reserve_gib':8.0,'reserve_fraction':0.03},
            'memory_pressure':{'device_headroom_gib':8.0,'retry_headroom_increment_gib':4.0,'max_retries':2}},
        'D':{'activation_offload':'none'},
        'M':{'activation_offload':'none','host_offload':{'enabled':True,'allocator_budget_gib':0.25,
            'cache_limit_gib':0.0,'reclaim_policy':'after_use','reserve_gib':8.0,'reserve_fraction':0.03}},
    }
    # These budgets, tolerances and repetitions are frozen before any performance run.
    write_json(evidence/'acceptance-spec.json',{'repetitions':3,'updates':6,'prompt_lengths':lengths,
        'modes':modes,'loss_atol':0.05,'loss_rtol':0.01,'adapter_atol':2e-5,'adapter_rtol':2e-3,
        'monitoring_overhead_target':0.03,'steady_policy_regression_target':0.10,
        'host_mem_available_floor_bytes':8*2**30,'natural_cuda_oom_required':False,
        'fixture':'generated native ledgers; same complete groups, branches and dtype in A/B/C/D'})
    timings={}
    attempts={}
    def training(label,policy,*,stop=None,parent=None,dry=False):
        cfg=copy.deepcopy(raw)
        cfg['run']['attempt_id']='host-offload-v2-9b-'+label
        cfg['run']['mode']='dry-run' if dry else 'smoke'
        cfg['run']['start']={'mode':'resume' if parent else 'fresh','checkpoint':str(parent) if parent else None}
        cfg['validation']['stop_after_update']=stop
        cfg['training']['memory'].update(copy.deepcopy(modes[policy]))
        file=evidence/(label+'.yaml')
        file.write_text(yaml.safe_dump(cfg,sort_keys=False))
        log=evidence/(label+'.log')
        started=time.monotonic()
        with log.open('w') as out:
            proc=subprocess.run([str(ROOT/('.venvs/runner/bin/dycon' if dry else '.venvs/rl/bin/dycon')),
                'lora-train','--training-config',str(file)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=420)
        assert proc.returncode==0,log.read_text()[-11000:]
        timings[label]=time.monotonic()-started
        if dry: return None
        directory=evidence/'artifacts/RL/checkpoints'/cfg['run']['attempt_id']
        attempts[label]=str(directory)
        with (evidence/(label+'-verify.log')).open('w') as out:
            checked=subprocess.run([str(ROOT/'.venvs/runner/bin/dycon'),'artifact','verify',cfg['run']['attempt_id'],
                '--scope','lora-attempt','--storage-root',cfg['runtime']['output_root']],cwd=ROOT,
                stdout=out,stderr=subprocess.STDOUT,timeout=120)
        assert checked.returncode==0,(evidence/(label+'-verify.log')).read_text()[-5000:]
        print(label,'passed',round(timings[label],2),flush=True)
        return directory
    training('dry-C','C',dry=True)
    values={}
    from safetensors.torch import load_file
    for repeat in range(3):
        paths={p:training(f'{p}-{repeat}',p) for p in modes}
        wanted=load_file(str(paths['D']/'adapter/adapter_model.safetensors'))
        baseline=[json.loads(x) for x in (paths['D']/'metrics.jsonl').read_text().splitlines()]
        for p,directory in paths.items():
            actual=load_file(str(directory/'adapter/adapter_model.safetensors'))
            for key in wanted:
                torch.testing.assert_close(actual[key],wanted[key],atol=2e-5,rtol=2e-3,msg=key)
            rows=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
            assert len(rows)==6 and rows[-1]['consumed_groups']==6
            for a,b in zip(rows,baseline):
                torch.testing.assert_close(torch.tensor(a['loss']),torch.tensor(b['loss']),atol=0.05,rtol=0.01)
                assert a['host_global_mem_available_bytes'] >= 8*2**30
            if p in {'A','B'}: assert sum(r['offload_d2h_bytes'] for r in rows)>0
            if p=='C': assert sum(r['offload_d2h_bytes'] for r in rows)==0
            if p in {'B','C','M'}:
                assert all(r['host_pool_reserved_bytes']==0 for r in rows)
                assert all(r['host_allocated_bytes']<=0.25*2**30 for r in rows)
            values[f'{p}-{repeat}']={'wall_seconds':timings[f'{p}-{repeat}'],
                'update_seconds':[r['seconds'] for r in rows],
                'host_allocated_bytes':[r.get('host_allocated_bytes') for r in rows],
                'd2h_bytes':sum(r['offload_d2h_bytes'] for r in rows),
                'h2d_bytes':sum(r['offload_h2d_bytes'] for r in rows),
                'gpu_peak_bytes':max(r['peak_allocated_bytes'] for r in rows)}
        # Keep a completion receipt even if a later repetition is interrupted.
        write_json(evidence/'ab-progress.json',{'attempts':attempts,'values':values,'completed_repetitions':repeat+1})
    paused=training('C-pause','C',stop=3)
    resumed=training('C-resume','C',parent=paused/'checkpoints/step-000003')
    full=Path(attempts['C-0'])
    for name in ['optimizer','scheduler','rng']:
        left,right=[torch.load(p/'checkpoints/step-000006/training-state.pt',weights_only=True,map_location='cpu')[name]
                    for p in [full,resumed]]
        assert_tree_close(left,right)
    assert read_json(full/'final-adapter.json')==read_json(resumed/'final-adapter.json')
    perf={}
    for mode in modes:
        samples=[statistics.median(values[f'{mode}-{i}']['update_seconds'][1:]) for i in range(3)]
        perf[mode]={'steady_median_seconds':statistics.median(samples),'range_seconds':[min(samples),max(samples)]}
    write_json(evidence/'acceptance.json',{'passed':True,'real_model':'Qwen3.5-9B','backend':'fla_chunk',
        'attempts':attempts,'timings':timings,'values':values,'performance':perf,'exact_resume':True,
        'all_adapters_close':True,'real_historical_long_workload':False,'natural_cuda_oom_observed':False,
        'C_pressure_regime':'wide GPU capacity, no offload; partial pressure covered by tiny GPU fixture'})
    regression=perf['C']['steady_median_seconds']/perf['D']['steady_median_seconds']-1
    monitoring=perf['M']['steady_median_seconds']/perf['D']['steady_median_seconds']-1
    write_json(evidence/'performance-verdict.json',{'policy_regression':regression,
        'host_governance_monitoring_overhead':monitoring,'policy_passed':regression<=0.10,
        'monitoring_passed':monitoring<=0.03,'passed':regression<=0.10 and monitoring<=0.03})
    assert regression<=0.10, f'Policy regression {regression:.2%} exceeds 10%'
    assert monitoring<=0.03, f'Host governance monitoring overhead {monitoring:.2%} exceeds 3%'

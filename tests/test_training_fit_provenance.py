import json
from core.io import write_json
from difficulty.training.provenance import fit_sources


def test_published_sources_exclude_validation_only_benchmarks(tmp_path):
    directory=tmp_path/'difficulty/datasets/data'
    rows=[{'rollout_id':'swe','model_fingerprint':'model','benchmark':'swebench','agent':'mini','run_id':'one'},
          {'rollout_id':'skills','model_fingerprint':'model','benchmark':'skillsbench','agent':'openhands','run_id':'two'}]
    write_json(directory/'rollouts.json',{'rollouts':rows})
    (directory/'samples.jsonl').write_text('\n'.join(json.dumps(r) for r in [
        {'sample_id':'a','rollout_id':'swe'},{'sample_id':'b','rollout_id':'skills'}]))
    write_json(directory/'receipt.json',{'state':'complete'})
    write_json(directory/'splits/split/split.json',{'strategy':'task_random_v1',
        'sample_ids':{'train':['a'],'validation':['b'],'test':[]}})
    train=fit_sources(tmp_path,'data','split','train')
    assert train['source_benchmarks']==['swebench']
    assert train['source_agents']==['mini']
    refit=fit_sources(tmp_path,'data','split','train_validation')
    assert refit['source_benchmarks']==['skillsbench','swebench']
    assert refit['source_run_ids']==['one','two']

"""Opt-in pinned-vLLM tests. Uses tiny synthetic logits/model; no benchmark data.

Run in latent-vllm with DYCON_REPLAY_TEST_GPU=an_idle_physical_index.
"""
import os
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

pytestmark = pytest.mark.skipif("DYCON_REPLAY_TEST_GPU" not in os.environ, reason="Explicit idle GPU required")


@pytest.fixture(scope="module")
def native():
    from hidden.backends.vllm.engine import configure_environment, gpu_reservation, verify_runtime
    root = Path(__file__).resolve().parents[1]
    gpu = int(os.environ["DYCON_REPLAY_TEST_GPU"])
    import sys
    configure_environment({"project_root":str(root),"serving":{"python":".venvs/easysteer/bin/python"},
                           "hidden":{"vllm":{"python":sys.executable,"gpus":[gpu]}}})
    pin = verify_runtime(root)
    with gpu_reservation(root,[gpu]):
        import torch
        from vllm import SamplingParams
        yield torch, SamplingParams, pin


def test_native_trace_api_and_restriction_gate(native):
    torch, SamplingParams, pin = native
    params=SamplingParams(temperature=.6, top_p=.95, top_k=20, logprobs=0,
                          trace_decode_token_ids=[3,4,2],max_tokens=3,detokenize=False)
    params._validate_trace_replay(NS(get_vocab_size=lambda:32),None)
    assert params.logprobs == 0 and params.prompt_logprobs is None
    for kwargs in ({"thinking_token_budget":3},{"bad_words":["x"]},{"prompt_logprobs":0},{"n":2}):
        with pytest.raises(ValueError):
            value=SamplingParams(trace_decode_token_ids=[3],max_tokens=1,**kwargs)
            value._validate_trace_replay(NS(get_vocab_size=lambda:32),None)
    assert pin["vllm_version"]=="0.29.0"


@pytest.mark.parametrize("temperature,k,p",[(.6,3,.8),(1.,5,.95),(.7,1,1.),(1.3,5,.01),(1.,3,1.)])
def test_native_processed_distribution_matches_float64_oracle(native,temperature,k,p):
    torch,_,_=native
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs
    from vllm.v1.worker.gpu.sample.trace_replay import apply_trace_tokens
    # Batch size >=8 exercises the native Triton path. Tie rules are tested separately.
    raw=np.array([[2.,1.9,1.2,-.7,-1.4],[3.,1.1,.7,-.4,-2.]]*4,dtype=np.float64)
    logits=torch.tensor(raw,dtype=torch.float32,device="cuda")/temperature
    processed=apply_top_k_top_p(logits,torch.full((8,),k,device="cuda",dtype=torch.int32),torch.full((8,),p,device="cuda"))
    expected=[]
    for row in raw/temperature:
        row=row.copy(); threshold=np.sort(row)[-k]; row[row<threshold]=-np.inf
        order=np.argsort(row,kind="stable"); sorted_row=row[order]
        probs=np.exp(sorted_row-np.max(sorted_row));probs/=probs.sum()
        exclude=np.cumsum(probs)<=1-p;exclude[-1]=False;row[order[exclude]]=-np.inf
        norm=np.log(np.exp(row-np.max(row)).sum())+np.max(row)
        expected.append(row-norm)
    ids=torch.arange(5,device="cuda").repeat(8,1)
    actual=compute_token_logprobs(processed,ids).cpu().numpy()
    np.testing.assert_array_equal(np.isneginf(actual),np.isneginf(expected))
    finite=np.isfinite(expected)
    np.testing.assert_allclose(actual[finite],np.asarray(expected)[finite],atol=1e-6,rtol=0)
    # Trace override changes token IDs after distribution computation, never logits.
    before=processed.clone(); sampled=torch.full((8,),4,device="cuda",dtype=torch.int64)
    traces=torch.tensor([[0,1]]*8,device="cuda",dtype=torch.int32)
    apply_trace_tokens(sampled,torch.arange(8,device="cuda",dtype=torch.int32),traces,
                       torch.full((8,),2,device="cuda",dtype=torch.int32),
                       torch.full((8,),7,device="cuda",dtype=torch.int32),torch.full((8,),7,device="cuda",dtype=torch.int32))
    assert sampled.tolist()==[0]*8
    assert torch.equal(before,processed)
    gathered=compute_token_logprobs(processed,sampled[:,None]).cpu().numpy().ravel()
    np.testing.assert_allclose(gathered,actual[:,0],atol=0,rtol=0)
    if k>1 and p>.5: assert any(x < -.01 for x in gathered if np.isfinite(x))


def test_native_tie_support_depends_on_masking_kernel(native):
    torch,_,_=native
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    # Pin the observed runtime difference; it must not be mistaken for historical equivalence.
    row=torch.tensor([[2.,2.,1.2,-.7,-1.4]],device="cuda")
    small=apply_top_k_top_p(row.clone(),torch.tensor([1],device="cuda"),torch.tensor([1.],device="cuda"))
    large=apply_top_k_top_p(row.repeat(8,1),torch.ones(8,device="cuda",dtype=torch.int32),torch.ones(8,device="cuda"))
    assert torch.isfinite(small[0]).tolist()==[True,True,False,False,False]
    assert torch.isfinite(large[0]).tolist()==[True,False,False,False,False]


def test_tiny_model_native_record_then_trace_replay(native,tmp_path):
    torch, SamplingParams,_=native
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from vllm import LLM
    from RL.dataset.replay.validation import target_logprobs
    torch.manual_seed(1729)
    config=LlamaConfig(vocab_size=64,hidden_size=64,intermediate_size=128,num_hidden_layers=1,
                       num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,
                       bos_token_id=1,eos_token_id=2,pad_token_id=0)
    checkpoint=tmp_path/"tiny-model"
    LlamaForCausalLM(config).save_pretrained(checkpoint)
    tok=Tokenizer(WordLevel({f"t{i}":i for i in range(64)},unk_token="t0"));tok.pre_tokenizer=Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok,unk_token="t0",bos_token="t1",eos_token="t2",pad_token="t0").save_pretrained(checkpoint)
    engine=LLM(model=str(checkpoint),dtype="float32",enforce_eager=True,async_scheduling=False,
               enable_trace_replay=True,logprobs_mode="processed_logprobs",generation_config="vllm",
               enable_prefix_caching=False,max_model_len=32,max_num_seqs=2,max_num_batched_tokens=32,
               kv_cache_memory_bytes=8*1024**2,language_model_only=True,seed=1729)
    prompts=[{"prompt_token_ids":[1,5,6]},{"prompt_token_ids":[1,5,6,7,8]}]
    try:
        params=[SamplingParams(temperature=.6,top_p=.95,top_k=20,seed=101+i,logprobs=0,
                               max_tokens=4,ignore_eos=True,detokenize=False) for i in range(2)]
        recorded=engine.generate(prompts,params,use_tqdm=False)
        completions=[list(o.outputs[0].token_ids) for o in recorded]
        traces=[SamplingParams(temperature=.6,top_p=.95,top_k=20,seed=101+i,logprobs=0,
                              max_tokens=len(c),ignore_eos=True,detokenize=False,trace_decode_token_ids=c) for i,c in enumerate(completions)]
        replayed=engine.generate(prompts,traces,use_tqdm=False)
        for original,replayed_output,c in zip(recorded,replayed,completions):
            original_values=target_logprobs(original,c);replayed_values=target_logprobs(replayed_output,c)
            np.testing.assert_allclose(original_values,replayed_values,atol=1e-5,rtol=0)
            assert any(abs(x)>.01 for x in replayed_values)
    finally:
        shutdown=getattr(engine.llm_engine,"shutdown",None)
        if shutdown:shutdown()
    # Exercise the actual isolated dispatcher/worker, model registration, receipts
    # and loader with a tiny synthetic capture, not a business dataset.
    import gc
    del engine
    gc.collect()
    from core.io import read_json, write_json
    from core.model_identity import register
    from test_rl_dataset import token_source, build
    from RL.dataset.config import ReplayConfig
    from RL.dataset.artifacts import publication,seal
    from RL.dataset.replay.service import prepare,dispatch
    from RL.dataset.replay.artifacts import check_replay,ReplayReader
    from RL.dataset.loader import iter_groups
    from core.io import sha256
    import yaml
    root=Path(__file__).resolve().parents[1]
    fingerprint=register(root,checkpoint)
    identity_path=root/'runtime/model-identities'/f'{fingerprint}.json'
    try:
        fixture=token_source(tmp_path/'capture');f,file,dc=fixture
        dc['audit']['enabled']=False;file.write_text(yaml.safe_dump(dc))
        source_file=tmp_path/'capture/runs'/f['run']/'resolved-config.json'
        source=read_json(source_file)
        source['model']={'path':str(checkpoint),'fingerprint':fingerprint,'served_name':'tiny-test','trust_remote_code':False}
        source['serving']['dtype']='float32';write_json(source_file,source)
        ds=build(fixture)
        replay=ReplayConfig.model_validate({'schema_version':1,'replay_id':'native-worker-test',
          'dataset':{'id':ds.name,'root':str(ds.parents[2])},
          'runtime':{'python':str(root/'.venvs/latent-vllm/bin/python'),'gpus':[int(os.environ['DYCON_REPLAY_TEST_GPU'])],
                     'max_model_len':32,'max_num_seqs':2,'max_num_batched_tokens':32,'kv_cache_memory_bytes':8*1024**2},
          'storage':{'output_root':str(tmp_path/'native-replayed'),'max_shard_bytes':1024}})
        cfg={'project_root':str(root),'generation':{'temperature':.6,'top_p':.95,'top_k':20}}
        plan=prepare(ds,cfg,replay)
        assert not plan['blocked']
        identity={'replay_id':replay.replay_id,'dataset_receipt_sha256':sha256(ds/'receipt.json')}
        # Module fixture holds the same shared reservation a formal parent holds.
        with publication(replay.storage.output_root,'rl-replay',replay.replay_id,signature=identity) as (stage,dest):
            write_json(stage/'job.json',{'project_root':str(root),'dataset':str(ds),'stage':str(stage),
                        'plan':plan,'identity':identity,'replay_config':replay.model_dump(),'recover':False})
            dispatch(replay.runtime.python,stage/'job.json',stage/'worker.log',120,root)
            seal(stage,'rl-replay',replay.replay_id)
        assert check_replay(dest,dataset_path=ds,with_values=True)['tokens']==18
        reader=ReplayReader(dest,ds,allow_estimates=True)
        # Unsupported historical tokens under top-k/p remain -inf, not fabricated.
        assert read_json(dest/'replay.json')['status']=='replay_estimate'
    finally:
        identity_path.unlink(missing_ok=True)

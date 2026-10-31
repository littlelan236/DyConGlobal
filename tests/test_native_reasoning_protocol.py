import re
from rollout.model import model_config
from runtime.serving.manager import ServingManager


def test_required_native_close_uses_shared_generation_path(cfg,tmp_path):
    import json
    cfg['model']['path']=str(tmp_path)
    (tmp_path/'tokenizer_config.json').write_text(json.dumps({'added_tokens_decoder':{
        '773':{'content':'<think>'},'774':{'content':'</think>'}}}))
    cfg['generation']['require_reasoning_close']=True
    cfg['capture']['enabled']=False
    cfg['serving']['reasoning_parser']='qwen3'
    body=model_config(cfg,False)['model_kwargs']['extra_body']
    grammar=body['structured_outputs']['grammar']
    assert 'Token(774)' in grammar and 'ExcludeToken(773)' in grammar
    assert 'Token(248069)' not in grammar  # Never hardcode another model's token IDs.
    args=ServingManager(cfg,tmp_path).build_args('python','127.0.0.1',8108)
    assert '--structured-outputs-config' in args
    assert 'enable_in_reasoning' in args[args.index('--structured-outputs-config')+1]

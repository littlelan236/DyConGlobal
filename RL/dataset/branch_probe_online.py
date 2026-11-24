"""Generate and score paired branches through independent managed online services."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import math
from pathlib import Path
from statistics import fmean
import time
from uuid import uuid4

import requests

from core.config import load_snapshot
from core.io import hash_json, read_json, sha256, write_json
from core.model_identity import verify_model
from rollout.decision.coordinator import ServingParallelClient
from runtime.serving.manager import ServingManager

from .branch_probe import _load_config, _selected_states, preflight


def _service_config(cfg, config, model, estimator, branches):
    settings = config['services'][model['step']]
    frozen = load_snapshot(cfg['project_root'], settings['model_run'])
    path = Path(config['models'][model['step']]).resolve()
    if Path(frozen['model']['path']).resolve() != path:
        raise ValueError('Export differs from frozen serving model')
    verify_model(frozen)
    if not model['base_matches_predictor']:
        raise ValueError('Export does not attest to the estimator base')
    service = deepcopy(frozen)
    service['serving'].update(base_url=f"http://127.0.0.1:{settings['port']}/v1",
                              gpus=[settings['gpu']], tensor_parallel_size=1,
                              max_model_len=config['max_model_len'], max_num_seqs=branches,
                              gpu_memory_utilization=settings.get('gpu_memory_utilization', 0.26))
    service['serving']['kv_cache_memory_bytes'] = None
    service['capture'].update(enabled=True, anchors=['think_close'], layers=estimator['sites'],
                              logprobs={'enabled': False})
    service['capture']['export']['enabled'] = False
    service['steering']['enabled'] = False
    service['generation']['require_reasoning_close'] = False
    old = read_json(Path(config['estimator_run']) / 'resolved-config.json')['parallel']
    service['parallel'] = {**old, 'enabled': True, 'branch_count': branches,
                           'dynamic_branch_admission': False, 'max_inflight_branches': branches,
                           'reasoning_loop': {'enabled': False},
                           'branch_probe': {'model_fingerprint': service['model']['fingerprint'],
                                            'base_fingerprint': estimator['model_fingerprint']}}
    service['config_hash'] = hash_json({k: v for k,v in service.items() if k != 'config_hash'})
    return service


def _sample_node(manager, state, sampling, branches):
    cfg = manager.cfg
    client = ServingParallelClient(cfg)
    controller = manager.capture_controller
    decision = 'probe-' + uuid4().hex
    session_id = 'cap-' + uuid4().hex
    prompt = state['prompt_token_ids']
    if hash_json(prompt) != state['prompt_sha256']:
        raise ValueError('Native prompt hash mismatch')
    deadline = time.time() + 900
    common = dict(session_id=session_id, task_id=state['group_id'], run_id=manager.run_id,
                  rollout_id=state['group_id'], step_id=state['agent_step'], anchors=['think_close'],
                  decision_id=decision, parent_prefix_sha256=state['prompt_sha256'],
                  message_index=state['message_index'], branch_count=branches,
                  decision_deadline_unix=deadline, query_id=decision)
    controller.start({**common, 'branch_id': 0})
    client.call('reserve', {'decision_id': decision, 'branches': list(range(branches)),
                            'decision_deadline_unix': deadline})

    def sample(branch):
        attempt = uuid4().hex
        seed = sampling['seed'] + branch
        body = {k:v for k,v in sampling.items() if k != 'seed'}
        body.update(model=cfg['model']['served_name'], prompt=prompt, seed=seed, n=1,
                    dycon_capture={**common, 'branch_id': branch, 'api_attempt_id': attempt})
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(client.base + '/v1/completions', headers=client.headers,
                                        json=body, timeout=720)
                if not response.ok:
                    raise ValueError(f"HTTP {response.status_code}: {response.text[:350]}")
                result = response.json()
            if len(result['choices']) != 1:
                raise ValueError('Expected exactly one completion')
            return {'branch_id': branch, 'api_attempt_id': attempt, 'seed': seed,
                    'finish_reason': result['choices'][0]['finish_reason']}
        except (requests.RequestException, ValueError) as error:
            return {'branch_id': branch, 'api_attempt_id': attempt, 'seed': seed, 'error': str(error)}

    try:
        with ThreadPoolExecutor(max_workers=branches) as pool:
            attempts = list(pool.map(sample, range(branches)))
        wait = 20 if not any('error' in x for x in attempts) else 0
        result = client.call('results', {'decision_id': decision, 'timeout': wait})
        if result['estimator_sha256'] != cfg['parallel']['estimator_sha256']:
            raise ValueError('Online estimator hash mismatch')
        rows = []
        for attempt, score in zip(attempts, result['results'], strict=True):
            native = controller.request_outputs.get(score.get('native_request_id'))
            if score['status'] == 'scored':
                if (not native or native['request_sha256'] != state['prompt_sha256']
                        or native['api_attempt_id'] != attempt['api_attempt_id']
                        or score['branch_id'] != attempt['branch_id']):
                    raise ValueError('Capture receipt does not bind the sampled branch')
                pos = score['absolute_position'] - len(prompt)
                ids = native['output_token_ids']
                if not 0 <= pos < len(ids) or ids[pos] != score['token_id']:
                    raise ValueError('Captured close token is not in the native completion')
            rows.append({**score, **attempt, 'native_receipt_sha256': native.get('ledger_sha256') if native else None,
                         'completion_tokens': len(native['output_token_ids']) if native else None})
        return rows
    finally:
        client.call('release', {'decision_id': decision})


def run(cfg, config_path, output):
    from runtime.serving.parallel import artifact_contract

    config = _load_config(config_path)
    sampling = dict(config['sampling'])
    branches = sampling.pop('branches')
    if type(branches) is not int or not 2 <= branches <= 32:
        raise ValueError('Expected 2 to 32 branches')
    if set(sampling) - {'temperature', 'top_p', 'max_tokens', 'seed', 'stop', 'stop_token_ids'}:
        raise ValueError('Unsupported sampling options')
    if sampling['temperature'] <= 0 or sampling['max_tokens'] < 1:
        raise ValueError('Expected free sampling with a positive token budget')
    proof = preflight(config_path)
    states = _selected_states(config)
    if any(len(x['prompt_token_ids']) + sampling['max_tokens'] > config['max_model_len'] for x in states):
        raise ValueError('Native prompt plus output budget exceeds max_model_len')
    estimator = read_json(Path(config['estimator_run']) / 'resolved-config.json')['parallel']
    contract = artifact_contract(estimator['estimator_ref'], proof['estimator_sha256'])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'inputs.json', {**proof, 'config': config, 'config_sha256': sha256(config_path)})
    results = {}
    for model in proof['models']:
        step = model['step']
        service = _service_config(cfg, config, model, contract, branches)
        with ServingManager(service, output / f'step{step}', run_id=f'branch-probe-{step}-{uuid4().hex}') as manager:
            results[step] = {}
            for state in states:
                rows = _sample_node(manager, state, sampling, branches)
                results[step][state['group_id']] = rows
                write_json(output / f'branches-step{step}.json', results[step])
    summary = []
    for state in states:
        group_id = state['group_id']
        row = {'group_id': group_id, 'agent_step': state['agent_step']}
        for step in (450, 1240):
            scores = [x['score'] for x in results[step][group_id]
                      if x['status'] == 'scored' and not x.get('error') and math.isfinite(x['score'])]
            row[f'step{step}'] = {'mean': fmean(scores) if scores else None, 'scored': len(scores),
                                  'requested': branches}
        row['mean_delta_1240_minus_450'] = (row['step1240']['mean'] - row['step450']['mean']
                                           if all(row[f'step{s}']['scored'] >= 2 for s in (450,1240)) else None)
        summary.append(row)
    report = {'output': str(output), 'states': summary}
    write_json(output / 'summary.json', report)
    if any(s['mean_delta_1240_minus_450'] is None for s in summary):
        raise ValueError('A paired node has fewer than two scored branches per checkpoint')
    return report

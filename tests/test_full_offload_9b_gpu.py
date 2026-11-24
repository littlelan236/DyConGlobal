"""Real 9B length ladder through formal CLI; records capacity failures explicitly.

Generated native token ledgers measure execution/capacity, not benchmark quality.
The test accepts only success or a diagnosed CUDA capacity boundary after at least
one committed update. A capacity boundary is NOT acceptance of longer lengths.
"""
import json
import os
from pathlib import Path
import subprocess
import time

import pytest
import yaml

pytestmark = pytest.mark.skipif(os.environ.get('DYCON_FULL_OFFLOAD_9B') != '1',
                               reason='Explicit real 9B full-policy capacity test')


@pytest.mark.parametrize('backend', ['reference_pytorch', 'fla_chunk'])
def test_real_9b_full_length_ladder(tmp_path, backend):
    from core.io import hash_json, read_json, write_json, sha256
    from RL.artifacts import base_identity
    from RL.dataset.artifacts import seal
    from RL.dataset.schema import spans
    from RL.grpo.schedule import order_groups
    from tests.test_grpo_prepared import ROOT, dataset
    from tests.test_full_policy import full_values
    ceiling = int(os.environ.get('DYCON_FULL_MAX_LENGTH', '32768'))
    lengths = [n for n in [2048, 4096, 8192, 16384, 32768] if n <= ceiling]
    raw = full_values()
    model = os.environ.get('DYCON_FULL_MODEL', '/VM/wangtianyi/DyConGlobal/models/Qwen3.5-9B')
    base = base_identity(model, ROOT, False)
    native = dataset(tmp_path, 'full-offload-length-ladder', range(len(lengths)), branch_count=2)
    metadata = read_json(native / 'dataset.json')
    metadata['cohort']['model'] = base
    metadata['description'] = 'Generated engineering native token fixture; not benchmark trajectories'
    write_json(native / 'dataset.json', metadata)
    rows = [json.loads(s) for s in (native / 'branches.jsonl').read_text().splitlines()]
    for row in rows:
        length = lengths[int(row['group_id'][1:])]
        prompt = [2 + i % 40 for i in range(length - 2)]
        completion = [5, 6 + int(row['branch_id'])]
        ledger = Path(row['native_evidence']['root']) / row['native_evidence']['ledger_ref']
        identity = {key: row[source] for key, source in [
            ('request_id', 'native_request_id'), ('session_id', 'session_id'),
            ('decision_id', 'decision_id'), ('branch_id', 'branch_id'),
            ('step_id', 'agent_step'), ('message_index', 'message_index'),
            ('tokenizer_fingerprint', 'tokenizer_fingerprint')]}
        write_json(ledger, {**identity, 'prompt_token_ids': prompt, 'output_token_ids': completion})
        row['native_evidence']['ledger_sha256'] = sha256(ledger)
        row.update(prompt_count=len(prompt), completion_count=len(completion),
            prompt_sha256=hash_json(prompt), completion_sha256=hash_json(completion),
            mask_spans=spans(len(prompt), len(completion)))
    (native / 'branches.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    seal(native, 'rl-dataset', 'full-offload-length-ladder')
    seed = next(s for s in range(10000) if order_groups(list(range(len(lengths))), seed=s,
        batch_size=1, algorithm='prefix_length_pool128_v1', prompt_length=lambda i: lengths[i]) == list(range(len(lengths))))
    raw['run'].update(attempt_id='real-9b-' + backend, mode='smoke', seed=42)
    raw['model'].update(path=model, dtype='bfloat16', gdn_backend=backend)
    raw['runtime'] = {'gpus': [int(s) for s in os.environ.get('DYCON_FULL_GPUS', '4,5').split(',')],
        'output_root': str(tmp_path), 'distributed_timeout_seconds': 240}
    raw['training'].update(max_updates=len(lengths), checkpoint_every_updates=len(lengths), learning_rate=1e-6)
    raw['training']['memory'].update(activation_offload='checkpoint_boundary', host_budget_gib=16.,
        loss_chunk_tokens=16, exact_length=True, attention_policy='flash_only',
        host_offload={'enabled': True, 'reclaim_policy': 'after_use',
                      'allocator_budget_gib': float(os.environ.get('DYCON_FULL_HOST_BUDGET', '8.25')),
                      'cache_limit_gib': 0., 'reserve_gib': 8., 'reserve_fraction': .03},
        oom_retry={'enabled': False}, memory_telemetry={'enabled': True, 'granularity': 'phase'})
    raw['validation'].update(max_sequence_length=max(lengths), stop_after_update=None)
    raw['algorithm'].update(dataset_root=str(tmp_path / 'sources'), dataset_id='full-offload-length-ladder')
    raw['grpo'].update(group_batch_size=1, shuffle_seed=seed, shuffle_algorithm='prefix_length_pool128_v1',
        top_k=1, top_k_fraction=None, kl_beta=0., ratio_guard={'enabled': False}, prefix_reuse={'mode': 'disabled'})
    raw['preflight'] = {'verify_identity': False, 'check_resources': True, 'audit_native_tokens': True}
    config = tmp_path / 'training.yaml'
    config.write_text(yaml.safe_dump(raw))
    log = tmp_path / 'training.log'
    started = time.monotonic()
    with log.open('w') as stream:
        result = subprocess.run([str(ROOT / '.venvs/rl/bin/dycon'), 'rl-train', '--training-config', str(config)],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=2700)
    attempt = tmp_path / 'RL/checkpoints' / raw['run']['attempt_id']
    metrics = [json.loads(s) for s in (attempt / 'metrics.jsonl').read_text().splitlines()] if (attempt / 'metrics.jsonl').exists() else []
    text = log.read_text()
    evidence = {'backend': backend, 'exit_code': result.returncode, 'seconds': time.monotonic() - started,
        'requested_lengths': lengths, 'committed_lengths': lengths[:len(metrics)], 'metrics': metrics,
        'capacity_verified_through': lengths[len(metrics)-1] if metrics else None,
        'longer_lengths_accepted': result.returncode == 0,
        'failure_tail': text[-16000:] if result.returncode else None}
    write_json(tmp_path / 'capacity-evidence.json', evidence)
    assert metrics, text[-16000:]
    if result.returncode:
        assert 'out of memory' in text.lower(), text[-16000:]
        assert not (attempt / 'receipt.json').exists()
    else:
        assert len(metrics) == len(lengths)
        assert (attempt / 'receipt.json').exists()
    for row in metrics:
        assert row['trainable_parameters'] == 8953803264
        assert row['policy_changed']
        for rank in row['ranks']:
            offload = rank['activation_offload']
            assert offload['offload_d2h_bytes'] > 0
            assert offload['offload_h2d_bytes'] == offload['offload_d2h_bytes']
            assert offload['offload_live_bytes'] == offload['host_pool_reserved_bytes'] == 0
    print('CAPACITY', backend, evidence['capacity_verified_through'], 'exit', result.returncode, flush=True)

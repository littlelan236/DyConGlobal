"""Read-only bounded summaries of memory JSONL; no Torch dependency."""
import json

from RL.observability import source

MAX_LINE = 2 * 1024 * 1024


def length_bucket(length):
    for lower, upper in zip((0, 8192, 16384, 32768, 49152, 65536, 81920),
                            (8192, 16384, 32768, 49152, 65536, 81920, float('inf'))):
        if lower <= length < upper:
            return f'{lower}-{int(upper)-1}' if upper != float('inf') else f'{lower}+'
    return 'unknown'


def summarize_files(paths, *, rank=None, phase=None, limit=100):
    if not 1 <= limit <= 1000: raise ValueError('memory summary limit must be 1..1000')
    if rank is not None and rank < 0: raise ValueError('rank must be nonnegative')
    buckets = {}
    warnings = []
    file_count = 0
    def record(row, outcome, groups):
        if rank is not None and row['rank'] != rank: return
        if phase is not None and row.get('phase') != phase: return
        group = groups.get(row.get('group_ordinal'), {})
        length = group.get('max_sequence_tokens', max(
            (g['max_sequence_tokens'] for g in groups.values()), default=0))
        mode = row.get('offload', {}).get('mode', 'unknown')
        key = (row['rank'], row.get('phase'), mode, length_bucket(length), outcome, row['event'])
        item = buckets.setdefault(key, {'rank': key[0], 'phase': key[1], 'offload_mode': mode,
            'sequence_tokens': key[3], 'outcome': outcome, 'sample_kind': key[5], 'samples': 0, 'seconds_sum': 0.,
            'peak_allocated_bytes_max': 0, 'peak_reserved_bytes_max': 0})
        item['samples'] += 1
        item['seconds_sum'] += row.get('seconds', 0.)
        for name in ('allocated', 'reserved'):
            item[f'peak_{name}_bytes_max'] = max(item[f'peak_{name}_bytes_max'], row.get(f'peak_{name}_bytes', 0))
    for path in paths:
        if not path.is_file(): continue
        file_count += 1
        pending, groups = [], {}
        with path.open('rb') as stream:
            line_number = 0
            while True:
                raw = stream.readline(MAX_LINE + 1)
                if not raw: break
                line_number += 1
                if len(raw) > MAX_LINE: raise ValueError(f'Memory event exceeds line limit: {path}:{line_number}')
                if not raw.endswith(b'\n'):
                    warnings.append(f'Incomplete last line ignored: {path.name}:{line_number}')
                    break
                row = json.loads(raw)
                event = row.get('event')
                if event in ('step_start', 'retry_start'):
                    for old in pending: record(old, 'uncommitted', groups)
                    pending = []
                    groups = {g['group_ordinal']: g for peer in row.get('offload', {}).get('ranks', [])
                              if peer['rank'] == row['rank'] for g in peer['groups']}
                if event in ('phase_end', 'oom'): pending.append(row)
                if event == 'attempt_result' and any(s['state'] != 'OK' for s in row['rank_results']):
                    local = next(s for s in row['rank_results'] if s['rank'] == row['rank'])
                    outcome = local['state'] if local['state'] != 'OK' else 'discarded_peer_failure'
                    for old in pending: record(old, outcome, groups)
                    pending = []
                if event == 'step_commit':
                    for old in pending: record(old, 'committed', groups)
                    pending = []
                if event == 'terminal_failure':
                    for old in pending: record(old, 'fatal', groups)
                    pending = []
        for old in pending: record(old, 'uncommitted', groups)
    rows = sorted(buckets.values(), key=lambda x: (x['rank'], x['phase'] or '', x['offload_mode'], x['sequence_tokens'], x['outcome']))
    return {'schema_version': 1, 'files': file_count, 'total_buckets': len(rows), 'buckets': rows[:limit],
            'truncated': len(rows) > limit, 'warnings': warnings,
            'capacity_claim': 'Observed intervals only; OOM requested allocations are not included in allocated peaks.'}


def summarize(cfg, attempt_id, *, storage_root=None, rank=None, phase=None, limit=100):
    directory, _, _, _ = source(cfg, attempt_id, storage_root)
    paths = [directory / 'memory-events.jsonl', *sorted((directory / 'distributed').glob('rank-*/memory-events.jsonl'))]
    return {'attempt_id': attempt_id, **summarize_files(paths, rank=rank, phase=phase, limit=limit)}

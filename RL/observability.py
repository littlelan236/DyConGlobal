"""Read-only RL snapshots and best-effort runtime stage telemetry (no Torch)."""
from collections import deque
from datetime import datetime
import hashlib
import io
import math
import os
from pathlib import Path
import statistics
import threading
import time
import warnings

from core.io import artifact_roots, hash_json, now, read_json, write_json
from RL.artifacts import _directory
from RL.metric_schema import SIGNAL_SCALARS

MAX_METRIC_BYTES = 64 * 1024 * 1024
SCALARS = ('consumed_execution_chunks', 'execution_window_cursor', 'completed_parent_groups',
           'cursor_schema', 'execution_schedule_sha256', 'consumed_half_ids_sha256', 'execution_round',
           'distinct_parent_groups_seen', 'planned_parent_groups', 'planned_execution_chunks',
           'parent_group_count', 'execution_chunk_count', 'parent_branch_count', 'branch_loss_weight', 'global_step', 'group_cursor', 'consumed_groups', 'consumed_branches',
           'consumed_tokens', 'loss', 'policy_loss', 'k3_penalty_mean', 'clip_fraction',
           'ratio_mean', 'positive_ratio_mean', 'negative_ratio_mean',
           'gradient_norm', 'learning_rate', 'seconds', 'step_seconds',
           'forward_seconds', 'backward_seconds', 'peak_allocated_bytes',
           'reward', 'reward_mean', 'mean_reward',
           'peak_reserved_bytes', 'free_bytes', 'observed_at', 'attempt_id', 'metrics_schema_version') + SIGNAL_SCALARS


def streaming_json():
    try:
        import ijson
    except ImportError as exc:
        raise ValueError('Missing ijson==3.5.1; install the locked runner dependencies') from exc
    return ijson


def process(pid):
    try:
        p = Path('/proc') / str(int(pid))
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': int(pid), 'starttime': fields[19],
                'command': [s.decode(errors='replace') for s in (p / 'cmdline').read_bytes().split(b'\0') if s]}
    except (OSError, ValueError, IndexError):
        return None


def matches(record):
    actual = process(record.get('pid', -1)) if record else None
    return bool(actual and actual == record)


def source(cfg, attempt_id, storage_root=None):
    directory = _directory(cfg, attempt_id, storage_root)
    attempt = read_json(directory / 'attempt.json')
    if attempt.get('attempt_id') != attempt_id:
        raise ValueError('Attempt ID does not match its directory')
    contract = attempt.get('contract') or {}
    if (directory / 'grpo-contract.json').exists():
        frozen = read_json(directory / 'grpo-contract.json')
        if contract and frozen != contract:
            raise ValueError('Attempt and GRPO contracts disagree')
        if not contract and attempt.get('status') == 'failed':
            attempt = {**attempt, 'contract': frozen}
            contract = frozen
    identity = {'schema_version': 1, 'attempt_id': attempt_id,
                'storage_root': str(directory.parents[2]), 'directory': str(directory),
                'contract_sha256': hash_json(contract)}
    cache = Path(cfg['project_root']) / 'runtime/rl-observability' / hash_json(identity)
    return directory, attempt, identity, cache


class MetricReader:
    """Read complete JSONL records only; offsets advance after successful validation.

    A 64 MiB line cap is explicit; unknown arrays are discarded by ijson events.
    File identity, prefix fingerprints and the last committed line detect rewrites.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.identity = None
        self.anchor = None
        self.prefix = None
        self.seen = {}
        self.history = deque(maxlen=256)

    def seek_tail(self, count):
        if not self.path.exists():
            return
        with self.path.open('rb') as stream:
            stream.seek(0, 2)
            position = stream.tell()
            newlines = 0
            while position > 0:
                length = min(position, 1024*1024)
                position -= length
                stream.seek(position)
                block = stream.read(length)
                found = block.count(b'\n')
                if found + newlines >= count+1:
                    end = len(block)
                    for _ in range(count+1-newlines):
                        end = block.rfind(b'\n', 0, end)
                    self.offset = position+end+1
                    return
                newlines += found

    def read(self):
        if not self.path.exists():
            if self.identity is not None:
                raise ValueError('Metrics source disappeared; explicit rebuild required')
            return
        with self.path.open('rb') as f:
            st = os.fstat(f.fileno())
            identity = (st.st_dev, st.st_ino)
            if self.identity is not None and (identity != self.identity or st.st_size < self.offset):
                raise ValueError('Metrics source replaced/truncated; explicit rebuild required')
            self.identity = identity
            if self.prefix:
                length, digest = self.prefix
                if hashlib.sha256(f.read(length)).hexdigest() != digest:
                    raise ValueError('Metrics prefix changed; explicit rebuild required')
            if self.anchor:
                start, length, digest = self.anchor
                f.seek(start)
                if hashlib.sha256(f.read(length)).hexdigest() != digest:
                    raise ValueError('Metrics committed line changed; explicit rebuild required')
            f.seek(self.offset)
            while True:
                start = f.tell()
                raw = f.readline(MAX_METRIC_BYTES + 1)
                if len(raw) > MAX_METRIC_BYTES:
                    raise ValueError(f'Metric exceeds {MAX_METRIC_BYTES} byte resource limit at {start}')
                if not raw or not raw.endswith(b'\n'):
                    break
                selected = {}
                try:
                    first = True
                    for prefix, event, value in streaming_json().parse(io.BytesIO(raw), use_float=True):
                        if first and (prefix, event) != ('', 'start_map'):
                            raise ValueError('Metric must be an object')
                        first = False
                        if '.' not in prefix and prefix in SCALARS and event in ('number', 'string', 'null', 'boolean'):
                            if prefix in selected:
                                raise ValueError('Duplicate metric field: ' + prefix)
                            if isinstance(value, float) and not math.isfinite(value):
                                raise ValueError('Nonfinite metric')
                            selected[prefix] = value
                    step = selected.get('global_step')
                    if type(step) is not int or step < 1:
                        raise ValueError('Metric global_step must be a positive integer')
                    digest = hashlib.sha256(raw).hexdigest()
                    if step in self.seen and self.seen[step] != digest:
                        raise ValueError(f'Conflicting metric step {step}')
                    if step not in self.seen and self.seen and step <= max(self.seen):
                        raise ValueError('Metric steps are out of order')
                except (ValueError, OverflowError, streaming_json().JSONError) as exc:
                    raise ValueError(f'Metrics parse error at byte {start}: {exc}') from exc
                duplicate = step in self.seen
                self.seen[step] = digest
                self.offset = f.tell()
                self.anchor = (start, len(raw), digest)
                if self.prefix is None:
                    self.prefix = (len(raw), digest)
                if not duplicate:
                    yield selected, digest


def age(value):
    try:
        return max(0., time.time() - datetime.fromisoformat(value).timestamp()) if value else None
    except (ValueError, TypeError):
        return None


def log_warnings(pid):
    """Bounded diagnostic tail of the exact trainer's redirected stdout, if any."""
    if not pid:
        return []
    try:
        path = Path('/proc')/str(pid['pid'])/'fd/1'
        if not path.is_file():
            return []
        with path.open('rb') as stream:
            stream.seek(0,2)
            length = stream.tell()
            stream.seek(max(0,length-65536))
            lines = stream.read(65536).decode('utf-8',errors='replace').splitlines()
        if length > 65536:
            lines = lines[1:]
        return [line[:400] for line in lines if any(word in line.lower() for word in ('warning:', 'falling back', 'allocator warning', 'nonfatal'))][-5:]
    except OSError:
        return []


def training_process(directory, attempt, live):
    recorded = live.get('process')
    if recorded and matches(recorded):
        return recorded
    # Old trainers did not register PIDs: bind exact --training-config path,
    # resolved config attempt ID/output root and creation time, never a name substring.
    import yaml
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        item = process(p.name)
        if not item:
            continue
        cmd = item['command']
        if 'lora-train' not in cmd or '--training-config' not in cmd:
            continue
        try:
            path = Path(cmd[cmd.index('--training-config') + 1])
            if not path.is_absolute():
                path = (p / 'cwd').resolve() / path
            config = yaml.safe_load(path.read_text())
            if (config['run']['attempt_id'] == directory.name
                    and Path(config['runtime']['output_root']).resolve() == directory.parents[2]):
                created = attempt.get('created_at')
                if created:
                    boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime '))
                    started = boot + int(item['starttime']) / os.sysconf('SC_CLK_TCK')
                    created_unix = datetime.fromisoformat(created).timestamp()
                    if not created_unix-3600 <= started <= created_unix+5:
                        continue
                return item
        except (OSError, ValueError, IndexError, KeyError, TypeError):
            continue
    return None


def snapshot(cfg, attempt_id, storage_root=None, recent=10, reader=None, *, verify_checkpoints=True):
    directory, attempt, identity, cache = source(cfg, attempt_id, storage_root)
    config = read_json(directory / 'resolved-config.json')
    contract = attempt.get('contract') or {}
    proof = contract.get('input', {})
    messages = []
    if reader is None:
        reader = MetricReader(directory / 'metrics.jsonl')
        reader.seek_tail(max(10,recent))
    try:
        reader.history.extend(m for m, _ in reader.read())
    except ValueError as exc:
        messages.append(str(exc))
    rows = reader.history
    latest = rows[-1] if rows else {}
    # Never substitute an inflight cursor for the last committed metric cursor.
    state = latest or attempt.get('final_state') or attempt.get('initial_state', {})
    live = read_json(cache / 'live.json') if (cache / 'live.json').exists() else {}
    pid = training_process(directory, attempt, live)
    messages.extend(log_warnings(pid))
    if live and (not pid or live.get('process') != pid):
        if attempt.get('status') not in ('paused','completed','failed','dry_run'):
            messages.append('Stage snapshot belongs to an exited/different process')
        live = {}
    checkpoint = None
    checkpoint_paths = (directory / 'checkpoints').glob('step-*') if verify_checkpoints else ()
    for path in sorted(checkpoint_paths, reverse=True):
        if not (path / 'receipt.json').is_file():
            continue
        cp = read_json(path / 'state.json')
        receipt = read_json(path / 'receipt.json')
        from core.io import sha256
        entry = receipt.get('files', {}).get('state.json', {})
        if entry.get('sha256') != sha256(path / 'state.json') or (contract and cp.get('contract') != contract):
            messages.append('Checkpoint state/contract receipt mismatch: ' + path.name)
            continue
        try:
            from RL.grpo.stream_resume import checkpoint_kind
            if checkpoint_kind(cp['contract']) == 'streaming':
                from RL.checkpoints import verify_checkpoint
                verify_checkpoint(path)
        except (OSError, ValueError, KeyError) as exc:
            messages.append('Invalid streaming checkpoint ' + path.name + ': ' + str(exc))
            continue
        checkpoint = {'path': str(path), 'created_at': receipt.get('created_at'),
                      'state': cp.get('state'), 'receipt_present': True}
        break
    if checkpoint:
        checkpoint_state = checkpoint.get('state') or {}
        if (checkpoint_state.get('global_step',0) > state.get('global_step',0)
                or checkpoint_state.get('global_step') == state.get('global_step')
                and (checkpoint_state.get('group_cursor') != state.get('group_cursor')
                     or any(checkpoint_state.get(k) != state.get(k) for k in (
                         'consumed_execution_chunks', 'execution_window_cursor',
                         'consumed_half_ids_sha256', 'execution_schedule_sha256')))):
            messages.append('Checkpoint and latest complete metric cursors disagree')
    terminal = attempt.get('status') in ('paused', 'completed', 'failed', 'dry_run')
    liveness = 'running' if pid else ('not_running' if terminal else 'not_running_with_stale_state')
    initial_step = attempt.get('initial_state', {}).get('global_step', 0)
    current = [r for r in rows if r.get('attempt_id', attempt_id) == attempt_id and r['global_step'] > initial_step]
    times = [r.get('step_seconds', r.get('seconds')) for r in current[-10:]]
    times = [x for x in times if type(x) in (int, float) and math.isfinite(x) and x > 0]
    maximum = proof.get('planned_updates', config.get('training', {}).get('max_updates'))
    step = state.get('global_step', 0)
    eta = {'samples': len(times), 'mean_step_seconds': statistics.mean(times) if times else None,
           'median_step_seconds': statistics.median(times) if times else None,
           'remaining_updates': max(0, maximum-step) if maximum is not None else None,
           'confidence': 'low' if len(times) < 3 or not pid else 'window_estimate',
           'basis': 'current attempt last 10 completed updates; excludes startup/checkpoint overhead'}
    eta['remaining_seconds'] = eta['mean_step_seconds'] * eta['remaining_updates'] if times and pid and maximum else None
    metrics_path = directory / 'metrics.jsonl'
    tb = cache / 'tensorboard'
    sync = read_json(tb / 'sync-cursor.json') if (tb / 'sync-cursor.json').exists() else None
    if sync and sync.get('error'):
        messages.append('TensorBoard synchronizer: ' + sync['error'])
    service = read_json(tb / 'service.json') if (tb / 'service.json').exists() else None
    visualization = None
    if service:
        visualization = {name + '_alive': matches(service.get(name)) for name in ('sync', 'tensorboard')}
        visualization.update(sync=sync, state=service.get('state'))
        if service.get('state') == 'running' and not all(visualization[k] for k in ('sync_alive', 'tensorboard_alive')):
            messages.append('An owned visualization process has exited')
    return {'schema_version': 1, 'attempt_id': attempt_id, 'source': identity,
            'status': attempt.get('status'), 'created_at': attempt.get('created_at'), 'ended_at': attempt.get('ended_at'),
            'observed_at': now(), 'observed_liveness': liveness, 'process': pid,
            'last_completed_update': step, 'global_step': step, 'max_updates': maximum,
            'percent': 100 * step/maximum if maximum else None,
            'execution_stop_target': config.get('validation', {}).get('stop_after_update') or maximum,
            'committed_consumed_groups': state.get('group_cursor'), 'planned_groups': proof.get('group_count'),
            **({key: state.get(key, proof.get(key)) for key in (
                'planned_parent_groups', 'planned_execution_chunks', 'consumed_execution_chunks',
                'distinct_parent_groups_seen', 'completed_parent_groups', 'execution_window_cursor')}
               if proof.get('execution_plan') else {}),
            'consumed_branches': state.get('consumed_branches'), 'consumed_tokens': state.get('consumed_tokens'),
            'inflight_update': live.get('inflight_update') if pid else None,
            'stage': live.get('stage', 'unknown'), 'stage_age_seconds': age(live.get('last_stage_changed_at')),
            'live': live or None, 'latest_metric': latest or None, 'recent_updates': list(rows)[-recent:] if recent else [],
            'checkpoint': checkpoint, 'recoverable_checkpoint_step': (checkpoint.get('state') or {}).get('global_step') if checkpoint else None,
            'source_freshness': {'metrics_age_seconds': max(0, time.time()-metrics_path.stat().st_mtime) if metrics_path.exists() else None,
                                 'heartbeat_age_seconds': age(live.get('process_alive_at')),
                                 'historical_wall_time_estimated': bool(latest and not latest.get('observed_at'))},
            'eta': eta, 'visualization': visualization, 'warnings': messages}


def active(cfg, storage_root=None, recent=0, *, progress_only=False):
    roots = [Path(storage_root).resolve()] if storage_root else artifact_roots(cfg, 'lora-attempt')
    candidates = set()
    errors = []
    for root in roots:
        for path in (root / 'RL/checkpoints').glob('*/attempt.json'):
            if path.is_symlink() or path.parent.is_symlink():
                continue
            try:
                attempt = read_json(path)
            except (OSError, ValueError) as exc:
                if not progress_only:
                    raise
                errors.append({'attempt_id': path.parent.name, 'storage_root': str(root), 'error': str(exc)[:400]})
                continue
            if attempt.get('status') not in ('completed', 'paused', 'failed', 'dry_run'):
                candidates.add((path.parent.name, str(root)))
    if not storage_root:
        # An existing trainer may use an explicitly configured physical root
        # which is absent from today's project artifact-scan roots.
        import yaml
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            item = process(entry.name)
            command = item['command'] if item else []
            if 'lora-train' not in command or '--training-config' not in command:
                continue
            try:
                config_path = Path(command[command.index('--training-config')+1])
                if not config_path.is_absolute():
                    config_path = (entry/'cwd').resolve()/config_path
                training = yaml.safe_load(config_path.read_text())
                root = Path(training['runtime']['output_root']).resolve()
                name = training['run']['attempt_id']
                directory, _, _, _ = source(cfg, name, root)
                if training_process(directory, read_json(directory/'attempt.json'), {}):
                    candidates.add((name, str(root)))
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                continue
        registry = Path(cfg['project_root'])/'runtime/rl-observability'
        for live_path in registry.glob('*/live.json'):
            try:
                live = read_json(live_path)
                binding = read_json(live_path.parent/'source.json')
                if matches(live.get('process')):
                    candidates.add((binding['attempt_id'], binding['storage_root']))
            except (OSError, ValueError, KeyError):
                continue
    if not progress_only:
        return {'schema_version': 1, 'attempts': [
            snapshot(cfg, name, root, recent) for name, root in sorted(candidates)]}
    found = []
    for name, root in sorted(candidates):
        try:
            found.append(snapshot(cfg, name, root, recent, verify_checkpoints=False))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append({'attempt_id': name, 'storage_root': root, 'error': str(exc)[:400]})
    return {'schema_version': 1, 'attempts': found, 'errors': errors}


class LiveTelemetry:
    """Optional diagnostic writes cannot fail the training main path."""
    def __init__(self, cfg, attempt_id, storage_root):
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self.data = {}
        self.last_write = 0.
        self.path = None
        try:
            _, _, identity, cache = source(cfg, attempt_id, storage_root)
            write_json(cache / 'source.json', identity)
            self.path = cache / 'live.json'
            self.data = {'process': process(os.getpid()), 'stage': 'training', 'last_stage_changed_at': now()}
        except Exception as exc:
            warnings.warn('RL observability unavailable: ' + str(exc))
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def _write(self):
        if self.path:
            try:
                write_json(self.path, {**self.data, 'process_alive_at': now()})
            except Exception as exc:
                warnings.warn('RL observability write failed: ' + str(exc))

    def emit(self, stage, **fields):
        with self.lock:
            if self.data.get('stage') != stage:
                self.data['last_stage_changed_at'] = now()
            self.data.update(stage=stage, **fields)
            if time.monotonic()-self.last_write >= 2 or stage in ('update_committed','checkpoint','exception','finalizing','completed','paused'):
                self._write()
                self.last_write = time.monotonic()

    def _heartbeat(self):
        while not self.stopped.wait(10):
            with self.lock:
                self._write()

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=1)

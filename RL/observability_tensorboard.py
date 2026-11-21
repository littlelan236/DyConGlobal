"""Managed replayable TensorBoard projection; all mutable state is external."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

from core.io import now, read_json, write_json
from core.locking import file_lock
from RL.observability import MetricReader, matches, process, source, training_process
from RL.metric_schema import SIGNAL_TAGS

TAGS = {**{k: 'train/' + k for k in ('loss', 'policy_loss', 'k3_penalty_mean',
        'clip_fraction', 'ratio_mean', 'gradient_norm', 'learning_rate')},
        **{'positive_ratio_mean': 'train/policy_ratio_positive_advantage',
           'negative_ratio_mean': 'train/policy_ratio_negative_advantage'},
        **{k: 'progress/' + k for k in ('group_cursor', 'consumed_branches', 'consumed_tokens',
            'consumed_execution_chunks', 'execution_window_cursor', 'completed_parent_groups',
            'distinct_parent_groups_seen', 'parent_group_count', 'execution_chunk_count',
            'planned_parent_groups', 'planned_execution_chunks')},
        **{k: 'timing/' + k for k in ('forward_seconds', 'backward_seconds')}}

K3_LOSS_TAG = 'train/k3_loss'
K3_LOSS_FRACTION_TAG = 'train/k3_loss_fraction'
PINNED_CHARTS = (
    ('Gradient L2 norm', TAGS['gradient_norm']),
    ('Total loss', TAGS['loss']),
    ('K3 (before beta)', SIGNAL_TAGS['k3_penalty_mean']),
    ('Weighted K3 / total loss', K3_LOSS_FRACTION_TAG),
    ('Policy ratio: positive advantage tokens', TAGS['positive_ratio_mean']),
    ('Policy ratio: negative advantage tokens', TAGS['negative_ratio_mean']),
    ('Training step duration (seconds)', 'timing/step_seconds'),
)
SERVICE_MODULE = 'RL.observability_tensorboard'

TERMINAL_ATTEMPT_STATES = frozenset(('paused', 'completed', 'failed', 'dry_run'))
DEFAULT_QUERY_LEASE_SECONDS = 1800
MAX_QUERY_LEASE_SECONDS = 7 * 24 * 60 * 60


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def scalar_values(row, *, include_k3=True, kl_beta=None):
    values = {tag: row[k] for k, tag in TAGS.items() if k in row}
    if not include_k3:
        values = {key: value for key, value in values.items()
                  if key not in ('train/k3_penalty_mean',)}
    # Drift is useful even when KL is not part of the optimization objective.
    # Existing train/* tags remain compatibility aliases for saved dashboards.
    values.update({tag: row[k] for k, tag in SIGNAL_TAGS.items() if k in row})
    if type(row.get('policy_changed')) is bool:
        values[SIGNAL_TAGS['policy_changed']] = int(row['policy_changed'])
    from RL.grpo.signals import throughput
    values.update({SIGNAL_TAGS[k]: v for k, v in throughput(row).items()
                   if k in SIGNAL_TAGS and k not in row})
    if 'step_seconds' in row or 'seconds' in row:
        values['timing/step_seconds'] = row.get('step_seconds', row.get('seconds'))
    for key, tag in [('peak_allocated_bytes', 'peak_allocated_gib'),
                     ('peak_reserved_bytes', 'peak_reserved_gib'), ('free_bytes', 'free_gib')]:
        if key in row:
            values['memory/' + tag] = row[key] / 1024**3
    penalty = row.get('k3_penalty_mean')
    if finite_number(kl_beta) and finite_number(penalty):
        weighted = kl_beta * penalty
        values[K3_LOSS_TAG] = weighted
        loss = row.get('loss')
        if finite_number(loss) and loss != 0:
            values[K3_LOSS_FRACTION_TAG] = weighted / loss
    attribution = row.get('gradient_attribution') or {}
    for name, component in attribution.get('components', {}).items():
        for field, value in component.items():
            values[f'gradient/{name}/{field}'] = value
    for field in ('total_norm', 'source_cosine', 'source_sum_residual_norm'):
        if field in attribution:
            values[f'gradient/{field}'] = attribution[field]
    if 'kl_backward_seconds' in attribution:
        values['timing/gradient_kl_backward_seconds'] = attribution['kl_backward_seconds']
    return {k: v for k, v in values.items() if finite_number(v)}


def pinned_summary():
    import re
    from tensorboard.compat.proto import summary_pb2, tensor_pb2, types_pb2
    from tensorboard.plugins.custom_scalar import layout_pb2, metadata

    layout = layout_pb2.Layout(version=0)
    category = layout.category.add(title='Pinned')
    for title, tag in PINNED_CHARTS:
        category.chart.add(title=title).multiline.tag.append('^' + re.escape(tag) + '$')
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(
        tag=metadata.CONFIG_SUMMARY_TAG,
        metadata=metadata.create_summary_metadata(),
        tensor=tensor_pb2.TensorProto(dtype=types_pb2.DT_STRING,
                                      string_val=[layout.SerializeToString()]),
    )])


def serve():
    import sys
    from absl import app
    from tensorboard import default, main_lib, program
    from tensorboard.plugins import base_plugin
    from tensorboard.plugins.custom_scalar.custom_scalars_plugin import CustomScalarsPlugin

    class PinnedPlugin(CustomScalarsPlugin):
        def frontend_metadata(self):
            return base_plugin.FrontendMetadata(
                element_name='tf-custom-scalar-dashboard', tab_name='Pinned')

    main_lib.global_init()
    plugins = [PinnedPlugin if plugin is CustomScalarsPlugin else plugin
               for plugin in default.get_plugins()]
    tb = program.TensorBoard(plugins=plugins)
    try:
        app.run(tb.main, flags_parser=tb.configure)
    except base_plugin.FlagsError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc


def server_flags(folder, port):
    return ['--logdir', str(folder / 'visible'), '--host', '127.0.0.1',
            '--port', str(port), '--reload_interval', '2', '--load_fast', 'false']


def belongs_to_source(name, command, folder, port):
    if name == 'sync':
        return command[1:] == ['-m', SERVICE_MODULE, '--worker', str(folder.parent / 'source.json')]
    flags = server_flags(folder, port)
    return command[1:] in (['-m', SERVICE_MODULE, '--serve', *flags],
                           ['-m', 'tensorboard.main', *flags])


def owned_process(name, record, folder, *, strict=False):
    registered = record.get(name)
    if not registered:
        return None
    actual = process(registered.get('pid', -1))
    if actual is None or actual['starttime'] != registered.get('starttime'):
        return None
    if actual != registered and registered.get('command') != []:
        return None
    if belongs_to_source(name, actual['command'], folder, record.get('port')):
        return actual
    # Older launchers could snapshot an empty argv before exec. Require the
    # full command for this source before recovering any such receipt.
    if strict and registered.get('command') != []:
        raise ValueError('Registered '+name+' command does not belong to this source; refusing to signal')
    return None


def worker(descriptor):
    # Invoked only by the public CLI's owned service manager, in analysis env.
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    binding = read_json(descriptor)
    folder = Path(descriptor).parent / 'tensorboard'
    directory = Path(binding['directory'])
    cursor_path = folder / 'sync-cursor.json'
    reader = MetricReader(directory / 'metrics.jsonl')
    kl_beta = binding.get('kl_beta')
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)

    def lifecycle_reason():
        lease_until = binding.get('lease_until')
        if lease_until is not None and time.time() >= lease_until:
            return 'query_lease_expired'
        if binding.get('lifecycle') != 'training':
            return None
        try:
            attempt = read_json(directory / 'attempt.json')
        except (OSError, ValueError):
            return 'attempt_metadata_unavailable'
        if attempt.get('status') in TERMINAL_ATTEMPT_STATES:
            return 'attempt_terminal'
        live_path = folder.parent / 'live.json'
        live = read_json(live_path) if live_path.exists() else {}
        current = training_process(directory, attempt, live)
        if current:
            binding['_training_seen'] = True
            binding['_training_missing_since'] = None
            return None
        if binding.get('_training_seen'):
            missing_since = binding.get('_training_missing_since') or time.time()
            binding['_training_missing_since'] = missing_since
            if time.time() - missing_since >= 30:
                return 'training_process_exited'
        return None

    def stop_peer(reason):
        service_path = folder / 'service.json'
        record = read_json(service_path) if service_path.exists() else {}
        owned = owned_process('tensorboard', record, folder)
        if owned:
            os.kill(owned['pid'], signal.SIGTERM)
        record.update(state='stopped', stopped_at=now(), auto_stop_reason=reason)
        write_json(service_path, record)
    with file_lock(folder / 'writer.lock', nonblocking=True):
        previous = read_json(cursor_path) if cursor_path.exists() else {}
        generation = folder / ('generation-' + str(time.time_ns()))
        events = generation / 'events' / binding['attempt_id']
        writer = EventFileWriter(str(events), max_queue_size=20, flush_secs=2)
        cursor = {'schema_version': 1, 'state': 'replaying', 'generation': str(generation),
                  'offset': 0, 'last_step': 0, 'updated_at': now(),
                  'parent_checkpoint': binding.get('parent_checkpoint'),
                  'inherited_through_step': binding.get('inherited_through_step', 0),
                  'historical_wall_time_estimated': False}
        try:
            # A restart checks the previous committed prefix before replay.
            if previous.get('identity'):
                reader.identity = tuple(previous['identity'])
                reader.offset = previous['offset']
                reader.anchor = previous.get('anchor')
                reader.prefix = previous.get('prefix')
                iterator = reader.read()
                next(iterator, None)
                iterator.close()
                reader = MetricReader(reader.path)
            elif previous.get('error'):
                raise ValueError('Previous synchronization failed; use --action rebuild')
            writer.add_event(Event(wall_time=time.time(), summary=pinned_summary()))
            write_json(cursor_path, cursor)
            first = True
            while not stopped:
                for row, digest in reader.read():
                    from datetime import datetime
                    observed = row.get('observed_at')
                    wall = datetime.fromisoformat(observed).timestamp() if observed else 0.
                    if not observed:
                        cursor['historical_wall_time_estimated'] = True
                    values = scalar_values(row, include_k3=kl_beta != 0, kl_beta=kl_beta)
                    values['progress/inherited'] = int(row['global_step'] <= binding.get('inherited_through_step', 0))
                    writer.add_event(Event(wall_time=wall, step=row['global_step'],
                        summary=Summary(value=[Summary.Value(tag=k, simple_value=v) for k,v in values.items()])))
                    writer.flush()
                    cursor.update(offset=reader.offset, last_step=row['global_step'], last_line_sha256=digest,
                                  updated_at=now(), identity=reader.identity, anchor=reader.anchor, prefix=reader.prefix)
                    write_json(cursor_path, cursor)
                if first:
                    writer.flush()
                    link = folder / ('.visible-' + str(os.getpid()))
                    link.symlink_to(generation / 'events', target_is_directory=True)
                    os.replace(link, folder / 'visible')
                    first = False
                cursor.update(state='following', updated_at=now())
                write_json(cursor_path, cursor)
                for _ in range(10):
                    if stopped:
                        break
                    reason = lifecycle_reason()
                    if reason:
                        cursor['stop_reason'] = reason
                        stopped = True
                        break
                    time.sleep(.2)
            cursor.update(state='stopped', stopped_at=now())
            write_json(cursor_path, cursor)
            if cursor.get('stop_reason'):
                stop_peer(cursor['stop_reason'])
        except BaseException as exc:
            cursor.update(state='failed', error=f'{type(exc).__name__}: {exc}', updated_at=now())
            write_json(cursor_path, cursor)
            raise
        finally:
            writer.close()


def status_at(folder):
    path = folder / 'service.json'
    record = read_json(path) if path.exists() else {'state': 'not_started'}
    cursor = read_json(folder / 'sync-cursor.json') if (folder / 'sync-cursor.json').exists() else None
    if cursor and cursor.get('stop_reason') and not record.get('auto_stop_reason'):
        record['auto_stop_reason'] = cursor['stop_reason']
    alive = {name + '_alive': bool(owned_process(name, record, folder)) for name in ('sync', 'tensorboard')}
    visible = folder / 'visible'
    event_files = list(visible.glob('*/events.out.tfevents.*')) if visible.exists() else []
    event_mtime = max((p.stat().st_mtime for p in event_files), default=None)
    return {**record, **alive, 'running': all(alive.values()), 'sync_status': cursor,
            'event_updated_unix': event_mtime,
            'event_age_seconds': time.time()-event_mtime if event_mtime else None}


def stop_owned(folder, record):
    for name in ('tensorboard', 'sync'):
        owned = owned_process(name, record, folder, strict=True)
        if not owned:
            continue
        # Identity contains exact argv AND proc starttime, excluding PID reuse.
        os.kill(owned['pid'], signal.SIGTERM)
        deadline = time.monotonic() + 10
        while matches(owned) and time.monotonic() < deadline:
            time.sleep(.1)
        if matches(owned):
            os.kill(owned['pid'], signal.SIGKILL)
            deadline = time.monotonic()+3
            while matches(owned) and time.monotonic() < deadline:
                time.sleep(.1)
        if matches(owned):
            raise RuntimeError('Owned process did not exit: ' + name)
    record.update(state='stopped', stopped_at=now(), stop_receipt={'confirmed_exited': True, 'at': now()})
    write_json(folder / 'service.json', record)


def manage(cfg, attempt_id, *, action='status', storage_root=None, port=16006, host='127.0.0.1',
           lease_seconds=DEFAULT_QUERY_LEASE_SECONDS):
    directory, attempt, identity, cache = source(cfg, attempt_id, storage_root)
    folder = cache / 'tensorboard'
    if not isinstance(lease_seconds, int) or not 1 <= lease_seconds <= MAX_QUERY_LEASE_SECONDS:
        raise ValueError(f'lease_seconds must be between 1 and {MAX_QUERY_LEASE_SECONDS}')
    if action == 'status':
        return status_at(folder)
    if host != '127.0.0.1':
        raise ValueError('RL TensorBoard must bind 127.0.0.1; use SSH forwarding')
    if not 1 <= port <= 65535:
        raise ValueError('Port must be between 1 and 65535')
    with file_lock(folder / 'service.lock', nonblocking=True):
        record = status_at(folder)
        if action == 'stop':
            stop_owned(folder, record)
            return status_at(folder)
        if action == 'renew':
            if not record['running']:
                raise ValueError('Cannot renew a stopped TensorBoard service')
            if record.get('lifecycle') != 'query_lease':
                return record
            record['lease_until'] = time.time() + lease_seconds
            record['lease_seconds'] = lease_seconds
            write_json(folder / 'service.json', record)
            binding = read_json(cache / 'source.json')
            binding['lease_until'] = record['lease_until']
            write_json(cache / 'source.json', binding)
            return status_at(folder)
        if record['running'] and action == 'start':
            if record.get('port') != port or record.get('host') != host:
                raise ValueError('Service already running on a different port/host; stop it first')
            return record
        stop_owned(folder, record)
        if action == 'rebuild':
            (folder / 'sync-cursor.json').unlink(missing_ok=True)
        python = Path(cfg['project_root']) / '.venvs/analysis/bin/python'
        if not python.exists():
            raise ValueError('Missing .venvs/analysis; install locked TensorBoard analysis environment')
        check = subprocess.run([str(python), '-c', 'import tensorboard; assert tensorboard.__version__ == "2.20.0"; import ijson'],
                               capture_output=True, text=True, timeout=15)
        if check.returncode:
            raise ValueError('Analysis requires tensorboard==2.20.0 and ijson==3.5.1: ' + check.stderr[-1000:])
        with socket.socket() as test:
            test.bind((host, port))
        lifecycle = 'training' if training_process(directory, attempt, {}) else 'query_lease'
        lease_until = time.time() + lease_seconds if lifecycle == 'query_lease' else None
        identity.update(lifecycle=lifecycle, lease_until=lease_until,
                        lease_seconds=lease_seconds if lease_until else None,
                        kl_beta=((attempt.get('contract') or {}).get('schedule') or {}).get('grpo', {}).get('kl_beta'),
                        parent_checkpoint=attempt.get('parent'),
                        inherited_through_step=attempt.get('initial_state', {}).get('global_step', 0))
        write_json(cache / 'source.json', identity)
        base = [str(python), '-m', SERVICE_MODULE]
        record = {'schema_version': 1, 'state': 'starting', 'attempt_id': attempt_id,
                  'host': host, 'port': port, 'logdir': str(folder / 'visible'), 'started_at': now(),
                  'url': f'http://127.0.0.1:{port}',
                  'ssh_forward': f'ssh -N -L {port}:127.0.0.1:{port} wangtianyi@10.249.187.9',
                  'lifecycle': lifecycle, 'lease_until': lease_until,
                  'lease_seconds': lease_seconds if lease_until else None}
        def launch(name, command):
            with (folder / (name + '.log')).open('ab') as log:
                child = subprocess.Popen(command, cwd=cfg['project_root'], stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 5
            while child.poll() is None and time.monotonic() < deadline:
                owned = process(child.pid)
                if owned and owned['command'] == command:
                    break
                time.sleep(.02)
            else:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=3)
                raise RuntimeError(name + ' did not exec the expected command during startup')
            record[name] = owned
            write_json(folder / 'service.json', record)
            return child
        try:
            sync = launch('sync', base + ['--worker', str(cache / 'source.json')])
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if sync.poll() is not None:
                    raise RuntimeError('Synchronizer failed: ' + json.dumps(status_at(folder).get('sync_status')))
                cursor = read_json(folder / 'sync-cursor.json') if (folder / 'sync-cursor.json').exists() else {}
                if cursor.get('state') == 'following' and cursor.get('updated_at', '') >= record['started_at']:
                    break
                time.sleep(.2)
            else:
                raise TimeoutError('Historical replay not ready within 180 seconds')
            tb = launch('tensorboard', base + ['--serve', *server_flags(folder, port)])
            deadline = time.monotonic()+25
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while time.monotonic() < deadline:
                if tb.poll() is not None:
                    raise RuntimeError('TensorBoard exited during startup')
                try:
                    with opener.open(record['url'], timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(.2)
            else:
                raise TimeoutError('TensorBoard HTTP readiness timed out')
            record.update(state='running', ready_at=now())
            write_json(folder / 'service.json', record)
        except BaseException:
            stop_owned(folder, record)
            raise
        return status_at(folder)


if __name__ == '__main__':
    import sys
    if sys.argv[1:2] == ['--serve']:
        del sys.argv[1]  # TensorBoard parses the remaining arguments.
        serve()
    else:
        parser = argparse.ArgumentParser(description='Internal CLI-managed RL observability worker')
        parser.add_argument('--worker', required=True)
        worker(parser.parse_args().worker)

"""Explicit, audited offline revisions of bounded run resource settings."""
from contextlib import ExitStack
from pathlib import Path
import copy
import json

import yaml

from application.execution.run_plan import load_plan
from core.config import Capture, CaptureExport, Parallel, Serving
from core.io import hash_json, identifier, now, read_json, sha256, write_json
from core.locking import file_lock
from core.processes import matches
from hidden.relocate_spool import assert_no_writers

FIELDS = ('spool_max_bytes', 'spool_high_watermark', 'spool_low_watermark')


def revise_capture_budget(root, run_id, *, revision_id, max_bytes=None, high=None, low=None, reason,
                          control_seconds=None, receipt_workers=None, runtime_values=None,
                          serving_gpu_memory_utilization=None, serving_placement=None):
    root = Path(root).resolve()
    run_id, revision_id = identifier(run_id), identifier(revision_id)
    if not reason or not reason.strip():
        raise ValueError('Explicit authorization/reason is required')
    run = root / 'runs' / run_id
    runtime = runtime_values is not None
    control = control_seconds is not None or receipt_workers is not None
    placement = serving_placement is not None
    serving = serving_gpu_memory_utilization is not None or placement
    if serving:
        if runtime or control or any(v is not None for v in (max_bytes, high, low)):
            raise ValueError('Serving memory revision cannot mix resource revision modes')
        if placement:
            if serving_gpu_memory_utilization is not None:
                raise ValueError('Use one serving revision mode')
            required = {'serving.gpus', 'serving.base_url'}
            sharded = required | {'serving.trajectory_sharding', 'serving.tensor_parallel_size',
                                  'serving.replica_base_urls', 'serving.max_cpu_workers_per_gpu',
                                  'serving.gpu_memory_utilization'}
            if set(serving_placement) not in (required, sharded):
                raise ValueError('Placement accepts devices/endpoint or the complete per_gpu_tp1 profile')
            if set(serving_placement) == sharded and (
                    serving_placement['serving.trajectory_sharding'] != 'per_gpu_tp1' or
                    serving_placement['serving.tensor_parallel_size'] != 1):
                raise ValueError('Topology conversion requires per_gpu_tp1 and TP=1')
            values = dict(serving_placement)
        else:
            values = {'serving.gpu_memory_utilization': serving_gpu_memory_utilization}
    elif runtime:
        values = dict(runtime_values)
        if set(values) != set(CONCURRENCY64_VALUES):
            raise ValueError('Only the declared concurrency profile fields may be revised')
    elif control:
        if control_seconds is None or receipt_workers is None or any(v is not None for v in (max_bytes, high, low)):
            raise ValueError("Specify control seconds and receipt workers together, without budget values")
        values = {"control_seconds": control_seconds, "receipt_workers": receipt_workers}
    else:
        values = dict(zip(FIELDS, (max_bytes, high, low)))
    field_prefix = "" if runtime or serving else "capture." if control else "capture.export."
    out = run / 'config-revisions' / revision_id
    with ExitStack() as locks:
        for path in [root / 'state/run-locks' / (run_id + '.lock'),
                     run / 'execution/supervisor.lock', run / 'execution/active-supervisor.lock']:
            locks.enter_context(file_lock(path, nonblocking=True))
        loaded = load_plan(root, run_id)
        if loaded['plan']['operation'] != 'rollout':
            raise ValueError('Capture budget revision requires a rollout run')
        latest = run / 'execution/latest.json'
        execution = None
        if latest.exists():
            execution = read_json(latest)['attempt']
            previous = run / 'execution/attempts' / identifier(execution)
            for name in ['process.json', 'supervisor.json']:
                if (previous / name).exists() and matches(read_json(previous / name)):
                    raise RuntimeError('Execution still active; cancel/wait before revising')
            if not (previous / 'exit.json').exists():
                raise RuntimeError('Previous execution has no exit receipt')
            term = previous / 'termination.json'
            if not term.exists() or not read_json(term).get('cleanup_complete'):
                raise RuntimeError('Previous execution cleanup is incomplete')
        if not (runtime or control or serving):
            assert_no_writers(root, [run_id])
        if (out / 'receipt.json').exists():
            receipt = read_json(out / 'receipt.json')
            if receipt['values'] != values or receipt['reason'] != reason:
                raise ValueError('Revision ID already binds different values/reason')
            if sha256(run / 'resolved-config.json') != receipt['after_files']['resolved-config.json']:
                raise ValueError('Run has a later revision; refusing to reuse this ID')
            return receipt
        if out.exists():
            raise RuntimeError('Incomplete revision audit exists; inspect before recovery')
        old = copy.deepcopy(loaded['config'])
        candidate = copy.deepcopy(old)
        if runtime or serving:
            for dotted, value in values.items():
                node = candidate
                keys = dotted.split('.')
                for key in keys[:-1]:
                    node = node[key]
                node[keys[-1]] = value
            if serving:
                Serving.model_validate(candidate['serving'])
                if placement and candidate['serving'].get('trajectory_sharding', 'none') != 'per_gpu_tp1' and len(candidate['serving']['gpus']) != candidate['serving']['tensor_parallel_size']:
                    raise ValueError('Placement must preserve the frozen tensor parallel size')
                if 'serving.gpu_memory_utilization' in values and candidate['serving'].get('kv_cache_memory_bytes') is not None:
                    raise ValueError('Explicit KV cache bytes override gpu_memory_utilization; revise the resource policy first')
            else:
                Capture.model_validate(candidate['capture'])
                if candidate.get('parallel_policy'):
                    from core.parallel_policy import validate_effective_parallel

                    validate_effective_parallel(candidate)
                else:
                    Parallel.model_validate({k:v for k,v in candidate['parallel'].items() if k != 'estimator_sha256'})
        elif control:
            candidate['capture'].update(values)
            Capture.model_validate(candidate['capture'])
        else:
            candidate['capture']['export'].update(values)
            CaptureExport.model_validate(candidate['capture']['export'])
        candidate['config_hash'] = hash_json({k: v for k, v in candidate.items()
                                            if k not in {'config_hash', 'selection', 'selection_sha256'}})
        plan = copy.deepcopy(loaded['plan'])
        plan.update(config_hash=candidate['config_hash'], config_revision=revision_id,
                    previous_plan_sha256=loaded['plan']['plan_sha256'])
        plan['plan_sha256'] = hash_json({k: v for k, v in plan.items() if k != 'plan_sha256'})
        human = yaml.safe_load((run / 'config.yaml').read_text())
        if runtime or serving:
            for dotted, value in values.items():
                node = human
                keys = dotted.split('.')
                for key in keys[:-1]:
                    node = node.setdefault(key, {})
                node[keys[-1]] = value
        elif control:
            human['capture'].update(values)
        else:
            human['capture']['export'].update(values)
        files = ['resolved-config.json', 'plan.json', 'config.yaml']
        originals = {name: (run / name).read_bytes() for name in files}
        protected = {name: sha256(run / name) for name in
                     ['selection.json', 'tasks.json', 'rollout/work-items.json', 'evidence.json']
                     if (run / name).exists()}
        proposed = {'resolved-config.json': (json.dumps(candidate, ensure_ascii=False, indent=2) + '\n').encode(),
                    'plan.json': (json.dumps(plan, ensure_ascii=False, indent=2) + '\n').encode(),
                    'config.yaml': yaml.safe_dump(human, sort_keys=False, allow_unicode=True).encode()}
        out.mkdir(parents=True)
        for phase, content in [('before', originals), ('after', proposed)]:
            (out / phase).mkdir()
            for name, value in content.items():
                (out / phase / name).write_bytes(value)
        write_json(out / 'progress.json', {'phase': 'prepared', 'prepared_at': now()})
        try:
            for name, value in proposed.items():
                # Atomic per-file replacement; run locks exclude all execution.
                temp = run / (name + '.budget-revision.part')
                temp.write_bytes(value)
                temp.replace(run / name)
            verified = load_plan(root, run_id)
            assert verified['config'] == candidate
            assert all(sha256(run / name) == digest for name, digest in protected.items())
        except BaseException:
            for name, value in originals.items():
                (run / name).write_bytes(value)
            write_json(out / 'progress.json', {'phase': 'rolled_back', 'at': now()})
            raise
        receipt = {'schema_version': 1, 'run_id': run_id, 'revision_id': revision_id,
                   'reason': reason, 'previous_execution': execution, 'values': values,
                   'changed_config_fields': [field_prefix + name for name in values],
                   'old_config_hash': old['config_hash'], 'new_config_hash': candidate['config_hash'],
                   'old_plan_sha256': loaded['plan']['plan_sha256'], 'new_plan_sha256': plan['plan_sha256'],
                   'before_files': {name: sha256(out / 'before' / name) for name in files},
                   'after_files': {name: sha256(run / name) for name in files},
                   'protected_files_unchanged': protected, 'completed_at': now()}
        write_json(out / 'receipt.json', receipt)
        write_json(run / 'events' / ('config-revision-' + revision_id + '.json'),
                   {'event': 'run.serving_placement_revised' if placement else 'run.serving_memory_revised' if serving else 'run.concurrency_profile_revised' if runtime else 'run.capture_control_revised' if control else 'run.capture_budget_revised', **receipt})
        write_json(out / 'progress.json', {'phase': 'complete', 'completed_at': now()})
        return receipt


def revise_capture_control(root, run_id, *, revision_id, control_seconds, receipt_workers, reason):
    """Use the same offline locking/audit mechanism for capture control resources."""
    return revise_capture_budget(root, run_id, revision_id=revision_id, reason=reason,
                                 control_seconds=control_seconds, receipt_workers=receipt_workers)


CONCURRENCY64_VALUES = {'capture.control_seconds': 600, 'capture.receipt_workers': 64, 'capture.control_workers': 16, 'capture.flush_timeout_seconds': 1800, 'capture.finalize_seconds': 900, 'capture.finalize_request_seconds': 7200, 'capture.admission_seconds': 21600, 'capture.export.upload_max_elapsed_seconds': 3600, 'capture.export.queue_size': 2048, 'parallel.control_request_seconds': 600, 'parallel.results_wait_seconds': 600, 'parallel.result_workers': 16, 'parallel.score_timeout_ms': 5000, 'parallel.decision_ttl_seconds': 64800}

def revise_concurrency(root, run_id, *, revision_id, reason):
    return revise_capture_budget(root, run_id, revision_id=revision_id, reason=reason,
                                 runtime_values=CONCURRENCY64_VALUES)


def revise_serving_memory(root, run_id, *, revision_id, gpu_memory_utilization, reason):
    """Revise only the executor memory fraction under the existing offline audit."""
    return revise_capture_budget(root, run_id, revision_id=revision_id, reason=reason,
                                 serving_gpu_memory_utilization=gpu_memory_utilization)


def revise_serving_placement(root, run_id, *, revision_id, gpus, base_url, reason,
                             trajectory_sharding=None, replica_base_urls=None,
                             max_cpu_workers_per_gpu=None, gpu_memory_utilization=None):
    """Audit a device move or explicit conversion to managed per-GPU TP=1."""
    values = {'serving.gpus': gpus, 'serving.base_url': base_url}
    options = (trajectory_sharding, replica_base_urls, max_cpu_workers_per_gpu,
               gpu_memory_utilization)
    if any(value is not None for value in options):
        if any(value is None for value in options) or trajectory_sharding != 'per_gpu_tp1':
            raise ValueError('Specify the complete per_gpu_tp1 profile, including memory and CPU slots')
        values.update({'serving.trajectory_sharding': trajectory_sharding,
                       'serving.tensor_parallel_size': 1,
                       'serving.replica_base_urls': replica_base_urls,
                       'serving.max_cpu_workers_per_gpu': max_cpu_workers_per_gpu,
                       'serving.gpu_memory_utilization': gpu_memory_utilization})
    return revise_capture_budget(root, run_id, revision_id=revision_id, reason=reason,
                                 serving_placement=values)

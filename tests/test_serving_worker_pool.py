"""Multi-GPU TP1 scheduling, resource ownership, and frozen configuration."""
import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from core.config import Serving, assert_frozen, resolve
from core.io import hash_json, read_json
from runtime.serving.pool import ServingPool


def sharded(cfg, cap=2):
    cfg = copy.deepcopy(cfg)
    cfg['serving'].update(
        management='managed', gpus=[2, 3], tensor_parallel_size=1,
        trajectory_sharding='per_gpu_tp1', max_cpu_workers_per_gpu=cap,
        base_url='http://127.0.0.1:18200/v1',
        replica_base_urls=['http://127.0.0.1:18200/v1', 'http://127.0.0.1:18201/v1'],
    )
    cfg['execution']['rollout_workers'] = 1  # New pool derives its own capacity.
    cfg['config_hash'] = hash_json({k: v for k, v in cfg.items() if k != 'config_hash'})
    return cfg


def services(monkeypatch, fail=None):
    instances = []
    controller = SimpleNamespace(cancel_run=lambda run: None, spool=None)

    class Manager:
        def __init__(self, cfg, output, run_id=None, capture_owner=None):
            self.cfg, self.output, self.run_id, self.owner = cfg, output, run_id, capture_owner
            self.runtime = {'capture_controller': controller, 'capture_control_url': 'shared-control'}
            self.closed = False
            instances.append(self)

        def __enter__(self):
            if len(instances) == fail:
                raise RuntimeError('startup failed')
            return self

        def __exit__(self, *args):
            self.closed = True

    monkeypatch.setattr('runtime.serving.pool.ServingManager', Manager)
    return instances


def test_sharding_validation_and_legacy_identity(cfg):
    import yaml
    from pathlib import Path
    source = Path(cfg['config_source'])
    raw = yaml.safe_load(source.read_text())
    for field in ('trajectory_sharding', 'replica_base_urls', 'max_cpu_workers_per_gpu', 'startup_mode'):
        raw['serving'].pop(field, None)
    source.write_text(yaml.safe_dump(raw))
    original = resolve(source)
    assert 'trajectory_sharding' not in original['serving']
    assert 'replica_base_urls' not in original['serving']
    assert 'max_cpu_workers_per_gpu' not in original['serving']
    Serving.model_validate(sharded(cfg)['serving'])
    mig = sharded(cfg)
    mig['serving']['gpus'] = ['MIG-' + 'a' * 36, 'MIG-' + 'b' * 36]
    mig['serving']['replica_base_urls'] = ['http://127.0.0.1:18200/v1', 'http://127.0.0.1:18201/v1']
    Serving.model_validate(mig['serving'])
    with pytest.raises(ValueError, match='exactly one'):
        from runtime.serving.resources import normalize_devices
        normalize_devices(mig['serving']['gpus'])
    for changes in [
        {'tensor_parallel_size': 2}, {'management': 'external'},
        {'max_cpu_workers_per_gpu': 0}, {'replica_base_urls': []},
        {'replica_base_urls': ['http://127.0.0.1:18200/v1', 'http://localhost:18200/v1']},
        {'replica_base_urls': ['http://127.0.0.1:18200/v1', 'http://example.com:18201/v1']},
    ]:
        bad = {**sharded(cfg)['serving'], **changes}
        with pytest.raises(ValueError):
            Serving.model_validate(bad)


def test_pool_capacity_reuses_free_gpu_and_preserves_snapshot(cfg, monkeypatch, tmp_path):
    instances = services(monkeypatch)
    frozen = sharded(cfg, cap=2)
    original = copy.deepcopy(frozen)
    with ServingPool(frozen, tmp_path / 'serving', 'pool') as pool:
        assert pool.capacity == 4
        assert [m.cfg['serving']['gpus'] for m in instances] == [[2], [3]]
        assert instances[0].owner is None
        assert instances[1].owner is instances[0]
        assert all(m.run_id == 'pool' for m in instances)
        workers = [pool.acquire() for _ in range(4)]
        assert [w.index for w in workers] == [0, 1, 0, 1]
        assert pool.acquire() is None
        pool.release(workers[1])
        assert pool.acquire().index == 1
        assert read_json(tmp_path / 'serving/worker-pool.json')['workers'][0]['active_cpu_workers'] == 2
    assert all(m.closed for m in instances)
    assert frozen == original
    assert_frozen(frozen)


def test_pool_startup_failure_closes_previously_started_services(cfg, monkeypatch, tmp_path):
    instances = services(monkeypatch, fail=2)
    with pytest.raises(RuntimeError, match='startup failed'):
        with ServingPool(sharded(cfg), tmp_path, 'pool'):
            pytest.fail('entered failed pool')
    assert instances[0].closed


def test_only_capture_owner_closes_shared_capture_controller(cfg, tmp_path):
    from runtime.serving.manager import ServingManager
    closed = []
    server = SimpleNamespace(close=lambda: closed.append(True))
    owner = ServingManager(cfg, tmp_path / 'owner', run_id='pool')
    replica = ServingManager(cfg, tmp_path / 'replica', run_id='pool', capture_owner=owner)
    owner.capture_server = replica.capture_server = server
    replica.__exit__(None, None, None)
    assert closed == []
    owner.__exit__(None, None, None)
    assert closed == [True]


def test_frozen_clone_accepts_sharding_and_freezes_per_gpu_capacity(cfg):
    from application.execution.run_plan import create_run, load_plan
    from tests.test_limits_retry import prepare
    # Management is retained from the source; only supported deployment fields change.
    cfg = copy.deepcopy(cfg)
    cfg['serving']['management'] = 'managed'
    cfg['config_hash'] = hash_json({k: v for k, v in cfg.items() if k != 'config_hash'})
    source, _ = prepare(cfg, run_id='managed-source-sharding', tasks=('a', 'b'))
    clone = create_run(cfg, 'clone-sharding', operation='rollout', from_run=source.name,
                       clone_overrides=(
                           'serving.trajectory_sharding=per_gpu_tp1',
                           'serving.gpus=[2,3]', 'serving.tensor_parallel_size=1',
                           'serving.base_url=http://127.0.0.1:18200/v1',
                           'serving.replica_base_urls=[http://127.0.0.1:18200/v1,http://127.0.0.1:18201/v1]',
                           'serving.max_cpu_workers_per_gpu=3',
                       ))
    loaded = load_plan(cfg['project_root'], clone.name)
    assert loaded['config']['serving']['max_cpu_workers_per_gpu'] == 3
    assert loaded['config']['serving']['gpus'] == [2, 3]
    assert_frozen(loaded['config'])


def test_cancel_reaches_all_engines_even_when_one_fails(cfg, monkeypatch, tmp_path):
    services(monkeypatch)
    endpoints = []

    def post(url, **kwargs):
        endpoints.append(url)
        if ':18200/' in url:
            raise RuntimeError('unavailable')
        return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr('runtime.serving.pool.requests.post', post)
    cfg = sharded(cfg)
    cfg['capture']['enabled'] = True
    with ServingPool(cfg, tmp_path, 'pool') as pool:
        errors = pool.cancel('pool', '000')
    assert len(endpoints) == 2
    assert len(errors) == 1
    assert errors[0]['worker_id'] == 0


def test_real_scheduler_refills_free_gpu_without_batch_barrier(cfg, monkeypatch):
    from rollout import runner
    from tests.test_limits_retry import fake_runtime, prepare

    cfg = sharded(cfg, cap=1)
    run, loaded = prepare(cfg, run_id='sharded-refill', tasks=('a', 'b', 'c'))
    slow, fast, next_task = [i['instance_id'] for i in loaded['work_items']['items']]
    factory, calls, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    services(monkeypatch)
    release_slow = threading.Event()
    c_started = threading.Event()
    order = []
    lock = threading.Lock()
    real_execute = runner._execute_attempt

    def execute(cfg, run_id, task, factory, runtime, sample, context, out):
        with lock:
            order.append((sample.instance_id, runtime['serving_worker_gpu'], cfg['serving']['base_url']))
        if sample.instance_id == slow:
            assert release_slow.wait(10)
        elif sample.instance_id == next_task:
            c_started.set()
        return real_execute(cfg, run_id, task, factory, runtime, sample, context, out)

    monkeypatch.setattr(runner, '_execute_attempt', execute)
    with ThreadPoolExecutor(max_workers=1) as outer:
        future = outer.submit(runner.run_manifest, cfg, run.name, True, factory)
        try:
            assert c_started.wait(10), 'free GPU did not receive c while a was running'
        finally:
            release_slow.set()
        result = future.result(timeout=10)
    assert len(result) == 3
    assignments = {iid: (gpu, url) for iid, gpu, url in order}
    assert assignments[slow] == (2, 'http://127.0.0.1:18200/v1')
    assert assignments[fast] == assignments[next_task] == (3, 'http://127.0.0.1:18201/v1')
    assert len(calls) == 3
    assert all('serving_assignment' in r for r in result.values())
    assert read_json(run / 'serving/worker-pool.json')['capacity'] == 2


def test_real_scheduler_obeys_per_gpu_cpu_limit(cfg, monkeypatch):
    from rollout import runner
    from tests.test_limits_retry import fake_runtime, prepare

    cfg = sharded(cfg, cap=2)
    run, _ = prepare(cfg, run_id='sharded-cap', tasks=tuple('abcdef'))
    factory, _, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    services(monkeypatch)
    barrier = threading.Barrier(4)
    lock = threading.Lock()
    active, peak = {}, {}
    started = []
    real_execute = runner._execute_attempt

    def execute(cfg, run_id, task, factory, runtime, sample, context, out):
        gpu = runtime['serving_worker_gpu']
        with lock:
            active[gpu] = active.get(gpu, 0) + 1
            peak[gpu] = max(peak.get(gpu, 0), active[gpu])
            started.append(sample.instance_id)
            first = len(started) <= 4
        try:
            if first:
                barrier.wait(timeout=10)
            return real_execute(cfg, run_id, task, factory, runtime, sample, context, out)
        finally:
            with lock:
                active[gpu] -= 1

    monkeypatch.setattr(runner, '_execute_attempt', execute)
    result = runner.run_manifest(cfg, run.name, resume=True, factory=factory)
    assert len(result) == 6
    assert peak == {2: 2, 3: 2}

from rollout.startup_failure_gate import StartupFailureGate


def result(i, status):
    return {'sample_key': 'sample-' + str(i), 'attempt_id': 'attempt-0', 'exit_status': status}


def test_threshold_counts_physical_startup_failures_and_deduplicates():
    gate = StartupFailureGate(3)
    assert not gate.observe(result(1, 'EnvironmentSetupFailed'))
    assert not gate.observe(result(1, 'EnvironmentSetupFailed'))
    assert not gate.observe(result(2, 'CaptureError'))
    assert gate.observe(result(3, 'WorkerFailure'))
    proof = gate.receipt('one-run')
    assert proof['blocked'] and proof['consecutive_startup_failures'] == 3
    assert len(proof['failures']) == 3


def test_model_failures_reset_infrastructure_count_and_waits_do_not_count():
    gate = StartupFailureGate(2)
    for status in ('reasoning_loop', 'LimitsExceeded', 'OpenHandsError', 'Submitted'):
        assert not gate.observe(result(status, 'CaptureError'))
        assert not gate.observe(result(status + '-wait', 'AdmissionWaiting'))
        assert not gate.observe(result(status + '-model', status))
        assert gate.consecutive == 0


def test_legacy_snapshots_without_limit_do_not_stop_dispatch():
    gate = StartupFailureGate(None)
    for i in range(10):
        assert not gate.observe(result(i, 'WorkerFailure'))


def test_threshold_remains_latched_while_existing_attempts_finish():
    gate = StartupFailureGate(1)
    assert gate.observe(result(1, 'EnvironmentSetupFailed'))
    assert gate.observe(result(2, 'Submitted'))
    assert gate.receipt('run')['blocked']


def test_runner_stops_only_its_pending_dispatch_after_frozen_threshold(cfg, monkeypatch):
    from tests.test_limits_retry import configure, fake_runtime, prepare
    from core.io import hash_json, read_json
    from rollout import runner
    cfg = configure(cfg, workers=1, retry=False)
    cfg['execution']['startup_failure_limit'] = 2
    cfg['config_hash'] = hash_json({k: v for k, v in cfg.items() if k != 'config_hash'})
    run, _ = prepare(cfg, tasks=('a', 'b', 'c', 'd'))
    factory, _, _, _ = fake_runtime(monkeypatch, ['Submitted'])
    executed = []
    def failed(*args, **kwargs):
        sample = kwargs.get('sample') or args[-1]
        executed.append(sample.instance_id)
        return {**sample.as_dict(), 'attempt_id': 'attempt-0',
                'exit_status': 'WorkerFailure', 'disposition': 'failed'}
    monkeypatch.setattr(runner, 'admitted_task', failed)
    runner.run_manifest(cfg, run.name, resume=True, factory=factory)
    assert len(executed) == 2
    assert read_json(run / 'status.json')['reason'] == 'StartupFailureThreshold'
    assert read_json(run / 'rollout/startup-failure-gate.json')['blocked']

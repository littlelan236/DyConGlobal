"""Run-local report projection."""

from __future__ import annotations

from core.io import read_json

from .run_plan import load_plan


def webarena_statistics(loaded):
    """Count every planned sample, including unfinished and ungraded attempts."""
    from application.execution.evidence import RunEvidence
    from rollout.retry import sample_accepted

    run, cfg = loaded['run'], loaded['config']
    evidence = RunEvidence(cfg['project_root'], loaded['plan']['run_id']).read()
    counts = dict.fromkeys(('resolved', 'unresolved', 'infra_error', 'ungraded', 'pending', 'unsubmitted'), 0)
    items = loaded['work_items']['items']
    for item in items:
        path = run / 'rollout/tasks' / item['instance_id'] / 'repetitions' / item['repetition_id'] / 'result.json'
        if not path.exists():
            counts['pending'] += 1
            continue
        result = read_json(path)
        if result.get('sample_key') != item['sample_key']:
            raise ValueError('Benchmark report sample identity changed')
        row = evidence['rollouts'].get(result.get('rollout_id'))
        if not sample_accepted(cfg, result):
            counts['infra_error'] += 1
        elif not row or row.get('submission_status') != 'submitted':
            counts['unsubmitted'] += 1
        elif row.get('evaluation_status') == 'completed':
            counts['resolved' if row.get('resolved') is True else 'unresolved'] += 1
        elif row.get('evaluation_status') in {'infra_error', 'ConnectionError'}:
            counts['infra_error'] += 1
        else:
            counts['ungraded'] += 1
    return {'benchmark': 'webarena_verified', 'revision': cfg['execution_context']['revision'],
            'planned_samples': len(items), 'success_rate_denominator': len(items),
            'success_rate': counts['resolved'] / len(items) if items else None,
            **counts, 'policy': 'selected evaluation of the final accepted attempt; all planned samples remain in denominator'}


def report(root, run_id):
    loaded = load_plan(root, run_id)
    run = loaded["run"]
    result = {"run_id": run_id, "operation": loaded["plan"]["operation"]}
    for name in ("status.json", "progress.json", "harness/summary.json", "artifacts.json"):
        path = run / name
        if path.exists():
            result[name.replace("/", "_").removesuffix(".json")] = read_json(path)
    result["paths"] = {"run": str(run), "tasks": str(run / "tasks.json")}
    from rollout.retry import count_projection, sampling_notice
    phases = run / "rollout/phases.json"
    if phases.exists():
        result["rollout_phases"] = read_json(phases)
        result.setdefault("progress", {}).update(result["rollout_phases"])
    result["rollout_counts"] = count_projection(run)
    result["sampling_notice"] = sampling_notice(loaded["config"])
    from rollout.retry import attempt_usage
    result["all_attempt_usage"] = attempt_usage(run)
    from runtime.containers.dependencies import failure_status
    result['environment_failures'] = failure_status(root, run_id)
    from .supervisor import cancellation_status
    result["cancellation"] = cancellation_status(run)
    workers = run / "serving/worker-pool.json"
    if workers.exists():
        result["serving_workers"] = read_json(workers)
    if (loaded['config'].get('benchmark', {}).get('backend') == 'webarena_verified'
            and loaded['plan']['operation'] == 'rollout'):
        result['benchmark_statistics'] = webarena_statistics(loaded)
    return result

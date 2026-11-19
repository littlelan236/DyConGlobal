"""Complete task ledger even when execution or external audit is incomplete."""
from collections import Counter
from pathlib import Path

from application.execution.run_plan import load_plan
from core.io import hash_json, identifier, now, read_json, sha256, write_json


def collect(root, run_id, output, *, audit_dir=None):
    root = Path(root).resolve()
    loaded = load_plan(root, run_id)
    run = loaded['run']
    harness = run / 'harness'
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(root / 'runs'):
        raise ValueError('Acceptance requires a new output directory outside runs')
    selection = read_json(harness / 'selection.json')
    summary_path = harness / 'summary.json'
    finished = summary_path.exists()
    summary = read_json(summary_path) if finished else {'outcomes': read_json(harness / 'progress.json') if (harness / 'progress.json').exists() else {}}
    records = {}
    for rid in selection['rollout_ids']:
        selected = selection['rollouts'][rid]
        task = selection['tasks'][selected['task_key']]['instance']
        iid = task['instance_id']
        out = summary.get('outcomes', {}).get(rid, {})
        audit = {}
        errors = []
        ref = out.get('verifier_audit', {})
        path = (Path(audit_dir) / 'tasks' / (rid + '.json') if audit_dir else Path(ref['path']) if ref.get('path') else None)
        if path and path.is_file():
            if not audit_dir and ref.get('sha256') != sha256(path):
                errors.append('Audit reference digest mismatch')
            else:
                audit = read_json(path)
        else:
            errors.append('Physical attempt audit missing')
        coverage = audit.get('coverage', {})
        required = coverage.get('required', 0)
        valid = bool(not errors and audit.get('classification') == 'ValidOutcome'
                     and audit.get('effective_resolved') is True and required > 0
                     and coverage.get('state_counts') == {'PASSED': required})
        attempts = out.get('attempts', [])
        environment = {}
        environment_hashes = {}
        if attempts:
            folder = harness / 'environment' / attempts[-1]['evaluation_task_id'] / iid
            for name in ['container-spec.json', 'environment-policy.json', 'environment.json',
                         'environment-after-eval.json', 'eval-execution.json']:
                target = folder / name
                if target.exists():
                    environment[name] = read_json(target)
                    environment_hashes[name] = sha256(target)
        policy = environment.get('environment-policy.json', {})
        probe = environment.get('environment.json', {})
        spec = environment.get('container-spec.json', {})
        execution = environment.get('eval-execution.json', {})
        scheme = {'image': task.get('nas_image'), 'python': probe.get('python'),
                  'packages_hash': probe.get('packages_hash'), 'requirements': policy.get('specs'),
                  'variables': policy.get('environment', {}).get('variables'),
                  'network_mode': spec.get('network_mode'),
                  'runner': execution.get('test_command', {}).get('runner'),
                  'plugin_sha256': (execution.get('pytest_binding') or {}).get('plugin_sha256'),
                  'resources': policy.get('environment', {}).get('variables', {}).get('DYCON_SWEBENCH_RESOURCES')}
        row = {'instance_id': iid, 'rollout_id': rid, 'execution_finished': bool(out),
               'physical_attempt_count': len(attempts), 'gold_valid': valid,
               'classification': audit.get('classification', 'Review'),
               'reason_code': audit.get('reason_code', 'execution_evidence_incomplete'),
               'reward': int(audit['effective_resolved']) if not errors and audit.get('classification') == 'ValidOutcome' and isinstance(audit.get('effective_resolved'), bool) else None,
               'official_result': audit.get('official_result', {}), 'coverage': coverage,
               'audit_path': str(path) if path else None, 'errors': errors,
               'missing_evidence': audit.get('missing_evidence', []),
               'environment_hashes': environment_hashes,
               'scheme': scheme, 'scheme_id': hash_json(scheme)}
        if iid in records:
            raise ValueError('Duplicate physical task in acceptance ledger')
        records[iid] = row
    audit_complete = (read_json(Path(audit_dir) / 'summary.json').get('complete')
                      if audit_dir and (Path(audit_dir) / 'summary.json').exists() else None)
    cleanup = {'checked': False, 'remaining_owned_containers': None}
    eval_id = summary.get('evaluation_id')
    if eval_id:
        import docker
        import hashlib
        owner = 'harness-' + hashlib.sha256((run_id + ':' + eval_id).encode()).hexdigest()[:24]
        client = None
        try:
            client = docker.DockerClient(base_url=loaded['config']['environment']['docker_host'], timeout=15)
            containers = client.containers.list(all=True, filters={'label': ['dycon.project=' + str(root), 'dycon.run=' + owner]})
            cleanup = {'checked': True, 'remaining_owned_containers': [c.id for c in containers],
                       'cleanup_complete': not containers, 'owner': owner}
        except Exception as exc:
            cleanup = {'checked': False, 'remaining_owned_containers': None, 'error': str(exc), 'owner': owner}
        finally:
            if client is not None:
                client.close()
    output.mkdir(parents=True)
    for iid, row in records.items():
        write_json(output / 'tasks' / (iid + '.json'), row)
    result = {'run_id': run_id, 'requested': len(records), 'execution_finished': finished,
              'audit_complete': audit_complete, 'gold_valid_count': sum(r['gold_valid'] for r in records.values()),
              'classification_counts': dict(Counter(r['classification'] for r in records.values())),
              'reason_counts': dict(Counter(r['reason_code'] for r in records.values())),
              'cleanup': cleanup, 'task_ids_sha256': hash_json(sorted(records)),
              'accepted_pair_count': 0, 'negative_completed': False,
              'pair_acceptance': 'Requires separate matching environment negative evidence',
              'records': records, 'at': now()}
    write_json(output / 'summary.json', result)
    return {k: v for k, v in result.items() if k != 'records'}

"""Independent historical submission replay through the existing verifier lifecycle.

Restores authenticated declared outputs, not an unsaved historical live sandbox.
Never invokes solve/run or substitutes an oracle. Explicit replacement uses backups.
"""
from __future__ import annotations

import asyncio
import copy
import json
import shlex
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from application.execution.run_plan import load_plan
from application.execution.subject import safe_output_path
from core.io import hash_json, identifier, now, read_json, sha256, write_json
from core.locking import file_lock
from runtime.agents.submission_restore import archive_inventory, RESTORE_CODE

POLICY = 'skillsbench-declared-submission-regrade-v1'
SUPPORTED = {
    'lean4-proof': ['/app/workspace/solution.lean'],
    'latex-formula-extraction': ['/root/latex_formula_extraction.md'],
    'simpo-code-reproduction': ['/root/loss.npz', '/root/python_info.txt'],
    'paper-anonymizer': ['/root/redacted'],
    'xlsx-recover-data': ['/root/nasa_budget_recovered.xlsx'],
    'fix-build-google-auto': ['/home/travis/build/failed'],
    'fix-build-agentops': ['/home/github/build/failed'],
    'spring-boot-jakarta-migration': ['/workspace'],
    'setup-fuzzing-py': ['/app'],
}


def member(base, relative):
    path = Path(base) / relative
    if path.is_symlink() or not path.resolve().is_relative_to(Path(base).resolve()):
        raise ValueError('Source escaped its owned directory')
    return path


def source_inventory(source, audit_rows):
    run = source['run']
    paths = {run / n for n in ['plan.json', 'selection.json', 'config.yaml', 'resolved-config.json',
                               'evidence.json', 'progress.json']}
    names = {'reward.txt', 'rewards.jsonl', 'evaluation-raw.json', 'evaluation.json',
             'result.json', 'attempt.json', 'evidence.json', 'progress.json', '05-settled.json'}
    paths.update(p for p in run.rglob('*') if p.is_file() and p.name in names)
    for row in audit_rows:
        for entry in row.get('source_manifest', []):
            p = Path(entry['path'])
            if entry.get('sha256') and (not p.is_file() or sha256(p) != entry['sha256']):
                raise ValueError('Audited source changed: ' + str(p))
            paths.add(p)
        attempt = member(run, row['attempt_path'])
        path = member(attempt, 'subject/subject.json')
        subject = read_json(path)
        paths.add(path)
        for output in subject['outputs']:
            p = member(path.parent, output['path'])
            if sha256(p) != output['sha256'] or p.stat().st_size != output['size']:
                raise ValueError('Changed original subject output')
            paths.add(p)
    return {str(p): sha256(p) for p in sorted(paths) if p.is_file()}


def replay_plan(source, target, audit):
    from application.execution.status import _sample_status_projection
    from catalog.skillsbench import verify_package
    projection, _ = _sample_status_projection(source, read_json(source['run'] / 'progress.json'))
    tasks = {t['instance_id']: (k, t) for k, t in target['config']['selection']['tasks'].items()}
    rows = [json.loads(line) for line in Path(audit).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r['classification'] in {'VerifierError', 'Review'}]
    selected, seen, checked = [], set(), set()
    for row in rows:
        current = projection.get(row['sample_key'])
        if (row['run_id'] != source['plan']['run_id'] or row['sample_key'] in seen or not current
                or any(current.get(k) != row[k] for k in ['instance_id', 'repetition_id', 'attempt_id'])
                or current.get('raw_reward') != row['raw_reward']):
            raise ValueError('Duplicate, changed or superseded source sample')
        seen.add(row['sample_key'])
        key, task = tasks[row['instance_id']]
        original = source['config']['selection']['tasks'][row['task_key']]
        if row['instance_id'] not in checked:
            verify_package(task)
            adaptation = task['instance'].get('verifier_adaptation', {})
            receipt_path = Path(adaptation['receipt_path'])
            if sha256(receipt_path) != adaptation['receipt_sha256']:
                raise ValueError('Task adaptation receipt changed')
            upstream = read_json(receipt_path)['upstream_task']
            if (upstream['instance']['package_sha256'] != original['instance']['package_sha256']
                    or task['instance']['images'] != original['instance']['images']):
                raise ValueError('New verifier does not derive from the source package and image')
            checked.add(row['instance_id'])
        attempt = member(source['run'], row['attempt_path'])
        subject_path = member(attempt, 'subject/subject.json')
        subject = read_json(subject_path)
        if current.get('subject_sha256') != sha256(subject_path):
            raise ValueError('Original subject identity changed')
        binding = subject['binding']
        if any(binding.get(k) != value for k, value in {
                'run': source['plan']['run_id'], 'sample': row['sample_key'],
                'attempt': row['attempt_id'], 'task': row['instance_id']}.items()):
            raise ValueError('Subject belongs to a different historical attempt')
        allowed = SUPPORTED.get(row['instance_id'], [])
        reason = None
        paths = [o['source_path'] for o in subject['outputs']]
        if row['instance_id'] not in SUPPORTED:
            reason = 'Required /opt/syzkaller submission was not exported; /app archive cannot restore it'
        elif set(paths) | set(subject['missing_outputs']) != set(allowed) or subject['rejected_outputs']:
            reason = 'Incomplete or unsupported declared output inventory'
        selected.append({'source': row, 'task_key': key, 'task': task,
                         'subject_path': str(subject_path), 'subject_sha256': sha256(subject_path),
                         'subject': subject, 'unavailable_reason': reason,
                         'replay_scope': 'Declared authenticated outputs in a fresh image; original full live state unavailable'})
    if not selected:
        raise ValueError('No historical VerifierError/Review candidates')
    return selected, rows


async def restore(env, item, folder):
    subject, requests = item['subject'], []
    allowed = SUPPORTED[item['source']['instance_id']]
    for index, row in enumerate(subject['outputs']):
        path = member(Path(item['subject_path']).parent, row['path'])
        if row['kind'] == 'directory_archive':
            request = await asyncio.to_thread(archive_inventory, path, row['sha256'], row['source_path'])
            request['kind'] = 'directory_archive'
        elif row['kind'] == 'file':
            request = {k: row[k] for k in ['kind', 'source_path', 'sha256', 'mode']}
        else:
            raise ValueError('Unsupported subject kind')
        remote = f'/tmp/dycon-regrade-output-{index}'
        await env.upload_file(path, remote)
        request.update(uploaded_path=remote, allowed_paths=allowed)
        requests.append(request)
    requests += [{'kind': 'missing', 'source_path': p, 'allowed_paths': allowed}
                 for p in subject['missing_outputs']]
    results = []
    for index, request in enumerate(requests):
        path = folder / f'restore-{index:03d}.json'
        write_json(path, request)
        remote = f'/tmp/dycon-regrade-request-{index}.json'
        await env.upload_file(path, remote)
        command = '/opt/dycon-wheels/python/bin/python3.12 -c ' + shlex.quote(RESTORE_CODE) + ' ' + shlex.quote(remote)
        result = await env.exec(command, user='root', timeout_sec=300)
        if result.return_code:
            raise RuntimeError('Submission restoration failed: ' + str(result.stderr)[-1500:])
        proof = json.loads(result.stdout)
        if proof.get('verified') is not True:
            raise ValueError('Runtime restored bytes do not match')
        results.append(proof)
    # The unexported SimPO source must never fall back to the image's initial code.
    if item['source']['instance_id'] == 'simpo-code-reproduction':
        result = await env.exec('rm -rf -- /root/SimPO', user='root', timeout_sec=30)
        if result.return_code:
            raise RuntimeError('Unable to exclude unexported SimPO source')
    write_json(folder / 'restoration.json', {'policy': POLICY, 'original_subject_sha256': item['subject_sha256'],
        'verified': True, 'outputs': results, 'scope': item['replay_scope']})


async def replay(cfg, run_id, item, output):
    from benchflow.environment.manifest import EnvironmentManifest
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.rollout_planes import DefaultRolloutPlanes
    from benchflow.usage_tracking import UsageTrackingConfig
    from runtime.agents.skillsbench import OwnedPlanes, SkillsBenchFactory, _oracle_zero_audit
    from runtime.agents.dependencies import verify_runtime_snapshot

    row = {'policy': POLICY, 'sample_key': item['source']['sample_key'],
           'task_id': item['source']['instance_id'], 'original_attempt_id': item['source']['attempt_id'],
           'original_classification': item['source']['classification'], 'original_reward': item['source']['raw_reward'],
           'original_subject_sha256': item['subject_sha256'], 'new_rewards': None,
           'classification': 'RegradeUnavailable', 'reason': item['unavailable_reason'],
           'model_executed': False, 'oracle_executed': False, 'verifier_executed': False,
           'cleanup_confirmed': False, 'replay_scope': item['replay_scope'], 'started_at': now()}
    output.mkdir(parents=True, exist_ok=False)
    if item['unavailable_reason']:
        row.update(finished_at=now(), cleanup_confirmed=True)
        write_json(output / 'result.json', row)
        return row
    task, cfg = item['task'], copy.deepcopy(cfg)
    cfg['skillsbench_validation'] = True
    task_id = row['task_id']
    attempt_id = 'regrade-' + row['sample_key']
    row.update(run_id=run_id, attempt_id=attempt_id, task_package_sha256=task['instance']['package_sha256'],
               dependency_snapshot_sha256=cfg['execution_context']['profile']['dependency_snapshot_sha256'])
    factory = rollout = planes = None
    try:
        factory = SkillsBenchFactory(cfg, run_id)
        await asyncio.to_thread(factory.lock_images, [task])
        image = factory.images[(task_id, 'main')]
        cfg['skillsbench_service_images'] = {s: r['tag'] for (iid, s), r in factory.images.items() if iid == task_id}
        cfg['skillsbench_daemon_capabilities'] = {'cpu_quota_supported': factory.cpu_quota_supported}
        sample = SimpleNamespace(instance_id=task_id, sample_key=row['sample_key'], task_key=item['task_key'])
        planes = OwnedPlanes(DefaultRolloutPlanes(), cfg, run_id, sample, attempt_id, image, output)
        task_source = Path(task['instance']['task_path'])
        planes.task_source = task_source
        write_json(output / 'attempt-started.json', {'attempt_id': attempt_id, 'run_id': run_id,
                   'task_key': item['task_key'], 'sample_key': row['sample_key'], 'ownership': planes.ownership})
        profile = cfg['execution_context']['profile']

        async def check(env):
            await verify_runtime_snapshot(profile)

        config = RolloutConfig(task_path=task_source, agent='oracle', model=None,
            skill_mode=profile['skill_mode'], skills_dir=task_source / 'environment/skills'
            if profile['skill_mode'] == 'with-skill' else None,
            sandbox_user=None, timeout=None, jobs_dir=output / 'benchflow', job_name=attempt_id,
            rollout_name=attempt_id, environment_manifest=EnvironmentManifest(name=task_id,
                image=image['tag'], keep_alive=True, isolation='per_task', owns_lifecycle=True),
            pre_agent_hooks=[check], usage_tracking=UsageTrackingConfig(mode='off'), planes=planes)
        rollout = await Rollout.create(config)
        await rollout.setup()
        await rollout.start()
        await rollout.install_agent()
        await restore(rollout._env, item, output)
        await rollout.verify()
        result = rollout.result
        row.update(new_rewards=result.rewards, verifier_error=result.verifier_error,
                   verifier_executed=True, agent_error=result.error)
        audit = _oracle_zero_audit(output, task, result.rewards)
        write_json(output / 'verifier-audit.json', audit)
        row['classification'] = 'ValidOutcome' if audit['classification'] == 'NotCandidate' else audit['classification']
        row['reason'] = audit.get('reason_code')
        if result.verifier_error or result.error:
            row.update(classification='VerifierError', reason='verifier_execution_incomplete')
        if task_id == 'simpo-code-reproduction':
            stages = list(output.glob('benchflow/**/verifier/simpo-stages.json'))
            data = read_json(stages[0]) if len(stages) == 1 else {}
            if (any(s.get('phase') == 'submission' and s.get('status') == 'completed'
                    for s in data.get('stages', [])) or (result.rewards or {}).get('reward') == 1):
                row.update(classification='RegradeUnavailable', new_rewards=None,
                           reason='SimPO requires a rerun, but /root/SimPO source was not exported')
    except Exception as exc:
        row.update(classification='RegradeExecutionError', reason=repr(exc))
        (output / 'exception.txt').write_text(traceback.format_exc())
    finally:
        if rollout is not None:
            try:
                await rollout.cleanup()
            except Exception as exc:
                row['cleanup_error'] = repr(exc)
        if factory is not None:
            try:
                remaining = factory.client.containers.list(all=True, filters={'label': [
                    'dycon.run=' + run_id, 'dycon.epoch=' + planes.ownership['epoch']]}) if planes else []
                row['cleanup_confirmed'] = not remaining and (output / 'sandbox-cleanup.json').exists()
            except Exception as exc:
                row['cleanup_error'] = repr(exc)
            await asyncio.to_thread(factory.cleanup_owned, run_id)
        if not row['cleanup_confirmed']:
            row['classification'] = 'RegradeExecutionError'
        row['finished_at'] = now()
        row['receipt_sha256'] = hash_json(row)
        write_json(output / 'result.json', row)
    return row


def regrade(root, run_id, source_run, audit_report, output_dir, *, workers=3, plan_only=False, replace_original=False):
    if not 1 <= workers <= 4 or run_id == source_run:
        raise ValueError('Use a distinct frozen target run and workers in 1..4')
    source, target = load_plan(root, source_run), load_plan(root, run_id)
    output = Path(output_dir).resolve()
    if not output.is_relative_to((target['run'] / 'validation').resolve()) or output == (target['run'] / 'validation').resolve():
        raise ValueError('Results must be in a new validation subdirectory of the target run')
    if target['config']['execution_context'].get('benchmark') != 'skillsbench':
        raise ValueError('Expected frozen SkillsBench target')
    audit_path = Path(audit_report).resolve()
    if not audit_path.is_relative_to(Path(root).resolve()):
        raise ValueError('Audit selection must be project owned')
    if replace_original:
        from application.execution.config_revision import _offline
        _offline(source['run'])
    selected, audit_rows = replay_plan(source, target, audit_path)
    output.mkdir(parents=True, exist_ok=False)
    before = source_inventory(source, audit_rows)
    write_json(output / 'source-before.json', before)
    write_json(output / 'plan.json', {'policy': POLICY, 'run_id': run_id, 'source_run_id': source_run,
        'audit_report': str(audit_path), 'audit_sha256': sha256(audit_path),
        'replace_original_with_backup': replace_original,
        'task_count': len({i['source']['instance_id'] for i in selected}), 'sample_count': len(selected),
        'samples': [{'sample_key': i['source']['sample_key'], 'task_id': i['source']['instance_id'],
                     'subject_sha256': i['subject_sha256'], 'unavailable_reason': i['unavailable_reason']}
                    for i in selected]})
    if plan_only:
        return {'planned': len(selected), 'output': str(output), 'verifier_executed': False}

    async def execute():
        from runtime.agents.skillsbench import SkillsBenchFactory
        cfg = copy.deepcopy(target['config'])
        semaphore, rows, network = asyncio.Semaphore(workers), [], None

        async def one(item):
            async with semaphore:
                row = await replay(cfg, run_id, item, output / 'samples' / item['source']['sample_key'])
                rows.append(row)
                counts = dict(Counter(r['classification'] for r in rows))
                write_json(output / 'progress.json', {'selected': len(selected), 'completed': len(rows), 'counts': counts})
                print(json.dumps({'sample': row['sample_key'], 'task': row['task_id'],
                    'classification': row['classification'], 'completed': len(rows), 'selected': len(selected)}), flush=True)

        try:
            if cfg['execution_context']['profile'].get('use_host_proxy'):
                network = SkillsBenchFactory(copy.deepcopy(cfg), run_id)
                await asyncio.to_thread(network.lock_images, [])
                cfg['skillsbench_network_proxy'] = network.cfg['skillsbench_network_proxy']
            # Run short file-only submissions first; expensive build jobs follow.
            ordered = sorted(selected, key=lambda i: (i['source']['instance_id'].startswith('fix-build-'),
                             i['source']['instance_id'] == 'spring-boot-jakarta-migration', i['source']['sample_key']))
            await asyncio.gather(*(one(item) for item in ordered))
        finally:
            if network:
                await asyncio.to_thread(network.cleanup_owned, run_id)
        return rows

    with file_lock(target['run'] / 'validation/.oracle-lifecycle.lock', nonblocking=True):
        results = asyncio.run(execute())
    changed = [p for p, expected in before.items() if not Path(p).is_file() or sha256(p) != expected]
    write_json(output / 'source-preservation.json', {'checked': len(before), 'changed': changed, 'verified': not changed})
    summary = {'policy': POLICY, 'run_id': run_id, 'source_run_id': source_run, 'sample_count': len(selected),
               'completed': len(results), 'counts': dict(Counter(r['classification'] for r in results)),
               'verifier_executed': sum(r['verifier_executed'] for r in results),
               'historical_outcomes_modified': False, 'sources_preserved': not changed,
               'complete': len(results) == len(selected) and not changed, 'output': str(output)}
    (output / 'samples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n'
                                                for r in sorted(results, key=lambda r: r['sample_key'])))
    write_json(output / 'summary.json', summary)
    if replace_original:
        if not summary['complete']:
            raise ValueError('Incomplete replay or changed sources; refusing historical replacement')
        from runtime.agents.submission_replacement import replace_outcomes
        replacement = replace_outcomes(root, source, target, selected, results, output, before)
        summary.update(replacement=replacement,
                       historical_outcomes_modified=replacement['applied_count'] > 0)
        write_json(output / 'summary.json', summary)
    return summary

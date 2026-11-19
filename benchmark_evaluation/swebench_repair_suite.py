"""Blocking repair validation through existing CLI, with immutable batch evidence."""
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import yaml

from application.execution.run_plan import load_plan
from catalog.store import Catalog
from core.config import Environment
from core.io import hash_json, identifier, now, read_json, sha256, write_json
from core.locking import file_lock


def selected_task_ids(root, plan, source_tasks):
    """A failed-only continuation is bound to a completed immutable ledger."""
    if not plan.get('failed_only_baseline'):
        return sorted(source_tasks)
    ref = plan['failed_only_baseline']
    path = Path(ref['path']).resolve()
    if (not path.is_relative_to(Path(root).resolve() / 'temp')
            or sha256(path) != ref['sha256']):
        raise ValueError('Failed-only baseline ledger changed')
    baseline = read_json(path)
    if (baseline['run_id'] != plan['source_run']
            or set(baseline['records']) != set(source_tasks)
            or not baseline.get('execution_finished') or not baseline.get('audit_complete')
            or not baseline.get('cleanup', {}).get('cleanup_complete')):
        raise ValueError('Failed-only baseline must be complete and match the source')
    ids = sorted(iid for iid, row in baseline['records'].items() if row.get('gold_valid') is not True)
    if not ids or set(plan.get('task_ids', [])) != set(ids) or len(plan['task_ids']) != len(ids):
        raise ValueError('Failed-only selection must contain exactly the unaccepted tasks')
    return ids


def publish_task_settings(root, source_tasks, updates):
    """Only environment/image metadata may change; benchmark payloads stay frozen."""
    from runtime.containers.repairs import environment_overrides
    with Catalog(root).transaction() as catalog:
        for key, task in catalog['tasks'].items():
            iid = task['instance_id']
            if iid not in updates:
                continue
            expected = copy.deepcopy(source_tasks[iid])
            actual = copy.deepcopy(task['instance'])
            for value in [expected, actual]:
                value.pop('environment_overrides', None)
                value.pop('nas_image', None)
            if actual != expected:
                raise ValueError('Benchmark payload changed: ' + iid)
            changes = updates[iid]
            if set(changes) - {'environment_overrides', 'nas_image'}:
                raise ValueError('Unsupported task repair field')
            task['instance'].update(copy.deepcopy(changes))


def reuse_batch(root, plan, name, selected, source_tasks):
    """Reuse only content-bound, complete baseline evidence, never new scores."""
    ref = plan.get('reuse_baseline_batches', {}).get(name)
    if ref is None:
        return None
    path = Path(ref['path']).resolve()
    if not path.is_relative_to(Path(root).resolve() / 'temp') or sha256(path) != ref['sha256']:
        raise ValueError('Reused baseline ledger changed: ' + name)
    result = read_json(path)
    cleanup = result.get('cleanup', {})
    if (set(result.get('records', {})) != set(selected) or not result.get('execution_finished')
            or not result.get('audit_complete') or not cleanup.get('checked')
            or not cleanup.get('cleanup_complete')):
        raise ValueError('Reused baseline evidence is incomplete: ' + name)
    old = load_plan(root, result['run_id'])
    old_tasks = {r['instance_id']: r['instance'] for r in old['config']['selection']['tasks'].values()}
    for iid in selected:
        if old_tasks.get(iid) != source_tasks[iid]:
            raise ValueError('Reused baseline task configuration changed: ' + iid)
    result.update(batch=name, reused=True, reuse_source=str(path), reuse_sha256=ref['sha256'])
    return result


def run(root, plan_path):
    root = Path(root).resolve()
    plan = read_json(plan_path)
    if plan.get('schema_version') != 1:
        raise ValueError('Unsupported repair validation plan')
    suite_id = identifier(plan['suite_id'])
    job = root / 'temp' / suite_id
    job.mkdir(parents=True, exist_ok=True)
    source_id = identifier(plan['source_run'])
    loaded = load_plan(root, source_id)
    source = loaded['config']
    source_run = loaded['run']
    source_tasks = {row['instance_id']: row['instance'] for row in source['selection']['tasks'].values()}
    ids = selected_task_ids(root, plan, source_tasks)
    if len(ids) != plan['task_count'] or hash_json(ids) != plan['task_ids_sha256']:
        raise ValueError('Repair plan task set does not match frozen source')
    negative_enabled = plan.get('run_negative', True)
    if type(negative_enabled) is not bool:
        raise ValueError('run_negative must be a boolean')
    for name in ('collection_smoke', 'timeout_diagnostics'):
        if set(plan.get(name, [])) - set(ids):
            raise ValueError('Repair batch escapes the selected task set')
    if set(plan.get('task_settings', {})) - set(ids):
        raise ValueError('Repair settings escape the selected task set')
    for experiment in plan.get('experiments', []):
        if (not experiment['representatives'] or set(experiment['representatives']) - set(ids)
                or set(experiment['task_settings']) - set(ids)):
            raise ValueError('Repair experiment escapes the selected task set')
    protected = {str(p): sha256(p) for name in ['plan.json', 'selection.json', 'resolved-config.json',
                 'config.yaml', 'tasks.json', 'harness/summary.json', 'harness/selection.json']
                 if (p := source_run / name).exists()}
    cli = str(root / '.venvs/runner/bin/dycon')
    started = now()
    stages = []
    batches = []
    completed = False
    def command(args, label, seconds, allowed=(0,)):
        logfile = job / 'logs' / (identifier(label) + '.log')
        logfile.parent.mkdir(parents=True, exist_ok=True)
        write_json(job / 'stage.json', {'stage': label, 'at': now()})
        with logfile.open('wb') as output:
            child = subprocess.Popen([cli, *args], cwd=root, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            write_json(job / 'command.json', {'stage': label, 'pid': child.pid, 'at': now()})
            try:
                rc = child.wait(timeout=seconds)
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
                raise
        receipt = {'stage': label, 'returncode': rc, 'log': str(logfile), 'sha256': sha256(logfile), 'at': now()}
        stages.append(receipt)
        write_json(job / 'commands' / (label + '.json'), receipt)
        if rc not in allowed:
            raise RuntimeError(label + ' failed; see ' + str(logfile))
        return rc

    def evaluate_batch(name, selected, wheelhouse, *, source_run_id=source_id, gold=True):
        selected = sorted(selected)
        if gold and name in {'collection-smoke', 'timeout-diagnostics'}:
            reused = reuse_batch(root, plan, name, selected, source_tasks)
            if reused is not None:
                batches.append({k: v for k, v in reused.items() if k != 'records'})
                write_json(job / 'batches.json', batches)
                return reused
        rid = identifier(suite_id + '-' + name)
        source_loaded = load_plan(root, source_run_id)
        variables = {**source_loaded['config']['environment']['variables'],
                     'DYCON_SWEBENCH_FROZEN_ENVIRONMENT': '1'}
        args = ['--set', 'environment.variables=' + yaml.safe_dump(variables, default_flow_style=True),
                '--set', 'environment.wheelhouse=' + yaml.safe_dump(wheelhouse).strip(),
                '--set', 'execution.infrastructure_retry_limit=0',
                'run', 'create', rid, '--operation', 'harness', '--from-run', source_run_id]
        for iid in selected:
            args += ['--task-id', iid]
        command(args, name + '-create', 120)
        old_selection = read_json(source_loaded['run'] / 'harness/selection.json')
        rollout_ids = [r for r, row in old_selection['rollouts'].items()
                       if old_selection['tasks'][row['task_key']]['instance_id'] in selected]
        if len(rollout_ids) != len(selected):
            raise ValueError('Expected exactly one frozen submission per selected task')
        args = ['harness', 'prepare', rid, '--source-run', old_selection.get('source_run_id', source_run_id)]
        for rollout_id in rollout_ids:
            args += ['--rollout-id', rollout_id]
        command(args, name + '-prepare', 120)
        # Compare only immutable benchmark inputs, retaining all required IDs.
        new_selection = read_json(root / 'runs' / rid / 'harness/selection.json')
        for task in new_selection['tasks'].values():
            iid = task['instance_id']
            expected = copy.deepcopy(source_tasks[iid])
            actual = copy.deepcopy(task['instance'])
            for payload in [expected, actual]:
                payload.pop('nas_image', None)
                payload.pop('environment_overrides', None)
            if expected != actual:
                raise ValueError('Frozen benchmark input changed in batch ' + name)
        evaluation_error = None
        try:
            command(['harness', 'run', rid, name, *(['--gold'] if gold else [])], name + '-evaluate', plan['stage_timeout_seconds'])
        except Exception as exc:
            evaluation_error = str(exc)
        finally:
            audit = job / 'audits' / rid
            audit_error = None
            try:
                command(['harness', 'audit', rid, '--output', str(audit)], name + '-audit', 1200, allowed=(0, 3))
            except Exception as exc:
                audit_error = str(exc)
            ledger = job / 'acceptance' / rid
            # Ledger/cleanup collection is an independent formal stage even
            # when the external audit reports incomplete evidence.
            command(['harness', 'acceptance', rid, '--output', str(ledger),
                     *(['--audit-dir', str(audit)] if audit.exists() else [])], name + '-ledger', 180)
            result = read_json(ledger / 'summary.json')
            result.update(batch=name, evaluation_error=evaluation_error, audit_error=audit_error)
            batches.append({k: v for k, v in result.items() if k != 'records'})
            write_json(job / 'batches.json', batches)
        if evaluation_error:
            raise RuntimeError(evaluation_error)
        return result

    previous_handler = signal.getsignal(signal.SIGTERM)
    def terminated(signum, frame):
        raise KeyboardInterrupt('Repair suite terminated')
    signal.signal(signal.SIGTERM, terminated)
    with file_lock(root / 'state' / (suite_id + '.lock'), nonblocking=True):
        if (job / 'started.json').exists():
            raise FileExistsError('Use a new repair suite ID')
        write_json(job / 'plan.json', plan)
        write_json(job / 'started.json', {'pid': os.getpid(), 'at': started, 'source_run': source_id,
                                        'task_count': len(ids), 'progress_polling': False})
        try:
            # Dependency preparation must succeed before spending time on
            # evaluation or image work. Existing publications stay immutable.
            wheelhouse = str(root / 'runtime/wheelhouse' / plan['wheelhouse_id'])
            if plan.get('wheelhouse_manifest_sha256'):
                from runtime.wheelhouse import verify
                if sha256(Path(wheelhouse) / 'manifest.json') != plan['wheelhouse_manifest_sha256']:
                    raise ValueError('Prepared wheelhouse manifest changed')
                write_json(job / 'wheelhouse-verified.json', verify(wheelhouse))
            else:
                command(['wheelhouse', 'build', plan['wheelhouse_id'], '--source', plan['wheel_source'],
                         '--source', source['environment']['wheelhouse'],
                         *(['--download-plan', plan['wheel_download_plan']] if plan.get('wheel_download_plan') else [])],
                         'wheelhouse-build', 3600)
            # Replay historical evidence into a new sidecar, never old results.
            command(['harness', 'audit', source_id, '--output', str(job / 'historical-replay')],
                    'historical-replay', 1200, allowed=(0, 3))
            if plan.get('collection_smoke'):
                smoke = evaluate_batch('collection-smoke', plan['collection_smoke'], source['environment']['wheelhouse'])
                if smoke['gold_valid_count'] != len(plan['collection_smoke']):
                    raise RuntimeError('Collection smoke failed; expansion gate remains closed')
            # Preserve diagnostic-only baseline before resources/dependencies.
            if plan.get('timeout_diagnostics'):
                evaluate_batch('timeout-diagnostics', plan['timeout_diagnostics'], source['environment']['wheelhouse'])
            image_updates = {}
            image_errors = []
            for image in plan['image_repairs']:
                args = ['swebench', 'images-repair', source_id, image['publication_id'], '--kind', image['kind']]
                for iid in image['task_ids']:
                    args += ['--task-id', iid]
                try:
                    command(args, 'images-' + image['kind'], 10800)
                    store = Path(source['paths']['nas_root']) / image['publication_id']
                    for iid, row in read_json(store / 'completion.json')['results'].items():
                        image_updates[iid] = row['reference']
                except Exception as exc:
                    image_errors.append({'kind': image['kind'], 'error': str(exc)})
            write_json(job / 'image-results.json', {'references': image_updates, 'errors': image_errors})
            settings = copy.deepcopy(plan['task_settings'])
            for iid, ref in image_updates.items():
                settings.setdefault(iid, {})['nas_image'] = ref
            publish_task_settings(root, source_tasks, settings)
            write_json(job / 'published-settings.json', settings)
            negative_source = plan.get('negative_source_run')
            negative_ids = ({r['instance_id'] for r in load_plan(root, negative_source)['config']['selection']['tasks'].values()}
                            if negative_enabled else set())
            # Version candidates are explicitly isolated and rolled back when
            # their representative gold controls do not meet the full contract.
            experiments = []
            for experiment in plan['experiments']:
                proposed = copy.deepcopy(settings)
                for iid, patch in experiment['task_settings'].items():
                    env = proposed.setdefault(iid, {}).setdefault('environment_overrides', {})
                    for key, value in patch.items():
                        if isinstance(value, dict):
                            env[key] = {**env.get(key, {}), **value}
                        else:
                            env[key] = value
                publish_task_settings(root, source_tasks, {iid: proposed[iid] for iid in experiment['task_settings']})
                result = evaluate_batch(experiment['name'], experiment['representatives'], wheelhouse)
                accepted = result['gold_valid_count'] == len(experiment['representatives'])
                controls = sorted(set(experiment['representatives']) & negative_ids)
                control_result = None
                if accepted and controls:
                    control_result = evaluate_batch(experiment['name'] + '-negative', controls, wheelhouse,
                                                    source_run_id=negative_source, gold=False)
                    accepted = all(control_result['records'][iid]['classification'] == 'ValidOutcome'
                        and control_result['records'][iid]['reward'] == 0
                        and control_result['records'][iid]['scheme_id'] == result['records'][iid]['scheme_id']
                        for iid in controls)
                if accepted:
                    settings = proposed
                else:
                    publish_task_settings(root, source_tasks, {iid: settings[iid] for iid in experiment['task_settings']})
                experiments.append({'name': experiment['name'], 'representatives_passed': accepted,
                                    'valid_count': result['gold_valid_count'], 'run_id': result['run_id'],
                                    'matching_negative_tasks': controls,
                                    'scheme_negative_coverage': 'checked' if control_result else 'uncovered'})
                write_json(job / 'experiments.json', experiments)
            write_json(job / 'final-settings.json', settings)
            final_batch = plan.get('final_batch', 'gold145')
            full = evaluate_batch(final_batch, ids, wheelhouse)
            applicable = sorted(iid for iid in negative_ids if full['records'].get(iid, {}).get('gold_valid'))
            pairs = []
            negative = None
            if applicable:
                negative = evaluate_batch('negative', applicable, wheelhouse, source_run_id=negative_source, gold=False)
                for iid in applicable:
                    pos, neg = full['records'][iid], negative['records'][iid]
                    if (neg['classification'] == 'ValidOutcome' and neg['reward'] == 0
                            and pos['scheme_id'] == neg['scheme_id']):
                        pairs.append(iid)
            write_json(job / 'pair-results.json', {'accepted_pairs': pairs, 'accepted_pair_count': len(pairs),
                       'negative_requested_count': len(applicable), 'negative_enabled': negative_enabled,
                       'uncovered_gold_schemes_remain': True})
            completed = True
            return {'suite_id': suite_id, 'gold_valid_count': full['gold_valid_count'],
                    'accepted_pair_count': len(pairs), 'task_count': len(ids)}
        finally:
            changed = [name for name, digest in protected.items() if sha256(name) != digest]
            write_json(job / 'completion.json', {'state': 'finished' if completed else 'failed',
                'started_at': started, 'finished_at': now(), 'task_count': len(ids),
                'batches': batches, 'stages': stages, 'historical_source_changes': changed,
                'error': str(sys.exc_info()[1]) if sys.exc_info()[1] else None,
                'acceptance_status': 'partial' if completed else 'incomplete',
                'all_tasks_repaired': bool(completed and batches and next((b['gold_valid_count'] == len(ids) for b in batches if b['batch'] == plan.get('final_batch', 'gold145')), False)),
                'negative_enabled': negative_enabled,
                'progress_polling': False})
            signal.signal(signal.SIGTERM, previous_handler)

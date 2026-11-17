"""Audited revisions of the complete editable run configuration."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import re
import shutil

import yaml

from application.execution.config_diff import diff_config_values
from application.execution.run_plan import _selection_from_tasks, load_plan
from core.config import Config, UniqueLoader, resolve_values
from core.io import hash_json, identifier, now, read_json, sha256, write_json
from core.locking import file_lock
from core.processes import matches


def parse_updates(assignments=(), patch=None):
    """Parse YAML values; final schema validation rejects unknown nested fields."""
    updates = []
    if patch is not None:
        raw = yaml.load(Path(patch).read_text(encoding='utf-8'), Loader=UniqueLoader)
        if not isinstance(raw, dict):
            raise ValueError('Revision patch must be a configuration mapping')
        def leaves(value, parts):
            if isinstance(value, dict) and value:
                for key, child in value.items():
                    if not isinstance(key, str) or '.' in key:
                        raise ValueError('Patch field names must be strings without dots')
                    leaves(child, [*parts, key])
            else:
                updates.append(('.'.join(parts), value))
        leaves(raw, [])
    for item in assignments:
        dotted, sep, value = item.partition('=')
        if not sep:
            raise ValueError('--set requires FIELD=YAML_VALUE')
        updates.append((dotted, yaml.load(value, Loader=UniqueLoader)))
    names = [name for name, _ in updates]
    for i, name in enumerate(names):
        if any(name == other or name.startswith(other + '.') or other.startswith(name + '.')
               for other in names[:i]):
            raise ValueError('Conflicting revision paths: ' + name)
    return dict(updates)


def _assign(raw, name, value):
    parts = name.split('.')
    if not all(parts) or parts[0] not in Config.model_fields:
        raise ValueError('Not an editable configuration field: ' + name)
    if name == 'parallel.max_inflight_branches':
        raise ValueError('parallel.max_inflight_branches is derived from serving.max_num_seqs')
    node = raw
    for key in parts[:-1]:
        if node.get(key) is None:
            node[key] = {}
        if not isinstance(node[key], dict):
            raise ValueError('Configuration path is not a mapping: ' + name)
        node = node[key]
    node[parts[-1]] = copy.deepcopy(value)


def _offline(run):
    latest = run / 'execution/latest.json'
    if not latest.exists():
        return None
    execution = identifier(read_json(latest)['attempt'])
    previous = run / 'execution/attempts' / execution
    for name in ['process.json', 'supervisor.json']:
        if (previous / name).exists() and matches(read_json(previous / name)):
            raise RuntimeError('Execution still active; cancel/wait before revising')
    if not (previous / 'exit.json').exists():
        raise RuntimeError('Previous execution has no exit receipt')
    if not (previous / 'termination.json').exists() or not read_json(previous / 'termination.json').get('cleanup_complete'):
        raise RuntimeError('Previous execution cleanup is incomplete')
    return execution


def _prepare(root, loaded, updates, replan):
    old = loaded['config']
    raw = copy.deepcopy({key: old[key] for key in Config.model_fields if key in old})
    raw.get('parallel', {}).pop('estimator_sha256', None)
    raw.get('steering', {}).pop('payload_sha256', None)
    raw.get('parallel', {}).pop('max_inflight_branches', None)
    for name, value in updates.items():
        _assign(raw, name, value)
    normalized = resolve_values(raw, root / 'config.yaml', run_policy=True,
                                source_sha256=old.get('config_source_sha256', ''))
    old_parallel = old.get('parallel', {})
    if ('estimator_sha256' not in old_parallel and
            normalized.get('parallel', {}).get('estimator_ref') == old_parallel.get('estimator_ref')):
        normalized.get('parallel', {}).pop('estimator_sha256', None)
    candidate = copy.deepcopy(old)
    for key in Config.model_fields:
        candidate.pop(key, None)
        if key in normalized:
            candidate[key] = normalized[key]
    # Artifact placement is configurable; an existing run stays at its physical
    # location. New artifact types use revised roots without moving old evidence.
    if candidate['paths'] != old['paths']:
        candidate['run_storage_root'] = str(loaded['run'].resolve().parent.parent)
    input_diff = diff_config_values(
        {k: old[k] for k in Config.model_fields if k in old},
        {k: candidate[k] for k in Config.model_fields if k in candidate})
    identity_paths = ('/benchmark/', '/generation/seed', '/parallel/branch_count', '/parallel/enabled',
                      '/agent/backend', '/agent/profile_ref', '/model/path',
                      '/model/fingerprint', '/model/served_name', '/execution/limits_retry')
    identity_changes = [row['path'] for row in input_diff
                        if any(row['path'].startswith(p) for p in identity_paths)]
    if identity_changes and not replan:
        raise ValueError('Identity changes require --replan to archive prior state: ' + ', '.join(identity_changes))
    selection = old['selection']
    if replan and candidate['benchmark'] != old['benchmark']:
        from catalog.store import Catalog
        selection = _selection_from_tasks(Catalog(root), cfg=candidate,
                                           partition=candidate['benchmark'].get('partition'))
        if not selection['task_keys']:
            raise ValueError('Revised benchmark selection is empty')
    candidate['selection'] = copy.deepcopy(selection)
    candidate['selection_sha256'] = hash_json(selection)
    if candidate.get('parallel_policy'):
        from core.parallel_policy import effective_policy
        proof = candidate['parallel_policy']
        proof['effective'] = effective_policy(candidate.get('parallel', {}))
        proof['effective_sha256'] = hash_json(proof['effective'])
    context_fields = ['benchmark', 'agent', 'generation', 'parallel', 'model', 'webarena', 'harbor',
                      'serving', 'environment', 'skillsbench_images', 'webarena_images']
    if replan or any(candidate.get(k) != old.get(k) for k in context_fields):
        from core.backends import execution_context
        candidate.pop('execution_context', None)
        context = execution_context(candidate, selection['tasks'])
        if context is not None:
            original_context = old.get('execution_context') or {}
            for section, field, digest in [('agent', 'profile_ref', 'profile_sha256'),
                                            ('benchmark', 'snapshot_ref', 'snapshot_sha256')]:
                if (candidate.get(section, {}).get(field) == old.get(section, {}).get(field)
                        and original_context.get(digest) is not None
                        and context.get(digest) != original_context[digest]):
                    raise ValueError('Frozen reference changed in place; publish and revise a new ' + section + '.' + field)
            candidate['execution_context'] = context
    from runtime.backends.registry import resolve
    candidate['execution_binding'] = resolve(candidate, selection['tasks']).as_dict()
    if candidate['environment'].get('wheelhouse') != old['environment'].get('wheelhouse'):
        if candidate['environment'].get('wheelhouse'):
            from runtime.wheelhouse import verify
            verify(candidate['environment']['wheelhouse'])
    candidate['config_hash'] = hash_json({k: v for k, v in candidate.items()
                                        if k not in {'config_hash', 'selection', 'selection_sha256'}})
    return candidate, identity_changes


def _documents(loaded, candidate, revision_id, replan):
    run = loaded['run']
    plan = copy.deepcopy(loaded['plan'])
    plan.update(config_hash=candidate['config_hash'], config_revision=revision_id,
                previous_plan_sha256=plan['plan_sha256'])
    documents = {'resolved-config.json': candidate}
    if replan:
        selection = candidate['selection']
        tasks = copy.deepcopy(loaded['tasks'])
        tasks.update(selection_sha256=hash_json(selection), tasks=[
            {'task_key': k, 'instance_id': selection['tasks'][k]['instance_id'], 'task': selection['tasks'][k]}
            for k in selection['task_keys']])
        tasks['tasks_sha256'] = hash_json(tasks['tasks'])
        plan.update(selection_sha256=hash_json(selection), tasks_sha256=tasks['tasks_sha256'])
        from rollout.sampling import make_work_items
        work = make_work_items(selection, run.name, plan['repeat_count'], candidate['generation']['seed'],
                               candidate.get('parallel', {}).get('branch_count', 1), extension=plan.get('sampling_extension'))
        work['items_sha256'] = hash_json(work['items'])
        documents.update({'selection.json': selection, 'tasks.json': tasks, 'rollout/work-items.json': work})
    plan['plan_sha256'] = hash_json({k: v for k, v in plan.items() if k != 'plan_sha256'})
    documents['plan.json'] = plan
    human = copy.deepcopy({k: candidate[k] for k in Config.model_fields if k in candidate})
    human.get('parallel', {}).pop('max_inflight_branches', None)
    human.get('parallel', {}).pop('estimator_sha256', None)
    human.get('steering', {}).pop('payload_sha256', None)
    result = {name: (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
              for name, value in documents.items()}
    result['config.yaml'] = yaml.safe_dump(human, sort_keys=False, allow_unicode=True).encode()
    return result


def _attempt_floors(run):
    path = run / 'rollout/replan-attempt-floors.json'
    floors = read_json(path) if path.exists() else {}
    for folder in (run / 'rollout/tasks').glob('*/repetitions/*/attempts'):
        key = folder.parents[2].name + '/' + folder.parent.name
        numbers = [int(m.group(1)) for p in folder.iterdir()
                   if (m := re.fullmatch(r'rep-\d+-attempt-(\d+)', p.name))]
        floors[key] = max([floors.get(key, -1), *numbers])
    return floors


def _put(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.revision.part')
    temporary.write_bytes(content)
    temporary.replace(path)


def _remove(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _rollback(run, out, journal):
    write_json(out / 'progress.json', {**journal, 'phase': 'rolling_back'})
    if journal['replan']:
        for name in journal['archive_names']:
            stored = out / 'archived-state' / name
            if stored.exists() or stored.is_symlink():
                _remove(run / name)
                stored.replace(run / name)
        for name in journal['new_roots']:
            if name not in journal['archive_names']:
                _remove(run / name)
    for name, digest in journal['before_files'].items():
        source = out / 'before' / name
        if digest is None:
            (run / name).unlink(missing_ok=True)
        else:
            if sha256(source) != digest:
                raise RuntimeError('Corrupt revision backup: ' + name)
            _put(run / name, source.read_bytes())
    write_json(out / 'progress.json', {**journal, 'phase': 'rolled_back', 'at': now()})


def revise_config(root, run_id, *, revision_id, updates=None, reason, replan=False,
                  dry_run=False, recover=False, expect_config_hash=None):
    root = Path(root).resolve()
    run_id, revision_id = identifier(run_id), identifier(revision_id)
    if not reason or not reason.strip():
        raise ValueError('Explicit authorization/reason is required')
    updates = updates or {}
    # The Python API has the same ambiguity checks as CLI patch parsing.
    names = list(updates)
    for i, name in enumerate(names):
        if not isinstance(name, str) or any(name.startswith(other + '.') or other.startswith(name + '.')
                                            for other in names[:i]):
            raise ValueError('Conflicting or invalid revision paths')
    if not updates and not recover:
        raise ValueError('Specify configuration updates with --set or --patch')
    if recover and (updates or dry_run or replan):
        raise ValueError('--recover cannot be combined with updates, preview or replan')
    loaded = load_plan(root, run_id) if not recover else None
    from core.io import run_path
    run = loaded['run'] if loaded else run_path(root, run_id)
    out = run / 'config-revisions' / revision_id
    request = {'updates': updates, 'reason': reason, 'replan': replan}
    if dry_run:
        if expect_config_hash and loaded['config']['config_hash'] != expect_config_hash:
            raise ValueError('Configuration changed since preview')
        candidate, identity_changes = _prepare(root, loaded, updates, replan)
        return {'dry_run': True, 'run_id': run_id, 'revision_id': revision_id,
                'old_config_hash': loaded['config']['config_hash'], 'new_config_hash': candidate['config_hash'],
                'replan': replan, 'identity_changes': identity_changes,
                'differences': diff_config_values(loaded['config'], candidate)}
    with ExitStack() as locks:
        for path in [root / 'state/run-locks' / (run_id + '.lock'),
                     run / 'execution/supervisor.lock', run / 'execution/active-supervisor.lock']:
            locks.enter_context(file_lock(path, nonblocking=True))
        execution = _offline(run)
        if recover:
            if not (out / 'progress.json').exists():
                # Preparation publishes the journal before touching run files.
                # An interruption while backing up therefore has no run writes.
                if not out.is_dir():
                    raise ValueError('Revision audit does not exist')
                load_plan(root, run_id)
                write_json(out / 'progress.json', {'phase': 'rolled_back', 'preparation_only': True, 'at': now()})
                return {'run_id': run_id, 'revision_id': revision_id, 'recovered': True}
            journal = read_json(out / 'progress.json')
            if (out / 'receipt.json').exists():
                receipt = read_json(out / 'receipt.json')
                if any(sha256(run / name) != digest for name, digest in receipt['after_files'].items()):
                    raise RuntimeError('Committed revision no longer matches current files')
                write_json(run / 'events' / ('config-revision-' + revision_id + '.json'),
                           {'event': 'run.config_revised', **receipt})
                write_json(out / 'progress.json', {**journal, 'phase': 'complete', 'completed_at': now()})
                return {'run_id': run_id, 'revision_id': revision_id, 'recovered': True, 'committed': True}
            if journal['phase'] not in {'prepared', 'committing', 'rolling_back'}:
                raise ValueError('Revision is not an incomplete transaction')
            _rollback(run, out, journal)
            return {'run_id': run_id, 'revision_id': revision_id, 'recovered': True}
        loaded = load_plan(root, run_id)
        if (out / 'receipt.json').exists():
            receipt = read_json(out / 'receipt.json')
            if receipt['request'] != request:
                raise ValueError('Revision ID already binds different values/reason')
            if sha256(run / 'resolved-config.json') != receipt['after_files']['resolved-config.json']:
                raise ValueError('Run has a later revision; refusing to reuse this ID')
            return receipt
        if out.exists():
            raise RuntimeError('Incomplete revision audit exists; inspect and use --recover or a new revision ID')
        if expect_config_hash and loaded['config']['config_hash'] != expect_config_hash:
            raise ValueError('Configuration changed since preview')
        candidate, identity_changes = _prepare(root, loaded, updates, replan)
        old_export = loaded['config'].get('capture', {}).get('export', {})
        new_export = candidate.get('capture', {}).get('export', {})
        # A writer reads its own frozen quota. Increasing only this stopped
        # run's quota does not move shared bytes or change other writers.
        quota_increase_only = (
            new_export.get('spool_max_bytes', 0) > old_export.get('spool_max_bytes', 0)
            and new_export == {**old_export, 'spool_max_bytes': new_export.get('spool_max_bytes')})
        if replan or (new_export != old_export and not quota_increase_only):
            from hidden.relocate_spool import assert_no_writers
            assert_no_writers(root, [run_id])
        proposed = _documents(loaded, candidate, revision_id, replan)
        floors = _attempt_floors(run) if replan else None
        retained = {'execution', 'logs', 'events', 'config-revisions', 'plan.json',
                    'config.yaml', 'resolved-config.json', 'selection.json', 'tasks.json', 'parallel-config.yaml'}
        archive_names = sorted(p.name for p in run.iterdir() if p.name not in retained) if replan else []
        new_roots = ['rollout', 'progress.json', 'status.json', 'artifacts.json']
        protected = {name: sha256(run / name) for name in
                     ['selection.json', 'tasks.json', 'rollout/work-items.json', 'evidence.json', 'progress.json']
                     if not replan and (run / name).exists()}
        before = {name: (run / name).read_bytes() if (run / name).exists() else None for name in proposed}
        journal = {'phase': 'prepared', 'replan': replan, 'archive_names': archive_names,
                   'new_roots': new_roots, 'before_files': {}, 'prepared_at': now()}
        for name, content in before.items():
            if content is not None:
                _put(out / 'before' / name, content)
            journal['before_files'][name] = sha256(out / 'before' / name) if content is not None else None
        for name, content in proposed.items():
            _put(out / 'after' / name, content)
        write_json(out / 'request.json', request)
        write_json(out / 'progress.json', journal)
        try:
            write_json(out / 'progress.json', {**journal, 'phase': 'committing'})
            if replan:
                (out / 'archived-state').mkdir()
                for name in archive_names:
                    (run / name).replace(out / 'archived-state' / name)
            for name, content in proposed.items():
                _put(run / name, content)
            if replan:
                write_json(run / 'rollout/replan-attempt-floors.json', floors)
                from rollout.retry import initialize_counts
                candidate_work = read_json(run / 'rollout/work-items.json')
                if loaded['plan']['operation'] == 'rollout':
                    initialize_counts(run, candidate_work)
                    from rollout.progress import Progress
                    Progress(run, candidate, candidate_work['items'])._write()
                else:
                    write_json(run / 'progress.json', {})
                write_json(run / 'status.json', {'stage': 'planned', 'run_id': run_id,
                           'operation': loaded['plan']['operation']})
                write_json(run / 'artifacts.json', {'schema_version': 1, 'run_id': run_id, 'artifacts': []})
            verified = load_plan(root, run_id, allow_incomplete_revision=revision_id)
            if verified['config'] != candidate or any(sha256(run / k) != v for k, v in protected.items()):
                raise RuntimeError('Revision verification failed')
        except BaseException:
            _rollback(run, out, journal)
            raise
        differences = diff_config_values(loaded['config'], candidate)
        receipt = {'schema_version': 1, 'run_id': run_id, 'revision_id': revision_id,
                   'request': request, 'previous_execution': execution, 'replan': replan,
                   'identity_changes': identity_changes, 'differences': differences,
                   'changed_config_fields': [row['path'].strip('/').replace('/', '.') for row in differences
                                             if row['path'] != '/config_hash'],
                   'old_config_hash': loaded['config']['config_hash'], 'new_config_hash': candidate['config_hash'],
                   'old_plan_sha256': loaded['plan']['plan_sha256'], 'new_plan_sha256': verified['plan']['plan_sha256'],
                   'before_files': journal['before_files'],
                   'after_files': {name: sha256(run / name) for name in proposed},
                   'protected_files_unchanged': protected, 'archived_state': archive_names, 'completed_at': now()}
        write_json(out / 'receipt.json', receipt)
        write_json(run / 'events' / ('config-revision-' + revision_id + '.json'),
                   {'event': 'run.config_revised', **receipt})
        write_json(out / 'progress.json', {**journal, 'phase': 'complete', 'completed_at': now()})
        return receipt

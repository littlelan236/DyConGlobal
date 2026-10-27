"""Explicit, audited offline revisions of capture storage resource settings only."""
from contextlib import ExitStack
from pathlib import Path
import copy
import json

import yaml

from application.execution.run_plan import load_plan
from core.config import Capture, CaptureExport
from core.io import hash_json, identifier, now, read_json, sha256, write_json
from core.locking import file_lock
from core.processes import matches
from hidden.relocate_spool import assert_no_writers

FIELDS = ('spool_max_bytes', 'spool_high_watermark', 'spool_low_watermark')


def revise_capture_budget(root, run_id, *, revision_id, max_bytes=None, high=None, low=None, reason,
                          control_seconds=None, receipt_workers=None):
    root = Path(root).resolve()
    run_id, revision_id = identifier(run_id), identifier(revision_id)
    if not reason or not reason.strip():
        raise ValueError('Explicit authorization/reason is required')
    run = root / 'runs' / run_id
    control = control_seconds is not None or receipt_workers is not None
    if control:
        if control_seconds is None or receipt_workers is None or any(v is not None for v in (max_bytes, high, low)):
            raise ValueError("Specify control seconds and receipt workers together, without budget values")
        values = {"control_seconds": control_seconds, "receipt_workers": receipt_workers}
    else:
        values = dict(zip(FIELDS, (max_bytes, high, low)))
    field_prefix = "capture." if control else "capture.export."
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
        if control:
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
        if control:
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
            assert verified['config']['capture'] == candidate['capture']
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
                   {'event': 'run.capture_control_revised' if control else 'run.capture_budget_revised', **receipt})
        write_json(out / 'progress.json', {'phase': 'complete', 'completed_at': now()})
        return receipt


def revise_capture_control(root, run_id, *, revision_id, control_seconds, receipt_workers, reason):
    """Use the same offline locking/audit mechanism for capture control resources."""
    return revise_capture_budget(root, run_id, revision_id=revision_id, reason=reason,
                                 control_seconds=control_seconds, receipt_workers=receipt_workers)

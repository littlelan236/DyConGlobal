"""Read-only, bounded evidence shared by online settlement and audit replay."""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

from core.io import hash_json, sha256

MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 96 * 1024 * 1024
TEXT_NAMES = {'test-stdout.txt', 'test-stderr.txt', 'test_output.log', 'pytest_output.txt'}


@lru_cache(maxsize=1)
def registry():
    data = json.loads(Path(__file__).with_name('verifier_contracts.json').read_text(encoding='utf-8'))
    return {c['package_sha256']: c for c in data['contracts']}


def frozen_contract(task):
    package = (task or {}).get('instance', {})
    result = {'verified': False, 'status': 'unknown', 'ctrf_expected': None,
              'task_completion_patterns': [], 'files': [], 'sources': {},
              'source': 'run_selected_package', 'package_sha256': package.get('package_sha256'),
              'score_mode': 'unsupported', 'capabilities': [], 'expected_reports': []}
    files = package.get('files', [])
    result['official_input_members'] = [m.get('path') for m in files
                                        if str(m.get('path','')).startswith(('verifier/','tests/'))]
    if not files or hash_json(files) != package.get('package_sha256'):
        return result
    registered = registry().get(package['package_sha256'])
    if registered and registered['instance_id'] != (task.get('instance_id') or package.get('instance_id')):
        result['status'] = 'identity_mismatch'
        return result
    root = Path(package.get('task_path', '')).resolve()
    try:
        indexed = {m['path']: m for m in files}
        if len(indexed) != len(files):
            raise ValueError('duplicate package member')
        wanted = registered['files'] if registered else {
            m['path']: m['sha256'] for m in files
            if m['path'] in {'test.sh', 'tests/test.sh', 'verifier/test.sh'}}
        for name, expected in wanted.items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('unsafe package member')
            path = root / relative
            if (path.is_symlink() or not path.resolve().is_relative_to(root)
                    or indexed[name]['sha256'] != expected or path.stat().st_size > MAX_FILE):
                raise ValueError('invalid frozen member')
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != expected:
                raise ValueError('changed frozen member')
            result['files'].append({'path': name, 'sha256': digest})
            if relative.suffix not in {'.npz','.npy'}:
                result['sources'][name] = payload.decode('utf-8')
        scripts = [v for k, v in result['sources'].items() if k in {'test.sh', 'tests/test.sh', 'verifier/test.sh'}]
        if not scripts:
            return result
        commands = '\n'.join(line for s in scripts for line in s.splitlines()
                             if not line.lstrip().startswith('#'))
        if '--ctrf' in commands:
            result['ctrf_expected'] = True
            result['expected_reports'] = sorted(set(re.findall(r'/logs/verifier/(ctrf[\w.-]*\.json)', commands)))
        elif registered and registered['score_mode'] == 'pytest_exit_binary':
            result['ctrf_expected'] = False
        if registered:
            result.update(score_mode=registered['score_mode'], capabilities=registered['capabilities'],
                          contract_id=package['package_sha256'], contract_version=1)
            result['command_labels'] = registered.get('command_labels', [])
            if 'agentops_matrix_v1' in registered['capabilities']:
                result['required_environments']=registered['required_environments']
            if result['score_mode'] in {'custom_groups', 'cif_fraction'}:
                result['ctrf_expected'] = False
            if 'fuzz_driver' in result['capabilities']:
                result['expected_reports'] = ['ctrf.json'] + [f'ctrf_{lib}.json' for lib in (
                    'arrow', 'ujson', 'black', 'ipython', 'minisgl')]
        result.update(verified=True, status='supported' if registered and result['score_mode'] != 'unsupported'
                      else 'unsupported')
    except (OSError, ValueError, KeyError, UnicodeError) as exc:
        result.update(verified=False, status='identity_mismatch', error=str(exc), sources={})
    return result


def verifier_files(attempt_dir):
    root = Path(attempt_dir)/'benchflow'
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob('*') if p.is_file()
                  and 'verifier' in p.relative_to(root).parts[:-1]
                  and 'trainer' not in p.relative_to(root).parts[:-1])


def runtime_lines(text):
    """Keep exception output/captured streams; never turn displayed source into events.

    Pytest E lines are accepted only as exception output. Assert expressions,
    expanded values and arrow/source lines remain source_context.
    """
    region = 'stream'
    for number, raw in enumerate(text.splitlines(), 1):
        line = re.sub(r'\x1b\[[0-9;]*m', '', raw)
        if re.match(r'^_{3,}.*_{3,}$', line.strip()):
            region = 'source_context'
        if re.match(r'^-{3,}\s*Captured (?:stdout|stderr|log)', line.strip()):
            region = 'captured_stream'
        if re.match(r'^={3,}.*(?:short test summary|FAILURES|ERRORS)', line.strip()):
            region = 'source_context'
        if line.startswith('E '):
            value = re.sub(r'^E\s+', '', line)
            if re.match(r'(?:[\w.]+(?:Error|Exception)|AssertionError):', value):
                yield number, value, 'runtime_exception'
            elif re.match(r'(?:error:|cause(?:d by)?:|go:|https?://|STD(?:OUT|ERR):)', value, re.I):
                yield number, value, 'runtime_exception'
            continue
        if line.lstrip().startswith(('>', '#')):
            continue
        if re.match(r'^\s{4,}\S', line) and region == 'source_context':
            continue
        if region != 'source_context' or re.match(r'^\S+(?:Error|Exception):', line):
            yield number, line, region


def normalize(attempt_dir, task, reward, evaluation_raw=None):
    attempt = Path(attempt_dir).resolve()
    contract = frozen_contract(task)
    snap = {'reward': reward, 'contract': contract, 'logs': [], 'ctrf_reports': [],
            'documents': {}, 'artifacts': {}, 'evidence': [], 'integrity_errors': [],
            'missing_evidence': [], 'source_changed': False, 'truncated': False,
            'attempt_path': str(attempt), 'raw': {}, 'source_checks': {}}
    total = 0

    def consume(path, role, *, required=False):
        nonlocal total
        relative = path.relative_to(attempt).as_posix()
        if not path.exists():
            if required:
                snap['missing_evidence'].append(relative)
                snap['integrity_errors'].append({'path': relative, 'error': 'required evidence missing'})
            snap['source_checks'][str(path)] = None
            return None
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(attempt):
                raise ValueError('unsafe evidence path')
            size = path.stat().st_size
            if size > MAX_FILE or total + size > MAX_TOTAL:
                snap['truncated'] = True
                raise ValueError('evidence read budget exceeded')
            data = path.read_bytes()
            total += len(data)
            digest = hashlib.sha256(data).hexdigest()
            snap['source_checks'][str(path)] = digest
            snap['evidence'].append({'path': relative, 'sha256': digest, 'size': len(data),
                                     'evidence_role': role})
            return data.decode('utf-8')
        except (OSError, ValueError, UnicodeError) as exc:
            snap['integrity_errors'].append({'path': relative, 'error': str(exc)})
            return None

    names = ['evaluation-raw.json', 'diagnostics/verifier-execution.json',
             'subject/subject.json', 'attempt-started.json']
    if 'agentops_ci' in contract['capabilities']:
        names.append('diagnostics/agentops-py37.json')
    if (attempt/'diagnostics/verifier-stages.json').is_file():
        names.append('diagnostics/verifier-stages.json')
    for name in names:
        text = consume(attempt/name, 'raw_evaluator' if name == 'evaluation-raw.json' else 'execution_receipt',
                       required=name == 'evaluation-raw.json' and evaluation_raw is None)
        if text is not None:
            try:
                doc = json.loads(text)
                if not isinstance(doc, dict):
                    raise ValueError('expected an object')
                snap['documents'][name] = doc
            except ValueError as exc:
                snap['integrity_errors'].append({'path': name, 'error': str(exc)})
    stages = snap['documents'].get('diagnostics/verifier-stages.json', {})
    for name, entry in stages.get('artifacts', {}).items():
        if not isinstance(entry, dict) or 'path' not in entry:
            continue
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts or not str(relative).replace('\\','/').startswith('diagnostics/verifier-stage-artifacts/'):
            snap['integrity_errors'].append({'path': str(relative), 'error': 'unsafe protected stage artifact'})
            continue
        text = consume(attempt/relative, 'protected_verifier_stage', required=True)
        if snap['source_checks'].get(str(attempt/relative)) != entry.get('sha256'):
            snap['integrity_errors'].append({'path': str(relative), 'error': 'protected stage hash changed'})
        if text is not None and 'data' in entry:
            try:
                parsed = [json.loads(line) for line in text.splitlines() if line.strip()] if name.endswith('.jsonl') else json.loads(text)
                if parsed != entry['data']:
                    raise ValueError('protected stage content differs from executor receipt')
            except ValueError as exc:
                snap['integrity_errors'].append({'path': str(relative), 'error': str(exc)})
    persisted = snap['documents'].get('evaluation-raw.json', {})
    snap['raw'] = evaluation_raw if evaluation_raw is not None else persisted
    if evaluation_raw is not None and persisted and hash_json(persisted) != hash_json(evaluation_raw):
        snap['integrity_errors'].append({'path': 'evaluation-raw.json', 'error': 'raw evaluator conflicts with persisted bytes'})
    raw_rewards = snap['raw'].get('rewards')
    if isinstance(raw_rewards, dict) and raw_rewards.get('reward') != reward:
        snap['integrity_errors'].append({'path': 'evaluation-raw.json', 'error': 'reward source mismatch'})
    files = verifier_files(attempt)
    snap['verifier_listing'] = [p.relative_to(attempt).as_posix() for p in files]
    for path in files:
        name = path.name
        role = None
        if name in TEXT_NAMES:
            role = 'verifier_stream'
        elif path.suffix == '.json' and 'ctrf' in name.lower():
            role = 'ctrf'
        elif name == 'reward.txt' or (path.parent.name == 'scores' and path.suffix == '.txt'):
            role = 'score_artifact'
        if role is None:
            continue
        text = consume(path, role)
        rel = path.relative_to(attempt).as_posix()
        if role == 'ctrf':
            row = {'path': rel, 'suite_id': path.stem, 'call_id': path.parent.relative_to(attempt).as_posix(),
                   'sha256': snap['source_checks'].get(str(path)), 'summary': {}, 'tests': [], 'valid': False}
            if text is not None:
                try:
                    doc = json.loads(text)
                    results = doc.get('results', {})
                    if not isinstance(results, dict):
                        raise ValueError('invalid CTRF results')
                    row.update(summary=results.get('summary') or {}, tests=results.get('tests') or [],
                               tool=results.get('tool'), valid=True)
                    if not isinstance(row['tests'], list) or not isinstance(row['summary'], dict):
                        raise ValueError('invalid CTRF tests/summary')
                    statuses = {'passed', 'failed', 'skipped', 'pending', 'other'}
                    if any(not isinstance(test, dict) or test.get('status') not in statuses
                           or not isinstance(test.get('name'), str) or not test['name']
                           for test in row['tests']):
                        raise ValueError('invalid CTRF test identity/status')
                    if any(type(value) is not int or value < 0 for key, value in row['summary'].items()
                           if key in statuses | {'tests'}):
                        raise ValueError('invalid CTRF summary count')
                except (ValueError, AttributeError) as exc:
                    row['valid'] = False
                    snap['integrity_errors'].append({'path': rel, 'error': str(exc)})
            snap['ctrf_reports'].append(row)
        elif role == 'verifier_stream' and text is not None:
            snap['logs'].append({'path': rel, 'text': text, 'events': list(runtime_lines(text)),
                                  'call_id': path.parent.relative_to(attempt).as_posix()})
        elif text is not None:
            snap['artifacts'][rel] = text
    for item in contract['files']:
        path = Path((task or {}).get('instance', {}).get('task_path', ''))/item['path']
        snap['source_checks'][str(path)] = item['sha256']
    subject = snap['documents'].get('subject/subject.json', {})
    started = snap['documents'].get('attempt-started.json', {})
    binding = subject.get('binding', {})
    if (started.get('attempt_id') and binding.get('attempt_id')
            and started['attempt_id'] != binding['attempt_id']):
        snap['integrity_errors'].append({'path':'subject/subject.json', 'error':'subject attempt identity mismatch'})
    return snap


def check_sources(snap):
    for name, old in snap['source_checks'].items():
        path = Path(name)
        try:
            new = sha256(path) if path.exists() else None
        except OSError:
            new = 'unreadable'
        if old != new:
            snap['source_changed'] = True
    current = [p.relative_to(Path(snap['attempt_path'])).as_posix()
               for p in verifier_files(snap['attempt_path'])]
    if current != snap['verifier_listing']:
        snap['source_changed'] = True

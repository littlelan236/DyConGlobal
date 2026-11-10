"""Conservative, attempt-local SkillsBench zero-reward verification evidence."""
from __future__ import annotations

import json
import re
from pathlib import Path

from core.io import hash_json, sha256

PYTEST_SUMMARY = re.compile(
    r"(?m)^\s*=+\s*(?=[^\n]*\b[1-9]\d*\s+(?:passed|failed|xfailed|xpassed)\b)"
    r"[^\n]*\b(?:passed|failed|xfailed|xpassed)\b[^\n]*=+\s*$", re.I)
TERMINAL_DEPENDENCY = re.compile(
    r"(?m)^.*(?:ERROR:\s*(?:Could not install packages|No matching distribution found|"
    r"Could not find a version that satisfies|THESE PACKAGES DO NOT MATCH THE HASHES)|"
    r"No solution found when resolving (?:tool )?dependencies|"
    r"(?:pytest|uvx|uv|python3?|pip3?):\s*(?:command )?not found|"
    r"(?:/[^\s:]+/)?pytest:\s*No such file or directory|"
    r"No module named ['\"]?(?:pytest|pytest_json_ctrf|pytest_ctrf)\b).*$", re.I)
DOWNLOAD_FAILURE = re.compile(r"Failed to (?:fetch|download)|Could not fetch URL", re.I)
UV_ABORT = re.compile(r"(?m)^\s*[×x]\s+Failed to (?:download|fetch)\b", re.I)
ABORT_CAUSE = re.compile(r"Caused by:|Failed to download after \d+ retr(?:y|ies)|"
                         r"Request failed after \d+ retr(?:y|ies)", re.I)
RECOVERY = re.compile(r"Successfully installed|Installed \d+ packages?|"
                      r"\bcollected [1-9]\d* items?\b|={2,}\s*test session starts", re.I)

DOWNLOAD_ABORT = re.compile(
    r'(?im)^\s*error:\s*(?:Failed to download\s+[\x60]?([A-Za-z0-9_.-]+)'
    r"|Request failed after\s+\d+\s+retr(?:y|ies))\b")
DOWNLOAD_FETCH = re.compile(
    r'(?im)^\s*Caused by:\s*Failed to fetch:?\s*[\x60]?'
    r'https?://[^/\s]+/(?:[^/\s]+/)*simple/([A-Za-z0-9_.-]+)/?')
INSTALL_CONTEXT = re.compile(
    r"(?im)^\s*(?:Looking in indexes:|Collecting\s+[A-Za-z0-9_.-]+|"
    r"Downloading\s+|Using CPython |Initialized project\s+|Resolved\s+\d+\s+packages?)")
DOWNLOAD_TIMEOUT = re.compile(
    r"(?im)(?:ReadTimeoutError|ConnectTimeoutError|WriteTimeoutError|PoolTimeout)")
MISSING_IMPORT = re.compile(
    r'(?im)No module named\s+([A-Za-z0-9_.-]+)')


def _package_name(name):
    return re.sub(r"[-_.]+", "-", (name or "").strip().lower())


def _missing_for_package(log, package, start):
    aliases = {
        "opencv-python": {"cv2"},
        "opencv-python-headless": {"cv2"},
        "pillow": {"pil"},
        "pyyaml": {"yaml"},
        "beautifulsoup4": {"bs4"},
        "pytest-json-ctrf": {"pytest_json_ctrf"},
        "scikit-learn": {"sklearn"},
        "python-dateutil": {"dateutil"},
    }
    expected = aliases.get(_package_name(package), {_package_name(package).replace("-", "_")})
    return any(match.group(1).strip(" '\"").lower() in expected
               for match in MISSING_IMPORT.finditer(log, start))


def _same_package_recovered(log, package, start):
    package = _package_name(package)
    tail = log[start:]
    for match in re.finditer(r"(?im)^\s*Successfully installed\s+([^\n]+)", tail):
        installed = {_package_name(item.split("==", 1)[0]) for item in match.group(1).split()}
        if package in installed:
            return True
    return bool(re.search(rf"(?im)^\s*\+\s+{re.escape(package)}==", tail))


def _download_failure(log):
    if not (INSTALL_CONTEXT.search(log) or DOWNLOAD_ABORT.search(log)
            or re.search(r"(?im)^\s*error:\s*Request failed after\s+\d+\s+retr(?:y|ies)", log)):
        return None
    match = DOWNLOAD_ABORT.search(log)
    if match:
        package = match.group(1)
        fetch = DOWNLOAD_FETCH.search(log, match.end())
        if package and (ABORT_CAUSE.search(log, match.end()) or fetch):
            if _same_package_recovered(log, package, match.end()):
                return None
            return package, _missing_for_package(log, package, match.end())
        if fetch:
            package = fetch.group(1)
            if not _same_package_recovered(log, package, match.end()):
                return package, _missing_for_package(log, package, match.end())
    request = re.search(
        r"(?im)^\s*error:\s*Request failed after\s+\d+\s+retr(?:y|ies)", log)
    if request:
        fetch = DOWNLOAD_FETCH.search(log, request.end())
        if fetch:
            package = fetch.group(1)
            if not _same_package_recovered(log, package, request.end()):
                return package, _missing_for_package(log, package, request.end())
    if re.search(r"(?im)^\s*ERROR:\s*Exception:", log) and DOWNLOAD_TIMEOUT.search(log):
        packages = list(re.finditer(r"(?im)^\s*Collecting\s+([A-Za-z0-9_.-]+)", log))
        if packages:
            package = packages[-1].group(1)
            if not _same_package_recovered(log, package, packages[-1].end()):
                return package, _missing_for_package(log, package, packages[-1].end())
    return None



def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeError):
        return None


def frozen_verifier_contract(task):
    """Read only files whose bytes match the run-selected package manifest.

    An unrecognized script is an unknown contract, never an assumed non-CTRF
    verifier. Non-CTRF completion patterns require a task-specific contract.
    """
    package = (task or {}).get('instance', {})
    contract = {'ctrf_expected': None, 'task_completion_patterns': [],
                'package_sha256': package.get('package_sha256'), 'files': [],
                'source': 'run_selected_package', 'verified': False}
    files = package.get('files', [])
    if not files or hash_json(files) != package.get('package_sha256'):
        return contract
    root = Path(package.get('task_path', '')).resolve()
    try:
        for member in files:
            relative = Path(member['path'])
            if relative.is_absolute() or '..' in relative.parts:
                return contract
            if relative.as_posix() not in {'tests/test.sh', 'test.sh', 'verifier/test.sh'}:
                continue
            path = root / relative
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                return contract
            if sha256(path) != member['sha256']:
                return contract
            text = path.read_text(encoding='utf-8')
            # Ignore comments: merely mentioning CTRF is not a verifier contract.
            commands = '\n'.join(line for line in text.splitlines()
                                 if not line.lstrip().startswith('#'))
            contract['files'].append({'path': member['path'], 'sha256': member['sha256']})
            contract['verified'] = True
            if re.search(r'--ctrf\b|(?:/logs/verifier/|[\s\"\x27])ctrf\.json\b', commands):
                contract['ctrf_expected'] = True
    except (OSError, ValueError, UnicodeError, KeyError):
        contract.update(ctrf_expected=None, verified=False)
    return contract


def _verifier_files(attempt_dir):
    root = Path(attempt_dir) / 'benchflow'
    if not root.is_dir():
        return []
    # Do not follow links outside the attempt or consume trainer/verifiers.jsonl.
    return sorted(p for p in root.rglob('*') if p.is_file() and not p.is_symlink()
                  and p.resolve().is_relative_to(root.resolve())
                  and 'verifier' in p.relative_to(root).parts[:-1]
                  and 'trainer' not in p.relative_to(root).parts[:-1])


def inspect_ctrf(attempt_dir):
    """Return (executed, present); all-skipped/empty reports prove no execution."""
    reports = [p for p in _verifier_files(attempt_dir)
               if p.suffix == '.json' and 'ctrf' in p.name.lower()]
    for path in reports:
        report = _read_json(path)
        results = report.get('results') if isinstance(report, dict) else None
        if not isinstance(results, dict):
            continue
        tests, summary = results.get('tests'), results.get('summary')
        if isinstance(tests, list) and any(isinstance(t, dict) and
                t.get('status') in {'passed', 'failed'} for t in tests):
            return True, True
        if isinstance(summary, dict):
            if any(type(summary.get(k)) is int and summary[k] > 0 for k in ('passed', 'failed')):
                return True, True
            total = summary.get('tests')
            if (type(total) is int and total > 0 and
                    sum(summary.get(k, 0) if type(summary.get(k, 0)) is int else 0
                        for k in ('skipped', 'pending')) < total):
                return True, True
    return False, bool(reports)


def classify_verifier_logs(logs, *, ctrf_executed=False, task_completion_patterns=(), unreadable=False):
    """Shared evidence policy with conservative terminal download detection."""
    if not unreadable:
        for log in logs:
            failure = _download_failure(log)
            if failure is not None and failure[1]:
                package, _ = failure
                return 'VerifierError', (
                    f'Verifier download failed for {package}; package remained missing at test collection')
    if ctrf_executed or any(PYTEST_SUMMARY.search(log) for log in logs):
        return 'ValidZero', 'Executed tests or pytest result summary found'
    if any(re.search(pattern, log, re.I) for pattern in task_completion_patterns for log in logs):
        return 'ValidZero', 'Task-specific verifier completion evidence found'
    if not unreadable:
        for index, log in enumerate(logs):
            failures = list(TERMINAL_DEPENDENCY.finditer(log))
            aborts = list(UV_ABORT.finditer(log)) if ABORT_CAUSE.search(log) else []
            failure = max(failures + aborts, key=lambda match: match.start(), default=None)
            if failure is None:
                continue
            if (RECOVERY.search(log[failure.end():]) or
                    any(RECOVERY.search(other) for i, other in enumerate(logs) if i != index)):
                continue
            if failure.re is TERMINAL_DEPENDENCY:
                return 'VerifierError', 'Verifier dependency/test launch failed: ' + failure.group(0).strip()[:180]
            return 'VerifierError', 'Verifier installer aborted after download/fetch failure'
        for log in logs:
            failure = _download_failure(log)
            if failure is not None:
                package, _ = failure
                return 'VerifierError', f'Verifier download failed for {package} before test execution'
    return 'Review', 'Insufficient evidence of completed tests or final verifier failure'

def classify_zero_reward_trajectory(attempt_dir, *, disposition, reward,
        evaluation_raw=None, evaluation_report=None, ctrf_expected=None,
        task_completion_patterns=()):
    """Return VerifierError, ValidZero, Review, or NotCandidate with a reason."""
    if disposition != 'accepted' or type(reward) not in (int, float) or reward != 0:
        return 'NotCandidate', 'Requires accepted and raw_reward.reward == 0'
    attempt_dir = Path(attempt_dir)
    raw = evaluation_raw if evaluation_raw is not None else _read_json(attempt_dir / 'evaluation-raw.json')
    report = evaluation_report if evaluation_report is not None else _read_json(attempt_dir / 'evaluation-report.json')
    raw = raw if isinstance(raw, dict) else {}
    report = report if isinstance(report, dict) else {}
    if raw.get('completed') is False and raw.get('error'):
        return 'VerifierError', 'evaluation-raw.json: completed=false with error'
    for name, document in [('evaluation-raw.json', raw), ('evaluation-report.json', report)]:
        for field in ('status', 'evaluation_status', 'success_policy'):
            if document.get(field) in ('completed_unknown', 'verifier_incomplete'):
                return 'VerifierError', name + ': ' + document[field]
    logs, unreadable = [], False
    for path in _verifier_files(attempt_dir):
        if path.name in {'test-stdout.txt', 'test-stderr.txt'}:
            try:
                logs.append(path.read_text(encoding='utf-8', errors='replace'))
            except OSError:
                unreadable = True
    executed, present = inspect_ctrf(attempt_dir)
    classification, reason = classify_verifier_logs(logs, ctrf_executed=executed,
        task_completion_patterns=task_completion_patterns, unreadable=unreadable)
    if classification != 'Review':
        return classification, reason
    if ctrf_expected is True and not present:
        return 'Review', 'Frozen verifier expects CTRF, but report is missing and no final failure is proven'
    if ctrf_expected is None and not present:
        return 'Review', 'CTRF contract unknown; missing report does not prove failure'
    if any(DOWNLOAD_FAILURE.search(log) for log in logs):
        return 'Review', 'Download/retry evidence does not establish an unrecovered final failure'
    return 'Review', 'Insufficient evidence of completed tests or final verifier failure'


def audit_zero_evaluation(attempt_dir, task, status, resolved, policy, raw):
    """Apply the audit before immutable rollout/evaluation receipts are written."""
    rewards = raw.get('rewards')
    reward = rewards.get('reward') if isinstance(rewards, dict) else None
    if status not in {'Submitted', 'AgentCompleted', 'AgentTimeout'} or reward != 0 or type(reward) not in (int, float):
        return status, resolved, policy, None
    contract = frozen_verifier_contract(task)
    classification, reason = classify_zero_reward_trajectory(attempt_dir,
        disposition='accepted', reward=reward, evaluation_raw=raw,
        ctrf_expected=contract['ctrf_expected'],
        task_completion_patterns=contract['task_completion_patterns'])
    evidence = []
    for path in [Path(attempt_dir) / 'evaluation-raw.json', *_verifier_files(attempt_dir)]:
        if path.name not in {'evaluation-raw.json', 'test-stdout.txt', 'test-stderr.txt'} and not (
                path.suffix == '.json' and 'ctrf' in path.name.lower()):
            continue
        try:
            evidence.append({'path': path.relative_to(attempt_dir).as_posix(), 'sha256': sha256(path)})
        except OSError:
            evidence.append({'path': path.relative_to(attempt_dir).as_posix(), 'unreadable': True})
    receipt = {'schema_version': 1, 'classification': classification, 'reason': reason,
               'contract': contract, 'reward': reward,
               'evidence_files': evidence,
               'policy': 'attempt_local_zero_reward_three_state_v1'}
    if classification == 'VerifierError':
        return 'VerifierError', None, 'verifier_incomplete', receipt
    return status, resolved, policy, receipt

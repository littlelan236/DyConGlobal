"""Structured dependency diagnostics shared by container, rollout and harness."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename

from core.io import hash_json, identifier, now, read_json, write_json
from core.locking import file_lock
from runtime.containers.errors import EnvironmentSetupFailed

INSTALL_FAILURE_MARKER = '__DYCON_DEPENDENCY_INSTALL_FAILED__'
_INSTALL_FAILURE = re.compile(r'^' + INSTALL_FAILURE_MARKER + r' rc=(\d+) command=(.*)$', re.M)

INSTALL_ERROR_GUARD = r'''
__dycon_dependency_error() {
    local dycon_rc="$1" dycon_cmd="$2"
    case "$dycon_cmd" in
        pip*\ install*|*/pip*\ install*|python*\ -m\ pip\ install*|*/python*\ -m\ pip\ install*|uv\ pip\ install*|conda\ install*|conda\ create*)
            printf '\n__DYCON_DEPENDENCY_INSTALL_FAILED__ rc=%s command=%s\n' "$dycon_rc" "$dycon_cmd" >&2
            exit "$dycon_rc"
            ;;
    esac
}
trap '__dycon_dependency_error "$?" "$BASH_COMMAND"' ERR
'''


def classify_dependency_failure(text, *, stage=None, returncode=None, requirements=(),
                                wheelhouse=None, reason=None, log_path=None, offline=False):
    text = re.sub(r'\x1b\[[0-9;]*m', '', text or '')
    marker = _INSTALL_FAILURE.search(text)
    if stage is None and marker is None:
        return None  # Test source/fixtures mentioning pip failures are not install failures.
    lower = text.lower()
    unavailable = []
    if offline and wheelhouse and Path(wheelhouse).is_dir():
        candidates = {}
        for path in Path(wheelhouse).glob('*.whl'):
            try:
                name, version, _, _ = parse_wheel_filename(path.name)
                candidates.setdefault(name, []).append(version)
            except ValueError:
                continue
        for value in requirements:
            requirement = Requirement(value)
            if not requirement.marker and not requirement.url and not any(
                    version in requirement.specifier for version in candidates.get(canonicalize_name(requirement.name), [])):
                unavailable.append(value)
    if reason is None:
        if any(s in lower for s in ('offline wheelhouse is empty', 'configured wheelhouse contains no wheels')):
            reason = 'wheelhouse_unavailable'
        elif 'conflicting offline requirement pins' in lower:
            reason = 'version_conflict'
        elif any(s in lower for s in ('requires a different python', 'not a supported wheel', 'requires-python')):
            reason = 'python_or_platform_incompatible'
        elif any(s in lower for s in ('temporary failure in name resolution', 'failed to establish a new connection',
                                     'connection reset', 'connectionerror', 'connection refused', 'proxyerror', 'sslerror')):
            reason = 'dependency_download_network_error'
        elif unavailable and any(s in lower for s in ('resolutionimpossible', 'conflicting dependencies',
                                                       'no matching distribution', 'could not find a version')):
            # pip can call a missing wheel + identical requirement/constraint a
            # ResolutionImpossible. Verify local candidates instead of inferring
            # mutually incompatible versions from that message alone.
            reason = 'missing_distribution'
        elif any(s in lower for s in ('no matching distribution found', 'could not find a version that satisfies',
                                     'packagesnotfounderror')):
            reason = 'missing_distribution'
        elif any(s in lower for s in ('resolutionimpossible', 'conflicting dependencies')):
            reason = 'version_conflict'
        elif marker or any(s in lower for s in ('pip subprocess', 'installing build dependencies', 'failed building wheel')):
            reason = 'dependency_install_failed'
        else:
            return None
    failed = []
    for pattern in (r'No matching distribution found for ([^\n\r]+)',
                    r'Could not find a version that satisfies the requirement ([^\n\r]+)'):
        for value in re.findall(pattern, text, re.I):
            value = value.split(' (from versions:')[0].strip()
            if value not in failed:
                failed.append(value)
    if unavailable and not failed:
        failed = unavailable
    detail = next((line.strip() for line in text.splitlines() if re.search(
        r'No matching distribution found|Could not find a version that satisfies|Conflicting offline requirement pins', line, re.I)), '')
    detail = detail or next((line.strip() for line in text.splitlines()
                   if re.search(r'ERROR:|ResolutionImpossible|Temporary failure|Conflicting offline|wheelhouse.*(?:empty|no wheels)', line, re.I)), '')
    result = {'failure_type': 'dependency_error', 'reason': reason,
              'stage': stage or 'harness_install', 'requirements': list(requirements),
              'failed_requirements': failed, 'wheelhouse': wheelhouse,
              'returncode': returncode if returncode is not None else int(marker.group(1)) if marker else None,
              'detail': (detail or text[-1500:])[:1500]}
    if marker:
        result['command'] = marker.group(2)[:1000]
    if log_path:
        result['log_path'] = str(log_path)
    if unavailable:
        result['candidate_check'] = {'offline': True, 'unavailable_wheel_requirements': unavailable}
    return result


def record_dependency_failure(cfg, context, failure):
    root = Path(cfg['project_root'])
    result = {**failure, 'run_id': context.run_id, 'instance_id': context.instance_id,
              'attempt_id': context.attempt_id, 'role': context.role, 'at': now(),
              'diagnostic_path': str(context.output_dir / 'dependency-failure.json')}
    try:
        write_json(context.output_dir / 'dependency-failure.json', result)
        run_id = identifier(context.run_id)
        index = root / 'state/environment-failures' / (run_id + '.json')
        with file_lock(index.with_suffix('.lock')):
            data = read_json(index) if index.exists() else {'schema_version': 1, 'run_id': run_id, 'failures': {}}
            key = hash_json({k: result[k] for k in ('instance_id', 'attempt_id', 'role', 'stage')})[:32]
            data['failures'][key] = result
            data.update(updated_at=now(), count=len(data['failures']),
                        reason_counts=dict(Counter(r['reason'] for r in data['failures'].values())))
            write_json(index, data)
    except Exception as exc:
        # Preserve the original installation failure even if diagnostic storage fails.
        result['recording_error'] = repr(exc)
    return result


def from_exception(exception):
    seen = set()
    while exception is not None and id(exception) not in seen:
        seen.add(id(exception))
        failure = getattr(exception, 'dependency_failure', None)
        if failure:
            return failure
        exception = exception.__cause__ or exception.__context__
    return None


def setup_error(cfg, context, message, *, stage, returncode=None, requirements=(),
                reason=None, log_path=None, wheelhouse=None, version_checks=()):
    failure = classify_dependency_failure(message, stage=stage, returncode=returncode,
        requirements=requirements, wheelhouse=wheelhouse or cfg['environment'].get('wheelhouse'),
        reason=reason, log_path=log_path, offline=not cfg['environment'].get('network_fallback', False))
    if failure:
        if version_checks:
            failure['version_checks'] = list(version_checks)
            failure['failed_requirements'] = [c['spec'] for c in version_checks]
        failure = record_dependency_failure(cfg, context, failure)
    return EnvironmentSetupFailed(message[-2000:], dependency_failure=failure)


def failure_status(root, run_id):
    path = Path(root) / 'state/environment-failures' / (identifier(run_id) + '.json')
    if not path.exists():
        return {'count': 0, 'reason_counts': {}, 'failures': {}, 'path': str(path)}
    return {**read_json(path), 'path': str(path)}

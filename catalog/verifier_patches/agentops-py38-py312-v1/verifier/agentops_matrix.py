"""Validate the protected five-interpreter AgentOps execution contract.

This exact module is also published inside the immutable verifier package.
"""
import re

REQUIRED = ('py38', 'py39', 'py310', 'py311', 'py312')
VERSIONS = {'py38': (3,8,20), 'py39': (3,9,25), 'py310': (3,10,22),
            'py311': (3,11,13), 'py312': (3,12,15)}


def validate_matrix(report, required=REQUIRED):
    issues, rows = [], []
    if not isinstance(report, dict) or not isinstance(report.get('testenvs'), dict):
        return {'complete': False, 'passed': False, 'issues': ['tox_report_missing_or_invalid'], 'environments': []}
    for name in required:
        env = report['testenvs'].get(name)
        if not isinstance(env, dict):
            issues.append(name + ':required_environment_missing')
            continue
        problems = []
        python = env.get('python', {})
        if tuple(python.get('version_info', [])[:3]) != VERSIONS.get(name):
            problems.append('required_interpreter_missing_or_wrong')
        setup = env.get('setup', [])
        latest = {}
        for cmd in setup:
            latest[cmd.get('run_id')] = cmd
        if not any(k == 'install_deps' for k in latest):
            problems.append('dependency_install_evidence_missing')
        if not any(k and k.startswith('install_package') for k in latest):
            problems.append('package_install_evidence_missing')
        if any(type(c.get('retcode')) is not int or c['retcode'] != 0 for c in latest.values()):
            problems.append('required_setup_failed')
        tests = env.get('test', [])
        test = tests[0] if tests else {}
        args = test.get('command', [])
        if ([str(a).rsplit('/', 1)[-1] for a in args[:1]] != ['coverage']
                or args[1:] != ['run', '--source', 'agentops', '-m', 'pytest']):
            problems.append('required_pytest_command_missing_or_changed')
        output = test.get('output', '')
        summaries = re.findall(r'^=+\s+(.+?)\s+in\s+[\d.]+s\s+=+\s*$', output, re.M)
        counts = {}
        if summaries:
            counts = {k: int(n) for n, k in re.findall(
                r'(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)', summaries[-1])}
        if (not re.search(r'^=+ test session starts =+', output, re.M)
                or counts.get('passed', 0) + counts.get('failed', 0) < 1
                or counts.get('error', 0) or counts.get('errors', 0)):
            problems.append('completed_test_results_missing')
        code = test.get('retcode')
        if type(code) is not int or code not in (0, 1) or bool(code) != bool(counts.get('failed', 0)):
            problems.append('pytest_exit_and_results_inconsistent')
        passed = code == 0 and not problems
        if passed:
            coverage = tests[1] if len(tests) > 1 else {}
            args = coverage.get('command', [])
            if (not args or str(args[0]).rsplit('/', 1)[-1] != 'coverage'
                    or args[1:] != ['report', '-m'] or coverage.get('retcode') != 0):
                problems.append('coverage_report_missing_or_failed')
        result = env.get('result', {})
        if result.get('success') is not (code == 0) or result.get('exit_code') != code:
            problems.append('environment_result_inconsistent')
        issues.extend(name + ':' + item for item in problems)
        rows.append({'environment': name, 'python': python, 'counts': counts,
                     'return_code': code, 'passed': code == 0, 'issues': problems})
    return {'complete': not issues, 'passed': not issues and all(r['passed'] for r in rows),
            'issues': issues, 'environments': rows}

"""Explicit frozen path-parameter contracts; never match test basenames."""
import json
import re

KEY = 'DYCON_SWEBENCH_TEST_IDENTITIES'
PATH = re.compile(r'/(?:testbed/pydicom/data/test_files|n/fs/p-swe-bench/temp/[^/]+/pydicom__pydicom__[^/]+/pydicom/data/test_files|n/fs/guest/[^/]+/\.pydicom/data)/')


def path_identity(nodeid):
    """Keep method, every parameter filename, order and suffix exactly intact."""
    return PATH.sub('/<pydicom-data>/', nodeid)


def contract(task):
    text = task.get('environment_overrides', {}).get('variables', {}).get(KEY)
    if not text:
        return []
    if task.get('repo') != 'pydicom/pydicom':
        raise ValueError('Path identity contracts are limited to pydicom')
    rows = json.loads(text)
    required = set(json.loads(task['FAIL_TO_PASS']) if isinstance(task['FAIL_TO_PASS'], str) else task['FAIL_TO_PASS'])
    required.update(json.loads(task['PASS_TO_PASS']) if isinstance(task['PASS_TO_PASS'], str) else task['PASS_TO_PASS'])
    if not isinstance(rows, list) or not rows or len(rows) > 1000:
        raise ValueError('Invalid path identity contract')
    for row in rows:
        if (set(row) != {'required', 'raw'} or row['required'] not in required
                or row['raw'] == row['required'] or '/testbed/pydicom/data/test_files/' not in row['raw']
                or path_identity(row['required']) != path_identity(row['raw'])):
            raise ValueError('Path identity contract changes the test or parameter vector')
    for key in ('required', 'raw'):
        if len({row[key] for row in rows}) != len(rows):
            raise ValueError('Ambiguous path identity contract')
    return rows


def parser_for(task, original):
    rows = contract(task)
    if not rows:
        return original

    def parse(log, spec):
        # Parse original raw IDs independently. The official options parser
        # discards all but the final path segment, including multi-path vectors.
        statuses = dict(original(log, spec))
        observed = {}
        for line in log.splitlines():
            match = re.fullmatch(r'(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS) (\S+)(?: .*)?', line)
            if match:
                state, nodeid = match.groups()
                observed.setdefault(nodeid, []).append(state)
        for row in rows:
            values = observed.get(row['raw'], [])
            if not values:
                continue
            if len(values) != 1 or row['required'] in observed or row['required'] in statuses:
                raise ValueError('Ambiguous physical path-parameter result')
            statuses[row['required']] = values[0]
        return statuses
    return parse


def validate_collection(task, execution):
    """Mapped results require a unique bound physical collection and call."""
    rows = contract(task)
    if not rows:
        return
    sessions = execution.get('pytest_sessions', [])
    if len(sessions) != 1 or sessions[0].get('binding') != execution.get('pytest_binding'):
        raise ValueError('Path identity contract lacks a bound physical session')
    session = sessions[0]
    for row in rows:
        if session.get('collected', []).count(row['raw']) != 1:
            raise ValueError('Path identity contract does not match current collection')
        phases = [r for r in session.get('reports', []) if r['nodeid'] == row['raw']]
        if sorted(r['when'] for r in phases) != ['call', 'setup', 'teardown']:
            raise ValueError('Path identity contract lacks complete physical phases')

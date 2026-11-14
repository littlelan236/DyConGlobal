"""Explicit read-only scoring adapters for content-bound frozen verifiers."""
from __future__ import annotations

import ast
import re

SUMMARY = re.compile(r'(?m)^\s*=+\s*([^\n]*\b\d+ (?:passed|failed|error|errors|skipped)[^\n]*)=+\s*$')


def pytest_counts(text):
    rows = []
    for match in SUMMARY.finditer(text):
        counts = {k.rstrip('s') if k.startswith('error') else k: int(v)
                  for v, k in re.findall(r'(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)', match.group(1))}
        if sum(counts.get(k, 0) for k in ('passed', 'failed', 'error')):
            rows.append({'line': text.count('\n', 0, match.start())+1, **counts})
    return rows


def report_completed(report):
    return report['valid'] and any(t.get('status') in {'passed', 'failed'} for t in report['tests']
                                     if isinstance(t, dict))


def score_check(snap):
    contract, reward = snap['contract'], snap['reward']
    mode = contract['score_mode']
    check = {'mode': mode, 'recorded_reward': reward, 'artifact_score': None, 'expected_score': None,
             'matched': False, 'format': None, 'source': None, 'complete': False}
    logs = [r for r in snap['logs'] if r['path'].endswith('test-stdout.txt')]
    text = '\n'.join(r['text'] for r in logs)
    reports = snap['ctrf_reports']
    counts = pytest_counts(text)
    final = counts[-1] if counts else {}
    check['pytest_counts'] = final
    expected = None
    score_artifacts = [v.strip() for k, v in snap['artifacts'].items() if k.endswith('/reward.txt')]
    try:
        values = [float(v) for v in score_artifacts]
    except ValueError:
        check['issue'] = 'invalid_score_artifact'
        return check
    if len(values) > 1:
        check['issue'] = 'duplicate_score_artifact'
        return check
    if values:
        check['artifact_score'] = values[0]
    if not contract['verified'] or contract['status'] != 'supported':
        check['issue'] = 'unsupported_scoring_contract'
        return check
    if mode == 'custom_groups':
        expected, details = custom_groups(text)
        check.update(details, format='python round(4), REWARD .4f', source='bound custom group events')
    elif mode == 'cif_fraction':
        expected, details = cif_fraction(text, contract)
        check.update(details, format='Python .2f', source='11 unique CIF case outcomes')
    elif mode == 'pytest_exit_binary':
        if final:
            expected = 0.0 if final.get('failed', 0) + final.get('error', 0) else 1.0
        check.update(format='pytest exit binary', source='final pytest outcome')
    elif mode == 'ctrf_fraction_round3':
        if len(reports) == 1:
            summary = reports[0]['summary']
            n, p = summary.get('tests'), summary.get('passed')
            if type(n) is int and n > 0 and type(p) is int and 0 <= p <= n and report_completed(reports[0]):
                expected = round(p/n, 3)
        check.update(format='Python round(3)', source='CTRF summary passed/tests')
    elif mode == 'pytest_fraction_round3':
        p, f = final.get('passed', 0), final.get('failed', 0)
        if p+f:
            expected = round(p/(p+f), 3)
        check.update(format='Python round(3)', source='final pytest passed/(passed+failed)')
    elif mode == 'command_success_fraction':
        matches = re.findall(r'^=== Success count: (\d+)/(\d+) commands passed ===\s*$', text, re.M)
        success = re.findall(r'^Successful\s*:\s*(\[[^\n]*\])\s*$', text, re.M)
        failed = re.findall(r'^Failed\s*:\s*(\[[^\n]*\])\s*$', text, re.M)
        if len(matches) == len(success) == len(failed) == 1:
            p, n = map(int, matches[0])
            try:
                a, b = ast.literal_eval(success[0]), ast.literal_eval(failed[0])
                labels = set(contract.get('command_labels', []))
                if (n > 0 and isinstance(a, list) and isinstance(b, list) and all(isinstance(v, str) for v in a+b)
                        and len(set(a+b)) == len(a+b) == n and len(a) == p and labels == set(a+b)):
                    expected = float(f'{p/n:.6f}')
                    check['command_counts'] = {'successful': p, 'total': n, 'unique_labels': sorted(labels)}
            except (ValueError, SyntaxError, TypeError):
                pass
        check.update(format='Python .6f', source='unique command outcomes')
    elif mode == 'suite_success_fraction_truncate2':
        names = [r['path'].split('/')[-1] for r in reports]
        required = contract['expected_reports']
        if sorted(names) == sorted(required) and all(report_completed(r) for r in reports):
            passed = sum(not any(t.get('status') == 'failed' for t in r['tests']) for r in reports)
            expected = (passed*100//6)/100
            check['suite_counts'] = {'passed': passed, 'total': 6}
        check.update(format='bc scale=2 truncation', source='six independent suite outcomes')
    elif mode == 'scenario_score_or_zero':
        matches = re.findall(r'^scenario_3: (\d+\.\d{3})(?: \((no score file|parse error)\))?$', text, re.M)
        finals = re.findall(r'^Final score: (\d+\.\d{3})$', text, re.M)
        if len(matches) == len(finals) == 1 and matches[0][0] == finals[0]:
            value, branch = matches[0]
            if branch == 'no score file' and final.get('failed', 0)+final.get('error', 0):
                # The frozen scenario fixture stops on invalid/missing submitted output.
                expected = 0.0
                check['branch'] = 'agent_output_hard_gate'
            elif not branch:
                artifacts = [v for k, v in snap['artifacts'].items() if k.endswith('/scores/scenario_3.txt')]
                if len(artifacts) == 1:
                    try:
                        if float(f'{float(artifacts[0]):.3f}') == float(value):
                            expected = float(value)
                    except ValueError:
                        check['issue'] = 'invalid_scenario_score_artifact'
        check.update(format='Python .3f', source='scenario_3 artifact or frozen output hard gate')
    elif mode == 'separate_score_script':
        if final.get('failed', 0)+final.get('error', 0):
            expected = 0.0
            check['branch'] = 'format_hard_gate'
        elif final.get('passed', 0):
            matches = re.findall(r'^Policy-ladder reward: ([\d.]+) \(([^\n]+)\)$', text, re.M)
            ladder = {
                'contains non-abort transaction ids': 0., 'exact unnecessary-abort policy': 1.,
                'no useful unnecessary-abort signal': .05, 'treats abort records broadly as unnecessary': .10,
                'small hard/soft abort boundary error': .85, 'hard/soft abort boundary error': .35,
                'includes hard aborts and misses unnecessary aborts': .15,
                'all or nearly all soft aborts treated as unnecessary': .45, 'right-boundary timestamp error': .55,
                'minor soft-abort necessity error': .85, 'mixed soft-abort policy': .55,
                'over-strict timestamp boundary': .25, 'over-strict timestamp-window policy': .35,
                'partial exact subset': .50, 'large exact subset': .70, 'nearly complete exact subset': .85}
            if len(matches) == 1 and matches[0][1] in ladder:
                expected = ladder[matches[0][1]]
                if float(matches[0][0]) != expected:
                    expected = None
                check['branch'] = matches[0][1]
        check.update(format='Python .12f', source='format gate and independently mapped ladder reason')
    check['expected_score'] = expected
    check['complete'] = expected is not None
    check['matched'] = expected is not None and expected == reward and (not values or values[0] == expected)
    return check


def custom_groups(text):
    marker = '=== Anti-cheat checks ==='
    if text.count(marker) != 1:
        return None, {'issue': 'missing_or_duplicate_custom_invocation'}
    text = text.split(marker, 1)[1]
    sections = re.split(r'=== Bug [123]:[^\n]*===', text)
    groups = [
        ['test_trl_source_exists', 'test_train_script_not_modified', 'test_reward_fn_not_modified'],
        ['test_selective_log_softmax_correctness', 'test_log_probs_nonpositive', 'test_log_probs_highest_token'],
        ['test_advantage_epsilon_reasonable', 'test_advantages_not_vanishing'],
        ['test_decode_preserves_plain_completions', 'test_decode_extracts_after_think',
         'test_decode_handles_incomplete_think', 'test_reward_nonzero_after_decode']]
    if len(sections) != 4:
        return None, {'issue': 'missing_group'}
    outcomes = []
    for section, required in zip(sections, groups):
        events = re.findall(r'^  (PASS|FAIL|ERROR): (test_\w+)(?: -- ([^\n]*))?$', section, re.M)
        if not events or len(events) > len(required):
            return None, {'issue': 'incomplete_group'}
        for index, (status, name, _) in enumerate(events):
            if name != required[index] or (status != 'PASS' and index != len(events)-1):
                return None, {'issue': 'invalid_group_short_circuit'}
            if status == 'ERROR':
                return None, {'issue': 'custom_scorer_exception'}
        success = all(e[0] == 'PASS' for e in events)
        if success and len(events) != len(required):
            return None, {'issue': 'incomplete_group'}
        outcomes.append(success)
    expected = round(sum(w for w, yes in zip((.35, .25, .40), outcomes[1:]) if yes), 4) if outcomes[0] else 0.
    finals = re.findall(r'^REWARD=([\d.]+)$', text, re.M)
    if len(finals) != 1 or float(finals[0]) != expected:
        return None, {'issue': 'custom_score_mismatch'}
    return expected, {'group_results': outcomes}


def cif_fraction(text, contract):
    marker = 'Testing Wyckoff Position Analysis'
    if text.count(marker) != 1:
        return None, {'issue': 'missing_or_duplicate_custom_invocation'}
    text = text.split(marker, 1)[1]
    source = contract['sources'].get('verifier/test_outputs.py', '')
    names = re.findall(r'^\s*"([^"\n]+\.cif)":\s*\{', source, re.M)
    cases = re.findall(r'^Test (\d+): ([^\n]+\.cif)\s*\n(.*?)(?=^Test \d+:|^Summary\s*$)', text, re.M|re.S)
    if len(names) != 11 or [name for _, name, _ in cases] != names or [int(i) for i, _, _ in cases] != list(range(1,12)):
        return None, {'issue': 'missing_or_duplicate_cif_case'}
    failed = []
    for _, name, body in cases:
        events = re.findall(r'^[✓✗] (PASSED|FAILED)([^\n]*)$', body, re.M)
        if len(events) != 1 or 'exception' in events[0][1]:
            return None, {'issue': 'cif_exception_or_ambiguous_case'}
        if events[0][0] == 'FAILED':
            if '  Expected:' not in body or '  Got:' not in body:
                return None, {'issue': 'cif_failure_unexplained'}
            failed.append(name)
    p = 11-len(failed)
    summaries = re.findall(r'Total tests: (\d+)\nPassed: (\d+)\nFailed: (\d+)', text)
    final = re.findall(r'^Score: ([\d.]+)$', text, re.M)
    failure_lists = re.findall(r'^Failed tests: ([^\n]+)$', text, re.M)
    expected = float(f'{p/11:.2f}')
    if (summaries != [('11', str(p), str(len(failed)))] or len(final) != 1
            or float(final[0]) != expected or (failed and failure_lists != [', '.join(failed)])):
        return None, {'issue': 'cif_count_or_score_mismatch'}
    return expected, {'case_counts': {'passed': p, 'failed': len(failed), 'total': 11}}

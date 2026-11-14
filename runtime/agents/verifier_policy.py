"""The single v2 decision function. Facts enter here; no execution or settlement."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from runtime.agents.verifier_scores import pytest_counts, report_completed, score_check

POLICY = 'skillsbench_verifier_audit_v2'
PYTEST_DEPS = {'pytest', 'pytest-json-ctrf', 'pytest_json_ctrf', 'pluggy', 'iniconfig',
               'packaging', 'pygments', 'exceptiongroup', 'typing-extensions', 'tomli'}


def failure(rule, code, stage, location, *, dependency=None, detail=None, suite='ctrf', call=None):
    return {'rule': rule, 'reason_code': code, 'stage': stage, 'suite_id': suite,
            'verifier_call': call, 'environment': stage, 'dependency': dependency,
            'ownership': 'verifier', 'state': 'verifier_aborted', 'required': True,
            'root_cause': 'unknown', 'location': location, 'detail': detail}


def location(row, line=None, pointer=None):
    return {'path': row['path'], **({'decoded_line': line, 'json_pointer': pointer} if pointer else {'line': line})}


def runtime_exceptions(report):
    for i, test in enumerate(report['tests']):
        if not isinstance(test, dict) or test.get('status') != 'failed':
            continue
        trace = str(test.get('trace', ''))
        # Pytest source expressions (>, assert expansions) are not event messages.
        exceptions = []
        for n, raw in enumerate(trace.splitlines(), 1):
            match = re.match(r'^E\s+((?:[\w.]+(?:Error|Exception)|AssertionError):.*)', raw)
            if match:
                exceptions.append((n, match.group(1)))
            elif re.match(r'^[\w.]+(?:Error|Exception):', raw):
                exceptions.append((n, raw))
        yield i, test, trace, exceptions


def exception_owner(contract, trace, message):
    """Reviewed agent-error endpoints; unknown exceptions never prove completion.

    Capabilities are authorized by the full frozen source identity, rather than
    by the task name, sample identity, status, or the recorded score.
    """
    caps = contract['capabilities']
    sources = '\n'.join(contract['sources'].values())
    if message.startswith('AssertionError:'):
        return 'task_assertion'
    if 'simpo_independent_reference' in caps and 'TimeoutExpired:' in message and 'timeout=600' in sources:
        return 'simpo_submitted_unit_test_timeout'
    missing = re.search(r"FileNotFoundError:.*['\"](/root/[^'\"]+)['\"]$", message)
    if missing:
        path = missing.group(1)
        output = re.search(r'(?m)^OUTPUT_FILE\s*=\s*[\x27\"]'+re.escape(path)+r'[\x27\"]', sources)
        if output and path in contract['sources'].get('task.md', ''):
            return 'required_submitted_output'
    if ('submitted_csv_values' in caps and 'ValueError:' in message
            and 'could not convert string' in message and 'loaded_csvs' in trace
            and 'np.loadtxt(path' in sources and 'csv_paths' in trace):
        return 'submitted_csv_value'
    if ('submitted_csv_values' in caps and 'ValueError: operands could not be broadcast together' in message
            and 'test_wigner_values_match_reference' in trace
            and 'np.allclose(actual, expected' in sources and 'loaded_csvs' in trace):
        return 'submitted_csv_shape'
    if ('submitted_parallel_code' in caps and '_pickle.UnpicklingError:' in message
            and 'workspace/parallel_solution.py:' in trace
            and 'from parallel_solution import batch_search_parallel' in sources):
        return 'submitted_parallel_code'
    if ('submitted_generator_ids' in caps and 'IndexError: index ' in message
            and 'out of bounds' in message and 'Pg[idx]' in trace
            and 'for g in report["generators"]:' in sources):
        return 'submitted_generator_id'
    if ('submitted_compression_report' in caps and 'FileNotFoundError:' in message
            and message.endswith("'compression_report.json'")
            and 'REPORT_PATH = "compression_report.json"' in sources):
        return 'submitted_compression_report'
    if ('submitted_scenario_json' in caps and 'FileNotFoundError:' in message
            and message.endswith("'/output/scenario_3.json'")
            and 'OUTPUT_DIR = Path("/output")' in sources):
        return 'submitted_scenario_json'
    if ('fuzz_driver' in caps and 'test_fuzz_driver_runs' in trace
            and re.search(r'AssertionError: Running fuzz\.py for \w+ failed with code (?:1|77):', trace)
            and re.search(r'File ["\x27]/app/(?:arrow|black|ujson|ipython|minisgl)/', trace)
            and 'fuzz.py' in sources
            and ('ModuleNotFoundError:' in message or 'ValueError:' in message or 'FileNotFoundError:' in message)):
        return 'submitted_fuzz_driver'
    return None


def nested_failures(snap, issues):
    contract = snap['contract']
    caps = contract['capabilities']
    failures = []
    for report in snap['ctrf_reports']:
        for index, test, trace, exceptions in runtime_exceptions(report):
            name = str(test.get('name', ''))
            messages = '\n'.join(e for _, e in exceptions)
            where = location(report, exceptions[-1][0] if exceptions else None,
                             f'/results/tests/{index}/trace')
            common = {'suite': report['suite_id'], 'call': report['call_id']}
            missing_module = re.search(r"(?:ModuleNotFoundError|ImportError): No module named ['\"]?([\w.]+)", messages)
            if missing_module and missing_module.group(1) in {'pytest','pytest_json_ctrf','pytest_ctrf'}:
                failures.append(failure('R3','verifier_dependency_missing','test_launch',where,
                                        dependency=missing_module.group(1),**common))
            missing_file = re.search(r"FileNotFoundError: [^\n]*['\"](/(?:verifier|tests)/[^'\"]+)['\"]", messages)
            if missing_file:
                member = re.sub(r'^/(?:verifier|tests)/','',missing_file.group(1))
                official = contract.get('official_input_members',[])
                if 'verifier/'+member in official or 'tests/'+member in official:
                    failures.append(failure('R4','official_reference_input_missing','official_input',where,
                                            dependency=missing_file.group(1),**common))
                else:
                    issues.append({'code':'reference_input_ownership_unknown','location':where})
            if exceptions:
                issues.append({'code': 'test_failure', 'name': name, 'raw_status': test.get('raw_status'),
                               'location': where, 'evidence_role': 'runtime_exception'})
            for line, message in exceptions:
                owner = exception_owner(contract, trace, message)
                if owner:
                    issues.append({'code': 'task_result_failure', 'ownership': 'agent',
                                   'endpoint': owner, 'location': location(report, line, f'/results/tests/{index}/trace')})
                elif not (missing_module and missing_module.group(1) in {'pytest','pytest_json_ctrf','pytest_ctrf'}) and not missing_file:
                    issues.append({'code': 'exception_ownership_unknown', 'exception': message.split(':', 1)[0],
                                   'location': location(report, line, f'/results/tests/{index}/trace')})
            if ('mathjax_fixture' in caps and 'mathjax_renderer' in trace
                    and 'MathJaxRenderer' in trace
                    and 'MathJax.startup.promise' in '\n'.join(contract['sources'].values())
                    and re.search(r'TimeoutError: Page\.wait_for_function: Timeout \d+ms exceeded', messages)):
                failures.append(failure('R1', 'required_fixture_initialization_timeout', 'mathjax_renderer',
                                        where, detail='Required official renderer initialization timed out.', **common))
            if 'required_go_build' in caps:
                # The assertion message carries the real child command stderr;
                # the source literal alone is never sufficient.
                actual = re.search(r'AssertionError: make (descriptions|all) failed:', messages)
                if actual and re.search(r'go:.*(?:download|toolchain)|golang\.org/toolchain', messages+trace):
                    tail = messages[messages.find(actual.group(0)):]
                    runtime_go = '\n'.join(re.sub(r'^E\s+', '', trace_line) for trace_line in trace.splitlines()
                                           if trace_line.startswith('E '))
                    if re.search(r'(?:EOF|TLS handshake|connection reset|timed? out)', tail+runtime_go, re.I):
                        failures.append(failure('R2', 'required_toolchain_download_failed', 'make '+actual.group(1),
                                                where, dependency='Go toolchain', **common))
            if ('fuzz_driver' in caps and report['suite_id'].startswith('ctrf_')
                    and 'test_fuzz_driver_runs' in name):
                actual = re.search(r'AssertionError: Running fuzz\.py for (\w+) failed with code (\d+):', messages)
                if actual and actual.group(2) == '2':
                    # Child streams are concatenated by the frozen helper. A
                    # terminal return code binds termination without inventing
                    # a chronology across those streams.
                    output = '\n'.join(re.sub(r'^E\s+', '', trace_line) for trace_line in trace.splitlines()
                                       if trace_line.startswith('E '))
                    if re.search(r'Request failed after \d+ retr(?:y|ies)', output, re.I):
                        package = re.search(r'(?:Failed to download\s+[`\x27]?([\w.-]+)|simple/([\w.-]+))', output, re.I)
                        failures.append(failure('R2', 'required_subsuite_dependency_download_failed',
                                                'fuzz_driver:'+actual.group(1), where,
                                                dependency=next((g for g in package.groups() if g), None) if package else 'uv dependency',
                                                detail='Bound uv child exited 2 after download retries; child stream ordering unknown.',
                                                **common))
    return failures


def install_failures(snap, recoveries, issues):
    failures = []
    script = '\n'.join(v for k, v in snap['contract']['sources'].items() if k.endswith('test.sh'))
    installer = bool(re.search(r'\b(?:uvx|uv|pip3?)\b', script))
    deps = set(PYTEST_DEPS)
    deps.update(re.findall(r'--with[=\s]+([\w.-]+)', script))
    for log in snap['logs']:
        # Auxiliary copied logs duplicate stdout and cannot establish order.
        if not log['path'].endswith(('test-stdout.txt', 'test-stderr.txt')):
            continue
        events = log['events']
        starts = [i for i, (_, t, _) in enumerate(events) if re.search(r'=+ test session starts', t)]
        for index, (line, value, role) in enumerate(events):
            if role == 'source_context':
                continue
            kind, obj, code = None, None, 'dependency_install_failed'
            # Only installer stream events before a pytest call, or explicit
            # command failure after its final summary, belong to this stage.
            terminal = re.search(r'^\s*(?:error:\s*(?:Request failed after \d+ retr(?:y|ies)|Failed to download)|'
                                 r'ERROR:\s*(?:No matching distribution|Could not find a version|Could not install packages|THESE PACKAGES)|'
                                 r'[×x]\s+Failed to download)', value, re.I)
            if terminal:
                window = '\n'.join(t for _, t, _ in events[max(0,index-5):min(len(events),index+10)])
                runtime = re.search(r'cpython[-\s]([\d.]+)|python-build-standalone', window, re.I)
                if runtime and re.search(r'Request failed after \d+ retr|(?:cause(?:d by)?:).*|TLS|EOF', window, re.I):
                    kind = 'python'
                    obj = 'CPython '+(runtime.group(1) or 'runtime')
                    code = 'python_runtime_download_failed'
                else:
                    match = re.search(r'(?:Failed to download\s+[`\x27]?|simple/|distribution found for\s+|requirement\s+)([\w.-]+)', window, re.I)
                    if match and (re.search(r'Request failed after|cause(?:d by)?:|No matching distribution|Could not find a version', window, re.I)):
                        obj = match.group(1).rstrip('.')
                        if obj.lower().replace('_','-') in {d.replace('_','-') for d in deps}:
                            kind = 'package'
                        elif installer and re.search(r'(?:pip3? install[^\n]*\b'+re.escape(obj)+r'\b)', script):
                            kind = 'package'
                        else:
                            issues.append({'code': 'dependency_ownership_unknown', 'dependency': obj,
                                           'location': location(log, line)})
            launch = re.search(r'(?:pytest|uvx|uv|python3?): (?:command not found|No such file or directory)|No module named [\x27\"]?pytest\b', value)
            if launch and (installer or 'pytest' in script):
                kind, obj, code = 'launch', launch.group(0), 'verifier_test_launch_failed'
            if not kind:
                continue
            # A runtime error inside arbitrary agent test output cannot become
            # an installer failure just by printing installer-like text.
            if starts and index > starts[0] and role == 'captured_stream':
                continue
            later = '\n'.join(t for _, t, _ in events[index+1:])
            normalized_obj = (obj or '').lower().replace('_','-')
            installed = bool(re.search(r'(?:Successfully installed[^\n]*\b'+re.escape(normalized_obj)+r'(?:[-=\s]|$)|'
                                       r'\+\s*'+re.escape(normalized_obj)+r'==)', later, re.I)) if kind == 'package' else False
            if kind == 'python':
                version = obj.split(' ',1)[1]
                installed = bool(re.search(r'(?:Installed|Successfully installed) (?:Python|CPython) '+re.escape(version), later, re.I))
            complete_later = bool(pytest_counts(later)) and bool(re.search(r'=+ test session starts', later))
            if installed:
                recoveries.append({'dependency': obj, 'stage': 'installer', 'location': location(log,line),
                                   'evaluation_recovered': False, 'call_id': log['call_id'],
                                   'later_test_summary_observed': complete_later,
                                   'environment_identity': 'unknown'})
                issues.append({'code': 'recovery_environment_unknown' if complete_later else
                               'install_recovered_tests_incomplete', 'dependency': obj})
                continue
            others = [v['text'] for v in snap['logs'] if v is not log and v['call_id'] == log['call_id']]
            if any(re.search(r'Successfully installed|Installed (?:CPython|Python)|test session starts', t, re.I) for t in others):
                # Cross-stream success is deliberately not used as recovery.
                issues.append({'code': 'ordering_unknown', 'dependency': obj, 'location': location(log,line)})
                continue
            failures.append(failure('R3' if kind == 'launch' else 'R2', code, 'verifier_install',
                                    location(log,line), dependency=obj, call=log['call_id']))
    return failures


def agentops_failures(snap, issues):
    if 'agentops_ci' not in snap['contract']['capabilities']:
        return []
    sources = '\n'.join(snap['contract']['sources'].values())
    logs = [log for log in snap['logs'] if log['path'].endswith('test-stdout.txt')]
    failures = []
    diagnostic = snap['documents'].get('diagnostics/agentops-py37.json', {})
    metadata_path = Path(__file__).with_name('verifier_package_metadata.json')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    metadata_verified = all(
        any(item['package'] == name and item['version'] == version
            and item['requires_python'] == '>=3.8.1,<4.0'
            and len(item.get('response_sha256', '')) == 64 for item in metadata)
        for name, version in [('langchain', '0.0.352'), ('langchain-core', '0.1.23')])
    for log in logs:
        text = log['text']
        # Captured child output begins with the official CI group markers;
        # source print statements cannot satisfy these runtime boundaries.
        run_tox = re.search(r'^##\[group\]Run tox\s*$', text, re.M)
        install = re.search(r'^##\[group\]Run [\x27\"]?pip install tox[\x27\"]?\s*$', text, re.M)
        if install:
            tail = text[install.end():run_tox.start() if run_tox else len(text)]
            abort = re.search(r'^##\[error\]Process completed with exit code 2\.', tail, re.M)
            if abort and not run_tox and 'pip install tox' in sources:
                issues.append({'code': 'stderr_not_captured'})
                failures.append(failure('R2', 'agentops_required_tox_install_aborted', 'pip_install_tox',
                                        location(log,text.count('\n',0,install.start())+1), dependency='tox',
                                        detail='Official required install exited 2 before Run tox; stderr was discarded.', call=log['call_id']))
        py37 = re.search(r'^\s*py37: (?:exit [1-9]|FAIL|failed)', text, re.M)
        constraints = 'PIP_CONSTRAINT' in sources and 'langchain==0.0.352' in sources and 'langchain-core==0.1.23' in sources
        owned = diagnostic.get('ownership', {})
        physical = snap['documents'].get('attempt-started.json', {}).get('attempt_id')
        bound = (physical and owned.get('attempt') == physical and owned.get('task') == 'fix-build-agentops')
        resolution = 'ResolutionImpossible' in diagnostic.get('stdout','') and diagnostic.get('source') == 'official tox py37 stderr after verifier'
        if py37 and constraints and bound and resolution and metadata_verified:
            issues.append({'code': 'published_python_compatibility_metadata', 'path': metadata_path.name,
                           'sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                           'releases': metadata})
            failures.append(failure('R2', 'agentops_py37_injected_incompatible_constraints', 'tox_py37',
                                    location(log,text.count('\n',0,py37.start())+1), dependency='langchain/langchain-core',
                                    detail='Pinned releases require Python >=3.8.1,<4; required py37 resolution failed.', call=log['call_id']))
        elif py37:
            issues.append({'code': 'required_py37_failure_ownership_unknown'})
    return failures


def decide(snap):
    contract = snap['contract']
    public_contract = {k:v for k,v in contract.items() if k != 'sources'}
    issues, recoveries = [], []
    reports = snap['ctrf_reports']
    check = score_check(snap)
    result = {'schema_version': 2, 'policy': POLICY, 'reward': snap['reward'],
              'classification': 'Review', 'primary_rule': 'R6', 'reason_code': 'insufficient_evidence',
              'reason': 'Evidence does not establish all required evaluation stages and score.',
              'contract': public_contract, 'task_contract': public_contract,
              'issues': issues, 'required_stages': [], 'suite_results': [], 'nested_failures': [],
              'dependency_failures': [], 'recoveries': recoveries, 'score_check': check,
              'input_output_ownership': {}, 'ordering_status': 'within_stream_only',
              'evidence': snap['evidence'], 'evidence_files': snap['evidence'],
              'source_manifest': [{'path': path, 'sha256': digest}
                                  for path, digest in sorted(snap['source_checks'].items())],
              'verifier_listing': snap['verifier_listing'],
              'ctrf_reports': reports, 'missing_evidence': snap['missing_evidence'],
              'integrity_errors': snap['integrity_errors'], 'source_changed': snap['source_changed'],
              'truncated': snap['truncated'], 'processed': not bool(snap['integrity_errors'] or snap['source_changed']),
              'coverage_limitations': ['Runtime reference input bytes are not attested for historical attempts.',
                                       'Saved scoring evidence is checked; scientific computations are not rerun.']}

    def conclude(classification, rule, code, reason):
        result.update(classification=classification, primary_rule=rule, reason_code=code, reason=reason)
        return result

    if snap['reward'] < 0:
        return conclude('Review','R6','invalid_reward_range','Reward is finite and below zero.')
    if snap['integrity_errors'] or snap['source_changed'] or snap['truncated']:
        return conclude('Review','R6','source_changed' if snap['source_changed'] else 'evidence_read_failed',
                        'Evidence integrity or read completeness is not reliable.')
    if not contract['verified']:
        return conclude('Review','R6','frozen_verifier_contract_unknown','Frozen contract identity could not be verified.')
    calls = {r['call_id'] for r in reports} | {r['call_id'] for r in snap['logs']
                                             if r['path'].endswith(('test-stdout.txt','test-stderr.txt'))}
    if len(calls) > 1:
        issues.append({'code':'final_verifier_call_unknown','calls':sorted(calls)})
        return conclude('Review','R6','final_verifier_call_unknown','Multiple verifier calls lack a final-call binding.')
    failures = nested_failures(snap, issues)
    failures += install_failures(snap, recoveries, issues)
    failures += agentops_failures(snap, issues)
    from runtime.agents.verifier_stages import simpo_evidence, command_outcomes, mathjax_failure, agentops_dependency_failures
    explicit_failures, explicit_recoveries, stage_issues = command_outcomes(snap)
    recoveries.extend(explicit_recoveries)
    issues.extend(stage_issues)
    for command in explicit_failures:
        failures.append(failure('R2' if command.get('phase') != 'task_test' else 'R1',
            'required_command_failed', command.get('phase'),
            {'path':'diagnostics/verifier-execution.json','sequence':command['sequence']},
            call=command['verifier_call_id'],suite=command['suite_id'],
            detail='Trusted executor records a nonzero required command without same-stage recovery.'))
    protected = simpo_evidence(snap)
    from runtime.agents.verifier_stages import agentops_matrix_evidence
    matrix=agentops_matrix_evidence(snap)
    result['agentops_matrix']=matrix
    tox_failures=agentops_dependency_failures(snap)
    result['protected_environment_failures']=tox_failures
    for command in tox_failures:
        failures.append(failure('R2','agentops_required_environment_dependency_failed',
            'tox_'+command['environment'],{'path':'diagnostics/verifier-stages.json','sequence':command['sequence']},
            call=command['call_id'],suite=command['environment'],
            detail=command.get('stderr') or 'Frozen required tox dependency installation failed.'))
    mathjax=mathjax_failure(snap)
    if mathjax:
        failures.append(failure('R4',mathjax['reason'],'mathjax_initialization',
            {'path':'diagnostics/verifier-stages.json'},call=mathjax['call_id'],
            detail='Protected local MathJax/browser initialization did not complete.'))
    result['protected_stages'] = protected
    if protected and protected.get('verified') and protected['outcome'] == 'verifier_error':
        failures.append(failure('R4','simpo_reference_integrity_failed','independent_reference',
            {'path':'diagnostics/verifier-stages.json'},call=protected['call_id'],
            detail=protected.get('error','Protected SimPO reference is invalid.')))

    raw = snap['raw']
    if raw.get('completed') is False and raw.get('error'):
        failures.append(failure('R1','verifier_incomplete','verifier',{'path':'evaluation-raw.json','field':'completed/error'},
                                detail='Raw evaluator reports completed=false with execution error.'))
    result['nested_failures'] = failures
    result['dependency_failures'] = [f for f in failures if f['rule'] in {'R2','R3'}]
    result['required_stages'] = failures.copy()
    missing = set(contract['expected_reports'])-{r['path'].split('/')[-1] for r in reports}
    result['expected_reports'] = contract['expected_reports']
    result['missing_reports'] = sorted(missing)
    for report in reports:
        state = 'completed_valid' if report_completed(report) else 'not_observed'
        names = [t.get('name') for t in report['tests'] if isinstance(t,dict)]
        if len(set(names)) != len(names):
            issues.append({'code':'duplicate_test_identity','path':report['path']})
        result['suite_results'].append({'suite_id':report['suite_id'],'verifier_call':report['call_id'],
                                        'state':state,'summary':report['summary']})
    if failures and any(i['code'] == 'duplicate_test_identity' for i in issues):
        return conclude('Review','R6','duplicate_test_identity',
                        'Duplicate test records leave failure and retry ordering unresolved.')
    if failures:
        if any(i['code']=='ordering_unknown' for i in issues) and not any(f['stage'] != 'verifier_install' for f in failures):
            return conclude('Review','R6','ordering_unknown','Cross-stream evidence cannot establish final failure or recovery.')
        first = sorted(failures,key=lambda f: f['rule'])[0]
        return conclude('VerifierError', first['rule'], first['reason_code'],
                        first['detail'] or f"Required {first['stage']} aborted without same-stage recovery.")
    if protected is not None:
        if not protected.get('verified'):
            return conclude('Review','R6',protected['reason'],'Protected stage evidence does not establish this repaired verifier call.')
        if protected['outcome'] == 'agent_failure':
            error=protected.get('error','').splitlines()[0] if protected.get('error') else ''
            absent = error.removeprefix('Missing output: ') if error.startswith('Missing output: ') else (
                error.removeprefix('Missing environment info: ') if error.startswith('Missing environment info: ') else None)
            if absent and absent not in snap['documents'].get('subject/subject.json',{}).get('missing_outputs',[]):
                return conclude('Review','R6','missing_output_not_attested_in_subject',
                                'Runtime output absence is not established by the immutable pre-verifier submission.')
    if missing:
        issues.append({'code':'required_suite_missing','reports':sorted(missing)})
    # Identity-bound ownership/conflict adapters use subject archive facts, not
    # sample IDs or the previous audit's classification.
    ownership = snap.get('ownership', {})
    result['input_output_ownership'] = ownership
    issues.extend(snap.get('ownership_issues', []))
    if ownership.get('reference_submission_alias'):
        return conclude('Review','R6','reference_and_submission_path_alias_with_missing_agent_outputs',
                        'Reference and required submitted output resolve to the same missing path.')
    if ownership.get('instruction_verifier_syntax_conflict'):
        return conclude('Review','R6','pddl_prompt_example_conflicts_with_standard_grammar',
                        'Submitted functional actions follow the prompt example but conflict with the verifier grammar.')
    blocking = {'ordering_unknown','dependency_ownership_unknown','install_recovered_tests_incomplete',
                'recovery_environment_unknown',
                'required_stage_identity_incomplete','required_stage_order_unknown',
                'required_py37_failure_ownership_unknown','duplicate_test_identity','archive_ownership_unknown',
                'build_failure_ownership_unknown','reference_input_ownership_unknown','exception_ownership_unknown'}
    if any(i['code'] in blocking for i in issues):
        return conclude('Review','R6',next(i['code'] for i in issues if i['code'] in blocking),
                        'Required stage ownership, ordering, or recovery remains unresolved.')
    if missing or any(not report_completed(r) for r in reports):
        return conclude('Review','R6','required_suite_incomplete','A required suite is absent or has no completed cases.')
    if not check['complete'] or not check['matched']:
        return conclude('Review','R6',check.get('issue','score_inconsistent_or_unproven'),
                        'Frozen scoring evidence is incomplete or conflicts with the recorded reward.')
    if (not reports and contract['score_mode'] not in {'custom_groups','cif_fraction'}
            and not (contract['ctrf_expected'] is False and check.get('pytest_counts'))):
        return conclude('Review','R6','test_evidence_incomplete','No complete suite report establishes the evaluation.')
    # Count differences are retained; the content-bound score adapter decides
    # which source is authorized rather than replacing one denominator with another.
    if len(reports) == 1 and check.get('pytest_counts'):
        p = check['pytest_counts']
        n = sum(p.get(k,0) for k in ('passed','failed','error','skipped'))
        c = reports[0]['summary'].get('tests')
        if c is not None and c != n:
            issues.append({'code':'ctrf_pytest_count_difference','pytest':n,'ctrf':c,
                           'score_source':check['source'],'interpretation':'CTRF groups parametrized/phase records; pytest counts items.'})
    result['required_stages'].extend(result['suite_results'])
    reason = ownership.get('agent_build_failure') or 'frozen_contract_completed_score_verified'
    return conclude('ValidOutcome','R5',reason,'Required evaluation completed or followed its frozen agent-failure gate; scoring matches.')

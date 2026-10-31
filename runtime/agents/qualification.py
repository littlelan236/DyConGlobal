"""Qualify observed OpenHands request roles without editing old profiles."""
from pathlib import Path

from core.io import hash_json, identifier, read_json, sha256, write_json
from runtime.agents.openhands import validate_profile


def qualify_profile(root, run_id, profile_id, profile_ref=None):
    from application.execution.run_plan import load_plan
    from application.execution.evidence import RunEvidence
    from rollout.adapters.openhands import parse_response
    loaded = load_plan(root, run_id)
    cfg, run = loaded['config'], loaded['run']
    if (cfg.get('execution_context', {}).get('benchmark') != 'skillsbench'
            or cfg['execution_context']['agent'] != 'openhands'):
        raise ValueError('Qualification requires a frozen SkillsBench/OpenHands run')
    RunEvidence(root, run_id).verify()
    files = sorted((run / 'rollout/tasks').rglob('requests/req-*.json'))
    if not files:
        raise ValueError('No actual OpenHands model requests to qualify')
    tools, called, observed, requests = set(), set(), set(), []
    for file in files:
        row = read_json(file)
        if row.get('role') not in {'agent_decision','condensation'}:
            raise ValueError('Unqualified model request in probe: ' + file.name)
        if row['binding'].get('profile_sha256') != cfg['execution_context']['profile_sha256']:
            raise ValueError('Request belongs to another agent profile')
        request = row['request']
        requests.append({'path': file.relative_to(run).as_posix(), 'sha256': sha256(file),
                         'request_id': row['request_id'], 'step_id': row['step_id'],
                         'role':row['role'],'status':row.get('status')})
        if row.get('status')!='completed' or row['role']=='condensation':continue
        message = parse_response(row['selected_response'], request)
        tools.update(t['function']['name'] for t in request.get('tools') or [])
        called.update(call['id'] for call in message.get('tool_calls') or [])
        observed.update(m['tool_call_id'] for m in request['messages']
                        if m.get('role') == 'tool' and m.get('tool_call_id'))
    pairs = called & observed
    installs = list((run / 'rollout/tasks').rglob('dependency-install.json'))
    events = list((run / 'rollout/tasks').rglob('openhands/acp-events.json'))
    if not pairs or not events or not installs or any(read_json(f)['return_code'] for f in installs):
        raise ValueError('Qualification requires installed pinned runtime, ACP events, and executed tool/result pairs')
    proof = {'schema_version': 1, 'source_run_id': run_id,
        'execution_context_sha256': cfg['execution_context']['context_sha256'],
        'source_profile_sha256': cfg['execution_context']['profile_sha256'],
        'requests': sorted(requests, key=lambda r: (r['step_id'] if r['step_id'] is not None else -1, r['request_id'])),
        'tool_names': sorted(tools), 'executed_tool_call_ids': sorted(pairs),
        'acp_events': [{'path': f.relative_to(run).as_posix(), 'sha256': sha256(f)} for f in events],
        'supported_roles': sorted({r['role'] for r in requests}), 'unknown_roles': 'reject_before_capture_or_parallel'}
    proof['receipt_sha256'] = hash_json(proof)
    template = cfg['execution_context']['profile']
    if profile_ref is not None:
        template_path = (Path(root) / profile_ref).resolve()
        if not template_path.is_relative_to(Path(root).resolve()):
            raise ValueError('Runtime profile must be project-owned')
        template = validate_profile(read_json(template_path))
        template['dependency_snapshot_sha256'] = sha256(Path(template['dependency_snapshot']) / 'manifest.json')
        excluded = {'request_roles', 'qualification', 'use_host_proxy', 'condenser_max_tokens'}
        source = validate_profile(cfg['execution_context']['profile'])
        if template.get('dependency_snapshot_sha256')!=source.get('dependency_snapshot_sha256'):
            from runtime.agents.dependencies import verify_snapshot
            old=verify_snapshot(source['dependency_snapshot']);new=verify_snapshot(template['dependency_snapshot'])
            def core_files(manifest):
                return {r['path']:r['sha256'] for r in manifest['files']
                    if r['path'].startswith('wheels/') and not r['path'].startswith('wheels/openhands-')}
            allowed={'benchflow_only_skills_v1','stock_condenser_for_minimal_BenchFlow_settings_v1'}
            if (old['source_archive_sha256']!=new['source_archive_sha256']
                    or old['cli_revision']!=new['cli_revision'] or core_files(old)!=core_files(new)
                    or any(a['policy'] not in allowed for a in new.get('source_adaptations',[]))):
                raise ValueError('Qualification cannot cross unverified dependency or CLI changes')
            proof['dependency_compatibility']={'source_receipt':old['receipt_sha256'],
                'target_receipt':new['receipt_sha256'],'identical_non_cli_wheels_sha256':hash_json(core_files(new)),
                'declared_cli_adaptations':sorted({a['policy'] for a in new.get('source_adaptations',[])})}
            excluded.update({'dependency_snapshot','dependency_snapshot_sha256'})
        if any(template.get(k) != source.get(k) for k in set(template) | set(source) if k not in excluded):
            raise ValueError('Qualification cannot transfer across agent, dependency, skills or tool profiles')
        proof['runtime_profile_ref'] = str(template_path.relative_to(root))
        proof['runtime_profile_sha256'] = sha256(template_path)
        proof['receipt_sha256'] = hash_json({k:v for k,v in proof.items() if k!='receipt_sha256'})
    profile = {**template, 'request_roles': 'openhands-1.22.1',
               'qualification': proof}
    if profile.get('dependency_snapshot'):
        profile['dependency_snapshot_sha256'] = sha256(Path(profile['dependency_snapshot']) / 'manifest.json')
    profile = validate_profile(profile)
    target = Path(root) / 'runtime/agent-profiles' / identifier(profile_id) / 'manifest.json'
    if target.exists() and read_json(target) != profile:
        raise FileExistsError('Qualification profile is immutable')
    if not target.exists():
        write_json(target, profile)
    return {'profile_ref': str(target.relative_to(root)), 'profile_sha256': sha256(target),
            'qualification_sha256': proof['receipt_sha256'], 'tool_result_pairs': len(pairs),
            'qualified_requests': len(files)}

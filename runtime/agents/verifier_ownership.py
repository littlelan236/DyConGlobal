"""Content-bound path/build responsibility adapters shared by both entries."""
from __future__ import annotations

import json
import re
from pathlib import Path

from application.execution.subject import read_archive_members


def enrich_ownership(snap):
    contract = snap['contract']
    caps = contract['capabilities']
    result, issues = {}, []
    snap.update(ownership=result, ownership_issues=issues)
    subject = snap['documents'].get('subject/subject.json', {})
    raw_text = '\n'.join(log['text'] for log in snap['logs'] if log['path'].endswith('test-stdout.txt'))
    runtime = '\n'.join(text for log in snap['logs'] for _, text, _ in log['events'])
    sources = '\n'.join(contract['sources'].values())
    if 'simpo_paths' in caps and re.search(r'AssertionError: Missing ground truth: /root/loss\.npz', runtime):
        cwd = re.search(r'^rootdir: (/[^\n]+)', raw_text, re.M)
        if (cwd and cwd.group(1).strip() == '/root'
                and re.search(r'os\.getcwd\(\), [\x27\"]loss\.npz[\x27\"]', sources)
                and re.search(r'out_path = [\x27\"]/root/loss\.npz[\x27\"]', sources)):
            result.update(cwd='/root', reference_path='/root/loss.npz', submission_path='/root/loss.npz',
                          reference_submission_alias=True,
                          agent_missing_outputs=subject.get('missing_outputs',[]))
            issues.extend([{'code':'reference_submission_path_alias'},{'code':'agent_outputs_missing',
                           'paths':subject.get('missing_outputs',[])}])
        else:
            issues.append({'code':'archive_ownership_unknown','detail':'cwd_unknown'})
    pddl_failed = ('pddl_syntax' in caps and ('PDDLReader' in raw_text or 'ParseException' in runtime))
    maven_failed = ('maven_submission' in caps and ('JavacTool' in runtime or 'maven-shade-plugin:3' in runtime))
    if not (pddl_failed or maven_failed):
        return
    rows = []
    attempt = Path(snap['attempt_path'])
    try:
        for output in subject.get('outputs',[]):
            if output.get('kind') != 'directory_archive':
                continue
            rel = Path(output['path'])
            path = attempt/'subject'/rel
            if rel.is_absolute() or '..' in rel.parts or not path.resolve().is_relative_to(attempt):
                raise ValueError('unsafe subject output')
            def wanted(name):
                p = Path(name)
                if pddl_failed:
                    return p.name == 'problem.json' or bool(re.fullmatch(r'task\d+\.txt', p.name))
                return p.name == 'pom.xml' or bool(re.fullmatch(r'patch_\d+\.diff',p.name))
            members = read_archive_members(path,output['sha256'],wanted)
            snap['source_checks'][str(path)] = output['sha256']
            snap['evidence'].append({'path':path.relative_to(attempt).as_posix(),'sha256':output['sha256'],
                                     'evidence_role':'submitted_archive'})
            for member in members:
                text = member.pop('data').decode('utf-8')
                rows.append((member['name'],text))
                snap['evidence'].append({'path':path.relative_to(attempt).as_posix(),
                                         **member,'evidence_role':'submitted_archive_member'})
        result['archive_members'] = [n for n,_ in rows]
        if pddl_failed:
            problem = next((json.loads(t) for n,t in rows if n.endswith('/problem.json')), None)
            required_plans = [r.get('plan_output') for r in problem] if isinstance(problem,list) else []
            plans = [(n,t) for n,t in rows if Path(n).name in required_plans]
            functional = (len(plans) == len(required_plans) == 2 and
                          all(re.search(r'\b\w+\([^()\n]*\)',text) for _,text in plans))
            instruction = contract['sources'].get('task.md','')
            if rows and functional and re.search(r'\b\w+\([^()\n]*\)',instruction) and 'PDDL' in instruction:
                result['instruction_verifier_syntax_conflict'] = True
            else:
                issues.append({'code':'archive_ownership_unknown','detail':'PDDL submission/grammar evidence missing'})
        if maven_failed:
            poms = [(n,t) for n,t in rows if n.endswith('pom.xml')]
            patches = [(n,t) for n,t in rows if re.search(r'patch_\d+\.diff$',n)]
            bad_plugin = any(re.search(r'<artifactId>maven-shade-plugin</artifactId>\s*<version>3</version>',t) for _,t in poms)
            exclusions = [(n,t) for n,t in poms if re.search(r'<exclusion>\s*<groupId>com\.sun</groupId>\s*<artifactId>tools</artifactId>',t)]
            patch_excludes = any(re.search(r'^\+.*<groupId>com\.sun</groupId>',t,re.M) and
                                 re.search(r'^\+.*<artifactId>tools</artifactId>',t,re.M) for _,t in patches)
            if bad_plugin and re.search(r'maven-shade-plugin:3(?::|\b)',runtime):
                result['agent_build_failure'] = 'google_auto_invalid_maven_plugin_version'
            elif len(exclusions)>=4 and patch_excludes and 'JavacTool' in runtime and '1.7.0' in raw_text:
                # Direct tools dependencies would change the classpath proof.
                direct_tools = any(re.search(r'<dependency>\s*<groupId>com\.sun</groupId>\s*<artifactId>tools</artifactId>',t) for _,t in poms)
                if not direct_tools:
                    result['agent_build_failure'] = 'google_auto_submitted_exclusion_removes_compiler_dependency'
            if not result.get('agent_build_failure'):
                issues.append({'code':'build_failure_ownership_unknown'})
    except (OSError, ValueError, UnicodeError, KeyError, json.JSONDecodeError) as exc:
        issues.append({'code':'archive_ownership_unknown','detail':str(exc)})

"""Shared OpenHands runtime/tool probe with explicit sandbox identity."""
from __future__ import annotations

import json
import os
import select
import subprocess
import tempfile
import time
from pathlib import Path


def probe(workspace, model_seconds, expected_user="agent", expected_home="/home/agent"):
    from openhands_cli.stores.agent_store import AgentStore
    from openhands.tools.terminal import TerminalAction, TerminalExecutor

    import pwd
    expected_uid = pwd.getpwnam(expected_user).pw_uid
    if os.getuid() != expected_uid or (os.getuid() == 0 and expected_user != 'root'):
        raise ValueError('OpenHands sandbox probe identity differs from expected agent')
    if os.environ.get('HOME') != expected_home or str(Path.home()) != expected_home:
        raise ValueError('OpenHands sandbox probe HOME differs from expected agent')
    os.chdir(workspace)
    os.environ.update(LLM_MODEL='openai/dycon-environment-probe', LLM_API_KEY='probe',
        LLM_BASE_URL='http://127.0.0.1:9/v1', DYCON_OPENHANDS_TERMINAL='subprocess',
        DYCON_OPENHANDS_MODEL_SECONDS=str(model_seconds), LITELLM_LOCAL_MODEL_COST_MAP='true',
        OPENHANDS_SUPPRESS_BANNER='1')
    agent = AgentStore().load_or_create(env_overrides_enabled=True, critic_disabled=True)
    for name, field in (("OPENHANDS_CONDENSER_MAX_SIZE", "max_size"),
                        ("OPENHANDS_CONDENSER_KEEP_FIRST", "keep_first")):
        if name in os.environ and getattr(agent.condenser, field) != int(os.environ[name]):
            raise ValueError("Published CLI did not apply " + name)
    terminal = next(t for t in agent.tools if t.name == 'terminal')
    if terminal.params.get('terminal_type') != 'subprocess' or agent.llm.timeout != model_seconds:
        raise ValueError('Published CLI does not consume the frozen terminal/SDK timeout contract')
    executor = TerminalExecutor(working_dir=workspace, terminal_type='subprocess')
    try:
        command = ('printf dycon_terminal_ok; id -u; pwd; '
                   'command -v python || true; command -v python3 || true; '
                   'if command -v python >/dev/null; then python -c \"import sys; print(sys.executable)\"; '
                   'elif command -v python3 >/dev/null; then python3 -c \"import sys; print(sys.executable)\"; fi')
        observation = executor.session.execute(TerminalAction(command=command, timeout=5))
        if 'dycon_terminal_ok' not in str(observation):
            raise ValueError('Native OpenHands subprocess tool did not execute')
    finally:
        executor.close()
    receipts = []
    # Match BenchFlow's ACP launch: its shell seeds settings before OpenHands
    # performs the session auth check, which precedes env override application.
    settings = Path.home() / '.openhands/agent_settings.json'
    original_settings = settings.read_bytes() if settings.exists() else None
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({'llm': {'model': os.environ['LLM_MODEL'],
        'api_key': os.environ['LLM_API_KEY'], 'base_url': os.environ['LLM_BASE_URL']}}))
    with tempfile.TemporaryFile(mode='w+') as errors:
        process = subprocess.Popen(['/opt/dycon-openhands/bin/openhands', 'acp', '--always-approve', '--override-with-envs'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, text=True, bufsize=1)
        try:
            for index, method, params in (
                    (1, 'initialize', {'protocolVersion': 1, 'clientCapabilities': {}}),
                    (2, 'session/new', {'cwd': workspace, 'mcpServers': []})):
                process.stdin.write(json.dumps({'jsonrpc': '2.0', 'id': index, 'method': method, 'params': params}) + '\n')
                process.stdin.flush()
                deadline = time.monotonic() + 40
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
                        raise TimeoutError('ACP environment probe timed out: ' + method)
                    line = process.stdout.readline()
                    if not line:
                        errors.seek(0)
                        raise RuntimeError('ACP exited before ' + method + ': ' + errors.read()[-4000:])
                    value = json.loads(line)
                    if value.get('id') == index:
                        if 'error' in value:
                            raise RuntimeError('ACP environment initialization failed: ' + str(value['error']))
                        receipts.append({'method': method, 'response': value})
                        break
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if original_settings is None:
                settings.unlink(missing_ok=True)
            else:
                settings.write_bytes(original_settings)
    return {'schema_version': 1, 'uid': os.getuid(), 'gid': os.getgid(), 'home': os.environ['HOME'],
            'cwd': os.getcwd(), 'terminal_backend': terminal.params['terminal_type'],
            'model_timeout_seconds': agent.llm.timeout, 'sdk_retries': agent.llm.num_retries,
            'condenser': {'max_size': agent.condenser.max_size,
                          'keep_first': agent.condenser.keep_first},
            'real_tool_executed': True, 'terminal_identity_output': str(observation),
            'path': os.environ.get('PATH'), 'acp': receipts, 'model_requests': 0, 'verified': True}


if __name__ == '__main__':
    import sys
    print(json.dumps(probe(sys.argv[1], int(sys.argv[2]), *sys.argv[3:])))

"""Minimal eval instrumentation; preserve official command and cleanup flow."""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import uuid
from pathlib import Path

START = ": '>>>>> Start Test Output'"
END = ": '>>>>> End Test Output'"


def get_test_command(script):
    lines = script.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.strip() == START]
    ends = [i for i, line in enumerate(lines) if line.strip() == END]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        return None
    command = "".join(lines[starts[0] + 1:ends[0]]).strip()
    if not command:
        return None
    # Parse only simple shell assignments; never execute shell syntax.
    try:
        words = shlex.split(command.splitlines()[0])
    except ValueError:
        words = []
    while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0], re.S):
        words.pop(0)
    executable = " ".join(words)
    runner = ("pytest" if re.match(r"(?:python\S* -m )?pytest\b", executable) else
              "tox" if re.match(r"tox\b", executable) else
              "sympy" if executable.startswith(("bin/test ", "python bin/test ")) else "unknown")
    scope = "command" if len(command.splitlines()) == 1 else "region"
    return {"command": command, "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
            "runner": runner, "scope": scope, "end_line": ends[0]}


def get_pytest_binding(script):
    rows = re.findall(r"(?m)^# __DYCON_PYTEST_BINDING__ (.+)$", script)
    return json.loads(rows[0]) if len(rows) == 1 else None


def original_script(script):
    # Only remove our delimited wrapper, never lines of the official command.
    script = re.sub(r"(?ms)^# __DYCON_INSTRUMENTATION_BEGIN__\n.*?^# __DYCON_INSTRUMENTATION_END__\n", "", script)
    return "".join(line for line in script.splitlines(keepends=True)
                   if not line.startswith(('__dycon_test_rc=', "printf '\\n__DYCON_TEST_RESULT__")))


def instrument(script, *, stack_seconds=280):
    details = get_test_command(script)
    if details is None:
        return script, None
    payload = json.dumps({k: details[k] for k in ("command_sha256", "runner", "scope")}, separators=(",", ":"))
    prefix = payload[:-1] + ',"returncode":'
    lines = script.splitlines(keepends=True)
    after = '__dycon_test_rc=$?\n' + "printf '\\n__DYCON_TEST_RESULT__ %s%s}\\n' " + shlex.quote(prefix) + ' "$__dycon_test_rc"\n'
    before = ""
    if details["runner"] in {"pytest", "tox"}:
        code = Path(__file__).with_name("swebench_pytest_plugin.py").read_text()
        token = uuid.uuid4().hex
        binding = {"token": token, "command_sha256": details["command_sha256"],
                   "plugin_sha256": hashlib.sha256(code.encode()).hexdigest()}
        folder = "/tmp/dycon-pytest-" + token
        module = "dycon_receipt_" + token
        source = "BINDING=" + repr(binding) + "\nRECEIPT_DIR=" + repr(folder) + "\n" + code
        startup_code = Path(__file__).with_name("swebench_startup_probe.py").read_text()
        startup = ("BINDING=" + repr(binding) + "\nRECEIPT_DIR=" + repr(folder)
                   + "\nSTACK_SECONDS=" + repr(max(1, int(stack_seconds))) + "\n" + startup_code)
        before = ("# __DYCON_INSTRUMENTATION_BEGIN__\n# __DYCON_PYTEST_BINDING__ " + json.dumps(binding) + "\n"
                  + "mkdir -m 700 " + shlex.quote(folder) + "\n"
                  + "cat > " + folder + "/" + module + ".py <<'DYCON_PLUGIN_EOF'\n" + source + "\nDYCON_PLUGIN_EOF\n"
                  + "cat > " + folder + "/sitecustomize.py <<'DYCON_STARTUP_EOF'\n" + startup + "\nDYCON_STARTUP_EOF\n"
                  + "export PYTHONPATH=" + folder + '${PYTHONPATH:+:$PYTHONPATH}\n'
                  + "export PYTEST_PLUGINS=" + module + '${PYTEST_PLUGINS:+,$PYTEST_PLUGINS}\n'
                  + 'export TOX_TESTENV_PASSENV="${TOX_TESTENV_PASSENV:+$TOX_TESTENV_PASSENV }PYTHONPATH PYTEST_PLUGINS"\n'
                  + 'export TOX_PASSENV="PYTHONPATH,PYTEST_PLUGINS"\n'
                  + "# __DYCON_INSTRUMENTATION_END__\n")
        # Session bodies stay out of merged stdout/stderr. The host collects
        # the directory before cleanup, including incomplete diagnostics.
        after += ("# __DYCON_INSTRUMENTATION_BEGIN__\n"
                  + "printf '%s%s}\\n' " + shlex.quote(prefix)
                  + ' "$__dycon_test_rc" > ' + folder + "/command.tmp\n"
                  + "mv " + folder + "/command.tmp " + folder + "/command.json\n"
                  + "# __DYCON_INSTRUMENTATION_END__\n")
    lines.insert(details["end_line"], after)
    if before:
        start = next(i for i, line in enumerate(lines) if line.strip() == START)
        lines.insert(start, before)
    return "".join(lines), details

"""Minimal eval instrumentation; preserve official command and cleanup flow."""
from __future__ import annotations

import hashlib
import json
import re
import shlex

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
    runner = ("pytest" if re.match(r"(?:python\S* -m )?pytest\b", command) else
              "tox" if re.match(r"tox\b", command) else
              "sympy" if command.startswith(("bin/test ", "python bin/test ")) else "unknown")
    scope = "command" if len(command.splitlines()) == 1 else "region"
    return {"command": command, "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
            "runner": runner, "scope": scope, "end_line": ends[0]}


def instrument(script):
    details = get_test_command(script)
    if details is None:
        return script, None
    payload = json.dumps({k: details[k] for k in ("command_sha256", "runner", "scope")}, separators=(",", ":"))
    # printf arguments are quoted independently; no interpolation of test source.
    prefix = payload[:-1] + ',"returncode":'
    lines = script.splitlines(keepends=True)
    lines.insert(details["end_line"],
                 '__dycon_test_rc=$?\n'
                 + "printf '\\n__DYCON_TEST_RESULT__ %s%s}\\n' "
                 + shlex.quote(prefix) + ' "$__dycon_test_rc"\n')
    return "".join(lines), details

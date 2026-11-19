"""Evidence-based Harbor verifier health; never replace the upstream raw reward."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

POLICY = "harbor-verifier-health-v2"
MAX_BYTES = 8 * 1024 * 1024


def audit_verifier(directory, attempt_id, *, operation=None):
    """Inspect exactly one trial, without importing Harbor or executing its files.

    Oracle failures invalidate a reference control, not ordinary model submissions.
    Completed assertions establish verifier execution health, not a passing
    Oracle control. trial_outcome separately rejects sub-one Oracle rewards.
    Missing or ambiguous health evidence is Review.
    """
    from core.io import identifier

    root = Path(directory).resolve()
    trial = root / "trials" / identifier(attempt_id)
    sources = []
    incomplete = []

    def read(relative):
        path = root / relative
        if not path.exists():
            return None
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            incomplete.append(str(relative))
            return None
        try:
            with path.open("rb") as stream:
                data = stream.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                incomplete.append(str(relative))
                return None
        except OSError:
            incomplete.append(str(relative))
            return None
        sources.append({"path": str(relative), "sha256": hashlib.sha256(data).hexdigest()})
        return data.decode("utf-8", "replace")

    prefix = trial.relative_to(root)
    log = read(prefix / "verifier/test-stdout.txt")
    exit_text = read(prefix / "agent/exit-code.txt") if operation == "oracle" else None
    exec_text = read(Path("verifier-execution.json"))

    def decision(classification, reason, evidence=None):
        return {"schema_version": 1, "policy": POLICY, "classification": classification,
                "reason_code": reason, "evidence": evidence, "sources": sources,
                "incomplete_sources": incomplete}

    if incomplete:
        return decision("Review", "unreadable_or_oversized_evidence")
    if exit_text is not None:
        try:
            code = int(exit_text.strip())
        except ValueError:
            return decision("Review", "invalid_oracle_exit_receipt")
        if code:
            return decision("VerifierError", "oracle_reference_failed", {"return_code": code})

    execution = None
    if exec_text is not None:
        try:
            execution = json.loads(exec_text)
        except ValueError:
            return decision("Review", "invalid_verifier_execution_receipt")
        if (not isinstance(execution, dict) or execution.get("attempt_id") != attempt_id
                or execution.get("schema_version") != 1):
            return decision("Review", "unbound_verifier_execution_receipt")
        if execution.get("exception_type"):
            return decision("VerifierError", "verifier_execution_exception",
                            {"exception_type": execution["exception_type"]})
        if type(execution.get("return_code")) is not int:
            return decision("Review", "invalid_verifier_exit_code")

    if log is None:
        return decision("Review", "missing_verifier_log")

    # Restrict signals to the final shell/runner failure. Earlier retry warnings
    # or strings printed by assertion diffs do not establish bootstrap failure.
    lines = log.splitlines()
    last = next((line.strip() for line in reversed(lines) if line.strip()), "")
    terminal_patterns = (
        ("verifier_tool_missing", r"^/tests/test\.sh: (?:line \d+: )?.*\b(?:uvx?|curl|wget|pytest|python[\d.]*)\b: (?:command )?not found$"),
        ("verifier_dependency_install_failed", r"^(?:E: Unable to locate package|E: Failed to fetch|ERROR: No matching distribution found|ERROR: Could not find a version|ERROR: Failed to install|error: (?:Failed to|No solution found)).*"),
        ("verifier_download_failed", r"^(?:curl: \(\d+\)|wget:|Caused by:|error:).*(?:timed out|certificate|SSL|TLS|resolve host|Failed to download|failed to download|connection|network).*"),
    )
    for reason, pattern in terminal_patterns:
        if re.search(pattern, last, re.I):
            return decision("VerifierError", reason, {"line": len(lines), "text": last[:600]})

    # pytest's final session summary provides execution/collection evidence.
    summaries = [line for line in lines if re.fullmatch(r"=+ .* in [\d.]+s(?: \([^)]*\))? =+", line.strip())]
    if summaries:
        summary = summaries[-1]
        if re.search(r"\b\d+ errors?\b|no tests ran", summary):
            return decision("VerifierError", "verifier_test_setup_or_collection_failed", {"summary": summary[:600]})
        if re.search(r"\b[1-9]\d* (?:passed|failed)\b", summary):
            if execution and execution["return_code"] not in (0, 1):
                return decision("VerifierError", "verifier_script_failed", {"return_code": execution["return_code"]})
            return decision("ValidOutcome", "tests_completed", {"summary": summary[:600]})
    if execution and execution["return_code"]:
        return decision("VerifierError", "verifier_script_failed", {"return_code": execution["return_code"]})
    return decision("Review", "insufficient_verifier_completion_evidence")

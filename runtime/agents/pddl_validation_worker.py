"""Isolated trusted-code worker for the SkillsBench PDDL semantic recheck CLI.

Only reads JSON, PDDL and plan text. It never imports submitted code or solves plans.
"""

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path


def evaluate(request):
    sys.dont_write_bytecode = True
    if importlib.metadata.version("unified-planning") != "1.3.0":
        raise ValueError("Revalidation requires unified_planning==1.3.0")
    spec = importlib.util.spec_from_file_location("trusted_pddl_verifier", request["verifier"])
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    rows = []
    for task in request["tasks"]:
        try:
            result = verifier.evaluate_plan(task["domain"], task["problem"], task["plan"])
        except verifier.PlanFormatError as exc:
            result = {"valid": False, "status": "INVALID_SYNTAX", "reason": str(exc)}
        normalized = None
        try:
            normalized = verifier.check_plan_format(task["plan"])
        except ValueError:
            pass
        rows.append({"id": task["id"], **result,
                     "normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest()
                     if normalized is not None else None})
    return {"plans": rows, "environment": {
        "python": platform.python_version(), "executable": sys.executable,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "unified_planning": "1.3.0", "execution": "host-isolated-python-semantic-validation",
        "container_replay": False, "oracle_or_planner_executed": False}}


if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result = evaluate(request)
    Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")

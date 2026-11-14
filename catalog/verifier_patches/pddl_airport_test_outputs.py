"""Airport plans: accept the documented notation and validate PDDL semantics.

Project correction pddl-airport-plan-v1 for unified_planning 1.3.0.
This file is copied into a new immutable task variant, never into old runs.
"""

import json
import re
from pathlib import Path

from unified_planning.engines.results import ValidationResultStatus
from unified_planning.exceptions import UPException
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import PlanValidator

PROBLEM_FILE = "/app/problem.json"
_NAME = r"[A-Za-z][A-Za-z0-9_-]*"
_FUNCTION = re.compile(rf"({_NAME})\(\s*({_NAME}(?:\s*,\s*{_NAME})*)?\s*\)")
_STANDARD = re.compile(rf"\(\s*({_NAME})(?:\s+{_NAME})*\s*\)")


class PlanFormatError(ValueError):
    """A submitted action line is outside the two documented grammars."""


def normalize_plan_text(text):
    """Normalize complete action lines only; never evaluate submission code."""
    actions = []
    for number, raw in enumerate(text.splitlines(), 1):
        # PDDL comments and separator lines carry no actions.
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        match = _FUNCTION.fullmatch(line)
        if match:
            name, arguments = match.groups()
            args = [] if arguments is None else [arg.strip() for arg in arguments.split(",")]
            actions.append("(" + " ".join([name, *args]) + ")")
        elif _STANDARD.fullmatch(line):
            actions.append(line)
        else:
            raise PlanFormatError(f"Invalid action syntax at line {number}: {raw}")
    # Empty plans are parsed too: only the validator decides whether init meets goal.
    return "\n".join(actions) + ("\n" if actions else "")


def check_plan_format(plan_file):
    return normalize_plan_text(Path(plan_file).read_text(encoding="utf-8"))


def evaluate_plan(domain_file, problem_file, plan_file):
    reader = PDDLReader()
    problem = reader.parse_problem(str(domain_file), str(problem_file))
    normalized = check_plan_format(plan_file)
    try:
        pred_plan = reader.parse_plan_string(problem, normalized)
    except (UPException, ValueError, AssertionError) as exc:
        return {"valid": False, "status": "INVALID_SYNTAX", "reason": str(exc),
                "inapplicable_action": None, "action_count": None,
                "validator_result": f"{type(exc).__name__}: {exc}"}
    with PlanValidator(problem_kind=problem.kind, plan_kind=pred_plan.kind) as validator:
        result = validator.validate(problem, pred_plan)
    if result.status not in (ValidationResultStatus.VALID, ValidationResultStatus.INVALID):
        raise RuntimeError(f"Validator returned no definitive result: {result}")
    return {
        "valid": result.status == ValidationResultStatus.VALID,
        "status": result.status.name,
        "reason": str(result.reason) if result.reason is not None else None,
        "inapplicable_action": str(result.inapplicable_action)
        if result.inapplicable_action is not None else None,
        "action_count": len(pred_plan.actions),
        "validator_result": str(result),
    }


def validate_plan(domain_file, problem_file, plan_file):
    result = evaluate_plan(domain_file, problem_file, plan_file)
    print(json.dumps(result, ensure_ascii=False))
    return result["valid"]


def load_problem():
    return json.loads(Path(PROBLEM_FILE).read_text(encoding="utf-8"))


class TestOutputFilesExist:
    def test_all_output_files_exist(self):
        for task in load_problem():
            assert Path(task["plan_output"]).is_file(), task["plan_output"]


class TestPlanCorrectness:
    def test_all_plans_valid(self):
        # Check all problems even if one submitted plan is invalid.
        failures = []
        for task in load_problem():
            try:
                if not validate_plan(task["domain"], task["problem"], task["plan_output"]):
                    failures.append(f"Invalid plan: {task['id']}")
            except (ValueError, KeyError) as exc:
                failures.append(f"Invalid plan: {task['id']}: {exc}")
        assert not failures, "\n".join(failures)

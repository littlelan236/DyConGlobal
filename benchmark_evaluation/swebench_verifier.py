"""Pure, versioned SWE-bench validity audit. Never rewrites official scores.

The collector owns I/O and invokes the installed official parser. This module
only consumes frozen evidence. Attribution supplied by a reviewer must be bound
to the exact evidence digest; log keywords alone never prove environment fault.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

from benchmark_evaluation.swebench_settlement import SCORED_CLASSIFICATIONS, review_policy

POLICY_VERSION = "swebench_verifier_audit_v5"
GROUPS = ("FAIL_TO_PASS", "PASS_TO_PASS")
STATES = {"PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS"}
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def evidence_digest(document):
    """Labels and collector errors are not part of the underlying evidence."""
    return digest({k: v for k, v in document.items() if k not in {"findings", "collection_errors"}})


def _ids(task, group):
    value = task.get(group, [])
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ValueError(f"Invalid {group} list")
    if len(value) != len(set(value)):
        raise ValueError(f"Duplicate {group} IDs")
    return value


def test_region(text):
    """Accept actual output/xtrace markers, not markers inside patches or prose."""
    text = _ANSI.sub("", text)
    def markers(label):
        pattern = r"(?m)^(?:\+\s*)?:?\s*['\"]?>>>>> " + label + r" Test Output(?:['\"])?\s*$"
        return list(re.finditer(pattern, text))
    starts, ends = markers("Start"), markers("End")
    if len(starts) != 1 or len(ends) != 1 or starts[0].end() >= ends[0].start():
        return "", False
    return text[starts[0].end():ends[0].start()], True


def failure_blocks(text):
    """Pytest traceback blocks; matching below rejects ambiguous short names."""
    matches = list(re.finditer(r"(?m)^_{3,}\s+(.+?)\s+_{3,}\s*$", text))
    blocks = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end():end].split("short test summary info")[0]
        blocks.append({"header": match.group(1), "body": body})
    return blocks


def _block_for(test_id, blocks, required_ids):
    if "::" not in test_id:
        # SymPy's official parser uses bare function names. Keep the full source
        # path in evidence and reject multiple blocks with that function name.
        matches = [b for b in blocks if b["header"].endswith(":" + test_id)]
        return matches[0] if len(matches) == 1 else None
    short = ".".join(test_id.split("::")[1:])
    if not short or sum(".".join(x.split("::")[1:]) == short for x in required_ids) != 1:
        return None
    matches = [b for b in blocks if re.sub(r"^ERROR at (?:setup|teardown) of ", "", b["header"])
               in {short, test_id}]
    return matches[0] if len(matches) == 1 else None


def _candidate_errors(body):
    # Only terminal pytest exception lines, never source, patch, or expected raises.
    terminal = "\n".join(re.findall(r"(?m)^E\s+(.+)$", body))
    terminal += "\n" + "\n".join(re.findall(r"(?m)^((?:RecursionError|DeprecationWarning):[^\n]*)$", body))
    tests = [
        ("test_network_dns_failure", r"Temporary failure in name resolution|Name or service not known|NameResolutionError"),
        ("test_network_unreachable", r"Network is unreachable|No route to host"),
        ("test_network_connection_reset", r"Connection reset by peer|RemoteDisconnected"),
        ("test_tls_trust_failure", r"CERTIFICATE_VERIFY_FAILED|unable to get local issuer certificate"),
        ("test_resource_http_error", r"HTTP Error (?:403|404|5\d\d)|403 Client Error"),
        ("missing_verifier_dependency", r"ImportError:.*(?:libGL|libXrender|libEGL).*cannot open shared object"),
        ("runtime_dependency_incompatible", r"AttributeError:.*(?:np\.Inf|`np\.Inf`|numpy.*(?:Inf|float|complex))"),
        ("python_runtime_incompatible", r"RecursionError:"),
        ("runtime_warning_as_error", r"DeprecationWarning:"),
        ("dependency_api_incompatible", r"(?:AttributeError|TypeError):.*(?:vtk|PointData|DataSetMapper)"),
        ("test_framework_compatibility", r"fixture .* not found|recursive dependency involving fixture|TypeError:.*expected exception"),
    ]
    return [code for code, pattern in tests if re.search(pattern, terminal, re.I)]


def _receipt(text, expected):
    rows = re.findall(r"(?m)^__DYCON_TEST_RESULT__ (\{[^\n]+\})\s*$", text)
    if len(rows) != 1:
        return None
    try:
        value = json.loads(rows[0])
    except ValueError:
        return None
    if value.get("command_sha256") != expected or type(value.get("returncode")) is not int:
        return None
    return value


def _inner_pytest(text, execution, statuses):
    """Accept one completed top-level session, bound to the actual host output.

    Multiple/nested sessions are ambiguous. Do not choose the one that passed.
    """
    binding = execution.get("pytest_binding")
    sidecar = execution.get("receipt_protocol") == 2
    rows = (execution.get("pytest_sessions", []) if sidecar else
            re.findall(r"(?m)^__DYCON_PYTEST_SESSION__ (.+)$", text))
    if not binding or len(rows) != 1:
        return None
    if not (execution.get("output_verified") is True or execution.get("output_sha256") == hashlib.sha256(text.encode()).hexdigest()):
        return None
    try:
        value = dict(rows[0]) if sidecar else json.loads(rows[0])
        if (value.get("schema_version") != 1 or value.get("binding") != binding
                or binding.get("command_sha256") != execution.get("test_command_sha256")
                or value.get("exitstatus") not in (0, 1) or value.get("collection_errors")
                or not value.get("executable") or not value.get("argv")
                or not value.get("pid") or not value.get("ancestors")):
            return None
        reports = value["reports"]
        collected = value["collected"]
        if len(collected) != len(set(collected)):
            return None
        by_test = {}
        for row in reports:
            phases = by_test.setdefault(row["nodeid"], {})
            if row["when"] in phases:
                return None
            phases[row["when"]] = row
        if set(by_test) != set(collected):
            return None
        any_failed = False
        for test, phases in by_test.items():
            if set(phases) != {"setup", "call", "teardown"}:
                return None
            if phases["setup"]["outcome"] != "passed" or phases["teardown"]["outcome"] != "passed":
                return None
            state = phases["call"]["outcome"]
            if state not in {"passed", "failed"}:
                return None
            if test in statuses and statuses[test] != state.upper():
                return None
            any_failed |= state == "failed"
        if bool(value["exitstatus"]) != any_failed:
            return None
        value["tests"] = by_test
        return value
    except (ValueError, KeyError, TypeError):
        return None


def _pytest_progress_token(task, test_id):
    # The official pytest parser takes the second whitespace token of every
    # status line, including output of intentionally failing nested pytest runs.
    # These three forms cannot name top-level pytest tests. Do not generalize
    # this to arbitrary malformed IDs, or to other repository parsers.
    return task.get("repo") == "pytest-dev/pytest" and bool(
        re.fullmatch(r"\[|\[\d+%\](?:-+)?", test_id))


def _gold_failure_reason(document, result, region):
    """Describe observations, without promoting candidates to causal facts."""
    outcome = document.get("attempt_result", {})
    detail = "\n".join(str(outcome.get(k, "")) for k in ("error", "failure_detail"))
    if ("failed to extract layer" in detail and "Lchown" in detail
            and ("subuid" in detail or "subgid" in detail)):
        return "container_image_ownership_unmapped", "preflight", True
    if document.get("official_result", {}).get("completed") is not True:
        return ("gold_evaluation_timeout" if outcome.get("failure_type") == "timeout"
                or document.get("execution", {}).get("timed_out") else "gold_evaluation_incomplete"), "test", False
    receipt = result["execution"].get("test_receipt") or {}
    if receipt.get("returncode") == 139 or re.search(r"/eval\.sh: line \d+:\s+\d+ Segmentation fault", region):
        return "gold_native_process_crash", "test", False
    if re.search(r"(?m)^_+ ERROR collecting .+ _+$", region):
        return "gold_test_collection_failed", "test", False
    if result.get("parameter_identity_mismatches"):
        return "required_test_parameter_identity_mismatch", "parse", True
    candidates = result["candidate_reason_codes"]
    for candidate, reason in (("runtime_warning_as_error", "gold_runtime_warning_failure"),
                              ("python_runtime_incompatible", "gold_runtime_recursion_failure"),
                              ("dependency_api_incompatible", "gold_dependency_api_failure")):
        if candidate in candidates:
            return reason, "test", False
    if result["coverage"].get("found", 0) < result["coverage"].get("required", 0):
        return "gold_required_test_evidence_missing", "parse", False
    return "gold_control_failed_unattributed", "test", False


def classify(document):
    """Audit scores; verified failing gold controls are ineligible negatives.

    VerifierError is an operational eligibility decision. A gold-control failure
    proves the control failed, not that a candidate environment cause is proven.
    Review is reserved for gold. When other evidence is inconclusive on a
    formal run, retain the official score if available.
    """
    official = dict(document.get("official_result", {}))
    result = {
        "schema_version": 1, "policy_version": POLICY_VERSION,
        "official_result": official, "classification": "Review",
        "reason_code": "required_test_evidence_missing", "stage": "test",
        "attribution": "unknown", "evidence_strength": "insufficient",
        "candidate_reason_codes": [], "additional_findings": [],
        "gold_control_failure": bool(document.get("gold") and official.get("resolved") is not True),
        "effective_resolved": None, "evidence": document.get("sources", []),
        "evidence_digest": evidence_digest(document), "coverage": {}, "execution": {},
        "missing_evidence": [],
        "cause_confirmed": False, "diagnostic_status": "unresolved",
        "gold_patch_issue_confirmed": False, "policy_basis": "test_evidence",
        "gold_provenance": dict(document.get("gold_provenance", {})),
        "parameter_identity_mismatches": [],
    }
    unknown = []
    if document.get("collection_errors"):
        unknown.extend(document["collection_errors"])
    task = document.get("task", {})
    identity = document.get("identity", {})
    if identity.get("instance_id") != task.get("instance_id") or not task.get("instance_id"):
        unknown.append("Instance identity mismatch or missing")
    report = document.get("report", {})
    if report and report.get("resolved") is not official.get("resolved"):
        unknown.append("Official outcome and report disagree")
    statuses = document.get("statuses", {})
    try:
        required = {g: _ids(task, g) for g in GROUPS}
    except (ValueError, TypeError) as exc:
        required = {g: [] for g in GROUPS}
        unknown.append(str(exc))
    required_ids = set(x for values in required.values() for x in values)
    if not required_ids:
        unknown.append("No frozen required test IDs")
    coverage, bad = [], []
    for group, tests in required.items():
        declared = report.get("tests_status", {}).get(group, {})
        successes, failures = set(declared.get("success", [])), set(declared.get("failure", []))
        if successes & failures or (successes | failures) - set(tests):
            unknown.append(f"Conflicting or unexpected report IDs: {group}")
        for test in tests:
            state = statuses.get(test, "MISSING")
            verdict = "success" if test in successes else "failure" if test in failures else None
            row = {"group": group, "test": test, "found": test in statuses,
                   "state": state, "official_verdict": verdict}
            coverage.append(row)
            if state not in STATES:
                unknown.append(f"Required test missing or invalid status: {test}")
            # Preserve official FAIL_ONLY semantics, but independently require evidence.
            if document.get("eval_type", "pass_and_fail") == "fail_only":
                expected = "failure" if state == "FAILED" else "success"
            else:
                expected = "success" if state in {"PASSED", "XFAIL"} else (
                    "failure" if state in {"FAILED", "ERROR", "MISSING"} else None)
            if verdict != expected or verdict is None:
                unknown.append(f"Report/state coverage conflict or unsupported skip: {test}")
            if verdict == "failure":
                bad.append(row)
    result["coverage"] = {"required": len(coverage), "found": sum(x["found"] for x in coverage),
                          "state_counts": dict(Counter(x["state"] for x in coverage)), "tests": coverage}
    if coverage and all(x["official_verdict"] is not None for x in coverage):
        if (not bad) != (official.get("resolved") is True):
            unknown.append("Official resolution conflicts with required test report")
    text = document.get("test_output", "")
    region, complete_markers = test_region(text)
    if not complete_markers:
        unknown.append("Missing, duplicate or out-of-order test boundaries")
    if not report:
        unknown.append("Missing official report")
    if not document.get("parser_found"):
        unknown.append("Official parser did not accept log")
    if report.get("patch_successfully_applied") is not True:
        unknown.append("Patch/test execution not established")
    if official.get("completed") is not True:
        unknown.append("Official evaluation incomplete")

    execution = document.get("execution", {})
    receipt = _receipt(region, execution.get("test_command_sha256")) if execution else None
    receipt_location = "test_region" if receipt else None
    if execution.get("output_sha256") == hashlib.sha256(text.encode()).hexdigest():
        full_receipt = _receipt(text, execution.get("test_command_sha256"))
        if not receipt or full_receipt != receipt:
            receipt = full_receipt
            receipt_location = "host_bound_merged_output" if receipt else None
    if execution.get("receipt_protocol") == 2:
        host_result = execution.get("sidecar_test_result")
        if (execution.get("output_verified") is True and isinstance(host_result, dict)
                and host_result.get("command_sha256") == execution.get("test_command_sha256")
                and type(host_result.get("returncode")) is int):
            if receipt is not None and receipt != host_result:
                unknown.append("Sidecar and stream test receipts disagree")
            receipt = host_result
            receipt_location = "host_collected_sidecar"
        else:
            unknown.append("Missing or unbound command sidecar")
    result["execution"] = {**execution, "test_receipt": receipt,
                           "receipt_location": receipt_location,
                           "historical_receipt_missing": not bool(execution)}
    if execution:
        if execution.get("timed_out"):
            unknown.append("Host recorded timeout; attribution requires review")
        if execution.get("returncode") not in (0, 1):
            unknown.append("Abnormal outer execution exit")
        if execution.get("test_command_sha256") and receipt is None:
            unknown.append("Expected test command receipt missing or conflicting")
    inner = _inner_pytest(text, execution, statuses)
    result["execution"]["inner_pytest"] = inner
    if receipt:
        rc, runner = receipt["returncode"], receipt.get("runner")
        if rc not in (0, 1) or receipt.get("scope") != "command":
            unknown.append("Test process interrupted, runner error or incomplete inner receipt")
        if (rc == 1 and runner not in {"pytest", "unittest", "sympy"}
                and not (runner == "tox" and inner and inner["exitstatus"] == 1
                         and required_ids.issubset(inner["tests"]))):
            unknown.append("Outer runner failure lacks inner test exit evidence")
        if rc and official.get("resolved") is True:
            unknown.append("Nonzero test command exit conflicts with official pass")
        if rc == 0 and bad:
            unknown.append("Successful test command conflicts with required failures")

    candidates = set()
    unattributed = {}
    blocks = failure_blocks(region)
    raw_states = {}
    for state, raw_id in re.findall(r"(?m)^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)[ \t]+([^\n]+)$", region):
        raw_id = raw_id.split(" - ", 1)[0].strip()
        raw_states.setdefault(raw_id, set()).add(state)
    artifacts = sorted(t for t in required_ids if _pytest_progress_token(task, t))
    result["coverage"]["parser_artifact_ids"] = artifacts
    if artifacts and not required_ids.difference(artifacts):
        unknown.append("Required test contract contains only parser artifacts")
    for test in required_ids:
        if test in artifacts:
            continue
        if len(raw_states.get(test, set())) > 1:
            unknown.append(f"Conflicting raw test states: {test}")
    for row in bad:
        block = _block_for(row["test"], blocks, required_ids)
        if block:
            candidates.update(_candidate_errors(block["body"]))
            if row["state"] == "FAILED" and not re.search(
                    r"(?m)^E\s+(?:AssertionError\b|assert\b|Failed: DID NOT (?:RAISE|WARN))", block["body"]):
                message = f"Required failure is not an attributed assertion: {row['test']}"
                unknown.append(message)
                unattributed[row["test"]] = {"message": message, "block": block}
        elif row["state"] in {"FAILED", "ERROR"}:
            unknown.append(f"Required failure block not uniquely identified: {row['test']}")
        if row["state"] == "ERROR":
            unknown.append(f"Required ERROR attribution unknown: {row['test']}")
    # Startup failures have no per-test block. Keep them candidates, not proof.
    if not statuses and complete_markers:
        candidates.update(_candidate_errors(region))
        if re.search(r"\b\d+ passed\b", region):
            candidates.add("parser_incomplete")
        if "Segmentation fault" in region:
            candidates.add("native_process_crash")
    if candidates:
        unknown.append("Runtime/test fault candidates need causal evidence")
    if document.get("gold") and official.get("resolved") is False:
        unknown.append("Gold control failed; cause not established by gold identity")

    confirmed, extras = [], []
    patch_issues = []
    subject_tests = set()
    sources = {(x.get("path"), x.get("sha256")) for x in document.get("sources", [])}
    for finding in document.get("findings", []):
        bound = (finding.get("evidence_digest") == result["evidence_digest"]
                 and bool(finding.get("evidence"))
                 and all((x.get("path"), x.get("sha256")) in sources for x in finding.get("evidence", [])))
        relevant = (finding.get("stage") in {"preflight", "bootstrap", "probe", "test_patch", "parse"}
                    or bool(set(finding.get("required_tests", [])) & required_ids))
        if not bound or not relevant:
            unknown.append("Unbound or unrelated attribution finding")
            continue
        extras.append(finding)
        if (finding.get("reason_code") == "subject_call_exception"
                and finding.get("attribution") == "subject"
                and finding.get("stage") == "test"
                and finding.get("evidence_strength") == "confirmed"
                and finding.get("recovered") is False and finding.get("basis")
                and finding.get("supporting_sources") and inner and not document.get("gold")
                and all((x.get("path"), x.get("sha256")) in sources for x in finding["supporting_sources"])):
            matched = set(finding.get("required_tests", []))
            valid = bool(matched) and matched.issubset(unattributed)
            for test in matched:
                block = unattributed.get(test, {}).get("block", {})
                call = inner["tests"].get(test, {}).get("call", {})
                exception = finding.get("exceptions", {}).get(test)
                valid = valid and bool(exception) and call.get("outcome") == "failed"
                valid = valid and finding.get("failure_block_sha256", {}).get(test) == digest(block)
                valid = valid and bool(re.search(r"(?m)^E\s+" + re.escape(exception or "") + r":", block.get("body", "")))
                crash_type = (call.get("crash", {}).get("message") or "").split(":", 1)[0]
                valid = valid and crash_type.rsplit(".", 1)[-1] == (exception or "").rsplit(".", 1)[-1]
            if valid:
                subject_tests.update(matched)
                continue
            unknown.append("Subject finding lacks matching call-phase exception evidence")
        if (finding.get("reason_code") == "gold_patch_defect"
                and finding.get("attribution") == "gold_patch"
                and finding.get("evidence_strength") == "confirmed"
                and finding.get("recovered") is False and finding.get("basis")
                and finding.get("supporting_sources")
                and bool(set(finding.get("required_tests", [])) &
                         {row["test"] for row in bad if row["state"] == "FAILED"})):
            patch_issues.append(finding)
        if (finding.get("evidence_strength") == "confirmed" and finding.get("attribution") == "verifier"
                and finding.get("recovered") is False):
            confirmed.append(finding)
        elif finding not in patch_issues:
            candidates.add(finding.get("reason_code", "unknown_attribution"))
    # Findings remove only the exact exception-attribution gap. All other
    # identity, execution, candidate, fixture and coverage guards still apply.
    unknown = [x for x in unknown if x not in {unattributed[t]["message"] for t in subject_tests}]
    # A dependency receipt comes from a scoped host stage or guarded installer.
    dependency = document.get("dependency_failure")
    if dependency and dependency.get("stage") in {"preflight", "bootstrap", "probe", "harness_install"}:
        # Host probe/bootstrap receipts or an actual installer trap establish
        # scope. A stale JSON record after a successful recovery is insufficient.
        boot = document.get("environment", {}).get("bootstrap-receipt.json", {})
        guarded = bool(re.search(r"(?m)^__DYCON_DEPENDENCY_INSTALL_FAILED__ rc=[1-9]\d* command=", text))
        failed_stage = dependency["stage"] in {"preflight", "probe"} or boot.get("status") == "failed"
        recovered = boot.get("status") == "ready" and dependency["stage"] == "bootstrap"
        if (guarded or failed_stage) and not recovered:
            confirmed.append({"reason_code": dependency.get("reason", "dependency_install_failed"),
                              "stage": dependency["stage"], "attribution": "verifier",
                              "evidence_strength": "confirmed", "dependency_failure": dependency})
        elif not recovered:
            unknown.append("Dependency failure lacks scoped terminal stage evidence")
    # An identity/integrity failure invalidates attribution to this attempt too.
    integrity_error = bool(document.get("collection_errors")) or identity.get("instance_id") != task.get("instance_id")
    verified_gold = bool(document.get("gold") and document.get("gold_provenance", {}).get("verified"))
    result["candidate_reason_codes"] = sorted(candidates)
    # Official parse_log_pytest_options canonicalizes an absolute parameter
    # path to its last path component. A frozen uncanonicalized ID can never
    # match that mapping. Report the one-to-one identity defect, never silently
    # use a similar test's success to turn the official reward into 1.
    if verified_gold and task.get("repo") == "pydicom/pydicom":
        for row in coverage:
            match = re.fullmatch(r"(.*?)\[(.*)\]", row["test"])
            if row["state"] != "MISSING" or not match:
                continue
            main, option = match.groups()
            if option.startswith("/") and not option.startswith("//") and "*" not in option:
                normalized = f"{main}[/{option.split('/')[-1]}]"
                if normalized in statuses:
                    result["parameter_identity_mismatches"].append({"required": row["test"],
                        "official_parser_id": normalized, "state": statuses[normalized]})
    # Explicit unittest skip reasons can establish a missing verifier tool on
    # gold. Require the exact required identity and terminal skip reason, rather
    # than a keyword anywhere in source/patch/expected-exception output.
    if verified_gold:
        for row in coverage:
            if row["state"] != "SKIPPED":
                continue
            pattern = r"(?m)^" + re.escape(row["test"]) + r" \.\.\. skipped ['\"]([\w+-]+) is mandatory for [^\n]+ tests['\"]\s*$"
            match = re.search(pattern, region)
            if match:
                finding = {"reason_code": "missing_verifier_tool", "stage": "test",
                    "attribution": "verifier", "evidence_strength": "confirmed", "recovered": False,
                    "required_tests": [row["test"]], "tool": match.group(1),
                    "basis": "Exact required gold test explicitly skipped for mandatory tool"}
                confirmed.append(finding)
                extras.append(finding)
                break
    submission = document.get("submission_failure", {})
    if (not document.get("gold") and not integrity_error and not confirmed
            and submission.get("reason_code") == "invalid_patch"
            and submission.get("attribution") == "subject"
            and submission.get("patch_sha256") == identity.get("patch_sha256")):
        result.update(classification="ValidOutcome", reason_code="invalid_patch",
                      stage="patch", attribution="subject", evidence_strength="confirmed",
                      effective_resolved=False, cause_confirmed=True, diagnostic_status="confirmed")
    elif confirmed and not integrity_error:
        primary = confirmed[0]
        result.update(classification="VerifierError", reason_code=primary["reason_code"],
                      stage=primary["stage"], attribution="verifier", evidence_strength="confirmed",
                      cause_confirmed=True, diagnostic_status="confirmed")
    elif verified_gold and not integrity_error and official.get("resolved") is not True:
        if patch_issues and not candidates and not [x for x in unknown if not x.startswith("Gold control failed;")]:
            result.update(classification="ValidOutcome", reason_code="gold_patch_defect",
                          attribution="gold_patch", evidence_strength="confirmed", effective_resolved=False,
                          gold_patch_issue_confirmed=True, cause_confirmed=True, diagnostic_status="confirmed")
        else:
            reason, stage, causal = _gold_failure_reason(document, result, region)
            result.update(classification="VerifierError", reason_code=reason, stage=stage,
                          attribution="verifier" if causal else "unknown", cause_confirmed=causal,
                          evidence_strength="confirmed" if causal else "control_failure",
                          diagnostic_status="confirmed" if causal else "needs_investigation",
                          policy_basis="verified_gold_control")
    elif not unknown:
        result.update(classification="ValidOutcome", reason_code="test_failed" if bad else "tests_passed",
                      attribution="subject" if bad else "none", evidence_strength="supported",
                      effective_resolved=official.get("resolved"), diagnostic_status="not_applicable")
    else:
        if integrity_error:
            result["reason_code"] = "evidence_integrity_error"
        elif candidates:
            result["reason_code"] = "failure_attribution_unknown"
        elif document.get("gold") and official.get("resolved") is False and bad:
            result["reason_code"] = "gold_control_failure_unexplained"
        elif not complete_markers or execution:
            result["reason_code"] = "execution_evidence_incomplete"
    result["candidate_reason_codes"] = sorted(candidates)
    result["additional_findings"] = extras
    result["missing_evidence"] = list(dict.fromkeys(unknown))
    result["blocker_codes"] = sorted({
        "subject_call_exception_unconfirmed" if x.startswith("Required failure is not an attributed assertion") else
        "inner_runner_receipt_missing" if x.startswith("Outer runner failure") else
        "required_pass_nonrequired_failure" if x.startswith("Nonzero test command exit conflicts") else
        "required_collection_incomplete" if x.startswith("Required test missing") else
        "evidence_conflict_or_missing" for x in unknown})
    return review_policy(result, gold=document.get("gold") is True)


def summarize(audits):
    counts = Counter(x["classification"] for x in audits)
    valid = [x for x in audits if x["classification"] == "ValidOutcome"]
    scored = [x for x in audits if x["classification"] in SCORED_CLASSIFICATIONS]
    passed = sum(x["effective_resolved"] is True for x in valid)
    return {"policy_version": POLICY_VERSION, "requested": len(audits), "classification_counts": dict(counts),
            "reason_counts": dict(Counter(x["reason_code"] for x in audits)),
            "diagnostic_status_counts": dict(Counter(x.get("diagnostic_status", "unresolved") for x in audits)),
            "unattributed_verifier_errors": sum(x["classification"] == "VerifierError" and
                                                not x.get("cause_confirmed", False) for x in audits),
            "gold_patch_defects": sum(x.get("gold_patch_issue_confirmed", False) for x in audits),
            "audit_coverage": len(valid) / len(audits) if audits else None,
            "valid_resolved": passed, "valid_resolve_rate": passed / len(valid) if valid else None,
            "scored": len(scored), "scored_resolved": sum(x["effective_resolved"] is True for x in scored),
            "official_resolved": sum(x["official_result"].get("resolved") is True for x in audits)}

import json
from pathlib import Path
import subprocess

import pytest

from benchmark_evaluation.swebench_receipts import instrument, get_test_command
from benchmark_evaluation.swebench_verifier import classify, digest, evidence_digest, summarize


def task():
    return {"instance_id": "org__repo-1", "repo": "org/repo", "version": "1",
            "base_commit": "abc", "eval_type": "pass_and_fail",
            "FAIL_TO_PASS": ["tests/test_x.py::test_x"],
            "PASS_TO_PASS": ["tests/test_x.py::test_y"]}


def base(*, statuses=None, report=None, output=None, findings=None, execution=None):
    t = task()
    statuses = statuses if statuses is not None else {t["FAIL_TO_PASS"][0]: "PASSED", t["PASS_TO_PASS"][0]: "PASSED"}
    report = report or {"resolved": True, "patch_successfully_applied": True,
                        "tests_status": {"FAIL_TO_PASS": {"success": t["FAIL_TO_PASS"], "failure": []},
                                          "PASS_TO_PASS": {"success": t["PASS_TO_PASS"], "failure": []}}}
    script = ": '>>>>> Start Test Output'\npytest -q\n: '>>>>> End Test Output'\n"
    return {"identity": {"instance_id": t["instance_id"]}, "task": t, "gold": False,
            "official_result": {"completed": True, "resolved": report.get("resolved")},
            "report": report, "statuses": statuses, "parser_found": True,
            "test_output": output or ": '>>>>> Start Test Output'\n1 passed\n: '>>>>> End Test Output'\n",
            "sources": [], "findings": findings or [], "execution": execution or {},
            "environment": {}}


def test_valid_pass_is_supported_outcome():
    result = classify(base())
    assert result["classification"] == "ValidOutcome"
    assert result["effective_resolved"] is True


def test_missing_required_evidence_is_review():
    t = task()
    report = {"resolved": False, "patch_successfully_applied": True,
              "tests_status": {"FAIL_TO_PASS": {"success": [], "failure": t["FAIL_TO_PASS"]},
                                "PASS_TO_PASS": {"success": [], "failure": t["PASS_TO_PASS"]}}}
    result = classify(base(statuses={}, report=report, output="bad output"))
    assert result["classification"] == "Review"
    assert result["effective_resolved"] is None


def test_confirmed_dependency_receipt_is_verifier_error():
    result = classify(base(findings=[{"reason_code": "missing_verifier_dependency", "stage": "bootstrap",
                                      "attribution": "verifier", "evidence_strength": "confirmed",
                                      "recovered": False, "required_tests": [], "evidence": [],
                                      "evidence_digest": "bad"}],
                           execution={"returncode": 1}))
    # Unbound findings do not get to override evidence.
    assert result["classification"] == "Review"


def test_instrumentation_has_single_scoped_receipt():
    script = "#!/bin/bash\nset -e\n: '>>>>> Start Test Output'\npytest -q\n: '>>>>> End Test Output'\n"
    details = get_test_command(script)
    wrapped, observed = instrument(script)
    assert details["runner"] == "pytest" and observed["command_sha256"] == details["command_sha256"]
    assert wrapped.count("__DYCON_TEST_RESULT__") == 1


def failed_document(*, error="AssertionError: expected 2, got 1", state="FAILED", gold=False):
    d = base()
    test = d["task"]["FAIL_TO_PASS"][0]
    d["statuses"][test] = state
    d["report"]["resolved"] = d["official_result"]["resolved"] = False
    d["report"]["tests_status"]["FAIL_TO_PASS"] = {"success": [], "failure": [test]}
    d["test_output"] = f"+ : '>>>>> Start Test Output'\n_____ test_x _____\nE   {error}\n+ : '>>>>> End Test Output'\n"
    d["gold"] = gold
    return d


def bind_finding(d, *, attribution="verifier", recovered=False):
    d["sources"] = [{"path": "test_output.txt", "sha256": "f" * 64}]
    d["findings"] = [{"reason_code": "test_framework_nose_compatibility", "stage": "test",
                      "attribution": attribution, "evidence_strength": "confirmed", "recovered": recovered,
                      "required_tests": d["task"]["FAIL_TO_PASS"], "evidence": d["sources"],
                      "evidence_digest": evidence_digest(d), "basis": "Pinned framework/source attribution"}]


def test_assertion_failure_stays_a_valid_subject_negative():
    result = classify(failed_document())
    assert result["classification"] == "ValidOutcome" and result["effective_resolved"] is False


def test_gold_does_not_prove_environment_cause():
    result = classify(failed_document(gold=True))
    assert result["classification"] == "Review" and result["gold_control_failure"] is True


@pytest.mark.parametrize("error,reason", [
    ("OSError: Network is unreachable", "test_network_unreachable"),
    ("socket.gaierror: Name or service not known", "test_network_dns_failure"),
    ("ConnectionResetError: Connection reset by peer", "test_network_connection_reset"),
    ("SSLError: CERTIFICATE_VERIFY_FAILED", "test_tls_trust_failure"),
    ("HTTPError: HTTP Error 403: Forbidden", "test_resource_http_error"),
    ("RecursionError: maximum recursion depth exceeded", "python_runtime_incompatible"),
    ("DeprecationWarning: collections.Iterable", "runtime_warning_as_error"),
])
def test_terminal_errors_are_candidates_until_causally_attributed(error, reason):
    result = classify(failed_document(error=error))
    assert result["classification"] == "Review"
    assert reason in result["candidate_reason_codes"]


def test_source_and_expected_exception_strings_do_not_trigger_rules():
    d = base()
    d["test_output"] = "diff --git a/foo b/foo\n+raise ConnectionError('Network is unreachable')\n" + d["test_output"].replace(
        "1 passed", "with pytest.raises(SSLError):\n    request()\n1 passed")
    assert classify(d)["classification"] == "ValidOutcome"


def test_exception_words_in_source_of_failed_assertion_are_ignored():
    d = failed_document()
    d["test_output"] = d["test_output"].replace("E   AssertionError", "    raise RecursionError()\nE   AssertionError")
    assert classify(d)["classification"] == "ValidOutcome"


def test_hash_bound_attribution_upgrades_only_this_evidence():
    d = failed_document(gold=True)
    bind_finding(d)
    assert classify(d)["classification"] == "VerifierError"
    d["test_output"] += "changed"
    assert classify(d)["classification"] == "Review"


@pytest.mark.parametrize("mutation", ["source", "required", "recovered", "subject"])
def test_findings_cannot_override_without_valid_scope_and_attribution(mutation):
    d = failed_document(gold=True)
    bind_finding(d)
    f = d["findings"][0]
    if mutation == "source":
        f["evidence"] = [{"path": "other", "sha256": "f" * 64}]
    elif mutation == "required":
        f["required_tests"] = ["unrelated"]
    elif mutation == "recovered":
        f["recovered"] = True
    else:
        f["attribution"] = "subject"
    assert classify(d)["classification"] == "Review"


def test_confirmed_primary_retains_unexplained_secondary_failures():
    d = failed_document(gold=True)
    bind_finding(d)
    result = classify(d)
    assert result["classification"] == "VerifierError"
    assert result["missing_evidence"] and result["additional_findings"]


def test_identity_or_collection_corruption_prevents_confirmed_attribution():
    d = failed_document(gold=True)
    d["collection_errors"] = ["Source changed"]
    bind_finding(d)
    assert classify(d)["reason_code"] == "evidence_integrity_error"


@pytest.mark.parametrize("state", ["SKIPPED", "XPASS", "MISSING", "arbitrary"])
def test_nonpass_states_are_not_fabricated_as_pass(state):
    d = base()
    d["statuses"][d["task"]["FAIL_TO_PASS"][0]] = state
    assert classify(d)["classification"] == "Review"


def test_xfail_preserves_official_pass_semantics():
    d = base()
    d["statuses"][d["task"]["FAIL_TO_PASS"][0]] = "XFAIL"
    assert classify(d)["effective_resolved"] is True


def test_report_omits_a_skipped_required_id():
    d = base()
    d["statuses"][d["task"]["FAIL_TO_PASS"][0]] = "SKIPPED"
    d["report"]["tests_status"]["FAIL_TO_PASS"]["success"] = []
    assert classify(d)["classification"] == "Review"


def test_report_false_with_all_successes_is_a_conflict():
    d = base()
    d["report"]["resolved"] = d["official_result"]["resolved"] = False
    assert classify(d)["classification"] == "Review"


@pytest.mark.parametrize("rc", [2, 3, 4, 5, 137, 139])
def test_abnormal_test_exit_cannot_be_hidden_by_successful_cleanup(rc):
    d = base()
    command_hash = "a" * 64
    row = {"command_sha256": command_hash, "returncode": rc, "runner": "pytest", "scope": "command"}
    d["execution"] = {"returncode": 0, "test_command_sha256": command_hash}
    d["test_output"] = d["test_output"].replace("1 passed", "1 passed\n__DYCON_TEST_RESULT__ " + json.dumps(row))
    assert classify(d)["classification"] == "Review"


def test_tox_exit_one_is_not_assumed_to_be_pytest_test_failure():
    d = failed_document()
    command_hash = "a" * 64
    row = {"command_sha256": command_hash, "returncode": 1, "runner": "tox", "scope": "command"}
    d["execution"] = {"returncode": 0, "test_command_sha256": command_hash}
    d["test_output"] = d["test_output"].replace("+ : '>>>>> End", "__DYCON_TEST_RESULT__ " + json.dumps(row) + "\n+ : '>>>>> End")
    assert classify(d)["classification"] == "Review"


def test_duplicate_receipts_are_rejected():
    d = base()
    row = {"command_sha256": "a" * 64, "returncode": 0, "runner": "pytest", "scope": "command"}
    d["execution"] = {"returncode": 0, "test_command_sha256": "a" * 64}
    d["test_output"] = d["test_output"].replace("1 passed", ("__DYCON_TEST_RESULT__ " + json.dumps(row) + "\n") * 2)
    assert classify(d)["classification"] == "Review"


def test_real_shell_retains_test_failure_after_cleanup_success(tmp_path):
    script = "#!/bin/bash\nset -uxo pipefail\n: '>>>>> Start Test Output'\nfalse\n: '>>>>> End Test Output'\ntrue\n"
    wrapped, details = instrument(script)
    result = subprocess.run(["bash"], input=wrapped, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
    assert result.returncode == 0
    receipts = [json.loads(line.split(" ", 1)[1]) for line in result.stdout.splitlines()
                if line.startswith("__DYCON_TEST_RESULT__ ")]
    assert len(receipts) == 1 and receipts[0]["returncode"] == 1
    assert receipts[0]["command_sha256"] == details["command_sha256"]


def test_bootstrap_failure_is_verifier_error_and_recovered_bootstrap_is_ignored():
    d = base()
    d["dependency_failure"] = {"stage": "bootstrap", "reason": "missing_distribution"}
    d["environment"]["bootstrap-receipt.json"] = {"status": "failed", "returncode": 1}
    assert classify(d)["classification"] == "VerifierError"
    d["environment"]["bootstrap-receipt.json"] = {"status": "ready", "returncode": 0}
    assert classify(d)["classification"] == "ValidOutcome"


def test_generic_error_string_does_not_bypass_install_guard():
    d = base()
    d["test_output"] = d["test_output"].replace("1 passed", "expected = 'ERROR: No matching distribution found'\n1 passed")
    assert classify(d)["classification"] == "ValidOutcome"


@pytest.mark.parametrize("name,candidate", [
    ("psf__requests-2931", "test_network_unreachable"),
    ("sympy__sympy-13146", "python_runtime_incompatible"),
    ("sympy__sympy-13001", "python_runtime_incompatible"),
    ("sympy__sympy-13865", "runtime_warning_as_error"),
    ("pyvista__pyvista-3675", None),
    ("pydicom__pydicom-1031", None),
    ("pydicom__pydicom-1428", None),
])
def test_real_gold_tracebacks_remain_review_without_causal_proof(name, candidate):
    p = Path(__file__).parent / "fixtures/swebench_verifier" / (name + ".json")
    raw = json.loads(p.read_text())
    test, state = raw["required"]["test"], raw["required"]["state"]
    d = failed_document(gold=True, state=state)
    d["identity"]["instance_id"] = d["task"]["instance_id"] = name
    d["task"].update(FAIL_TO_PASS=[test], PASS_TO_PASS=[])
    d["statuses"] = {test: state}
    d["report"]["tests_status"] = {"FAIL_TO_PASS": {"success": [], "failure": [test]},
                                    "PASS_TO_PASS": {"success": [], "failure": []}}
    d["test_output"] = raw["test_output"]
    result = classify(d)
    assert result["classification"] == "Review" and result["effective_resolved"] is None
    if candidate:
        assert candidate in result["candidate_reason_codes"]
    if name.endswith("13865"):
        assert "python_runtime_incompatible" not in result["candidate_reason_codes"]


def test_summary_is_mutually_exclusive_and_does_not_make_review_a_zero():
    docs = [base(), failed_document(), failed_document(gold=True), failed_document(gold=True)]
    bind_finding(docs[-1])
    results = [classify(d) for d in docs]
    summary = summarize(results)
    assert sum(summary["classification_counts"].values()) == 4
    assert summary["valid_resolved"] == 1 and summary["valid_resolve_rate"] == 0.5
    assert [x["effective_resolved"] for x in results] == [True, False, None, None]

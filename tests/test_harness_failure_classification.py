from benchmark_evaluation.harness import classify_harness_failure


def test_classifies_patch_apply_failure_before_generic_error():
    result = classify_harness_failure(
        evidence="EvaluationError: >>>>> Patch Apply Failed:\npatch: Only garbage was found in the patch input"
    )
    assert result["failure_type"] == "invalid_patch"


def test_classifies_network_exception_separately():
    result = classify_harness_failure(exception=ConnectionError("Connection reset by peer"))
    assert result["failure_type"] == "network_error"


def test_classifies_test_timeout_separately():
    result = classify_harness_failure(evidence="Timeout error: 300 seconds exceeded.")
    assert result["failure_type"] == "timeout"


def test_unknown_failure_is_explicitly_other_error():
    result = classify_harness_failure(exception=RuntimeError("unexpected evaluator failure"))
    assert result["failure_type"] == "other_error"

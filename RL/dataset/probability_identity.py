"""Separate the model function from factual execution evidence, without Torch."""
from core.io import hash_json

RAW_CONTRACT = {"temperature": 1.0, "normalization": "full_vocabulary",
                "logits_stage": "pre_transform", "targets": "completion_tokens"}
FUNCTION_FIELDS = ("checkpoint_fingerprint", "tokenizer_fingerprint",
                   "adapter_fingerprint", "effective_model_overrides")
REDUCTION = "fp32_log_softmax_full_vocabulary_v1"
# These are measurements, not conditions that change the scoring computation.
# Original runtime bytes (including these fields) stay in source provenance.
RUNTIME_MEASUREMENTS = {"startup_and_capture_seconds", "queue_metrics", "dispatch_observations"}


def split_identity(scoring):
    if (not isinstance(scoring, dict) or not set(FUNCTION_FIELDS) <= scoring.keys()
            or not all(isinstance(scoring[k], str) and scoring[k].strip()
                       for k in FUNCTION_FIELDS[:2])
            or scoring["adapter_fingerprint"] is not None and
               not (isinstance(scoring["adapter_fingerprint"], str) and scoring["adapter_fingerprint"].strip())
            or not isinstance(scoring["effective_model_overrides"], dict)
            or scoring.get("collector_dtype") not in {"float16", "bfloat16", "float32"}
            or not isinstance(scoring.get("runtime"), dict) or not scoring["runtime"]
            or not isinstance(scoring.get("implementation"), dict) or not scoring["implementation"]):
        raise ValueError("Incomplete scoring function/execution cohort metadata")
    function = {k: scoring[k] for k in FUNCTION_FIELDS}
    function["raw_contract"] = RAW_CONTRACT
    cohort = {k: scoring[k] for k in ("collector_dtype", "runtime", "implementation")}
    cohort["runtime"] = {k: v for k, v in cohort["runtime"].items() if k not in RUNTIME_MEASUREMENTS}
    if not cohort["runtime"]:
        raise ValueError("Execution cohort has no factual runtime identity")
    cohort["reduction_contract"] = REDUCTION
    return function, cohort


def canonical_scoring(manifest):
    """Validate schema 3 and expose the existing training compatibility interface."""
    function = manifest.get("scoring_function_identity", {})
    cohorts = manifest.get("execution_cohorts", {})
    if (manifest.get("schema_version") != 3
            or manifest.get("stream") != "raw_t1_output_tokens"
            or manifest.get("probability_semantics") != "raw_t1_output_tokens_v1"
            or manifest.get("raw_contract") != RAW_CONTRACT
            or manifest.get("values_dtype") != "<f4"
            or manifest.get("index") != "branches.jsonl"
            or not isinstance(cohorts, dict) or not cohorts
            or set(function) != {*FUNCTION_FIELDS, "raw_contract"}
            or function["raw_contract"] != RAW_CONTRACT):
        raise ValueError("Canonical probability raw/function/cohort contract mismatch")
    dtypes = set()
    for key, cohort in cohorts.items():
        if (not isinstance(cohort, dict) or hash_json(cohort) != key
                or cohort.get("reduction_contract") != REDUCTION):
            raise ValueError("Invalid execution cohort identity")
        normalized_function, _ = split_identity({**function, **cohort})
        if normalized_function != function:
            raise ValueError("Invalid scoring function identity")
        dtypes.add(cohort["collector_dtype"])
    if len(dtypes) != 1:
        raise ValueError("Probability sources cannot mix collector compute dtype")
    return {**{k: function[k] for k in FUNCTION_FIELDS}, "collector_dtype": dtypes.pop(),
            "runtime": {k: v["runtime"] for k, v in cohorts.items()},
            "implementation": {k: v["implementation"] for k, v in cohorts.items()}}

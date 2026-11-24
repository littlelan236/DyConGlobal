"""Failure rules for explicitly bound online RL runs; ordinary runs stay unchanged."""
LIMIT_FAILURES = frozenset({'LimitsExceeded', 'reasoning_loop'})


def failure_policy(cfg):
    return cfg.get('execution', {}).get('online_rl_failure_policy') or {}


def baseline_failure_reward(cfg, source, record):
    reward = failure_policy(cfg).get('baseline_failure_reward')
    status = record.get('termination_reason', record.get('agent_exit_status', record.get('exit_status')))
    return reward if source == 'baseline' and reward is not None and status in LIMIT_FAILURES else None


def verifier_error(outcome):
    if not isinstance(outcome, dict):
        return False
    audit = outcome.get('verifier_audit') or {}
    raw = outcome.get('raw_outcome') or outcome
    return (audit.get('classification') == 'VerifierError'
            or outcome.get('status') == 'VerifierError'
            or raw.get('status') in {'infra_error', 'ConnectionError', 'Timeout', 'error'})


def verifier_discard(outcome):
    marker = (outcome or {}).get('online_rl_discard') or {}
    return marker.get('reason') == 'verifier_error_retries_exhausted' and verifier_needs_retry(outcome)


def verifier_needs_retry(outcome):
    if verifier_error(outcome):
        return True
    official = outcome.get('official_result') if isinstance(outcome, dict) else None
    audit = (outcome.get('verifier_audit') or {}) if isinstance(outcome, dict) else {}
    return (isinstance(official, dict) and type(official.get('resolved')) is not bool
            and not (audit.get('classification') == 'ValidOutcome'
                     and audit.get('reason_code') == 'missing_patch'
                     and outcome.get('resolved') is False))

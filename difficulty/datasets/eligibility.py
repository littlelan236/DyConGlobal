"""Explicit evaluation eligibility, separate from terminal workload labels."""
import math

from application.execution.subject import subject_digest


def validate_policy(policy):
    if policy not in {'resolved_only', 'positive_reward', 'accepted_terminal', 'submitted'}:
        raise ValueError('Unknown training eligibility policy: ' + str(policy))


def positive_reward(rewards):
    value = rewards.get('reward') if isinstance(rewards, dict) else None
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def eligible_evaluation(row, evidence, rollout_id, policy):
    validate_policy(policy)
    if policy == 'submitted':
        # Workload-to-submission labels require no benchmark-success outcome.
        return (row.get('submission_status') == 'submitted'
                and row.get('termination_reason') == 'Submitted'
                and row.get('evaluation_subject', {}).get('kind') not in {'workspace', 'web_trace'})
    if row.get('evaluation_status') != 'completed':
        return False
    from benchmark_evaluation.swebench_settlement import training_label
    value = training_label(row, evidence, rollout_id)
    if type(value) is not bool:
        return False
    if policy == 'resolved_only':
        return value is True
    if row.get('evaluation_subject', {}).get('kind') != 'workspace':
        return False
    evaluation = evidence['evaluations'].get(row.get('selected_evaluation_id'), {})
    outcome = evaluation.get('outcomes', {}).get(rollout_id, {})
    key, digest = subject_digest(row)
    return (outcome.get('status') == 'completed'
            and outcome.get(key) == digest
            and outcome.get('resolved') == row.get('resolved')
            and ((policy == 'accepted_terminal' and type(row.get('resolved')) is bool)
                 or (policy == 'positive_reward' and positive_reward(outcome.get('rewards')))))


def verify_positive_report(row, report):
    key, digest = subject_digest(row)
    if (row.get('evaluation_status') != 'completed'
            or report.get('evaluation_status') != 'completed'
            or report.get('evaluation_id') != row.get('selected_evaluation_id')
            or report.get(key) != digest
            or report.get('resolved') != row.get('resolved')
            or report.get('attempt_id') != row.get('attempt_id')
            or not positive_reward(report.get('raw_rewards'))):
        raise ValueError('Positive reward selection does not bind a completed positive official evaluation')


def verify_terminal_report(row, report):
    """Bind actual-terminal labels to an evaluated workspace, including zero reward."""
    key, digest = subject_digest(row)
    if (row.get('evaluation_subject', {}).get('kind') != 'workspace'
            or type(row.get('resolved')) is not bool
            or row.get('evaluation_status') != 'completed'
            or report.get('evaluation_status') != 'completed'
            or report.get('evaluation_id') != row.get('selected_evaluation_id')
            or report.get(key) != digest
            or report.get('resolved') != row.get('resolved')
            or report.get('attempt_id') != row.get('attempt_id')):
        raise ValueError('Accepted terminal selection must bind a completed exact official evaluation')

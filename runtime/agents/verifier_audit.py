"""Shared SkillsBench below-one verifier policy and explicit zero-API adapters."""
from __future__ import annotations

import math

from runtime.agents.verifier_evidence import (
    check_sources,
    frozen_contract,
    normalize,
    verifier_files,
)
from runtime.agents.verifier_ownership import enrich_ownership
from runtime.agents.verifier_policy import POLICY, decide


def finite_reward(value):
    return type(value) in (int, float) and math.isfinite(value)


def is_candidate(value):
    return finite_reward(value) and value < 1


def frozen_verifier_contract(task):
    contract = frozen_contract(task)
    return {k: v for k, v in contract.items() if k != 'sources'}


_verifier_files = verifier_files


def classify_attempt(attempt_dir, task, *, reward, evaluation_raw=None):
    """Normalize once for either entry, then invoke the only final classifier."""
    if not is_candidate(reward):
        return {'schema_version':2, 'policy':POLICY, 'classification':'NotCandidate',
                'reason_code':'not_finite_sub_one_reward', 'reward':reward}
    snap = normalize(attempt_dir, task, reward, evaluation_raw)
    if snap['contract']['verified']:
        enrich_ownership(snap)
    check_sources(snap)
    return decide(snap)


def audit_zero_evaluation(attempt_dir, task, status, resolved, policy, raw):
    """Historical name; v2 audits all finite rewards below one before settlement."""
    rewards = raw.get('rewards')
    reward = rewards.get('reward') if isinstance(rewards,dict) else None
    negative_range_review = (status == 'UnsupportedRewardPolicy' and is_candidate(reward) and reward < 0)
    if (status not in {'Submitted','AgentCompleted','AgentTimeout'} and not negative_range_review) or not is_candidate(reward):
        return status, resolved, policy, None
    receipt = classify_attempt(attempt_dir, task, reward=reward, evaluation_raw=raw)
    if receipt['classification'] == 'VerifierError':
        return 'VerifierError', None, 'verifier_incomplete', receipt
    return status, resolved, policy, receipt


def classify_zero_reward_trajectory(attempt_dir, *, disposition, reward,
        evaluation_raw=None, evaluation_report=None, ctrf_expected=None,
        task_completion_patterns=(), task=None):
    """Deprecated zero-only tuple API. Supply task identity for a v2 decision.

    Final evaluation-report fields and completion regexes are intentionally not
    evidence. ValidZero is only a label translation for this zero-only API.
    """
    if disposition != 'accepted' or not finite_reward(reward) or reward != 0:
        return 'NotCandidate', 'Requires accepted numeric zero reward'
    result = classify_attempt(attempt_dir, task, reward=reward, evaluation_raw=evaluation_raw)
    label = 'ValidZero' if result['classification'] == 'ValidOutcome' else result['classification']
    return label, result.get('reason',result['reason_code'])


def inspect_ctrf(attempt_dir):
    """Legacy inspection only; it is never a final evaluation decision."""
    import json
    reports = []
    for path in verifier_files(attempt_dir):
        if path.suffix == '.json' and 'ctrf' in path.name.lower():
            try:
                reports.append(json.loads(path.read_text(encoding='utf-8')))
            except (OSError,ValueError):
                reports.append({})
    executed = any(any(t.get('status') in {'passed','failed'} for t in r.get('results',{}).get('tests',[])
                       if isinstance(t,dict)) for r in reports if isinstance(r,dict))
    return executed, bool(reports)


def classify_verifier_logs(logs, *, ctrf_executed=False, task_completion_patterns=(), unreadable=False):
    """Deprecated insufficient-evidence adapter; logs alone cannot authorize v2."""
    return 'Review', 'Frozen contract and attempt evidence required by skillsbench_verifier_audit_v2'

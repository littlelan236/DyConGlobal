"""Torch-free scalar allowlist and organized TensorBoard tags."""
from RL.grpo.k3_diagnostics import K3_SCALARS

SIGNAL_TAGS = {
    "trainable_parameters": "full/trainable_parameters",
    "global_grad_norm": "full/global_grad_norm",
    "policy_changed": "full/policy_changed",
    "microsteps": "full/microsteps",
    "compute_seconds": "timing/compute_seconds",
    "loss": "loss/total", "policy_loss": "loss/policy",
    "positive_logprob_change": "learning/logprob/positive_advantage",
    "negative_logprob_change": "learning/logprob/negative_advantage",
    "positive_ratio_mean": "learning/ratio/positive_advantage",
    "negative_ratio_mean": "learning/ratio/negative_advantage",
    "preference_gap": "learning/preference/gap",
    "advantage_logprob_alignment": "learning/preference/advantage_alignment",
    "preference_positive_group_fraction": "learning/preference/positive_group_fraction",
    "preference_group_count": "learning/support/preference_groups",
    "positive_advantage_tokens": "learning/support/positive_tokens",
    "negative_advantage_tokens": "learning/support/negative_tokens",
    "ratio_mean": "drift/ratio/mean", "ratio_min": "drift/ratio/min", "ratio_max": "drift/ratio/max",
    "ratio_low_token_fraction": "drift/ratio/below_clip_token_fraction",
    "ratio_high_token_fraction": "drift/ratio/above_clip_token_fraction",
    "clip_fraction": "policy/clipping/outside_interval_objective_fraction",
    "ratio_guard_excluded_fraction": "policy/guard/excluded_token_fraction",
    "ratio_guard_excluded_tokens": "policy/guard/excluded_tokens",
    "ratio_guard_eligible_tokens": "policy/guard/eligible_tokens",
    "k3_cap": "drift/k3/configured_cap",
    "k3_cap_clipped_tokens": "drift/k3/clipped_tokens",
    "k3_cap_eligible_tokens": "drift/k3/cap_eligible_tokens",
    "k3_cap_clipped_fraction": "drift/k3/clipped_token_fraction",
    "k3_raw_penalty_mean": "drift/k3/raw_mean_after_guard",
    "k3_raw_nonfinite_tokens": "drift/k3/raw_nonfinite_tokens",
    "k3_capped_penalty_max": "drift/k3/capped_max",
    "fully_guarded_branches": "policy/guard/fully_excluded_branches",
    "fully_guarded_groups": "policy/guard/fully_excluded_groups",
    "policy_active_tokens": "policy/active/tokens",
    "gradient_norm": "optimizer/gradient/preclip_norm",
    "gradient_clip_threshold": "optimizer/gradient/clip_threshold",
    "gradient_clip_scale": "optimizer/gradient/clip_scale",
    "gradient_parameter_coverage": "optimizer/gradient/parameter_coverage",
    "learning_rate": "optimizer/learning_rate/current",
    "next_learning_rate": "optimizer/learning_rate/next",
    "parameter_update_sample_norm": "optimizer/update/sample_absolute_norm",
    "parameter_update_sample_relative_norm": "optimizer/update/sample_relative_norm",
    "parameter_before_sample_norm": "optimizer/update/sample_before_norm",
    "parameter_update_sample_elements": "optimizer/update/sample_elements",
    "retry_time_fraction": "timing/retry_fraction",
    "evaluated_at_global_step": "progress/evaluated_at_global_step",
}
for field, folder in (("policy_active", "policy/active"), ("policy_saturated", "policy/clipping/saturated"),
                      ("ratio_guard_excluded", "policy/guard")):
    for suffix in ("objective_fraction", "advantage_weighted_fraction", "positive_token_fraction", "negative_token_fraction"):
        SIGNAL_TAGS[field + "_" + suffix] = folder + "/" + suffix
for suffix in ("min", "p01", "p50", "p99", "max"):
    SIGNAL_TAGS["log_ratio_" + suffix] = "drift/log_ratio/" + suffix
SIGNAL_TAGS["k3_penalty_mean"] = "drift/k3/objective_mean_after_guard"
for field in K3_SCALARS:
    SIGNAL_TAGS[field] = "drift/k3/" + field.removeprefix("k3_")
for field in ("updates_per_second", "groups_per_second", "branches_per_second", "completion_tokens_per_second",
              "policy_active_tokens_per_second"):
    SIGNAL_TAGS[field] = "throughput/" + field
for field in ("queue_wait_seconds", "failed_attempt_seconds", "successful_attempt_seconds", "gradient_sync_seconds",
              "rank_critical_path_seconds", "checkpoint_seconds"):
    SIGNAL_TAGS[field] = "timing/" + field

for field in ("cpu_rss_bytes", "cpu_hwm_bytes", "host_active_bytes", "host_allocated_bytes",
              "host_active_peak_bytes", "host_allocated_peak_bytes", "host_native_cache_bytes",
              "host_pool_reserved_bytes", "host_pool_idle_bytes", "host_pool_leased_bytes",
              "host_pool_pending_bytes", "host_actual_budget_bytes",
              "offload_selected_boundaries", "offload_retained_boundaries"):
    SIGNAL_TAGS[field] = "memory/process/" + field
# A host-global sample belongs to the recording rank; never sum across ranks.
SIGNAL_TAGS["host_global_mem_available_bytes"] = "memory/host_global/mem_available_bytes"

SIGNAL_SCALARS = tuple(SIGNAL_TAGS) + ("group_count", "branch_count", "token_count", "ratio_guard_enabled",
    "ratio_guard_threshold", "positive_branches", "negative_branches")

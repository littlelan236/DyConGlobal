# Frozen KL reference recomputation

Set objective.kl.computation: recompute, beta: 0.02 and k3_cap: 1.0.
The default legacy mode retains old checkpoint semantics. No KL-specific forward
runs when beta is zero. Recompute is supported for LoRA offline DyCon, offline
baseline, mixed offline training and online bootstrap, including followers.
Full-policy training rejects this mode because adapter swapping cannot freeze
the updated base weights.

RL/policy_reference.py owns the frozen initial adapter scorer. The historical
RL.online_policy.InitialPolicyScorer import remains compatible. Fresh runs freeze
the initial policy; resume reuses the saved initial adapter. Computation mode
is part of the checkpoint contract.

DyCon retains its historical PPO denominator source. A group reference forward
runs under eval/no_grad before the policy prefix graph is created. It reuses the
group prompt and holds only detached completion logprobs, then releases them at
group end. Adapter replacement never occurs across a live policy graph.
Baseline and online paths recompute before each training unit, overriding any
stored KL reference values. Baseline units can be merged traces.

The old reference-cache artifact role bindings remain historical PPO anchor
provenance; mixed attempts explicitly record their effective recomputed KL
identity in the top-level kl_reference contract. Ratios do not change.

Validation: independent FP32 frozen-model loss/gradient oracle over two updates,
prefix enabled/disabled, checkpointing enabled/disabled; forced recompute for
both source types; zero-beta skip; exception restoration; real service exact
resume with an external PPO denominator; tiny CUDA BF16/FLA mixed gradients.

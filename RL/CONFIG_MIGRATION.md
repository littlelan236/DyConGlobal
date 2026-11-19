# Explicit configuration revision on LoRA GRPO resume

Use the same formal `dycon lora-train --training-config YAML` entrypoint.
`run.start.config_migration: null` retains the existing exact-resume admission.
An explicit revision is bound to both source and target schedule hashes:

```yaml
run:
  start:
    mode: resume
    checkpoint: /ABSOLUTE/ATTEMPT/checkpoints/step-NNNNNN
    config_migration:
      source_schedule_sha256: SOURCE_HASH
      target_schedule_sha256: TARGET_HASH
      reason: Explicitly requested configuration revision
```

Compute these hashes with the existing `core.io.hash_json` and
`RL.config.schedule_contract` helpers. The source is the complete checkpoint's
`state.json.contract.schedule`; the target is the parsed new training config.
No checkpoint or frozen manifest is edited. A mismatched hash rejects admission.

This is a general revision envelope, not a two-field memory/topology whitelist.
The full new configuration is parsed and its changes recorded. Compatible future
execution settings (memory allocation/reclamation, clipping, checkpoint cadence,
learning rate, weight decay, scheduler settings and diagnostics) may be revised.
Optimizer moments, optimizer step, scheduler completed step, parameters, RNG and
cursors are loaded first. Revised optimizer settings are then explicitly applied;
loading old optimizer groups must not silently undo the requested learning rate.

Configuration permission is distinct from state compatibility. A changed model
or adapter tensor inventory/backend must pass existing tensor and execution
identity checks. Changing a frozen dataset/selection, advantage, anchor, logical
batch/window, accumulation or complete schedule requires separately defined
replanning/reset semantics and is rejected here. This interface never interprets
an incompatible cursor as progress through a new plan, or silently warm-starts.
Full-policy FSDP2 migration remains unsupported by this LoRA transition path.

`config-transition.json` records source receipt, both schedule hashes, reason,
source state, schedule/configuration differences and the previous transition
hash. It travels with new checkpoints. Old source bytes remain immutable.
Default exact-mode behavior and source receipt/tensor verification stay enabled.
The guarantee is state-preserving continuation with revised future settings,
not an unchanged optimization trajectory or bitwise cross-topology identity.

`runtime.topology_policy: preserve_updates` now additionally admits a single
worker whole-parent checkpoint as source for 2/4/8 workers. A source without a
distributed contract retains that original logical contract; new physical worker
metadata is recorded separately. Its single RNG state initializes the target
bookkeeping states; every forward/backward task must still prove no RNG use.
The same logical parents, branch weights, optimizer windows and anchor survive.

Validation: `tests/test_rl_config_migration.py` exercises hash-bound admission,
immutable/backend rejection, optimizer settings after moment restoration, and
real tiny-Qwen prepared/streaming single-to-two Gloo checkpoint round trips.
Existing topology tests cover 2/4/8-worker remapping.

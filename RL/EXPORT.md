# LoRA checkpoint export for rollout serving

`dycon lora-export` verifies a complete project-owned update checkpoint and its
adapter/base binding, then publishes a full model for the existing rollout API.
Run it from `.venvs/rl` (Torch and safetensors); help remains available in runner.

```bash
.venvs/rl/bin/dycon lora-export --checkpoint /absolute/checkpoints/step-000450 \
  --base-model /absolute/models/Qwen3.5-9B --output /absolute/models/new-serving-model \
  --dtype float16
```

There is no implicit latest selection. Only the project's plain, fixed-rank,
bias-free all-linear LoRA export is accepted; DoRA, RS-LoRA and extra saved
modules are rejected. All frozen adapter tensor names/ranks must match the base
shard index. The operation uses CPU FP32 matrix products and addition, then
rounds once to the requested serving dtype. It preserves every base tensor,
including unloaded MTP states, and copies tokenizer/template/config assets.
FP16 exported weights keep ordinary FP16 serving for both experimental arms;
this is a merged inference representation, not bitwise BF16 training forwards.

Publication uses a same-filesystem staging directory and refuses overwrite.
`lora-export-receipt.json` records the exact source checkpoint/adapter hashes,
base identity, compute/output dtype and output file hashes. The final model
is registered through `core.model_identity.register`; put the returned path and
fingerprint into root config before creating a run through `dycon run create`.
Export uses no GPU and does not start or modify training or rollout.

`--dry-run` verifies source identities and reports the plan without importing
Torch or writing a model. It does update the existing base identity registry.
The export receipt remains in the model directory. Training resume must continue
using the original complete checkpoint, not the merged serving directory.

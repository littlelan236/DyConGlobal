# Historical long-context validation

Recorded 2026-11-10, reconstructed from existing evidence during sparse-interface
implementation. No new GPU training or capacity test was run for this report.

The separate 2026-11-14 adaptive-offload and whole-step OOM retry implementation
has CPU validation and a CLI dry-run; its GPU acceptance is deferred until the
required GPUs are free. The historical measurements below do not validate that
new path. See [OOM recovery validation](OOM_RECOVERY_VALIDATION.md).

The preserved long-context implementation was an uncommitted working-tree change
on `feature/lora-long-context`, over HEAD `4f1e2b3`. CPU memory tests recorded
68 passed; the earlier existing-function regression log records 95 passed.
The 2K bf16 numerical audit records identical dense/chunked loss
0.5168296742941573 and maximum adapter-gradient absolute difference
7.301568984985352e-7 against fixed atol=2e-5 / rtol=0.002, with frozen base unchanged.

Historical dense CE capacity evidence:

| Input length | Fresh peak allocated GiB | Fresh peak reserved GiB | Fresh mean seconds/update | Recorded steps |
|---|---:|---:|---:|---|
| 2048 | 20.32 | 21.54 | 1.83 | 2 fresh + step-3 resume + 1 warm-start |
| 4096 | 21.16 | 23.26 | 3.68 | 2 fresh + step-3 resume + 1 warm-start |
| 8192 | 24.23 | 25.20 | 12.11 | 2 fresh + step-3 resume + 1 warm-start |
| 16384 | 30.37 | 31.41 | 40.34 | 2 fresh + step-3 resume + 1 warm-start |
| 24576 | 36.51 | 38.11 | 82.56 | 2 fresh + step-3 resume + 1 warm-start |

These fresh measurements used CE/MLP chunks of 512, exact-length microbatch one,
Flash-only full attention, layer checkpointing and **no CPU activation offload**.
They include both linear and full attention; the installed linear/convolution
kernels were PyTorch reference implementations. The separate bf16 audit exercised
checkpoint-boundary CPU transfers and recorded 512 MiB each direction; it must
not be confused with capacity-ladder configuration.

Evidence remains in:

- `temp/lora-long-context-20261110/cpu-memory-final.log`
- `temp/lora-long-context-20261110/regressions.log`
- `temp/lora-long-context-20261110/bf16-equivalence.json`
- `temp/lora-long-context-20261110/ablations-20261110-140837/report.json`
- `temp/lora-long-context-20261110/capacity-20261110-141843/progress.json`
- `/data3/wangtianyi/DyConGlobal/RL/checkpoints/lora-long-20261110-141843-LENGTH[-resume|-reload]/`

The progress evidence contains fresh updates 1/2, independent resume update 3,
and adapter warm-start update 1 for every table length. This report summarizes
historical files; it does not rerun GPU checks or re-certify artifacts against the
new sparse source contract. Old complete checkpoints require explicit adapter
warm-start under the new source/supervision contract.

The user stopped further capacity work. 32K and longer are not accepted, and an
unexecuted 24K preset draft is not a passed test. Sparse CPU/dry-run acceptance is
reported separately in `SPARSE_SUPERVISION.md`; no new sparse GPU, throughput or
capacity conclusion follows from this historical dense evidence.

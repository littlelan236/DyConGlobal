# RL server deployment

Environment installation does not start a run. Use the same checkout and formal
`dycon lora-train --training-config YAML` entrypoint for training.

## Install the environments

On Linux, install Git, `flock` (util-linux), and uv first. The validated deployment
used uv 0.13.8. Python is pinned by the repository's `.python-version` to 3.12.15;
uv can download that interpreter if it is unavailable. Configure `HTTPS_PROXY`
and `HTTP_PROXY` in the shell when needed; keep credentials and host proxy
addresses outside tracked configuration.

From the checkout root:

```bash
bash runtime/scripts/setup_rl.sh --with-runner
```

This installs runner, then RL, using their checked-in locks. To install just one:

```bash
bash runtime/scripts/setup_runner.sh
bash runtime/scripts/setup_rl.sh
```

The scripts work from any checkout path. Locks contain dependency packages;
the current checkout is registered separately with `--no-deps -e .` after the
locked build tools are installed. Root `egg-info` metadata from earlier editable installs is refreshed
through the locked build backend so it cannot shadow the current dependencies.
Existing `.venvs/runner` / `.venvs/rl` links to external storage are supported.
An existing environment with a different
Python version is rejected rather than replaced. For an exact clean deployment,
start with new environment directories; install does not remove unrelated
packages from an existing environment.

Installers share a per-user setup lock and run sequentially to avoid duplicate
downloads and cache-lock timeouts. Defaults are `UV_LOCK_TIMEOUT=900` and
`UV_HTTP_TIMEOUT=90`; environment variables can override both. `UV_BIN` selects
another uv executable. No proxy or cache location is hard-coded into locks.

RL installation enforces the hashes in `locks/rl.txt`. Both environments finish
with `uv pip check`, an ijson import and training CLI help. PEFT is pinned to
0.21.2 in both `locks/rl.in` and the package's RL extra, matching the validated
training environment. `ijson==3.5.1` is included for metric reads. TensorBoard
and Matplotlib retain their separate analysis/runner ownership; FLA and
causal-conv kernels are not installed implicitly.

## Update the RL lock without upgrading the training stack

Edit `locks/rl.in` and matching package extras in `pyproject.toml`, then compile
with the existing lock as a version constraint:

```bash
uv pip compile --python .venvs/rl/bin/python --generate-hashes \
  --constraint locks/rl.txt locks/rl.in --output-file locks/rl.txt
git diff -- locks/rl.in locks/rl.txt pyproject.toml
```

Review package versions and hashes before applying the result. Do not add
`--upgrade` for deployment fixes. Deliberate dependency upgrades require a
separate compatibility review and revised constraints. The runner lock must not
contain an editable checkout or a host's absolute project path.

## Verify a new installation

Run the relevant existing CPU regressions in the freshly installed environments:

```bash
CUDA_VISIBLE_DEVICES='' .venvs/rl/bin/python -m pytest -q \
  tests/test_grpo_objective.py tests/test_rl_training_signals.py \
  tests/test_grpo_token_diagnostics.py
CUDA_VISIBLE_DEVICES='' .venvs/runner/bin/python -m pytest -q \
  tests/test_capture_background_recovery.py tests/test_capture_quota_writer_guard.py
.venvs/runner/bin/dycon config-check
```

These are CPU tests with temporary synthetic fixtures, not the production RL
run. `pip check` alone cannot detect undeclared imports; the metric-reader tests
exercise the ijson dependency. Configuration schema validation and CLI help do
not establish model/data availability or GPU training correctness.

## Prepare host paths before training

Use the project's existing root `config.yaml` and explicit task training YAML;
do not add implicit config overlays. Keep host edits and dated execution YAMLs
local. Resolve and verify the training YAML's model path, dataset/replay roots,
native-ledger absolute paths and writable output root. A copied dataset receipt
does not by itself establish that every referenced native ledger is readable.

On 10.249.189.84 the copied input mirror initially lives at
`/home/wangtianyi/rl-migration-inputs/VM/wangtianyi/DyConGlobal`. Source artifacts
may retain `/VM/...` absolute references, so changing only dataset/replay roots
does not relocate all inputs. Arrange the required mount/path mapping with the
administrator, or use an explicitly supported artifact relocation workflow;
do not rewrite frozen receipts or create mappings over existing directories.

For a new training YAML, inspect current `lora-train --help`, enable the
documented data/resource preflight checks and run the formal dry-run first.
Confirm the checkpoint/start mode, fresh attempt ID, GPU availability and
physical output location before an explicitly authorized real launch. Dry-run
does not prove a successful GPU optimizer update. See [training](TRAINING.md),
[GRPO](GRPO.md) and [streaming resume](STREAMING_RESUME.md).

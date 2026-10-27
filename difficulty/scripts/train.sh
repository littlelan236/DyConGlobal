#!/usr/bin/env bash
# A subshell keeps source from changing the caller's cwd, flags, environment or traps.
(
    set -e
    task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
    training_config="${1:-difficulty/training/config.yaml}"
    cd -- "$task_root"
    exec "$task_root/.venvs/analysis/bin/python" -m core.cli difficulty-train --training-config "$training_config"
)

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
UV="${UV_BIN:-$HOME/.local/bin/uv}"
"$UV" venv --python 3.12 --allow-existing .venvs/runner
LOCK="locks/runner.in"
if [ -f locks/runner.txt ]; then LOCK="locks/runner.txt"; fi
"$UV" pip install --python .venvs/runner/bin/python -r "$LOCK"
"$UV" pip install --python .venvs/runner/bin/python --no-deps --no-build-isolation -e .

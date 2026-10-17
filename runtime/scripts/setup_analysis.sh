#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
UV="${UV_BIN:-$HOME/.local/bin/uv}"
"$UV" venv --python 3.12 --allow-existing .venvs/analysis
LOCK="locks/analysis.in"
if [ -f locks/analysis.txt ]; then LOCK="locks/analysis.txt"; fi
"$UV" pip install --python .venvs/analysis/bin/python -r "$LOCK"
"$UV" pip install --python .venvs/analysis/bin/python --no-deps --no-build-isolation -e .

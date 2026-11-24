#!/usr/bin/env bash
# Shared installer only; training continues to use the formal dycon CLI.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

usage() {
  echo 'Usage: setup_runner.sh | setup_rl.sh [--with-runner]'
  echo 'Requires uv and flock. Uses .python-version and checked-in dependency locks.'
}

MODE="${1:-}"
if [ "$MODE" != runner ] && [ "$MODE" != rl ]; then usage >&2; exit 2; fi
shift
ENVIRONMENTS=("$MODE")
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then usage; exit 0; fi
if [ "$MODE" = rl ] && [ "${1:-}" = --with-runner ]; then
  ENVIRONMENTS=(runner rl)
  shift
fi
if [ "$#" -ne 0 ]; then usage >&2; exit 2; fi

UV="${UV_BIN:-$HOME/.local/bin/uv}"
if [ -z "${UV_BIN:-}" ] && [ ! -x "$UV" ]; then UV=uv; fi
command -v "$UV" >/dev/null || { echo 'uv is required; install it before setup.' >&2; exit 1; }
command -v flock >/dev/null || { echo 'flock (util-linux) is required.' >&2; exit 1; }
PYTHON_VERSION="$(tr -d '\r\n' < .python-version)"
export UV_LOCK_TIMEOUT="${UV_LOCK_TIMEOUT:-900}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-90}"

# Serialize these installers across checkouts sharing a user's download cache.
# Other uv callers still use uv's distribution-cache locks.
SETUP_LOCK_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/dycon"
mkdir -p "$SETUP_LOCK_DIR"
exec 9>"$SETUP_LOCK_DIR/python-environment-setup.lock"
if ! flock -w "$UV_LOCK_TIMEOUT" 9; then
  echo "Another environment setup holds $SETUP_LOCK_DIR/python-environment-setup.lock; retry after it finishes." >&2
  exit 1
fi

for ENVIRONMENT in "${ENVIRONMENTS[@]}"; do
  LOCK="locks/$ENVIRONMENT.txt"
  test -f "$LOCK" || { echo "Missing dependency lock: $LOCK" >&2; exit 1; }
  PYTHON=".venvs/$ENVIRONMENT/bin/python"
  # Never replace an existing environment with a different Python in place.
  if [ -x "$PYTHON" ]; then
    INSTALLED_VERSION="$("$PYTHON" -c 'import platform; print(platform.python_version())')"
    if [ "$INSTALLED_VERSION" != "$PYTHON_VERSION" ]; then
      echo "$PYTHON uses $INSTALLED_VERSION; expected $PYTHON_VERSION. Create a new environment explicitly." >&2
      exit 1
    fi
  else
    "$UV" venv --python "$PYTHON_VERSION" ".venvs/$ENVIRONMENT"
  fi
  INSTALL_OPTIONS=()
  if [ "$ENVIRONMENT" = rl ]; then INSTALL_OPTIONS+=(--require-hashes); fi
  "$UV" pip install --python "$PYTHON" "${INSTALL_OPTIONS[@]}" -r "$LOCK"
  # A root egg-info from a previous editable install can shadow fresh dist-info.
  # Refresh it through the locked build backend before registering this checkout.
  "$PYTHON" -c 'from setuptools import setup; setup()' egg_info
  "$UV" pip install --python "$PYTHON" --no-deps --no-build-isolation -e .
  "$UV" pip check --python "$PYTHON"
  "$PYTHON" -c 'import ijson; assert ijson.__version__ == "3.5.1"'
  ".venvs/$ENVIRONMENT/bin/dycon" lora-train --help >/dev/null
  echo "$ENVIRONMENT environment ready at $ROOT/.venvs/$ENVIRONMENT"
done

#!/bin/bash
test "${UV_OFFLINE:-}" = 1 || { echo "Frozen offline verifier resources required" >&2; exit 86; }
# Install dependencies
command -v curl >/dev/null || exit 86
command -v pdftotext >/dev/null || exit 86

# Install uv
source $HOME/.local/bin/env

# Ensure logs directory exists
mkdir -p /logs/verifier

# Run pytest
uvx \
  --with pytest==8.4.1 \
  --with pytest-json-ctrf==0.3.5 \
  pytest --ctrf /logs/verifier/ctrf.json /verifier/test_outputs.py -rA -v

PYTEST_EXIT_CODE=$?

if [ $PYTEST_EXIT_CODE -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi

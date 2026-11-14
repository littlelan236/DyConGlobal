#!/bin/bash
test "${UV_OFFLINE:-}" = 1 || { echo "Frozen offline verifier resources required" >&2; exit 86; }
# pytest is pre-installed in Dockerfile
mkdir -p /logs/verifier
pytest --ctrf /logs/verifier/ctrf.json /verifier/test_outputs.py -rA -v
if [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
exit 0

#!/usr/bin/env bash
# One-shot reproduction: tests, then the full pipeline.
set -euo pipefail
PY="${PYTHON:-python}"
"$PY" -m unittest discover -s tests -t .
"$PY" -m razerpay_fraud demo --out out

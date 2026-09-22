#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON=${PYTHON:-python}

cd "$ROOT/miner/pearl-gemm"

export PEARL_SM120_BACKEND=1
export PEARL_GEMM_FORCE_BUILD=1

echo "Rebuilding Pearl GEMM + SM120 extension from $(git -C "$ROOT" rev-parse --short HEAD)"
"$PYTHON" -m pip install -e . --no-build-isolation

echo "Running SM120 extension smoke test"
export PEARL_SM120_FUSED=1
export PEARL_SM120_SEARCH_ONLY=1
"$PYTHON" tests/sm120_extension_smoke.py

echo "SM120 v08 extension rebuild and smoke test passed."

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

export PEARL_SM120_BACKEND=1
export PEARL_GEMM_FORCE_BUILD=1

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required for this Pearl workspace but was not found in PATH." >&2
  echo "The rebuild does not require pip." >&2
  exit 2
fi

cd "$ROOT"

echo "Syncing Pearl's uv workspace"
uv sync --package pearl-gemm --package pearl-gemm-build-utils

VENV_PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "ERROR: uv sync did not create $VENV_PYTHON" >&2
  exit 2
fi

echo "Rebuilding Pearl GEMM + SM120 extension from $(git rev-parse --short HEAD)"
uv pip install   --python "$VENV_PYTHON"   --reinstall   --no-build-isolation   --no-deps   -e "$ROOT/miner/pearl-gemm"

echo "Running SM120 extension smoke test"
export PEARL_SM120_FUSED=1
export PEARL_SM120_SEARCH_ONLY=1
export PEARL_SM120_WMMA=1
"$VENV_PYTHON" "$ROOT/miner/pearl-gemm/tests/sm120_extension_smoke.py"

echo "SM120 v08 extension rebuild and smoke test passed."

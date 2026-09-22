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

echo "Syncing Pearl's vLLM miner workspace package"
# vllm-miner owns the vllm.general_plugins entry point that registers
# quantization='pearl'. It depends on pearl-gemm, so syncing it installs both
# the runtime plugin and the CUDA package instead of leaving stock vLLM unable
# to recognize the Pearl quantization method.
uv sync --package vllm-miner

VENV_PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "ERROR: uv sync did not create $VENV_PYTHON" >&2
  exit 2
fi

echo "Rebuilding Pearl GEMM + SM120 extension from $(git rev-parse --short HEAD)"
uv pip install   --python "$VENV_PYTHON"   --reinstall   --no-build-isolation   --no-deps   -e "$ROOT/miner/pearl-gemm"

echo "Verifying Pearl vLLM plugin is installed"
"$VENV_PYTHON" - <<'PY'
from importlib.metadata import entry_points

eps = [
    ep
    for ep in entry_points(group="vllm.general_plugins")
    if ep.name == "register_pearl_mining_plugin"
]
if len(eps) != 1:
    raise SystemExit(
        "ERROR: Pearl vLLM plugin entry point is missing; "
        f"found {len(eps)} matching entries"
    )
ep = eps[0]
print(f"Pearl vLLM plugin: {ep.name} -> {ep.value}")
PY

echo "Running SM120 extension smoke test"
export PEARL_SM120_FUSED=1
export PEARL_SM120_SEARCH_ONLY=1
export PEARL_SM120_WMMA=1
"$VENV_PYTHON" "$ROOT/miner/pearl-gemm/tests/sm120_extension_smoke.py"

echo "SM120 v08 extension rebuild and smoke test passed."

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
VENV_PYTHON="$ROOT/.venv/bin/python"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required for the Pearl workspace." >&2
  exit 2
fi
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "ERROR: missing workspace interpreter: $VENV_PYTHON" >&2
  exit 2
fi

cd "$ROOT"

# Phase 1: restore the historical Pearl extension exactly through its normal
# build path. Do NOT let SM120 feature flags leak into this build.
unset PEARL_SM120_BACKEND || true
unset PEARL_SM120_ONLY || true
unset PEARL_SM120_FUSED || true
unset PEARL_SM120_SEARCH_ONLY || true
unset PEARL_SM120_WMMA || true
unset PEARL_SM120_QUANT_CUTE || true
export PEARL_GEMM_FORCE_BUILD=1

echo "[1/5] Syncing the original vLLM/Pearl workspace"
uv sync --package vllm-miner

echo "[2/5] Restoring the historical pearl_gemm_cuda extension"
uv pip install \
  --python "$VENV_PYTHON" \
  --reinstall \
  --no-build-isolation \
  --no-deps \
  -e "$ROOT/miner/pearl-gemm"

readarray -t LEGACY_INFO < <("$VENV_PYTHON" - <<'PY'
import hashlib
import pearl_gemm_cuda
path = pearl_gemm_cuda.__file__
with open(path, "rb") as f:
    digest = hashlib.sha256(f.read()).hexdigest()
print(path)
print(digest)
PY
)
LEGACY_PATH="${LEGACY_INFO[0]}"
LEGACY_SHA="${LEGACY_INFO[1]}"
echo "historical extension: $LEGACY_PATH"
echo "historical sha256:    $LEGACY_SHA"

echo "[3/5] Verifying Pearl vLLM plugin registration"
"$VENV_PYTHON" - <<'PY'
from importlib.metadata import entry_points
eps = [
    ep for ep in entry_points(group="vllm.general_plugins")
    if ep.name == "register_pearl_mining_plugin"
]
if len(eps) != 1:
    raise SystemExit(
        "ERROR: expected exactly one Pearl vLLM plugin entry point; "
        f"found {len(eps)}"
    )
print(f"Pearl plugin: {eps[0].name} -> {eps[0].value}")
PY

# Phase 2: reproduce the uploaded working topology. Build ONLY the separate
# pearl_sm120_cuda extension. PEARL_SM120_ONLY exists specifically so this
# step cannot rebuild or replace pearl_gemm_cuda.
echo "[4/5] Building only the separate SM120 backend"
SITE_PACKAGES=$("$VENV_PYTHON" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["platlib"])
PY
)

cd "$ROOT/miner/pearl-gemm"
PEARL_SM120_BACKEND=1 \
PEARL_SM120_ONLY=1 \
PEARL_GEMM_FORCE_BUILD=1 \
MAX_JOBS="${MAX_JOBS:-1}" \
"$VENV_PYTHON" setup.py build_ext --build-lib "$SITE_PACKAGES"

# Prove the custom build did not mutate the historical extension.
readarray -t LEGACY_AFTER < <("$VENV_PYTHON" - <<'PY'
import hashlib
import pearl_gemm_cuda
path = pearl_gemm_cuda.__file__
with open(path, "rb") as f:
    digest = hashlib.sha256(f.read()).hexdigest()
print(path)
print(digest)
PY
)
if [[ "${LEGACY_AFTER[0]}" != "$LEGACY_PATH" || "${LEGACY_AFTER[1]}" != "$LEGACY_SHA" ]]; then
  echo "ERROR: SM120-only build modified the historical Pearl extension." >&2
  echo "before: $LEGACY_PATH $LEGACY_SHA" >&2
  echo "after:  ${LEGACY_AFTER[0]} ${LEGACY_AFTER[1]}" >&2
  exit 3
fi

echo "[5/5] Running the original SM120 extension smoke test"
export PEARL_SM120_BACKEND=1
export PEARL_SM120_FUSED=1
export PEARL_SM120_SEARCH_ONLY=1
"$VENV_PYTHON" "$ROOT/miner/pearl-gemm/tests/sm120_extension_smoke.py"

echo "Aligned v08 build passed."
echo "Historical pearl_gemm_cuda remained unchanged during the SM120 build."

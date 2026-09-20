#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BUILD="$ROOT/build/sm120"
ARTIFACTS="$ROOT/artifacts/build"
mkdir -p "$BUILD" "$ARTIFACTS"

if ! command -v nvcc >/dev/null; then
  echo BLOCKER_TOOLCHAIN_CUDA_TOO_OLD >&2
  exit 2
fi
if ! git -C "$ROOT/miner/pearl-gemm/third_party/cutlass" describe --tags --exact-match 2>/dev/null | grep -qx 'v4.6.3'; then
  echo "BLOCKER_CUTLASS_4_6_3_REQUIRED" >&2
  exit 2
fi
CUDA_VERSION=$(nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\),.*/\1/p')
case "$CUDA_VERSION" in
  12.[89]*|1[3-9].*) ;;
  *) echo BLOCKER_TOOLCHAIN_CUDA_TOO_OLD >&2; exit 2 ;;
esac

cmake -S "$ROOT/miner/pearl-gemm/csrc/sm120" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build "$BUILD" --parallel
ctest --test-dir "$BUILD" --output-on-failure

{
  echo "pearl_revision=$(git -C "$ROOT" rev-parse HEAD)"
  echo "branch=$(git -C "$ROOT" branch --show-current)"
  echo "cuda=$CUDA_VERSION"
  echo "cmake=$(cmake --version | head -1)"
  echo "sm120_math_backend=branch_3_exact_integer_fallback"
  echo "cutlass_revision=$(git -C "$ROOT/miner/pearl-gemm/third_party/cutlass" rev-parse HEAD)"
  echo "cutlass_tag=$(git -C "$ROOT/miner/pearl-gemm/third_party/cutlass" describe --tags --exact-match)"
  echo "cuda_arch=sm_120"
} > "$ARTIFACTS/manifest.txt"
printf 'branch_3_exact_integer_fallback\n' > "$ARTIFACTS/sm120_math_backend.txt"

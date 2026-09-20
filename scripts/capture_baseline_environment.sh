#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT="$ROOT/artifacts/baseline/environment.txt"
mkdir -p "$(dirname "$OUT")"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "[pearl_git_status]"
  git -C "$ROOT" status --short
  echo "[pearl_revision]"
  git -C "$ROOT" rev-parse HEAD
  echo "[nvcc]"
  nvcc --version
  echo "[nvidia_smi]"
  nvidia-smi -q
  echo "[krig_command]"
  ps -eo pid=,args= | grep '[k]rig-miner' || true
} > "$OUT"

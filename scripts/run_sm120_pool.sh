#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
GATES="$ROOT/artifacts/results/release_gates.json"
if [ ! -f "$GATES" ] || ! grep -q '"release_enabled": true' "$GATES"; then
  echo "SM120 pool submission is disabled: Package 9 release gates are not proven." >&2
  exit 3
fi
if [ "${PEARL_SM120_BACKEND:-0}" != 1 ]; then
  echo "Set PEARL_SM120_BACKEND=1 only after the release gates are proven." >&2
  exit 3
fi
if [ -z "${PEARL_SM120_MINER_COMMAND:-}" ]; then
  echo "BLOCKER_OPERATOR_COMMAND_REQUIRED: set PEARL_SM120_MINER_COMMAND to the existing miner launch command." >&2
  exit 3
fi

# Credentials, node URLs, wallet configuration, and model arguments remain in the
# operator's existing environment/command.  This wrapper only adds the experimental
# backend flags and never creates or changes payout credentials.
export PEARL_SM120_FUSED=1
export PEARL_SM120_SEARCH_ONLY=1
export PEARL_SM120_EVENT_LOG="${PEARL_SM120_EVENT_LOG:-$ROOT/artifacts/run/miner_events.jsonl}"
exec bash -lc "$PEARL_SM120_MINER_COMMAND"

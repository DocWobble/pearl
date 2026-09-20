#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p "$ROOT/artifacts/results"
"$ROOT/scripts/build_sm120.sh"
if ! python3 "$ROOT/miner/tools/benchmark_sm120.py" --release-gates; then
  printf '%s\n' 'KRIG' > "$ROOT/artifacts/results/production_recommendation.txt"
  echo "KRIG: SM120 release gates are not proven; A/B schedule was not started."
  exit 0
fi

if [ -z "${KRIG_COMMAND:-}" ] || [ -z "${CUSTOM_COMMAND:-}" ]; then
  echo "BLOCKER_OPERATOR_COMMAND_REQUIRED: set KRIG_COMMAND and CUSTOM_COMMAND." >&2
  exit 3
fi

RUN_DIR="$ROOT/artifacts/run"
mkdir -p "$RUN_DIR"
schedule_file="$RUN_DIR/ab_schedule.jsonl"
: > "$schedule_file"

run_window() {
  local label="$1" command="$2" warmup="$3" measured="$4"
  local log="$RUN_DIR/${label}.log"
  local started ended pid
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  PEARL_SM120_EVENT_LOG="$RUN_DIR/${label}_events.jsonl" \
    timeout --signal=INT "$((warmup + measured))" bash -lc "$command" >"$log" 2>&1 &
  pid=$!
  sleep "$warmup"
  printf '{"event":"measurement_started","label":"%s","warmup_seconds":%s,"measured_seconds":%s,"wallclock_utc":"%s"}\n' "$label" "$warmup" "$measured" "$started" >> "$schedule_file"
  sleep "$measured"
  wait "$pid" || true
  ended=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '{"event":"measurement_finished","label":"%s","wallclock_utc":"%s"}\n' "$label" "$ended" >> "$schedule_file"
}

# The first window for each miner receives the specified five-minute warm-up;
# the second window reuses the same miner without adding an unrecorded interval.
run_window krig_1 "$KRIG_COMMAND" 300 1800
run_window custom_1 "$CUSTOM_COMMAND" 300 1800
run_window krig_2 "$KRIG_COMMAND" 0 1800
run_window custom_2 "$CUSTOM_COMMAND" 0 1800

python3 "$ROOT/miner/tools/compare_pool_work.py" \
  --krig "$RUN_DIR/krig_events.jsonl" \
  --custom "$RUN_DIR/custom_events.jsonl" \
  --output "$ROOT/artifacts/results/pool_work_comparison.json"

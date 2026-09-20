#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT="$ROOT/artifacts/baseline/krig"
mkdir -p "$OUT"
if ! pgrep -f '[k]rig-miner' >/dev/null; then
  echo "Krig is not running; this script only observes an existing working baseline." >&2
  exit 2
fi
curl -fsS http://127.0.0.1:12000/metrics > "$OUT/metrics_before_warmup.prom"
echo "Warm-up started: $(date --iso-8601=seconds)" > "$OUT/capture.txt"
sleep 300
curl -fsS http://127.0.0.1:12000/metrics > "$OUT/metrics_before_measurement.prom"
echo "Measurement started: $(date --iso-8601=seconds)" >> "$OUT/capture.txt"
for _ in $(seq 1 180); do
  date --iso-8601=seconds >> "$OUT/sample_times.txt"
  nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits >> "$OUT/gpu.csv"
  curl -fsS http://127.0.0.1:12000/metrics >> "$OUT/metrics_samples.prom"
  sleep 10
done
curl -fsS http://127.0.0.1:12000/metrics > "$OUT/metrics_after_measurement.prom"
echo "Measurement completed: $(date --iso-8601=seconds)" >> "$OUT/capture.txt"

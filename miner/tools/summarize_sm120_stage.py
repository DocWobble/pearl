#!/usr/bin/env python3
"""Summarize opt-in SM120 stage timing lines from a miner log."""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

STAGES = ("noising_ms", "seed_copy_ms", "search_ms", "inference_ms", "total_ms")


def parse(path: Path) -> dict[str, list[float]]:
    values = {stage: [] for stage in STAGES}
    for line in path.read_text(errors="replace").splitlines():
        if "SM120_STAGE " not in line:
            continue
        fields = {
            token.split("=", 1)[0]: token.split("=", 1)[1]
            for token in line.split()
            if "=" in token
        }
        for stage in STAGES:
            if stage in fields:
                values[stage].append(float(fields[stage]))
    return values


def percentile(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    if not ys:
        return 0.0
    idx = min(len(ys) - 1, max(0, round((len(ys) - 1) * p)))
    return ys[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    values = parse(args.log)
    samples = len(values["total_ms"])
    if not samples:
        raise SystemExit("no SM120_STAGE samples found")

    print(f"samples: {samples}")
    print("| stage | median ms | mean ms | p90 ms | median share |")
    print("|---|---:|---:|---:|---:|")
    total_med = statistics.median(values["total_ms"])
    for stage in STAGES:
        xs = values[stage]
        med = statistics.median(xs)
        share = (med / total_med * 100.0) if total_med else 0.0
        print(
            f"| {stage} | {med:.3f} | {statistics.fmean(xs):.3f} | "
            f"{percentile(xs, 0.90):.3f} | {share:.1f}% |"
        )


if __name__ == "__main__":
    main()

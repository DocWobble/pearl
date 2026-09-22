#!/usr/bin/env python3
"""Compare custom SM120 and Krig hashrate in the same PearlHash H/s unit."""
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

KRIG_RE = re.compile(
    r'^krig_miner_hashes_per_second\{gpu="[^"]+"\}\s+([0-9.eE+-]+)\s*$'
)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def krig_samples(path: Path) -> list[float]:
    values: list[float] = []
    for line in path.read_text(errors="replace").splitlines():
        match = KRIG_RE.match(line.strip())
        if match:
            values.append(float(match.group(1)) / 1e12)
    return values


def custom_samples(path: Path) -> list[float]:
    values: list[float] = []
    for line in path.read_text(errors="replace").splitlines():
        if "SM120_RATE " not in line:
            continue
        fields = dict(
            token.split("=", 1)
            for token in line.split()
            if "=" in token
        )
        raw = fields.get("rolling_ths")
        if raw is None or float(raw) <= 0:
            raw = fields.get("hashrate_ths")
        if raw is not None and float(raw) > 0:
            values.append(float(raw))
    return values


def summarize(name: str, values: list[float]) -> dict[str, float | int | str]:
    if not values:
        raise SystemExit(f"no {name} hashrate samples found")
    return {
        "miner": name,
        "samples": len(values),
        "median_ths": statistics.median(values),
        "mean_ths": statistics.fmean(values),
        "p90_ths": percentile(values, 0.90),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--krig-prom", type=Path, required=True)
    parser.add_argument("--custom-log", type=Path, required=True)
    args = parser.parse_args()

    krig = summarize("krig", krig_samples(args.krig_prom))
    custom = summarize("custom", custom_samples(args.custom_log))
    ratio = float(custom["median_ths"]) / float(krig["median_ths"])

    print("| miner | samples | median TH/s | mean TH/s | p90 TH/s |")
    print("|---|---:|---:|---:|---:|")
    for row in (krig, custom):
        print(
            f"| {row['miner']} | {row['samples']} | {row['median_ths']:.6f} | "
            f"{row['mean_ths']:.6f} | {row['p90_ths']:.6f} |"
        )
    print(f"custom/krig median hashrate ratio: {ratio:.6f}x")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark exact SM120 search kernels on a geometry observed in a miner log."""
from __future__ import annotations

import argparse
import os
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import torch

import pearl_sm120_cuda

GEOM_RE = re.compile(r"SM120_STAGE .*?m=(\d+) n=(\d+) k=(\d+)")


def geometry_from_log(path: Path) -> tuple[int, int, int]:
    counts: Counter[tuple[int, int, int]] = Counter()
    for line in path.read_text(errors="replace").splitlines():
        match = GEOM_RE.search(line)
        if match:
            counts[tuple(map(int, match.groups()))] += 1
    if not counts:
        raise SystemExit(f"no SM120_STAGE geometry found in {path}")
    geometry, count = counts.most_common(1)[0]
    print(f"benchmark geometry from log: m={geometry[0]} n={geometry[1]} k={geometry[2]} samples={count}")
    return geometry


def bench(mode: str, a: torch.Tensor, bt: torch.Tensor, key: torch.Tensor,
          target: torch.Tensor, m: int, n: int, k: int, iterations: int) -> tuple[float, float]:
    os.environ["PEARL_SM120_KERNEL"] = mode
    for _ in range(2):
        result = pearl_sm120_cuda.search(a, bt, key, target, m, n, k, 1)
        if result["cuda_errors"]:
            raise RuntimeError(f"{mode} warmup failed: {result}")
    torch.cuda.synchronize()

    samples: list[float] = []
    expected_work = m * n * k
    for _ in range(iterations):
        start = time.perf_counter()
        result = pearl_sm120_cuda.search(a, bt, key, target, m, n, k, 1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if result["cuda_errors"]:
            raise RuntimeError(f"{mode} failed: {result}")
        if int(result["valid_candidate_work"]) != expected_work:
            raise RuntimeError(
                f"{mode} work mismatch: {result['valid_candidate_work']} != {expected_work}"
            )
        samples.append(elapsed)

    median_s = statistics.median(samples)
    ths = expected_work / median_s / 1e12
    return median_s * 1000.0, ths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    os.environ["PEARL_SM120_BACKEND"] = "1"
    os.environ["PEARL_SM120_FUSED"] = "1"
    os.environ["PEARL_SM120_SEARCH_ONLY"] = "1"

    m, n, k = geometry_from_log(args.log)
    if m % 16 or n % 16 or k % 128:
        raise SystemExit(f"noncanonical geometry: {(m, n, k)}")

    a = torch.zeros((1, m, k), device="cuda", dtype=torch.int8)
    bt = torch.zeros((n, k), device="cuda", dtype=torch.int8)
    key = torch.arange(32, dtype=torch.uint8)
    target = torch.zeros(32, dtype=torch.uint8)

    rows: list[tuple[str, float, float]] = []
    for mode in ("scalar", "dp4a", "wmma"):
        ms, ths = bench(mode, a, bt, key, target, m, n, k, args.iterations)
        rows.append((mode, ms, ths))

    print("| kernel | median search ms | search-only TH/s |")
    print("|---|---:|---:|")
    for mode, ms, ths in rows:
        print(f"| {mode} | {ms:.3f} | {ths:.6f} |")

    best = min(rows, key=lambda row: row[1])
    scalar_ms = next(ms for mode, ms, _ in rows if mode == "scalar")
    print(f"BEST_KERNEL={best[0]}")
    print(f"BEST_SPEEDUP_VS_SCALAR={scalar_ms / best[1]:.3f}x")


if __name__ == "__main__":
    main()

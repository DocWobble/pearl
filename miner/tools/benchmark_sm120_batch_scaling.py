#!/usr/bin/env python3
"""Measure SM120 work granularity across candidate-bank sizes.

This is a throughput microbenchmark, not a protocol-valid multi-candidate miner.
It feeds distinct synthetic A buffers through the exact search kernel so we can
measure whether larger candidate banks improve normalized work/second or merely
make each launch proportionally longer.

The result answers the v12 handoff question:
    normalized Pearl work / main launch
versus
    main-launch duration

Run only after the usual SM120 parity gates have passed.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import statistics
import time
from pathlib import Path

import torch

import pearl_sm120_cuda


GEOM_RE = re.compile(r"SM120_STAGE .*?m=(\d+) n=(\d+) k=(\d+)")
DEFAULT_KRIG_WORK_PER_LAUNCH = 1.483e12


def geometry_from_log(path: Path) -> tuple[int, int, int]:
    counts: dict[tuple[int, int, int], int] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = GEOM_RE.search(line)
        if not match:
            continue
        geometry = tuple(map(int, match.groups()))
        counts[geometry] = counts.get(geometry, 0) + 1
    if not counts:
        raise SystemExit(f"no SM120_STAGE geometry found in {path}")
    geometry, count = max(counts.items(), key=lambda item: item[1])
    print(
        "benchmark geometry from log: "
        f"m={geometry[0]} n={geometry[1]} k={geometry[2]} samples={count}"
    )
    return geometry


def parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def parse_modes(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    allowed = {
        "scalar",
        "dp4a",
        "wmma",
        "wmma64x64",
        "wmma64x128",
        "wmma128x64",
    }
    if not result or any(mode not in allowed for mode in result):
        raise argparse.ArgumentTypeError(
            "expected comma-separated modes from: " + ",".join(sorted(allowed))
        )
    return result


def mib(num_bytes: int) -> float:
    return num_bytes / (1024.0 * 1024.0)


def bench(
    mode: str,
    batch: int,
    bt: torch.Tensor,
    key: torch.Tensor,
    target: torch.Tensor,
    m: int,
    n: int,
    k: int,
    iterations: int,
    warmup: int,
) -> dict[str, float | int | str]:
    os.environ["PEARL_SM120_KERNEL"] = mode

    # Distinct rows avoid accidentally benchmarking a single expanded view.
    # Values do not need protocol-valid seed provenance for this kernel-level
    # throughput experiment; canonical parity is gated separately.
    a = torch.empty((batch, m, k), device="cuda", dtype=torch.int8)
    a.random_(-128, 128)

    expected_work = batch * m * n * k
    target_tests = batch * (m // 16) * (n // 16)

    for _ in range(warmup):
        result = pearl_sm120_cuda.search(a, bt, key, target, m, n, k, 1)
        if result["cuda_errors"]:
            raise RuntimeError(f"{mode} batch={batch} warmup failed: {result}")
        if int(result["valid_candidate_work"]) != expected_work:
            raise RuntimeError(
                f"{mode} batch={batch} work mismatch: "
                f"{result['valid_candidate_work']} != {expected_work}"
            )

    torch.cuda.synchronize()
    samples: list[float] = []
    cache_hits: list[int] = []
    cache_misses: list[int] = []

    for _ in range(iterations):
        start = time.perf_counter()
        result = pearl_sm120_cuda.search(a, bt, key, target, m, n, k, 1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if result["cuda_errors"]:
            raise RuntimeError(f"{mode} batch={batch} failed: {result}")
        if int(result["valid_candidate_work"]) != expected_work:
            raise RuntimeError(
                f"{mode} batch={batch} work mismatch: "
                f"{result['valid_candidate_work']} != {expected_work}"
            )

        samples.append(elapsed)
        cache_hits.append(int(result["b_cache_hits"]))
        cache_misses.append(int(result["b_cache_misses"]))

    median_s = statistics.median(samples)
    return {
        "mode": mode,
        "batch": batch,
        "median_ms": median_s * 1000.0,
        "work": expected_work,
        "target_tests": target_tests,
        "ths": expected_work / median_s / 1e12,
        "ns_per_target": median_s * 1e9 / target_tests,
        "a_mib": mib(batch * m * k),
        "cache_hits": sum(cache_hits),
        "cache_misses": sum(cache_misses),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--batches", type=parse_ints, default=parse_ints("1,2,4,8,16"))
    parser.add_argument(
        "--modes",
        type=parse_modes,
        default=parse_modes("wmma,wmma64x64,wmma64x128,wmma128x64"),
    )
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--krig-work-per-launch",
        type=float,
        default=DEFAULT_KRIG_WORK_PER_LAUNCH,
        help="inferred/observed normalized Krig work per dominant launch",
    )
    parser.add_argument(
        "--max-free-vram-fraction",
        type=float,
        default=0.70,
        help="skip A banks that alone exceed this fraction of currently free VRAM",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise SystemExit(f"SM120 GPU required, found capability={torch.cuda.get_device_capability(0)}")
    if args.iterations < 1 or args.warmup < 0:
        raise SystemExit("invalid iteration count")
    if not 0.05 <= args.max_free_vram_fraction <= 0.95:
        raise SystemExit("--max-free-vram-fraction must be in [0.05, 0.95]")

    os.environ["PEARL_SM120_BACKEND"] = "1"
    os.environ["PEARL_SM120_FUSED"] = "1"
    os.environ["PEARL_SM120_SEARCH_ONLY"] = "1"
    os.environ["PEARL_SM120_REUSE_ALLOC"] = "1"
    os.environ["PEARL_SM120_CACHE_B"] = "1"

    m, n, k = geometry_from_log(args.log)
    if m % 16 or n % 16 or k % 128:
        raise SystemExit(f"noncanonical geometry: {(m, n, k)}")

    work_per_candidate = m * n * k
    tests_per_candidate = (m // 16) * (n // 16)
    krig_equivalent_batch = max(1, math.ceil(args.krig_work_per_launch / work_per_candidate))

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    bt_bytes = n * k
    max_a_bytes = int(free_bytes * args.max_free_vram_fraction)

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"free_vram_mib={mib(free_bytes):.1f} total_vram_mib={mib(total_bytes):.1f}")
    print(f"work_per_candidate={work_per_candidate}")
    print(f"target_tests_per_candidate={tests_per_candidate}")
    print(f"b_transposed_mib={mib(bt_bytes):.1f}")
    print(f"krig_reference_work_per_launch={args.krig_work_per_launch:.0f}")
    print(f"krig_equivalent_candidate_batch={krig_equivalent_batch}")
    print(
        "krig_equivalent_a_bank_mib="
        f"{mib(krig_equivalent_batch * m * k):.1f}"
    )

    bt = torch.zeros((n, k), device="cuda", dtype=torch.int8)
    key = torch.arange(32, dtype=torch.uint8)
    # All-zero target makes winners effectively impossible, keeping the
    # measured path on the normal losing-search fast path.
    target = torch.zeros(32, dtype=torch.uint8)

    rows: list[dict[str, float | int | str]] = []
    for mode in args.modes:
        for batch in args.batches:
            a_bytes = batch * m * k
            if a_bytes > max_a_bytes:
                print(
                    f"SKIP mode={mode} batch={batch}: "
                    f"A bank {mib(a_bytes):.1f} MiB exceeds "
                    f"{args.max_free_vram_fraction:.0%} of free VRAM"
                )
                continue
            try:
                row = bench(
                    mode,
                    batch,
                    bt,
                    key,
                    target,
                    m,
                    n,
                    k,
                    args.iterations,
                    args.warmup,
                )
            except RuntimeError as exc:
                print(f"FAIL mode={mode} batch={batch}: {exc}")
                continue
            rows.append(row)
            torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("no benchmark rows completed")

    baseline: dict[str, float] = {}
    for row in rows:
        mode = str(row["mode"])
        if int(row["batch"]) == 1:
            baseline[mode] = float(row["ths"])

    print()
    print("| kernel | batch | median ms | normalized work/launch | target tests/launch | TH/s | vs batch-1 TH/s | ns/target | A MiB | B-cache h/m |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        mode = str(row["mode"])
        ths = float(row["ths"])
        rel = ths / baseline[mode] if mode in baseline else float("nan")
        print(
            f"| {mode} | {int(row['batch'])} | {float(row['median_ms']):.3f} | "
            f"{int(row['work'])} | {int(row['target_tests'])} | {ths:.6f} | "
            f"{rel:.3f}x | {float(row['ns_per_target']):.3f} | "
            f"{float(row['a_mib']):.1f} | "
            f"{int(row['cache_hits'])}/{int(row['cache_misses'])} |"
        )

    best = max(rows, key=lambda row: float(row["ths"]))
    print()
    print(f"BEST_MODE={best['mode']}")
    print(f"BEST_BATCH={best['batch']}")
    print(f"BEST_THS={float(best['ths']):.6f}")
    print(f"BEST_MEDIAN_MS={float(best['median_ms']):.3f}")
    print(f"BEST_WORK_PER_LAUNCH={best['work']}")
    print(f"BEST_TARGET_TESTS_PER_LAUNCH={best['target_tests']}")

    same_mode = [row for row in rows if row["mode"] == best["mode"]]
    one = next((row for row in same_mode if int(row["batch"]) == 1), None)
    if one is not None:
        print(
            "BEST_MODE_THS_GAIN_VS_BATCH1="
            f"{float(best['ths']) / float(one['ths']):.3f}x"
        )


if __name__ == "__main__":
    main()

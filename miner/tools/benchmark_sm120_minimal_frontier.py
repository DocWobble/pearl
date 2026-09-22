#!/usr/bin/env python3
"""Measure the exact Pearl SM120 search frontier with all non-search machinery removed.

This benchmark constructs one fixed canonical noise field, then repeatedly searches
all disjoint 16x16 tiles.  Each tile is one exact Pearl jackpot evaluation.  The
headline rate is therefore the same normalized work unit used by Krig:

    valid_work = tile_count * 16 * 16 * K = M * N * K

The purpose is not to claim pool performance.  It answers a narrower question:
can the current exact search circuit itself outrun the best measured same-card
Krig rate once vLLM, inference output, host noising, and per-candidate setup are
removed?
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import pearl_sm120_cuda

KRIG_MAX_THS = 181.92


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def one_case(side: int, k: int, iterations: int, warmups: int) -> dict[str, float | int]:
    m = n = side
    device = torch.device("cuda")

    a_seed = torch.arange(32, dtype=torch.uint8)
    b_seed = torch.arange(31, -1, -1, dtype=torch.uint8)
    target = torch.zeros(32, dtype=torch.uint8)
    a_rows = torch.arange(m, device=device, dtype=torch.int32)
    b_cols = torch.arange(n, device=device, dtype=torch.int32)

    prep_start = time.perf_counter()
    a_noise, b_noise_t = pearl_sm120_cuda.noise(
        a_seed, b_seed, a_rows, b_cols, k, 128
    )
    torch.cuda.synchronize()
    prep_ms = (time.perf_counter() - prep_start) * 1000.0

    expected_work = m * n * k
    expected_tiles = (m // 16) * (n // 16)

    for _ in range(warmups):
        result = pearl_sm120_cuda.search(
            a_noise.unsqueeze(0), b_noise_t, a_seed, target, m, n, k, 1
        )
        if result["cuda_errors"]:
            raise RuntimeError(f"warmup failed side={side} k={k}: {result}")

    samples_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = pearl_sm120_cuda.search(
            a_noise.unsqueeze(0), b_noise_t, a_seed, target, m, n, k, 1
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if result["cuda_errors"]:
            raise RuntimeError(f"search failed side={side} k={k}: {result}")
        if int(result["valid_candidate_work"]) != expected_work:
            raise RuntimeError(
                f"work mismatch side={side} k={k}: "
                f"{result['valid_candidate_work']} != {expected_work}"
            )
        samples_ms.append(elapsed_ms)

    med_ms = statistics.median(samples_ms)
    med_ths = expected_work / (med_ms / 1000.0) / 1e12
    p90_ms = pct(samples_ms, 0.90)
    p90_ths = expected_work / (p90_ms / 1000.0) / 1e12
    tile_rate = expected_tiles / (med_ms / 1000.0)

    del a_noise, b_noise_t, a_rows, b_cols
    torch.cuda.empty_cache()

    return {
        "side": side,
        "k": k,
        "tiles": expected_tiles,
        "work": expected_work,
        "prep_ms": prep_ms,
        "median_ms": med_ms,
        "p90_ms": p90_ms,
        "median_ths": med_ths,
        "p90_ths": p90_ths,
        "tiles_s": tile_rate,
        "krig_ratio": med_ths / KRIG_MAX_THS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sides", default="256,512,1024,2048,4096")
    parser.add_argument("--k", default="2048,4096,8192,16384,32768,65536")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--kernel", default="wmma", choices=("wmma", "dp4a", "scalar"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise SystemExit(f"SM120 GPU required, got {torch.cuda.get_device_capability(0)}")

    os.environ["PEARL_SM120_BACKEND"] = "1"
    os.environ["PEARL_SM120_FUSED"] = "1"
    os.environ["PEARL_SM120_SEARCH_ONLY"] = "1"
    os.environ["PEARL_SM120_REUSE_ALLOC"] = "1"
    os.environ["PEARL_SM120_KERNEL"] = args.kernel

    sides = [int(x) for x in args.sides.split(",") if x.strip()]
    ks = [int(x) for x in args.k.split(",") if x.strip()]
    for side in sides:
        if side < 16 or side % 16:
            raise SystemExit(f"side must be a positive multiple of 16: {side}")
    for k in ks:
        if k < 2048 or k > 65536 or k % 128:
            raise SystemExit(f"K must be canonical rank-128 geometry: {k}")

    props = torch.cuda.get_device_properties(0)
    print(f"GPU={props.name}")
    print(f"KERNEL={args.kernel}")
    print(f"KRIG_MAX_THS={KRIG_MAX_THS:.2f}")
    print()

    rows: list[dict[str, float | int]] = []
    free_bytes, _ = torch.cuda.mem_get_info()
    reserve = 2 * 1024**3

    for k in ks:
        for side in sides:
            operand_bytes = 2 * side * k
            if operand_bytes + reserve > free_bytes:
                print(
                    f"SKIP side={side} k={k}: operands={operand_bytes / 2**30:.2f} GiB "
                    f"would leave <2 GiB free"
                )
                continue
            row = one_case(side, k, args.iterations, args.warmups)
            rows.append(row)
            print(
                f"side={side:4d} k={k:5d} tiles={row['tiles']:7d} "
                f"prep={row['prep_ms']:8.3f} ms search={row['median_ms']:8.3f} ms "
                f"TH/s={row['median_ths']:9.3f} Krig={row['krig_ratio']:7.3f}x"
            )

    if not rows:
        raise SystemExit("no benchmark cases ran")

    rows.sort(key=lambda r: float(r["median_ths"]), reverse=True)
    best = rows[0]

    print()
    print("| side | K | tiles/search | noise prep ms | search median ms | p90 ms | median TH/s | vs Krig MAX |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['side']} | {row['k']} | {row['tiles']} | "
            f"{row['prep_ms']:.3f} | {row['median_ms']:.3f} | {row['p90_ms']:.3f} | "
            f"{row['median_ths']:.3f} | {row['krig_ratio']:.4f}x |"
        )

    print()
    print(
        "BEST "
        f"side={best['side']} k={best['k']} "
        f"search_ms={best['median_ms']:.3f} "
        f"ths={best['median_ths']:.6f} "
        f"krig_ratio={best['krig_ratio']:.6f}"
    )
    if float(best["median_ths"]) > KRIG_MAX_THS:
        print("FRONTIER_RESULT=SEARCH_CIRCUIT_BEATS_KRIG")
    else:
        print("FRONTIER_RESULT=SEARCH_CIRCUIT_BELOW_KRIG")


if __name__ == "__main__":
    main()

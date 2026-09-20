#!/usr/bin/env python3
"""Run the local SM120 search path for the handoff's stress-gate duration."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

import pearl_sm120_cuda


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=30 * 60)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/sm120_stress.json"))
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[2] / args.output
    if not torch.cuda.is_available():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "requested_seconds": args.seconds,
            "status": "blocked",
            "gate_pass": False,
            "error": "CUDA GPU required for SM120 stress gate",
        }, indent=2) + "\n")
        raise SystemExit("CUDA GPU required for SM120 stress gate")
    os.environ["PEARL_SM120_BACKEND"] = "1"
    os.environ["PEARL_SM120_FUSED"] = "1"
    os.environ["PEARL_SM120_SEARCH_ONLY"] = "1"
    a = torch.randint(-64, 65, (args.batch, 128, 2048), device="cuda", dtype=torch.int8)
    b = torch.randint(-64, 65, (128, 2048), device="cuda", dtype=torch.int8)
    key = torch.arange(32, dtype=torch.uint8)
    target = torch.zeros(32, dtype=torch.uint8)
    started = time.monotonic()
    calls = 0
    errors = 0
    overflow = False
    winners = 0
    while time.monotonic() - started < args.seconds:
        result = pearl_sm120_cuda.search(a, b, key, target, 128, 128, 2048, 1)
        calls += 1
        errors += int(result["cuda_errors"])
        overflow = overflow or bool(result["winner_overflow"])
        winners += int(result["winners"])
    torch.cuda.synchronize()
    report = {
        "requested_seconds": args.seconds, "elapsed_seconds": time.monotonic() - started,
        "batch": args.batch, "calls": calls, "cuda_errors": errors,
        "winner_overflow": overflow, "winners": winners,
        "gate_pass": errors == 0 and not overflow,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

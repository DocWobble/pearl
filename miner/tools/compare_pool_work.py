#!/usr/bin/env python3
"""Compare miner event logs using difficulty-normalized accepted work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TWO256 = 1 << 256


def _target(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.startswith("0x"):
            return int(text, 16)
        if len(text) == 64 and all(c in "0123456789abcdefABCDEF" for c in text):
            return int.from_bytes(bytes.fromhex(text), "little")
        return int(text, 0)
    except ValueError:
        return None


def _records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: event must be an object")
        records.append(value)
    return records


def summarize(path: Path) -> dict[str, Any]:
    records = _records(path)
    submissions = [r for r in records if r.get("event") in {"share_submitted", "share_response"}]
    accepted = stale = rejected = 0
    normalized_accepted = normalized_stale = 0.0
    expected_hash_work_accepted = expected_hash_work_stale = 0.0
    for record in submissions:
        result = str(record.get("result", record.get("status", ""))).upper()
        target = _target(
            record.get(
                "adjusted_target",
                record.get("share_target", record.get("target")),
            )
        )
        work = (TWO256 / (target + 1)) if target is not None and target >= 0 else 0.0
        hash_work = float(record.get("expected_hash_work", 0.0) or 0.0)
        if not hash_work and work:
            daf = float(record.get("daf", 0.0) or 0.0)
            hash_work = work * daf
        if result == "ACCEPTED":
            accepted += 1
            normalized_accepted += work
            expected_hash_work_accepted += hash_work
        elif result == "STALE":
            stale += 1
            normalized_stale += work
            expected_hash_work_stale += hash_work
        elif result in {"REJECTED", "INVALID"}:
            rejected += 1
    times = [int(r["monotonic_ns"]) for r in records if r.get("monotonic_ns") is not None]
    duration = (max(times) - min(times)) / 1e9 if len(times) >= 2 else 0.0
    if duration <= 0:
        duration = float(next((r["duration_seconds"] for r in records if r.get("duration_seconds")), 0.0))
    hours = duration / 3600.0 if duration > 0 else 0.0
    watts = [float(r["gpu_watts"]) for r in records if r.get("gpu_watts") is not None]
    return {
        "path": str(path),
        "duration_seconds": duration,
        "accepted": accepted,
        "stale": stale,
        "rejected": rejected,
        "normalized_accepted_work": normalized_accepted,
        "normalized_stale_work": normalized_stale,
        "normalized_accepted_work_per_hour": normalized_accepted / hours if hours else 0.0,
        "normalized_stale_work_per_hour": normalized_stale / hours if hours else 0.0,
        "pool_inferred_hashes_per_second": (
            expected_hash_work_accepted / duration if duration else 0.0
        ),
        "pool_inferred_ths": (
            expected_hash_work_accepted / duration / 1e12 if duration else 0.0
        ),
        "pool_inferred_stale_ths": (
            expected_hash_work_stale / duration / 1e12 if duration else 0.0
        ),
        "average_gpu_watts": sum(watts) / len(watts) if watts else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--krig", type=Path, required=True)
    parser.add_argument("--custom", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/pool_work_comparison.json"))
    args = parser.parse_args()
    result = {"krig": summarize(args.krig), "custom": summarize(args.custom)}
    result["custom_vs_krig_work_ratio"] = (
        result["custom"]["normalized_accepted_work_per_hour"] /
        result["krig"]["normalized_accepted_work_per_hour"]
        if result["krig"]["normalized_accepted_work_per_hour"] else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("| miner | pool-inferred TH/s | accepted work/hour | stale work/hour | accepted | stale | rejected | avg watts |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in ("krig", "custom"):
        s = result[name]
        watts = "n/a" if s["average_gpu_watts"] is None else f"{s['average_gpu_watts']:.2f}"
        print(
            f"| {name} | {s['pool_inferred_ths']:.6g} | "
            f"{s['normalized_accepted_work_per_hour']:.6g} | "
            f"{s['normalized_stale_work_per_hour']:.6g} | {s['accepted']} | "
            f"{s['stale']} | {s['rejected']} | {watts} |"
        )


if __name__ == "__main__":
    main()

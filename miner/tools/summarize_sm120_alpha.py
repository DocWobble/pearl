#!/usr/bin/env python3
"""Summarize the latest SM120 rate/diagnostic telemetry from a miner log."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_fields(line: str, marker: str) -> dict[str, str]:
    payload = line.split(marker, 1)[1].strip()
    fields: dict[str, str] = {}
    for token in payload.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()

    latest_rate: dict[str, str] | None = None
    latest_verbose: dict[str, str] | None = None
    alpha_active = False
    rate_lines = 0
    for line in args.log.read_text(errors="replace").splitlines():
        if "SM120_ALPHA_ACTIVE " in line:
            alpha_active = True
        if "SM120_RATE " in line:
            latest_rate = parse_fields(line, "SM120_RATE ")
            rate_lines += 1
        if "SM120_ALPHA " in line:
            latest_verbose = parse_fields(line, "SM120_ALPHA ")

    if latest_rate is None and latest_verbose is None:
        raise SystemExit("No SM120 telemetry found in log")

    print(json.dumps({
        "alpha_active_seen": alpha_active,
        "rate_lines": rate_lines,
        "latest_rate": latest_rate,
        "latest_verbose": latest_verbose,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

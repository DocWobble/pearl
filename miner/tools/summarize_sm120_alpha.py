#!/usr/bin/env python3
"""Summarize the latest SM120_ALPHA telemetry line from a miner log."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_fields(line: str) -> dict[str, str]:
    payload = line.split("SM120_ALPHA ", 1)[1].strip()
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

    latest: dict[str, str] | None = None
    alpha_active = False
    alpha_lines = 0
    for line in args.log.read_text(errors="replace").splitlines():
        if "SM120_ALPHA_ACTIVE " in line:
            alpha_active = True
        if "SM120_ALPHA " in line:
            latest = parse_fields(line)
            alpha_lines += 1

    if latest is None:
        raise SystemExit("No SM120_ALPHA telemetry found in log")

    print(json.dumps({
        "alpha_active_seen": alpha_active,
        "telemetry_lines": alpha_lines,
        "latest": latest,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

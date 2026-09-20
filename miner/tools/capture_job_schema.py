#!/usr/bin/env python3
"""Classify ownership of an actual pool job without changing it.

Read one JSON job envelope from --input or stdin and write a stable schema report.
The classifier is intentionally conservative: absent fields are reported as unknown, never as
miner-controlled.  It does not connect to a pool or send a message.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def locate(obj: Any, names: set[str]) -> bool:
    if isinstance(obj, dict):
        return any(key.lower() in names or locate(value, names) for key, value in obj.items())
    if isinstance(obj, list):
        return any(locate(value, names) for value in obj)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/job_schema.json"))
    args = parser.parse_args()
    source = args.input.read_text() if args.input else __import__("sys").stdin.read()
    if not source.strip():
        raise SystemExit("capture_job_schema.py requires a captured JSON job envelope")
    envelope = json.loads(source)
    groups = {
        "POOL_CONTROLS_ROOT_A": {"hash_a", "root_a", "a_root", "matrix_a_root"},
        "POOL_CONTROLS_ROOT_B": {"hash_b", "root_b", "b_root", "matrix_b_root"},
        "POOL_CONTROLS_M": {"m", "rows", "matrix_m"},
        "POOL_CONTROLS_N": {"n", "cols", "columns", "matrix_n"},
        "POOL_CONTROLS_K": {"k", "common_dim", "common_dimension", "matrix_k"},
        "POOL_CONTROLS_MINING_CONFIG": {"mining_config", "miningconfiguration", "config"},
    }
    result = {"source": str(args.input) if args.input else "stdin", "classification": {}}
    for name, aliases in groups.items():
        result["classification"][name] = locate(envelope, aliases)
    # Missing fields are unknown, never evidence that the miner may mutate a
    # commitment.  This keeps the mutable-A branch disabled until a real pool
    # envelope proves the ownership rule.
    has_root_a = result["classification"]["POOL_CONTROLS_ROOT_A"]
    has_root_b = result["classification"]["POOL_CONTROLS_ROOT_B"]
    explicit_mutable_a = isinstance(envelope, dict) and envelope.get("miner_controls_root_a") is True
    result["MUTABLE_A_ROOT"] = True if explicit_mutable_a else (False if has_root_a else None)
    result["FIXED_B_REUSE"] = True if has_root_b else None
    result["status"] = "captured" if result["classification"]["POOL_CONTROLS_MINING_CONFIG"] else "incomplete_job_schema"
    result["mutable_a_release_eligible"] = bool(
        result["status"] == "captured"
        and explicit_mutable_a
        and result["classification"]["POOL_CONTROLS_M"]
        and result["classification"]["POOL_CONTROLS_N"]
        and result["classification"]["POOL_CONTROLS_K"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

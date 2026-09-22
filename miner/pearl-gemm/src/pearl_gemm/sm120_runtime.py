"""Narrow runtime gate for the experimental SM120 miner backend.

The gate does not authorize a submission.  It only identifies the hardware and the explicit
operator flag; release_gates.json remains the authority for enabling a live custom path.
"""
from __future__ import annotations

import os
from pathlib import Path


FEATURE_FLAGS = (
    "PEARL_SM120_CACHE_B",
    "PEARL_SM120_REUSE_ALLOC",
    "PEARL_SM120_DOUBLE_BUFFER",
    "PEARL_SM120_IMPLICIT_ZERO_BASE",
    "PEARL_SM120_SEARCH_ONLY",
    "PEARL_SM120_FUSED",
    "PEARL_SM120_WMMA",
)


def backend_importable() -> bool:
    try:
        import pearl_sm120_cuda  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def requested() -> bool:
    return os.environ.get("PEARL_SM120_BACKEND") == "1"


def feature_enabled(name: str, default: bool = False) -> bool:
    if name not in FEATURE_FLAGS:
        raise ValueError(f"unknown SM120 feature flag: {name}")
    value = os.environ.get(name)
    return default if value is None else value == "1"


def feature_state() -> dict[str, bool]:
    return {name: feature_enabled(name) for name in FEATURE_FLAGS}


def supported(device_capability: tuple[int, int]) -> bool:
    return device_capability == (12, 0)


def release_authorized(root: Path) -> bool:
    gate_file = root / "artifacts" / "results" / "release_gates.json"
    return gate_file.exists() and '"release_enabled": true' in gate_file.read_text()

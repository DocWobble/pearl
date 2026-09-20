#!/usr/bin/env python3
"""Run SM120 release checks and write a machine-readable gate report."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run(command: list[str], cwd: Path) -> dict[str, object]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-gates", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=Path("build/sm120"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    report_path = root / "artifacts" / "results" / "release_gates.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ctest = run(["ctest", "--output-on-failure"], root / args.build_dir)
    fixture_path = root / "artifacts" / "parity" / "reference_vectors.json"
    fixture_summary = {"count": 0, "all_verified": False, "all_target_passes": False}
    if fixture_path.exists():
        fixtures = json.loads(fixture_path.read_text()).get("vectors", [])
        fixture_summary = {
            "count": len(fixtures),
            "all_verified": bool(fixtures) and all(v.get("canonical_verifier_accepts") for v in fixtures),
            "all_target_passes": bool(fixtures) and all(v.get("target_passes") for v in fixtures),
        }
    rust_gate = {"returncode": None, "stdout": "", "stderr": ""}
    corrupt_gate = {"returncode": None, "stdout": "", "stderr": ""}
    if args.release_gates:
        cargo = os.environ.get("CARGO", "/home/director/.cargo/bin/cargo")
        rust_gate = run([cargo, "test", "--manifest-path", "zk-pow/Cargo.toml", "--test",
                         "sm120_verifier_gate", "--", "--ignored", "--nocapture"], root)
        corrupt_gate = run([cargo, "test", "--manifest-path", "zk-pow/Cargo.toml", "--test",
                            "sm120_corrupt_gate", "--", "--nocapture"], root)
    gpu_parity_path = root / "artifacts" / "results" / "sm120_gpu_parity.json"
    gpu_parity = json.loads(gpu_parity_path.read_text()) if gpu_parity_path.exists() else {}
    stress_path = root / "artifacts" / "results" / "sm120_stress.json"
    stress = json.loads(stress_path.read_text()) if stress_path.exists() else {}
    tuning_path = root / "artifacts" / "results" / "sm120_tuning.json"
    tuning = json.loads(tuning_path.read_text()) if tuning_path.exists() else {}
    # Gates 1-3 are not represented by a green CTest alone. They remain false until vectors,
    # 1000 canonical proofs, and corrupt-fixture rejection are executed through Pearl's verifier.
    gates = {
        # GPU parity currently covers selected proof tiles; roots and proof reconstruction still
        # remain outside the SM120 search path, so the full parity gate stays closed.
        "GATE_1_reference_vector_parity": False,
        "GATE_2_canonical_verifier_1000_of_1000": rust_gate["returncode"] == 0,
        "GATE_3_corrupt_proofs_rejected": corrupt_gate["returncode"] == 0,
        "GATE_4_30m_cuda_stress": stress.get("requested_seconds") == 1800 and stress.get("gate_pass") is True,
        "GATE_5_no_winner_overflow": tuning.get("status") == "complete" and bool(tuning.get("results")) and all(not r.get("winner_overflow") for r in tuning.get("results", [])),
        # A local replay test proves only the state-machine branch.  This gate requires a
        # recorded, generation-tagged submission against the real pool adapter.
        "GATE_6_no_stale_generation_submission": False,
    }
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "ctest": ctest,
              "fixture_summary": fixture_summary, "gpu_parity": gpu_parity,
              "rust_verifier_gate": rust_gate, "corrupt_gate": corrupt_gate,
              "stress": stress, "tuning": tuning,
              "gates": gates, "release_enabled": all(gates.values()),
              "reason": "Live submission remains disabled until every canonical gate is proven."}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.release_gates and not report["release_enabled"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

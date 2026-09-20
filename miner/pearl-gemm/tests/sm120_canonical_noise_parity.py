"""Compare canonical Rust fixture noise against the SM120 CUDA generator."""

import json
import os
import time
from pathlib import Path

import torch

import pearl_sm120_cuda


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    fixture_path = root / "artifacts" / "parity" / "reference_vectors.json"
    fixtures = json.loads(fixture_path.read_text())["vectors"]
    for index, fixture in enumerate(fixtures):
        a_seed = torch.tensor(list(bytes.fromhex(fixture["a_noise_seed_hex"])), dtype=torch.uint8)
        b_seed = torch.tensor(list(bytes.fromhex(fixture["b_noise_seed_hex"])), dtype=torch.uint8)
        a_rows = torch.tensor(fixture["a_row_indices"], device="cuda", dtype=torch.int32)
        b_cols = torch.tensor(fixture["b_col_indices"], device="cuda", dtype=torch.int32)
        noise_a, noise_b_t = pearl_sm120_cuda.noise(a_seed, b_seed, a_rows, b_cols, 2048, 128)
        assert noise_a.dtype == torch.int8 and noise_b_t.dtype == torch.int8
        assert noise_a.shape == (len(fixture["a_row_indices"]), 2048)
        assert noise_b_t.shape == (len(fixture["b_col_indices"]), 2048)
        assert noise_a[0, :16].cpu().tolist() == fixture["sampled_noise_a"], fixture["id"]
        assert noise_b_t[0, :16].cpu().tolist() == fixture["sampled_noise_b_transposed"], fixture["id"]
        # The canonical proof selects a 4x8 tile. Zero-padding the remaining rows and columns
        # leaves its XOR fold unchanged, letting the 16x16 CUDA backend reproduce the exact
        # selected-tile transcript without inventing a different committed matrix.
        a_noised = torch.zeros((16, 2048), device="cuda", dtype=torch.int8)
        b_noised_t = torch.zeros((16, 2048), device="cuda", dtype=torch.int8)
        a_base = torch.tensor(fixture["selected_a_rows"], device="cuda", dtype=torch.int8)
        b_base = torch.tensor(fixture["selected_b_cols_transposed"], device="cuda", dtype=torch.int8)
        a_noised[: a_base.shape[0]] = a_base + noise_a
        b_noised_t[: b_base.shape[0]] = b_base + noise_b_t
        observed_words = pearl_sm120_cuda.transcript(a_noised, b_noised_t, 2048, 128).cpu().tolist()
        expected_words = [word if word < 2**31 else word - 2**32 for word in fixture["final_jackpot_words_le"]]
        assert observed_words == expected_words, fixture["id"]
        key = torch.tensor(list(bytes.fromhex(fixture["a_noise_seed_hex"])), dtype=torch.uint8)
        target = torch.full((32,), 255, dtype=torch.uint8)
        result = pearl_sm120_cuda.search(a_noised.unsqueeze(0), b_noised_t, key, target, 16, 16, 2048, index)
        assert result["winners"] == 1 and result["cuda_errors"] == 0, fixture["id"]
        assert result["winner_descriptors"][0]["jackpot_hash"].hex() == fixture["final_keyed_blake3_hex"], fixture["id"]
        if index % 32 == 0:
            print("verified", index + 1, "of", len(fixtures), fixture["id"])
    report = {
        "checked_at": time.time(),
        "fixtures": len(fixtures),
        "noise_samples_match": True,
        "transcript_words_match": True,
        "keyed_blake3_match": True,
    }
    report_path = root / "artifacts" / "results" / "sm120_gpu_parity.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("canonical noise/transcript/hash parity passed", len(fixtures), "fixtures")


if __name__ == "__main__":
    os.environ["PEARL_SM120_BACKEND"] = "1"
    os.environ["PEARL_SM120_FUSED"] = "1"
    os.environ["PEARL_SM120_SEARCH_ONLY"] = "1"
    main()

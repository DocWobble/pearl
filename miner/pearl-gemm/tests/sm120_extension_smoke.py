"""Live CUDA smoke test for the separately built SM120 PyTorch extension."""

import os

import torch

import pearl_sm120_cuda


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for SM120 extension smoke test")
    # (m, k) and (n, k) are the legal rank-128 INT8 matrix layout consumed by
    # the fused backend.  An all-FF target guarantees a winner without
    # pretending this is a proof-valid pool share.
    candidate_a = torch.arange(16 * 2048, device="cuda", dtype=torch.int8).reshape(1, 16, 2048)
    b_transposed = torch.arange(16 * 2048, device="cuda", dtype=torch.int8).reshape(16, 2048)
    key = torch.arange(32, dtype=torch.uint8)
    target = torch.full((32,), 255, dtype=torch.uint8)
    os.environ["PEARL_SM120_BACKEND"] = "1"
    os.environ["PEARL_SM120_FUSED"] = "1"
    os.environ["PEARL_SM120_SEARCH_ONLY"] = "1"
    result = pearl_sm120_cuda.search(candidate_a, b_transposed, key, target, 16, 16, 2048, 7)
    assert result["candidates"] == 1 and result["valid_candidate_work"] == 16 * 16 * 2048, result
    assert result["target_tests"] == 1, result
    assert result["winners"] == 1, result
    assert result["winner_overflow"] is False, result
    assert result["cuda_errors"] == 0, result
    assert result["winner_descriptors"] == [{
        "generation": 7, "candidate_index": 0, "tile_row": 0, "tile_col": 0,
        "reconstruction_slot": 0, "jackpot_hash": result["winner_descriptors"][0]["jackpot_hash"],
    }], result

    # Production v08 entry point keeps the key and target CUDA-resident and
    # launches on PyTorch's current stream.
    device_result = pearl_sm120_cuda.search_device(
        candidate_a, b_transposed, key.cuda(), target.cuda(), 16, 16, 2048, 9
    )
    assert device_result["candidates"] == 1
    assert device_result["target_tests"] == 1
    assert device_result["winners"] == 1 and device_result["cuda_errors"] == 0, device_result
    assert device_result["winner_descriptors"][0]["generation"] == 9

    # WMMA is the v08 production default. Run the DP4A fallback over the exact
    # same operands and require an identical jackpot hash, not merely a winner.
    wmma_hash = device_result["winner_descriptors"][0]["jackpot_hash"]
    os.environ["PEARL_SM120_WMMA"] = "0"
    dp4a_result = pearl_sm120_cuda.search_device(
        candidate_a, b_transposed, key.cuda(), target.cuda(), 16, 16, 2048, 10
    )
    assert dp4a_result["candidates"] == 1
    assert dp4a_result["target_tests"] == 1
    assert dp4a_result["winners"] == 1 and dp4a_result["cuda_errors"] == 0, dp4a_result
    assert dp4a_result["winner_descriptors"][0]["generation"] == 10
    assert dp4a_result["winner_descriptors"][0]["jackpot_hash"] == wmma_hash
    os.environ["PEARL_SM120_WMMA"] = "1"
    # Exercise the production-shaped entry point: base operands and canonical noise
    # seeds enter the extension, and noised operands are materialized on the GPU.
    rows = torch.arange(16, device="cuda", dtype=torch.int32)
    cols = torch.arange(16, device="cuda", dtype=torch.int32)
    base_a = torch.zeros((1, 16, 2048), device="cuda", dtype=torch.int8)
    base_b = torch.zeros((16, 2048), device="cuda", dtype=torch.int8)
    noised_result = pearl_sm120_cuda.noise_search(
        base_a, base_b, key, key, rows, cols, target, 16, 16, 2048, 8
    )
    assert noised_result["candidates"] == 1
    assert noised_result["winners"] == 1 and noised_result["cuda_errors"] == 0, noised_result
    print(result)


if __name__ == "__main__":
    main()

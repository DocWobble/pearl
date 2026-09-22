# SM120 fork notes

This branch, `sm120-pool-miner-v06`, is the active RTX 50-series / SM120 fork.
It is intentionally separate from upstream `pearl-research-labs/pearl`; model
weights are never versioned here.

## Modified implementation boundary

- `miner/pearl-gemm/src/pearl_gemm/sm120_reference.py` is the canonical
  integer noising oracle.
- `pearl_gemm_interface.py` selects that oracle and SM120 search only when
  explicitly enabled with `PEARL_SM120_BACKEND=1`, `PEARL_SM120_FUSED=1`, and
  `PEARL_SM120_SEARCH_ONLY=1`. It prevents unsupported Hopper WGMMA/TMA paths
  from being used on SM120.
- `miner/pearl-gemm/csrc/sm120/` remains the architecture-specific search
  backend. Its result must still be proven by a gateway submission and node
  acceptance; an inference response is not a mining reward.
- `miner/vllm-miner/entrypoint.sh` owns gateway shutdown to permit clean
  restarts without a stale Unix socket.

## Build note

The pinned CUTLASS submodule is deliberately left at its upstream commit. For
CUDA 13 SM120 builds, apply
`miner/pearl-gemm/patches/cutlass-sm120-tma-gate.patch` inside
`miner/pearl-gemm/third_party/cutlass` before building the extension.

## Deliberately excluded

- model weights and wallet material;
- generated tuning/results and runtime logs;
- claims of earnings absent a gateway submission and accepted node result.

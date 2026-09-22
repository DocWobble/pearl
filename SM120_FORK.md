# SM120 fork notes

This branch line is the RTX 50-series / SM120 fork. `sm120-functional-alpha-v07` adds live, protocol-level search accounting on top of the v06 implementation.
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

## Functional alpha telemetry

The v07 alpha passes the current mining-job generation into every SM120 search and emits
protocol-level counters from completed searches.  Set `PEARL_SM120_LOG_EVERY` to control
the reporting interval (default 64 search calls).  A live run should first print
`SM120_ALPHA_ACTIVE` and then periodic `SM120_ALPHA` lines.

The key measurement is `tests`: the number of completed 16x16 Pearl jackpot target
comparisons represented by that search call:

`candidates * (m / 16) * (n / 16)`

`job_tests_s` and `total_tests_s` report those target comparisons per second.  The
`*_expected_hits` fields integrate the actual adjusted uint256 target used by each call,
so `job_mean_hit_s` is the current job-window mean time to one target hit under the
observed layer mix.  On a direct-node SOLO run that target is the adjusted block target;
on a pool run it represents the target supplied by that pool job.

The alpha fails closed instead of silently entering legacy Hopper WGMMA/TMA when the
SM120 backend is requested incorrectly.  Irregular/non-proof geometry uses the exact
architecture-neutral inference fallback and is not counted as mining work.

Current alpha boundary: the live vLLM seam still supplies one candidate matrix per search
call.  The fused kernel fans that candidate over all complete 16x16 Pearl tiles, but the
dedicated fixed-B/mutable-A candidate-bank loop remains the next optimization stage.

To summarize a captured terminal log:

`python miner/tools/summarize_sm120_alpha.py /path/to/.sm120-miner.log`

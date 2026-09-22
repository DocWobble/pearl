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


## Throughput/search v08

`sm120-throughput-search-v08` keeps the v07 proof/submission boundary and
moves the hot search path closer to a real standalone miner.

### Throughput changes

- `DeviceScratch` is persistent per CUDA device and
  `PEARL_SM120_REUSE_ALLOC=1` is enabled by default on the SM120 live path.
  The winner queue is no longer allocated and freed for every search call.
- The vLLM seam calls `pearl_sm120_cuda.search_device`, leaving the 32-byte
  jackpot key and 32-byte target resident on CUDA instead of copying both back
  to the CPU before every search.
- Losing searches clear and copy only the compact queue/analytics header.
  Winner descriptors are copied only when a winner actually exists.
- The fused rank-128 inner loop uses exact signed INT8 `__dp4a` groups of four
  instead of one scalar multiply per K element. Rank checkpoints stay at the
  same 128-element boundaries.
- The 256-word checkpoint XOR uses warp shuffles plus eight warp partials,
  reducing the full-block synchronization tree to two barriers per checkpoint.
  The CUDA parity test hashes an independently accumulated scalar transcript
  and requires the fused DP4A result to match byte-for-byte.
- `miner/tools/benchmark_sm120_request_batch.py` sweeps prompt/prefill size and
  request batch size in one process using measured rolling TH/s. This is preferable to multiple
  competing workload processes, which were observed to divide the same GPU
  throughput rather than increase it.

### Apples-to-apples Pearl hashrate

The captured Krig Prometheus baseline defines:

`krig_miner_hashes_per_second = tiles_per_second * DAF`

where for the captured K=2048 workload DAF is `16*16*2048 = 524288`.
The SM120 backend's `valid_candidate_work` is exactly the corresponding
quantity:

`candidates * m * n * k`

Therefore v08 reports:

`Pearl H/s = sum(valid_candidate_work) / elapsed_seconds`

and prints it as `hashrate_ths`, `rolling_ths`, and `job_ths` on
`SM120_RATE` lines. `tiles_s` remains a separate diagnostic counting final
16x16 jackpot target comparisons per second.

Use:

`python miner/tools/compare_sm120_krig_hashrate.py --krig-prom artifacts/baseline/krig/metrics_samples.prom --custom-log /path/to/.sm120-miner.log`

for a direct median/mean/p90 TH/s comparison.

### Search-distribution diagnostics

The fused kernel also keeps nested counters for final hashes with at least
8, 12, 16, 20, and 24 leading zero bits. The counters do not modify the real
target and add an atomic operation only after the first 1/256 tail event.
Telemetry reports observed/expected tail ratios. These counters are the gate
for any future candidate-source steering: no source-order strategy is treated
as advantageous unless it shows reproducible tail enrichment on unseen jobs.

The fixed-B/mutable-A standalone candidate engine remains gated by Package 3
of the v0.6 handoff. B caching is deliberately not enabled automatically in
the current vLLM seam because its complete cache identity is not yet populated
there. Proof-validity and real-job ownership tests come before that optimization.


### Pool-inferred effective hashrate

Accepted custom shares now log `adjusted_target`, `daf`, and
`expected_hash_work`. For a share:

`expected_hash_work = DAF * 2^256 / (adjusted_target + 1)`

so summing accepted expected hash work and dividing by elapsed seconds gives a
pool-inferred H/s in the same dimensional unit as Krig's local hashrate.
`compare_pool_work.py` reports this as pool-inferred TH/s when those fields
are present. This is intentionally independent of the local CUDA counter.


### Tensor-core search

v08 now prefers an exact signed-INT8 WMMA `m16n16k16` search kernel with
INT32 accumulation. One warp owns one complete 16x16 Pearl proof tile. Eight
K=16 WMMA operations advance each rank-128 checkpoint; the accumulator remains
live across checkpoints, is stored through `wmma::store_matrix_sync`, XOR
folded exactly, and fed into the unchanged 16-word transcript and keyed BLAKE3
path.

`PEARL_SM120_WMMA=1` is the default. Setting
`PEARL_SM120_WMMA=0` selects the exact DP4A fallback. The extension smoke
test executes both kernels on identical operands and requires their 32-byte
jackpot hashes to match exactly. The dense CUDA parity test independently
constructs the scalar cumulative transcript and also requires the fused winner
hash to match it.

The WMMA path is the primary throughput experiment for the 5070 Ti. It must
compile and pass those parity checks on the physical SM120 card before its
reported TH/s is treated as valid.

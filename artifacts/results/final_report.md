# Pearl SM120 miner report

This report is regenerated as release gates and live comparison evidence become available.

- Git revision: `5b09d844e4069440933722c51495ac24a7bb4886`
- Selected math backend: exact signed INT8 operands with INT32 accumulation; C++ scalar fallback pending a measured CUTLASS alternative
- Selected kernel tile: not selected
- Selected batch size: not selected
- Enabled optimization flags: `PEARL_SM120_BACKEND`, `PEARL_SM120_FUSED`, and `PEARL_SM120_SEARCH_ONLY` are required for the experimental search path; cache/reuse/double-buffer/implicit-zero remain independently gated
- SM120 build: CUDA 12.8 PyTorch extension compiled with `-gencode arch=compute_120,code=sm_120`
- SM120 GPU smoke: one candidate, 32,768 valid INT8-to-INT32 operations, one deliberate all-FF-target winner, zero CUDA errors, no queue overflow
- SM120 CMake suite: 6 of 6 passed on the RTX 5070 Ti, including the CUDA dense-reference test
- Canonical parity corpus: 160 verified rank-128/K=2048 fixtures (100 normal, 20 boundary, 20 Legacy, 20 Salted); all adjusted-target comparisons pass
- GPU parity implementation: CUDA reproduces selected noise, all 16 transcript words, and keyed BLAKE3 for the corpus when the host-GPU parity test is run
- Integrated search paths: the active `pearl_gemm_interface.noisy_gemm` path and the `miner-base.NoisyGemm.gemm` fallback can dispatch exact SM120 winner search; Pearl's existing GEMM still produces the exact output/denoising data and proof callback
- GPU noising path: `noise_search` generates canonical selected-row/column noise and adds it on-device, with an explicit zero-base branch
- B cache: generation-keyed device-side B state is copied on misses and reused only when the full dependency key matches
- Canonical verifier gates: 1,000/1,000 accepted; 20 deliberate corrupted fixtures rejected
- Generation/cache guards: job replacement and disconnect reject old winners; B-cache keys include generation, job/hash/target, dimensions, rank, certificate, noise seed, and B storage identity
- Canonical Rust baseline: mined MoE proof verified; malformed dimension, corrupted routing root, and tampered routing hash were rejected
- Release gates: GATE_2 and GATE_3 proven; GATE_1, GATE_4, GATE_5, and GATE_6 remain closed
- Krig baseline: 30-minute connected-pool sample completed; 27 accepted, 0 rejected, 0 additional stale shares; 93,794 iterations
- Custom A/B results: not run; custom submission is disabled
- Final two-hour result: not run
- Selected hardware profile: not selected
- Production recommendation: **KRIG**

The exact remaining bottleneck is release evidence: the 30-minute custom CUDA stress run,
deterministic tuning measurement, GPU execution of the full parity test, and a live generation-tagged
pool submission record remain. The active dispatch is opt-in and has not been accepted as the
production path until GPU parity and throughput evidence are recorded. The canonical proof builder
is preserved behind both dispatch paths, but no custom candidate is eligible for submission while
any gate is closed.
The current machine-side stress artifact records `CUDA GPU required for SM120 stress gate`, and the
tuning artifact records `No CUDA GPUs are available`; no tuning header was generated from those
blocked runs.

#pragma once

#include <array>
#include <cstdint>
#include <cuda_runtime_api.h>
#include "sm120_job_state.h"
#include "sm120_metrics.h"
#include "sm120_winner_queue.cuh"

namespace pearl::sm120 {

struct CandidateBatch {
  const int8_t* a_noised{};  // candidate-major, row-major [batch][m][k]
  const int8_t* b_noised_t{};  // row-major [n][k]
  uint32_t count{};
  uint64_t b_identity{};  // backing-storage identity, supplied by the owner of B
};

struct SearchResult {
  Metrics metrics;
  bool winner_overflow{};
  std::array<WinnerDescriptor, kWinnerQueueCapacity> winners{};
};

// Per-worker device state. It owns only backend-private allocations: candidate tensors remain
// owned by the caller. `reserve` grows monotonically and never reallocates for a smaller job.
struct DeviceScratch {
  DeviceWinnerQueue* winner_queue{};
  cudaStream_t stream_compute{};
  cudaStream_t stream_prepare{};
  cudaEvent_t buffer_ready[2]{};
  uint32_t max_m{}, max_n{}, max_k{};
  bool streams_enabled{};
  bool b_cache_valid{};
  BCacheKey b_cache_key{};
  int8_t* b_cache_device{};
  uint64_t b_cache_bytes{};

  DeviceScratch() = default;
  DeviceScratch(const DeviceScratch&) = delete;
  DeviceScratch& operator=(const DeviceScratch&) = delete;
  ~DeviceScratch();
  bool reserve(uint32_t m, uint32_t n, uint32_t k);
  bool observe_b(const JobContext& job, const CandidateBatch& candidates);
  const int8_t* cached_b() const { return b_cache_device; }
  void release();
};

// Exact integer transcript reduction only. This entry point deliberately returns descriptors
// and counters; proof reconstruction and network submission remain outside the GPU backend.
SearchResult search_job(const JobContext& job, const CandidateBatch& candidates, DeviceScratch& scratch);
SearchResult search_job(const JobContext& job, const CandidateBatch& candidates);
// Generate canonical selected-row/column noise on the device, add it to the supplied
// base operands, and run the exact search path without a host-side noised copy.
SearchResult noise_and_search_job(const JobContext& job, const int8_t* a_base,
                                  const int8_t* b_base_t, const uint32_t* a_rows,
                                  const uint32_t* b_cols, uint32_t candidate_count,
                                  uint64_t b_identity, DeviceScratch& scratch,
                                  cudaStream_t stream = nullptr);
bool runtime_enabled();
int dense_reference_tile_launch(const int8_t* a, const int8_t* b_transposed,
                                uint32_t k, uint32_t rank, uint32_t* transcript_out);
int generate_noise_launch(const uint8_t* a_noise_seed, const uint8_t* b_noise_seed,
                          const uint32_t* a_rows, uint32_t a_row_count,
                          const uint32_t* b_cols, uint32_t b_col_count,
                          uint32_t k, uint32_t rank, int8_t* noise_a, int8_t* noise_b_t,
                          cudaStream_t stream);
int add_noise_candidates_launch(const int8_t* base, const int8_t* noise, int8_t* out,
                               uint64_t elements_per_candidate, uint32_t candidate_count,
                               cudaStream_t stream);
int repeat_candidates_launch(const int8_t* source, int8_t* out,
                             uint64_t elements_per_candidate, uint32_t candidate_count,
                             cudaStream_t stream);
int fused_search_launch(const CandidateBatch& candidates, const DeviceSearchConfig& config,
                        DeviceWinnerQueue* queue, cudaStream_t stream);

}  // namespace pearl::sm120

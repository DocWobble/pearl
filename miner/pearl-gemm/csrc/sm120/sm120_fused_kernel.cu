#include <cuda_runtime.h>
#include "sm120_backend.h"
#include "sm120_blake3.cuh"
#include "sm120_transcript.cuh"

namespace pearl::sm120 {

__device__ bool hash_meets_target(const uint8_t hash[32], const uint8_t target[32]) {
  for (int i = 31; i >= 0; --i) {
    if (hash[i] < target[i]) return true;
    if (hash[i] > target[i]) return false;
  }
  return true;
}

__global__ void fused_search_kernel(const int8_t* candidates_a, const int8_t* b_transposed,
                                    uint32_t candidate_count, DeviceSearchConfig config,
                                    DeviceWinnerQueue* queue) {
  const uint32_t lane = threadIdx.x;
  const uint32_t tiles_w = config.n / 16;
  const uint32_t tiles = (config.m / 16) * tiles_w;
  const uint32_t candidate = blockIdx.x / tiles;
  const uint32_t tile = blockIdx.x % tiles;
  if (candidate >= candidate_count || lane >= 256 || config.m % 16 || config.n % 16 ||
      config.rank != 128 || config.k == 0 || config.k > 65536 || config.k % config.rank) return;
  const uint32_t tile_row = tile / tiles_w, tile_col = tile % tiles_w;
  const int8_t* a = candidates_a + uint64_t(candidate) * config.m * config.k + uint64_t(tile_row * 16) * config.k;
  b_transposed += uint64_t(tile_col * 16) * config.k;
  const uint32_t row = lane / 16, col = lane % 16;
  int32_t cumulative = 0;
  __shared__ uint32_t folded[256];
  __shared__ uint32_t transcript[16];
  __shared__ uint8_t final_hash[32];
  if (lane < 16) transcript[lane] = 0;
  __syncthreads();
  for (uint32_t p = 0, chunk = 0; p < config.k; p += config.rank, ++chunk) {
    for (uint32_t x = p; x < p + config.rank; ++x)
      cumulative += int32_t(a[row * config.k + x]) * int32_t(b_transposed[col * config.k + x]);
    folded[lane] = static_cast<uint32_t>(cumulative);
    __syncthreads();
    for (uint32_t stride = 128; stride; stride >>= 1) {
      if (lane < stride) folded[lane] ^= folded[lane + stride];
      __syncthreads();
    }
    if (lane == 0) transcript[chunk % 16] = rotl32(transcript[chunk % 16], 13) ^ folded[0];
    __syncthreads();
  }
  if (lane == 0) {
    keyed_blake3_64(transcript, config.jackpot_key, final_hash);
    if (hash_meets_target(final_hash, config.share_target)) {
      const uint32_t slot = atomicAdd(&queue->winner_count, 1);
      if (slot < kWinnerQueueCapacity) {
        WinnerDescriptor& winner = queue->winners[slot];
        winner.id.generation = config.generation;
        winner.id.candidate_index = config.candidate_base + candidate;
        winner.tile_row = tile_row; winner.tile_col = tile_col; winner.reconstruction_slot = candidate;
        for (uint32_t i = 0; i < 32; ++i) winner.jackpot_hash.bytes[i] = final_hash[i];
      } else queue->overflow_flag = 1;
    }
  }
}

int fused_search_launch(const CandidateBatch& candidates, const DeviceSearchConfig& config,
                        DeviceWinnerQueue* queue, cudaStream_t stream) {
  const uint32_t tiles = (config.m / 16) * (config.n / 16);
  fused_search_kernel<<<candidates.count * tiles, 256, 0, stream>>>(candidates.a_noised, candidates.b_noised_t,
                                                                      candidates.count, config, queue);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace pearl::sm120

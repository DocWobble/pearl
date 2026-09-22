#include <cstdlib>
#include <cuda_runtime.h>
#include <mma.h>

#include "sm120_backend.h"
#include "sm120_blake3.cuh"
#include "sm120_transcript.cuh"

namespace pearl::sm120 {

namespace wmma = nvcuda::wmma;

__device__ bool hash_meets_target(const uint8_t hash[32], const uint8_t target[32]) {
  for (int i = 31; i >= 0; --i) {
    if (hash[i] < target[i]) return true;
    if (hash[i] > target[i]) return false;
  }
  return true;
}

__device__ void record_deep_tail(const uint8_t hash[32], DeviceWinnerQueue* queue) {
  if (hash[31] != 0) return;
  atomicAdd(reinterpret_cast<unsigned long long*>(&queue->tail_d8), 1ULL);
  if ((hash[30] & 0xF0u) != 0) return;
  atomicAdd(reinterpret_cast<unsigned long long*>(&queue->tail_d12), 1ULL);
  if (hash[30] != 0) return;
  atomicAdd(reinterpret_cast<unsigned long long*>(&queue->tail_d16), 1ULL);
  if ((hash[29] & 0xF0u) != 0) return;
  atomicAdd(reinterpret_cast<unsigned long long*>(&queue->tail_d20), 1ULL);
  if (hash[29] != 0) return;
  atomicAdd(reinterpret_cast<unsigned long long*>(&queue->tail_d24), 1ULL);
}

__device__ void record_winner(const uint8_t final_hash[32],
                              uint32_t candidate, uint32_t tile_row,
                              uint32_t tile_col,
                              const DeviceSearchConfig& config,
                              DeviceWinnerQueue* queue) {
  const uint8_t* share_target =
      config.share_target_device != nullptr ? config.share_target_device
                                            : config.share_target;
  record_deep_tail(final_hash, queue);
  if (!hash_meets_target(final_hash, share_target)) return;

  const uint32_t slot = atomicAdd(&queue->winner_count, 1);
  if (slot < kWinnerQueueCapacity) {
    WinnerDescriptor& winner = queue->winners[slot];
    winner.id.generation = config.generation;
    winner.id.candidate_index = config.candidate_base + candidate;
    winner.tile_row = tile_row;
    winner.tile_col = tile_col;
    winner.reconstruction_slot = candidate;
    for (uint32_t i = 0; i < 32; ++i)
      winner.jackpot_hash.bytes[i] = final_hash[i];
  } else {
    queue->overflow_flag = 1;
  }
}

__global__ void fused_search_dp4a_kernel(
    const int8_t* __restrict__ candidates_a,
    const int8_t* __restrict__ b_transposed,
    uint32_t candidate_count, DeviceSearchConfig config,
    DeviceWinnerQueue* __restrict__ queue) {
  const uint32_t lane = threadIdx.x;
  const uint32_t tiles_w = config.n / 16;
  const uint32_t tiles = (config.m / 16) * tiles_w;
  const uint32_t candidate = blockIdx.x / tiles;
  const uint32_t tile = blockIdx.x % tiles;
  if (candidate >= candidate_count || lane >= 256 || config.m % 16 ||
      config.n % 16 || config.rank != 128 || config.k == 0 ||
      config.k > 65536 || config.k % config.rank) {
    return;
  }

  const uint32_t tile_row = tile / tiles_w;
  const uint32_t tile_col = tile % tiles_w;
  const int8_t* a =
      candidates_a + uint64_t(candidate) * config.m * config.k +
      uint64_t(tile_row * 16) * config.k;
  const int8_t* b_tile =
      b_transposed + uint64_t(tile_col * 16) * config.k;
  const uint32_t row = lane / 16;
  const uint32_t col = lane % 16;

  int32_t cumulative = 0;
  __shared__ uint32_t warp_folded[8];
  __shared__ uint32_t transcript[16];
  __shared__ uint8_t final_hash[32];

  if (lane < 16) transcript[lane] = 0;
  __syncthreads();

  for (uint32_t p = 0, chunk = 0; p < config.k; p += 128, ++chunk) {
#pragma unroll 4
    for (uint32_t x = p; x < p + 128; x += 4) {
      const auto* a4 =
          reinterpret_cast<const int32_t*>(a + row * config.k + x);
      const auto* b4 =
          reinterpret_cast<const int32_t*>(b_tile + col * config.k + x);
      cumulative = __dp4a(*a4, *b4, cumulative);
    }

    uint32_t folded = static_cast<uint32_t>(cumulative);
    constexpr uint32_t kFullWarpMask = 0xffffffffu;
    for (uint32_t offset = 16; offset; offset >>= 1)
      folded ^= __shfl_down_sync(kFullWarpMask, folded, offset);
    if ((lane & 31u) == 0) warp_folded[lane >> 5] = folded;
    __syncthreads();

    if (lane < 32) {
      uint32_t block_folded = lane < 8 ? warp_folded[lane] : 0u;
      for (uint32_t offset = 16; offset; offset >>= 1)
        block_folded ^=
            __shfl_down_sync(kFullWarpMask, block_folded, offset);
      if (lane == 0)
        transcript[chunk % 16] =
            rotl32(transcript[chunk % 16], 13) ^ block_folded;
    }
    __syncthreads();
  }

  if (lane == 0) {
    const uint8_t* jackpot_key =
        config.jackpot_key_device != nullptr ? config.jackpot_key_device
                                             : config.jackpot_key;
    keyed_blake3_64(transcript, jackpot_key, final_hash);
    record_winner(final_hash, candidate, tile_row, tile_col, config, queue);
  }
}

__global__ void fused_search_wmma_kernel(
    const int8_t* __restrict__ candidates_a,
    const int8_t* __restrict__ b_transposed,
    uint32_t candidate_count, DeviceSearchConfig config,
    DeviceWinnerQueue* __restrict__ queue) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t tiles_w = config.n / 16;
  const uint32_t tiles = (config.m / 16) * tiles_w;
  const uint32_t candidate = blockIdx.x / tiles;
  const uint32_t tile = blockIdx.x % tiles;
  if (candidate >= candidate_count || config.m % 16 || config.n % 16 ||
      config.rank != 128 || config.k == 0 || config.k > 65536 ||
      config.k % 128) {
    return;
  }

  const uint32_t tile_row = tile / tiles_w;
  const uint32_t tile_col = tile % tiles_w;
  const int8_t* a =
      candidates_a + uint64_t(candidate) * config.m * config.k +
      uint64_t(tile_row * 16) * config.k;
  const int8_t* b_tile =
      b_transposed + uint64_t(tile_col * 16) * config.k;

  __shared__ __align__(32) signed char smem_a[16 * 16];
  __shared__ __align__(32) signed char smem_b[16 * 16];
  __shared__ __align__(32) int32_t smem_acc[16 * 16];
  __shared__ uint32_t transcript[16];
  __shared__ uint8_t final_hash[32];

  wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char,
                 wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char,
                 wmma::col_major>
      b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_frag;
  wmma::fill_fragment(acc_frag, 0);

  if (lane < 16) transcript[lane] = 0;
  __syncwarp();

  for (uint32_t p = 0, chunk = 0; p < config.k; p += 128, ++chunk) {
#pragma unroll
    for (uint32_t kk = 0; kk < 128; kk += 16) {
      // One half-warp copies a complete aligned 16x16 INT8 A tile and the
      // equivalent column-major B tile into 32-byte-aligned shared storage.
      if (lane < 16) {
        reinterpret_cast<int4*>(smem_a)[lane] =
            *reinterpret_cast<const int4*>(
                a + uint64_t(lane) * config.k + p + kk);
        reinterpret_cast<int4*>(smem_b)[lane] =
            *reinterpret_cast<const int4*>(
                b_tile + uint64_t(lane) * config.k + p + kk);
      }
      __syncwarp();

      wmma::load_matrix_sync(a_frag, smem_a, 16);
      wmma::load_matrix_sync(b_frag, smem_b, 16);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Store through the WMMA API instead of depending on architecture-specific
    // fragment lane mapping. XOR is order-independent, so each lane folds eight
    // of the 256 exact INT32 accumulator values and the warp reduces them.
    wmma::store_matrix_sync(smem_acc, acc_frag, 16, wmma::mem_row_major);
    __syncwarp();

    uint32_t folded = 0;
#pragma unroll
    for (uint32_t i = lane; i < 256; i += 32)
      folded ^= static_cast<uint32_t>(smem_acc[i]);
    constexpr uint32_t kFullWarpMask = 0xffffffffu;
    for (uint32_t offset = 16; offset; offset >>= 1)
      folded ^= __shfl_down_sync(kFullWarpMask, folded, offset);

    if (lane == 0)
      transcript[chunk % 16] =
          rotl32(transcript[chunk % 16], 13) ^ folded;
    __syncwarp();
  }

  if (lane == 0) {
    const uint8_t* jackpot_key =
        config.jackpot_key_device != nullptr ? config.jackpot_key_device
                                             : config.jackpot_key;
    keyed_blake3_64(transcript, jackpot_key, final_hash);
    record_winner(final_hash, candidate, tile_row, tile_col, config, queue);
  }
}

int fused_search_launch(const CandidateBatch& candidates,
                        const DeviceSearchConfig& config,
                        DeviceWinnerQueue* queue, cudaStream_t stream) {
  const uint32_t tiles = (config.m / 16) * (config.n / 16);
  const char* wmma_env = std::getenv("PEARL_SM120_WMMA");
  const bool use_wmma = wmma_env == nullptr || wmma_env[0] != '0';

  if (use_wmma) {
    fused_search_wmma_kernel<<<candidates.count * tiles, 32, 0, stream>>>(
        candidates.a_noised, candidates.b_noised_t, candidates.count, config,
        queue);
  } else {
    fused_search_dp4a_kernel<<<candidates.count * tiles, 256, 0, stream>>>(
        candidates.a_noised, candidates.b_noised_t, candidates.count, config,
        queue);
  }
  return static_cast<int>(cudaGetLastError());
}

}  // namespace pearl::sm120

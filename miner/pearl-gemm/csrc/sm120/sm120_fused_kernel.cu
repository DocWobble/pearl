#include <cstdlib>
#include <cstring>
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

__device__ void record_winner(const uint8_t final_hash[32], uint32_t candidate,
                              uint32_t tile_row, uint32_t tile_col,
                              const DeviceSearchConfig& config,
                              DeviceWinnerQueue* queue) {
  if (!hash_meets_target(final_hash, config.share_target)) return;
  const uint32_t slot = atomicAdd(&queue->winner_count, 1);
  if (slot < kWinnerQueueCapacity) {
    WinnerDescriptor& winner = queue->winners[slot];
    winner.id.generation = config.generation;
    winner.id.candidate_index = config.candidate_base + candidate;
    winner.tile_row = tile_row;
    winner.tile_col = tile_col;
    winner.reconstruction_slot = candidate;
    for (uint32_t i = 0; i < 32; ++i) {
      winner.jackpot_hash.bytes[i] = final_hash[i];
    }
  } else {
    queue->overflow_flag = 1;
  }
}

// Golden correctness path preserved verbatim in the optimized extension so the
// same binary can A/B its arithmetic against faster implementations.
__global__ void fused_search_scalar_kernel(
    const int8_t* candidates_a, const int8_t* b_transposed,
    uint32_t candidate_count, DeviceSearchConfig config,
    DeviceWinnerQueue* queue) {
  const uint32_t lane = threadIdx.x;
  const uint32_t tiles_w = config.n / 16;
  const uint32_t tiles = (config.m / 16) * tiles_w;
  const uint32_t candidate = blockIdx.x / tiles;
  const uint32_t tile = blockIdx.x % tiles;
  if (candidate >= candidate_count || lane >= 256 || config.m % 16 ||
      config.n % 16 || config.rank != 128 || config.k == 0 ||
      config.k > 65536 || config.k % config.rank) return;

  const uint32_t tile_row = tile / tiles_w;
  const uint32_t tile_col = tile % tiles_w;
  const int8_t* a = candidates_a +
      uint64_t(candidate) * config.m * config.k +
      uint64_t(tile_row * 16) * config.k;
  const int8_t* bt = b_transposed + uint64_t(tile_col * 16) * config.k;
  const uint32_t row = lane / 16;
  const uint32_t col = lane % 16;

  int32_t cumulative = 0;
  __shared__ uint32_t folded[256];
  __shared__ uint32_t transcript[16];
  __shared__ uint8_t final_hash[32];

  if (lane < 16) transcript[lane] = 0;
  __syncthreads();

  for (uint32_t p = 0, chunk = 0; p < config.k; p += config.rank, ++chunk) {
    for (uint32_t x = p; x < p + config.rank; ++x) {
      cumulative += int32_t(a[row * config.k + x]) *
                    int32_t(bt[col * config.k + x]);
    }

    folded[lane] = static_cast<uint32_t>(cumulative);
    __syncthreads();
    for (uint32_t stride = 128; stride; stride >>= 1) {
      if (lane < stride) folded[lane] ^= folded[lane + stride];
      __syncthreads();
    }
    if (lane == 0) {
      transcript[chunk % 16] =
          rotl32(transcript[chunk % 16], 13) ^ folded[0];
    }
    __syncthreads();
  }

  if (lane == 0) {
    keyed_blake3_64(transcript, config.jackpot_key, final_hash);
    record_winner(final_hash, candidate, tile_row, tile_col, config, queue);
  }
}

// Exact integer path using four signed INT8 MACs per instruction. The 256
// independent accumulator cells are still one thread each, but rank-boundary
// XOR now reduces in registers and uses only eight shared words.
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
      config.k > 65536 || config.k % config.rank) return;

  const uint32_t tile_row = tile / tiles_w;
  const uint32_t tile_col = tile % tiles_w;
  const int8_t* a = candidates_a +
      uint64_t(candidate) * config.m * config.k +
      uint64_t(tile_row * 16) * config.k;
  const int8_t* bt = b_transposed + uint64_t(tile_col * 16) * config.k;
  const uint32_t row = lane / 16;
  const uint32_t col = lane % 16;

  int32_t cumulative = 0;
  __shared__ uint32_t warp_folded[8];
  __shared__ uint32_t transcript[16];
  __shared__ uint8_t final_hash[32];

  if (lane < 16) transcript[lane] = 0;
  __syncthreads();

  constexpr uint32_t kFullWarpMask = 0xffffffffu;
  for (uint32_t p = 0, chunk = 0; p < config.k; p += 128, ++chunk) {
#pragma unroll
    for (uint32_t x = p; x < p + 128; x += 4) {
      const int packed_a =
          *reinterpret_cast<const int*>(a + uint64_t(row) * config.k + x);
      const int packed_b =
          *reinterpret_cast<const int*>(bt + uint64_t(col) * config.k + x);
      cumulative = __dp4a(packed_a, packed_b, cumulative);
    }

    uint32_t folded = static_cast<uint32_t>(cumulative);
    for (uint32_t offset = 16; offset; offset >>= 1) {
      folded ^= __shfl_down_sync(kFullWarpMask, folded, offset);
    }
    if ((lane & 31u) == 0) warp_folded[lane >> 5] = folded;
    __syncthreads();

    if (lane < 32) {
      uint32_t block_folded = lane < 8 ? warp_folded[lane] : 0u;
      for (uint32_t offset = 16; offset; offset >>= 1) {
        block_folded ^=
            __shfl_down_sync(kFullWarpMask, block_folded, offset);
      }
      if (lane == 0) {
        transcript[chunk % 16] =
            rotl32(transcript[chunk % 16], 13) ^ block_folded;
      }
    }
    __syncthreads();
  }

  if (lane == 0) {
    keyed_blake3_64(transcript, config.jackpot_key, final_hash);
    record_winner(final_hash, candidate, tile_row, tile_col, config, queue);
  }
}

// Tensor-core path. One warp owns one 16x16 Pearl tile. The INT32 WMMA
// accumulator remains live across the complete K dimension. Every 128-wide rank
// boundary is materialized to shared memory only long enough to reproduce the
// protocol-required XOR observation before accumulation continues.
__global__ void fused_search_wmma_kernel(
    const int8_t* __restrict__ candidates_a,
    const int8_t* __restrict__ b_transposed,
    uint32_t candidate_count, DeviceSearchConfig config,
    DeviceWinnerQueue* __restrict__ queue) {
  const uint32_t lane = threadIdx.x;
  const uint32_t tiles_w = config.n / 16;
  const uint32_t tiles = (config.m / 16) * tiles_w;
  const uint32_t candidate = blockIdx.x / tiles;
  const uint32_t tile = blockIdx.x % tiles;
  if (candidate >= candidate_count || lane >= 32 || config.m % 16 ||
      config.n % 16 || config.rank != 128 || config.k == 0 ||
      config.k > 65536 || config.k % 128) return;

  const uint32_t tile_row = tile / tiles_w;
  const uint32_t tile_col = tile % tiles_w;
  const int8_t* a = candidates_a +
      uint64_t(candidate) * config.m * config.k +
      uint64_t(tile_row * 16) * config.k;
  const int8_t* bt = b_transposed + uint64_t(tile_col * 16) * config.k;

  __shared__ __align__(32) signed char smem_a[16 * 16];
  __shared__ __align__(32) signed char smem_b[16 * 16];
  __shared__ __align__(32) int32_t smem_acc[16 * 16];
  __shared__ uint32_t transcript[16];
  __shared__ uint8_t final_hash[32];

  wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char,
                 wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char,
                 wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_frag;
  wmma::fill_fragment(acc_frag, 0);

  if (lane < 16) transcript[lane] = 0;
  __syncwarp();

  for (uint32_t p = 0, chunk = 0; p < config.k; p += 128, ++chunk) {
#pragma unroll
    for (uint32_t kk = 0; kk < 128; kk += 16) {
      if (lane < 16) {
        reinterpret_cast<int4*>(smem_a)[lane] =
            *reinterpret_cast<const int4*>(
                a + uint64_t(lane) * config.k + p + kk);
        reinterpret_cast<int4*>(smem_b)[lane] =
            *reinterpret_cast<const int4*>(
                bt + uint64_t(lane) * config.k + p + kk);
      }
      __syncwarp();

      wmma::load_matrix_sync(a_frag, smem_a, 16);
      wmma::load_matrix_sync(b_frag, smem_b, 16);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::store_matrix_sync(smem_acc, acc_frag, 16, wmma::mem_row_major);
    __syncwarp();

    uint32_t folded = 0;
#pragma unroll
    for (uint32_t i = lane; i < 256; i += 32) {
      folded ^= static_cast<uint32_t>(smem_acc[i]);
    }
    constexpr uint32_t kFullWarpMask = 0xffffffffu;
    for (uint32_t offset = 16; offset; offset >>= 1) {
      folded ^= __shfl_down_sync(kFullWarpMask, folded, offset);
    }

    if (lane == 0) {
      transcript[chunk % 16] =
          rotl32(transcript[chunk % 16], 13) ^ folded;
    }
    __syncwarp();
  }

  if (lane == 0) {
    keyed_blake3_64(transcript, config.jackpot_key, final_hash);
    record_winner(final_hash, candidate, tile_row, tile_col, config, queue);
  }
}

// Macro-tiled tensor-core path.  A thread block owns many adjacent 16x16
// Pearl hash tiles so A and B panels are fetched once and reused across the
// block instead of being re-read independently by every hash tile.
//
// Each warp still owns exactly one protocol-visible 16x16 tile and retains its
// own cumulative INT32 WMMA fragment across K.  The rank-boundary XOR is taken
// directly over the distributed accumulator fragment, so no C tile is written
// to shared/global memory merely to observe parity.
template <int MacroM, int MacroN>
__global__ void fused_search_wmma_macro_kernel(
    const int8_t* __restrict__ candidates_a,
    const int8_t* __restrict__ b_transposed,
    uint32_t candidate_count, DeviceSearchConfig config,
    DeviceWinnerQueue* __restrict__ queue) {
  static_assert(MacroM % 16 == 0 && MacroN % 16 == 0);
  constexpr uint32_t kTilesM = MacroM / 16;
  constexpr uint32_t kTilesN = MacroN / 16;
  constexpr uint32_t kWarps = kTilesM * kTilesN;
  constexpr uint32_t kThreads = kWarps * 32;
  static_assert(kThreads <= 1024);

  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid >> 5;
  const uint32_t lane = tid & 31u;

  const uint32_t total_tiles_m = config.m / 16;
  const uint32_t total_tiles_n = config.n / 16;
  const uint32_t macro_grid_m = (total_tiles_m + kTilesM - 1) / kTilesM;
  const uint32_t macro_grid_n = (total_tiles_n + kTilesN - 1) / kTilesN;
  const uint32_t macros_per_candidate = macro_grid_m * macro_grid_n;
  const uint32_t candidate = blockIdx.x / macros_per_candidate;
  const uint32_t macro = blockIdx.x % macros_per_candidate;

  if (candidate >= candidate_count || config.m % 16 || config.n % 16 ||
      config.rank != 128 || config.k == 0 || config.k > 65536 ||
      config.k % 128) {
    return;
  }

  const uint32_t macro_row = macro / macro_grid_n;
  const uint32_t macro_col = macro % macro_grid_n;
  const uint32_t local_tile_row = warp / kTilesN;
  const uint32_t local_tile_col = warp % kTilesN;
  const uint32_t tile_row = macro_row * kTilesM + local_tile_row;
  const uint32_t tile_col = macro_col * kTilesN + local_tile_col;
  const bool active = warp < kWarps &&
                      tile_row < total_tiles_m &&
                      tile_col < total_tiles_n;

  const int8_t* candidate_a =
      candidates_a + uint64_t(candidate) * config.m * config.k;
  const uint32_t a_row_base = macro_row * MacroM;
  const uint32_t b_col_base = macro_col * MacroN;

  __shared__ __align__(32) signed char smem_a[MacroM * 16];
  __shared__ __align__(32) signed char smem_b[MacroN * 16];
  __shared__ uint32_t final_transcripts[kWarps * 16];
  __shared__ uint8_t final_hashes[kWarps * 32];

  wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char,
                 wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char,
                 wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_frag;
  wmma::fill_fragment(acc_frag, 0);

  // Lane 0..15 each owns one jackpot word for this warp/tile.
  uint32_t jackpot_word = 0;
  constexpr uint32_t kFullWarpMask = 0xffffffffu;

  for (uint32_t p = 0, chunk = 0; p < config.k; p += 128, ++chunk) {
#pragma unroll
    for (uint32_t kk = 0; kk < 128; kk += 16) {
      // Cooperative 16-byte panel loads.  One block load feeds every warp.
      // A has MacroM rows, B^T has MacroN rows.
      if (tid < MacroM) {
        const uint32_t row = a_row_base + tid;
        int4 value = make_int4(0, 0, 0, 0);
        if (row < config.m) {
          value = *reinterpret_cast<const int4*>(
              candidate_a + uint64_t(row) * config.k + p + kk);
        }
        reinterpret_cast<int4*>(smem_a)[tid] = value;
      }
      if (tid >= MacroM && tid < MacroM + MacroN) {
        const uint32_t local_col = tid - MacroM;
        const uint32_t col = b_col_base + local_col;
        int4 value = make_int4(0, 0, 0, 0);
        if (col < config.n) {
          value = *reinterpret_cast<const int4*>(
              b_transposed + uint64_t(col) * config.k + p + kk);
        }
        reinterpret_cast<int4*>(smem_b)[local_col] = value;
      }
      __syncthreads();

      if (active) {
        const signed char* a_tile =
            smem_a + local_tile_row * 16 * 16;
        const signed char* b_tile =
            smem_b + local_tile_col * 16 * 16;
        wmma::load_matrix_sync(a_frag, a_tile, 16);
        wmma::load_matrix_sync(b_frag, b_tile, 16);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      }
      // Every warp must finish reading the panel before it is overwritten.
      __syncthreads();
    }

    if (active) {
      // A WMMA accumulator fragment collectively contains the 256 logical
      // INT32 cells exactly once. XOR is order-independent, so the fragment
      // can be reduced in registers without materializing the C tile.
      uint32_t folded = 0;
#pragma unroll
      for (int i = 0; i < acc_frag.num_elements; ++i) {
        folded ^= static_cast<uint32_t>(acc_frag.x[i]);
      }
      for (uint32_t offset = 16; offset; offset >>= 1) {
        folded ^= __shfl_down_sync(kFullWarpMask, folded, offset);
      }
      const uint32_t xored_tile =
          __shfl_sync(kFullWarpMask, folded, 0);
      if (lane == (chunk & 15u)) {
        jackpot_word = rotl32(jackpot_word, 13) ^ xored_tile;
      }
    }
  }

  if (active) {
    if (lane < 16) {
      final_transcripts[warp * 16 + lane] = jackpot_word;
    }
    __syncwarp();

    if (lane == 0) {
      uint8_t* final_hash = final_hashes + warp * 32;
      keyed_blake3_64(final_transcripts + warp * 16,
                      config.jackpot_key, final_hash);
      record_winner(final_hash, candidate, tile_row, tile_col, config, queue);
    }
  }
}

enum class SearchKernel {
  kScalar,
  kDp4a,
  kWmma,
  kWmma64x64,
  kWmma64x128,
  kWmma128x64,
};

SearchKernel selected_kernel() {
  const char* value = std::getenv("PEARL_SM120_KERNEL");
  if (value == nullptr || value[0] == '\0' || std::strcmp(value, "wmma") == 0) {
    return SearchKernel::kWmma;
  }
  if (std::strcmp(value, "dp4a") == 0) return SearchKernel::kDp4a;
  if (std::strcmp(value, "scalar") == 0) return SearchKernel::kScalar;
  if (std::strcmp(value, "wmma64x64") == 0) return SearchKernel::kWmma64x64;
  if (std::strcmp(value, "wmma64x128") == 0) return SearchKernel::kWmma64x128;
  if (std::strcmp(value, "wmma128x64") == 0) return SearchKernel::kWmma128x64;
  // Unknown values fail conservatively into the golden arithmetic path.
  return SearchKernel::kScalar;
}

int fused_search_launch(const CandidateBatch& candidates,
                        const DeviceSearchConfig& config,
                        DeviceWinnerQueue* queue, cudaStream_t stream) {
  const uint32_t tiles = (config.m / 16) * (config.n / 16);
  const uint32_t blocks = candidates.count * tiles;

  switch (selected_kernel()) {
    case SearchKernel::kWmma:
      fused_search_wmma_kernel<<<blocks, 32, 0, stream>>>(
          candidates.a_noised, candidates.b_noised_t, candidates.count, config,
          queue);
      break;
    case SearchKernel::kDp4a:
      fused_search_dp4a_kernel<<<blocks, 256, 0, stream>>>(
          candidates.a_noised, candidates.b_noised_t, candidates.count, config,
          queue);
      break;
    case SearchKernel::kWmma64x64: {
      constexpr uint32_t kTilesM = 4, kTilesN = 4;
      const uint32_t grid_m = ((config.m / 16) + kTilesM - 1) / kTilesM;
      const uint32_t grid_n = ((config.n / 16) + kTilesN - 1) / kTilesN;
      const uint32_t macro_blocks = candidates.count * grid_m * grid_n;
      fused_search_wmma_macro_kernel<64, 64>
          <<<macro_blocks, 512, 0, stream>>>(
              candidates.a_noised, candidates.b_noised_t, candidates.count,
              config, queue);
      break;
    }
    case SearchKernel::kWmma64x128: {
      constexpr uint32_t kTilesM = 4, kTilesN = 8;
      const uint32_t grid_m = ((config.m / 16) + kTilesM - 1) / kTilesM;
      const uint32_t grid_n = ((config.n / 16) + kTilesN - 1) / kTilesN;
      const uint32_t macro_blocks = candidates.count * grid_m * grid_n;
      fused_search_wmma_macro_kernel<64, 128>
          <<<macro_blocks, 1024, 0, stream>>>(
              candidates.a_noised, candidates.b_noised_t, candidates.count,
              config, queue);
      break;
    }
    case SearchKernel::kWmma128x64: {
      constexpr uint32_t kTilesM = 8, kTilesN = 4;
      const uint32_t grid_m = ((config.m / 16) + kTilesM - 1) / kTilesM;
      const uint32_t grid_n = ((config.n / 16) + kTilesN - 1) / kTilesN;
      const uint32_t macro_blocks = candidates.count * grid_m * grid_n;
      fused_search_wmma_macro_kernel<128, 64>
          <<<macro_blocks, 1024, 0, stream>>>(
              candidates.a_noised, candidates.b_noised_t, candidates.count,
              config, queue);
      break;
    }
    case SearchKernel::kScalar:
    default:
      fused_search_scalar_kernel<<<blocks, 256, 0, stream>>>(
          candidates.a_noised, candidates.b_noised_t, candidates.count, config,
          queue);
      break;
  }

  return static_cast<int>(cudaGetLastError());
}

}  // namespace pearl::sm120

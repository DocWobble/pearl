#include <cuda_runtime.h>
#include <cstdint>
#include "sm120_transcript.cuh"

namespace pearl::sm120 {

// One CUDA block evaluates one legal 16x16 hash tile.  This is the correctness fallback:
// signed int8 products accumulate exactly into signed int32 and each full rank chunk updates
// the same transcript definition as miner_base.noisy_gemm.Transcript.
__global__ void dense_reference_tile(const int8_t* a, const int8_t* bt, uint32_t k,
                                     uint32_t rank, uint32_t* transcript_out) {
  const uint32_t lane = threadIdx.x;
  if (lane >= 256 || rank != 128 || k == 0 || k > 65536 || k % rank != 0) return;
  const uint32_t row = lane / 16;
  const uint32_t col = lane % 16;
  int32_t cumulative = 0;
  __shared__ uint32_t folded[256];
  __shared__ uint32_t transcript[16];
  if (lane < 16) transcript[lane] = 0;
  __syncthreads();
  for (uint32_t p = 0, chunk = 0; p < k; p += rank, ++chunk) {
    for (uint32_t x = p; x < p + rank; ++x)
      cumulative += int32_t(a[row * k + x]) * int32_t(bt[col * k + x]);
    folded[lane] = static_cast<uint32_t>(cumulative);
    __syncthreads();
    for (uint32_t stride = 128; stride; stride >>= 1) {
      if (lane < stride) folded[lane] ^= folded[lane + stride];
      __syncthreads();
    }
    if (lane == 0) transcript[chunk % 16] = rotl32(transcript[chunk % 16], 13) ^ folded[0];
    __syncthreads();
  }
  if (lane < 16) transcript_out[lane] = transcript[lane];
}

int dense_reference_tile_launch(const int8_t* a, const int8_t* b_transposed,
                                uint32_t k, uint32_t rank, uint32_t* transcript_out) {
  dense_reference_tile<<<1, 256>>>(a, b_transposed, k, rank, transcript_out);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace pearl::sm120

#include <cuda_runtime.h>
#include <cstdint>
#include "sm120_blake3.cuh"

namespace pearl::sm120 {

// Exact direct materialization for the zero-base branch.  It is intentionally separate from
// the search kernel so no consensus-relevant noising is fused before vector parity passes.
__global__ void add_noise_i8(const int8_t* base, const int8_t* noise, int8_t* out, uint64_t count) {
  const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) out[index] = static_cast<int8_t>(int(base[index]) + int(noise[index]));
}

__global__ void add_noise_candidates_i8(const int8_t* base, const int8_t* noise, int8_t* out,
                                        uint64_t elements_per_candidate, uint32_t candidate_count) {
  const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t count = elements_per_candidate * candidate_count;
  if (index < count) out[index] = static_cast<int8_t>(int(base[index]) + int(noise[index % elements_per_candidate]));
}

__global__ void repeat_candidates_i8(const int8_t* source, int8_t* out,
                                     uint64_t elements_per_candidate, uint32_t candidate_count) {
  const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t count = elements_per_candidate * candidate_count;
  if (index < count) out[index] = source[index % elements_per_candidate];
}

namespace {

__device__ __constant__ uint8_t kLabelA[32] = {'A','_','t','e','n','s','o','r'};
__device__ __constant__ uint8_t kLabelB[32] = {'B','_','t','e','n','s','o','r'};

__device__ void random_hash(uint32_t index, const uint8_t seed[32], const uint8_t key[32],
                            uint32_t prepend_index, uint8_t out[32]) {
  uint32_t message[16]{};
  message[prepend_index] = index + 1;
  for (uint32_t i = 0; i < 8; ++i) {
    message[8 + i] = uint32_t(seed[4 * i]) | uint32_t(seed[4 * i + 1]) << 8 |
                     uint32_t(seed[4 * i + 2]) << 16 | uint32_t(seed[4 * i + 3]) << 24;
  }
  keyed_blake3_64(message, key, out);
}

__device__ int8_t uniform_value(uint32_t row, uint32_t col, uint32_t rank,
                                const uint8_t label[32], const uint8_t key[32]) {
  const uint32_t linear = row * rank + col;
  uint8_t hash[32];
  random_hash(linear / 32, label, key, 0, hash);
  return static_cast<int8_t>(hash[linear % 32] & 63u) - 32;
}

__device__ uint32_t permutation_word(uint32_t reduction_index, const uint8_t label[32], const uint8_t key[32]) {
  uint8_t hash[32];
  random_hash(reduction_index / 8, label, key, 1, hash);
  const uint32_t offset = (reduction_index % 8) * 4;
  return uint32_t(hash[offset]) | uint32_t(hash[offset + 1]) << 8 |
         uint32_t(hash[offset + 2]) << 16 | uint32_t(hash[offset + 3]) << 24;
}

__device__ int8_t noise_value(uint32_t matrix_index, uint32_t reduction_index, uint32_t rank,
                              const uint8_t label[32], const uint8_t key[32]) {
  const uint32_t random = permutation_word(reduction_index, label, key);
  const uint32_t first = random & (rank - 1);
  const uint32_t second = first ^ (1u + uint32_t((uint64_t(rank - 1) * random) >> 32));
  return static_cast<int8_t>(int(uniform_value(matrix_index, first, rank, label, key)) -
                             int(uniform_value(matrix_index, second, rank, label, key)));
}

__global__ void generate_noise_a_kernel(const uint8_t* seed, const uint32_t* rows,
                                        uint32_t row_count, uint32_t k, uint32_t rank, int8_t* out) {
  const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t count = uint64_t(row_count) * k;
  if (index >= count) return;
  const uint32_t row_slot = index / k;
  out[index] = noise_value(rows[row_slot], index % k, rank, kLabelA, seed);
}

__global__ void generate_noise_b_kernel(const uint8_t* seed, const uint32_t* cols,
                                        uint32_t col_count, uint32_t k, uint32_t rank, int8_t* out) {
  const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t count = uint64_t(col_count) * k;
  if (index >= count) return;
  const uint32_t col_slot = index / k;
  out[index] = noise_value(cols[col_slot], index % k, rank, kLabelB, seed);
}

}  // namespace

int generate_noise_launch(const uint8_t* a_noise_seed, const uint8_t* b_noise_seed,
                          const uint32_t* a_rows, uint32_t a_row_count,
                          const uint32_t* b_cols, uint32_t b_col_count,
                          uint32_t k, uint32_t rank, int8_t* noise_a, int8_t* noise_b_t,
                          cudaStream_t stream) {
  constexpr uint32_t threads = 128;
  const uint64_t a_count = uint64_t(a_row_count) * k;
  const uint64_t b_count = uint64_t(b_col_count) * k;
  generate_noise_a_kernel<<<(a_count + threads - 1) / threads, threads, 0, stream>>>(
      a_noise_seed, a_rows, a_row_count, k, rank, noise_a);
  if (cudaGetLastError() != cudaSuccess) return 1;
  generate_noise_b_kernel<<<(b_count + threads - 1) / threads, threads, 0, stream>>>(
      b_noise_seed, b_cols, b_col_count, k, rank, noise_b_t);
  return static_cast<int>(cudaGetLastError());
}

int add_noise_candidates_launch(const int8_t* base, const int8_t* noise, int8_t* out,
                                uint64_t elements_per_candidate, uint32_t candidate_count,
                                cudaStream_t stream) {
  constexpr uint32_t threads = 256;
  const uint64_t count = elements_per_candidate * candidate_count;
  add_noise_candidates_i8<<<(count + threads - 1) / threads, threads, 0, stream>>>(
      base, noise, out, elements_per_candidate, candidate_count);
  return static_cast<int>(cudaGetLastError());
}

int repeat_candidates_launch(const int8_t* source, int8_t* out,
                             uint64_t elements_per_candidate, uint32_t candidate_count,
                             cudaStream_t stream) {
  constexpr uint32_t threads = 256;
  const uint64_t count = elements_per_candidate * candidate_count;
  repeat_candidates_i8<<<(count + threads - 1) / threads, threads, 0, stream>>>(
      source, out, elements_per_candidate, candidate_count);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace pearl::sm120

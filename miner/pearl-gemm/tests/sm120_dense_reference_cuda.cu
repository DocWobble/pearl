#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include "sm120_backend.h"
#include "sm120_blake3.cuh"
#include "sm120_transcript.cuh"

__global__ void blake3_probe(const uint32_t* message, const uint8_t* key, uint8_t* output) {
  pearl::sm120::keyed_blake3_64(message, key, output);
}

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
  constexpr uint32_t k = 2048;
  std::array<int8_t, 16 * k> host_a{};
  std::array<int8_t, 16 * k> host_bt{};
  for (uint32_t i = 0; i < host_a.size(); ++i) host_a[i] = int8_t((i * 17) % 127 - 63);
  for (uint32_t i = 0; i < host_bt.size(); ++i) host_bt[i] = int8_t((i * 29) % 127 - 63);
  int8_t *a = nullptr, *bt = nullptr;
  uint32_t* actual = nullptr;
  assert(cudaMalloc(&a, host_a.size()) == cudaSuccess);
  assert(cudaMalloc(&bt, host_bt.size()) == cudaSuccess);
  assert(cudaMalloc(&actual, 16 * sizeof(uint32_t)) == cudaSuccess);
  assert(cudaMemcpy(a, host_a.data(), host_a.size(), cudaMemcpyHostToDevice) == cudaSuccess);
  assert(cudaMemcpy(bt, host_bt.data(), host_bt.size(), cudaMemcpyHostToDevice) == cudaSuccess);
  assert(pearl::sm120::dense_reference_tile_launch(a, bt, k, 128, actual) == int(cudaSuccess));
  std::array<uint32_t, 16> observed{};
  assert(cudaMemcpy(observed.data(), actual, sizeof(observed), cudaMemcpyDeviceToHost) == cudaSuccess);
  std::array<uint32_t, 16> expected_transcript{};
  std::array<int32_t, 16 * 16> cumulative{};
  for (uint32_t chunk = 0; chunk < k / 128; ++chunk) {
    for (uint32_t row = 0; row < 16; ++row) for (uint32_t col = 0; col < 16; ++col) {
      auto& sum = cumulative[row * 16 + col];
      for (uint32_t x = chunk * 128; x < (chunk + 1) * 128; ++x)
        sum += int32_t(host_a[row*k+x]) * int32_t(host_bt[col*k+x]);
    }
    uint32_t folded = 0;
    for (const int32_t sum : cumulative) folded ^= static_cast<uint32_t>(sum);
    expected_transcript[chunk % 16] = pearl::sm120::rotl32(expected_transcript[chunk % 16], 13) ^ folded;
  }
  for (uint32_t i = 0; i < 16; ++i) assert(observed[i] == expected_transcript[i]);
  uint32_t host_words[16]; uint8_t host_key[32]; uint8_t host_digest[32]{};
  for (uint32_t i = 0; i < 32; ++i) host_key[i] = uint8_t(i);
  for (uint32_t i = 0; i < 16; ++i) {
    host_words[i] = uint32_t(uint8_t(4*i*37+11)) | uint32_t(uint8_t((4*i+1)*37+11)) << 8 |
                    uint32_t(uint8_t((4*i+2)*37+11)) << 16 | uint32_t(uint8_t((4*i+3)*37+11)) << 24;
  }
  uint32_t* d_words = nullptr; uint8_t *d_key = nullptr, *d_digest = nullptr;
  assert(cudaMalloc(&d_words, sizeof(host_words)) == cudaSuccess && cudaMalloc(&d_key, sizeof(host_key)) == cudaSuccess && cudaMalloc(&d_digest, sizeof(host_digest)) == cudaSuccess);
  cudaMemcpy(d_words, host_words, sizeof(host_words), cudaMemcpyHostToDevice); cudaMemcpy(d_key, host_key, sizeof(host_key), cudaMemcpyHostToDevice);
  blake3_probe<<<1, 1>>>(d_words, d_key, d_digest); assert(cudaGetLastError() == cudaSuccess);
  assert(cudaMemcpy(host_digest, d_digest, sizeof(host_digest), cudaMemcpyDeviceToHost) == cudaSuccess);
  const uint8_t expected_digest[32] = {0xc7,0x8c,0x38,0x1a,0x60,0x92,0xf1,0x7d,0x12,0xa0,0x1f,0xf0,0x32,0xe5,0x2e,0x83,0xcf,0x0f,0x93,0x45,0x67,0xda,0x16,0xe8,0xcb,0x3b,0x76,0xd1,0x05,0x80,0x67,0x6a};
  for (uint32_t i = 0; i < 32; ++i) assert(host_digest[i] == expected_digest[i]);
  cudaFree(d_digest); cudaFree(d_key); cudaFree(d_words);
  setenv("PEARL_SM120_BACKEND", "1", 1);
  setenv("PEARL_SM120_FUSED", "1", 1);
  setenv("PEARL_SM120_SEARCH_ONLY", "1", 1);
  pearl::sm120::JobContext job{};
  job.generation = 17; job.m = 16; job.n = 16; job.k = k; job.rank = 128;
  for (uint32_t i = 0; i < 32; ++i) { job.jackpot_key.bytes[i] = host_key[i]; job.share_target.little_endian[i] = 0xff; }
  pearl::sm120::CandidateBatch batch{a, bt, 1};
  pearl::sm120::DeviceScratch scratch;
  const auto search = pearl::sm120::search_job(job, batch, scratch);
  assert(search.metrics.cuda_errors == 0 && search.metrics.candidates == 1 && search.metrics.winner_count == 1);
  assert(search.winners[0].id.generation == 17 && search.winners[0].id.candidate_index == 0);
  const auto* queue_first = scratch.winner_queue;
  const auto repeated = pearl::sm120::search_job(job, batch, scratch);
  assert(repeated.metrics.cuda_errors == 0 && repeated.metrics.winner_count == 1);
  assert(scratch.winner_queue == queue_first && queue_first != nullptr);
  cudaFree(actual); cudaFree(bt); cudaFree(a);
  return 0;
}

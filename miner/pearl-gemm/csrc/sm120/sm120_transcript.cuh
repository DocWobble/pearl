#pragma once

#include <cstdint>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

namespace pearl::sm120 {

constexpr uint32_t kJackpotWords = 16;
constexpr uint32_t kHashAccumulateRotation = 13;

__host__ __device__ constexpr uint32_t rotl32(uint32_t value, uint32_t amount) {
  return (value << amount) | (value >> (32U - amount));
}

struct Transcript {
  uint32_t words[kJackpotWords]{};

  __host__ __device__ void update(uint32_t rank_chunk, uint32_t xored_tile) {
    const uint32_t index = rank_chunk % kJackpotWords;
    words[index] = rotl32(words[index], kHashAccumulateRotation) ^ xored_tile;
  }
};

}  // namespace pearl::sm120

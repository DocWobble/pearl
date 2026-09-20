#pragma once

#include <cstdint>

#if !defined(__CUDACC__) && !defined(__host__)
#define __host__
#endif
#if !defined(__CUDACC__) && !defined(__device__)
#define __device__
#endif

namespace pearl::sm120 {

// Canonical BLAKE3's single-chunk, 64-byte keyed mode.  Pearl's jackpot is exactly sixteen
// little-endian u32 words, so this is the exact `blake3(jackpot, key=commitment_hash)` path;
// there is no tree reduction or output truncation in this case.  Constants and the seven-round
// schedule match miner/pearl-gemm/csrc/blake3/blake3.cuh and pearl-blake3.
constexpr bool kCanonicalBlake3Wired = true;
constexpr uint32_t kBlake3Iv[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                                   0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
constexpr uint32_t kBlake3KeyedHash = 16u;
constexpr uint32_t kBlake3ChunkStart = 1u;
constexpr uint32_t kBlake3ChunkEnd = 2u;
constexpr uint32_t kBlake3Root = 8u;

__host__ __device__ inline uint32_t rotr32(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32u - n));
}

__host__ __device__ inline void blake3_g(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
                                         uint32_t mx, uint32_t my) {
  a = a + b + mx; d = rotr32(d ^ a, 16); c += d; b = rotr32(b ^ c, 12);
  a = a + b + my; d = rotr32(d ^ a, 8);  c += d; b = rotr32(b ^ c, 7);
}

__host__ __device__ inline void blake3_round(uint32_t state[16], const uint32_t m[16]) {
  blake3_g(state[0], state[4], state[8], state[12], m[0], m[1]);
  blake3_g(state[1], state[5], state[9], state[13], m[2], m[3]);
  blake3_g(state[2], state[6], state[10], state[14], m[4], m[5]);
  blake3_g(state[3], state[7], state[11], state[15], m[6], m[7]);
  blake3_g(state[0], state[5], state[10], state[15], m[8], m[9]);
  blake3_g(state[1], state[6], state[11], state[12], m[10], m[11]);
  blake3_g(state[2], state[7], state[8], state[13], m[12], m[13]);
  blake3_g(state[3], state[4], state[9], state[14], m[14], m[15]);
}

__host__ __device__ inline void keyed_blake3_64(const uint32_t message[16], const uint8_t key[32],
                                                 uint8_t output[32]) {
  uint32_t state[16];
  uint32_t block[16];
  for (uint32_t i = 0; i < 8; ++i) {
    state[i] = uint32_t(key[4*i]) | (uint32_t(key[4*i+1]) << 8) |
               (uint32_t(key[4*i+2]) << 16) | (uint32_t(key[4*i+3]) << 24);
    // Keep the IV as a function-local literal array so nvcc materializes it for device code.
    constexpr uint32_t iv[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                                0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
    state[i + 8] = iv[i];
  }
  state[12] = 0; state[13] = 0; state[14] = 64;
  state[15] = kBlake3KeyedHash | kBlake3ChunkStart | kBlake3ChunkEnd | kBlake3Root;
  for (uint32_t i = 0; i < 16; ++i) block[i] = message[i];
  for (uint32_t round = 0; round < 7; ++round) {
    blake3_round(state, block);
    if (round != 6) {
      const uint32_t previous[16] = {block[0], block[1], block[2], block[3], block[4], block[5], block[6], block[7],
                                     block[8], block[9], block[10], block[11], block[12], block[13], block[14], block[15]};
      constexpr uint32_t p[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
      for (uint32_t i = 0; i < 16; ++i) block[i] = previous[p[i]];
    }
  }
  for (uint32_t i = 0; i < 8; ++i) {
    const uint32_t word = state[i] ^ state[i + 8];
    output[4*i] = uint8_t(word); output[4*i+1] = uint8_t(word >> 8);
    output[4*i+2] = uint8_t(word >> 16); output[4*i+3] = uint8_t(word >> 24);
  }
}

}  // namespace pearl::sm120

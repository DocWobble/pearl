#include <array>
#include <cassert>
#include <cstdint>
#include "sm120_blake3.cuh"

int main() {
  constexpr std::array<uint8_t, 32> expected = {
      0xc7, 0x8c, 0x38, 0x1a, 0x60, 0x92, 0xf1, 0x7d, 0x12, 0xa0, 0x1f, 0xf0, 0x32, 0xe5, 0x2e, 0x83,
      0xcf, 0x0f, 0x93, 0x45, 0x67, 0xda, 0x16, 0xe8, 0xcb, 0x3b, 0x76, 0xd1, 0x05, 0x80, 0x67, 0x6a};
  uint8_t key[32];
  uint32_t words[16];
  for (uint32_t i = 0; i < 32; ++i) key[i] = uint8_t(i);
  for (uint32_t i = 0; i < 16; ++i) {
    uint8_t bytes[4];
    for (uint32_t j = 0; j < 4; ++j) bytes[j] = uint8_t((4*i+j) * 37 + 11);
    words[i] = uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 | uint32_t(bytes[2]) << 16 | uint32_t(bytes[3]) << 24;
  }
  uint8_t actual[32]{};
  pearl::sm120::keyed_blake3_64(words, key, actual);
  for (uint32_t i = 0; i < 32; ++i) assert(actual[i] == expected[i]);
  return 0;
}

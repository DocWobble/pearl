#include <cassert>
#include <cstdint>
#include "sm120_transcript.cuh"

int main() {
  using namespace pearl::sm120;
  Transcript t{};
  for (uint32_t chunk = 0; chunk < 64; ++chunk) t.update(chunk, chunk * 0x9e3779b9u);
  // Reproduce the calculation independently so a constant/layout regression is caught.
  uint32_t expected[16]{};
  for (uint32_t chunk = 0; chunk < 64; ++chunk) {
    const uint32_t i = chunk % 16;
    expected[i] = (expected[i] << 13 | expected[i] >> 19) ^ (chunk * 0x9e3779b9u);
  }
  for (uint32_t i = 0; i < 16; ++i) assert(t.words[i] == expected[i]);
  return 0;
}

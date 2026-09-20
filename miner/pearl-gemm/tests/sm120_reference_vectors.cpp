#include <array>
#include <cassert>
#include <cstdint>
#include "sm120_transcript.cuh"

int main() {
  using namespace pearl::sm120;
  static_assert(kJackpotWords == 16);
  static_assert(kHashAccumulateRotation == 13);
  assert(rotl32(0x80000000u, 1) == 1u);
  Transcript transcript{};
  transcript.update(0, 0x11223344u);
  assert(transcript.words[0] == 0x11223344u);
  transcript.update(16, 0x01020304u);
  assert(transcript.words[0] == (rotl32(0x11223344u, 13) ^ 0x01020304u));
  return 0;
}

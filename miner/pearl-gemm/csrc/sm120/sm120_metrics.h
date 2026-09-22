#pragma once

#include <cstdint>

namespace pearl::sm120 {

struct Metrics {
  uint64_t candidates{};
  uint64_t target_tests{};
  uint64_t valid_candidate_work{};
  uint64_t winner_count{};
  uint64_t overflow_count{};
  uint64_t cuda_errors{};
  uint64_t b_cache_hits{};
  uint64_t b_cache_misses{};
  uint64_t tail_d8{};
  uint64_t tail_d12{};
  uint64_t tail_d16{};
  uint64_t tail_d20{};
  uint64_t tail_d24{};
};

}  // namespace pearl::sm120

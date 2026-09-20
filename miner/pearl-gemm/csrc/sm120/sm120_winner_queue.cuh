#pragma once

#include <cstdint>
#include "sm120_job_state.h"

namespace pearl::sm120 {

constexpr uint32_t kWinnerQueueCapacity = 64;

struct DeviceWinnerQueue {
  uint32_t winner_count{};
  uint32_t overflow_flag{};
  WinnerDescriptor winners[kWinnerQueueCapacity]{};
};

struct DeviceSearchConfig {
  uint32_t m{}, n{}, k{}, rank{};
  uint64_t generation{}, candidate_base{};
  uint8_t jackpot_key[32]{};
  uint8_t share_target[32]{};
};

}  // namespace pearl::sm120

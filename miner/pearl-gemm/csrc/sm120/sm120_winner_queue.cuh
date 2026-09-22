#pragma once

#include <cstdint>
#include "sm120_job_state.h"

namespace pearl::sm120 {

constexpr uint32_t kWinnerQueueCapacity = 64;

struct DeviceWinnerQueue {
  uint32_t winner_count{};
  uint32_t overflow_flag{};
  uint64_t tail_d8{};
  uint64_t tail_d12{};
  uint64_t tail_d16{};
  uint64_t tail_d20{};
  uint64_t tail_d24{};
  WinnerDescriptor winners[kWinnerQueueCapacity]{};
};

struct DeviceSearchConfig {
  uint32_t m{}, n{}, k{}, rank{};
  uint64_t generation{}, candidate_base{};
  uint8_t jackpot_key[32]{};
  uint8_t share_target[32]{};
  const uint8_t* jackpot_key_device{};
  const uint8_t* share_target_device{};
};

}  // namespace pearl::sm120

#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>

#include "sm120_job_state.h"

namespace pearl::sm120 {

// Owns the single submit-eligible job generation. GPU work may finish after a replacement,
// but its descriptors are never allowed past either of these checks.
class JobGenerationManager {
 public:
  std::shared_ptr<const JobContext> publish(JobContext next_job);
  void disconnect();
  std::shared_ptr<const JobContext> active() const;
  bool allow_proof_construction(const WinnerDescriptor& winner) const;
  bool allow_submission(const WinnerDescriptor& winner) const;

 private:
  bool matches_active_locked(const WinnerDescriptor& winner) const;

  mutable std::mutex mutex_;
  uint64_t next_generation_{};
  bool submission_enabled_{};
  std::shared_ptr<const JobContext> active_;
  // Keep the active job plus the two immediately preceding immutable contexts alive.
  std::deque<std::shared_ptr<const JobContext>> history_;
};

}  // namespace pearl::sm120

#include "sm120_job_manager.h"

namespace pearl::sm120 {

std::shared_ptr<const JobContext> JobGenerationManager::publish(JobContext next_job) {
  std::lock_guard<std::mutex> lock(mutex_);
  next_job.generation = ++next_generation_;
  active_ = std::make_shared<const JobContext>(std::move(next_job));
  history_.push_back(active_);
  while (history_.size() > 3) history_.pop_front();
  submission_enabled_ = true;
  return active_;
}

void JobGenerationManager::disconnect() {
  std::lock_guard<std::mutex> lock(mutex_);
  submission_enabled_ = false;
}

std::shared_ptr<const JobContext> JobGenerationManager::active() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return active_;
}

bool JobGenerationManager::matches_active_locked(const WinnerDescriptor& winner) const {
  return submission_enabled_ && active_ != nullptr && winner.id.generation == active_->generation;
}

bool JobGenerationManager::allow_proof_construction(const WinnerDescriptor& winner) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return matches_active_locked(winner);
}

bool JobGenerationManager::allow_submission(const WinnerDescriptor& winner) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return matches_active_locked(winner);
}

}  // namespace pearl::sm120

#include <cassert>
#include <cstdlib>
#include "sm120_backend.h"
#include "sm120_job_manager.h"

int main() {
  using namespace pearl::sm120;
  JobGenerationManager manager;
  const auto old_job = manager.publish(JobContext{});
  WinnerDescriptor old_winner{};
  old_winner.id.generation = old_job->generation;
  assert(manager.allow_proof_construction(old_winner));
  assert(manager.allow_submission(old_winner));

  const auto active_job = manager.publish(JobContext{});
  assert(active_job->generation == old_job->generation + 1);
  assert(!manager.allow_proof_construction(old_winner));
  assert(!manager.allow_submission(old_winner));

  WinnerDescriptor active_winner{};
  active_winner.id.generation = active_job->generation;
  assert(manager.allow_proof_construction(active_winner));
  assert(manager.allow_submission(active_winner));
  manager.disconnect();
  assert(!manager.allow_proof_construction(active_winner));
  assert(!manager.allow_submission(active_winner));

  setenv("PEARL_SM120_CACHE_B", "1", 1);
  pearl::sm120::DeviceScratch scratch;
  pearl::sm120::CandidateBatch batch{};
  batch.b_identity = 0x1234;
  assert(!scratch.observe_b(*active_job, batch));
  assert(scratch.observe_b(*active_job, batch));
  const auto next_job = manager.publish(JobContext{});
  assert(!scratch.observe_b(*next_job, batch));
  return 0;
}

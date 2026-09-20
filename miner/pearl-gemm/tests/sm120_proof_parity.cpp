#include <cassert>
#include "sm120_backend.h"
#include "sm120_blake3.cuh"

int main() {
  using namespace pearl::sm120;
  // This is a submission guard, not a claim of proof parity. Proof parity stays closed
  // until fixtures are constructed and accepted by Pearl's Rust verifier.
  assert(kCanonicalBlake3Wired);
  JobContext job{};
  CandidateBatch batch{};
  assert(search_job(job, batch).metrics.winner_count == 0);
  return 0;
}

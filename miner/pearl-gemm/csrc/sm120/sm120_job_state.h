#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pearl::sm120 {

struct Uint256 { uint8_t little_endian[32]{}; };
struct Hash256 { uint8_t bytes[32]{}; };

// This object is copied into a shared_ptr and never mutated after publication.
struct JobContext {
  uint64_t generation{};
  std::string job_id;
  std::vector<uint8_t> header_bytes;
  std::vector<uint8_t> mining_config_bytes;
  uint32_t certificate_version{};
  uint32_t m{}, n{}, k{}, rank{128};
  Uint256 block_target{}, share_target{};
  Hash256 job_key{};
  Hash256 jackpot_key{};
  Hash256 hash_b{};
  Hash256 b_noise_seed{};
  Uint256 bound_b{};
  // Serialized periodic-pattern offsets are retained with the immutable job so a
  // proof adapter cannot accidentally reconstruct a winner under a different pattern.
  std::vector<int32_t> rows_pattern;
  std::vector<int32_t> cols_pattern;
  uint64_t dependency_fingerprint{};
};

struct CandidateIdentity {
  uint64_t generation{};
  uint64_t candidate_index{};
  Hash256 hash_a{}, hash_b{};
  uint32_t t_rows{}, t_cols{};
};

struct WinnerDescriptor {
  CandidateIdentity id;
  Hash256 jackpot_hash{};
  uint32_t tile_row{}, tile_col{}, reconstruction_slot{};
};

struct BCacheKey {
  uint64_t generation{};
  Hash256 job_key{};
  Hash256 hash_b{};
  Uint256 bound_b{};
  uint32_t n{}, k{}, rank{}, certificate_version{};
  Hash256 b_noise_seed{};
  uint64_t b_identity{};
  bool operator==(const BCacheKey& other) const;
};

struct SubmissionRecord {
  CandidateIdentity id;
  Uint256 applicable_share_target{};
  uint64_t found_monotonic_ns{}, submitted_monotonic_ns{}, response_monotonic_ns{};
  enum class Result { kAccepted, kStale, kRejected, kDisconnected } result{Result::kDisconnected};
  std::string reason;
};

}  // namespace pearl::sm120

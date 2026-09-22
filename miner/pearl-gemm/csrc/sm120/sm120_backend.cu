#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "sm120_backend.h"

namespace pearl::sm120 {

bool runtime_enabled() {
  const char* value = std::getenv("PEARL_SM120_BACKEND");
  if (value == nullptr || value[0] != '1') return false;
  int device = 0;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess) return false;
  return properties.major == 12 && properties.minor == 0;
}

namespace {

bool flag_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] == '1';
}

SearchResult search_job_impl(const JobContext& job, const CandidateBatch& candidates,
                             DeviceScratch& scratch,
                             const uint8_t* jackpot_key_device,
                             const uint8_t* share_target_device,
                             cudaStream_t caller_stream) {
  SearchResult result{};
  const bool reuse_alloc = flag_enabled("PEARL_SM120_REUSE_ALLOC");
  if (!reuse_alloc) scratch.release();
  if (!runtime_enabled() || job.rank != 128 || job.m == 0 || job.n == 0 ||
      job.m % 16 || job.n % 16 || job.k < 16 * job.rank || job.k > 65536 ||
      job.k % job.rank != 0 || candidates.a_noised == nullptr ||
      candidates.b_noised_t == nullptr || !flag_enabled("PEARL_SM120_FUSED") ||
      !flag_enabled("PEARL_SM120_SEARCH_ONLY")) {
    if (!reuse_alloc) scratch.release();
    return result;
  }

  DeviceWinnerQueue host_queue{};
  if (!scratch.reserve(job.m, job.n, job.k)) {
    result.metrics.cuda_errors = 1;
    if (!reuse_alloc) scratch.release();
    return result;
  }

  CandidateBatch search_candidates = candidates;
  if (flag_enabled("PEARL_SM120_CACHE_B")) {
    if (scratch.observe_b(job, candidates)) result.metrics.b_cache_hits = 1;
    else result.metrics.b_cache_misses = 1;
    if (scratch.cached_b() == nullptr) {
      result.metrics.cuda_errors = 1;
      if (!reuse_alloc) scratch.release();
      return result;
    }
    search_candidates.b_noised_t = scratch.cached_b();
  }

  const cudaStream_t stream =
      caller_stream != nullptr ? caller_stream :
      (scratch.streams_enabled ? scratch.stream_compute : nullptr);
  constexpr size_t kQueueHeaderBytes = offsetof(DeviceWinnerQueue, winners);

  if (cudaMemsetAsync(scratch.winner_queue, 0, kQueueHeaderBytes, stream) != cudaSuccess) {
    result.metrics.cuda_errors = 1;
    if (!reuse_alloc) scratch.release();
    return result;
  }

  DeviceSearchConfig config{};
  config.m = job.m;
  config.n = job.n;
  config.k = job.k;
  config.rank = job.rank;
  config.generation = job.generation;
  config.jackpot_key_device = jackpot_key_device;
  config.share_target_device = share_target_device;
  if (jackpot_key_device == nullptr || share_target_device == nullptr) {
    for (uint32_t i = 0; i < 32; ++i) {
      config.jackpot_key[i] = job.jackpot_key.bytes[i];
      config.share_target[i] = job.share_target.little_endian[i];
    }
  }

  bool failed =
      fused_search_launch(search_candidates, config, scratch.winner_queue, stream) != 0 ||
      cudaStreamSynchronize(stream) != cudaSuccess ||
      cudaMemcpy(&host_queue, scratch.winner_queue, kQueueHeaderBytes,
                 cudaMemcpyDeviceToHost) != cudaSuccess;

  if (!failed && host_queue.winner_count > 0) {
    const uint32_t copied =
        host_queue.winner_count < kWinnerQueueCapacity ? host_queue.winner_count
                                                       : kWinnerQueueCapacity;
    const auto* device_winners = reinterpret_cast<const char*>(scratch.winner_queue) +
                                 offsetof(DeviceWinnerQueue, winners);
    if (cudaMemcpy(host_queue.winners, device_winners,
                   sizeof(WinnerDescriptor) * copied,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      failed = true;
    }
  }

  if (failed) {
    result.metrics.cuda_errors = 1;
  } else {
    result.metrics.candidates = candidates.count;
    result.metrics.target_tests =
        uint64_t(candidates.count) * (job.m / 16) * (job.n / 16);
    result.metrics.valid_candidate_work =
        uint64_t(candidates.count) * job.m * job.n * job.k;
    result.metrics.winner_count = host_queue.winner_count;
    result.winner_overflow = host_queue.overflow_flag != 0;
    result.metrics.overflow_count = host_queue.overflow_flag ? 1 : 0;
    result.metrics.tail_d8 = host_queue.tail_d8;
    result.metrics.tail_d12 = host_queue.tail_d12;
    result.metrics.tail_d16 = host_queue.tail_d16;
    result.metrics.tail_d20 = host_queue.tail_d20;
    result.metrics.tail_d24 = host_queue.tail_d24;

    const uint32_t copied =
        host_queue.winner_count < kWinnerQueueCapacity ? host_queue.winner_count
                                                       : kWinnerQueueCapacity;
    for (uint32_t i = 0; i < copied; ++i) result.winners[i] = host_queue.winners[i];
  }

  if (!reuse_alloc) scratch.release();
  return result;
}

}  // namespace

bool BCacheKey::operator==(const BCacheKey& other) const {
  return generation == other.generation &&
      std::memcmp(job_key.bytes, other.job_key.bytes, 32) == 0 &&
      std::memcmp(hash_b.bytes, other.hash_b.bytes, 32) == 0 &&
      std::memcmp(bound_b.little_endian, other.bound_b.little_endian, 32) == 0 &&
      n == other.n && k == other.k && rank == other.rank &&
      certificate_version == other.certificate_version &&
      std::memcmp(b_noise_seed.bytes, other.b_noise_seed.bytes, 32) == 0 &&
      b_identity == other.b_identity;
}

DeviceScratch::~DeviceScratch() { release(); }

bool DeviceScratch::reserve(uint32_t m, uint32_t n, uint32_t k) {
  if (winner_queue != nullptr) {
    max_m = max_m > m ? max_m : m;
    max_n = max_n > n ? max_n : n;
    max_k = max_k > k ? max_k : k;
    return true;
  }
  if (cudaMalloc(&winner_queue, sizeof(DeviceWinnerQueue)) != cudaSuccess) {
    winner_queue = nullptr;
    return false;
  }
  max_m = m;
  max_n = n;
  max_k = k;
  if (flag_enabled("PEARL_SM120_DOUBLE_BUFFER")) {
    if (cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&stream_prepare, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&buffer_ready[0], cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&buffer_ready[1], cudaEventDisableTiming) != cudaSuccess) {
      release();
      return false;
    }
    streams_enabled = true;
  }
  return true;
}

bool DeviceScratch::observe_b(const JobContext& job, const CandidateBatch& candidates) {
  if (!flag_enabled("PEARL_SM120_CACHE_B")) return false;
  BCacheKey key{};
  key.generation = job.generation;
  key.job_key = job.job_key;
  key.hash_b = job.hash_b;
  key.bound_b = job.bound_b;
  key.n = job.n;
  key.k = job.k;
  key.rank = job.rank;
  key.certificate_version = job.certificate_version;
  key.b_noise_seed = job.b_noise_seed;
  key.b_identity = candidates.b_identity;
  const uint64_t bytes = uint64_t(job.n) * job.k * sizeof(int8_t);

  if (candidates.b_noised_t == nullptr) {
    const bool hit = b_cache_valid && b_cache_key == key && b_cache_bytes == bytes;
    b_cache_key = key;
    b_cache_bytes = bytes;
    b_cache_valid = true;
    return hit;
  }

  const bool hit = b_cache_valid && b_cache_device != nullptr &&
                   b_cache_bytes == bytes && b_cache_key == key;
  if (hit) return true;
  if (b_cache_device != nullptr && b_cache_bytes != bytes) {
    cudaFree(b_cache_device);
    b_cache_device = nullptr;
    b_cache_bytes = 0;
  }
  if (b_cache_device == nullptr &&
      cudaMalloc(&b_cache_device, bytes) != cudaSuccess) {
    b_cache_valid = false;
    return false;
  }
  if (cudaMemcpy(b_cache_device, candidates.b_noised_t, bytes,
                 cudaMemcpyDeviceToDevice) != cudaSuccess) {
    cudaFree(b_cache_device);
    b_cache_device = nullptr;
    b_cache_bytes = 0;
    b_cache_valid = false;
    return false;
  }
  b_cache_bytes = bytes;
  b_cache_key = key;
  b_cache_valid = true;
  return false;
}

void DeviceScratch::release() {
  if (buffer_ready[0]) cudaEventDestroy(buffer_ready[0]);
  if (buffer_ready[1]) cudaEventDestroy(buffer_ready[1]);
  if (stream_compute) cudaStreamDestroy(stream_compute);
  if (stream_prepare) cudaStreamDestroy(stream_prepare);
  if (b_cache_device) cudaFree(b_cache_device);
  if (winner_queue) cudaFree(winner_queue);
  winner_queue = nullptr;
  stream_compute = nullptr;
  stream_prepare = nullptr;
  buffer_ready[0] = nullptr;
  buffer_ready[1] = nullptr;
  max_m = max_n = max_k = 0;
  streams_enabled = false;
  b_cache_valid = false;
  b_cache_key = {};
  b_cache_device = nullptr;
  b_cache_bytes = 0;
}

SearchResult search_job(const JobContext& job, const CandidateBatch& candidates,
                        DeviceScratch& scratch) {
  return search_job_impl(job, candidates, scratch, nullptr, nullptr, nullptr);
}

SearchResult search_job_device_params(const JobContext& job,
                                      const CandidateBatch& candidates,
                                      DeviceScratch& scratch,
                                      const uint8_t* jackpot_key_device,
                                      const uint8_t* share_target_device,
                                      cudaStream_t stream) {
  if (jackpot_key_device == nullptr || share_target_device == nullptr) {
    SearchResult result{};
    result.metrics.cuda_errors = 1;
    return result;
  }
  return search_job_impl(job, candidates, scratch, jackpot_key_device,
                         share_target_device, stream);
}

SearchResult search_job(const JobContext& job, const CandidateBatch& candidates) {
  DeviceScratch scratch;
  return search_job(job, candidates, scratch);
}

SearchResult noise_and_search_job(const JobContext& job, const int8_t* a_base,
                                  const int8_t* b_base_t, const uint32_t* a_rows,
                                  const uint32_t* b_cols, uint32_t candidate_count,
                                  uint64_t b_identity, DeviceScratch& scratch,
                                  cudaStream_t stream) {
  SearchResult result{};
  if (!runtime_enabled() || !flag_enabled("PEARL_SM120_FUSED") ||
      !flag_enabled("PEARL_SM120_SEARCH_ONLY") || a_base == nullptr ||
      b_base_t == nullptr || a_rows == nullptr || b_cols == nullptr ||
      candidate_count == 0 || job.rank != 128 || job.m == 0 || job.n == 0 ||
      job.m % 16 || job.n % 16 || job.k < 16 * job.rank || job.k > 65536 ||
      job.k % job.rank != 0) {
    return result;
  }

  const cudaStream_t prep = stream == nullptr ? 0 : stream;
  const uint64_t a_elements = uint64_t(job.m) * job.k;
  const uint64_t b_elements = uint64_t(job.n) * job.k;
  int8_t *a_noise = nullptr, *b_noise = nullptr, *a_noised = nullptr,
         *b_noised = nullptr;
  const auto cleanup = [&]() {
    if (a_noise) cudaFree(a_noise);
    if (b_noise) cudaFree(b_noise);
    if (a_noised) cudaFree(a_noised);
    if (b_noised) cudaFree(b_noised);
  };

  if (cudaMalloc(&a_noise, a_elements) != cudaSuccess ||
      cudaMalloc(&b_noise, b_elements) != cudaSuccess ||
      cudaMalloc(&a_noised, a_elements * candidate_count) != cudaSuccess ||
      cudaMalloc(&b_noised, b_elements) != cudaSuccess) {
    cleanup();
    result.metrics.cuda_errors = 1;
    return result;
  }

  const bool implicit_zero_base = flag_enabled("PEARL_SM120_IMPLICIT_ZERO_BASE");
  if (generate_noise_launch(job.jackpot_key.bytes, job.b_noise_seed.bytes,
                            a_rows, job.m, b_cols, job.n, job.k, job.rank,
                            a_noise, b_noise, prep) != 0 ||
      (implicit_zero_base
           ? repeat_candidates_launch(a_noise, a_noised, a_elements,
                                      candidate_count, prep)
           : add_noise_candidates_launch(a_base, a_noise, a_noised, a_elements,
                                         candidate_count, prep)) != 0 ||
      (implicit_zero_base
           ? repeat_candidates_launch(b_noise, b_noised, b_elements, 1, prep)
           : add_noise_candidates_launch(b_base_t, b_noise, b_noised, b_elements,
                                         1, prep)) != 0 ||
      cudaStreamSynchronize(prep) != cudaSuccess) {
    cleanup();
    result.metrics.cuda_errors = 1;
    return result;
  }

  CandidateBatch candidates{a_noised, b_noised, candidate_count, b_identity};
  result = search_job(job, candidates, scratch);
  cleanup();
  return result;
}

}  // namespace pearl::sm120

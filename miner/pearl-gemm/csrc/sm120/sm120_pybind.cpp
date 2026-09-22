#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <memory>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <unordered_map>

#include "sm120_backend.h"
#include "host_signal_header.hpp"

namespace {

namespace py = pybind11;

void check_cuda_i8(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA-resident");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == at::kChar, name, " must be signed int8");
}

void check_cuda_u8_32(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                  tensor.scalar_type() == at::kByte && tensor.numel() == 32,
              name, " must be a contiguous CUDA uint8 tensor containing 32 bytes");
}

void check_cpu_u8_seed(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu() && tensor.is_contiguous() &&
                  tensor.scalar_type() == at::kByte && tensor.numel() == 32,
              name, " must be a contiguous CPU uint8 tensor containing 32 bytes");
}

void check_cuda_i32_indices(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                  tensor.scalar_type() == at::kInt,
              name, " must be a contiguous CUDA int32 tensor");
}

void check_same_cuda_device(const torch::Tensor& reference,
                            const torch::Tensor& other,
                            const char* name) {
  TORCH_CHECK(other.is_cuda() && other.get_device() == reference.get_device(),
              name, " must be on the same CUDA device as candidate_a");
}

pearl::sm120::DeviceScratch& persistent_scratch(const torch::Tensor& tensor) {
  const int device = tensor.get_device();
  TORCH_CHECK(cudaSetDevice(device) == cudaSuccess,
              "failed to select CUDA device for SM120 search");
  static thread_local std::unordered_map<
      int, std::unique_ptr<pearl::sm120::DeviceScratch>> scratches;
  auto& scratch = scratches[device];
  if (!scratch) scratch = std::make_unique<pearl::sm120::DeviceScratch>();
  return *scratch;
}

void sm120_signal_header(torch::Tensor header, int64_t m, int64_t n, int64_t k,
                         int64_t tile_row, int64_t tile_col) {
  TORCH_CHECK(header.device().is_cpu() && header.is_contiguous() &&
                  header.scalar_type() == at::kChar &&
                  header.numel() >= static_cast<int64_t>(sizeof(HostSignalHeader)),
              "host signal header must be a contiguous CPU int8 buffer");
  header.zero_();
  auto* value = reinterpret_cast<HostSignalHeader*>(header.data_ptr<int8_t>());
  value->status = kSignalTriggered;
  value->gridDim[0] = static_cast<uint32_t>((m + 15) / 16);
  value->gridDim[1] = static_cast<uint32_t>((n + 15) / 16);
  value->gridDim[2] = 1;
  value->blockDim[0] = 256;
  value->blockDim[1] = 1;
  value->blockDim[2] = 1;
  value->blockIdx[0] = static_cast<uint32_t>(tile_row);
  value->blockIdx[1] = static_cast<uint32_t>(tile_col);
  value->blockIdx[2] = 0;
  value->tileCoord[0] = static_cast<uint32_t>(tile_row);
  value->tileCoord[1] = static_cast<uint32_t>(tile_col);
  value->tileCoord[2] = 0;
  value->threadIdx[0] = value->threadIdx[1] = value->threadIdx[2] = 0;
  value->num_registers_per_thread = 16;
  for (uint16_t i = 0; i < value->num_registers_per_thread; ++i) {
    value->thread_rows[i] = static_cast<uint8_t>(i);
    value->thread_cols[i] = static_cast<uint8_t>(i);
  }
  value->mma_size = {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)};
  value->mma_tile_size = {16, 16, static_cast<int>(k)};
}

py::dict result_dict(const pearl::sm120::SearchResult& result) {
  py::dict response;
  response["candidates"] = result.metrics.candidates;
  response["target_tests"] = result.metrics.target_tests;
  response["valid_candidate_work"] = result.metrics.valid_candidate_work;
  response["winners"] = result.metrics.winner_count;
  response["winner_overflow"] = result.winner_overflow;
  response["cuda_errors"] = result.metrics.cuda_errors;
  response["b_cache_hits"] = result.metrics.b_cache_hits;
  response["b_cache_misses"] = result.metrics.b_cache_misses;
  response["tail_d8"] = result.metrics.tail_d8;
  response["tail_d12"] = result.metrics.tail_d12;
  response["tail_d16"] = result.metrics.tail_d16;
  response["tail_d20"] = result.metrics.tail_d20;
  response["tail_d24"] = result.metrics.tail_d24;

  py::list winners;
  const uint64_t copied =
      result.metrics.winner_count < pearl::sm120::kWinnerQueueCapacity
          ? result.metrics.winner_count
          : pearl::sm120::kWinnerQueueCapacity;
  for (uint64_t i = 0; i < copied; ++i) {
    const auto& winner = result.winners[i];
    py::dict item;
    item["generation"] = winner.id.generation;
    item["candidate_index"] = winner.id.candidate_index;
    item["tile_row"] = winner.tile_row;
    item["tile_col"] = winner.tile_col;
    item["reconstruction_slot"] = winner.reconstruction_slot;
    item["jackpot_hash"] = py::bytes(
        reinterpret_cast<const char*>(winner.jackpot_hash.bytes), 32);
    winners.append(item);
  }
  response["winner_descriptors"] = winners;
  return response;
}

pearl::sm120::CandidateBatch candidate_batch(torch::Tensor candidate_a,
                                             torch::Tensor b_transposed,
                                             int64_t m, int64_t n, int64_t k) {
  TORCH_CHECK(m > 0 && n > 0 && k >= 16 * 128 && m % 16 == 0 &&
                  n % 16 == 0 && k <= 65536 && k % 128 == 0,
              "SM120 dense backend requires canonical K >= 16*rank and legal 16x16 tiles");
  const auto batch = candidate_a.numel() / (m * k);
  TORCH_CHECK(batch > 0 && candidate_a.numel() == batch * m * k &&
                  b_transposed.numel() == n * k,
              "candidate-major A and transposed B sizes do not match m,n,k");
  return {candidate_a.data_ptr<int8_t>(), b_transposed.data_ptr<int8_t>(),
          uint32_t(batch),
          reinterpret_cast<uint64_t>(b_transposed.data_ptr<int8_t>())};
}

py::dict sm120_search(torch::Tensor candidate_a, torch::Tensor b_transposed,
                      torch::Tensor jackpot_key_cpu,
                      torch::Tensor share_target_cpu, int64_t m, int64_t n,
                      int64_t k, int64_t generation) {
  check_cuda_i8(candidate_a, "candidate_a");
  check_cuda_i8(b_transposed, "b_transposed");
  check_same_cuda_device(candidate_a, b_transposed, "b_transposed");
  check_cpu_u8_seed(jackpot_key_cpu, "jackpot_key");
  check_cpu_u8_seed(share_target_cpu, "share_target");

  pearl::sm120::JobContext job{};
  job.generation = generation;
  job.m = m;
  job.n = n;
  job.k = k;
  job.rank = 128;
  const auto* key = jackpot_key_cpu.data_ptr<uint8_t>();
  const auto* target = share_target_cpu.data_ptr<uint8_t>();
  for (uint32_t i = 0; i < 32; ++i) {
    job.jackpot_key.bytes[i] = key[i];
    job.share_target.little_endian[i] = target[i];
  }

  auto candidates = candidate_batch(candidate_a, b_transposed, m, n, k);
  auto& scratch = persistent_scratch(candidate_a);
  return result_dict(pearl::sm120::search_job(job, candidates, scratch));
}

py::dict sm120_search_device(
    torch::Tensor candidate_a, torch::Tensor b_transposed,
    torch::Tensor jackpot_key_cuda, torch::Tensor share_target_cuda,
    int64_t m, int64_t n, int64_t k, int64_t generation) {
  check_cuda_i8(candidate_a, "candidate_a");
  check_cuda_i8(b_transposed, "b_transposed");
  check_cuda_u8_32(jackpot_key_cuda, "jackpot_key");
  check_cuda_u8_32(share_target_cuda, "share_target");
  check_same_cuda_device(candidate_a, b_transposed, "b_transposed");
  check_same_cuda_device(candidate_a, jackpot_key_cuda, "jackpot_key");
  check_same_cuda_device(candidate_a, share_target_cuda, "share_target");

  pearl::sm120::JobContext job{};
  job.generation = generation;
  job.m = m;
  job.n = n;
  job.k = k;
  job.rank = 128;

  auto candidates = candidate_batch(candidate_a, b_transposed, m, n, k);
  auto& scratch = persistent_scratch(candidate_a);
  return result_dict(pearl::sm120::search_job_device_params(
      job, candidates, scratch, jackpot_key_cuda.data_ptr<uint8_t>(),
      share_target_cuda.data_ptr<uint8_t>()));
}

py::dict sm120_noise_search(torch::Tensor candidate_a_base,
                            torch::Tensor b_base_t,
                            torch::Tensor a_noise_seed_cpu,
                            torch::Tensor b_noise_seed_cpu,
                            torch::Tensor a_rows_cuda,
                            torch::Tensor b_cols_cuda,
                            torch::Tensor share_target_cpu, int64_t m, int64_t n,
                            int64_t k, int64_t generation) {
  check_cuda_i8(candidate_a_base, "candidate_a_base");
  check_cuda_i8(b_base_t, "b_base_t");
  check_cpu_u8_seed(a_noise_seed_cpu, "a_noise_seed");
  check_cpu_u8_seed(b_noise_seed_cpu, "b_noise_seed");
  check_cpu_u8_seed(share_target_cpu, "share_target");
  check_cuda_i32_indices(a_rows_cuda, "a_rows");
  check_cuda_i32_indices(b_cols_cuda, "b_cols");
  check_same_cuda_device(candidate_a_base, b_base_t, "b_base_t");
  check_same_cuda_device(candidate_a_base, a_rows_cuda, "a_rows");
  check_same_cuda_device(candidate_a_base, b_cols_cuda, "b_cols");

  TORCH_CHECK(m > 0 && n > 0 && k >= 16 * 128 && m % 16 == 0 &&
                  n % 16 == 0 && k <= 65536 && k % 128 == 0,
              "SM120 noising search requires canonical rank-128 geometry");
  const auto batch = candidate_a_base.numel() / (m * k);
  TORCH_CHECK(batch > 0 && candidate_a_base.numel() == batch * m * k &&
                  b_base_t.numel() == n * k &&
                  a_rows_cuda.numel() == m && b_cols_cuda.numel() == n,
              "base operand and index sizes do not match m,n,k");

  pearl::sm120::JobContext job{};
  job.generation = generation;
  job.m = m;
  job.n = n;
  job.k = k;
  job.rank = 128;
  const auto* a_seed = a_noise_seed_cpu.data_ptr<uint8_t>();
  const auto* b_seed = b_noise_seed_cpu.data_ptr<uint8_t>();
  const auto* target = share_target_cpu.data_ptr<uint8_t>();
  for (uint32_t i = 0; i < 32; ++i) {
    job.jackpot_key.bytes[i] = a_seed[i];
    job.b_noise_seed.bytes[i] = b_seed[i];
    job.share_target.little_endian[i] = target[i];
  }

  auto& scratch = persistent_scratch(candidate_a_base);
  const auto result = pearl::sm120::noise_and_search_job(
      job, candidate_a_base.data_ptr<int8_t>(), b_base_t.data_ptr<int8_t>(),
      reinterpret_cast<const uint32_t*>(a_rows_cuda.data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(b_cols_cuda.data_ptr<int32_t>()),
      uint32_t(batch), reinterpret_cast<uint64_t>(b_base_t.data_ptr<int8_t>()),
      scratch);
  return result_dict(result);
}

py::tuple sm120_noise(torch::Tensor a_noise_seed_cpu,
                      torch::Tensor b_noise_seed_cpu,
                      torch::Tensor a_rows_cuda, torch::Tensor b_cols_cuda,
                      int64_t k, int64_t rank) {
  check_cpu_u8_seed(a_noise_seed_cpu, "a_noise_seed");
  check_cpu_u8_seed(b_noise_seed_cpu, "b_noise_seed");
  check_cuda_i32_indices(a_rows_cuda, "a_rows");
  check_cuda_i32_indices(b_cols_cuda, "b_cols");
  TORCH_CHECK(k >= 16 * rank && k <= 65536 && rank == 128 && k % rank == 0,
              "canonical SM120 noise requires rank 128 and K >= 16*rank");
  const auto options =
      torch::TensorOptions().device(a_rows_cuda.device()).dtype(torch::kInt8);
  auto noise_a = torch::empty({a_rows_cuda.numel(), k}, options);
  auto noise_b_t = torch::empty({b_cols_cuda.numel(), k}, options);
  auto a_seed_cuda =
      a_noise_seed_cpu.to(a_rows_cuda.device(), torch::kUInt8, false, true);
  auto b_seed_cuda =
      b_noise_seed_cpu.to(b_cols_cuda.device(), torch::kUInt8, false, true);
  TORCH_CHECK(
      pearl::sm120::generate_noise_launch(
          a_seed_cuda.data_ptr<uint8_t>(), b_seed_cuda.data_ptr<uint8_t>(),
          reinterpret_cast<const uint32_t*>(a_rows_cuda.data_ptr<int32_t>()),
          a_rows_cuda.numel(),
          reinterpret_cast<const uint32_t*>(b_cols_cuda.data_ptr<int32_t>()),
          b_cols_cuda.numel(), k, rank, noise_a.data_ptr<int8_t>(),
          noise_b_t.data_ptr<int8_t>(), nullptr) == 0,
      "SM120 canonical noise launch failed");
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess,
              "SM120 canonical noise execution failed");
  return py::make_tuple(noise_a, noise_b_t);
}

torch::Tensor sm120_transcript(torch::Tensor a_noised,
                               torch::Tensor b_noised_t, int64_t k,
                               int64_t rank) {
  check_cuda_i8(a_noised, "a_noised");
  check_cuda_i8(b_noised_t, "b_noised_t");
  TORCH_CHECK(a_noised.numel() == 16 * k &&
                  b_noised_t.numel() == 16 * k,
              "transcript probe requires exactly one 16x16 tile");
  TORCH_CHECK(k >= 16 * rank && k <= 65536 && rank == 128 && k % rank == 0,
              "canonical transcript requires rank 128 and K >= 16*rank");
  const auto options =
      torch::TensorOptions().device(a_noised.device()).dtype(torch::kInt);
  auto transcript = torch::empty({16}, options);
  TORCH_CHECK(pearl::sm120::dense_reference_tile_launch(
                  a_noised.data_ptr<int8_t>(),
                  b_noised_t.data_ptr<int8_t>(), k, rank,
                  reinterpret_cast<uint32_t*>(
                      transcript.data_ptr<int32_t>())) == 0,
              "SM120 transcript launch failed");
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess,
              "SM120 transcript execution failed");
  return transcript;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("search", &sm120_search, "Exact SM120 search-only backend");
  m.def("search_device", &sm120_search_device,
        "Exact SM120 search using device-resident key/target");
  m.def("noise_search", &sm120_noise_search,
        "Canonical GPU noising followed by exact SM120 search");
  m.def("signal_header", &sm120_signal_header,
        "Write a validated SM120 winner into Pearl's host signal format");
  m.def("noise", &sm120_noise,
        "Canonical selected-row/column rank-128 noise generator");
  m.def("transcript", &sm120_transcript,
        "Exact rank-boundary transcript for one 16x16 tile");
}

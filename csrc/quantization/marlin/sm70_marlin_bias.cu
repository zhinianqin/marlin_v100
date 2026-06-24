#include "quantization/marlin/sm70_marlin_bias.cuh"

#include <c10/cuda/CUDAException.h>

namespace marlin::sm70 {

namespace {

__global__ void sm70_marlin_dense_bias_init_kernel(
    cutlass::half_t* __restrict__ c,
    cutlass::half_t const* __restrict__ b_bias, int64_t m, int64_t n) {
  int64_t const linear =
      int64_t(blockIdx.x) * blockDim.x + int64_t(threadIdx.x);
  int64_t const total = m * n;
  if (linear >= total) {
    return;
  }

  int const logical_col = static_cast<int>(linear % n);
  c[linear] = b_bias[sm70_marlin_bias_storage_index(logical_col)];
}

__global__ void sm70_marlin_moe_bias_init_kernel(
    cutlass::half_t* __restrict__ c,
    cutlass::half_t const* __restrict__ b_bias,
    float const* __restrict__ global_scale,
    int32_t const* __restrict__ sorted_token_ids,
    int32_t const* __restrict__ expert_ids,
    int32_t const* __restrict__ num_tokens_past_padded,
    float const* __restrict__ topk_weights, int sorted_token_count, int n,
    int moe_block_size, int expanded_token_count, bool mul_topk_weights) {
  int64_t const linear =
      int64_t(blockIdx.x) * blockDim.x + int64_t(threadIdx.x);
  int64_t const total = int64_t(sorted_token_count) * n;
  if (linear >= total) {
    return;
  }

  int const route_row = static_cast<int>(linear / n);
  if (route_row >= num_tokens_past_padded[0]) {
    return;
  }

  int const sorted_id = sorted_token_ids[route_row];
  if (sorted_id < 0 || sorted_id >= expanded_token_count) {
    return;
  }

  int const expert = expert_ids[route_row / moe_block_size];
  if (expert < 0) {
    return;
  }

  int const logical_col = static_cast<int>(linear % n);
  float value = __half2float(reinterpret_cast<half const*>(b_bias)[
      int64_t(expert) * n + sm70_marlin_bias_storage_index(logical_col)]);
  if (mul_topk_weights && global_scale != nullptr) {
    value *= global_scale[expert];
  }
  if (mul_topk_weights) {
    value *= topk_weights[sorted_id];
  }
  c[int64_t(sorted_id) * n + logical_col] = cutlass::half_t(value);
}

}  // namespace

void launch_sm70_marlin_dense_bias_init(
    cutlass::half_t* c, cutlass::half_t const* b_bias, int64_t m, int64_t n,
    cudaStream_t stream) {
  int64_t const total = m * n;
  if (total == 0) {
    return;
  }
  constexpr int kThreads = 256;
  int const blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  sm70_marlin_dense_bias_init_kernel<<<blocks, kThreads, 0, stream>>>(
      c, b_bias, m, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_sm70_marlin_moe_bias_init(
    cutlass::half_t* c, cutlass::half_t const* b_bias, float const* global_scale,
    int32_t const* sorted_token_ids, int32_t const* expert_ids,
    int32_t const* num_tokens_past_padded, float const* topk_weights,
    int sorted_token_count, int n, int moe_block_size,
    int expanded_token_count, bool mul_topk_weights, cudaStream_t stream) {
  int64_t const total = int64_t(sorted_token_count) * n;
  if (total == 0) {
    return;
  }
  constexpr int kThreads = 256;
  int const blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  sm70_marlin_moe_bias_init_kernel<<<blocks, kThreads, 0, stream>>>(
      c, b_bias, global_scale, sorted_token_ids, expert_ids,
      num_tokens_past_padded, topk_weights, sorted_token_count, n,
      moe_block_size, expanded_token_count, mul_topk_weights);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace marlin::sm70

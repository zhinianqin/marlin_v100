#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

namespace marlin::sm70 {

template <typename T>
CUTLASS_HOST_DEVICE T sm70_marlin_bias_arg(T value) {
  return value;
}

CUTLASS_HOST_DEVICE
int sm70_marlin_bias_storage_index(int logical_col) {
  constexpr int kInverseScalePermSingle[32] = {
      0,  1,  8,  9,  16, 17, 24, 25, 2,  3,  10,
      11, 18, 19, 26, 27, 4,  5,  12, 13, 20, 21,
      28, 29, 6,  7,  14, 15, 22, 23, 30, 31};
  return (logical_col & ~31) + kInverseScalePermSingle[logical_col & 31];
}

void launch_sm70_marlin_dense_bias_init(
    cutlass::half_t* c, cutlass::half_t const* b_bias, int64_t m, int64_t n,
    cudaStream_t stream);

void launch_sm70_marlin_moe_bias_init(
    cutlass::half_t* c, cutlass::half_t const* b_bias, float const* global_scale,
    int32_t const* sorted_token_ids, int32_t const* expert_ids,
    int32_t const* num_tokens_past_padded, float const* topk_weights,
    int sorted_token_count, int n, int moe_block_size,
    int expanded_token_count, bool mul_topk_weights, cudaStream_t stream);

}  // namespace marlin::sm70

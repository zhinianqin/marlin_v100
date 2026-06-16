/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

#include "marlin.cuh"
#include "core/registration.h"
#include "core/scalar_type.hpp"

torch::Tensor sm70_marlin_u4b8_gemm(torch::Tensor& a, torch::Tensor& c,
                                    torch::Tensor& b_q_weight,
                                    torch::Tensor& b_scales, int64_t size_m,
                                    int64_t size_n, int64_t size_k,
                                    int64_t group_size);
torch::Tensor sm70_marlin_u4_gemm(torch::Tensor& a, torch::Tensor& c,
                                  torch::Tensor& b_q_weight,
                                  torch::Tensor& b_scales,
                                  torch::Tensor& b_zeros, int64_t size_m,
                                  int64_t size_n, int64_t size_k,
                                  int64_t group_size);
torch::Tensor sm70_marlin_u8_gemm(torch::Tensor& a, torch::Tensor& c,
                                  torch::Tensor& b_q_weight,
                                  torch::Tensor& b_scales,
                                  torch::Tensor& b_zeros, int64_t size_m,
                                  int64_t size_n, int64_t size_k,
                                  int64_t group_size);
torch::Tensor sm70_marlin_u8b128_gemm(torch::Tensor& a, torch::Tensor& c,
                                      torch::Tensor& b_q_weight,
                                      torch::Tensor& b_scales, int64_t size_m,
                                      int64_t size_n, int64_t size_k,
                                      int64_t group_size);
torch::Tensor sm70_marlin_fp8_gemm(torch::Tensor& a, torch::Tensor& c,
                                   torch::Tensor& b_q_weight,
                                   torch::Tensor& b_scales, int64_t size_m,
                                   int64_t size_n, int64_t size_k,
                                   int64_t group_size);
torch::Tensor sm70_marlin_nvfp4_gemm(torch::Tensor& a, torch::Tensor& c,
                                     torch::Tensor& b_q_weight,
                                     torch::Tensor& b_scales,
                                     torch::Tensor& global_scale,
                                     int64_t size_m, int64_t size_n,
                                     int64_t size_k, int64_t group_size);
torch::Tensor sm70_marlin_mxfp4_gemm(torch::Tensor& a, torch::Tensor& c,
                                     torch::Tensor& b_q_weight,
                                     torch::Tensor& b_scales, int64_t size_m,
                                     int64_t size_n, int64_t size_k,
                                     int64_t group_size);

torch::Tensor marlin_gemm(
    torch::Tensor& a, std::optional<torch::Tensor> c_or_none,
    torch::Tensor& b_q_weight,
    std::optional<torch::Tensor> const& b_bias_or_none, torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& a_scales_or_none,
    std::optional<torch::Tensor> const& global_scale_or_none,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none, torch::Tensor& workspace,
    vllm::ScalarTypeId const& b_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float) {
  if (g_idx_or_none.has_value()) {
    TORCH_CHECK(g_idx_or_none.value().numel() == 0,
                "SM70 Marlin does not support act_order g_idx.");
  }
  if (perm_or_none.has_value()) {
    TORCH_CHECK(perm_or_none.value().numel() == 0,
                "SM70 Marlin does not support act_order perm.");
  }
  (void)workspace;
  (void)is_k_full;
  (void)use_atomic_add;
  (void)use_fp32_reduce;

  vllm::ScalarTypeId a_type_id, c_type_id, s_type_id;

  auto c_dtype = a.dtype();
  if (a.scalar_type() == at::ScalarType::Half) {
    a_type_id = vllm::kFloat16.id();
    c_type_id = vllm::kFloat16.id();
  } else if (a.scalar_type() == at::ScalarType::BFloat16) {
    a_type_id = vllm::kBFloat16.id();
    c_type_id = vllm::kBFloat16.id();
  } else {
    c_dtype = b_scales.dtype();
    if (b_scales.scalar_type() == at::ScalarType::Half) {
      c_type_id = vllm::kFloat16.id();
    } else if (b_scales.scalar_type() == at::ScalarType::BFloat16) {
      c_type_id = vllm::kBFloat16.id();
    } else {
      c_type_id = vllm::kBFloat16.id();

      TORCH_CHECK(c_or_none.has_value(), "c must be passed for W4A8-FP4");
      torch::Tensor c = c_or_none.value();
      c_dtype = c.dtype();

      if (c.scalar_type() == at::ScalarType::Half) {
        c_type_id = vllm::kFloat16.id();
      } else if (c.scalar_type() == at::ScalarType::BFloat16) {
        c_type_id = vllm::kBFloat16.id();
      } else {
        TORCH_CHECK(false, "unsupported c dtype");
      }
    }

    if (a.scalar_type() == at::ScalarType::Float8_e4m3fn) {
      a_type_id = vllm::kFE4M3fn.id();
    } else if (a.scalar_type() == at::ScalarType::Char) {
      a_type_id = vllm::kS8.id();
    } else {
      TORCH_CHECK(false, "unsupported `a` scalar_type");
    }
  }

  s_type_id = c_type_id;
  if (b_type_id == vllm::kFE2M1f.id()) {
    if (b_scales.scalar_type() == at::ScalarType::Float8_e4m3fn) {
      s_type_id = vllm::kFE4M3fn.id();
    } else if (b_scales.scalar_type() == at::ScalarType::Float8_e8m0fnu) {
      s_type_id = vllm::kFE8M0fnu.id();
    } else {
      TORCH_CHECK(false,
                  "When b_type = float4_e2m1f on the SM70 build, b_scales "
                  "must be float8_e4m3fn for NVFP4 or float8_e8m0fnu for "
                  "MXFP4 scales.");
    }
  }

  vllm::ScalarType a_type = vllm::ScalarType::from_id(a_type_id);
  vllm::ScalarType b_type = vllm::ScalarType::from_id(b_type_id);
  vllm::ScalarType c_type = vllm::ScalarType::from_id(c_type_id);
  vllm::ScalarType s_type = vllm::ScalarType::from_id(s_type_id);

  TORCH_CHECK(a_type == vllm::kFloat16,
              "SM70 build only supports float16 activations.");
  TORCH_CHECK(c_type == vllm::kFloat16,
              "SM70 build only supports float16 outputs.");
  TORCH_CHECK(s_type == vllm::kFloat16 ||
                  (b_type == vllm::kFE2M1f &&
                   (s_type == vllm::kFE4M3fn ||
                    s_type == vllm::kFE8M0fnu)),
              "SM70 build only supports float16 scales, except FP4 uses "
              "float8_e4m3fn scales for NVFP4 or float8_e8m0fnu scales for "
              "MXFP4.");
  TORCH_CHECK(b_type == vllm::kU4 || b_type == vllm::kU4B8 ||
                  b_type == vllm::kU8 || b_type == vllm::kU8B128 ||
                  b_type == vllm::kFE4M3fn || b_type == vllm::kFE2M1f,
              "SM70 Marlin currently implements only uint4, "
              "uint4b8, uint8, uint8b128, fp8_e4m3fn, nvfp4, and "
              "mxfp4 dense weights.");

  int pack_factor = 32 / b_type.size_bits();

  // Verify A
  TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
              ", size_m = ", size_m);
  TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
              ", size_k = ", size_k);

  // Verify B
  TORCH_CHECK(
      size_k % MARLIN_NAMESPACE_NAME::tile_size == 0, "size_k = ", size_k,
      " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  TORCH_CHECK((size_k / MARLIN_NAMESPACE_NAME::tile_size) == b_q_weight.size(0),
              "Shape mismatch: b_q_weight.size(0) = ", b_q_weight.size(0),
              ", size_k = ", size_k,
              ", tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  TORCH_CHECK(
      b_q_weight.size(1) % MARLIN_NAMESPACE_NAME::tile_size == 0,
      "b_q_weight.size(1) = ", b_q_weight.size(1),
      " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  int actual_size_n =
      (b_q_weight.size(1) / MARLIN_NAMESPACE_NAME::tile_size) * pack_factor;
  TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
              ", actual_size_n = ", actual_size_n);

  // Verify device and strides
  TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.stride(1) == 1, "A.stride(1) is not 1");
  // We use int4 (16 bytes) to load A, so A must aligned to 16 bytes
  TORCH_CHECK(a.stride(0) % 8 == 0, "A.stride(0) must divisible by 8");
  TORCH_CHECK(((uint64_t)a.data_ptr()) % 16 == 0, "A must aligned to 16 bytes");

  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");
  TORCH_CHECK(b_scales.scalar_type() == at::ScalarType::Half ||
                  (b_type == vllm::kFE2M1f &&
                   (b_scales.scalar_type() ==
                        at::ScalarType::Float8_e4m3fn ||
                    b_scales.scalar_type() ==
                        at::ScalarType::Float8_e8m0fnu)),
              "SM70 build only supports float16 scales, except FP4 uses "
              "float8_e4m3fn scales for NVFP4 or float8_e8m0fnu scales for "
              "MXFP4.");

  torch::Tensor a_scales;
  auto options = torch::TensorOptions().dtype(c_dtype).device(a.device());
  auto options_fp32 =
      torch::TensorOptions().dtype(at::kFloat).device(a.device());

  if (a_scales_or_none.has_value()) {
    a_scales = a_scales_or_none.value();
    TORCH_CHECK(a_type.size_bits() == 8,
                "a_scales can only be used for 8bit activation.");
  } else {
    a_scales = torch::empty({0}, options_fp32);
    TORCH_CHECK(a_type.size_bits() != 8,
                "the a_scales parameter must be passed for 8bit activation.");
  }

  // Alloc buffers
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  torch::Tensor c;
  if (c_or_none.has_value()) {
    c = c_or_none.value();
    TORCH_CHECK(c.device().is_cuda(), "c is not on GPU");
    TORCH_CHECK(c.is_contiguous(), "c is not contiguous");
    TORCH_CHECK(c.scalar_type() == at::ScalarType::Half,
                "SM70 build only supports float16 outputs.");
    TORCH_CHECK(c.size(0) == size_m, "Shape mismatch: c.size(0) = ", c.size(0),
                ", size_m = ", size_m);
    TORCH_CHECK(c.size(1) == size_n, "Shape mismatch: c.size(1) = ", c.size(1),
                ", size_n = ", size_n);
  } else {
    c = torch::empty({size_m, size_n}, options);
  }
  if (size_m == 0) return c;

  // Detect groupsize and act_order
  int num_groups = -1;
  int group_size = -1;

  int rank = b_scales.sizes().size();
  TORCH_CHECK(rank == 2, "b_scales rank = ", rank, " is not 2");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales dim 1 = ", b_scales.size(1),
              " is not size_n = ", size_n);
  num_groups = b_scales.size(0);

  if (num_groups > 1) {
    TORCH_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
                ", is not divisible by b_scales.size(0) = ",
                b_scales.size(0));
    group_size = size_k / num_groups;
  } else {
    group_size = -1;
  }

  torch::Tensor global_scale;
  if (global_scale_or_none.has_value()) {
    global_scale = global_scale_or_none.value();
    TORCH_CHECK(
        b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn,
        "SM70 Marlin supports global_scale only for "
        "nvfp4 format.");
    TORCH_CHECK(global_scale.device().is_cuda(), "global_scale is not on GPU");
    TORCH_CHECK(global_scale.is_contiguous(),
                "global_scale is not contiguous");
    TORCH_CHECK(global_scale.scalar_type() == at::ScalarType::Float,
                "SM70 Marlin nvfp4 expects fp32 global_scale.");
    TORCH_CHECK(global_scale.numel() == 1,
                "SM70 Marlin nvfp4 expects a single global_scale "
                "value.");
  } else {
    global_scale = torch::empty({0}, options_fp32);
    TORCH_CHECK(!(b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn),
                "the global_scale parameter must be passed for nvfp4 format.");
  }

  bool has_bias = b_bias_or_none.has_value();
  torch::Tensor b_bias;
  if (has_bias) {
    b_bias = b_bias_or_none.value();
    TORCH_CHECK(b_bias.device().is_cuda(), "b_bias is not on GPU");
    TORCH_CHECK(b_bias.is_contiguous(), "b_bias is not contiguous");
    TORCH_CHECK(b_bias.size(0) == size_n, "b_bias.size(0) != size_n");
    TORCH_CHECK(b_bias.stride(0) == 1, "b_bias.stride(0) != 1");
  } else {
    b_bias = torch::empty({0}, options);
  }

  torch::Tensor b_zeros;
  if (b_zeros_or_none.has_value()) {
    b_zeros = b_zeros_or_none.value();
    TORCH_CHECK(b_zeros.device().is_cuda(), "b_zeros is not on GPU");
    TORCH_CHECK(b_zeros.is_contiguous(),
                "b_zeros is not contiguous");
  } else {
    b_zeros = torch::empty({0}, options);
  }
  bool has_zp = b_zeros.size(-1) > 0;

  TORCH_CHECK(!has_zp || b_type == vllm::kU4 || b_type == vllm::kU8,
              "SM70 Marlin does not support zero-point "
              "metadata for this quant type.");

  // Verify fp16 zero-point metadata for dense uint4/uint8.
  if (has_zp) {
    TORCH_CHECK(is_zp_float,
                "SM70 Marlin received b_zeros but "
                "is_zp_float is false. Packed integer zero-points are not "
                "supported on the dense SM70 uint4/uint8 path.");
    int rank = b_zeros.sizes().size();
    TORCH_CHECK(rank == 2, "b_zeros rank = ", rank, " is not 2");
    TORCH_CHECK(b_zeros.scalar_type() == at::ScalarType::Half,
                "SM70 Marlin uint4/uint8 dense path expects fp16 "
                "zero points.");
    TORCH_CHECK(b_zeros.size(1) == size_n,
                "b_zeros dim 1 = ", b_zeros.size(1),
                " is not size_n = ", size_n);
    TORCH_CHECK(num_groups == b_zeros.size(0),
                "b_zeros dim 0 = ", b_zeros.size(0),
                " is not num_groups = ", num_groups);
  } else {
    TORCH_CHECK(!is_zp_float,
                "is_zp_float is true but b_zeros was not provided.");
  }

  TORCH_CHECK(size_n % MARLIN_NAMESPACE_NAME::min_thread_n == 0,
              "SM70 Marlin requires size_n % 64 == 0. "
              "size_n = ",
              size_n, ", min_thread_n = ", MARLIN_NAMESPACE_NAME::min_thread_n);

  TORCH_CHECK(a_scales.scalar_type() == at::ScalarType::Float,
              "scalar type of a_scales must be float");
  TORCH_CHECK(global_scale.scalar_type() == at::ScalarType::Float,
              "scalar type of global_scale must be float");
  if (a_type.size_bits() == 16) {
    TORCH_CHECK(
        a.scalar_type() == c.scalar_type(),
        "scalar type of a must be the same with c for 16 bit activation");
  }

  TORCH_CHECK(b_q_weight.scalar_type() == at::ScalarType::Int,
              "SM70 Marlin expects int32 packed weights.");
  TORCH_CHECK(!has_bias,
              "SM70 Marlin does not support bias. TODO: add epilogue bias fusion.");
  TORCH_CHECK((b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn) ||
                  global_scale.numel() == 0,
              "SM70 Marlin supports global_scale only for "
              "nvfp4 format.");
  TORCH_CHECK(size_k % 32 == 0,
              "SM70 Marlin requires size_k % 32 == 0.");
  TORCH_CHECK(size_n % 64 == 0,
              "SM70 Marlin requires size_n % 64 == 0.");
  TORCH_CHECK(group_size == -1 || group_size > 0,
              "SM70 Marlin received invalid group_size = ",
              group_size);

  if (b_type == vllm::kU4) {
    TORCH_CHECK(size_k % 32 == 0,
                "SM70 Marlin uint4 dense path requires size_k % 32 == 0.");
    TORCH_CHECK(
        has_zp && is_zp_float,
        "SM70 Marlin uint4 dense path requires fp16 zero points.");
    return sm70_marlin_u4_gemm(a, c, b_q_weight, b_scales, b_zeros, size_m,
                               size_n, size_k, group_size);
  }

  if (b_type == vllm::kU8) {
    TORCH_CHECK(size_k % 32 == 0,
                "SM70 Marlin uint8 dense path requires size_k % 32 == 0.");
    TORCH_CHECK(
        has_zp && is_zp_float,
        "SM70 Marlin uint8 dense path requires fp16 zero points.");
    return sm70_marlin_u8_gemm(a, c, b_q_weight, b_scales, b_zeros, size_m,
                               size_n, size_k, group_size);
  }

  if (b_type == vllm::kU8B128) {
    TORCH_CHECK(
        size_k % 32 == 0,
        "SM70 Marlin uint8b128 dense path requires size_k % 32 == 0.");
    TORCH_CHECK(!has_zp && !is_zp_float,
                "SM70 Marlin uint8b128 does not support "
                "zero-point metadata.");
    return sm70_marlin_u8b128_gemm(a, c, b_q_weight, b_scales, size_m, size_n,
                                   size_k, group_size);
  }

  if (b_type == vllm::kFE4M3fn) {
    TORCH_CHECK(
        size_k % 32 == 0,
        "SM70 Marlin fp8_e4m3 dense path requires size_k % 32 == 0.");
    TORCH_CHECK(group_size == -1 || group_size == 128,
                "SM70 Marlin fp8_e4m3 supports only group_size -1 or "
                "128. Got ",
                group_size);
    TORCH_CHECK(!has_zp && !is_zp_float,
                "SM70 Marlin fp8_e4m3 does not support zero-point "
                "metadata.");
    return sm70_marlin_fp8_gemm(a, c, b_q_weight, b_scales, size_m, size_n,
                                size_k, group_size);
  }

  if (b_type == vllm::kFE2M1f) {
    TORCH_CHECK(
        size_k % 32 == 0,
        "SM70 Marlin nvfp4/mxfp4 dense path requires size_k % 32 == 0.");
    TORCH_CHECK(!has_zp && !is_zp_float,
                "SM70 Marlin nvfp4/mxfp4 does not support zero-point "
                "metadata.");
    if (s_type == vllm::kFE4M3fn) {
      TORCH_CHECK(global_scale.numel() == 1,
                  "the global_scale parameter must be passed for nvfp4 format.");
      TORCH_CHECK(group_size == 16,
                  "SM70 Marlin nvfp4 supports only group_size 16. "
                  "Got ",
                  group_size);
      return sm70_marlin_nvfp4_gemm(a, c, b_q_weight, b_scales, global_scale,
                                    size_m, size_n, size_k, group_size);
    }

    TORCH_CHECK(s_type == vllm::kFE8M0fnu,
                "SM70 Marlin mxfp4 expects float8_e8m0fnu "
                "mxfp4 scales.");
    TORCH_CHECK(group_size == 32,
                "SM70 Marlin mxfp4 supports only group_size 32 "
                "when global_scale is not provided. Got ",
                group_size);
    return sm70_marlin_mxfp4_gemm(a, c, b_q_weight, b_scales, size_m, size_n,
                                  size_k, group_size);
  }

  TORCH_CHECK(!has_zp && !is_zp_float,
              "SM70 Marlin uint4b8 does not support zero-point "
              "metadata.");
  return sm70_marlin_u4b8_gemm(a, c, b_q_weight, b_scales, size_m, size_n,
                               size_k, group_size);
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("marlin_gemm", &marlin_gemm);
}

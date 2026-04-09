#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>

#include <cstdint>
#include <vector>

#include "core/scalar_type.hpp"

namespace marlin_sm70_cutlass {

inline std::vector<int64_t> make_weight_perm_u4() {
  std::vector<int64_t> perm_list;
  perm_list.reserve(1024);
  for (int i = 0; i < 32; ++i) {
    std::vector<int64_t> perm1;
    int col = i / 4;
    for (int block : {0, 1}) {
      for (int row : {2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4),
                      2 * (i % 4 + 4) + 1}) {
        perm1.push_back(16 * row + col + 8 * block);
      }
    }
    for (int j = 0; j < 4; ++j) {
      for (auto p : perm1) {
        perm_list.push_back(p + 256 * j);
      }
    }
  }

  std::vector<int64_t> interleaved;
  interleaved.reserve(perm_list.size());
  for (int chunk = 0; chunk < static_cast<int>(perm_list.size()); chunk += 8) {
    static constexpr int kInterleave[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    for (int idx : kInterleave) {
      interleaved.push_back(perm_list[chunk + idx]);
    }
  }
  return interleaved;
}

inline std::vector<int64_t> invert_perm(const std::vector<int64_t>& perm) {
  std::vector<int64_t> inv(perm.size());
  for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
    inv[perm[i]] = i;
  }
  return inv;
}

inline std::vector<int64_t> make_scale_perm(bool single_group) {
  std::vector<int64_t> perm;
  if (!single_group) {
    perm.reserve(64);
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        perm.push_back(i + 8 * j);
      }
    }
  } else {
    perm.reserve(32);
    for (int i = 0; i < 4; ++i) {
      for (int j : {0, 1, 8, 9, 16, 17, 24, 25}) {
        perm.push_back(2 * i + j);
      }
    }
  }
  return perm;
}

inline torch::Tensor dequantize_uint4b8_weight_cpu(torch::Tensor q_weight_cpu,
                                                   torch::Tensor scales_cpu,
                                                   int64_t size_k,
                                                   int64_t size_n,
                                                   int64_t num_groups) {
  TORCH_CHECK(q_weight_cpu.device().is_cpu(), "q_weight_cpu must be on CPU");
  TORCH_CHECK(scales_cpu.device().is_cpu(), "scales_cpu must be on CPU");
  TORCH_CHECK(q_weight_cpu.scalar_type() == at::ScalarType::Int,
              "q_weight_cpu must be int32");
  TORCH_CHECK(scales_cpu.scalar_type() == at::ScalarType::Half,
              "scales_cpu must be float16");

  int64_t group_size = num_groups > 1 ? (size_k / num_groups) : -1;
  bool single_group = group_size == -1 || group_size >= size_k;

  const auto perm = make_weight_perm_u4();
  const auto inv_perm = invert_perm(perm);
  const auto scale_perm = make_scale_perm(single_group);
  const auto inv_scale_perm = invert_perm(scale_perm);

  auto weight_cpu = torch::empty({size_k, size_n},
                                 torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

  auto q_contig = q_weight_cpu.contiguous();
  auto s_contig = scales_cpu.contiguous();

  const int32_t* q_ptr = q_contig.data_ptr<int32_t>();
  const at::Half* s_ptr = s_contig.data_ptr<at::Half>();
  at::Half* w_ptr = weight_cpu.data_ptr<at::Half>();

  std::vector<float> scales(size_n * num_groups, 0.0f);
  const int64_t scale_chunk = static_cast<int64_t>(scale_perm.size());
  for (int64_t base = 0; base < num_groups * size_n; base += scale_chunk) {
    for (int64_t i = 0; i < scale_chunk; ++i) {
      scales[base + i] = static_cast<float>(s_ptr[base + inv_scale_perm[i]]);
    }
  }

  const int64_t packed_rows = size_k / 16;
  const int64_t packed_cols = q_contig.size(1);
  const int64_t unpacked_cols = packed_cols * 8;
  std::vector<int32_t> unpacked_perm(unpacked_cols);
  std::vector<int32_t> unpacked(unpacked_cols);

  for (int64_t row = 0; row < packed_rows; ++row) {
    for (int64_t col = 0; col < packed_cols; ++col) {
      uint32_t packed = static_cast<uint32_t>(q_ptr[row * packed_cols + col]);
      for (int nibble = 0; nibble < 8; ++nibble) {
        unpacked_perm[col * 8 + nibble] = static_cast<int32_t>((packed >> (4 * nibble)) & 0xF);
      }
    }

    for (int64_t base = 0; base < unpacked_cols; base += static_cast<int64_t>(perm.size())) {
      for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
        unpacked[base + i] = unpacked_perm[base + inv_perm[i]];
      }
    }

    for (int64_t n_tile = 0; n_tile < size_n / 16; ++n_tile) {
      for (int64_t row_in_tile = 0; row_in_tile < 16; ++row_in_tile) {
        int64_t k_idx = row * 16 + row_in_tile;
        int64_t group_idx = group_size == -1 ? 0 : (k_idx / group_size);
        for (int64_t col_in_tile = 0; col_in_tile < 16; ++col_in_tile) {
          int64_t n_idx = n_tile * 16 + col_in_tile;
          int64_t flat_idx = n_tile * 256 + row_in_tile * 16 + col_in_tile;
          float scale = scales[group_idx * size_n + n_idx];
          float q = static_cast<float>(unpacked[flat_idx] - 8);
          w_ptr[k_idx * size_n + n_idx] = at::Half(q * scale);
        }
      }
    }
  }

  return weight_cpu;
}

inline torch::Tensor dequantize_uint4b8_weight_cuda(torch::Tensor q_weight,
                                                    torch::Tensor scales,
                                                    int64_t size_k,
                                                    int64_t size_n,
                                                    int64_t num_groups,
                                                    torch::Device device) {
  auto weight_cpu = dequantize_uint4b8_weight_cpu(
      q_weight.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32),
                  /*non_blocking=*/false, /*copy=*/true),
      scales.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat16),
                /*non_blocking=*/false, /*copy=*/true),
      size_k, size_n, num_groups);
  return weight_cpu.to(device, torch::kFloat16, /*non_blocking=*/false, /*copy=*/true);
}

using Sm70Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70>;

inline torch::Tensor run_cutlass_half_gemm(torch::Tensor a,
                                           torch::Tensor b,
                                           torch::Tensor out) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda(), "CUTLASS GEMM expects CUDA tensors");
  TORCH_CHECK(a.scalar_type() == at::ScalarType::Half &&
                  b.scalar_type() == at::ScalarType::Half &&
                  out.scalar_type() == at::ScalarType::Half,
              "CUTLASS GEMM expects float16 tensors");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(),
              "CUTLASS GEMM expects contiguous tensors");

  auto c_input = torch::zeros_like(out);

  cutlass::gemm::GemmCoord problem_size(
      static_cast<int>(a.size(0)),
      static_cast<int>(b.size(1)),
      static_cast<int>(a.size(1)));

  typename Sm70Gemm::Arguments args{
      problem_size,
      {reinterpret_cast<cutlass::half_t const*>(a.data_ptr<at::Half>()),
       static_cast<int>(a.stride(0))},
      {reinterpret_cast<cutlass::half_t const*>(b.data_ptr<at::Half>()),
       static_cast<int>(b.stride(0))},
      {reinterpret_cast<cutlass::half_t const*>(c_input.data_ptr<at::Half>()),
       static_cast<int>(c_input.stride(0))},
      {reinterpret_cast<cutlass::half_t*>(out.data_ptr<at::Half>()),
       static_cast<int>(out.stride(0))},
      {1.0f, 0.0f},
      1};

  size_t workspace_size = Sm70Gemm::get_workspace_size(args);
  auto workspace = torch::empty(
      {static_cast<long>(workspace_size)},
      torch::TensorOptions().dtype(torch::kUInt8).device(a.device()));

  Sm70Gemm gemm_op;
  auto status = gemm_op.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS SM70 GEMM cannot implement the requested problem.");
  status = gemm_op.initialize(args, workspace.data_ptr(),
                              at::cuda::getCurrentCUDAStream(a.get_device()));
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS SM70 GEMM initialization failed.");
  status = gemm_op(at::cuda::getCurrentCUDAStream(a.get_device()));
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS SM70 GEMM execution failed.");
  return out;
}

inline bool can_use_dense_cutlass(torch::Tensor const& a,
                                  torch::Tensor const& b_q_weight,
                                  torch::Tensor const& b_scales,
                                  bool has_bias,
                                  bool has_act_order,
                                  bool has_zp,
                                  bool is_zp_float,
                                  int64_t b_type_id) {
  return a.scalar_type() == at::ScalarType::Half &&
         b_q_weight.scalar_type() == at::ScalarType::Int &&
         b_scales.scalar_type() == at::ScalarType::Half &&
         !has_bias && !has_act_order && !has_zp && !is_zp_float &&
         b_type_id == vllm::kU4B8.id();
}

inline torch::Tensor run_dense_cutlass(torch::Tensor a,
                                       std::optional<torch::Tensor> c_or_none,
                                       torch::Tensor b_q_weight,
                                       torch::Tensor b_scales,
                                       int64_t size_m,
                                       int64_t size_n,
                                       int64_t size_k) {
  auto weight = dequantize_uint4b8_weight_cuda(
      b_q_weight, b_scales, size_k, size_n, b_scales.size(0), a.device());
  auto out = c_or_none.has_value()
                 ? c_or_none.value()
                 : torch::empty({size_m, size_n},
                                torch::TensorOptions().dtype(torch::kHalf).device(a.device()));
  return run_cutlass_half_gemm(a.contiguous(), weight.contiguous(), out);
}

inline bool can_use_moe_cutlass(torch::Tensor const& a,
                                torch::Tensor const& b_q_weight,
                                torch::Tensor const& b_scales,
                                bool has_bias,
                                bool has_act_order,
                                bool has_zp,
                                bool is_zp_float,
                                int64_t b_type_id) {
  static_cast<void>(a);
  static_cast<void>(b_q_weight);
  static_cast<void>(b_scales);
  static_cast<void>(has_bias);
  static_cast<void>(has_act_order);
  static_cast<void>(has_zp);
  static_cast<void>(is_zp_float);
  static_cast<void>(b_type_id);
  return false;
}

inline torch::Tensor run_moe_cutlass(torch::Tensor a,
                                     std::optional<torch::Tensor> c_or_none,
                                     torch::Tensor b_q_weight,
                                     torch::Tensor b_scales,
                                     torch::Tensor sorted_token_ids,
                                     torch::Tensor expert_ids,
                                     torch::Tensor num_tokens_post_pad,
                                     torch::Tensor topk_weights,
                                     int64_t moe_block_size,
                                     bool mul_topk_weights,
                                     int64_t size_n,
                                     int64_t size_k) {
  static_cast<void>(a);
  static_cast<void>(c_or_none);
  static_cast<void>(b_q_weight);
  static_cast<void>(b_scales);
  static_cast<void>(sorted_token_ids);
  static_cast<void>(expert_ids);
  static_cast<void>(num_tokens_post_pad);
  static_cast<void>(topk_weights);
  static_cast<void>(moe_block_size);
  static_cast<void>(mul_topk_weights);
  static_cast<void>(size_n);
  static_cast<void>(size_k);
  TORCH_CHECK(false,
              "SM70 MoE CUTLASS fallback has been retired; use the GPU Marlin kernel path.");
}

}  // namespace marlin_sm70_cutlass

#include "core/registration.h"

#include <torch/library.h>

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // Note about marlin kernel 'workspace' arguments:
  // Technically these should be mutable since they are modified by the kernel.
  // But since they are set back to zero once the kernel is finished we can
  // hand wave and say that they have no net effect.
  //
  // The reason to mark 'workspace' as immutable is so that they don't
  // interfere with using ScalarType arguments in the ops.
  ops.def(
      "marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, "
      "Tensor? b_bias_or_none,Tensor b_scales, "
      "Tensor? a_scales, Tensor? global_scale, Tensor? b_zeros_or_none, "
      "Tensor? g_idx_or_none, Tensor? perm_or_none, Tensor workspace, "
      "int b_type_id, SymInt size_m, SymInt size_n, SymInt size_k, "
      "bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, "
      "bool is_zp_float) -> Tensor");

  ops.def(
      "gptq_marlin_repack(Tensor b_q_weight, Tensor perm, "
      "SymInt size_k, SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor");

  ops.def(
      "awq_marlin_repack(Tensor b_q_weight, SymInt size_k, "
      "SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor");

  ops.def(
      "marlin_int4_fp8_preprocess(Tensor qweight, "
      "Tensor? qzeros_or_none, bool inplace) -> Tensor");

  ops.def(
      "sm70_cutlass_matmul_probe(Tensor a, Tensor b, int cta_m, int cta_n, "
      "int cta_k, int warps, int stages, int a_path, int b_path) -> Tensor");
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

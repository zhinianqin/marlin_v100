#include "core/registration.h"
#include "moe_ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output, bool renormalize, Tensor? "
      "bias) -> ()");
  m.impl("topk_softmax", torch::kCUDA, &topk_softmax);

  m.def(
      "topk_sigmoid(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output, bool renormalize, Tensor? "
      "bias) -> ()");
  m.impl("topk_sigmoid", torch::kCUDA, &topk_sigmoid);

  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts,"
      "                     int block_size, Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad,"
      "                     Tensor? maybe_expert_map) -> ()");
  m.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

  m.def(
      "batched_moe_align_block_size(int max_tokens_per_batch,"
      "                     int block_size, Tensor expert_num_tokens,"
      "                     Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad) -> ()");
  m.impl("batched_moe_align_block_size", torch::kCUDA,
         &batched_moe_align_block_size);

  m.def(
      "moe_wna16_marlin_gemm(Tensor! a, Tensor? c_or_none,"
      "Tensor! b_q_weight, Tensor? b_bias_or_none,"
      "Tensor! b_scales, Tensor? a_scales, Tensor? global_scale, Tensor? "
      "b_zeros_or_none,"
      "Tensor? g_idx_or_none, Tensor? perm_or_none, Tensor? c_tmp_or_none,"
      "Tensor sorted_token_ids,"
      "Tensor! expert_ids, Tensor! num_tokens_past_padded,"
      "Tensor! topk_weights, int moe_block_size, int top_k, "
      "bool mul_topk_weights, int b_type_id,"
      "int size_m, int size_n, int size_k,"
      "bool is_k_full, bool use_atomic_add,"
      "bool use_fp32_reduce, bool is_zp_float,"
      "int thread_k, int thread_n, int blocks_per_sm) -> Tensor");

  m.def(
      "grouped_topk(Tensor scores, int n_group, int "
      "topk_group, int topk, bool renormalize, float "
      "routed_scaling_factor, Tensor bias, int scoring_func) -> (Tensor, "
      "Tensor)");
  m.impl("grouped_topk", torch::kCUDA, &grouped_topk);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

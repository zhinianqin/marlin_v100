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
  #define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

#include "quantization/marlin/marlin.cuh"
#include "quantization/marlin/marlin_dtypes.cuh"
#include "quantization/marlin/dequant.h"
#include "quantization/marlin/marlin_mma.h"
#include "core/scalar_type.hpp"

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

namespace MARLIN_NAMESPACE_NAME {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700

template <typename scalar_t,  // compute dtype, half or nv_float16
          const vllm::ScalarTypeId b_type_id,  // weight MarlinScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const bool m_block_size_8,  // whether m_block_size == 8
                                      // only works when thread_m_blocks == 1
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const bool has_act_order,  // whether act_order is enabled
          const int group_blocks,    // number of consecutive 16x16 blocks
                                     // with a separate quantization scale
          const bool is_zp_float     // is zero point of float16 type?
          >
__global__ void Marlin(
    const int4* __restrict__ A,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C,        // fp16 output buffer of shape mxn
    int4* __restrict__ C_tmp,    // fp32 tmp output buffer (for reduce)
    const int4* __restrict__ scales_ptr,  // fp16 quantization scales of shape
                                          // (k/groupsize)xn
    const int4* __restrict__ zp_ptr,      // 4bit packed zero-points of shape
                                          // (k/groupsize)x(n/pack_factor)
    const int* __restrict__ g_idx,        // int32 group indices of shape k
    const int32_t* __restrict__ sorted_token_ids_ptr,        // moe sorted_ids
    const int32_t* __restrict__ expert_ids_ptr,              // moe expert ids
    const int32_t* __restrict__ num_tokens_past_padded_ptr,  // moe num tokens
    const float* __restrict__ topk_weights_ptr,              // moe top weights
    int top_k,              // num of experts per token
    bool mul_topk_weights,  // mul topk weights or not
    int num_groups,         // number of scale groups per output channel
    int prob_m,             // batch dimension m
    int prob_n,             // output dimension n
    int prob_k,             // reduction dimension k
    int* locks,             // extra global storage for barrier synchronization
    bool use_atomic_add,    // whether to use atomic add to reduce
    bool use_fp32_reduce    // whether to use fp32 global reduce
) {}

}  // namespace MARLIN_NAMESPACE_NAME

#else

// Instruction for loading a full 16x16 matrix fragment of operand A from shared
// memory, directly in tensor core layout.
template <int count, vllm::ScalarTypeId type_id>
__device__ inline void ldsm(typename MarlinScalarType<type_id>::FragA& frag_a,
                            const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  constexpr uint32_t kFullWarpMask = 0xffffffffu;
  int lane = threadIdx.x & 31;
  int lane_group = lane >> 2;
  uint32_t word_byte_offset =
      static_cast<uint32_t>((lane & 0x3) * sizeof(uint32_t));

  a[0] = 0;
  a[1] = 0;
  a[2] = 0;
  a[3] = 0;

  if constexpr (count == 4) {
    uint32_t smem0 = __shfl_sync(kFullWarpMask, smem, lane_group) +
                     word_byte_offset;
    uint32_t smem1 = __shfl_sync(kFullWarpMask, smem, 8 + lane_group) +
                     word_byte_offset;
    uint32_t smem2 = __shfl_sync(kFullWarpMask, smem, 16 + lane_group) +
                     word_byte_offset;
    uint32_t smem3 = __shfl_sync(kFullWarpMask, smem, 24 + lane_group) +
                     word_byte_offset;
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(a[0]) : "r"(smem0));
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(a[1]) : "r"(smem1));
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(a[2]) : "r"(smem2));
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(a[3]) : "r"(smem3));

  } else if constexpr (count == 2) {
    uint32_t smem0 = __shfl_sync(kFullWarpMask, smem, lane_group) +
                     word_byte_offset;
    uint32_t smem1 = __shfl_sync(kFullWarpMask, smem, 8 + lane_group) +
                     word_byte_offset;
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(a[0]) : "r"(smem0));
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(a[1]) : "r"(smem1));
  } else if constexpr (count == 1) {
    uint32_t smem0 = __shfl_sync(kFullWarpMask, smem, lane_group) +
                     word_byte_offset;
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(a[0]) : "r"(smem0));
  } else {
    static_assert(count == 1 || count == 2 || count == 4, "invalid count");
  }
}

// Multiply dequantized values by the corresponding quantization scale; used
// only for grouped quantization.
template <vllm::ScalarTypeId type_id>
__device__ inline void scale(typename MarlinScalarType<type_id>::FragB& frag_b,
                             typename MarlinScalarType<type_id>::FragS& frag_s,
                             int i) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<type_id>::scalar_t2;
  scalar_t2 s = MarlinScalarType<type_id>::num2num2(
      reinterpret_cast<scalar_t*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

template <vllm::ScalarTypeId type_id>
__device__ inline void scale_and_sub(
    typename MarlinScalarType<type_id>::FragB& frag_b,
    typename MarlinScalarType<type_id>::scalar_t s,
    typename MarlinScalarType<type_id>::scalar_t zp) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<type_id>::scalar_t2;
  scalar_t2 s2 = MarlinScalarType<type_id>::num2num2(s);
  scalar_t2 zp2 = MarlinScalarType<type_id>::num2num2(zp);
  frag_b[0] = __hfma2(frag_b[0], s2, __hneg2(zp2));
  frag_b[1] = __hfma2(frag_b[1], s2, __hneg2(zp2));
}

template <vllm::ScalarTypeId type_id>
__device__ inline void sub_zp(
    typename MarlinScalarType<type_id>::FragB& frag_b,
    typename MarlinScalarType<type_id>::scalar_t2& frag_zp, int i) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<type_id>::scalar_t2;
  scalar_t2 zp = MarlinScalarType<type_id>::num2num2(
      reinterpret_cast<scalar_t*>(&frag_zp)[i]);
  frag_b[0] = __hsub2(frag_b[0], zp);
  frag_b[1] = __hsub2(frag_b[1], zp);
}

// Given 2 floats multiply by 2 scales (halves)
template <vllm::ScalarTypeId type_id>
__device__ inline void scale_float(
    float* c, typename MarlinScalarType<type_id>::FragS& s) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  scalar_t* s_ptr = reinterpret_cast<scalar_t*>(&s);
  c[0] = __fmul_rn(c[0], MarlinScalarType<type_id>::num2float(s_ptr[0]));
  c[1] = __fmul_rn(c[1], MarlinScalarType<type_id>::num2float(s_ptr[1]));
}

// Wait until barrier reaches `count`, then lock for current threadblock.
__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      // Guarantee that subsequent writes by this threadblock will be visible
      // globally.
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

// Release barrier and increment visitation count.
__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    // Make sure that all writes since acquiring this barrier are visible
    // globally, while releasing the barrier.
    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                 :
                 : "l"(lock), "r"(val));
  }
}

// Wait until value of lock to be negative, and then add 1
__device__ inline void wait_negative_and_add(int* lock) {
  if (threadIdx.x == 0) {
    int state = 0;
    do
      // Guarantee that subsequent writes by this threadblock will be visible
      // globally.
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    while (state >= 0);
    atomicAdd(lock, 1);
  }
  __syncthreads();
}

template <const vllm::ScalarTypeId a_type_id,  // A ScalarType id
          const vllm::ScalarTypeId b_type_id,  // B ScalarType id
          const vllm::ScalarTypeId c_type_id,  // C ScalarType id
          const vllm::ScalarTypeId s_type_id,  // B_SCALE ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const bool m_block_size_8,  // whether m_block_size == 8
                                      // only works when thread_m_blocks == 1
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const int group_blocks,  // number of consecutive 16x16 blocks
                                   // with a separate quantization scale
          const bool is_zp_float   // is zero point of float16 type?
          >
__global__ void Marlin(
    const int4* __restrict__ A,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C,        // fp16 output buffer of shape mxn
    int4* __restrict__ C_tmp,    // fp32 tmp output buffer (for reduce)
    const int4* __restrict__ b_bias_ptr,
    // float scales of input matrix, only used when is_a_8bit == true.
    // shape (m,)
    const float* __restrict__ a_scales_ptr,
    // fp16 quantization scales. shape (k/groupsize, n)
    const int4* __restrict__ scales_ptr,
    // fp16 global scale (for nvfp4// only)
    const uint16_t* __restrict__ global_scale_ptr,
    // 4bit packed zero-points of shape
    // (k/groupsize, n/pack_factor)
    const int4* __restrict__ zp_ptr,
    // compatibility parameter
    const int* __restrict__ g_idx,
    const int32_t* __restrict__ sorted_token_ids_ptr,        // moe sorted_ids
    const int32_t* __restrict__ expert_ids_ptr,              // moe expert ids
    const int32_t* __restrict__ num_tokens_past_padded_ptr,  // moe num tokens
    const float* __restrict__ topk_weights_ptr,              // moe top weights
    int top_k,              // num of experts per token
    bool mul_topk_weights,  // mul topk weights or not
    int num_groups,         // number of scale groups per output channel
    int prob_m,             // batch dimension m
    int prob_n,             // output dimension n
    int prob_k,             // reduction dimension k
    int* locks,             // extra global storage for barrier synchronization
    bool has_bias,
    bool use_atomic_add,  // whether to use atomic add to reduce
    bool use_fp32_reduce  // whether to use fp32 global reduce
) {
  static_assert(group_blocks != 0,
                "SM70 Marlin kernels do not support act_order "
                "(group_blocks == 0).");
  // Each threadblock processes one "stripe" of the B matrix with (roughly) the
  // same size, which might involve multiple column "slices" (of width 16 *
  // `thread_n_blocks`). Stripes are defined as shown in the 3x3 matrix 5 SM
  // example:
  //   0 1 3
  //   0 2 3
  //   1 2 4
  // While this kind of partitioning makes things somewhat more complicated, it
  // ensures good utilization of all SMs for many kinds of shape and GPU
  // configurations, while requiring as few slow global cross-threadblock
  // reductions as possible.

  // Volta TensorCore only supports fp16
  if constexpr (a_type_id != vllm::kFloat16.id() && a_type_id != vllm::kS8.id())
    return;

  int num_tokens_past_padded = num_tokens_past_padded_ptr[0];
  constexpr int moe_block_size = m_block_size_8 ? 8 : (16 * thread_m_blocks);

  constexpr bool use_fp16_accum = false;
  using Adtype = MarlinScalarType<a_type_id>;
  using Cdtype = MarlinScalarType<c_type_id>;

  using scalar_t = typename MarlinScalarType<a_type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<a_type_id>::scalar_t2;
  using scalar_32bit_t = typename MarlinScalarType<a_type_id>::scalar_32bit_t;

  using c_scalar_t = typename MarlinScalarType<c_type_id>::scalar_t;
  using c_scalar_t2 = typename MarlinScalarType<c_type_id>::scalar_t2;

  using Sm70FragA = detail::Sm70DirectAFragment<m_block_size_8 ? 1 : 2>;
  using FragB = typename MarlinScalarType<a_type_id>::FragB;
  using Sm70FragC = detail::Sm70Accumulator<m_block_size_8>;
  using FragS = typename MarlinScalarType<c_type_id>::FragS;
  using FragZP = typename MarlinScalarType<c_type_id>::FragZP;

  extern __shared__ int4 sh[];
  static constexpr auto a_type = vllm::ScalarType::from_id(a_type_id);
  static constexpr auto b_type = vllm::ScalarType::from_id(b_type_id);
  static constexpr auto c_type = vllm::ScalarType::from_id(c_type_id);
  static constexpr auto s_type = vllm::ScalarType::from_id(s_type_id);
  if constexpr (b_type == vllm::kFE2M1f) {
    static_assert(s_type == vllm::kFE4M3fn && group_blocks == 1 ||
                  s_type == vllm::kFE8M0fnu && group_blocks == 2);
  } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
    static_assert(s_type == vllm::kBFloat16);
  } else if constexpr (std::is_same<scalar_t, half>::value) {
    static_assert(s_type == vllm::kFloat16);
  }

  static_assert(std::is_same<scalar_t, c_scalar_t>::value);
  constexpr bool has_zp = b_type == vllm::kU4 || b_type == vllm::kU8;
  constexpr bool is_int_type = b_type == vllm::kU4 || b_type == vllm::kU8 ||
                               b_type == vllm::kS4 || b_type == vllm::kS8 ||
                               b_type == vllm::kU4B8 || b_type == vllm::kU8B128;
  // see comments of dequant.h for more details
  constexpr bool dequant_skip_flop =
      b_type == vllm::kFE4M3fn ||
      b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn ||
      has_zp && !is_zp_float && !std::is_same<scalar_t, nv_bfloat16>::value ||
      has_zp && !is_zp_float && !(b_type == vllm::kU8);

  c_scalar_t2 global_scale;
  constexpr int sh_red_rows = moe_block_size;
  constexpr int sh_red_cols = 16 * thread_n_blocks;
  constexpr int sh_red_swizzle_mask = 0x7;
  constexpr int sh_red_stride = sh_red_cols + 8;

  constexpr int pack_factor = 32 / b_type.size_bits();
  static_assert(thread_m_blocks == 1 || !m_block_size_8);
  const int group_size = group_blocks == -1 ? prob_k : prob_k / num_groups;
  const int scales_expert_stride =
      prob_n * prob_k / group_size / (b_type == vllm::kFE2M1f ? 16 : 8);
  const int zp_expert_stride =
      is_zp_float ? prob_n * prob_k / group_size / 8
                  : prob_n * prob_k / group_size / (pack_factor * 4);
  const int b_bias_expert_stride = prob_n / 8;

  // parallel: num valid moe blocks
  int parallel = num_tokens_past_padded / moe_block_size;

  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;

  int global_mn_tiles = parallel * n_tiles;
  int part2_mn_tiles = global_mn_tiles;
  int part1_mn_iters = 0;
  bool in_part2 = false;

  // we use DP + two-tile SK here
  // part1: DP
  // part2: two-tile SK
  // see https://github.com/vllm-project/vllm/pull/24722 for more details
  if (global_mn_tiles > gridDim.x) {
    part2_mn_tiles = global_mn_tiles % gridDim.x;
    if (part2_mn_tiles * 3 <= gridDim.x) part2_mn_tiles += gridDim.x;
    part1_mn_iters = (global_mn_tiles - part2_mn_tiles) / gridDim.x;
  }

  int iters = div_ceil(k_tiles * part2_mn_tiles, gridDim.x);

  if constexpr (group_blocks != -1) {
    if (group_blocks >= thread_k_blocks) {
      // Ensure that the number of tiles in each stripe is a multiple of the
      // groupsize; this avoids an annoying special case where a stripe starts
      // in the middle of group.
      iters = (group_blocks / thread_k_blocks) *
              div_ceil(iters, (group_blocks / thread_k_blocks));
    }
  }

  int slice_row = 0;
  int slice_col_par = blockIdx.x;
  int slice_col;
  int slice_iters =
      k_tiles;  // number of threadblock tiles in the current slice
  // total number of active threadblocks in the current slice
  int slice_count = 1;
  // index of threadblock in current slice; numbered bottom to top
  int slice_idx = 0;

  int par_id = 0;
  int block_id = -1;
  int64_t expert_id = 0;  // use int64 to avoid computation result overflow
  int old_expert_id = 0;
  int64_t B_expert_off = 0;

  int4* sh_block_sorted_ids_int4 = sh;
  int4* sh_rd_block_sorted_ids_int4 =
      sh_block_sorted_ids_int4 + moe_block_size / 4;
  int4* sh_block_topk_weights_int4 =
      sh_rd_block_sorted_ids_int4 + moe_block_size / 4;
  // sh_block_topk_weights_int4 only need (moe_block_size / 4);
  // but we pad to align to 256 bytes
  int4* sh_new = sh_block_topk_weights_int4 + moe_block_size / 2;
  int32_t* sh_block_sorted_ids =
      reinterpret_cast<int*>(sh_block_sorted_ids_int4);
  int32_t* sh_rd_block_sorted_ids =
      reinterpret_cast<int*>(sh_rd_block_sorted_ids_int4);
  c_scalar_t2* sh_block_topk_weights =
      reinterpret_cast<c_scalar_t2*>(sh_block_topk_weights_int4);

  int32_t block_num_valid_tokens = 0;
  int32_t locks_off = 0;

  // We can easily implement parallel problem execution by just remapping
  // indices and advancing global pointers
  if (part2_mn_tiles >= gridDim.x) {
    // when part2_mn_tiles >= sms
    // then there are at most $sms$ conflict tile blocks
    locks_off = blockIdx.x;
  } else {
    locks_off = (iters * blockIdx.x) / k_tiles - 1;
  }

  int prob_m_top_k = prob_m * top_k;
  // read moe block data given block_id
  // block_sorted_ids / block_num_valid_tokens / block_topk_weights
  auto read_moe_block_data = [&](int block_id) {
    block_num_valid_tokens = moe_block_size;

    cp_async4_pred(sh_block_sorted_ids_int4 + threadIdx.x,
                   reinterpret_cast<const int4*>(sorted_token_ids_ptr) +
                       (block_id * moe_block_size / 4 + threadIdx.x),
                   threadIdx.x < moe_block_size / 4);

    cp_async_fence();
    cp_async_wait<0>();

    __syncthreads();

    if (threadIdx.x >= threads - 32) {
      constexpr int size_per_thread = div_ceil(moe_block_size, 32);
      int lane_id = threadIdx.x - (threads - 32);

      int local_count = 0;
  #pragma unroll
      for (int i = 0; i < size_per_thread; i++) {
        int j = lane_id * size_per_thread + i;
        if (j < moe_block_size) {
          int idx = sh_block_sorted_ids[j];
          if (idx < prob_m_top_k) local_count++;
        }
      }

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700

      if constexpr (moe_block_size >= 16)
        local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 16);
      if constexpr (moe_block_size >= 8)
        local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 8);
      if constexpr (moe_block_size >= 4)
        local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 4);
      if constexpr (moe_block_size >= 2)
        local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 2);

      local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 1);
      block_num_valid_tokens = local_count;
  #else
      block_num_valid_tokens = __reduce_add_sync(0xffffffff, local_count);
  #endif

      if (lane_id == 0)
        reinterpret_cast<int*>(sh_new)[0] = block_num_valid_tokens;
    }

    if (threadIdx.x < moe_block_size) {
      int idx = sh_block_sorted_ids[threadIdx.x];
      sh_rd_block_sorted_ids[threadIdx.x] = idx / top_k;

      if (mul_topk_weights) {
        idx = idx < prob_m_top_k ? idx : 0;
        c_scalar_t2 topk_weight_val =
            Cdtype::num2num2(Cdtype::float2num(topk_weights_ptr[idx]));
        if constexpr (b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn) {
          topk_weight_val = __hmul2(topk_weight_val, global_scale);
        }
        sh_block_topk_weights[threadIdx.x] = topk_weight_val;
      }
    }

    __syncthreads();

    block_num_valid_tokens = reinterpret_cast<int*>(sh_new)[0];
    __syncthreads();
  };

  // when move to next moe block, find the next block_id and expert_id
  // and then read moe block data
  auto update_next_moe_block_data = [&]() {
    if (par_id >= parallel) return;

    old_expert_id = expert_id;
    block_id = par_id;
    expert_id = expert_ids_ptr[block_id];

    if constexpr (b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn) {
      uint16_t val = global_scale_ptr[expert_id];
      global_scale = Cdtype::num2num2(*reinterpret_cast<c_scalar_t*>(&val));
    }

    B_expert_off = expert_id * prob_n * prob_k / (pack_factor * 4);
    scales_ptr += (expert_id - old_expert_id) * scales_expert_stride;
    if constexpr (has_zp) {
      zp_ptr += (expert_id - old_expert_id) * zp_expert_stride;
    }
    if (has_bias) {
      b_bias_ptr += (expert_id - old_expert_id) * b_bias_expert_stride;
    }

    read_moe_block_data(block_id);
  };

  // Compute all information about the current slice which is required for
  // synchronization.
  bool first_init = true;
  auto init_part2_slice = [&]() {
    slice_iters =
        iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= part2_mn_tiles) slice_iters = 0;
    if (slice_iters == 0) return;
    if (slice_row + slice_iters > k_tiles) slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * div_ceil(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = div_ceil(k_tiles - col_off, iters);
      if (col_off > 0) slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0) slice_idx--;
      }
    }
    if (part2_mn_tiles >= gridDim.x) {
      if (slice_count > 1 && slice_idx == slice_count - 1) {
        locks_off++;
      }
    } else {
      locks_off++;
    }

    if (first_init && use_atomic_add && slice_count > 1 && slice_idx == 0) {
      constexpr int threads_per_m = 16 * thread_n_blocks / 8;
      int m_per_thread =
          div_ceil(block_num_valid_tokens, threads / threads_per_m);
      for (int i = 0; i < m_per_thread; i++) {
        int row = threads / threads_per_m * i + threadIdx.x / threads_per_m;
        if (row < block_num_valid_tokens) {
          int64_t sorted_row = sh_block_sorted_ids[row];
          int col = slice_col * 16 * thread_n_blocks / 8 +
                    threadIdx.x % threads_per_m;
          C[sorted_row * prob_n / 8 + col] = {0, 0, 0, 0};
        }
      }
      // After write zero to output, write a negative value to lock.
      // Every SM that processes the same slice would wait for
      // the negative value, and then atomicAdd 1 to it.
      // After all SMs are processed, the lock value would back to 0 again.
      __syncthreads();
      if (threadIdx.x == 0) locks[locks_off] = 1 - slice_count;
    }

    if (slice_col == n_tiles) {
      slice_col = 0;
      par_id++;
      update_next_moe_block_data();
    }
  };

  auto init_part1_slice = [&]() {
    if (part1_mn_iters) {
      part1_mn_iters--;
      par_id = slice_col_par / n_tiles;
      slice_col = slice_col_par % n_tiles;
      slice_iters = k_tiles;
      update_next_moe_block_data();
    }
  };

  auto init_slice = [&]() {
    if (!in_part2 && !part1_mn_iters) {
      in_part2 = true;
      slice_col_par = (iters * blockIdx.x) / k_tiles;
      slice_row = (iters * blockIdx.x) % k_tiles;
      slice_col = (slice_col_par + global_mn_tiles - part2_mn_tiles) % n_tiles;
      par_id = (slice_col_par + global_mn_tiles - part2_mn_tiles) / n_tiles;
      update_next_moe_block_data();
    }
    if (!in_part2) {
      init_part1_slice();
    } else {
      init_part2_slice();
      first_init = false;
    }
  };

  init_slice();

  // A sizes/strides

  // stride of the A matrix in global memory
  int a_gl_stride = prob_k / 8;
  // stride of an A matrix tile in shared memory
  constexpr int a_sh_stride = 16 * thread_k_blocks / 8;
  // delta between subsequent A tiles in global memory
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;
  // between subsequent accesses within a tile
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
  // between shared memory writes
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
  // overall size of a tile
  constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);
  // number of shared write iterations for a tile
  constexpr int a_sh_wr_iters = div_ceil(a_sh_stage, a_sh_wr_delta);

  // B sizes/strides
  int b_gl_stride = 16 * prob_n / (pack_factor * 4);
  constexpr int b_sh_stride = ((thread_n_blocks * 16) * 16 / pack_factor) / 4;
  constexpr int b_thread_vecs = b_type.size_bits() == 4 ? 1 : 2;
  constexpr int b_sh_stride_threads = b_sh_stride / b_thread_vecs;

  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
  constexpr int b_sh_wr_delta = threads * b_thread_vecs;
  constexpr int b_sh_stage =
      b_sh_stride * thread_k_blocks;
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  // Scale sizes/strides
  int s_gl_stride = prob_n / (b_type == vllm::kFE2M1f ? 16 : 8);
  constexpr int s_sh_stride =
      16 * thread_n_blocks / (b_type == vllm::kFE2M1f ? 16 : 8);
  constexpr int s_tb_groups = group_blocks != -1 && group_blocks < thread_k_blocks
          ? thread_k_blocks / group_blocks
          : 1;
  constexpr int s_sh_stage = s_tb_groups * s_sh_stride;
  int s_gl_rd_delta = s_gl_stride;

  constexpr int tb_n_warps = thread_n_blocks / 4;

  // Zero-points sizes/strides
  int zp_gl_stride = is_zp_float ? prob_n / 8 : (prob_n / pack_factor) / 4;
  constexpr int zp_sh_stride = is_zp_float
                                   ? 16 * thread_n_blocks / 8
                                   : ((16 * thread_n_blocks) / pack_factor) / 4;
  constexpr int zp_tb_groups = s_tb_groups;
  constexpr int zp_sh_stage = has_zp ? zp_tb_groups * zp_sh_stride : 0;
  int zp_gl_rd_delta = zp_gl_stride;

  // Global A read index of current thread.
  int a_gl_rd_row = threadIdx.x / a_gl_rd_delta_o;
  int a_gl_rd_col = a_gl_rd_delta_o * slice_row + threadIdx.x % a_gl_rd_delta_o;
  // Shared write index of current thread.
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  int b_gl_rd;
  if (threads <= b_sh_stride) {
    b_gl_rd = threadIdx.x;
  } else {
    b_gl_rd =
        b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
  }

  b_gl_rd += B_expert_off + b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  [[maybe_unused]] auto b_sh_rd = threadIdx.x * b_thread_vecs;
  b_sh_rd += b_sh_rd / b_sh_stride * (b_sh_stride * (b_sh_wr_iters - 1));

  int s_gl_rd;
  if constexpr (group_blocks == -1) {
    s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
  } else if constexpr (group_blocks >= thread_k_blocks) {
    s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
              s_sh_stride * slice_col + threadIdx.x;
  } else {
    s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                             threadIdx.x / s_sh_stride) +
              s_sh_stride * slice_col + threadIdx.x % s_sh_stride;
  }
  auto s_sh_wr = threadIdx.x;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stage;

  // Zero-points
  int zp_gl_rd;
  if constexpr (has_zp) {
    if constexpr (group_blocks == -1) {
      zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
    } else if constexpr (group_blocks >= thread_k_blocks) {
      zp_gl_rd = zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                 zp_sh_stride * slice_col + threadIdx.x;
    } else {
      zp_gl_rd = zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                                 threadIdx.x / zp_sh_stride) +
                 zp_sh_stride * slice_col + threadIdx.x % zp_sh_stride;
    }
  }
  auto zp_sh_wr = threadIdx.x;
  bool zp_sh_wr_pred = zp_sh_stage > 0 && threadIdx.x < zp_sh_stage;

  // We use a different scale layout for grouped and column-wise quantization as
  // we scale a `half2` tile in column-major layout in the former and in
  // row-major in the latter case.
  int s_sh_rd;
  if constexpr (group_blocks != -1)
    s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) +
              detail::sm70_atom_rowcol(threadIdx.x % 32);
  else if constexpr (group_blocks == -1 &&
                     (m_block_size_8 || (has_zp && !dequant_skip_flop)))
    s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 8;
  else
    s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) % 4;

  int bias_sh_wr = threadIdx.x;
  int bias_gl_rd = (thread_n_blocks * 16 / 8) * slice_col + threadIdx.x;

  // Zero-points have the same read layout as the scales
  // (without column-wise case)
  constexpr int num_col_threads = 8;
  constexpr int num_row_threads = 4;
  constexpr int num_ints_per_thread = 8 / pack_factor;
  int zp_sh_rd;
  if constexpr (has_zp) {
    if constexpr (is_zp_float) {
      if constexpr (group_blocks != -1) {
        zp_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) +
                   detail::sm70_atom_rowcol(threadIdx.x % 32);
      }
    } else {
      zp_sh_rd = num_ints_per_thread * num_col_threads *
                     ((threadIdx.x / 32) % tb_n_warps) +
                 num_ints_per_thread *
                     detail::sm70_atom_rowcol(threadIdx.x % 32);
    }
  }

  // To ensure that writing and reading A tiles to/from shared memory, the
  // latter in fragment format, is fully bank conflict free, we need to use a
  // rather fancy XOR-based layout. The key here is that neither reads nor
  // writes of the 16-byte `int4` blocks of 8 consecutive threads involve the
  // same shared memory banks. Further, it seems (based on NSight-Compute) that
  // each warp must also write a consecutive memory segment?
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ (row % 8);
  };
  constexpr int sm70_m_halves = m_block_size_8 ? 1 : 2;
  int warp_id = threadIdx.x / 32;
  int warp_row = warp_id / tb_n_warps;
  int warp_col = warp_id % tb_n_warps;
  int lane = threadIdx.x & 31;
  int sm70_lane_row = detail::sm70_atom_rowcol(lane);
  constexpr int sm70_b_j_groups = 4;
  // Since the computation of this remapping is non-trivial and, due to our main
  // loop unrolls, all shared memory accesses are static, we simply precompute
  // both transformed reads and writes.
  int a_sh_wr_trans[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_direct_bytes[b_sh_wr_iters][thread_m_blocks][2][sm70_m_halves][2][2];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
  #pragma unroll
    for (int j = 0; j < thread_m_blocks; j++) {
    #pragma unroll
      for (int k_block = 0; k_block < 2; ++k_block) {
      #pragma unroll
        for (int m_half = 0; m_half < sm70_m_halves; ++m_half) {
          int row = 16 * j + 8 * m_half + sm70_lane_row;
        #pragma unroll
          for (int k_slice = 0; k_slice < 2; ++k_slice) {
        #pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
              int col = 16 * (warp_row * b_sh_wr_iters + i) + 8 * k_block +
                        4 * k_slice + 2 * pair;
              int chunk = transform_a(row * a_sh_stride + col / 8);
              a_sh_rd_direct_bytes[i][j][k_block][m_half][k_slice][pair] =
                  chunk * sizeof(int4) +
                  ((col & 0x7) / 2) * sizeof(uint32_t);
            }
          }
        }
      }
    }
  }
  int b_sh_rd_direct_bytes[b_sh_wr_iters][sm70_b_j_groups][b_thread_vecs][4];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; ++i) {
  #pragma unroll
    for (int j = 0; j < sm70_b_j_groups; ++j) {
      int local_k_block = warp_row * b_sh_wr_iters + i;
      int local_n_block = 4 * warp_col + j;
    #pragma unroll
      for (int vec = 0; vec < b_thread_vecs; ++vec) {
        int chunk = b_sh_stride * local_k_block +
                    local_n_block * (8 * b_thread_vecs) +
                    sm70_lane_row * b_thread_vecs + vec;
      #pragma unroll
        for (int row_group = 0; row_group < 4; ++row_group) {
          b_sh_rd_direct_bytes[i][j][vec][row_group] =
              chunk * sizeof(int4) + row_group * sizeof(uint32_t);
        }
      }
    }
  }

  // Since B-accesses have non-constant stride they have to be computed at
  // runtime; we break dependencies between subsequent accesses with a tile by
  // maintining multiple pointers (we have enough registers), a tiny
  // optimization.

  // Shared memory storage for global fetch pipelines.
  constexpr int sh_red_size =
      div_ceil(sh_red_rows * sh_red_stride * int(sizeof(float)),
               int(sizeof(int4)));
  constexpr int sh_b_size = stages * b_sh_stage;
  int4* sh_b = sh_new;
  int4* sh_red = sh_new;
  float* sh_red_f32 = reinterpret_cast<float*>(sh_red);

  constexpr int sh_size_b_red_min =
      (sh_red_size < sh_b_size ? sh_red_size : sh_b_size);
  constexpr int sh_size_b_red_max =
      (sh_red_size > sh_b_size ? sh_red_size : sh_b_size);
  constexpr int sh_bias_size = (thread_n_blocks * 16 / 8);
  constexpr int sh_b_red_bias_size =
      sh_size_b_red_max > (sh_size_b_red_min + sh_bias_size)
          ? sh_size_b_red_max
          : (sh_size_b_red_min + sh_bias_size);

  int4* sh_bias = sh_new + sh_size_b_red_min;
  int4* sh_zp = sh_new + sh_b_red_bias_size;
  constexpr int sh_s_size = stages * s_sh_stage;
  int4* sh_s = sh_zp + (stages * zp_sh_stage);
  int4* sh_a = sh_s + sh_s_size;

  // Register storage for double buffer of shared memory reads.
  using Sm70FragBQuant = detail::Sm70DirectBQuant<b_thread_vecs>;
  Sm70FragA frag_a[2][thread_m_blocks];
  Sm70FragBQuant frag_b_quant[2][4];
  Sm70FragC sm70_frag_c[thread_m_blocks][4];
  FragS frag_s[2][4];
  int frag_qzp[2][num_ints_per_thread];  // Zero-points
  FragZP frag_zp;                        // Zero-points in fp16
  FragZP frag_zpf[2];                    // Zero-points in fp16 in HQQ


  // Zero accumulators.
  auto zero_accums = [&]() {
  #pragma unroll
    for (int i = 0; i < thread_m_blocks; ++i) {
    #pragma unroll
      for (int j = 0; j < 4; ++j) {
        detail::zero_sm70_accumulator(sm70_frag_c[i][j]);
      }
    }
  };

  // Asynchronously fetch the next A, B and s tile from global to the next
  // shared memory pipeline location.
  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4* sh_a_stage = sh_a + moe_block_size * a_sh_stride * pipe;
  #pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        int row = a_gl_rd_delta_i / a_gl_stride * i + a_gl_rd_row;
        int64_t sorted_row = 0;
        if (!m_block_size_8 || row < 8)
          sorted_row = sh_rd_block_sorted_ids[row];
        int64_t true_idx =
            sorted_row * a_gl_stride + a_gl_rd_col + a_gl_rd_delta_o * a_off;
        cp_async4_pred(&sh_a_stage[a_sh_wr_trans[i]], &A[true_idx],
                       row < block_num_valid_tokens);
      }

      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < (b_sh_wr_iters * b_thread_vecs); i++) {
        constexpr int count = div_ceil(b_sh_stride, threads);
        int b_gl_idx =
            b_gl_rd + (i % count) * threads +
            b_gl_stride * (i / count) * div_ceil(threads, b_sh_stride);

        cp_async4(&sh_b_stage[threads * i + threadIdx.x], &B[b_gl_idx]);
      }

      b_gl_rd += b_gl_rd_delta_o;

      if constexpr (group_blocks != -1) {
        int4* sh_s_stage = sh_s + s_sh_stage * pipe;

        // Only fetch scales if this tile starts a new group
        if (pipe % div_ceil(group_blocks, thread_k_blocks) == 0) {
          if (s_sh_wr_pred) {
            cp_async4(&sh_s_stage[s_sh_wr], &scales_ptr[s_gl_rd]);
          }
          s_gl_rd += s_gl_rd_delta * s_tb_groups;
        }
      }

      if constexpr (has_zp && group_blocks != -1) {
        int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

        // Only fetch zero points if this tile starts a new group
        if (pipe % div_ceil(group_blocks, thread_k_blocks) == 0) {
          if (zp_sh_wr_pred) {
            cp_async4(&sh_zp_stage[zp_sh_wr], &zp_ptr[zp_gl_rd]);
          }
          zp_gl_rd += zp_gl_rd_delta * zp_tb_groups;
        }
      }
    }
    // Insert a fence even when we are winding down the pipeline to ensure that
    // waiting is also correct at this point.
    cp_async_fence();
  };

  auto fetch_col_zp_to_shared = [&]() {
    if (zp_sh_wr_pred) {
      cp_async4(&sh_zp[zp_sh_wr], &zp_ptr[zp_gl_rd]);
    }
  };

  auto fetch_col_scale_to_shared = [&]() {
    if (s_sh_wr_pred) {
      cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
    }
  };

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering
    // and can only issue the next fetch when it is guaranteed that the previous
    // shared memory load is fully complete (as it may otherwise be
    // overwritten).
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  // Load the next sub-tile from the current location in the shared memory pipe
  // into the current register buffer.
  auto fetch_to_registers = [&](int k, int pipe) {
    int4* sh_a_stage = sh_a + moe_block_size * a_sh_stride * pipe;
  #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++) {
      detail::load_sm70_direct_a<sm70_m_halves>(
          frag_a[k % 2][i], sh_a_stage,
          a_sh_rd_direct_bytes[k % b_sh_wr_iters][i]);
    }
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;

  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      detail::load_sm70_direct_b<b_thread_vecs>(
          frag_b_quant[k % 2][j], sh_b_stage,
          b_sh_rd_direct_bytes[k % b_sh_wr_iters][j]);
    }
  };
  auto fetch_scales_to_registers = [&](int k, int full_pipe) {
    int pipe = full_pipe % stages;
    using IT1 = int4;
    using IT0 = int2;
    constexpr int group_blocks2 = div_ceil(group_blocks, 1);

    if constexpr (group_blocks == -1) {
        // load only when starting a new slice
        if (k == 0 && full_pipe == 0 && dequant_skip_flop) {
          reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd];
          reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
        }
    } else if constexpr (group_blocks != -1) {
        if constexpr (group_blocks >= thread_k_blocks) {
          constexpr int g = group_blocks / thread_k_blocks;
          if (pipe % g == 0) {
            if (k % b_sh_wr_iters == 0) {
              int4* sh_s_stage = sh_s + s_sh_stage * (g * (pipe / g));
              reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
            } else {
              reinterpret_cast<int4*>(&frag_s[1])[0] =
                  reinterpret_cast<int4*>(&frag_s[0])[0];
            }
          }
        } else if (group_blocks2 < b_sh_wr_iters || k % b_sh_wr_iters == 0) {
          auto warp_id = threadIdx.x / 32;
          int warp_row = warp_id / tb_n_warps;

          int k_blocks = b_sh_wr_iters * warp_row + k % b_sh_wr_iters;
          int cur_group_id = k_blocks / group_blocks2;

          int4* sh_s_stage = sh_s + s_sh_stage * pipe;

          if constexpr (b_type_id != vllm::kFE2M1f.id()) {
            reinterpret_cast<int4*>(&frag_s[k % 2])[0] =
                sh_s_stage[s_sh_rd + cur_group_id * s_sh_stride];
          } else {
            reinterpret_cast<int2*>(&frag_s[k % 2])[0] =
                reinterpret_cast<int2*>(
                    sh_s_stage)[s_sh_rd + cur_group_id * (2 * s_sh_stride)];
          }
        } else if (group_blocks >= b_sh_wr_iters) {
          if constexpr (b_type_id != vllm::kFE2M1f.id()) {
            reinterpret_cast<int4*>(&frag_s[1])[0] =
                reinterpret_cast<int4*>(&frag_s[0])[0];
          } else {
            reinterpret_cast<int2*>(&frag_s[1])[0] =
                reinterpret_cast<int2*>(&frag_s[0])[0];
          }
        }
      }
  };

  auto fetch_zp_to_registers = [&](int k, int full_pipe) {
    static_assert(!has_zp || group_blocks != 0);

    if constexpr (has_zp && !is_zp_float) {
      int pipe = full_pipe % stages;

      if constexpr (group_blocks == -1) {
        // load only when starting a new slice
        if (k == 0 && full_pipe == 0) {
  #pragma unroll
          for (int i = 0; i < num_ints_per_thread; i++) {
            frag_qzp[k % 2][i] = (reinterpret_cast<int*>(sh_zp))[zp_sh_rd + i];
          }
        }
      } else if constexpr (group_blocks >= thread_k_blocks) {
        constexpr int g = group_blocks / thread_k_blocks;
        if (pipe % g == 0 && k % b_sh_wr_iters == 0) {
          int4* sh_zp_stage = sh_zp + zp_sh_stage * (g * (pipe / g));
  #pragma unroll
          for (int i = 0; i < num_ints_per_thread; i++) {
            frag_qzp[k % 2][i] =
                (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
          }
        }
      } else {
        auto warp_id = threadIdx.x / 32;

        int warp_row = warp_id / tb_n_warps;

        int k_blocks = b_sh_wr_iters * warp_row + k % b_sh_wr_iters;
        int cur_group_id = k_blocks / div_ceil(group_blocks, 1);

        int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

        sh_zp_stage += cur_group_id * zp_sh_stride;

  #pragma unroll
        for (int i = 0; i < num_ints_per_thread; i++) {
          frag_qzp[k % 2][i] =
              (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
        }
      }
    }

    else if constexpr (has_zp && is_zp_float) {
      int pipe = full_pipe % stages;

      if constexpr (group_blocks != -1) {
        if constexpr (group_blocks >= thread_k_blocks) {
          constexpr int g = group_blocks / thread_k_blocks;
          if (pipe % g == 0 && k % b_sh_wr_iters == 0) {
            int4* sh_zp_stage = sh_zp + zp_sh_stage * (g * (pipe / g));
            reinterpret_cast<int4*>(&frag_zpf[k % 2])[0] =
                sh_zp_stage[zp_sh_rd];
          }
        } else if (group_blocks < b_sh_wr_iters || k % b_sh_wr_iters == 0) {
          auto warp_id = threadIdx.x / 32;

          int warp_row = warp_id / tb_n_warps;
          int k_blocks = b_sh_wr_iters * warp_row + k % b_sh_wr_iters;
          int cur_group_id = k_blocks / group_blocks;

          int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

          reinterpret_cast<int4*>(&frag_zpf[k % 2])[0] =
              sh_zp_stage[zp_sh_rd + cur_group_id * zp_sh_stride];
        }
      }
    }
  };

  auto dequant_data = [&](int q, scalar_32bit_t* frag_b_ptr, int zp = 0) {
    if constexpr (a_type.size_bits() != b_type.size_bits()) {
      dequant<scalar_32bit_t, b_type_id, dequant_skip_flop>(q, frag_b_ptr);
    }
  };

  // Execute the actual tensor core matmul of a sub-tile.
  bool is_first_matmul_in_slice = true;
  auto matmul = [&](int k, int pipe) {
    int k2 = k % 2;
    constexpr int g =
        group_blocks > 0 ? div_ceil(group_blocks, thread_k_blocks) : 1;
    const bool is_new_zp =
        ((group_blocks > 0) && (group_blocks < b_sh_wr_iters || k == 0)) &&
            (pipe % g == 0) ||
        (group_blocks == -1 && is_first_matmul_in_slice);
    if constexpr (has_zp && !is_zp_float) {
      if (is_new_zp) {
        if constexpr (group_blocks == -1) is_first_matmul_in_slice = false;
        int zp_quant_0, zp_quant_1;

        if constexpr (b_type.size_bits() == 4) {
          zp_quant_0 = frag_qzp[k2][0];
          zp_quant_1 = zp_quant_0 >> 8;
        } else {
          static_assert(b_type.size_bits() == 8);
          zp_quant_0 = frag_qzp[k2][0];
          zp_quant_1 = frag_qzp[k2][1];
        }

        dequant_data(zp_quant_0, reinterpret_cast<scalar_32bit_t*>(&frag_zp));
        dequant_data(zp_quant_1,
                     reinterpret_cast<scalar_32bit_t*>(&frag_zp) + 2);
      }
    }
    if constexpr (!dequant_skip_flop && has_zp && is_zp_float) {
      if (is_new_zp) {
        reinterpret_cast<int4*>(&frag_zp)[0] =
            reinterpret_cast<int4*>(&frag_zpf[k2])[0];
      }
    }

    if constexpr (b_type == vllm::kFE2M1f) {
      int s_quant_0 = reinterpret_cast<int*>(frag_s[k2])[0];
      int s_quant_1 = reinterpret_cast<int*>(frag_s[k2])[1];

      dequant_fp8_scales<c_scalar_t2, s_type_id>(
          s_quant_0, reinterpret_cast<c_scalar_t2*>(&frag_s[k2]));
      dequant_fp8_scales<c_scalar_t2, s_type_id>(
          s_quant_1, reinterpret_cast<c_scalar_t2*>(&frag_s[k2]) + 2);
    }

  // We have the m dimension as the inner loop in order to encourage overlapping
  // dequantization and matmul operations.
  #pragma unroll
    for (int j = 0; j < 4; j++) {
      scalar_t scale_vals[2];
      if constexpr (!dequant_skip_flop && has_zp && !is_zp_float &&
                           group_blocks == -1) {
        int idx = (threadIdx.x / 4) % 2;
        scalar_t2 s2 = Adtype::nums2num2(
            reinterpret_cast<scalar_t*>(&frag_s[j / 2][j % 2 * 2 + 0])[idx],
            reinterpret_cast<scalar_t*>(&frag_s[j / 2][j % 2 * 2 + 1])[idx]);
        if (is_new_zp) frag_zp[j] = __hmul2(frag_zp[j], s2);
        scale_vals[0] = s2.x;
        scale_vals[1] = s2.y;
      } else if constexpr (!dequant_skip_flop && has_zp && group_blocks != -1) {
        if (is_new_zp)
          frag_zp[j] = __hmul2(frag_zp[j],
                               *reinterpret_cast<scalar_t2*>(&frag_s[k2][j]));
        scale_vals[0] = frag_s[k2][j][0].x;
        scale_vals[1] = frag_s[k2][j][0].y;
      }

      auto dequant_sm70_half = [&](int out_half, FragB (&frag_b_group)[4]) {
      #pragma unroll
        for (int row_group = 0; row_group < 4; ++row_group) {
          int b_quant;
          if constexpr (b_type.size_bits() == 4) {
            uint32_t packed_word = frag_b_quant[k2][j].words[0][row_group];
            b_quant =
                static_cast<int>(out_half == 0 ? packed_word : (packed_word >> 8));
          } else {
            static_assert(b_type.size_bits() == 8);
            b_quant = static_cast<int>(frag_b_quant[k2][j].words[out_half]
                                                          [row_group]);
          }
          dequant_data(b_quant, reinterpret_cast<scalar_32bit_t*>(
                                    &frag_b_group[row_group]));
        }
      };

      auto apply_sm70_post_ops = [&](FragB (&frag_b_group)[4], int out_half) {
      #pragma unroll
        for (int row_group = 0; row_group < 4; ++row_group) {
          if constexpr (dequant_skip_flop && has_zp && !is_zp_float) {
            sub_zp<a_type_id>(frag_b_group[row_group], frag_zp[j], out_half);
          }

          if constexpr (!dequant_skip_flop && has_zp && !is_zp_float) {
            scalar_t zp_val = out_half == 0 ? frag_zp[j].x : frag_zp[j].y;
            scale_and_sub<a_type_id>(frag_b_group[row_group],
                                     scale_vals[out_half], zp_val);
          } else if constexpr (group_blocks != -1) {
            scale<a_type_id>(frag_b_group[row_group], frag_s[k2][j], out_half);
          }
        }
      };

      #pragma unroll
      for (int out_half = 0; out_half < 2; ++out_half) {
        FragB frag_b_group[4];
        dequant_sm70_half(out_half, frag_b_group);
        apply_sm70_post_ops(frag_b_group, out_half);

        #pragma unroll
        for (int i = 0; i < thread_m_blocks; i++) {
          if constexpr (m_block_size_8) {
            detail::mma_sm70_direct_a_m8_half(
                frag_a[k2][i], frag_b_group,
                sm70_frag_c[i][j].accum[out_half][0]);
          } else {
            detail::mma_sm70_direct_a_native(
                frag_a[k2][i], frag_b_group, sm70_frag_c[i][j].accum[out_half]);
          }
        }
      }
    }
  };

  auto sh_red_index = [&](int row, int col) {
    int phys_col = col ^ (row & sh_red_swizzle_mask);
    return row * sh_red_stride + phys_col;
  };

  auto sh_red_load = [&](int row, int col) {
    return sh_red_f32[sh_red_index(row, col)];
  };

  auto clear_sh_red = [&]() {
    for (int idx = threadIdx.x; idx < sh_red_rows * sh_red_stride;
         idx += threads) {
      sh_red_f32[idx] = 0.0f;
    }
    __syncthreads();
  };

  auto dump_sm70_to_smem = [&]() {
    clear_sh_red();
    if (detail::sm70_atom_is_canonical_lane(lane)) {
  #pragma unroll
      for (int i = 0; i < thread_m_blocks; ++i) {
  #pragma unroll
        for (int j = 0; j < 4; ++j) {
          if constexpr (m_block_size_8) {
  #pragma unroll
            for (int out_half = 0; out_half < 2; ++out_half) {
  #pragma unroll
              for (int vid = 0; vid < 8; ++vid) {
                int row = detail::sm70_atom_c_dst_n(lane, vid);
                int col = 16 * sm70_b_j_groups * warp_col + 16 * j +
                          8 * out_half +
                          detail::sm70_atom_c_dst_m(lane, vid);
                atomicAdd(&sh_red_f32[sh_red_index(row, col)],
                          sm70_frag_c[i][j].accum[out_half][0][vid]);
              }
            }
          } else {
  #pragma unroll
            for (int out_half = 0; out_half < 2; ++out_half) {
  #pragma unroll
              for (int m_half = 0; m_half < 2; ++m_half) {
  #pragma unroll
                for (int vid = 0; vid < 8; ++vid) {
                  int row = 16 * i + 8 * m_half +
                            detail::sm70_atom_c_dst_m(lane, vid);
                  int col = 16 * sm70_b_j_groups * warp_col + 16 * j +
                            8 * out_half +
                            detail::sm70_atom_c_dst_n(lane, vid);
                  atomicAdd(&sh_red_f32[sh_red_index(row, col)],
                            sm70_frag_c[i][j].accum[out_half][m_half][vid]);
                }
              }
            }
          }
        }
      }
    }
    __syncthreads();
  };

  auto logical_tile_row_valid = [&](int row) { return row < block_num_valid_tokens; };

  auto logical_tile_global_row = [&](int row) -> int64_t {
    return sh_block_sorted_ids[row];
  };

  auto logical_tile_add = [&](int row, int col, float val) {
    sh_red_f32[sh_red_index(row, col)] += val;
  };

  auto logical_param_col = [&](int col) {
    int block_col = col & 0x1f;
    int permuted = 8 * ((block_col & 0x7) >> 1) +
                   2 * ((block_col >> 3) & 0x3) + (block_col & 0x1);
    return (col & ~0x1f) + permuted;
  };

  auto logical_tile_scale_num = [&](int col) -> c_scalar_t {
    if constexpr (group_blocks == -1) {
      return reinterpret_cast<c_scalar_t*>(sh_s)[logical_param_col(col)];
    } else {
      return reinterpret_cast<c_scalar_t*>(sh_s)[col];
    }
  };

  auto logical_tile_bias_num = [&](int col) -> c_scalar_t {
    return reinterpret_cast<c_scalar_t*>(sh_bias)[logical_param_col(col)];
  };

  auto apply_column_scales_to_logical_tile = [&]() {
    if constexpr (group_blocks == -1 &&
                  b_type.size_bits() == 8 &&
                  (has_zp && dequant_skip_flop || !has_zp)) {
      for (int linear = threadIdx.x; linear < sh_red_rows * sh_red_cols;
           linear += threads) {
        int row = linear / sh_red_cols;
        int col = linear % sh_red_cols;
        if (logical_tile_row_valid(row)) {
          sh_red_f32[sh_red_index(row, col)] *=
              Cdtype::num2float(logical_tile_scale_num(col));
        }
      }
      __syncthreads();
    }
  };

  auto global_reduce_fp16 = [&](bool first = false, bool last = false) {
    constexpr int vecs_per_row = sh_red_cols / 8;
    int c_gl_stride = prob_n / 8;
    int c_gl_base = slice_col * vecs_per_row;

    for (int linear = threadIdx.x; linear < sh_red_rows * vecs_per_row;
         linear += threads) {
      int row = linear / vecs_per_row;
      int col_vec = linear % vecs_per_row;
      int col_base = 8 * col_vec;
      if (!logical_tile_row_valid(row)) {
        continue;
      }

      int64_t c_idx = logical_tile_global_row(row) * c_gl_stride +
                      c_gl_base + col_vec;
      if (!first) {
        int4 existing = C[c_idx];
        c_scalar_t* c_red_f16 = reinterpret_cast<c_scalar_t*>(&existing);
      #pragma unroll
        for (int k = 0; k < 8; ++k) {
          logical_tile_add(row, col_base + k, Cdtype::num2float(c_red_f16[k]));
        }
      }

      if (!last) {
        alignas(16) c_scalar_t c_f16[8];
      #pragma unroll
        for (int k = 0; k < 8; ++k) {
          c_f16[k] = Cdtype::float2num(sh_red_load(row, col_base + k));
        }
        C[c_idx] = *reinterpret_cast<int4*>(c_f16);
      }
    }
    __syncthreads();
  };

  auto global_reduce_fp32 = [&](bool first = false, bool last = false) {
    constexpr int tb_m = thread_m_blocks * 16;
    constexpr int tb_n = thread_n_blocks * 16;
    constexpr int c_size = tb_m * tb_n * sizeof(float) / 16;
    constexpr int vecs_per_row = sh_red_cols / 4;
    int c_cur_offset = locks_off * c_size;

    for (int linear = threadIdx.x; linear < sh_red_rows * vecs_per_row;
         linear += threads) {
      int row = linear / vecs_per_row;
      int col_base = 4 * (linear % vecs_per_row);
      if (!logical_tile_row_valid(row)) {
        continue;
      }

      if (!first) {
        int4 existing = C_tmp[c_cur_offset + linear];
        float* c_red_f32 = reinterpret_cast<float*>(&existing);
      #pragma unroll
        for (int k = 0; k < 4; ++k) {
          logical_tile_add(row, col_base + k, c_red_f32[k]);
        }
      }

      if (!last) {
        alignas(16) float c_f32[4];
      #pragma unroll
        for (int k = 0; k < 4; ++k) {
          c_f32[k] = sh_red_load(row, col_base + k);
        }
        C_tmp[c_cur_offset + linear] = *reinterpret_cast<int4*>(c_f32);
      }
    }
    __syncthreads();
  };

  auto write_result = [&](bool last) {
    constexpr int vecs_per_row = sh_red_cols / 8;
    int c_gl_stride = prob_n / 8;
    int c_gl_base = slice_col * vecs_per_row;
    float global_scale_f = 1.0f;
    if constexpr (b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn) {
      global_scale_f =
          Cdtype::num2float(reinterpret_cast<c_scalar_t*>(&global_scale)[0]);
    }

    for (int linear = threadIdx.x; linear < sh_red_rows * vecs_per_row;
         linear += threads) {
      int row = linear / vecs_per_row;
      int col_vec = linear % vecs_per_row;
      int col_base = 8 * col_vec;
      if (!logical_tile_row_valid(row)) {
        continue;
      }

      alignas(16) c_scalar_t packed[8];
      float vals[8];
      float topk_weight_f = 1.0f;
      if (mul_topk_weights) {
        topk_weight_f =
            Cdtype::num2float(reinterpret_cast<c_scalar_t*>(
                                  &sh_block_topk_weights[row])[0]);
      }

    #pragma unroll
      for (int k = 0; k < 8; ++k) {
        vals[k] = sh_red_load(row, col_base + k);
      }

      if constexpr (group_blocks == -1 &&
                    b_type.size_bits() == 4 &&
                    (has_zp && dequant_skip_flop || !has_zp)) {
      #pragma unroll
        for (int k = 0; k < 8; ++k) {
          vals[k] *= Cdtype::num2float(logical_tile_scale_num(col_base + k));
        }
      }

      if constexpr (b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn) {
        if (!mul_topk_weights) {
        #pragma unroll
          for (int k = 0; k < 8; ++k) {
            vals[k] *= global_scale_f;
          }
        }
      }

      if (has_bias && last) {
      #pragma unroll
        for (int k = 0; k < 8; ++k) {
          vals[k] += Cdtype::num2float(logical_tile_bias_num(col_base + k));
        }
      }

      if (mul_topk_weights) {
      #pragma unroll
        for (int k = 0; k < 8; ++k) {
          vals[k] *= topk_weight_f;
        }
      }

    #pragma unroll
      for (int k = 0; k < 8; ++k) {
        packed[k] = Cdtype::float2num(vals[k]);
      }

      int64_t c_idx = logical_tile_global_row(row) * c_gl_stride +
                      c_gl_base + col_vec;
      if (use_atomic_add && slice_count > 1) {
        c_scalar_t2* c_half2 = reinterpret_cast<c_scalar_t2*>(&C[c_idx]);
        c_scalar_t2* packed_half2 = reinterpret_cast<c_scalar_t2*>(packed);
      #pragma unroll
        for (int a = 0; a < 4; ++a) {
          atomicAdd(&c_half2[a], packed_half2[a]);
        }
      } else {
        C[c_idx] = *reinterpret_cast<int4*>(packed);
      }
    }
    __syncthreads();
  };

  // Start global fetch and register load pipelines.
  auto start_pipes = [&]() {

  #pragma unroll
    for (int i = 0; i < stages - 1; i++) {
      if constexpr (has_zp && !is_zp_float && group_blocks == -1) {
        if (i == 0) {
          fetch_col_zp_to_shared();
          if constexpr (!dequant_skip_flop) {
            fetch_col_scale_to_shared();
          }
        }
      }
      fetch_to_shared(i, i, i < slice_iters);
    }

    zero_accums();
    wait_for_stage();
    fetch_to_registers(0, 0);
    fetch_scales_to_registers(0, 0);
    fetch_zp_to_registers(0, 0);
    a_gl_rd_col += a_gl_rd_delta_o * (stages - 1);
  };
  if (slice_iters) {
    start_pipes();
  }

  // Main loop.
  while (slice_iters) {
    // We unroll over both the global fetch and the register load pipeline to
    // ensure all shared memory accesses are static. Note that both pipelines
    // have even length meaning that the next iteration will always start at
    // index 0.

  #pragma unroll
    for (int pipe = 0; pipe < stages;) {
  #pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        fetch_scales_to_registers(k + 1, pipe);
        fetch_zp_to_registers(k + 1, pipe);
        if (k == b_sh_wr_iters - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe,
                          slice_iters >= stages);
          pipe++;
          wait_for_stage();
        }

        matmul(k, pipe - (k >= b_sh_wr_iters - 2 ? 1 : 0));
      }
      slice_iters--;
      if (slice_iters == 0) {
        break;
      }
    }

    a_gl_rd_col += a_gl_rd_delta_o * stages;

    // Process results and, if necessary, proceed to the next column slice.
    // While this pattern may not be the most readable, other ways of writing
    // the loop seemed to noticeably worse performance after compilation.
    if (slice_iters == 0) {
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      if constexpr (group_blocks == -1 &&
                    (has_zp && dequant_skip_flop || !has_zp)) {
        if (b_type.size_bits() == 8 || (last || use_atomic_add)) {
          if (s_sh_wr_pred) {
            cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
          }
          cp_async_fence();
        }
      }

      if (has_bias && last) {
        __syncthreads();
        cp_async4_pred(&sh_bias[bias_sh_wr], &b_bias_ptr[bias_gl_rd],
                       threadIdx.x < 16 * thread_n_blocks / 8);
        cp_async_fence();
      }

      dump_sm70_to_smem();

      if constexpr (group_blocks == -1 &&
                    (has_zp && dequant_skip_flop || !has_zp)) {
        if (b_type.size_bits() == 8 || (last || use_atomic_add)) {
          cp_async_wait<0>();
          __syncthreads();
        }
      }

      apply_column_scales_to_logical_tile();

      if (slice_count > 1 && !use_atomic_add) {
        // only globally reduce if there is more than one block in a slice
        barrier_acquire(&locks[locks_off], slice_idx);
        if (use_fp32_reduce) {
          global_reduce_fp32(slice_idx == 0, last);
        } else {
          global_reduce_fp16(slice_idx == 0, last);
        }
        barrier_release(&locks[locks_off], last);
      }

      if (has_bias && last) {
        cp_async_wait<0>();
        __syncthreads();
      }

      if (use_atomic_add && slice_count > 1 && slice_idx != 0)
        wait_negative_and_add(&locks[locks_off]);
      if (last || use_atomic_add)
        // only the last block in a slice actually writes the result
        write_result(last);
      slice_row = 0;
      if (!in_part2) {
        slice_col_par += gridDim.x;
      } else {
        slice_col_par++;
        slice_col++;
      }
      is_first_matmul_in_slice = true;
      init_slice();

      if (slice_iters) {
        a_gl_rd_col =
            a_gl_rd_delta_o * slice_row + threadIdx.x % a_gl_rd_delta_o;
        b_gl_rd = B_expert_off + b_gl_stride * (threadIdx.x / b_sh_stride) +
                  (threadIdx.x % b_sh_stride);
        b_gl_rd += b_sh_stride * slice_col + b_gl_rd_delta_o * slice_row;

        bias_gl_rd = (thread_n_blocks * 16 / 8) * slice_col + threadIdx.x;
        // Update slice k/n for scales loading
        if constexpr (group_blocks == -1) {
          s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
          zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
        } else if constexpr (group_blocks >= thread_k_blocks) {
          s_gl_rd =
              s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
              s_sh_stride * slice_col + threadIdx.x;
          zp_gl_rd =
              zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
              zp_sh_stride * slice_col + threadIdx.x;
        } else {
          s_gl_rd =
              s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                             threadIdx.x / s_sh_stride) +
              s_sh_stride * slice_col + threadIdx.x % s_sh_stride;
          zp_gl_rd =
              zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                              threadIdx.x / zp_sh_stride) +
              zp_sh_stride * slice_col + threadIdx.x % zp_sh_stride;
        }
        start_pipes();
      }
    }
  }
}

}  // namespace MARLIN_NAMESPACE_NAME

#endif

#include "marlin_dtypes.cuh"

#include <assert.h>

namespace MARLIN_NAMESPACE_NAME {

namespace detail {

constexpr uint32_t kFullWarpMask = 0xffffffffu;

// A single Volta m8n8k4 atom uses the low 2 lane bits for the row/column
// within a 4x4 quad and lane bit 4 to select the upper 4 rows / columns.
__device__ __forceinline__ int sm70_atom_rowcol(int lane) {
  return (lane & 0x3) + 4 * ((lane >> 4) & 0x1);
}

// Canonical inverse mapping from an 8x8 logical output coordinate back to the
// raw SM70 atom lane/register that owns that value. Lane bits 2 and 3 are
// duplicate modes, so we intentionally pick the canonical variant with those
// bits cleared.
__device__ __forceinline__ int sm70_atom_c_src_lane(int m_local, int n) {
  return (m_local & 0x1) + 2 * ((n >> 1) & 0x1) +
         16 * ((m_local >> 2) & 0x1);
}

__device__ __forceinline__ int sm70_atom_c_src_vid(int m_local, int n) {
  return 2 * ((m_local >> 1) & 0x1) + (n & 0x1) + 4 * ((n >> 2) & 0x1);
}

template <typename FragC>
__device__ __forceinline__ void scatter_sm70_atoms_to_sm80_fragment_half(
    const float* accum, FragC& frag_c, int half_idx) {
  float* dst = reinterpret_cast<float*>(&frag_c);
  int lane = threadIdx.x & 31;
  int start_i = half_idx * 2;

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int actual_i = start_i + i;
    int m = (lane >> 2) + 8 * (actual_i >> 1);
    int n = 2 * (lane & 0x3) + (actual_i & 0x1);
    int m_local = m & 0x7;
    int sm70_lane = sm70_atom_c_src_lane(m_local, n);
    int want_vid = sm70_atom_c_src_vid(m_local, n);
    float value = 0.0f;
#pragma unroll
    for (int vid = 0; vid < 8; ++vid) {
      float shuffled = __shfl_sync(kFullWarpMask, accum[vid], sm70_lane);
      if (vid == want_vid) {
        value = shuffled;
      }
    }
    dst[actual_i] += value;
  }
}

__device__ __forceinline__ void run_sm70_atom_packed(uint32_t a0, uint32_t a1,
                                            uint32_t b0, uint32_t b1,
                                            float* accum) {
  float d0 = accum[0];
  float d1 = accum[1];
  float d2 = accum[2];
  float d3 = accum[3];
  float d4 = accum[4];
  float d5 = accum[5];
  float d6 = accum[6];
  float d7 = accum[7];

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32"
               "{%0, %1, %2, %3, %4, %5, %6, %7},"
               "{%8, %9},"
               "{%10, %11},"
               "{%12, %13, %14, %15, %16, %17, %18, %19};\n"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5),
                 "=f"(d6), "=f"(d7)
               : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "f"(accum[0]),
                 "f"(accum[1]), "f"(accum[2]), "f"(accum[3]), "f"(accum[4]),
                 "f"(accum[5]), "f"(accum[6]), "f"(accum[7]));
#else
  assert(false && "SM70 inline PTX mma requires __CUDA_ARCH__ >= 700");
#endif

  accum[0] = d0;
  accum[1] = d1;
  accum[2] = d2;
  accum[3] = d3;
  accum[4] = d4;
  accum[5] = d5;
  accum[6] = d6;
  accum[7] = d7;
}

template <bool kTrans, typename FragC>
__device__ __forceinline__ void emulate_turing_m16n8k8(uint32_t x0, uint32_t x1,
                                              uint32_t y, FragC& frag_c) {
  int lane = threadIdx.x & 31;
  int atom_rowcol = sm70_atom_rowcol(lane);
  
  // 只分配 8 个 float，重用同一块内存，降低寄存器压力
  float accum[8];

#pragma unroll
  for (int half_idx = 0; half_idx < 2; ++half_idx) {
    // 每次开始前初始化 accum
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      accum[i] = 0.0f;
    }

    uint32_t x = (half_idx == 0) ? x0 : x1;

#pragma unroll
    for (int k_slice = 0; k_slice < 2; ++k_slice) {
      int lane_in_quad_0 = 2 * k_slice;
      int lane_in_quad_1 = lane_in_quad_0 + 1;
      int a_source_lane_0 = 4 * (atom_rowcol & 0x7) + lane_in_quad_0;
      int a_source_lane_1 = 4 * (atom_rowcol & 0x7) + lane_in_quad_1;
      int b_source_lane_0 = 4 * atom_rowcol + lane_in_quad_0;
      int b_source_lane_1 = 4 * atom_rowcol + lane_in_quad_1;

      uint32_t a0;
      uint32_t a1;
      uint32_t b0;
      uint32_t b1;

      if constexpr (!kTrans) {
        a0 = __shfl_sync(kFullWarpMask, x, a_source_lane_0);
        a1 = __shfl_sync(kFullWarpMask, x, a_source_lane_1);
        b0 = __shfl_sync(kFullWarpMask, y, b_source_lane_0);
        b1 = __shfl_sync(kFullWarpMask, y, b_source_lane_1);
      } else {
        b0 = __shfl_sync(kFullWarpMask, y, b_source_lane_0);
        b1 = __shfl_sync(kFullWarpMask, y, b_source_lane_1);
        a0 = __shfl_sync(kFullWarpMask, x, a_source_lane_0);
        a1 = __shfl_sync(kFullWarpMask, x, a_source_lane_1);
      }

      run_sm70_atom_packed(a0, a1, b0, b1, accum);
    }

    // 计算完一半后，立即通过 Scatter 将结果写回 frag_c 的对应位置
    scatter_sm70_atoms_to_sm80_fragment_half(accum, frag_c, half_idx);
  }
}

}  // namespace detail

template <vllm::ScalarTypeId type_id, bool use_fp16_accum, int k_size = 16>
__device__ __noinline__ void mma(
    const typename MarlinScalarType<type_id>::FragA& a_frag,
    const typename MarlinScalarType<type_id>::FragB& frag_b,
    typename MarlinScalarType<type_id>::FragC& frag_c, int idx = 0) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;

  static_cast<void>(idx);
  static_assert(k_size == 16, "SM70 inline PTX mma only supports k_size=16.");
  static_assert(std::is_same<scalar_t, half>::value,
                "SM70 inline PTX mma currently supports fp16 inputs only.");
  static_assert(!use_fp16_accum,
                "SM70 inline PTX mma currently supports fp32 accumulation only.");

  detail::emulate_turing_m16n8k8<false>(a[0], a[1], b[0], frag_c);
  detail::emulate_turing_m16n8k8<false>(a[2], a[3], b[1], frag_c);
}

template <vllm::ScalarTypeId type_id, bool use_fp16_accum, int k_size = 16>
__device__ __noinline__ void mma_trans(
    const typename MarlinScalarType<type_id>::FragA& a_frag,
    const typename MarlinScalarType<type_id>::FragB& frag_b,
    const typename MarlinScalarType<type_id>::FragB& frag_b2,
    typename MarlinScalarType<type_id>::FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  const uint32_t* b2 = reinterpret_cast<const uint32_t*>(&frag_b2);
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;

  static_assert(k_size == 16, "SM70 inline PTX mma only supports k_size=16.");
  static_assert(std::is_same<scalar_t, half>::value,
                "SM70 build only supports fp16 mma kernels.");
  static_assert(!use_fp16_accum,
                "SM70 inline PTX mma currently supports fp32 accumulation only.");

  detail::emulate_turing_m16n8k8<true>(b[0], b2[0], a[0], frag_c);
  detail::emulate_turing_m16n8k8<true>(b[1], b2[1], a[1], frag_c);
}

}  // namespace MARLIN_NAMESPACE_NAME

#include "marlin_dtypes.cuh"

#include <assert.h>

namespace MARLIN_NAMESPACE_NAME {

template <int count, vllm::ScalarTypeId type_id>
__device__ inline void ldsm(typename MarlinScalarType<type_id>::FragA& frag_a,
                            const int4* smem_base, int smem_idx) {
  static_assert(type_id == vllm::kFloat16.id(), "SM70 warp loader only supports fp16");
  static_assert(count == 2 || count == 4, "invalid count");
  int lane = threadIdx.x & 31;
  uint32_t* dst = reinterpret_cast<uint32_t*>(&frag_a);
  constexpr uint32_t kMask = 0xffffffffu;
  int lane_group = lane >> 2;
  int lane_col = lane & 0x3;

#pragma unroll
  for (int reg = 0; reg < count; ++reg) {
    int src_lane = 8 * reg + lane_group;
    int src_idx = __shfl_sync(kMask, smem_idx, src_lane);
    const half* src = reinterpret_cast<const half*>(&smem_base[src_idx]);
    half lo = src[2 * lane_col + 0];
    half hi = src[2 * lane_col + 1];
    dst[reg] = static_cast<uint32_t>(__half_as_ushort(lo)) |
               (static_cast<uint32_t>(__half_as_ushort(hi)) << 16);
  }

#pragma unroll
  for (int reg = count; reg < 4; ++reg) {
    dst[reg] = 0u;
  }
}

namespace detail {

constexpr uint32_t kFullWarpMask = 0xffffffffu;

template <typename FragT>
__device__ inline half shfl_frag_half(const FragT& frag, int src_lane,
                                      int src_half_idx) {
  const uint32_t* words = reinterpret_cast<const uint32_t*>(&frag);
  uint32_t packed = words[src_half_idx / 2];
  packed = __shfl_sync(kFullWarpMask, packed, src_lane);
  uint16_t bits = static_cast<uint16_t>(
      (src_half_idx & 1) ? (packed >> 16) : (packed & 0xffffu));
  return __ushort_as_half(bits);
}

template <typename FragA>
__device__ inline half fetch_sm80_a_value(const FragA& frag_a, int m, int k) {
  int src_lane = 4 * (m & 0x7) + ((k & 0x7) >> 1);
  int src_half_idx = 4 * (k >> 3) + 2 * (m >> 3) + (k & 0x1);
  return shfl_frag_half(frag_a, src_lane, src_half_idx);
}

template <typename FragBLike>
__device__ inline half fetch_sm80_b_value(const FragBLike& frag_b, int n, int k) {
  int src_lane = 4 * n + ((k & 0x7) >> 1);
  int src_half_idx = 2 * (k >> 3) + (k & 0x1);
  return shfl_frag_half(frag_b, src_lane, src_half_idx);
}

template <typename FragB>
__device__ inline half fetch_sm80_trans_a_value(const FragB& frag_b,
                                                const FragB& frag_b2, int m,
                                                int k) {
  int src_lane = 4 * (m & 0x7) + ((k & 0x7) >> 1);
  int synth_half_idx = 4 * (k >> 3) + 2 * (m >> 3) + (k & 0x1);

  if (synth_half_idx < 2) {
    return shfl_frag_half(frag_b, src_lane, synth_half_idx);
  }
  if (synth_half_idx < 4) {
    return shfl_frag_half(frag_b2, src_lane, synth_half_idx - 2);
  }
  if (synth_half_idx < 6) {
    return shfl_frag_half(frag_b, src_lane, synth_half_idx - 2);
  }
  return shfl_frag_half(frag_b2, src_lane, synth_half_idx - 4);
}

// A single Volta m8n8k4 atom uses the low 2 lane bits for the row/column
// within a 4x4 quad and lane bit 4 to select the upper 4 rows / columns.
__device__ inline int sm70_atom_rowcol(int lane) {
  return (lane & 0x3) + 4 * ((lane >> 4) & 0x1);
}

// Canonical inverse mapping from an 8x8 logical output coordinate back to the
// raw SM70 atom lane/register that owns that value. Lane bits 2 and 3 are
// duplicate modes, so we intentionally pick the canonical variant with those
// bits cleared.
__device__ inline int sm70_atom_c_src_lane(int m_local, int n) {
  return (m_local & 0x1) + 2 * ((n >> 1) & 0x1) +
         16 * ((m_local >> 2) & 0x1);
}

__device__ inline int sm70_atom_c_src_vid(int m_local, int n) {
  return 2 * ((m_local >> 1) & 0x1) + (n & 0x1) + 4 * ((n >> 2) & 0x1);
}

template <typename FragC>
__device__ inline void scatter_sm70_atoms_to_sm80_fragment(
    const float* accum_lo, const float* accum_hi, FragC& frag_c) {
  float* dst = reinterpret_cast<float*>(&frag_c);
  int lane = threadIdx.x & 31;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int m = (lane >> 2) + 8 * (i >> 1);
    int n = 2 * (lane & 0x3) + (i & 0x1);
    int half = m >> 3;
    int m_local = m & 0x7;
    int sm70_lane = sm70_atom_c_src_lane(m_local, n);
    int want_vid = sm70_atom_c_src_vid(m_local, n);
    float value = 0.0f;
#pragma unroll
    for (int vid = 0; vid < 8; ++vid) {
      float reg = half ? accum_hi[vid] : accum_lo[vid];
      float shuffled = __shfl_sync(kFullWarpMask, reg, sm70_lane);
      if (vid == want_vid) {
        value = shuffled;
      }
    }
    dst[i] += value;
  }
}

__device__ inline void run_sm70_atom(const half* a_vals, const half* b_vals,
                                     float* accum) {
  auto pack = [](half x, half y) {
    return static_cast<uint32_t>(__half_as_ushort(x)) |
           (static_cast<uint32_t>(__half_as_ushort(y)) << 16);
  };

  uint32_t a0 = pack(a_vals[0], a_vals[1]);
  uint32_t a1 = pack(a_vals[2], a_vals[3]);
  uint32_t b0 = pack(b_vals[0], b_vals[1]);
  uint32_t b1 = pack(b_vals[2], b_vals[3]);

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

template <typename FragA, typename FragB, typename FragC>
__device__ inline void mma_fp32_sm70(const FragA& a_frag, const FragB& frag_b,
                                     FragC& frag_c) {
  int lane = threadIdx.x & 31;
  int atom_rowcol = sm70_atom_rowcol(lane);
  float accum_lo[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float accum_hi[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
  for (int k_slice = 0; k_slice < 4; ++k_slice) {
    half b_vals[4];
    half a_vals_lo[4];
    half a_vals_hi[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int k = static_cast<int>(4 * k_slice + i);
      b_vals[i] = fetch_sm80_b_value(frag_b, atom_rowcol, k);
      a_vals_lo[i] = fetch_sm80_a_value(a_frag, atom_rowcol, k);
      a_vals_hi[i] = fetch_sm80_a_value(a_frag, atom_rowcol + 8, k);
    }

    run_sm70_atom(a_vals_lo, b_vals, accum_lo);
    run_sm70_atom(a_vals_hi, b_vals, accum_hi);
  }

  scatter_sm70_atoms_to_sm80_fragment(accum_lo, accum_hi, frag_c);
}

template <typename FragA, typename FragB, typename FragC>
__device__ inline void mma_fp32_sm70_trans(const FragA& a_frag,
                                           const FragB& frag_b,
                                           const FragB& frag_b2,
                                           FragC& frag_c) {
  int lane = threadIdx.x & 31;
  int atom_rowcol = sm70_atom_rowcol(lane);
  float accum_lo[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float accum_hi[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
  for (int k_slice = 0; k_slice < 4; ++k_slice) {
    half b_vals[4];
    half a_vals_lo[4];
    half a_vals_hi[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int k = 4 * k_slice + i;
      b_vals[i] = fetch_sm80_b_value(a_frag, atom_rowcol, k);
      a_vals_lo[i] = fetch_sm80_trans_a_value(frag_b, frag_b2, atom_rowcol, k);
      a_vals_hi[i] =
          fetch_sm80_trans_a_value(frag_b, frag_b2, atom_rowcol + 8, k);
    }

    run_sm70_atom(a_vals_lo, b_vals, accum_lo);
    run_sm70_atom(a_vals_hi, b_vals, accum_hi);
  }

  scatter_sm70_atoms_to_sm80_fragment(accum_lo, accum_hi, frag_c);
}

}  // namespace detail

template <class...>
inline constexpr bool always_false_v = false;

template <vllm::ScalarTypeId type_id, bool use_fp16_accum, int k_size = 16>
__device__ inline void mma(
    const typename MarlinScalarType<type_id>::FragA& a_frag,
    const typename MarlinScalarType<type_id>::FragB& frag_b,
    typename MarlinScalarType<type_id>::FragC& frag_c, int idx = 0) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  static_cast<void>(idx);
  if constexpr (k_size == 16) {
    static_assert(std::is_same<scalar_t, half>::value,
                  "SM70 inline PTX mma currently supports fp16 inputs only.");
    static_assert(!use_fp16_accum,
                  "SM70 inline PTX mma currently supports fp32 accumulation only.");
    detail::mma_fp32_sm70(a_frag, frag_b, frag_c);
  } else {
    static_assert(always_false_v<scalar_t>,
                  "SM70 inline PTX mma only supports k_size=16.");
  }
}

template <vllm::ScalarTypeId type_id, bool use_fp16_accum, int k_size = 16>
__device__ inline void mma_trans(
    const typename MarlinScalarType<type_id>::FragA& a_frag,
    const typename MarlinScalarType<type_id>::FragB& frag_b,
    const typename MarlinScalarType<type_id>::FragB& frag_b2,
    typename MarlinScalarType<type_id>::FragC& frag_c) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  static_assert(std::is_same<scalar_t, half>::value,
                "SM70 build only supports fp16 mma kernels.");
  static_assert(!use_fp16_accum,
                "SM70 inline PTX mma currently supports fp32 accumulation only.");
  static_assert(k_size == 16, "SM70 inline PTX mma only supports k_size=16.");
  detail::mma_fp32_sm70_trans(a_frag, frag_b, frag_b2, frag_c);
}

}  // namespace MARLIN_NAMESPACE_NAME

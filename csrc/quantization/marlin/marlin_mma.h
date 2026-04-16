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

template <int m_halves>
struct Sm70DirectAFragment {
  uint32_t words[2][m_halves][2][2];
};

template <int vecs>
struct Sm70DirectBQuant {
  uint32_t words[vecs][4];
};

__device__ __forceinline__ uint32_t load_sm70_shared_u32(const void* smem_ptr,
                                                         int byte_offset) {
  uint32_t smem =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr)) + byte_offset;
  uint32_t value;
  asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(value) : "r"(smem));
  return value;
}

template <int m_halves>
__device__ __forceinline__ void load_sm70_direct_a(
    Sm70DirectAFragment<m_halves>& frag_a, const void* smem_ptr,
    const int byte_offsets[2][m_halves][2][2]) {
#pragma unroll
  for (int k_block = 0; k_block < 2; ++k_block) {
#pragma unroll
    for (int m_half = 0; m_half < m_halves; ++m_half) {
#pragma unroll
      for (int k_slice = 0; k_slice < 2; ++k_slice) {
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          frag_a.words[k_block][m_half][k_slice][pair] =
              load_sm70_shared_u32(
                  smem_ptr, byte_offsets[k_block][m_half][k_slice][pair]);
        }
      }
    }
  }
}

template <int vecs>
__device__ __forceinline__ void load_sm70_direct_b(
    Sm70DirectBQuant<vecs>& frag_b, const void* smem_ptr,
    const int byte_offsets[vecs][4]) {
#pragma unroll
  for (int vec = 0; vec < vecs; ++vec) {
#pragma unroll
    for (int row_group = 0; row_group < 4; ++row_group) {
      frag_b.words[vec][row_group] =
          load_sm70_shared_u32(smem_ptr, byte_offsets[vec][row_group]);
    }
  }
}

__device__ __forceinline__ void zero_sm70_accum(float* accum) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    accum[i] = 0.0f;
  }
}

template <typename FragB, typename FragC>
__device__ __noinline__ void mma_sm70_direct_a_native(
    const Sm70DirectAFragment<2>& a_frag, const FragB (&frag_b_q)[4],
    FragC& frag_c) {
  uint32_t b_words[4][2];
#pragma unroll
  for (int row_group = 0; row_group < 4; ++row_group) {
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b_q[row_group]);
    b_words[row_group][0] = b[0];
    b_words[row_group][1] = b[1];
  }

#pragma unroll
  for (int k_block = 0; k_block < 2; ++k_block) {
#pragma unroll
    for (int m_half = 0; m_half < 2; ++m_half) {
      float accum[8];
      zero_sm70_accum(accum);

      run_sm70_atom_packed(a_frag.words[k_block][m_half][0][0],
                           a_frag.words[k_block][m_half][0][1],
                           b_words[0][k_block], b_words[1][k_block], accum);
      run_sm70_atom_packed(a_frag.words[k_block][m_half][1][0],
                           a_frag.words[k_block][m_half][1][1],
                           b_words[2][k_block], b_words[3][k_block], accum);

      scatter_sm70_atoms_to_sm80_fragment_half(accum, frag_c, m_half);
    }
  }
}

template <typename FragB, typename FragC>
__device__ __noinline__ void mma_sm70_direct_a_m8_half(
    const Sm70DirectAFragment<1>& a_frag, const FragB (&frag_b_q)[4],
    FragC& frag_c, int out_half) {
  uint32_t b_words[4][2];
#pragma unroll
  for (int row_group = 0; row_group < 4; ++row_group) {
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b_q[row_group]);
    b_words[row_group][0] = b[0];
    b_words[row_group][1] = b[1];
  }

#pragma unroll
  for (int k_block = 0; k_block < 2; ++k_block) {
    float accum[8];
    zero_sm70_accum(accum);

    run_sm70_atom_packed(b_words[0][k_block], b_words[1][k_block],
                         a_frag.words[k_block][0][0][0],
                         a_frag.words[k_block][0][0][1], accum);
    run_sm70_atom_packed(b_words[2][k_block], b_words[3][k_block],
                         a_frag.words[k_block][0][1][0],
                         a_frag.words[k_block][0][1][1], accum);

    scatter_sm70_atoms_to_sm80_fragment_half(accum, frag_c, out_half);
  }
}

}  // namespace detail

}  // namespace MARLIN_NAMESPACE_NAME

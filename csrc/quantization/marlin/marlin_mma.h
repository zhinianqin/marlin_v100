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

__device__ __forceinline__ bool sm70_atom_is_canonical_lane(int lane) {
  return (lane & 0xc) == 0;
}

// Canonical inverse mapping from the raw SM70 atom lane/register ownership back
// to the logical 8x8 output coordinates. This lets us dump the native
// accumulator directly to shared memory without warp shuffles.
__device__ __forceinline__ int sm70_atom_c_dst_m(int lane, int vid) {
  return (lane & 0x1) + 2 * ((vid >> 1) & 0x1) + 4 * ((lane >> 4) & 0x1);
}

__device__ __forceinline__ int sm70_atom_c_dst_n(int lane, int vid) {
  return (vid & 0x1) + 2 * ((lane >> 1) & 0x1) + 4 * ((vid >> 2) & 0x1);
}

template <int stride>
__device__ __forceinline__ int sm70_transform_a_index(int idx) {
  int row = idx / stride;
  return stride * row + ((idx % stride) ^ (row % 8));
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

struct Sm70DirectAFragment {
  uint32_t words[2][2][2];
};

struct Sm70Accumulator {
  float accum[8];
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

template <int m_atoms, int k_steps_per_warp, int a_sh_stride>
__device__ __forceinline__ void load_sm70_direct_a_runtime(
    Sm70DirectAFragment (&frag_a)[m_atoms],
    const void* smem_ptr, int warp_k_group, int sm70_lane_row,
    int tile_k_step) {
#pragma unroll
  for (int m_atom = 0; m_atom < m_atoms; ++m_atom) {
#pragma unroll
    for (int k_block = 0; k_block < 2; ++k_block) {
      int row = 8 * m_atom + sm70_lane_row;
      int local_k_base =
          16 * (warp_k_group * k_steps_per_warp + tile_k_step) + 8 * k_block;
#pragma unroll
      for (int k_slice = 0; k_slice < 2; ++k_slice) {
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          int col = local_k_base + 4 * k_slice + 2 * pair;
          int chunk =
              sm70_transform_a_index<a_sh_stride>(row * a_sh_stride + col / 8);
          int byte_offset =
              chunk * sizeof(int4) + ((col & 0x7) / 2) * sizeof(uint32_t);
          frag_a[m_atom].words[k_block][k_slice][pair] =
              load_sm70_shared_u32(smem_ptr, byte_offset);
        }
      }
    }
  }
}

template <int vecs, int k_steps_per_warp, int b_sh_stride, int warp_j_groups>
__device__ __forceinline__ void load_sm70_direct_b_runtime(
    Sm70DirectBQuant<vecs> (&frag_b)[warp_j_groups], const void* smem_ptr,
    int warp_k_group, int j_group_base, int sm70_lane_row, int tile_k_step) {
#pragma unroll
  for (int j = 0; j < warp_j_groups; ++j) {
    int local_k_block = warp_k_group * k_steps_per_warp + tile_k_step;
    int local_n_block = j_group_base + j;
#pragma unroll
    for (int vec = 0; vec < vecs; ++vec) {
      int chunk = b_sh_stride * local_k_block +
                  local_n_block * (8 * vecs) + sm70_lane_row * vecs + vec;
#pragma unroll
      for (int row_group = 0; row_group < 4; ++row_group) {
        int byte_offset =
            chunk * sizeof(int4) + row_group * sizeof(uint32_t);
        frag_b[j].words[vec][row_group] =
            load_sm70_shared_u32(smem_ptr, byte_offset);
      }
    }
  }
}

__device__ __forceinline__ void zero_sm70_accum(float* accum) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    accum[i] = 0.0f;
  }
}

__device__ __forceinline__ void zero_sm70_accumulator(
    Sm70Accumulator& accum) {
  zero_sm70_accum(accum.accum);
}

template <typename FragB>
__device__ __forceinline__ void mma_sm70_direct_a_atom(
    const Sm70DirectAFragment& a_frag, const FragB (&frag_b_q)[4],
    float* raw_accum) {
  uint32_t b_words[4][2];
#pragma unroll
  for (int row_group = 0; row_group < 4; ++row_group) {
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b_q[row_group]);
    b_words[row_group][0] = b[0];
    b_words[row_group][1] = b[1];
  }

#pragma unroll
  for (int k_block = 0; k_block < 2; ++k_block) {
    run_sm70_atom_packed(b_words[0][k_block], b_words[1][k_block],
                         a_frag.words[k_block][0][0],
                         a_frag.words[k_block][0][1], raw_accum);
    run_sm70_atom_packed(b_words[2][k_block], b_words[3][k_block],
                         a_frag.words[k_block][1][0],
                         a_frag.words[k_block][1][1], raw_accum);
  }
}

}  // namespace detail

}  // namespace MARLIN_NAMESPACE_NAME

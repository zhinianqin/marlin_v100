#include "core/registration.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>
#include <torch/library.h>
#include <cstdint>
#include <type_traits>

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/layout/tensor_op_multiplicand_sm70.h"
#include "cute/algorithm/cooperative_gemm.hpp"
#include "cute/tensor.hpp"
#include "cute/swizzle.hpp"
#include "cute/swizzle_layout.hpp"

namespace {

enum ProbeAPath : int64_t {
  kAPathCuteShared = 0,
  kAPathDirectGlobal = 1,
  kAPathCutlassThreadblock = 2,
  kAPathSm70Atom = 3,
};

enum ProbeBPath : int64_t {
  kBPathCuteShared = 0,
};

bool is_aligned_16(const void* ptr) {
  return (reinterpret_cast<std::uintptr_t>(ptr) & 0xfu) == 0;
}

void validate_sm70_cutlass_matmul_probe_inputs(
    const at::Tensor& a, const at::Tensor& b, int64_t cta_m, int64_t cta_n,
    int64_t cta_k, int64_t warps, int64_t stages, int64_t a_path,
    int64_t b_path) {
  TORCH_CHECK(a.device().is_cuda(), "sm70_cutlass_matmul_probe: A must be CUDA");
  TORCH_CHECK(b.device().is_cuda(), "sm70_cutlass_matmul_probe: B must be CUDA");
  TORCH_CHECK(a.get_device() == b.get_device(),
              "sm70_cutlass_matmul_probe: A and B must be on the same CUDA device");
  TORCH_CHECK(a.scalar_type() == at::ScalarType::Half,
              "sm70_cutlass_matmul_probe: A must be float16");
  TORCH_CHECK(b.scalar_type() == at::ScalarType::Half,
              "sm70_cutlass_matmul_probe: B must be float16");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
              "sm70_cutlass_matmul_probe: A and B must be rank-2 tensors");
  TORCH_CHECK(a.size(1) == b.size(0),
              "sm70_cutlass_matmul_probe: shape mismatch, A is ", a.sizes(),
              " and B is ", b.sizes());
  TORCH_CHECK(a.is_contiguous(),
              "sm70_cutlass_matmul_probe: A must be contiguous row-major");
  TORCH_CHECK(b_path == kBPathCuteShared,
              "sm70_cutlass_matmul_probe: unknown B path id ", b_path);
  TORCH_CHECK(b.is_contiguous(),
              "sm70_cutlass_matmul_probe: B must be contiguous row-major");
  TORCH_CHECK(a.size(0) > 0 && a.size(1) > 0 && b.size(1) > 0,
              "sm70_cutlass_matmul_probe: M, N, and K must be positive");
  TORCH_CHECK(stages == 2,
              "sm70_cutlass_matmul_probe: only a 2-stage software pipeline is wired");
  TORCH_CHECK(a_path == kAPathCuteShared || a_path == kAPathDirectGlobal ||
                  a_path == kAPathCutlassThreadblock ||
                  a_path == kAPathSm70Atom,
              "sm70_cutlass_matmul_probe: unknown A path id ", a_path);
  TORCH_CHECK(a_path != kAPathDirectGlobal,
              "sm70_cutlass_matmul_probe: A direct-global path is TODO; only "
              "CUTLASS 3 CuTe shared-memory path id 0, extracted CUTLASS "
              "threadblock path id 2, and SM70 atom path id 3 are available");
  TORCH_CHECK(cta_m == 8 || cta_m == 16 || cta_m == 32 || cta_m == 48 ||
                  cta_m == 64 || cta_m == 128 || cta_m == 256 ||
                  cta_m == 512,
              "sm70_cutlass_matmul_probe: cta_m must be 8, 16, 32, 48, "
              "64, 128, 256, or 512");
  TORCH_CHECK(cta_n == 32 || cta_n == 64 || cta_n == 128 || cta_n == 256 ||
                  cta_n == 512,
              "sm70_cutlass_matmul_probe: cta_n must be 32, 64, 128, "
              "256, or 512");
  TORCH_CHECK(cta_k == 32 || cta_k == 64 || cta_k == 128,
              "sm70_cutlass_matmul_probe: cta_k must be 32, 64, or 128");
  TORCH_CHECK(warps == 4 || warps == 8,
              "sm70_cutlass_matmul_probe: warps must be 4 or 8");
  if (a_path == kAPathCuteShared || a_path == kAPathSm70Atom) {
    TORCH_CHECK(a.size(0) % cta_m == 0,
                "sm70_cutlass_matmul_probe: current CuTe/SM70 atom probe "
                "requires M divisible by cta_m");
  }
  TORCH_CHECK(b.size(1) % cta_n == 0,
              "sm70_cutlass_matmul_probe: current CuTe probe requires N divisible by cta_n");
  TORCH_CHECK(a.size(1) % cta_k == 0,
              "sm70_cutlass_matmul_probe: current CuTe probe requires K divisible by cta_k");
  if (a_path == kAPathCuteShared || a_path == kAPathSm70Atom) {
    TORCH_CHECK(is_aligned_16(a.data_ptr()),
                "sm70_cutlass_matmul_probe: CuTe vectorized A copy requires A "
                "to be 16-byte aligned");
    TORCH_CHECK(is_aligned_16(b.data_ptr()),
                "sm70_cutlass_matmul_probe: CuTe vectorized B copy requires B "
                "to be 16-byte aligned");
  }

  int major = 0;
  int minor = 0;
  int device = a.get_device();
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  TORCH_CHECK(major == 7 && minor == 0,
              "sm70_cutlass_matmul_probe: this probe only supports SM70 GPUs, got sm",
              major, minor);
}

template <int CTA_M, int CTA_N, int Warps>
struct Sm70AtomLayout {
  static_assert(CTA_M == 8 || CTA_M == 16 || CTA_M == 32 || CTA_M == 48 ||
                    CTA_M == 64,
                "SM70 CuTe native probe supports CTA_M in {8,16,32,48,64}.");
  static_assert(CTA_N == 64 || CTA_N == 128 || CTA_N == 256,
                "SM70 CuTe native probe supports CTA_N in {64,128,256}.");
  static_assert(Warps == 4 || Warps == 8,
                "SM70 CuTe native probe supports 4 or 8 warps.");

  static constexpr int kAtomM =
      CTA_M == 8 ? 1 : (CTA_M == 16 ? 2 : 4);
  static_assert(kAtomM * 8 <= CTA_M,
                "SM70 CuTe native atom layout must not pad logical M.");
  static constexpr int kAtomN = (Warps * 4) / kAtomM;
  using Type =
      cute::Layout<cute::Shape<cute::Int<kAtomM>, cute::Int<kAtomN>,
                               cute::Int<1>>>;
};

template <int CTA_M, int CTA_K>
CUTE_HOST_DEVICE auto make_smem_a_layout() {
  using namespace cute;
  constexpr int kAtomK = CTA_K < 64 ? CTA_K : 64;
  auto atom_layout =
      composition(Swizzle<3, 3, 3>{},
                  Layout<Shape<Int<8>, Int<kAtomK>>,
                         Stride<Int<kAtomK>, Int<1>>>{});
  return tile_to_shape(atom_layout, make_shape(Int<CTA_M>{}, Int<CTA_K>{}));
}

template <int CTA_N, int CTA_K>
CUTE_HOST_DEVICE auto make_smem_b_layout() {
  using namespace cute;
  constexpr int kAtomN = CTA_N < 64 ? CTA_N : 64;
  auto atom_layout =
      composition(Swizzle<3, 3, 3>{},
                  Layout<Shape<Int<kAtomN>, Int<8>>,
                         Stride<Int<1>, Int<kAtomN>>>{});
  return tile_to_shape(atom_layout, make_shape(Int<CTA_N>{}, Int<CTA_K>{}));
}

__device__ __forceinline__ cutlass::half_t half_from_uint4_lane(
    uint4 packet, int lane) {
  uint32_t word = lane < 2 ? packet.x : (lane < 4 ? packet.y
                                                  : (lane < 6 ? packet.z
                                                              : packet.w));
  uint16_t bits = static_cast<uint16_t>((lane & 1) ? (word >> 16)
                                                   : (word & 0xffffu));
  return cutlass::half_t::bitcast(bits);
}

__device__ __forceinline__ uint16_t half_bits_from_uint4_lane(uint4 packet,
                                                              int lane) {
  uint32_t word = lane < 2 ? packet.x : (lane < 4 ? packet.y
                                                  : (lane < 6 ? packet.z
                                                              : packet.w));
  return static_cast<uint16_t>((lane & 1) ? (word >> 16) : (word & 0xffffu));
}

template <int CTA_M, int CTA_K, int Threads, class SmemTensor>
__device__ __forceinline__ void copy_a_gmem_to_smem_bounded(
    const cutlass::half_t* __restrict__ a, SmemTensor& sA, int m0, int k0,
    int k) {
  static_assert(CTA_K == 32, "SM70 CuTe native A copy expects CTA_K=32.");
  constexpr int kVecHalf = 8;
  static_assert(CTA_K % kVecHalf == 0,
                "SM70 CuTe native A vector copy expects CTA_K multiple of 8.");
  constexpr int kPacketsPerRow = CTA_K / kVecHalf;
  constexpr int kPackets = CTA_M * kPacketsPerRow;
  for (int linear = int(threadIdx.x); linear < kPackets;
       linear += Threads) {
    int row = linear / kPacketsPerRow;
    int packet_col = linear - row * kPacketsPerRow;
    int col = packet_col * kVecHalf;
    uint4 packet =
        *reinterpret_cast<const uint4*>(a + (m0 + row) * k + (k0 + col));
    CUTE_UNROLL
    for (int i = 0; i < kVecHalf; ++i) {
      sA(row, col + i) = half_from_uint4_lane(packet, i);
    }
  }
}

template <int CTA_N, int CTA_K, int Threads, class SmemTensor>
__device__ __forceinline__ void copy_b_gmem_to_smem_bounded(
    const cutlass::half_t* __restrict__ b, SmemTensor& sB, int n0, int k0,
    int n) {
  static_assert(CTA_K == 32, "SM70 CuTe native B copy expects CTA_K=32.");
  constexpr int kVecHalf = 8;
  static_assert(CTA_N % kVecHalf == 0,
                "SM70 CuTe native B vector copy expects CTA_N multiple of 8.");
  constexpr int kPacketsPerK = CTA_N / kVecHalf;
  constexpr int kPackets = CTA_K * kPacketsPerK;
  for (int linear = int(threadIdx.x); linear < kPackets;
       linear += Threads) {
    int kk = linear / kPacketsPerK;
    int packet_col = linear - kk * kPacketsPerK;
    int col = packet_col * kVecHalf;
    uint4 packet =
        *reinterpret_cast<const uint4*>(b + (k0 + kk) * n + (n0 + col));
    CUTE_UNROLL
    for (int i = 0; i < kVecHalf; ++i) {
      sB(col + i, kk) = half_from_uint4_lane(packet, i);
    }
  }
}

__device__ __forceinline__ int sm70_atom_rowcol(int lane) {
  return (lane & 0x3) + 4 * ((lane >> 4) & 0x1);
}

__device__ __forceinline__ bool sm70_atom_is_canonical_lane(int lane) {
  return (lane & 0xc) == 0;
}

__device__ __forceinline__ int sm70_atom_c_dst_m(int lane, int vid) {
  return (lane & 0x1) + 2 * ((vid >> 1) & 0x1) + 4 * ((lane >> 4) & 0x1);
}

__device__ __forceinline__ int sm70_atom_c_dst_n(int lane, int vid) {
  return (vid & 0x1) + 2 * ((lane >> 1) & 0x1) + 4 * ((vid >> 2) & 0x1);
}

template <int Stride>
__device__ __forceinline__ int sm70_transform_a_index(int idx) {
  int row = idx / Stride;
  return Stride * row + ((idx % Stride) ^ (row & 0x7));
}

__device__ __forceinline__ uint32_t load_sm70_shared_u32(const void* smem_ptr,
                                                         int byte_offset) {
  uint32_t smem =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr)) + byte_offset;
  uint32_t value;
  asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(value) : "r"(smem));
  return value;
}

__device__ __forceinline__ void load_sm70_shared_u32x4(
    uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3,
    const void* smem_ptr, int byte_offset) {
  uint32_t smem =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr)) + byte_offset;
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
               : "r"(smem));
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
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4),
                 "=f"(d5), "=f"(d6), "=f"(d7)
               : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "f"(accum[0]),
                 "f"(accum[1]), "f"(accum[2]), "f"(accum[3]),
                 "f"(accum[4]), "f"(accum[5]), "f"(accum[6]),
                 "f"(accum[7]));
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

struct Sm70AtomAFragment {
  uint32_t words[2][2][2];
};

struct Sm70AtomBFragment {
  uint32_t words[4][2];
};

struct Sm70AtomAccumulator {
  float accum[8];
};

__device__ __forceinline__ void zero_sm70_atom_accumulator(
    Sm70AtomAccumulator& accum) {
  CUTE_UNROLL
  for (int i = 0; i < 8; ++i) {
    accum.accum[i] = 0.0f;
  }
}

template <int CTA_K>
__device__ __forceinline__ void load_sm70_atom_a_fragment(
    Sm70AtomAFragment& frag, const int4* __restrict__ sh_a, int m_atom,
    int k16, int lane_row) {
  static_assert(CTA_K == 128, "SM70 atom A fragment loader expects CTA_K=128.");
  constexpr int kAShStride = CTA_K / 8;
  CUTE_UNROLL
  for (int k_block = 0; k_block < 2; ++k_block) {
    int row = 8 * m_atom + lane_row;
    int local_k_base = 16 * k16 + 8 * k_block;
    CUTE_UNROLL
    for (int k_slice = 0; k_slice < 2; ++k_slice) {
      CUTE_UNROLL
      for (int pair = 0; pair < 2; ++pair) {
        int col = local_k_base + 4 * k_slice + 2 * pair;
        int chunk =
            sm70_transform_a_index<kAShStride>(row * kAShStride + col / 8);
        int byte_offset =
            chunk * int(sizeof(int4)) + ((col & 0x7) / 2) * int(sizeof(uint32_t));
        frag.words[k_block][k_slice][pair] =
            load_sm70_shared_u32(sh_a, byte_offset);
      }
    }
  }
}

template <int CTA_K>
__device__ __forceinline__ void load_sm70_atom_b_fragment(
    Sm70AtomBFragment& frag, const int4* __restrict__ sh_b,
    int n_atom, int k16, int lane_row) {
  static_assert(CTA_K == 128, "SM70 atom B fragment loader expects CTA_K=128.");
  constexpr int kNAtoms = 64 / 8;
  CUTE_UNROLL
  for (int k_block = 0; k_block < 2; ++k_block) {
    int chunk = ((2 * k16 + k_block) * kNAtoms + n_atom) * 8 + lane_row;
    load_sm70_shared_u32x4(frag.words[0][k_block],
                           frag.words[1][k_block],
                           frag.words[2][k_block],
                           frag.words[3][k_block], sh_b,
                           chunk * int(sizeof(int4)));
  }
}

__device__ __forceinline__ void mma_sm70_atom(
    const Sm70AtomAFragment& a_frag, const Sm70AtomBFragment& b_frag,
    float* raw_accum) {
  CUTE_UNROLL
  for (int k_block = 0; k_block < 2; ++k_block) {
    run_sm70_atom_packed(b_frag.words[0][k_block],
                         b_frag.words[1][k_block],
                         a_frag.words[k_block][0][0],
                         a_frag.words[k_block][0][1], raw_accum);
    run_sm70_atom_packed(b_frag.words[2][k_block],
                         b_frag.words[3][k_block],
                         a_frag.words[k_block][1][0],
                         a_frag.words[k_block][1][1], raw_accum);
  }
}

template <int CTA_M, int CTA_K, int Threads>
__device__ __forceinline__ void copy_sm70_atom_a_gmem_to_smem(
    const cutlass::half_t* __restrict__ a, int4* __restrict__ sh_a, int m0,
    int k0, int k) {
  static_assert(CTA_K == 128, "SM70 atom A copy expects CTA_K=128.");
  constexpr int kVecHalf = 8;
  constexpr int kAShStride = CTA_K / kVecHalf;
  constexpr int kPackets = CTA_M * kAShStride;
  for (int linear = int(threadIdx.x); linear < kPackets; linear += Threads) {
    int row = linear / kAShStride;
    int packet_col = linear - row * kAShStride;
    int sh_idx = sm70_transform_a_index<kAShStride>(linear);
    sh_a[sh_idx] = *reinterpret_cast<const int4*>(
        a + (m0 + row) * k + (k0 + packet_col * kVecHalf));
  }
}

template <int CTA_N, int CTA_K, int Threads>
__device__ __forceinline__ void copy_sm70_atom_b_gmem_to_smem(
    const cutlass::half_t* __restrict__ b, int4* __restrict__ sh_b, int n0,
    int k0, int n) {
  static_assert(CTA_N == 64, "SM70 atom B copy expects CTA_N=64.");
  static_assert(CTA_K == 128, "SM70 atom B copy expects CTA_K=128.");
  constexpr int kVecHalf = 8;
  constexpr int kNAtoms = CTA_N / 8;
  constexpr int kKPairs = CTA_K / 2;
  constexpr int kPackets = kKPairs * kNAtoms;
  for (int linear = int(threadIdx.x); linear < kPackets; linear += Threads) {
    int k_pair = linear / kNAtoms;
    int n_atom = linear - k_pair * kNAtoms;
    int kk = 2 * k_pair;
    int col = n_atom * kVecHalf;
    uint4 packet0 =
        *reinterpret_cast<const uint4*>(b + (k0 + kk) * n + (n0 + col));
    uint4 packet1 =
        *reinterpret_cast<const uint4*>(b + (k0 + kk + 1) * n + (n0 + col));
    int k16 = kk / 16;
    int k_block = (kk & 0x8) >> 3;
    int row_group = (kk & 0x7) >> 1;
    CUTE_UNROLL
    for (int i = 0; i < kVecHalf; ++i) {
      int chunk = ((2 * k16 + k_block) * kNAtoms + n_atom) * 8 + i;
      uint32_t low = static_cast<uint32_t>(half_bits_from_uint4_lane(packet0, i));
      uint32_t high =
          static_cast<uint32_t>(half_bits_from_uint4_lane(packet1, i));
      reinterpret_cast<uint32_t*>(&sh_b[chunk])[row_group] =
          low | (high << 16);
    }
  }
}

template <int CTA_M, int CTA_N, int CTA_K, int Warps>
__global__ __launch_bounds__(Warps * 32, 1)
void sm70_atom_gemm_kernel(const cutlass::half_t* __restrict__ a,
                           const cutlass::half_t* __restrict__ b,
                           cutlass::half_t* __restrict__ c, int m, int n,
                           int k) {
  static_assert(CTA_M == 8 || CTA_M == 16,
                "SM70 atom probe supports CTA_M in {8,16}.");
  static_assert(CTA_N == 64, "SM70 atom probe supports CTA_N=64.");
  static_assert(CTA_K == 128, "SM70 atom probe supports CTA_K=128.");
  static_assert(Warps == 4, "SM70 atom probe supports 4 warps.");

  constexpr int kThreads = Warps * 32;
  constexpr int kMAtoms = CTA_M / 8;
  constexpr int kNAtoms = CTA_N / 8;
  constexpr int kNAtomsPerWarp = kNAtoms / Warps;
  static_assert(kNAtoms % Warps == 0,
                "SM70 atom probe requires whole N atoms per warp.");
  constexpr int kAInt4Count = CTA_M * (CTA_K / 8);
  constexpr int kASmemBytes = kAInt4Count * int(sizeof(int4));

  extern __shared__ char smem[];
  int4* sh_a = reinterpret_cast<int4*>(smem);
  int4* sh_b = reinterpret_cast<int4*>(smem + kASmemBytes);

  int lane = int(threadIdx.x) & 31;
  int warp_id = int(threadIdx.x) >> 5;
  int lane_row = sm70_atom_rowcol(lane);
  int m0 = int(blockIdx.x) * CTA_M;
  int n0 = int(blockIdx.y) * CTA_N;

  Sm70AtomAccumulator accum[kMAtoms][kNAtomsPerWarp];
  CUTE_UNROLL
  for (int mi = 0; mi < kMAtoms; ++mi) {
    CUTE_UNROLL
    for (int ni = 0; ni < kNAtomsPerWarp; ++ni) {
      zero_sm70_atom_accumulator(accum[mi][ni]);
    }
  }

  int k_tiles = k / CTA_K;
  CUTE_NO_UNROLL
  for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
    int k0 = k_tile * CTA_K;
    copy_sm70_atom_a_gmem_to_smem<CTA_M, CTA_K, kThreads>(a, sh_a, m0, k0, k);
    copy_sm70_atom_b_gmem_to_smem<CTA_N, CTA_K, kThreads>(b, sh_b, n0, k0, n);
    __syncthreads();

    CUTE_UNROLL
    for (int k16 = 0; k16 < CTA_K / 16; ++k16) {
      Sm70AtomAFragment a_frag[kMAtoms];
      CUTE_UNROLL
      for (int mi = 0; mi < kMAtoms; ++mi) {
        load_sm70_atom_a_fragment<CTA_K>(a_frag[mi], sh_a, mi, k16,
                                         lane_row);
      }
      CUTE_UNROLL
      for (int ni = 0; ni < kNAtomsPerWarp; ++ni) {
        int n_atom = warp_id * kNAtomsPerWarp + ni;
        Sm70AtomBFragment b_frag;
        load_sm70_atom_b_fragment<CTA_K>(b_frag, sh_b, n_atom, k16, lane_row);
        CUTE_UNROLL
        for (int mi = 0; mi < kMAtoms; ++mi) {
          mma_sm70_atom(a_frag[mi], b_frag, accum[mi][ni].accum);
        }
      }
    }
    __syncthreads();
  }

  if (sm70_atom_is_canonical_lane(lane)) {
    CUTE_UNROLL
    for (int mi = 0; mi < kMAtoms; ++mi) {
      CUTE_UNROLL
      for (int ni = 0; ni < kNAtomsPerWarp; ++ni) {
        int n_atom = warp_id * kNAtomsPerWarp + ni;
        CUTE_UNROLL
        for (int vid = 0; vid < 8; ++vid) {
          int row = 8 * mi + sm70_atom_c_dst_n(lane, vid);
          int col = 8 * n_atom + sm70_atom_c_dst_m(lane, vid);
          c[(m0 + row) * n + (n0 + col)] =
              static_cast<cutlass::half_t>(accum[mi][ni].accum[vid]);
        }
      }
    }
  }
}

template <int CTA_M, int CTA_N, int CTA_K, int Warps>
at::Tensor run_sm70_atom_gemm(const at::Tensor& a, const at::Tensor& b) {
  c10::cuda::CUDAGuard device_guard(a.device());
  at::Tensor out = at::empty({a.size(0), b.size(1)}, a.options());
  TORCH_CHECK(is_aligned_16(out.data_ptr()),
              "sm70_cutlass_matmul_probe: SM70 atom C output must be 16-byte "
              "aligned");

  constexpr int kThreads = Warps * 32;
  constexpr int kAInt4Count = CTA_M * (CTA_K / 8);
  constexpr int kBInt4Count = (CTA_K / 8) * CTA_N;
  size_t smem_bytes =
      (static_cast<size_t>(kAInt4Count) + static_cast<size_t>(kBInt4Count)) *
      sizeof(int4);

  auto kernel = sm70_atom_gemm_kernel<CTA_M, CTA_N, CTA_K, Warps>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes)));

  dim3 grid(static_cast<unsigned>(a.size(0) / CTA_M),
            static_cast<unsigned>(b.size(1) / CTA_N));
  dim3 block(kThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

  kernel<<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<const cutlass::half_t*>(a.data_ptr<at::Half>()),
      reinterpret_cast<const cutlass::half_t*>(b.data_ptr<at::Half>()),
      reinterpret_cast<cutlass::half_t*>(out.data_ptr<at::Half>()),
      static_cast<int>(a.size(0)), static_cast<int>(b.size(1)),
      static_cast<int>(a.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <int CTA_M, int CTA_N, int CTA_K, int Warps>
__global__ __launch_bounds__(Warps * 32, 1)
void sm70_cute_gemm_kernel(const cutlass::half_t* __restrict__ a,
                           const cutlass::half_t* __restrict__ b,
                           cutlass::half_t* __restrict__ c, int m, int n,
                           int k) {
  using namespace cute;

  static_assert(CTA_K == 32, "SM70 CuTe native probe supports CTA_K=32.");
  static_assert(Sm70AtomLayout<CTA_M, CTA_N, Warps>::kAtomM * 8 <= CTA_M,
                "SM70 CuTe native kernel must not use padded-M atom layout.");
  constexpr int kThreads = Warps * 32;
  auto tiled_mma = make_tiled_mma(
      SM70_8x8x4_F32F16F16F32_TN{},
      typename Sm70AtomLayout<CTA_M, CTA_N, Warps>::Type{},
      Tile<Int<CTA_M>, Int<CTA_N>, Int<4>>{});

  Tensor mC = make_tensor(make_gmem_ptr(c), make_shape(m, n),
                          make_stride(n, Int<1>{}));

  auto cta_tiler = make_shape(Int<CTA_M>{}, Int<CTA_N>{}, Int<CTA_K>{});
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

  auto sA_layout = make_smem_a_layout<CTA_M, CTA_K>();
  auto sB_layout = make_smem_b_layout<CTA_N, CTA_K>();

  extern __shared__ char smem[];
  auto* smem_a = reinterpret_cast<cutlass::half_t*>(smem);
  auto* smem_b = reinterpret_cast<cutlass::half_t*>(
      smem + sizeof(cutlass::half_t) * cosize(sA_layout));
  Tensor sA = make_tensor(make_smem_ptr(smem_a), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smem_b), sB_layout);

  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  int m0 = int(blockIdx.x) * CTA_M;
  int n0 = int(blockIdx.y) * CTA_N;
  int k_tiles = k / CTA_K;
  CUTE_NO_UNROLL
  for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
    int k0 = k_tile * CTA_K;
    copy_a_gmem_to_smem_bounded<CTA_M, CTA_K, kThreads>(a, sA, m0, k0, k);
    copy_b_gmem_to_smem_bounded<CTA_N, CTA_K, kThreads>(b, sB, n0, k0, n);
    __syncthreads();

    if constexpr (CTA_M == 32 || CTA_M == 64) {
      cute::cooperative_gemm(threadIdx.x, tiled_mma, sA, sB, tCrC, identity{},
                             identity{});
    } else {
      cute::detail::cooperative_gemm_predication(thr_mma, sA, sB, tCrC,
                                                 identity{}, identity{});
    }
    __syncthreads();
  }

  Tensor cC = make_identity_tensor(make_shape(Int<CTA_M>{}, Int<CTA_N>{}));
  Tensor tCcC = thr_mma.partition_C(cC);
  CUTE_UNROLL
  for (int i = 0; i < size(tCrC); ++i) {
    if (elem_less(tCcC(i), make_coord(Int<CTA_M>{}, Int<CTA_N>{}))) {
      tCgC(i) = static_cast<cutlass::half_t>(tCrC(i));
    }
  }
}

template <int CTA_M, int CTA_N, int CTA_K, int Warps>
at::Tensor run_sm70_cute_gemm(const at::Tensor& a, const at::Tensor& b) {
  c10::cuda::CUDAGuard device_guard(a.device());
  at::Tensor out = at::empty({a.size(0), b.size(1)}, a.options());
  TORCH_CHECK(is_aligned_16(out.data_ptr()),
              "sm70_cutlass_matmul_probe: CuTe C output must be 16-byte "
              "aligned");

  constexpr int kThreads = Warps * 32;
  auto sA_layout = make_smem_a_layout<CTA_M, CTA_K>();
  auto sB_layout = make_smem_b_layout<CTA_N, CTA_K>();
  size_t smem_bytes =
      sizeof(cutlass::half_t) *
      (static_cast<size_t>(cute::cosize(sA_layout)) +
       static_cast<size_t>(cute::cosize(sB_layout)));

  auto kernel = sm70_cute_gemm_kernel<CTA_M, CTA_N, CTA_K, Warps>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes)));

  dim3 grid(static_cast<unsigned>((a.size(0) + CTA_M - 1) / CTA_M),
            static_cast<unsigned>(b.size(1) / CTA_N));
  dim3 block(kThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

  kernel<<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<const cutlass::half_t*>(a.data_ptr<at::Half>()),
      reinterpret_cast<const cutlass::half_t*>(b.data_ptr<at::Half>()),
      reinterpret_cast<cutlass::half_t*>(out.data_ptr<at::Half>()),
      static_cast<int>(a.size(0)), static_cast<int>(b.size(1)),
      static_cast<int>(a.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <int CTA_M, int CTA_N, int CTA_K, int Warps>
struct Sm70ThreadblockWarpShape;

template <>
struct Sm70ThreadblockWarpShape<32, 128, 32, 4> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

template <>
struct Sm70ThreadblockWarpShape<32, 256, 32, 4> {
  using Type = cutlass::gemm::GemmShape<32, 64, 32>;
};

template <>
struct Sm70ThreadblockWarpShape<64, 64, 32, 4> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

template <>
struct Sm70ThreadblockWarpShape<64, 128, 32, 4> {
  using Type = cutlass::gemm::GemmShape<32, 64, 32>;
};

template <>
struct Sm70ThreadblockWarpShape<64, 128, 32, 8> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

template <>
struct Sm70ThreadblockWarpShape<64, 256, 32, 4> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70ThreadblockWarpShape<64, 256, 32, 8> {
  using Type = cutlass::gemm::GemmShape<32, 64, 32>;
};

template <>
struct Sm70ThreadblockWarpShape<64, 512, 32, 8> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70ThreadblockWarpShape<128, 64, 32, 4> {
  using Type = cutlass::gemm::GemmShape<64, 32, 32>;
};

template <>
struct Sm70ThreadblockWarpShape<128, 64, 32, 8> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

template <>
struct Sm70ThreadblockWarpShape<128, 128, 32, 4> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70ThreadblockWarpShape<128, 128, 32, 8> {
  using Type = cutlass::gemm::GemmShape<64, 32, 32>;
};

template <>
struct Sm70ThreadblockWarpShape<128, 256, 32, 8> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70ThreadblockWarpShape<256, 64, 32, 4> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70ThreadblockWarpShape<256, 64, 32, 8> {
  using Type = cutlass::gemm::GemmShape<64, 32, 32>;
};

template <>
struct Sm70ThreadblockWarpShape<256, 128, 32, 8> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70ThreadblockWarpShape<512, 64, 32, 8> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <int CTA_M, int CTA_N, int CTA_K, int Warps>
struct Sm70ThreadblockGemmTraits {
  static_assert(CTA_K == 32,
                "Extracted SM70 threadblock path supports only K=32");

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using ThreadblockShape = cutlass::gemm::GemmShape<CTA_M, CTA_N, CTA_K>;
  using WarpShape =
      typename Sm70ThreadblockWarpShape<CTA_M, CTA_N, CTA_K, Warps>::Type;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      cutlass::arch::OpClassTensorOp, 2, cutlass::arch::OpMultiplyAdd>;
  static_assert(MmaCore::kThreads == Warps * 32,
                "SM70 pure GEMM launch threads must match CUTLASS warp count.");
  using OutputOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, ElementAccumulator>;
  using Mma = typename cutlass::gemm::threadblock::DefaultMma<
      ElementA, LayoutA, 128 / cutlass::sizeof_bits<ElementA>::value,
      ElementB, LayoutB, 128 / cutlass::sizeof_bits<ElementB>::value,
      ElementAccumulator, LayoutC, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm70, ThreadblockShape, WarpShape, InstructionShape, 2,
      cutlass::arch::OpMultiplyAdd, false,
      cutlass::gemm::SharedMemoryClearOption::kNone>::ThreadblockMma;
  // TensorOp 8x8x4 on Volta requires the predefined permuted SMEM layouts.
  // Keep these assertions close to the reusable traits so the later probe
  // experiments do not drift away from CUTLASS' bank-conflict-free layouts.
  using ExpectedSmemLayoutA =
      cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<
          cutlass::sizeof_bits<ElementA>::value, ThreadblockShape::kK>;
  using ExpectedSmemLayoutB =
      cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<
          cutlass::sizeof_bits<ElementB>::value>;
  using ActualSmemLayoutA = typename Mma::SmemIteratorA::Layout;
  using ActualSmemLayoutB = typename Mma::SmemIteratorB::Layout;
  static_assert(std::is_same<ActualSmemLayoutA, ExpectedSmemLayoutA>::value,
                "SM70 A operand must use the predefined Volta row-major "
                "crosswise shared-memory layout for TensorOp 8x8x4.");
  static_assert(std::is_same<ActualSmemLayoutB, ExpectedSmemLayoutB>::value,
                "SM70 B operand must use the predefined Volta shared-memory "
                "layout selected for TensorOp 8x8x4.");
  static int const kPartitionsK = ThreadblockShape::kK / WarpShape::kK;
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
          ThreadblockShape, typename Mma::Operator, kPartitionsK, OutputOp,
          OutputOp::kCount>::Epilogue;

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };
};

template <int CTA_M, int CTA_N, int CTA_K, int Warps>
__global__ __launch_bounds__(Warps * 32, 1)
void sm70_cutlass_threadblock_gemm_kernel(
    const cutlass::half_t* __restrict__ a,
    const cutlass::half_t* __restrict__ b,
    cutlass::half_t* __restrict__ c, int m, int n, int k) {
  using Traits = Sm70ThreadblockGemmTraits<CTA_M, CTA_N, CTA_K, Warps>;
  using Mma = typename Traits::Mma;
  using Epilogue = typename Traits::Epilogue;

  extern __shared__ char smem[];
  auto& shared_storage =
      *reinterpret_cast<typename Traits::SharedStorage*>(smem);

  int thread_idx = threadIdx.x;
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_idx = threadIdx.x % 32;

  cutlass::MatrixCoord tb_offset_A{int(blockIdx.x) * CTA_M, 0};
  cutlass::MatrixCoord tb_offset_B{0, int(blockIdx.y) * CTA_N};
  cutlass::MatrixCoord tb_offset_C{int(blockIdx.x) * CTA_M,
                                   int(blockIdx.y) * CTA_N};

  typename Traits::LayoutA layout_a(k);
  typename Traits::LayoutB layout_b(n);
  typename Traits::LayoutC layout_c(n);

  typename Mma::IteratorA iterator_A(
      typename Mma::IteratorA::Params(layout_a),
      const_cast<cutlass::half_t*>(a), cutlass::MatrixCoord(m, k), thread_idx,
      tb_offset_A);
  typename Mma::IteratorB iterator_B(
      typename Mma::IteratorB::Params(layout_b),
      const_cast<cutlass::half_t*>(b), cutlass::MatrixCoord(k, n), thread_idx,
      tb_offset_B);

  Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
  typename Mma::FragmentC accumulators;
  accumulators.clear();

  int gemm_k_iterations = (k + CTA_K - 1) / CTA_K;
  mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

  typename Traits::OutputOp output_op({1.0f, 0.0f});
  typename Epilogue::OutputTileIterator iterator_C(
      typename Epilogue::OutputTileIterator::Params(layout_c),
      c, cutlass::MatrixCoord(m, n), thread_idx, tb_offset_C);
  typename Epilogue::OutputTileIterator iterator_D(
      typename Epilogue::OutputTileIterator::Params(layout_c),
      c, cutlass::MatrixCoord(m, n), thread_idx, tb_offset_C);

  Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
  epilogue(output_op, iterator_D, accumulators, iterator_C);
}

template <int CTA_M, int CTA_N, int CTA_K, int Warps>
at::Tensor run_sm70_cutlass_threadblock_gemm(const at::Tensor& a,
                                             const at::Tensor& b) {
  c10::cuda::CUDAGuard device_guard(a.device());
  at::Tensor out = at::empty({a.size(0), b.size(1)}, a.options());

  using Traits = Sm70ThreadblockGemmTraits<CTA_M, CTA_N, CTA_K, Warps>;
  constexpr int kThreads = Warps * 32;
  auto kernel = sm70_cutlass_threadblock_gemm_kernel<CTA_M, CTA_N, CTA_K, Warps>;
  size_t smem_bytes = sizeof(typename Traits::SharedStorage);
  if (smem_bytes >= (48u << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
  }

  dim3 grid(static_cast<unsigned>((a.size(0) + CTA_M - 1) / CTA_M),
            static_cast<unsigned>(b.size(1) / CTA_N));
  dim3 block(kThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

  kernel<<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<const cutlass::half_t*>(a.data_ptr<at::Half>()),
      reinterpret_cast<const cutlass::half_t*>(b.data_ptr<at::Half>()),
      reinterpret_cast<cutlass::half_t*>(out.data_ptr<at::Half>()),
      static_cast<int>(a.size(0)), static_cast<int>(b.size(1)),
      static_cast<int>(a.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void validate_sm70_cutlass_explicit_warp_probe_inputs(
    const at::Tensor& a, const at::Tensor& b, int64_t cta_m, int64_t cta_n,
    int64_t cta_k, int64_t warps, int64_t warp_m, int64_t warp_n,
    int64_t warp_k) {
  validate_sm70_cutlass_matmul_probe_inputs(
      a, b, cta_m, cta_n, cta_k, warps, 2, kAPathCutlassThreadblock,
      kBPathCuteShared);
  TORCH_CHECK(warp_m == 32 || warp_m == 64,
              "sm70_cutlass_matmul_explicit_warp_probe: warp_m must be 32 or 64");
  TORCH_CHECK(warp_n == 32 || warp_n == 64,
              "sm70_cutlass_matmul_explicit_warp_probe: warp_n must be 32 or 64");
  TORCH_CHECK(warp_k == 16 || warp_k == 32,
              "sm70_cutlass_matmul_explicit_warp_probe: warp_k must be 16 or 32");
  TORCH_CHECK(warp_k != 16,
              "sm70_cutlass_matmul_explicit_warp_probe: WarpK=16 is rejected "
              "for this stock CUTLASS diagnostic probe because it does not use "
              "the production SM70 Marlin phase-aware K-offset helper");
  TORCH_CHECK(cta_m % warp_m == 0 && cta_n % warp_n == 0 &&
                  cta_k % warp_k == 0,
              "sm70_cutlass_matmul_explicit_warp_probe: CTA shape must be "
              "divisible by explicit warp shape");
  TORCH_CHECK((cta_m / warp_m) * (cta_n / warp_n) * (cta_k / warp_k) == warps,
              "sm70_cutlass_matmul_explicit_warp_probe: explicit warp shape "
              "must decompose the CTA into the requested warp count");
}

template <int CTA_M, int CTA_N, int CTA_K, int Warps, int WarpM, int WarpN,
          int WarpK>
struct Sm70ExplicitWarpThreadblockGemmTraits {
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using ThreadblockShape = cutlass::gemm::GemmShape<CTA_M, CTA_N, CTA_K>;
  using WarpShape = cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  static_assert((CTA_M / WarpM) * (CTA_N / WarpN) * (CTA_K / WarpK) == Warps,
                "Explicit warp shape must decompose the CTA into Warps.");
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      cutlass::arch::OpClassTensorOp, 2, cutlass::arch::OpMultiplyAdd>;
  static_assert(MmaCore::kThreads == Warps * 32,
                "SM70 explicit warp probe launch threads must match CUTLASS warp count.");
  using OutputOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, ElementAccumulator>;
  using Mma = typename cutlass::gemm::threadblock::DefaultMma<
      ElementA, LayoutA, 128 / cutlass::sizeof_bits<ElementA>::value,
      ElementB, LayoutB, 128 / cutlass::sizeof_bits<ElementB>::value,
      ElementAccumulator, LayoutC, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm70, ThreadblockShape, WarpShape, InstructionShape, 2,
      cutlass::arch::OpMultiplyAdd, false,
      cutlass::gemm::SharedMemoryClearOption::kNone>::ThreadblockMma;
  using ExpectedSmemLayoutA =
      cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<
          cutlass::sizeof_bits<ElementA>::value, ThreadblockShape::kK>;
  using ExpectedSmemLayoutB =
      cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<
          cutlass::sizeof_bits<ElementB>::value>;
  using ActualSmemLayoutA = typename Mma::SmemIteratorA::Layout;
  using ActualSmemLayoutB = typename Mma::SmemIteratorB::Layout;
  static_assert(std::is_same<ActualSmemLayoutA, ExpectedSmemLayoutA>::value,
                "SM70 explicit warp probe A operand must use Volta row-major "
                "crosswise shared-memory layout.");
  static_assert(std::is_same<ActualSmemLayoutB, ExpectedSmemLayoutB>::value,
                "SM70 explicit warp probe B operand must use Volta B-congruous "
                "shared-memory layout.");
  static int const kPartitionsK = ThreadblockShape::kK / WarpShape::kK;
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
          ThreadblockShape, typename Mma::Operator, kPartitionsK, OutputOp,
          OutputOp::kCount>::Epilogue;

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };
};

template <int CTA_M, int CTA_N, int CTA_K, int Warps, int WarpM, int WarpN,
          int WarpK>
__global__ __launch_bounds__(Warps * 32, 1)
void sm70_cutlass_explicit_warp_gemm_kernel(
    const cutlass::half_t* __restrict__ a,
    const cutlass::half_t* __restrict__ b,
    cutlass::half_t* __restrict__ c, int m, int n, int k) {
  using Traits = Sm70ExplicitWarpThreadblockGemmTraits<
      CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN, WarpK>;
  using Mma = typename Traits::Mma;
  using Epilogue = typename Traits::Epilogue;

  extern __shared__ char smem[];
  auto& shared_storage =
      *reinterpret_cast<typename Traits::SharedStorage*>(smem);

  int thread_idx = threadIdx.x;
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_idx = threadIdx.x % 32;

  cutlass::MatrixCoord tb_offset_A{int(blockIdx.x) * CTA_M, 0};
  cutlass::MatrixCoord tb_offset_B{0, int(blockIdx.y) * CTA_N};
  cutlass::MatrixCoord tb_offset_C{int(blockIdx.x) * CTA_M,
                                   int(blockIdx.y) * CTA_N};

  typename Traits::LayoutA layout_a(k);
  typename Traits::LayoutB layout_b(n);
  typename Traits::LayoutC layout_c(n);

  typename Mma::IteratorA iterator_A(
      typename Mma::IteratorA::Params(layout_a),
      const_cast<cutlass::half_t*>(a), cutlass::MatrixCoord(m, k), thread_idx,
      tb_offset_A);
  typename Mma::IteratorB iterator_B(
      typename Mma::IteratorB::Params(layout_b),
      const_cast<cutlass::half_t*>(b), cutlass::MatrixCoord(k, n), thread_idx,
      tb_offset_B);

  Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
  typename Mma::FragmentC accumulators;
  accumulators.clear();

  int gemm_k_iterations = (k + CTA_K - 1) / CTA_K;
  mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

  typename Traits::OutputOp output_op({1.0f, 0.0f});
  typename Epilogue::OutputTileIterator iterator_C(
      typename Epilogue::OutputTileIterator::Params(layout_c),
      c, cutlass::MatrixCoord(m, n), thread_idx, tb_offset_C);
  typename Epilogue::OutputTileIterator iterator_D(
      typename Epilogue::OutputTileIterator::Params(layout_c),
      c, cutlass::MatrixCoord(m, n), thread_idx, tb_offset_C);

  Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
  epilogue(output_op, iterator_D, accumulators, iterator_C);
}

template <int CTA_M, int CTA_N, int CTA_K, int Warps, int WarpM, int WarpN,
          int WarpK>
at::Tensor run_sm70_cutlass_explicit_warp_gemm(const at::Tensor& a,
                                               const at::Tensor& b) {
  c10::cuda::CUDAGuard device_guard(a.device());
  at::Tensor out = at::empty({a.size(0), b.size(1)}, a.options());

  using Traits = Sm70ExplicitWarpThreadblockGemmTraits<
      CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN, WarpK>;
  constexpr int kThreads = Warps * 32;
  auto kernel = sm70_cutlass_explicit_warp_gemm_kernel<
      CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN, WarpK>;
  size_t smem_bytes = sizeof(typename Traits::SharedStorage);
  if (smem_bytes >= (48u << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
  }

  dim3 grid(static_cast<unsigned>((a.size(0) + CTA_M - 1) / CTA_M),
            static_cast<unsigned>(b.size(1) / CTA_N));
  dim3 block(kThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

  kernel<<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<const cutlass::half_t*>(a.data_ptr<at::Half>()),
      reinterpret_cast<const cutlass::half_t*>(b.data_ptr<at::Half>()),
      reinterpret_cast<cutlass::half_t*>(out.data_ptr<at::Half>()),
      static_cast<int>(a.size(0)), static_cast<int>(b.size(1)),
      static_cast<int>(a.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

#define DISPATCH_CUTE(CM, CN, CK, W)                                      \
  if (cta_m == CM && cta_n == CN && cta_k == CK && warps == W) {          \
    return run_sm70_cute_gemm<CM, CN, CK, W>(a, b);                       \
  }

#define DISPATCH_CUTE_N(CM, W)                                            \
  DISPATCH_CUTE(CM, 64, 32, W)                                            \
  DISPATCH_CUTE(CM, 128, 32, W)                                           \
  DISPATCH_CUTE(CM, 256, 32, W)

#define DISPATCH_CUTE_M(W)                                                \
  DISPATCH_CUTE_N(8, W)                                                   \
  DISPATCH_CUTE_N(16, W)                                                  \
  DISPATCH_CUTE_N(32, W)                                                  \
  DISPATCH_CUTE_N(48, W)                                                  \
  DISPATCH_CUTE_N(64, W)

at::Tensor dispatch_sm70_cute_gemm(const at::Tensor& a, const at::Tensor& b,
                                   int64_t cta_m, int64_t cta_n,
                                   int64_t cta_k, int64_t warps) {
  DISPATCH_CUTE_M(4)
  DISPATCH_CUTE_M(8)

  TORCH_CHECK(false,
              "sm70_cutlass_matmul_probe: unsupported CUTLASS 3 CuTe native config "
              "cta_m=", cta_m, ", cta_n=", cta_n, ", cta_k=", cta_k,
              ", warps=", warps,
              ". Supported native CuTe shapes are CTA_M in {8, 16, 32, "
              "48, 64}, CTA_N in {64, 128, 256}, CTA_K=32, and warps in "
              "{4, 8}.");
}

#undef DISPATCH_CUTE_M
#undef DISPATCH_CUTE_N
#undef DISPATCH_CUTE

#define DISPATCH_SM70_ATOM(CM)                                           \
  if (cta_m == CM && cta_n == 64 && cta_k == 128 && warps == 4) {        \
    return run_sm70_atom_gemm<CM, 64, 128, 4>(a, b);                     \
  }

at::Tensor dispatch_sm70_atom_gemm(const at::Tensor& a, const at::Tensor& b,
                                   int64_t cta_m, int64_t cta_n,
                                   int64_t cta_k, int64_t warps) {
  DISPATCH_SM70_ATOM(8)
  DISPATCH_SM70_ATOM(16)

  TORCH_CHECK(false,
              "sm70_cutlass_matmul_probe: unsupported SM70 atom config "
              "cta_m=", cta_m, ", cta_n=", cta_n, ", cta_k=", cta_k,
              ", warps=", warps,
              ". Supported SM70 atom shapes are CTA_M in {8, 16}, CTA_N=64, "
              "CTA_K=128, and warps=4.");
}

#undef DISPATCH_SM70_ATOM

#define DISPATCH_THREADBLOCK(CM, CN, CK, W)                               \
  if (cta_m == CM && cta_n == CN && cta_k == CK && warps == W) {          \
    return run_sm70_cutlass_threadblock_gemm<CM, CN, CK, W>(a, b);        \
  }

at::Tensor dispatch_sm70_cutlass_threadblock_gemm(
    const at::Tensor& a, const at::Tensor& b, int64_t cta_m, int64_t cta_n,
    int64_t cta_k, int64_t warps, int64_t /*b_path*/) {
  DISPATCH_THREADBLOCK(32, 128, 32, 4)
  DISPATCH_THREADBLOCK(32, 256, 32, 4)
  DISPATCH_THREADBLOCK(64, 64, 32, 4)
  DISPATCH_THREADBLOCK(64, 128, 32, 4)
  DISPATCH_THREADBLOCK(64, 128, 32, 8)
  DISPATCH_THREADBLOCK(64, 256, 32, 4)
  DISPATCH_THREADBLOCK(64, 256, 32, 8)
  DISPATCH_THREADBLOCK(64, 512, 32, 8)
  DISPATCH_THREADBLOCK(128, 64, 32, 4)
  DISPATCH_THREADBLOCK(128, 64, 32, 8)
  DISPATCH_THREADBLOCK(128, 128, 32, 4)
  DISPATCH_THREADBLOCK(128, 128, 32, 8)
  DISPATCH_THREADBLOCK(128, 256, 32, 8)
  DISPATCH_THREADBLOCK(256, 64, 32, 4)
  DISPATCH_THREADBLOCK(256, 64, 32, 8)
  DISPATCH_THREADBLOCK(256, 128, 32, 8)
  DISPATCH_THREADBLOCK(512, 64, 32, 8)

  TORCH_CHECK(false,
              "sm70_cutlass_matmul_probe: unsupported extracted CUTLASS "
              "threadblock config cta_m=", cta_m, ", cta_n=", cta_n,
              ", cta_k=", cta_k, ", warps=", warps,
              ". Supported CTA_M/CTA_N/warps shapes are 32x128/4, "
              "32x256/4, 64x64/4, 64x128/4, 64x128/8, 64x256/4, "
              "64x256/8, 64x512/8, 128x64/4, 128x64/8, 128x128/4, "
              "128x128/8, 128x256/8, 256x64/4, 256x64/8, "
              "256x128/8, and 512x64/8 with CTA_K 32.");
}

#undef DISPATCH_THREADBLOCK

at::Tensor sm70_cutlass_matmul_probe(const at::Tensor& a, const at::Tensor& b,
                                     int64_t cta_m, int64_t cta_n,
                                     int64_t cta_k, int64_t warps,
                                     int64_t stages, int64_t a_path,
                                     int64_t b_path) {
  validate_sm70_cutlass_matmul_probe_inputs(
      a, b, cta_m, cta_n, cta_k, warps, stages, a_path, b_path);
  if (a_path == kAPathCutlassThreadblock) {
    return dispatch_sm70_cutlass_threadblock_gemm(a, b, cta_m, cta_n, cta_k,
                                                  warps, b_path);
  }
  if (a_path == kAPathSm70Atom) {
    return dispatch_sm70_atom_gemm(a, b, cta_m, cta_n, cta_k, warps);
  }
  return dispatch_sm70_cute_gemm(a, b, cta_m, cta_n, cta_k, warps);
}

#define DISPATCH_EXPLICIT_WARP(CM, CN, CK, W, WM, WN, WK)                 \
  if (cta_m == CM && cta_n == CN && cta_k == CK && warps == W &&          \
      warp_m == WM && warp_n == WN && warp_k == WK) {                    \
    return run_sm70_cutlass_explicit_warp_gemm<CM, CN, CK, W, WM, WN,     \
                                               WK>(a, b);                 \
  }

at::Tensor sm70_cutlass_matmul_explicit_warp_probe(
    const at::Tensor& a, const at::Tensor& b, int64_t cta_m, int64_t cta_n,
    int64_t cta_k, int64_t warps, int64_t warp_m, int64_t warp_n,
    int64_t warp_k) {
  validate_sm70_cutlass_explicit_warp_probe_inputs(
      a, b, cta_m, cta_n, cta_k, warps, warp_m, warp_n, warp_k);

  DISPATCH_EXPLICIT_WARP(64, 64, 32, 4, 32, 32, 32)

  TORCH_CHECK(false,
              "sm70_cutlass_matmul_explicit_warp_probe: unsupported explicit "
              "warp config cta_m=", cta_m, ", cta_n=", cta_n,
              ", cta_k=", cta_k, ", warps=", warps, ", warp_m=", warp_m,
              ", warp_n=", warp_n, ", warp_k=", warp_k,
              ". This diagnostic probe currently instantiates "
              "64x64x32x4x32x32x32.");
}

#undef DISPATCH_EXPLICIT_WARP

}  // namespace

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("sm70_cutlass_matmul_probe", &sm70_cutlass_matmul_probe);
  m.impl("sm70_cutlass_matmul_explicit_warp_probe",
         &sm70_cutlass_matmul_explicit_warp_probe);
}

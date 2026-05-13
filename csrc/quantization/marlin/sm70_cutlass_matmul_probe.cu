#include "core/registration.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>
#include <torch/library.h>
#include <type_traits>

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/layout/tensor_op_multiplicand_sm70.h"
#include "cute/tensor.hpp"
#include "cute/swizzle.hpp"
#include "cute/swizzle_layout.hpp"

namespace {

enum ProbeAPath : int64_t {
  kAPathCuteShared = 0,
  kAPathDirectGlobal = 1,
  kAPathCutlassThreadblock = 2,
};

enum ProbeBPath : int64_t {
  kBPathCuteShared = 0,
};

void check_probe_inputs(const at::Tensor& a, const at::Tensor& b,
                        int64_t cta_m, int64_t cta_n, int64_t cta_k,
                        int64_t warps, int64_t stages, int64_t a_path,
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
                  a_path == kAPathCutlassThreadblock,
              "sm70_cutlass_matmul_probe: unknown A path id ", a_path);
  TORCH_CHECK(a_path != kAPathDirectGlobal,
              "sm70_cutlass_matmul_probe: A direct-global path is TODO; only "
              "CUTLASS 3 CuTe shared-memory path id 0 and extracted CUTLASS "
              "threadblock path id 2 are available");
  TORCH_CHECK(cta_m == 8 || cta_m == 16 || cta_m == 32 || cta_m == 64 ||
                  cta_m == 128,
              "sm70_cutlass_matmul_probe: cta_m must be 8, 16, 32, 64, or 128");
  TORCH_CHECK(cta_n == 32 || cta_n == 64 || cta_n == 128 || cta_n == 256,
              "sm70_cutlass_matmul_probe: cta_n must be 32, 64, 128, or 256");
  TORCH_CHECK(cta_k == 32 || cta_k == 64 || cta_k == 128,
              "sm70_cutlass_matmul_probe: cta_k must be 32, 64, or 128");
  TORCH_CHECK(warps == 4 || warps == 8,
              "sm70_cutlass_matmul_probe: warps must be 4 or 8");
  TORCH_CHECK(a.size(0) % cta_m == 0,
              "sm70_cutlass_matmul_probe: current CuTe probe requires M divisible by cta_m");
  TORCH_CHECK(b.size(1) % cta_n == 0,
              "sm70_cutlass_matmul_probe: current CuTe probe requires N divisible by cta_n");
  TORCH_CHECK(a.size(1) % cta_k == 0,
              "sm70_cutlass_matmul_probe: current CuTe probe requires K divisible by cta_k");

  int major = 0;
  int minor = 0;
  int device = a.get_device();
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  TORCH_CHECK(major == 7 && minor == 0,
              "sm70_cutlass_matmul_probe: this probe only supports SM70 GPUs, got sm",
              major, minor);
}

template <int Warps>
struct Sm70AtomLayout;

template <>
struct Sm70AtomLayout<4> {
  using Type = cute::Layout<cute::Shape<cute::Int<4>, cute::Int<4>, cute::Int<1>>>;
};

template <>
struct Sm70AtomLayout<8> {
  using Type = cute::Layout<cute::Shape<cute::Int<4>, cute::Int<8>, cute::Int<1>>>;
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

template <int CTA_M, int CTA_N, int CTA_K, int Warps>
__global__ __launch_bounds__(Warps * 32, 1)
void sm70_cute_gemm_kernel(const cutlass::half_t* __restrict__ a,
                           const cutlass::half_t* __restrict__ b,
                           cutlass::half_t* __restrict__ c, int m, int n,
                           int k) {
  using namespace cute;

  constexpr int kThreads = Warps * 32;
  constexpr int kCopyAThrK = CTA_K / 8;
  constexpr int kCopyAThrM = kThreads / kCopyAThrK;
  constexpr int kCopyBThrN = CTA_N / 8;
  constexpr int kCopyBThrK = kThreads / kCopyBThrN;
  auto tiled_mma = make_tiled_mma(
      SM70_8x8x4_F32F16F16F32_TN{},
      typename Sm70AtomLayout<Warps>::Type{},
      Tile<Int<CTA_M>, Int<CTA_N>, Int<4>>{});
  TiledCopy copy_a = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint128_t>, cutlass::half_t>{},
      Layout<Shape<Int<kCopyAThrM>, Int<kCopyAThrK>>,
             Stride<Int<kCopyAThrK>, Int<1>>>{},
      Layout<Shape<Int<1>, Int<8>>>{});
  TiledCopy copy_b = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint128_t>, cutlass::half_t>{},
      Layout<Shape<Int<kCopyBThrN>, Int<kCopyBThrK>>,
             Stride<Int<1>, Int<kCopyBThrN>>>{},
      Layout<Shape<Int<8>, Int<1>>>{});

  Tensor mA = make_tensor(make_gmem_ptr(a), make_shape(m, k),
                          make_stride(k, Int<1>{}));
  Tensor mB = make_tensor(make_gmem_ptr(b), make_shape(n, k),
                          make_stride(Int<1>{}, n));
  Tensor mC = make_tensor(make_gmem_ptr(c), make_shape(m, n),
                          make_stride(n, Int<1>{}));

  auto cta_tiler = make_shape(Int<CTA_M>{}, Int<CTA_N>{}, Int<CTA_K>{});
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

  auto sA_layout = make_smem_a_layout<CTA_M, CTA_K>();
  auto sB_layout = make_smem_b_layout<CTA_N, CTA_K>();

  extern __shared__ char smem[];
  auto* smem_a = reinterpret_cast<cutlass::half_t*>(smem);
  auto* smem_b = reinterpret_cast<cutlass::half_t*>(
      smem + sizeof(cutlass::half_t) * cosize(sA_layout));
  Tensor sA = make_tensor(make_smem_ptr(smem_a), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smem_b), sB_layout);

  auto thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tAsA = thr_copy_a.partition_D(sA);
  Tensor tArA = make_fragment_like(tAsA);

  auto thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  Tensor tBsB = thr_copy_b.partition_D(sB);
  Tensor tBrB = make_fragment_like(tBsB);

  copy(copy_a, tAgA(_, _, _, 0), tArA);
  copy(copy_b, tBgB(_, _, _, 0), tBrB);

  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrA = thr_mma.partition_fragment_A(sA);
  Tensor tCrB = thr_mma.partition_fragment_B(sB);
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  auto s2r_thr_copy_a =
      make_tiled_copy_A(Copy_Atom<DefaultCopy, cutlass::half_t>{},
                        tiled_mma)
          .get_thread_slice(threadIdx.x);
  Tensor tCsA = s2r_thr_copy_a.partition_S(sA);
  Tensor tCrA_copy_view = s2r_thr_copy_a.retile_D(tCrA);

  auto s2r_thr_copy_b =
      make_tiled_copy_B(Copy_Atom<DefaultCopy, cutlass::half_t>{},
                        tiled_mma)
          .get_thread_slice(threadIdx.x);
  Tensor tCsB = s2r_thr_copy_b.partition_S(sB);
  Tensor tCrB_copy_view = s2r_thr_copy_b.retile_D(tCrB);

  copy(tArA, tAsA);
  copy(tBrB, tBsB);
  __syncthreads();

  copy(tCsA(_, _, 0), tCrA_copy_view(_, _, 0));
  copy(tCsB(_, _, 0), tCrB_copy_view(_, _, 0));

  int k_tiles = k / CTA_K;
  auto k_blocks = size<2>(tCrA);
  CUTE_NO_UNROLL
  for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
    CUTE_UNROLL
    for (int k_block = 0; k_block < k_blocks; ++k_block) {
      if (k_block == k_blocks - 1) {
        __syncthreads();
        copy(tArA, tAsA);
        copy(tBrB, tBsB);
        __syncthreads();
      }

      int k_block_next = (k_block + 1) % k_blocks;
      copy(tCsA(_, _, k_block_next), tCrA_copy_view(_, _, k_block_next));
      copy(tCsB(_, _, k_block_next), tCrB_copy_view(_, _, k_block_next));

      if (k_block == 0) {
        int k_tile_next = k_tile + 1 < k_tiles ? k_tile + 1 : k_tile;
        copy(copy_a, tAgA(_, _, _, k_tile_next), tArA);
        copy(copy_b, tBgB(_, _, _, k_tile_next), tBrB);
      }

      gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
    }
  }

  CUTE_UNROLL
  for (int i = 0; i < size(tCrC); ++i) {
    tCgC(i) = static_cast<cutlass::half_t>(tCrC(i));
  }
}

template <int CTA_M, int CTA_N, int CTA_K, int Warps>
at::Tensor run_sm70_cute_gemm(const at::Tensor& a, const at::Tensor& b) {
  c10::cuda::CUDAGuard device_guard(a.device());
  at::Tensor out = at::empty({a.size(0), b.size(1)}, a.options());

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
struct Sm70ThreadblockWarpShape;

template <int CTA_K>
struct Sm70ThreadblockWarpShape<64, 64, CTA_K, 4> {
  using Type = cutlass::gemm::GemmShape<32, 32, CTA_K>;
};

template <int CTA_K>
struct Sm70ThreadblockWarpShape<64, 128, CTA_K, 4> {
  using Type = cutlass::gemm::GemmShape<32, 64, CTA_K>;
};

template <int CTA_K>
struct Sm70ThreadblockWarpShape<64, 128, CTA_K, 8> {
  using Type = cutlass::gemm::GemmShape<32, 32, CTA_K>;
};

template <int CTA_K>
struct Sm70ThreadblockWarpShape<64, 256, CTA_K, 8> {
  using Type = cutlass::gemm::GemmShape<32, 64, CTA_K>;
};

template <int CTA_K>
struct Sm70ThreadblockWarpShape<128, 64, CTA_K, 4> {
  using Type = cutlass::gemm::GemmShape<64, 32, CTA_K>;
};

template <int CTA_K>
struct Sm70ThreadblockWarpShape<128, 64, CTA_K, 8> {
  using Type = cutlass::gemm::GemmShape<32, 32, CTA_K>;
};

template <int CTA_K>
struct Sm70ThreadblockWarpShape<128, 128, CTA_K, 4> {
  using Type = cutlass::gemm::GemmShape<64, 64, CTA_K>;
};

template <int CTA_K>
struct Sm70ThreadblockWarpShape<128, 128, CTA_K, 8> {
  using Type = cutlass::gemm::GemmShape<64, 32, CTA_K>;
};

template <int CTA_K>
struct Sm70ThreadblockWarpShape<128, 256, CTA_K, 8> {
  using Type = cutlass::gemm::GemmShape<64, 64, CTA_K>;
};

template <int CTA_M, int CTA_N, int CTA_K, int Warps>
struct Sm70ThreadblockGemmTraits {
  static_assert(CTA_K == 32 || CTA_K == 64 || CTA_K == 128,
                "Extracted SM70 threadblock path supports K=32, 64, or 128");

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

#define DISPATCH_CUTE(CM, CN, CK, W)                                      \
  if (cta_m == CM && cta_n == CN && cta_k == CK && warps == W) {          \
    return run_sm70_cute_gemm<CM, CN, CK, W>(a, b);                       \
  }

#define DISPATCH_CUTE_K(CM, CN, W)                                        \
  DISPATCH_CUTE(CM, CN, 32, W)                                            \
  DISPATCH_CUTE(CM, CN, 64, W)                                            \
  DISPATCH_CUTE(CM, CN, 128, W)

#define DISPATCH_CUTE_N(CM, W)                                            \
  DISPATCH_CUTE_K(CM, 32, W)                                              \
  DISPATCH_CUTE_K(CM, 64, W)                                              \
  DISPATCH_CUTE_K(CM, 128, W)                                             \
  DISPATCH_CUTE_K(CM, 256, W)

#define DISPATCH_CUTE_M(W)                                                \
  DISPATCH_CUTE_N(8, W)                                                   \
  DISPATCH_CUTE_N(16, W)                                                  \
  DISPATCH_CUTE_N(32, W)                                                  \
  DISPATCH_CUTE_N(64, W)                                                  \
  DISPATCH_CUTE_N(128, W)

at::Tensor dispatch_sm70_cute_gemm(const at::Tensor& a, const at::Tensor& b,
                                   int64_t cta_m, int64_t cta_n,
                                   int64_t cta_k, int64_t warps) {
  DISPATCH_CUTE_M(4)
  DISPATCH_CUTE_M(8)

  TORCH_CHECK(false,
              "sm70_cutlass_matmul_probe: unsupported CUTLASS 3 CuTe config "
              "cta_m=", cta_m, ", cta_n=", cta_n, ", cta_k=", cta_k,
              ", warps=", warps);
}

#undef DISPATCH_CUTE_M
#undef DISPATCH_CUTE_N
#undef DISPATCH_CUTE_K
#undef DISPATCH_CUTE

#define DISPATCH_THREADBLOCK(CM, CN, CK, W)                               \
  if (cta_m == CM && cta_n == CN && cta_k == CK && warps == W) {          \
    return run_sm70_cutlass_threadblock_gemm<CM, CN, CK, W>(a, b);        \
  }

#define DISPATCH_THREADBLOCK_K(CM, CN, W)                                  \
  DISPATCH_THREADBLOCK(CM, CN, 32, W)                                      \
  DISPATCH_THREADBLOCK(CM, CN, 64, W)                                      \
  DISPATCH_THREADBLOCK(CM, CN, 128, W)

at::Tensor dispatch_sm70_cutlass_threadblock_gemm(
    const at::Tensor& a, const at::Tensor& b, int64_t cta_m, int64_t cta_n,
    int64_t cta_k, int64_t warps, int64_t /*b_path*/) {
  DISPATCH_THREADBLOCK_K(64, 64, 4)
  DISPATCH_THREADBLOCK_K(64, 128, 4)
  DISPATCH_THREADBLOCK_K(64, 128, 8)
  DISPATCH_THREADBLOCK_K(64, 256, 8)
  DISPATCH_THREADBLOCK_K(128, 64, 4)
  DISPATCH_THREADBLOCK_K(128, 64, 8)
  DISPATCH_THREADBLOCK_K(128, 128, 4)
  DISPATCH_THREADBLOCK_K(128, 128, 8)
  DISPATCH_THREADBLOCK_K(128, 256, 8)

  TORCH_CHECK(false,
              "sm70_cutlass_matmul_probe: unsupported extracted CUTLASS "
              "threadblock config cta_m=", cta_m, ", cta_n=", cta_n,
              ", cta_k=", cta_k, ", warps=", warps,
              ". Supported CTA_M/CTA_N/warps shapes are 64x64/4, "
              "64x128/4, 64x128/8, 64x256/8, 128x64/4, 128x64/8, "
              "128x128/4, 128x128/8, and 128x256/8 with CTA_K 32, "
              "64, or 128.");
}

#undef DISPATCH_THREADBLOCK_K
#undef DISPATCH_THREADBLOCK

at::Tensor sm70_cutlass_matmul_probe(const at::Tensor& a, const at::Tensor& b,
                                     int64_t cta_m, int64_t cta_n,
                                     int64_t cta_k, int64_t warps,
                                     int64_t stages, int64_t a_path,
                                     int64_t b_path) {
  check_probe_inputs(a, b, cta_m, cta_n, cta_k, warps, stages, a_path, b_path);
  if (a_path == kAPathCutlassThreadblock) {
    return dispatch_sm70_cutlass_threadblock_gemm(a, b, cta_m, cta_n, cta_k,
                                                  warps, b_path);
  }
  return dispatch_sm70_cute_gemm(a, b, cta_m, cta_n, cta_k, warps);
}

}  // namespace

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("sm70_cutlass_matmul_probe", &sm70_cutlass_matmul_probe);
}

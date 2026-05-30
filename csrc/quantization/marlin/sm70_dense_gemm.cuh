#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime_api.h>
#include <torch/library.h>

#include <type_traits>

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/layout/tensor_op_multiplicand_sm70.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "quantization/marlin/sm70_dense_common.cuh"

namespace marlin::sm70_dense {

template <typename Spec, int CtaM, int CtaN, int Warps, int GroupSize>
struct Sm70DenseGemmTraits {
  static_assert(CtaM == 32 || CtaM == 64 || CtaM == 128 || CtaM == 256,
                "SM70 dense supports CTA_M in {32, 64, 128, 256}.");
  static_assert(CtaN == 64 || CtaN == 128 || CtaN == 256,
                "SM70 dense supports CTA_N in {64, 128, 256}.");
  static_assert(Warps == 4 || Warps == 8,
                "SM70 dense supports 4 or 8 warps.");
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using ThreadblockShape = cutlass::gemm::GemmShape<CtaM, CtaN, kCtaK>;
  using WarpShape = typename Sm70DenseWarpShape<CtaM, CtaN, Warps>::Type;
  static_assert(WarpShape::kM <= 64 && WarpShape::kN <= 64,
                "SM70 dense keeps per-warp M/N no larger than 64.");
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using MmaCore = cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      cutlass::arch::OpClassTensorOp, 2, cutlass::arch::OpMultiplyAdd>;
  static_assert(MmaCore::kThreads == Warps * 32,
                "SM70 dense launch threads must match CUTLASS warp count.");
  using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA,
      128 / cutlass::sizeof_bits<ElementA>::value>;
  using IteratorB = typename Spec::template IteratorB<
      ThreadblockShape, typename MmaCore::IteratorThreadMapB, GroupSize>;
  using Mma = cutlass::gemm::threadblock::MmaPipelined<
      ThreadblockShape, IteratorA, typename MmaCore::SmemIteratorA, IteratorB,
      typename MmaCore::SmemIteratorB, ElementAccumulator, LayoutC,
      typename MmaCore::MmaPolicy>;
  using OutputOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, ElementAccumulator>;
  using ExpectedSmemLayoutB =
      cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<
          cutlass::sizeof_bits<ElementB>::value>;
  using ActualSmemLayoutB = typename Mma::SmemIteratorB::Layout;
  static_assert(std::is_same<ActualSmemLayoutB, ExpectedSmemLayoutB>::value,
                "SM70 dense B operand must use CUTLASS' predefined Volta "
                "B-congruous shared-memory layout.");
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

template <typename SharedStorage, typename Kernel>
inline size_t configure_dynamic_smem(Kernel kernel) {
  size_t smem_bytes = sizeof(SharedStorage);
  if (smem_bytes >= (48u << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
  }
  return smem_bytes;
}

inline dim3 cta_grid(int64_t size_m, int64_t size_n, int cta_m, int cta_n) {
  return dim3(static_cast<unsigned>((size_m + cta_m - 1) / cta_m),
              static_cast<unsigned>(size_n / cta_n));
}

template <int CtaM, int CtaN, int Warps, typename Launcher>
torch::Tensor dispatch_group_size(Launcher const& launcher,
                                  int64_t group_size,
                                  char const* quant_name) {
  switch (group_size) {
    case -1:
      return launcher.template operator()<CtaM, CtaN, Warps, -1>();
    case 32:
      return launcher.template operator()<CtaM, CtaN, Warps, 32>();
    case 64:
      return launcher.template operator()<CtaM, CtaN, Warps, 64>();
    case 128:
      return launcher.template operator()<CtaM, CtaN, Warps, 128>();
    default:
      TORCH_CHECK(false, "SM70 CUTLASS ", quant_name,
                  " prototype supports only group_size -1, 32, 64, or 128. "
                  "Got ",
                  group_size);
  }
  return torch::Tensor();
}

template <typename Launcher>
torch::Tensor dispatch_geometry(Launcher const& launcher,
                                Sm70DenseCtaGeometry geometry,
                                int64_t /*size_n*/, int64_t /*size_k*/,
                                int64_t group_size,
                                char const* quant_name) {
#define DISPATCH_SM70_DENSE_CTA(CM, CN, W)                               \
  if (geometry.cta_m == CM && geometry.cta_n == CN &&                     \
      geometry.warps == W) {                                              \
    return dispatch_group_size<CM, CN, W>(launcher, group_size,            \
                                          quant_name);                     \
  }

  DISPATCH_SM70_DENSE_CTA(32, 128, 4)
  DISPATCH_SM70_DENSE_CTA(32, 256, 4)
  DISPATCH_SM70_DENSE_CTA(64, 64, 4)
  DISPATCH_SM70_DENSE_CTA(64, 128, 4)
  DISPATCH_SM70_DENSE_CTA(64, 128, 8)
  DISPATCH_SM70_DENSE_CTA(64, 256, 4)
  DISPATCH_SM70_DENSE_CTA(64, 256, 8)
  DISPATCH_SM70_DENSE_CTA(128, 64, 4)
  DISPATCH_SM70_DENSE_CTA(128, 64, 8)
  DISPATCH_SM70_DENSE_CTA(128, 128, 4)
  DISPATCH_SM70_DENSE_CTA(128, 128, 8)
  DISPATCH_SM70_DENSE_CTA(128, 256, 8)
  DISPATCH_SM70_DENSE_CTA(256, 64, 4)
  DISPATCH_SM70_DENSE_CTA(256, 64, 8)
  DISPATCH_SM70_DENSE_CTA(256, 128, 8)

#undef DISPATCH_SM70_DENSE_CTA

  TORCH_CHECK(false, "Unreachable SM70 ", quant_name,
              " CTA geometry dispatch.");
}

}  // namespace marlin::sm70_dense

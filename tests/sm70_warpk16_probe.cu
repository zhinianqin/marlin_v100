#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Standalone SM70 explicit-warp-shape unit/probe.
//
// Build without compiling the full project:
//
//   mkdir -p build/sm70_probe
//   /usr/local/cuda-12.8/bin/nvcc -std=c++17 -O2 -arch=sm_70 \
//     -I/root/source/repos/cutlass/include -Icsrc \
//     -I/usr/local/cuda-12.8/include \
//     tests/sm70_warpk16_probe.cu \
//     -o build/sm70_probe/sm70_warpk16_probe
//
// Usage:
//
//   ./build/sm70_probe/sm70_warpk16_probe --smoke
//   ./build/sm70_probe/sm70_warpk16_probe --diagnose-warpk16
//   ./build/sm70_probe/sm70_warpk16_probe \
//     32x64x32x4x32x32x16 single atomic 32 64 32
//   ./build/sm70_probe/sm70_warpk16_probe \
//     32x64x32x4x32x32x16 single full k0 32 64 32
//   ./build/sm70_probe/sm70_warpk16_probe \
//     32x64x32x4x32x32x16 single atomic_noreduce k1 32 64 32
//
// The suite intentionally avoids stock CUTLASS illegal-address repro cases so
// CUDA context poisoning stays isolated to explicit one-case invocations.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/functional.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/half.h"
#include "cutlass/layout/tensor_op_multiplicand_sm70.h"
#include "quantization/marlin/sm70_marlin_mma.cuh"

namespace {

enum class RunStatus {
  kPass = 0,
  kUnsupported = 1,
  kMismatch = 2,
  kCudaError = 3,
};

#define CUDA_CHECK_BOOL(call)                                                \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "  \
                << cudaGetErrorString(status) << std::endl;                  \
      return false;                                                          \
    }                                                                        \
  } while (0)

#define CUDA_CHECK_STATUS(call)                                              \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "  \
                << cudaGetErrorString(status) << std::endl;                  \
      return RunStatus::kCudaError;                                          \
    }                                                                        \
  } while (0)

struct Geometry {
  int cta_m = 0;
  int cta_n = 0;
  int cta_k = 0;
  int warps = 0;
  int warp_m = 0;
  int warp_n = 0;
  int warp_k = 0;
};

enum class ProbeMode {
  kFull,
  kMainloop,
  kEpilogue,
  kAtomic,
  kAtomicNoReduce,
  kDirectPartitions,
};

enum class MainloopKind {
  kStock,
  kCustom,
  kSingle,
  kSinglePhase,
};

enum class InputPattern {
  kAll,
  kPartition0Only,
  kPartition1Only,
};

enum class KGroupMode {
  kLocal,
  kGlobal,
};

enum class ExpectedOutcome {
  kPass,
  kMismatch,
};

bool parse_mainloop_kind(char const* text, MainloopKind& kind) {
  if (std::strcmp(text, "stock") == 0) {
    kind = MainloopKind::kStock;
    return true;
  }
  if (std::strcmp(text, "custom") == 0) {
    kind = MainloopKind::kCustom;
    return true;
  }
  if (std::strcmp(text, "single") == 0) {
    kind = MainloopKind::kSingle;
    return true;
  }
  if (std::strcmp(text, "single_phase") == 0) {
    kind = MainloopKind::kSinglePhase;
    return true;
  }
  return false;
}

char const* mainloop_name(MainloopKind kind) {
  switch (kind) {
    case MainloopKind::kStock:
      return "stock";
    case MainloopKind::kCustom:
      return "custom";
    case MainloopKind::kSingle:
      return "single";
    case MainloopKind::kSinglePhase:
      return "single_phase";
  }
  return "unknown";
}

bool parse_input_pattern(char const* text, InputPattern& pattern) {
  if (std::strcmp(text, "all") == 0) {
    pattern = InputPattern::kAll;
    return true;
  }
  if (std::strcmp(text, "k0") == 0) {
    pattern = InputPattern::kPartition0Only;
    return true;
  }
  if (std::strcmp(text, "k1") == 0) {
    pattern = InputPattern::kPartition1Only;
    return true;
  }
  return false;
}

bool parse_kgroup_mode(char const* text, KGroupMode& mode) {
  if (std::strcmp(text, "local") == 0) {
    mode = KGroupMode::kLocal;
    return true;
  }
  if (std::strcmp(text, "global") == 0) {
    mode = KGroupMode::kGlobal;
    return true;
  }
  return false;
}

char const* kgroup_mode_name(KGroupMode mode) {
  switch (mode) {
    case KGroupMode::kLocal:
      return "local";
    case KGroupMode::kGlobal:
      return "global";
  }
  return "unknown";
}

char const* input_pattern_name(InputPattern pattern) {
  switch (pattern) {
    case InputPattern::kAll:
      return "all";
    case InputPattern::kPartition0Only:
      return "k0";
    case InputPattern::kPartition1Only:
      return "k1";
  }
  return "unknown";
}

bool parse_mode(char const* text, ProbeMode& mode) {
  if (std::strcmp(text, "full") == 0) {
    mode = ProbeMode::kFull;
    return true;
  }
  if (std::strcmp(text, "mainloop") == 0) {
    mode = ProbeMode::kMainloop;
    return true;
  }
  if (std::strcmp(text, "epilogue") == 0) {
    mode = ProbeMode::kEpilogue;
    return true;
  }
  if (std::strcmp(text, "atomic") == 0) {
    mode = ProbeMode::kAtomic;
    return true;
  }
  if (std::strcmp(text, "atomic_noreduce") == 0) {
    mode = ProbeMode::kAtomicNoReduce;
    return true;
  }
  if (std::strcmp(text, "direct") == 0) {
    mode = ProbeMode::kDirectPartitions;
    return true;
  }
  return false;
}

char const* mode_name(ProbeMode mode) {
  switch (mode) {
    case ProbeMode::kFull:
      return "full";
    case ProbeMode::kMainloop:
      return "mainloop";
    case ProbeMode::kEpilogue:
      return "epilogue";
    case ProbeMode::kAtomic:
      return "atomic";
    case ProbeMode::kAtomicNoReduce:
      return "atomic_noreduce";
    case ProbeMode::kDirectPartitions:
      return "direct";
  }
  return "unknown";
}

bool parse_geometry(char const* text, Geometry& geometry) {
  int fields[7] = {};
  char tail = 0;
  int const parsed = std::sscanf(text, "%dx%dx%dx%dx%dx%dx%d%c",
                                 &fields[0], &fields[1], &fields[2],
                                 &fields[3], &fields[4], &fields[5],
                                 &fields[6], &tail);
  if (parsed != 7) {
    return false;
  }
  geometry = {fields[0], fields[1], fields[2], fields[3],
              fields[4], fields[5], fields[6]};
  return true;
}

std::string geometry_label(Geometry geometry) {
  return std::to_string(geometry.cta_m) + "x" +
         std::to_string(geometry.cta_n) + "x" +
         std::to_string(geometry.cta_k) + "x" +
         std::to_string(geometry.warps) + "x" +
         std::to_string(geometry.warp_m) + "x" +
         std::to_string(geometry.warp_n) + "x" +
         std::to_string(geometry.warp_k);
}

bool print_device() {
  int device = 0;
  CUDA_CHECK_BOOL(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK_BOOL(cudaGetDeviceProperties(&prop, device));
  std::cout << "device=" << prop.name << " capability=" << prop.major << "."
            << prop.minor << std::endl;
  return true;
}

template <int CTA_M, int CTA_N, int CTA_K, int WarpM, int WarpN, int WarpK,
          int SmemAKBlock>
struct ExplicitRowRowMmaCore {
  using Shape = cutlass::gemm::GemmShape<CTA_M, CTA_N, CTA_K>;
  using WarpShape = cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using ElementA = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = cutlass::half_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = cutlass::layout::RowMajor;

  using WarpCount = cutlass::gemm::GemmShape<
      Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN,
      Shape::kK / WarpShape::kK>;

  static int const kWarpSize =
      cutlass::gemm::warp::WarpSize<cutlass::arch::OpClassTensorOp>::value;
  static int const kThreads = WarpCount::kCount * kWarpSize;
  static int const kAccessSizeInBits = 128;

  using SmemLayoutA = cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<ElementA>::value, SmemAKBlock>;
  using SmemLayoutB =
      cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<
          cutlass::sizeof_bits<ElementB>::value>;

  using IteratorThreadMapA = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
      cutlass::layout::PitchLinearShape<4, 8>,
      kAccessSizeInBits / cutlass::sizeof_bits<ElementA>::value>;

  using SmemIteratorA = cutlass::transform::threadblock::RegularTileIterator<
      cutlass::MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
      IteratorThreadMapA>;

  using IteratorThreadMapB = cutlass::transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<Shape::kN, Shape::kK>, kThreads,
      cutlass::layout::PitchLinearShape<8, 4>,
      kAccessSizeInBits / cutlass::sizeof_bits<ElementB>::value>;

  using SmemIteratorB = cutlass::transform::threadblock::RegularTileIterator<
      cutlass::MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
      IteratorThreadMapB>;

  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 16, 4>, 32, ElementA,
                         LayoutA, ElementB, LayoutB, ElementC,
                         cutlass::layout::RowMajor,
                         cutlass::arch::OpMultiplyAdd>,
      cutlass::MatrixShape<1, 1>>;

  using MmaTensorOp = cutlass::gemm::warp::MmaVoltaTensorOp<
      WarpShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB, ElementC,
      LayoutC, Policy>;

  using MmaPolicy = cutlass::gemm::threadblock::MmaPolicy<
      MmaTensorOp, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>,
      WarpCount::kK>;
};

template <
    typename Shape_, typename IteratorA_, typename SmemIteratorA_,
    typename IteratorB_, typename SmemIteratorB_, typename ElementC_,
    typename LayoutC_, typename Policy_,
    typename TransformA_ = cutlass::NumericArrayConverter<
        typename SmemIteratorA_::Element, typename IteratorA_::Element,
        IteratorA_::Fragment::kElements>,
    typename TransformB_ = cutlass::NumericArrayConverter<
        typename SmemIteratorB_::Element, typename IteratorB_::Element,
        IteratorB_::Fragment::kElements>,
    bool UseGlobalKGroup = false,
    bool UsePhaseAwareInitialKOffset = false>
class ProbeSm70MarlinMmaSingleStage
    : public cutlass::gemm::threadblock::MmaBase<Shape_, Policy_, 1> {
 public:
  using Base = cutlass::gemm::threadblock::MmaBase<Shape_, Policy_, 1>;

  using Shape = Shape_;
  using IteratorA = IteratorA_;
  using IteratorB = IteratorB_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using Policy = Policy_;
  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;

  using FragmentA = typename IteratorA::Fragment;
  using FragmentB = typename IteratorB::Fragment;
  using FragmentC = typename Policy::Operator::FragmentC;
  using Operator = typename Policy::Operator;

  static_assert(Base::kStages == 1,
                "ProbeSm70MarlinMmaSingleStage requires one stage.");

 private:
  using WarpFragmentA = typename Operator::FragmentA;
  using WarpFragmentB = typename Operator::FragmentB;

 protected:
  SmemIteratorA smem_iterator_A_;
  SmemIteratorB smem_iterator_B_;
  TransformA transform_A_;
  TransformB transform_B_;
  int warp_idx_k_;

 public:
  CUTLASS_DEVICE
  ProbeSm70MarlinMmaSingleStage(
      typename Base::SharedStorage& shared_storage, int thread_idx,
      int warp_idx, int lane_idx, TransformA transform_A = TransformA(),
      TransformB transform_B = TransformB())
      : Base(shared_storage, thread_idx, warp_idx, lane_idx),
        smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
        smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
        transform_A_(transform_A),
        transform_B_(transform_B),
        warp_idx_k_(0) {
    int const warp_idx_mn =
        warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    warp_idx_k_ = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int const warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int const warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    if constexpr (UsePhaseAwareInitialKOffset) {
      this->warp_tile_iterator_A_.add_tile_offset({warp_idx_m, 0});
      this->warp_tile_iterator_B_.add_tile_offset({0, warp_idx_n});

      // Volta crosswise warp iterators maintain both pointer_ and an internal
      // byte_offset_ phase while advancing across k-groups. add_tile_offset()
      // moves pointer_ directly and resets k_group_idx_, so use operator++()
      // here to test whether WarpK=16 partition1 needs the missing phase flip.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Base::kWarpGemmIterations * warp_idx_k_; ++i) {
        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;
      }
    } else {
      this->warp_tile_iterator_A_.add_tile_offset(
          {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k_});
      this->warp_tile_iterator_B_.add_tile_offset(
          {Base::kWarpGemmIterations * warp_idx_k_, warp_idx_n});
    }
  }

  CUTLASS_DEVICE
  void operator()(int gemm_k_iterations, FragmentC& accum,
                  IteratorA iterator_A, IteratorB iterator_B,
                  FragmentC const& src_accum) {
    accum = src_accum;

    FragmentA tb_frag_A;
    FragmentB tb_frag_B;

    tb_frag_A.clear();
    tb_frag_B.clear();

    iterator_A.load(tb_frag_A);
    iterator_B.load(tb_frag_B);

    ++iterator_A;
    ++iterator_B;

    WarpFragmentA warp_frag_A;
    WarpFragmentB warp_frag_B;

    Operator warp_mma;

    iterator_A.clear_mask(gemm_k_iterations <= 1);
    iterator_B.clear_mask(gemm_k_iterations <= 1);

    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      this->smem_iterator_A_.store(transform_A_(tb_frag_A));
      this->smem_iterator_B_.store(transform_B_(tb_frag_B));

      __syncthreads();

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
           ++warp_mma_k) {
        int const k_group =
            UseGlobalKGroup
                ? (warp_idx_k_ * Base::kWarpGemmIterations + warp_mma_k)
                : warp_mma_k;
        this->warp_tile_iterator_A_.set_kgroup_index(k_group);
        this->warp_tile_iterator_B_.set_kgroup_index(k_group);

        this->warp_tile_iterator_A_.load(warp_frag_A);
        this->warp_tile_iterator_B_.load(warp_frag_B);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        warp_mma(accum, warp_frag_A, warp_frag_B, accum);
      }

      this->warp_tile_iterator_A_.add_tile_offset(
          {0, -Base::kWarpGemmIterations});
      this->warp_tile_iterator_B_.add_tile_offset(
          {-Base::kWarpGemmIterations, 0});

      __syncthreads();

      tb_frag_A.clear();
      tb_frag_B.clear();

      iterator_A.load(tb_frag_A);
      iterator_B.load(tb_frag_B);

      ++iterator_A;
      ++iterator_B;

      iterator_A.clear_mask(gemm_k_iterations <= 2);
      iterator_B.clear_mask(gemm_k_iterations <= 2);
    }
  }
};

template <int CTA_M, int CTA_N, int CTA_K, int Warps, int WarpM, int WarpN,
          int WarpK, int SmemAKBlock>
struct ExplicitWarpGemmTraits {
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
  using MmaCore =
      ExplicitRowRowMmaCore<CTA_M, CTA_N, CTA_K, WarpM, WarpN, WarpK,
                            SmemAKBlock>;

  static_assert((CTA_M / WarpM) * (CTA_N / WarpN) * (CTA_K / WarpK) == Warps,
                "Explicit warp shape must decompose the CTA into Warps.");

  using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>, ElementA,
      LayoutA, 1, typename MmaCore::IteratorThreadMapA,
      128 / cutlass::sizeof_bits<ElementA>::value>;
  using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>, ElementB,
      LayoutB, 0, typename MmaCore::IteratorThreadMapB,
      128 / cutlass::sizeof_bits<ElementB>::value>;

  using Mma = cutlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator, LayoutC,
      typename MmaCore::MmaPolicy>;
  using CustomMma = marlin::sm70::Sm70MarlinMmaPipelined<
      ThreadblockShape, typename Mma::IteratorA, typename Mma::SmemIteratorA,
      typename Mma::IteratorB, typename Mma::SmemIteratorB, ElementAccumulator,
      LayoutC, typename Mma::Policy>;
  using SingleMma = ProbeSm70MarlinMmaSingleStage<
      ThreadblockShape, typename Mma::IteratorA, typename Mma::SmemIteratorA,
      typename Mma::IteratorB, typename Mma::SmemIteratorB, ElementAccumulator,
      LayoutC, typename Mma::Policy, cutlass::NumericArrayConverter<
                                        typename Mma::SmemIteratorA::Element,
                                        typename Mma::IteratorA::Element,
                                        Mma::IteratorA::Fragment::kElements>,
      cutlass::NumericArrayConverter<typename Mma::SmemIteratorB::Element,
                                     typename Mma::IteratorB::Element,
                                     Mma::IteratorB::Fragment::kElements>,
      false>;
  using GlobalKGroupSingleMma = ProbeSm70MarlinMmaSingleStage<
      ThreadblockShape, typename Mma::IteratorA, typename Mma::SmemIteratorA,
      typename Mma::IteratorB, typename Mma::SmemIteratorB, ElementAccumulator,
      LayoutC, typename Mma::Policy, cutlass::NumericArrayConverter<
                                        typename Mma::SmemIteratorA::Element,
                                        typename Mma::IteratorA::Element,
                                        Mma::IteratorA::Fragment::kElements>,
      cutlass::NumericArrayConverter<typename Mma::SmemIteratorB::Element,
                                     typename Mma::IteratorB::Element,
                                     Mma::IteratorB::Fragment::kElements>,
      true, false>;
  using PhaseAwareSingleMma = ProbeSm70MarlinMmaSingleStage<
      ThreadblockShape, typename Mma::IteratorA, typename Mma::SmemIteratorA,
      typename Mma::IteratorB, typename Mma::SmemIteratorB, ElementAccumulator,
      LayoutC, typename Mma::Policy, cutlass::NumericArrayConverter<
                                        typename Mma::SmemIteratorA::Element,
                                        typename Mma::IteratorA::Element,
                                        Mma::IteratorA::Fragment::kElements>,
      cutlass::NumericArrayConverter<typename Mma::SmemIteratorB::Element,
                                     typename Mma::IteratorB::Element,
                                     Mma::IteratorB::Fragment::kElements>,
      false, true>;
  using Operator = typename Mma::Operator;

  using ExpectedSmemLayoutA =
      cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<
          cutlass::sizeof_bits<ElementA>::value, SmemAKBlock>;
  using ExpectedSmemLayoutB =
      cutlass::layout::RowMajorVoltaTensorOpMultiplicandBCongruous<
          cutlass::sizeof_bits<ElementB>::value>;
  using ActualSmemLayoutA = typename Mma::SmemIteratorA::Layout;
  using ActualSmemLayoutB = typename Mma::SmemIteratorB::Layout;
  static_assert(std::is_same<ActualSmemLayoutA, ExpectedSmemLayoutA>::value,
                "Unexpected A shared-memory layout.");
  static_assert(std::is_same<ActualSmemLayoutB, ExpectedSmemLayoutB>::value,
                "Unexpected B shared-memory layout.");

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator, ElementAccumulator>;

  static int const kPartitionsK = ThreadblockShape::kK / WarpShape::kK;
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
          ThreadblockShape, Operator, kPartitionsK, OutputOp,
          OutputOp::kCount>::Epilogue;

};

// Local copy of the fp16 atomic epilogue used only by this standalone probe.
// Keeping it here avoids including sm70_marlin_splitk.cuh, which intentionally
// depends on PyTorch/C10 helpers in the production extension build.
template <typename Traits, bool ReducePartitionsK>
class ProbeAtomicFp16Epilogue {
 public:
  using CutlassEpilogue = typename Traits::Epilogue;
  using SharedStorage = typename CutlassEpilogue::Base::SharedStorage;
  using AccumulatorTile = typename CutlassEpilogue::AccumulatorTile;
  using AccumulatorFragmentIterator =
      typename CutlassEpilogue::AccumulatorFragmentIterator;
  using WarpTileIterator = typename CutlassEpilogue::WarpTileIterator;
  using SharedLoadIterator = typename CutlassEpilogue::SharedLoadIterator;
  using OutputTileIterator = typename CutlassEpilogue::OutputTileIterator;
  using ThreadMap = typename OutputTileIterator::ThreadMap;

 private:
  WarpTileIterator warp_tile_iterator_;
  SharedLoadIterator shared_load_iterator_;
  int warp_k_;

  CUTLASS_DEVICE
  void atomic_store_fragment(OutputTileIterator const& destination_iterator,
                             typename SharedLoadIterator::Fragment const& frag,
                             cutlass::half_t* __restrict__ c, int n) const {
    float const* frag_ptr = reinterpret_cast<float const*>(&frag);
    half* c_half = reinterpret_cast<half*>(c);
    int const thread_start_row = destination_iterator.thread_start_row();
    int const thread_start_column = destination_iterator.thread_start_column();
    int const extent_row = destination_iterator.extent_row();

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
         ++cluster) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int const frag_row_idx =
              row + ThreadMap::Iterations::kRow *
                        (group + ThreadMap::Iterations::kGroup * cluster);
          int const row_offset =
              row * ThreadMap::Delta::kRow +
              group * ThreadMap::Delta::kGroup +
              cluster * ThreadMap::Delta::kCluster;
          int const logical_row = thread_start_row + row_offset;

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn;
               ++column) {
            int const logical_column_base =
                thread_start_column + column * ThreadMap::Delta::kColumn;
            int const frag_base =
                (frag_row_idx * ThreadMap::Iterations::kColumn + column) *
                ThreadMap::kElementsPerAccess;

            if (logical_row < extent_row) {
              CUTLASS_PRAGMA_UNROLL
              for (int e = 0; e < ThreadMap::kElementsPerAccess; ++e) {
                int64_t const offset =
                    int64_t(logical_row) * n + logical_column_base + e;
                atomicAdd(c_half + offset,
                          __float2half_rn(frag_ptr[frag_base + e]));
              }
            }
          }
        }
      }
    }
  }

 public:
  CUTLASS_DEVICE
  ProbeAtomicFp16Epilogue(SharedStorage& shared_storage, int thread_idx,
                          int warp_idx, int lane_idx)
      : warp_tile_iterator_(shared_storage.reference(), lane_idx),
        shared_load_iterator_(shared_storage.reference(), thread_idx),
        warp_k_(0) {
    using WarpCount = typename CutlassEpilogue::WarpCount;
    int const warp_k = warp_idx / (WarpCount::kM * WarpCount::kN);
    int const warp_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
    int const warp_m = warp_mn % WarpCount::kM;
    int const warp_n = warp_mn / WarpCount::kM;
    warp_k_ = warp_k;

    cutlass::MatrixCoord warp_offset{warp_k * WarpCount::kM + warp_m,
                                     warp_n};
    warp_tile_iterator_.add_tile_offset(warp_offset);
  }

  CUTLASS_DEVICE
  void operator()(OutputTileIterator destination_iterator,
                  AccumulatorTile const& accumulators,
                  cutlass::half_t* __restrict__ c, int n) {
    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
      __syncthreads();

      typename AccumulatorFragmentIterator::Fragment accum_fragment;
      accum_fragment_iterator.load(accum_fragment);
      ++accum_fragment_iterator;
      warp_tile_iterator_.store(accum_fragment);

      __syncthreads();

      typename SharedLoadIterator::Fragment aligned_accum_fragment;
      if constexpr (!ReducePartitionsK) {
        shared_load_iterator_.add_pointer_offset(
            warp_k_ * CutlassEpilogue::kSmemPointerOffset);
      }
      shared_load_iterator_.load(aligned_accum_fragment);
      if constexpr (!ReducePartitionsK) {
        shared_load_iterator_.add_pointer_offset(
            -warp_k_ * CutlassEpilogue::kSmemPointerOffset);
      }

      if constexpr (ReducePartitionsK && CutlassEpilogue::kPartitionsK > 1) {
        cutlass::plus<typename SharedLoadIterator::Fragment> add_fragments;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < CutlassEpilogue::kPartitionsK; ++i) {
          typename SharedLoadIterator::Fragment aligned_addend_fragment;
          shared_load_iterator_.add_pointer_offset(
              CutlassEpilogue::kSmemPointerOffset);
          shared_load_iterator_.load(aligned_addend_fragment);
          aligned_accum_fragment =
              add_fragments(aligned_accum_fragment, aligned_addend_fragment);
        }

        shared_load_iterator_.add_pointer_offset(
            (1 - CutlassEpilogue::kPartitionsK) *
            CutlassEpilogue::kSmemPointerOffset);
      }

      atomic_store_fragment(destination_iterator, aligned_accum_fragment, c,
                            n);
      ++destination_iterator;
    }
  }
};

template <int CTA_M, int CTA_N, int CTA_K, int Warps, int WarpM, int WarpN,
          int WarpK, int SmemAKBlock, bool UseCustomMma, bool UseSingleMma,
          bool UsePhaseAwareInitialKOffset, bool UseGlobalKGroup, bool DoMma,
          bool DoEpilogue, bool UseAtomicEpilogue,
          bool ReduceAtomicPartitionsK, bool DirectPartitions>
__global__ __launch_bounds__(Warps * 32, 1)
void explicit_warp_kernel(cutlass::half_t const* __restrict__ a,
                          cutlass::half_t const* __restrict__ b,
                          cutlass::half_t* __restrict__ c, int m, int n,
                          int k) {
  using Traits = ExplicitWarpGemmTraits<CTA_M, CTA_N, CTA_K, Warps, WarpM,
                                        WarpN, WarpK, SmemAKBlock>;
  using PipelinedMma =
      typename std::conditional<UseCustomMma, typename Traits::CustomMma,
                                typename Traits::Mma>::type;
  using LocalSingleMma = typename std::conditional<
      UsePhaseAwareInitialKOffset, typename Traits::PhaseAwareSingleMma,
      typename Traits::SingleMma>::type;
  using LocalSingleOrPipelined =
      typename std::conditional<UseSingleMma, LocalSingleMma, PipelinedMma>::type;
  using Mma = typename std::conditional<
      UseSingleMma && UseGlobalKGroup, typename Traits::GlobalKGroupSingleMma,
      LocalSingleOrPipelined>::type;
  using Epilogue = typename Traits::Epilogue;
  using AtomicEpilogue =
      ProbeAtomicFp16Epilogue<Traits, ReduceAtomicPartitionsK>;

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  extern __shared__ char smem[];
  auto& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

  int const thread_idx = threadIdx.x;
  int const warp_idx = cutlass::canonical_warp_idx_sync();
  int const lane_idx = threadIdx.x % 32;
  using WarpCount = typename Epilogue::WarpCount;
  int const warp_k = warp_idx / (WarpCount::kM * WarpCount::kN);

  cutlass::MatrixCoord tb_offset_A{int(blockIdx.x) * CTA_M, 0};
  cutlass::MatrixCoord tb_offset_B{0, int(blockIdx.y) * CTA_N};
  cutlass::MatrixCoord tb_offset_C{int(blockIdx.x) * CTA_M,
                                   int(blockIdx.y) * CTA_N};

  typename Traits::LayoutA layout_a(k);
  typename Traits::LayoutB layout_b(n);
  typename Traits::LayoutC layout_c(n);

  typename Mma::IteratorA iterator_A(
      typename Mma::IteratorA::Params(layout_a),
      const_cast<cutlass::half_t*>(a), cutlass::MatrixCoord(m, k),
      thread_idx, tb_offset_A);
  typename Mma::IteratorB iterator_B(
      typename Mma::IteratorB::Params(layout_b),
      const_cast<cutlass::half_t*>(b), cutlass::MatrixCoord(k, n),
      thread_idx, tb_offset_B);

  Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
  typename Mma::FragmentC accumulators;
  accumulators.clear();

  if constexpr (DoMma) {
    int gemm_k_iterations = (k + CTA_K - 1) / CTA_K;
    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
  }

  if constexpr (UseAtomicEpilogue) {
    typename AtomicEpilogue::OutputTileIterator iterator_D(
        typename AtomicEpilogue::OutputTileIterator::Params(layout_c), c,
        cutlass::MatrixCoord(m, n), thread_idx, tb_offset_C);

    AtomicEpilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx,
                            lane_idx);
    epilogue(iterator_D, accumulators, c, n);
  } else if constexpr (DirectPartitions) {
    float* c_float = reinterpret_cast<float*>(c);
    cutlass::TensorRef<float, cutlass::layout::RowMajor> ref_c(
        c_float + int64_t(warp_k) * int64_t(m) * int64_t(n),
        cutlass::layout::RowMajor(n));
    typename Traits::Operator::IteratorC iterator_C(ref_c, lane_idx);
    int const warp_idx_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
    int const warp_m = warp_idx_mn % WarpCount::kM;
    int const warp_n = warp_idx_mn / WarpCount::kM;
    iterator_C.add_tile_offset({int(blockIdx.x) * (CTA_M / WarpM) + warp_m,
                                int(blockIdx.y) * (CTA_N / WarpN) + warp_n});
    iterator_C.store(accumulators);
  } else if constexpr (DoEpilogue) {
    typename Traits::OutputOp output_op({1.0f, 0.0f});
    typename Epilogue::OutputTileIterator iterator_C(
        typename Epilogue::OutputTileIterator::Params(layout_c), c,
        cutlass::MatrixCoord(m, n), thread_idx, tb_offset_C);
    typename Epilogue::OutputTileIterator iterator_D(
        typename Epilogue::OutputTileIterator::Params(layout_c), c,
        cutlass::MatrixCoord(m, n), thread_idx, tb_offset_C);

    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(output_op, iterator_D, accumulators, iterator_C);
  } else if (thread_idx == 0) {
    c[size_t(blockIdx.x) * size_t(CTA_M) * size_t(n) +
      size_t(blockIdx.y) * size_t(CTA_N)] = cutlass::half_t(0.0f);
  }
}

template <int CTA_M, int CTA_N, int CTA_K, int Warps, int WarpM, int WarpN,
          int WarpK, int SmemAKBlock, bool UseCustomMma, bool UseSingleMma,
          bool UsePhaseAwareInitialKOffset = false>
RunStatus run_geometry_impl(int m, int n, int k, ProbeMode mode,
                            MainloopKind mainloop, InputPattern pattern,
                            KGroupMode kgroup_mode) {
  using Traits = ExplicitWarpGemmTraits<CTA_M, CTA_N, CTA_K, Warps, WarpM,
                                        WarpN, WarpK, SmemAKBlock>;
  using PipelinedMma =
      typename std::conditional<UseCustomMma, typename Traits::CustomMma,
                                typename Traits::Mma>::type;
  using Mma =
      typename std::conditional<UseSingleMma, typename Traits::SingleMma,
                                PipelinedMma>::type;
  using Epilogue = typename Traits::Epilogue;
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  std::vector<cutlass::half_t> host_a(size_t(m) * size_t(k));
  std::vector<cutlass::half_t> host_b(size_t(k) * size_t(n));
  size_t const output_partitions =
      mode == ProbeMode::kDirectPartitions ? Traits::kPartitionsK : 1;
  std::vector<cutlass::half_t> host_c(size_t(output_partitions) * size_t(m) *
                                      size_t(n));
  std::vector<float> host_c_float(size_t(output_partitions) * size_t(m) *
                                  size_t(n));
  std::vector<float> reference(size_t(m) * size_t(n), 0.0f);

  std::mt19937 rng(17);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& value : host_a) {
    value = cutlass::half_t(dist(rng));
  }
  for (auto& value : host_b) {
    value = cutlass::half_t(dist(rng));
  }

  auto keep_k = [pattern](int kk) {
    if (pattern == InputPattern::kAll) {
      return true;
    }
    if (pattern == InputPattern::kPartition0Only) {
      return kk < WarpK;
    }
    return kk >= WarpK && kk < 2 * WarpK;
  };
  if (pattern != InputPattern::kAll) {
    for (int row = 0; row < m; ++row) {
      for (int kk = 0; kk < k; ++kk) {
        if (!keep_k(kk)) {
          host_a[size_t(row) * size_t(k) + kk] = cutlass::half_t(0.0f);
        }
      }
    }
    for (int kk = 0; kk < k; ++kk) {
      for (int col = 0; col < n; ++col) {
        if (!keep_k(kk)) {
          host_b[size_t(kk) * size_t(n) + col] = cutlass::half_t(0.0f);
        }
      }
    }
  }

  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float accum = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        accum += float(host_a[size_t(row) * size_t(k) + kk]) *
                 float(host_b[size_t(kk) * size_t(n) + col]);
      }
      reference[size_t(row) * size_t(n) + col] = accum;
    }
  }

  cutlass::half_t* dev_a = nullptr;
  cutlass::half_t* dev_b = nullptr;
  cutlass::half_t* dev_c = nullptr;
  size_t const output_bytes =
      mode == ProbeMode::kDirectPartitions
          ? host_c_float.size() * sizeof(float)
          : host_c.size() * sizeof(cutlass::half_t);
  CUDA_CHECK_STATUS(cudaMalloc(&dev_a, host_a.size() * sizeof(cutlass::half_t)));
  CUDA_CHECK_STATUS(cudaMalloc(&dev_b, host_b.size() * sizeof(cutlass::half_t)));
  CUDA_CHECK_STATUS(cudaMalloc(&dev_c, output_bytes));
  CUDA_CHECK_STATUS(cudaMemcpy(dev_a, host_a.data(),
                               host_a.size() * sizeof(cutlass::half_t),
                               cudaMemcpyHostToDevice));
  CUDA_CHECK_STATUS(cudaMemcpy(dev_b, host_b.data(),
                               host_b.size() * sizeof(cutlass::half_t),
                               cudaMemcpyHostToDevice));
  CUDA_CHECK_STATUS(cudaMemset(dev_c, 0, output_bytes));

  auto kernel = explicit_warp_kernel<CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN,
                                     WarpK, SmemAKBlock, UseCustomMma,
                                     UseSingleMma,
                                     UsePhaseAwareInitialKOffset, false, true,
                                     true, false, true,
                                     false>;
  auto kernel_global =
      explicit_warp_kernel<CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN, WarpK,
                           SmemAKBlock, UseCustomMma, UseSingleMma,
                           UsePhaseAwareInitialKOffset, true, true, true,
                           false, true, false>;
  if (mode == ProbeMode::kMainloop) {
    kernel = explicit_warp_kernel<CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN,
                                  WarpK, SmemAKBlock, UseCustomMma,
                                  UseSingleMma,
                                  UsePhaseAwareInitialKOffset, false, true,
                                  false, false, true,
                                  false>;
    kernel_global =
        explicit_warp_kernel<CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN, WarpK,
                             SmemAKBlock, UseCustomMma, UseSingleMma,
                             UsePhaseAwareInitialKOffset, true, true, false,
                             false, true, false>;
  } else if (mode == ProbeMode::kEpilogue) {
    kernel = explicit_warp_kernel<CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN,
                                  WarpK, SmemAKBlock, UseCustomMma,
                                  UseSingleMma,
                                  UsePhaseAwareInitialKOffset, false, false,
                                  true, false, true,
                                  false>;
    kernel_global =
        explicit_warp_kernel<CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN, WarpK,
                             SmemAKBlock, UseCustomMma, UseSingleMma,
                             UsePhaseAwareInitialKOffset, true, false, true,
                             false, true, false>;
  } else if (mode == ProbeMode::kAtomic) {
    kernel = explicit_warp_kernel<CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN,
                                  WarpK, SmemAKBlock, UseCustomMma,
                                  UseSingleMma,
                                  UsePhaseAwareInitialKOffset, false, true,
                                  false, true, true,
                                  false>;
    kernel_global =
        explicit_warp_kernel<CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN, WarpK,
                             SmemAKBlock, UseCustomMma, UseSingleMma,
                             UsePhaseAwareInitialKOffset, true, true, false,
                             true, true, false>;
  } else if (mode == ProbeMode::kAtomicNoReduce) {
    kernel = explicit_warp_kernel<CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN,
                                  WarpK, SmemAKBlock, UseCustomMma,
                                  UseSingleMma,
                                  UsePhaseAwareInitialKOffset, false, true,
                                  false, true, false,
                                  false>;
    kernel_global =
        explicit_warp_kernel<CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN, WarpK,
                             SmemAKBlock, UseCustomMma, UseSingleMma,
                             UsePhaseAwareInitialKOffset, true, true, false,
                             true, false, false>;
  } else if (mode == ProbeMode::kDirectPartitions) {
    kernel = explicit_warp_kernel<CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN,
                                  WarpK, SmemAKBlock, UseCustomMma,
                                  UseSingleMma,
                                  UsePhaseAwareInitialKOffset, false, true,
                                  false, false, true,
                                  true>;
    kernel_global =
        explicit_warp_kernel<CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN, WarpK,
                             SmemAKBlock, UseCustomMma, UseSingleMma,
                             UsePhaseAwareInitialKOffset, true, true, false,
                             false, true, true>;
  }
  if (kgroup_mode == KGroupMode::kGlobal) {
    kernel = kernel_global;
  }
  size_t const smem_bytes = sizeof(SharedStorage);
  if (smem_bytes >= (48u << 10)) {
    CUDA_CHECK_STATUS(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
  }

  dim3 grid(static_cast<unsigned>((m + CTA_M - 1) / CTA_M),
            static_cast<unsigned>(n / CTA_N));
  dim3 block(Warps * 32);
  kernel<<<grid, block, smem_bytes>>>(dev_a, dev_b, dev_c, m, n, k);
  cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    std::cerr << "kernel launch failed: " << cudaGetErrorString(launch_status)
              << std::endl;
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return RunStatus::kCudaError;
  }
  cudaError_t sync_status = cudaDeviceSynchronize();
  if (sync_status != cudaSuccess) {
    std::cerr << "kernel sync failed: " << cudaGetErrorString(sync_status)
              << std::endl;
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return RunStatus::kCudaError;
  }

  if (mode == ProbeMode::kDirectPartitions) {
    CUDA_CHECK_STATUS(cudaMemcpy(host_c_float.data(), dev_c, output_bytes,
                                 cudaMemcpyDeviceToHost));
  } else {
    CUDA_CHECK_STATUS(cudaMemcpy(host_c.data(), dev_c, output_bytes,
                                 cudaMemcpyDeviceToHost));
  }

  std::vector<float> actual_values(size_t(m) * size_t(n), 0.0f);
  std::vector<double> partition_l1(output_partitions, 0.0);
  std::vector<double> partition_l2(output_partitions, 0.0);
  std::vector<int> partition_nonzero(output_partitions, 0);
  for (size_t part = 0; part < output_partitions; ++part) {
    size_t const part_offset = part * size_t(m) * size_t(n);
    for (int i = 0; i < m * n; ++i) {
      float const value = mode == ProbeMode::kDirectPartitions
                              ? host_c_float[part_offset + i]
                              : float(host_c[part_offset + i]);
      actual_values[i] += value;
      partition_l1[part] += std::abs(value);
      partition_l2[part] += double(value) * double(value);
      partition_nonzero[part] += std::abs(value) > 1.0e-3f;
    }
  }

  float max_abs = 0.0f;
  float max_rel = 0.0f;
  int mismatches = 0;
  double actual_l2 = 0.0;
  double expected_l2 = 0.0;
  double dot = 0.0;
  double actual_l1 = 0.0;
  double expected_l1 = 0.0;
  int actual_nonzero = 0;
  int expected_nonzero = 0;
  int first_mismatch[6] = {-1, -1, -1, -1, -1, -1};
  for (int i = 0; i < m * n; ++i) {
    float const actual = actual_values[i];
    float const expected = reference[i];
    float const abs_diff = std::abs(actual - expected);
    float const rel_diff = abs_diff / std::max(1.0f, std::abs(expected));
    actual_l2 += double(actual) * double(actual);
    expected_l2 += double(expected) * double(expected);
    dot += double(actual) * double(expected);
    actual_l1 += std::abs(actual);
    expected_l1 += std::abs(expected);
    actual_nonzero += std::abs(actual) > 1.0e-3f;
    expected_nonzero += std::abs(expected) > 1.0e-3f;
    max_abs = std::max(max_abs, abs_diff);
    max_rel = std::max(max_rel, rel_diff);
    if (abs_diff > 0.25f && rel_diff > 0.05f) {
      if (mismatches < 6) {
        first_mismatch[mismatches] = i;
      }
      ++mismatches;
    }
  }
  double const cosine =
      dot / std::max(1.0e-20, std::sqrt(actual_l2 * expected_l2));

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  std::cout << "geometry=" << CTA_M << "x" << CTA_N << "x" << CTA_K << "x"
            << Warps << "x" << WarpM << "x" << WarpN << "x" << WarpK
            << " mainloop=" << mainloop_name(mainloop)
            << " mode=" << mode_name(mode)
            << " input=" << input_pattern_name(pattern)
            << " kgroup=" << kgroup_mode_name(kgroup_mode)
            << " layout_a_kblock=" << SmemAKBlock << " m=" << m
            << " n=" << n << " k=" << k
            << " kPartitionsK=" << Traits::kPartitionsK
            << " smem=" << smem_bytes << " max_abs=" << max_abs
            << " max_rel=" << max_rel << " mismatches=" << mismatches << "/"
            << (m * n) << " actual_l1=" << actual_l1
            << " expected_l1=" << expected_l1 << " actual_l2="
            << std::sqrt(actual_l2) << " expected_l2="
            << std::sqrt(expected_l2) << " cosine=" << cosine
            << " actual_nonzero=" << actual_nonzero
            << " expected_nonzero=" << expected_nonzero << std::endl;
  if (mode == ProbeMode::kDirectPartitions) {
    std::cout << "direct_partitions:";
    for (size_t part = 0; part < output_partitions; ++part) {
      std::cout << " p" << part << "_l1=" << partition_l1[part]
                << " p" << part << "_l2=" << std::sqrt(partition_l2[part])
                << " p" << part
                << "_nonzero=" << partition_nonzero[part];
    }
    std::cout << std::endl;
  }

  if (mismatches > 0) {
    std::cout << "first_mismatches:";
    for (int idx : first_mismatch) {
      if (idx < 0) {
        continue;
      }
      int const row = idx / n;
      int const col = idx % n;
      float const actual = actual_values[idx];
      float const expected = reference[idx];
      std::cout << " (r=" << row << ",c=" << col << ",actual=" << actual
                << ",expected=" << expected
                << ",diff=" << (actual - expected) << ")";
    }
    std::cout << std::endl;

    std::cout << "nearest_reference_for_first_mismatches:";
    for (int idx : first_mismatch) {
      if (idx < 0) {
        continue;
      }
      float const actual = actual_values[idx];
      int best_idx = -1;
      float best_diff = std::numeric_limits<float>::infinity();
      for (int ref_idx = 0; ref_idx < m * n; ++ref_idx) {
        float const diff = std::abs(actual - reference[ref_idx]);
        if (diff < best_diff) {
          best_diff = diff;
          best_idx = ref_idx;
        }
      }
      std::cout << " (actual_r=" << (idx / n) << ",actual_c=" << (idx % n)
                << " -> ref_r=" << (best_idx / n)
                << ",ref_c=" << (best_idx % n) << ",diff=" << best_diff
                << ")";
    }
    std::cout << std::endl;

    int nearest_exact = 0;
    int same_row_nearest = 0;
    int same_col_nearest = 0;
    int row_delta_hist[32] = {};
    int col_delta_hist[64] = {};
    for (int idx = 0; idx < m * n; ++idx) {
      float const actual = actual_values[idx];
      int best_idx = -1;
      float best_diff = std::numeric_limits<float>::infinity();
      for (int ref_idx = 0; ref_idx < m * n; ++ref_idx) {
        float const diff = std::abs(actual - reference[ref_idx]);
        if (diff < best_diff) {
          best_diff = diff;
          best_idx = ref_idx;
        }
      }
      if (best_diff < 1.0e-4f) {
        ++nearest_exact;
        int const actual_r = idx / n;
        int const actual_c = idx % n;
        int const ref_r = best_idx / n;
        int const ref_c = best_idx % n;
        same_row_nearest += actual_r == ref_r;
        same_col_nearest += actual_c == ref_c;
        if (m <= 32) {
          row_delta_hist[(ref_r - actual_r + m) % m]++;
        }
        if (n <= 64) {
          col_delta_hist[(ref_c - actual_c + n) % n]++;
        }
      }
    }
    std::cout << "nearest_exact_count=" << nearest_exact
              << " same_row=" << same_row_nearest
              << " same_col=" << same_col_nearest;
    if (m <= 32) {
      std::cout << " row_delta_hist:";
      for (int i = 0; i < m; ++i) {
        if (row_delta_hist[i]) {
          std::cout << " d" << i << "=" << row_delta_hist[i];
        }
      }
    }
    if (n <= 64) {
      std::cout << " col_delta_hist:";
      for (int i = 0; i < n; ++i) {
        if (col_delta_hist[i]) {
          std::cout << " d" << i << "=" << col_delta_hist[i];
        }
      }
    }
    std::cout << std::endl;
  }

  return mode == ProbeMode::kMainloop || mismatches == 0
             ? RunStatus::kPass
             : RunStatus::kMismatch;
}

template <int CTA_M, int CTA_N, int CTA_K, int Warps, int WarpM, int WarpN,
          int WarpK, int SmemAKBlock>
RunStatus run_geometry(int m, int n, int k, ProbeMode mode,
                       MainloopKind mainloop, InputPattern pattern,
                       KGroupMode kgroup_mode) {
  if (mainloop == MainloopKind::kCustom) {
    return run_geometry_impl<CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN, WarpK,
                             SmemAKBlock, true, false>(
        m, n, k, mode, mainloop, pattern, kgroup_mode);
  }
  if (mainloop == MainloopKind::kSingle) {
    return run_geometry_impl<CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN, WarpK,
                             SmemAKBlock, false, true>(
        m, n, k, mode, mainloop, pattern, kgroup_mode);
  }
  if (mainloop == MainloopKind::kSinglePhase) {
    return run_geometry_impl<CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN, WarpK,
                             SmemAKBlock, false, true, true>(
        m, n, k, mode, mainloop, pattern, kgroup_mode);
  }
  return run_geometry_impl<CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN, WarpK,
                           SmemAKBlock, false, false>(
      m, n, k, mode, mainloop, pattern, kgroup_mode);
}

template <int CM, int CN, int CK, int W, int WM, int WN, int WK>
RunStatus run_geometry_default_layout(int m, int n, int k, ProbeMode mode,
                                      MainloopKind mainloop,
                                      InputPattern pattern,
                                      KGroupMode kgroup_mode) {
  return run_geometry<CM, CN, CK, W, WM, WN, WK, CK>(
      m, n, k, mode, mainloop, pattern, kgroup_mode);
}

#define DISPATCH_GEOMETRY(CM, CN, CK, W, WM, WN, WK)                       \
  if (geometry.cta_m == CM && geometry.cta_n == CN &&                       \
      geometry.cta_k == CK && geometry.warps == W &&                        \
      geometry.warp_m == WM && geometry.warp_n == WN &&                     \
      geometry.warp_k == WK) {                                              \
    return run_geometry_default_layout<CM, CN, CK, W, WM, WN, WK>(          \
        m, n, k, mode, mainloop, pattern, kgroup_mode);                     \
  }

RunStatus dispatch(Geometry geometry, int m, int n, int k, ProbeMode mode,
                   MainloopKind mainloop, InputPattern pattern,
                   KGroupMode kgroup_mode) {
  DISPATCH_GEOMETRY(32, 64, 32, 4, 32, 32, 16)
  DISPATCH_GEOMETRY(64, 64, 32, 4, 32, 32, 32)
  DISPATCH_GEOMETRY(32, 64, 64, 4, 32, 32, 32)
  DISPATCH_GEOMETRY(32, 64, 64, 4, 32, 64, 16)
  DISPATCH_GEOMETRY(32, 64, 128, 4, 32, 64, 32)
  DISPATCH_GEOMETRY(32, 128, 64, 8, 32, 32, 32)
  DISPATCH_GEOMETRY(32, 128, 64, 8, 32, 64, 16)
  std::cerr << "Unsupported probe geometry " << geometry_label(geometry)
            << ". Add it to tests/sm70_warpk16_probe.cu if needed."
            << std::endl;
  return RunStatus::kUnsupported;
}

#undef DISPATCH_GEOMETRY

int run_expected_case(char const* name, Geometry geometry,
                      MainloopKind mainloop, ProbeMode mode, int m, int n,
                      int k, InputPattern pattern, KGroupMode kgroup_mode,
                      ExpectedOutcome expected) {
  std::cout << "[ RUN      ] " << name << " geometry="
            << geometry_label(geometry) << " mainloop="
            << mainloop_name(mainloop) << " mode=" << mode_name(mode)
            << " input=" << input_pattern_name(pattern)
            << " kgroup=" << kgroup_mode_name(kgroup_mode) << " M=" << m
            << " N=" << n << " K=" << k << " expected="
            << (expected == ExpectedOutcome::kPass ? "pass" : "mismatch")
            << std::endl;

  RunStatus const status =
      dispatch(geometry, m, n, k, mode, mainloop, pattern, kgroup_mode);
  bool const matched =
      (expected == ExpectedOutcome::kPass && status == RunStatus::kPass) ||
      (expected == ExpectedOutcome::kMismatch &&
       status == RunStatus::kMismatch);
  if (!matched) {
    std::cerr << "[  FAILED  ] " << name
              << " returned status=" << static_cast<int>(status)
              << std::endl;
    return 1;
  }

  std::cout << "[       OK ] " << name << std::endl;
  return 0;
}

int run_smoke_suite() {
  if (!print_device()) {
    return 1;
  }

  int failures = 0;
  failures += run_expected_case(
      "baseline stock WarpK32", {64, 64, 32, 4, 32, 32, 32},
      MainloopKind::kStock, ProbeMode::kFull, 64, 64, 128,
      InputPattern::kAll, KGroupMode::kLocal, ExpectedOutcome::kPass);
  failures += run_expected_case(
      "custom kPartitionsK2 WarpK32", {32, 64, 64, 4, 32, 32, 32},
      MainloopKind::kCustom, ProbeMode::kFull, 32, 64, 128,
      InputPattern::kAll, KGroupMode::kLocal, ExpectedOutcome::kPass);
  failures += run_expected_case(
      "custom kPartitionsK4 WarpK32 multi-CTA-K", {32, 64, 128, 4, 32, 64, 32},
      MainloopKind::kCustom, ProbeMode::kFull, 32, 64, 768,
      InputPattern::kAll, KGroupMode::kLocal, ExpectedOutcome::kPass);
  failures += run_expected_case(
      "single kPartitionsK2 WarpK32", {32, 64, 64, 4, 32, 32, 32},
      MainloopKind::kSingle, ProbeMode::kFull, 32, 64, 128,
      InputPattern::kAll, KGroupMode::kLocal, ExpectedOutcome::kPass);
  failures += run_expected_case(
      "WarpK16 single partition", {32, 64, 32, 4, 32, 32, 16},
      MainloopKind::kSingle, ProbeMode::kFull, 32, 64, 16,
      InputPattern::kAll, KGroupMode::kLocal, ExpectedOutcome::kPass);

  if (failures == 0) {
    std::cout << "[==========] smoke suite passed" << std::endl;
  }
  return failures == 0 ? 0 : 2;
}

int run_diagnose_warpk16_suite() {
  if (!print_device()) {
    return 1;
  }

  int failures = 0;
  Geometry const g{32, 64, 32, 4, 32, 32, 16};
  Geometry const g_warpk32{32, 64, 64, 4, 32, 32, 32};

  failures += run_expected_case(
      "WarpK16 single full K16", g, MainloopKind::kSingle, ProbeMode::kFull,
      32, 64, 16, InputPattern::kAll, KGroupMode::kLocal, ExpectedOutcome::kPass);
  failures += run_expected_case(
      "WarpK16 single atomic K16", g, MainloopKind::kSingle,
      ProbeMode::kAtomic, 32, 64, 16, InputPattern::kAll,
      KGroupMode::kLocal, ExpectedOutcome::kPass);
  failures += run_expected_case(
      "WarpK16 single full K32 k0 only", g, MainloopKind::kSingle,
      ProbeMode::kFull, 32, 64, 32, InputPattern::kPartition0Only,
      KGroupMode::kLocal, ExpectedOutcome::kPass);
  failures += run_expected_case(
      "WarpK16 single full K32 k1 only", g, MainloopKind::kSingle,
      ProbeMode::kFull, 32, 64, 32, InputPattern::kPartition1Only,
      KGroupMode::kLocal, ExpectedOutcome::kMismatch);
  failures += run_expected_case(
      "WarpK16 single atomic K32 k0 only", g, MainloopKind::kSingle,
      ProbeMode::kAtomic, 32, 64, 32, InputPattern::kPartition0Only,
      KGroupMode::kLocal, ExpectedOutcome::kPass);
  failures += run_expected_case(
      "WarpK16 single atomic K32 k1 only", g, MainloopKind::kSingle,
      ProbeMode::kAtomic, 32, 64, 32, InputPattern::kPartition1Only,
      KGroupMode::kLocal, ExpectedOutcome::kMismatch);
  failures += run_expected_case(
      "WarpK16 single direct K32 k0 only", g, MainloopKind::kSingle,
      ProbeMode::kDirectPartitions, 32, 64, 32,
      InputPattern::kPartition0Only, KGroupMode::kLocal,
      ExpectedOutcome::kPass);
  failures += run_expected_case(
      "WarpK16 single direct K32 k1 row permutation", g,
      MainloopKind::kSingle, ProbeMode::kDirectPartitions, 32, 64, 32,
      InputPattern::kPartition1Only, KGroupMode::kLocal,
      ExpectedOutcome::kMismatch);
  failures += run_expected_case(
      "WarpK16 phase-aware direct K32 k1 fixed", g,
      MainloopKind::kSinglePhase, ProbeMode::kDirectPartitions, 32, 64, 32,
      InputPattern::kPartition1Only, KGroupMode::kLocal,
      ExpectedOutcome::kPass);
  failures += run_expected_case(
      "WarpK16 phase-aware full K32 fixed", g,
      MainloopKind::kSinglePhase, ProbeMode::kFull, 32, 64, 32,
      InputPattern::kAll, KGroupMode::kLocal, ExpectedOutcome::kPass);
  failures += run_expected_case(
      "WarpK32 single direct K64 k1 baseline", g_warpk32,
      MainloopKind::kSingle, ProbeMode::kDirectPartitions, 32, 64, 64,
      InputPattern::kPartition1Only, KGroupMode::kLocal,
      ExpectedOutcome::kPass);
  failures += run_expected_case(
      "WarpK16 single full K32 mismatch repro", g, MainloopKind::kSingle,
      ProbeMode::kFull, 32, 64, 32, InputPattern::kAll,
      KGroupMode::kLocal, ExpectedOutcome::kMismatch);
  failures += run_expected_case(
      "WarpK16 single atomic K32 mismatch repro", g, MainloopKind::kSingle,
      ProbeMode::kAtomic, 32, 64, 32, InputPattern::kAll,
      KGroupMode::kLocal, ExpectedOutcome::kMismatch);
  failures += run_expected_case(
      "WarpK16 custom full K32 fixed", g, MainloopKind::kCustom,
      ProbeMode::kFull, 32, 64, 32, InputPattern::kAll,
      KGroupMode::kLocal, ExpectedOutcome::kPass);
  failures += run_expected_case(
      "WarpK16 custom full K64 fixed", g, MainloopKind::kCustom,
      ProbeMode::kFull, 32, 64, 64, InputPattern::kAll,
      KGroupMode::kLocal, ExpectedOutcome::kPass);
  failures += run_expected_case(
      "WarpK16 custom full K128 fixed", g, MainloopKind::kCustom,
      ProbeMode::kFull, 32, 64, 128, InputPattern::kAll,
      KGroupMode::kLocal, ExpectedOutcome::kPass);
  failures += run_expected_case(
      "WarpK16 custom 32x64x64 full K128 fixed", {32, 64, 64, 4, 32, 64, 16},
      MainloopKind::kCustom, ProbeMode::kFull, 32, 64, 128,
      InputPattern::kAll, KGroupMode::kLocal, ExpectedOutcome::kPass);
  failures += run_expected_case(
      "WarpK16 custom 32x128x64 full K128 fixed",
      {32, 128, 64, 8, 32, 64, 16}, MainloopKind::kCustom, ProbeMode::kFull,
      32, 128, 128, InputPattern::kAll, KGroupMode::kLocal,
      ExpectedOutcome::kPass);

  if (failures == 0) {
    std::cout
        << "[==========] WarpK16 diagnosis reproduced and production "
           "pipelined phase-aware path passed"
        << std::endl;
  }
  return failures == 0 ? 0 : 2;
}

void print_usage(char const* argv0) {
  std::cerr
      << "Usage: " << argv0 << "\n"
      << "  --smoke\n"
      << "  --diagnose-warpk16\n"
      << "  CTA_MxCTA_NxCTA_KxWarpsxWarpMxWarpNxWarpK "
         "[stock|custom|single|single_phase] [full|mainloop|epilogue|atomic] "
         "[all|k0|k1] [local|global] [M N K]\n"
      << "Example: " << argv0
      << " 32x64x32x4x32x32x16 single_phase direct k1 local 32 64 32\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--smoke") == 0) {
    return run_smoke_suite();
  }
  if (argc == 2 && std::strcmp(argv[1], "--diagnose-warpk16") == 0) {
    return run_diagnose_warpk16_suite();
  }
  if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 ||
                    std::strcmp(argv[1], "-h") == 0)) {
    print_usage(argv[0]);
    return 0;
  }

  if (argc < 2 || argc > 9) {
    print_usage(argv[0]);
    return 1;
  }

  Geometry geometry;
  if (!parse_geometry(argv[1], geometry)) {
    std::cerr << "Invalid geometry '" << argv[1]
              << "'. Expected CTA_MxCTA_NxCTA_KxWarpsxWarpMxWarpNxWarpK."
              << std::endl;
    return 1;
  }

  ProbeMode mode = ProbeMode::kFull;
  MainloopKind mainloop = MainloopKind::kStock;
  InputPattern pattern = InputPattern::kAll;
  KGroupMode kgroup_mode = KGroupMode::kLocal;
  int arg_index = 2;
  if (argc >= 3 && parse_mainloop_kind(argv[2], mainloop)) {
    arg_index = 3;
  }
  if (argc > arg_index && parse_mode(argv[arg_index], mode)) {
    ++arg_index;
  }
  if (argc > arg_index && parse_input_pattern(argv[arg_index], pattern)) {
    ++arg_index;
  }
  if (argc > arg_index && parse_kgroup_mode(argv[arg_index], kgroup_mode)) {
    ++arg_index;
  }

  int m = geometry.cta_m * 2;
  int n = geometry.cta_n * 2;
  int k = geometry.cta_k * 4;
  if (argc - arg_index == 3) {
    m = std::atoi(argv[arg_index]);
    n = std::atoi(argv[arg_index + 1]);
    k = std::atoi(argv[arg_index + 2]);
  } else if (argc != arg_index) {
    std::cerr << "Expected either no explicit M/N/K or all three values."
              << std::endl;
    return 1;
  }
  if (m <= 0 || n <= 0 || k <= 0 || n % geometry.cta_n != 0) {
    std::cerr << "Invalid problem size M=" << m << " N=" << n << " K=" << k
              << "; N must be positive and divisible by CTA_N." << std::endl;
    return 1;
  }

  if (!print_device()) {
    return 1;
  }

  return static_cast<int>(
      dispatch(geometry, m, n, k, mode, mainloop, pattern, kgroup_mode));
}

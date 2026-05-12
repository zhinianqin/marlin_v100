#include "core/registration.h"
#include "quantization/marlin/dequant.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>
#include <torch/library.h>
#include <type_traits>

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/layout/tensor_op_multiplicand_sm70.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

namespace {

constexpr int kCtaM = 128;
constexpr int kCtaN = 256;
constexpr int kCtaK = 32;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kQuantTileK = 16;
constexpr int kQuantTileN = 64;
constexpr int kU4ValuesPerWord = 8;
constexpr int kU4WordsPerTile = kQuantTileK * kQuantTileN / kU4ValuesPerWord;

template <typename Shape_, typename ThreadMap_, int GroupSize_, bool FullTile_>
class Sm70U4B8IteratorB {
 public:
  using Shape = Shape_;
  using ThreadMap = ThreadMap_;
  static int const kGroupSize = GroupSize_;
  static bool const kFullTile = FullTile_;
  static bool const kCacheScales = GroupSize_ == -1;
  using Element = cutlass::half_t;
  using Fragment = cutlass::Array<
      Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
  static_assert(Shape::kN == kCtaN,
                "The SM70 kU4B8 IteratorB expects CTA_N=256.");
  static_assert(Shape::kK == kCtaK,
                "The SM70 kU4B8 IteratorB expects CTA_K=32.");
  static_assert(ThreadMap::Iterations::kStrided == 1,
                "The SM70 kU4B8 IteratorB expects one K-strided iteration.");
  static_assert(ThreadMap::Iterations::kContiguous == 4,
                "The SM70 kU4B8 IteratorB expects four 64-column accesses.");
  static_assert(ThreadMap::Delta::kContiguous == kQuantTileN,
                "The SM70 kU4B8 IteratorB expects 64-column deltas.");
  static_assert(ThreadMap::kElementsPerAccess == kU4ValuesPerWord,
                "The SM70 kU4B8 IteratorB expects one packed int4 word per "
                "access.");

  struct Params {
    int size_k;
    int size_n;

    CUTLASS_HOST_DEVICE
    Params() : size_k(0), size_n(0) {}

    CUTLASS_HOST_DEVICE
    Params(int size_k_, int size_n_) : size_k(size_k_), size_n(size_n_) {}
  };

 private:
  uint32_t const* qweight_;
  half const* scales_;
  Params params_;
  cutlass::layout::PitchLinearCoord thread_offset_;
  int qweight_offsets_[ThreadMap::Iterations::kCount];
  int k_offset_;
  int n_offset_;
  int tile_k_end_;
  int next_k_advance_;
  bool mask_enabled_;
  mutable int cached_group_;
  mutable half2 cached_scales_[ThreadMap::Iterations::kContiguous * 4];

 public:
  CUTLASS_DEVICE
  Sm70U4B8IteratorB(Params const& params, uint32_t const* qweight,
                    half const* scales, int thread_id,
                    cutlass::MatrixCoord const& threadblock_offset)
      : qweight_(qweight),
        scales_(scales),
        params_(params),
        thread_offset_(ThreadMap::initial_offset(thread_id)),
        k_offset_(threadblock_offset.row()),
        n_offset_(threadblock_offset.column()),
        tile_k_end_(threadblock_offset.row() +
                    initial_k_advance(params.size_k)),
        next_k_advance_(initial_k_advance(params.size_k)),
        mask_enabled_(true),
        cached_group_(-2) {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int const idx = c + s * ThreadMap::Iterations::kContiguous;
        int const logical_k = threadblock_offset.row() +
                              thread_offset_.strided() +
                              s * ThreadMap::Delta::kStrided;
        int const logical_n = threadblock_offset.column() +
                              thread_offset_.contiguous() +
                              c * ThreadMap::Delta::kContiguous;
        qweight_offsets_[idx] =
            qweight_offset_from_logical(params_, logical_k, logical_n);
      }
    }
  }

  CUTLASS_DEVICE
  Sm70U4B8IteratorB& operator++() {
    int const k_advance = next_k_advance_;
    int const k_advance_qwords =
        (k_advance / kQuantTileK) * (params_.size_n * 2);
    k_offset_ += k_advance;
    int const next_tile_k_end = k_offset_ + Shape::kK;
    tile_k_end_ = next_tile_k_end < params_.size_k ? next_tile_k_end
                                                   : params_.size_k;
    next_k_advance_ = Shape::kK;
    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < ThreadMap::Iterations::kCount; ++idx) {
      qweight_offsets_[idx] += k_advance_qwords;
    }
    return *this;
  }

  CUTLASS_DEVICE
  void clear_mask(bool enable = true) {
    if (enable) {
      mask_enabled_ = false;
    }
  }

  CUTLASS_DEVICE
  void enable_mask() { mask_enabled_ = true; }

  CUTLASS_DEVICE
  static int initial_k_advance(int size_k) {
    int const residue_k = size_k % Shape::kK;
    return residue_k == 0 ? Shape::kK : residue_k;
  }

  CUTLASS_DEVICE
  int scale_group(int logical_k) const {
    if constexpr (kGroupSize == -1) {
      return 0;
    } else {
      static_assert(kGroupSize == 32 || kGroupSize == 64 ||
                        kGroupSize == 128,
                    "SM70 kU4B8 prototype only specializes group sizes "
                    "-1, 32, 64, and 128.");
      return logical_k / kGroupSize;
    }
  }

  CUTLASS_DEVICE
  static int initial_qweight_offset(
      Params const& params,
      cutlass::layout::PitchLinearCoord const& thread_offset,
      cutlass::MatrixCoord const& threadblock_offset) {
    static_assert(ThreadMap::Delta::kContiguous == kQuantTileN,
                  "The SM70 kU4B8 prototype expects 64-column B access "
                  "deltas.");

    int const logical_k = threadblock_offset.row() + thread_offset.strided();
    int const logical_n =
        threadblock_offset.column() + thread_offset.contiguous();
    return qweight_offset_from_logical(params, logical_k, logical_n);
  }

  CUTLASS_DEVICE
  static int qweight_offset_from_logical(Params const& params, int logical_k,
                                         int logical_n) {
    int const k_tile = logical_k / kQuantTileK;
    int const local_k = logical_k - k_tile * kQuantTileK;
    int const n_tile = logical_n / kQuantTileN;
    int const local_n_vec =
        (logical_n - n_tile * kQuantTileN) / ThreadMap::kElementsPerAccess;

    return k_tile * (params.size_n * 2) + n_tile * kU4WordsPerTile +
           local_k * (kQuantTileN / kU4ValuesPerWord) + local_n_vec;
  }

  CUTLASS_DEVICE
  void refresh_scale_cache(int group) const {
    if constexpr (!kCacheScales) {
      return;
    } else {
      if (cached_group_ == group) {
        return;
      }

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int const logical_n =
            n_offset_ + thread_offset_.contiguous() +
            c * ThreadMap::Delta::kContiguous;
        if constexpr (kFullTile) {
          half2 const* scale_vec =
              reinterpret_cast<half2 const*>(scales_ + group * params_.size_n +
                                             logical_n);
          half2* cache = cached_scales_ + c * 4;
          cache[0] = scale_vec[0];
          cache[1] = scale_vec[1];
          cache[2] = scale_vec[2];
          cache[3] = scale_vec[3];
        } else if (logical_n + ThreadMap::kElementsPerAccess <=
                   params_.size_n) {
          half2 const* scale_vec =
              reinterpret_cast<half2 const*>(scales_ + group * params_.size_n +
                                             logical_n);
          half2* cache = cached_scales_ + c * 4;
          cache[0] = scale_vec[0];
          cache[1] = scale_vec[1];
          cache[2] = scale_vec[2];
          cache[3] = scale_vec[3];
        }
      }
      cached_group_ = group;
    }
  }

  CUTLASS_DEVICE
  void load_full_tile(Fragment& frag) const {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        constexpr int kAccess = ThreadMap::kElementsPerAccess;
        int const frag_base =
            (c + s * ThreadMap::Iterations::kContiguous) * kAccess;
        int const qword_offset =
            qweight_offsets_[c + s * ThreadMap::Iterations::kContiguous];

        uint32_t const qword = qweight_[qword_offset];
        half2 const* scale_vec;
        if constexpr (!kCacheScales) {
          int const logical_n =
              n_offset_ + thread_offset_.contiguous() +
              c * ThreadMap::Delta::kContiguous;
          int const logical_k =
              k_offset_ + thread_offset_.strided() +
              s * ThreadMap::Delta::kStrided;
          int const group = scale_group(logical_k);
          scale_vec =
              reinterpret_cast<half2 const*>(scales_ + group * params_.size_n +
                                             logical_n);
        } else {
          scale_vec = cached_scales_ + c * 4;
        }

        half2 deq01_23[2];
        half2 deq45_67[2];
        marlin::dequant<half2, vllm::kU4B8.id(), false>(
            static_cast<int>(qword), deq01_23);
        marlin::dequant<half2, vllm::kU4B8.id(), false>(
            static_cast<int>(qword >> 8), deq45_67);

        half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
        frag_vec[0] = __hmul2(deq01_23[0], scale_vec[0]);
        frag_vec[1] = __hmul2(deq01_23[1], scale_vec[1]);
        frag_vec[2] = __hmul2(deq45_67[0], scale_vec[2]);
        frag_vec[3] = __hmul2(deq45_67[1], scale_vec[3]);
      }
    }
  }

  CUTLASS_DEVICE
  void load_residue_tile(Fragment& frag) const {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        constexpr int kAccess = ThreadMap::kElementsPerAccess;
        int const frag_base =
            (c + s * ThreadMap::Iterations::kContiguous) * kAccess;
        int const logical_n =
            n_offset_ + thread_offset_.contiguous() +
            c * ThreadMap::Delta::kContiguous;
        int const logical_k =
            k_offset_ + thread_offset_.strided() +
            s * ThreadMap::Delta::kStrided;

        bool const valid = logical_k < tile_k_end_ &&
                           logical_n + kAccess <= params_.size_n;
        if (!valid) {
          half2 const zero = __float2half2_rn(0.0f);
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          frag_vec[0] = zero;
          frag_vec[1] = zero;
          frag_vec[2] = zero;
          frag_vec[3] = zero;
          continue;
        }

        int const qword_offset =
            qweight_offsets_[c + s * ThreadMap::Iterations::kContiguous];

        uint32_t const qword = qweight_[qword_offset];
        half2 const* scale_vec;
        if constexpr (!kCacheScales) {
          int const group = scale_group(logical_k);
          scale_vec =
              reinterpret_cast<half2 const*>(scales_ + group * params_.size_n +
                                             logical_n);
        } else {
          scale_vec = cached_scales_ + c * 4;
        }

        half2 deq01_23[2];
        half2 deq45_67[2];
        marlin::dequant<half2, vllm::kU4B8.id(), false>(
            static_cast<int>(qword), deq01_23);
        marlin::dequant<half2, vllm::kU4B8.id(), false>(
            static_cast<int>(qword >> 8), deq45_67);

        half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
        frag_vec[0] = __hmul2(deq01_23[0], scale_vec[0]);
        frag_vec[1] = __hmul2(deq01_23[1], scale_vec[1]);
        frag_vec[2] = __hmul2(deq45_67[0], scale_vec[2]);
        frag_vec[3] = __hmul2(deq45_67[1], scale_vec[3]);
      }
    }
  }

  CUTLASS_DEVICE
  void load(Fragment& frag) const {
    if (!mask_enabled_) {
      return;
    }

    int const first_logical_k = k_offset_ + thread_offset_.strided();
    int const current_group = scale_group(first_logical_k);
    refresh_scale_cache(current_group);

    if constexpr (kFullTile) {
      load_full_tile(frag);
    } else {
      load_residue_tile(frag);
    }
  }
};

template <int GroupSize, bool FullTile>
struct Sm70U4B8GemmTraits {
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using ThreadblockShape = cutlass::gemm::GemmShape<kCtaM, kCtaN, kCtaK>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using MmaCore = cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      cutlass::arch::OpClassTensorOp, 2, cutlass::arch::OpMultiplyAdd>;
  static_assert(MmaCore::kThreads == kThreads,
                "SM70 kU4B8 launch threads must match CUTLASS warp count.");
  using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA,
      128 / cutlass::sizeof_bits<ElementA>::value>;
  using IteratorB = Sm70U4B8IteratorB<
      ThreadblockShape, typename MmaCore::IteratorThreadMapB, GroupSize,
      FullTile>;
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
                "kU4B8 B operand must be stored through CUTLASS' predefined "
                "Volta B-congruous shared-memory layout.");
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

template <int GroupSize, bool FullTile>
__global__ __launch_bounds__(kThreads, 1)
void sm70_marlin_u4b8_gemm_kernel(
    cutlass::half_t const* __restrict__ a,
    uint32_t const* __restrict__ b_q_weight,
    cutlass::half_t const* __restrict__ b_scales,
    cutlass::half_t* __restrict__ c, int m, int n, int k, int lda) {
  using Traits = Sm70U4B8GemmTraits<GroupSize, FullTile>;
  using Mma = typename Traits::Mma;
  using Epilogue = typename Traits::Epilogue;

  extern __shared__ char smem[];
  auto& shared_storage =
      *reinterpret_cast<typename Traits::SharedStorage*>(smem);

  int thread_idx = threadIdx.x;
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_idx = threadIdx.x % 32;

  cutlass::MatrixCoord tb_offset_A{int(blockIdx.x) * kCtaM, 0};
  cutlass::MatrixCoord tb_offset_B{0, int(blockIdx.y) * kCtaN};
  cutlass::MatrixCoord tb_offset_C{int(blockIdx.x) * kCtaM,
                                   int(blockIdx.y) * kCtaN};

  typename Traits::LayoutA layout_a(lda);
  typename Traits::LayoutC layout_c(n);

  typename Mma::IteratorA iterator_A(
      typename Mma::IteratorA::Params(layout_a),
      const_cast<cutlass::half_t*>(a), cutlass::MatrixCoord(m, k), thread_idx,
      tb_offset_A);
  typename Mma::IteratorB iterator_B(
      typename Mma::IteratorB::Params(k, n),
      reinterpret_cast<uint32_t const*>(b_q_weight),
      reinterpret_cast<half const*>(b_scales), thread_idx, tb_offset_B);

  Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
  typename Mma::FragmentC accumulators;
  accumulators.clear();

  int const gemm_k_iterations = (k + kCtaK - 1) / kCtaK;
  mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

  typename Traits::OutputOp output_op({1.0f, 0.0f});
  typename Epilogue::OutputTileIterator iterator_C(
      typename Epilogue::OutputTileIterator::Params(layout_c), c,
      cutlass::MatrixCoord(m, n), thread_idx, tb_offset_C);
  typename Epilogue::OutputTileIterator iterator_D(
      typename Epilogue::OutputTileIterator::Params(layout_c), c,
      cutlass::MatrixCoord(m, n), thread_idx, tb_offset_C);

  Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
  epilogue(output_op, iterator_D, accumulators, iterator_C);
}

}  // namespace

template <int GroupSize, bool FullTile>
torch::Tensor launch_sm70_marlin_u4b8_gemm(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, int64_t size_m, int64_t size_n, int64_t size_k) {
  auto kernel = sm70_marlin_u4b8_gemm_kernel<GroupSize, FullTile>;
  size_t smem_bytes =
      sizeof(typename Sm70U4B8GemmTraits<GroupSize, FullTile>::SharedStorage);
  if (smem_bytes >= (48u << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
  }

  dim3 grid(static_cast<unsigned>((size_m + kCtaM - 1) / kCtaM),
            static_cast<unsigned>((size_n + kCtaN - 1) / kCtaN));
  dim3 block(kThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

  kernel<<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<cutlass::half_t const*>(a.data_ptr<at::Half>()),
      reinterpret_cast<uint32_t const*>(b_q_weight.data_ptr<int32_t>()),
      reinterpret_cast<cutlass::half_t const*>(b_scales.data_ptr<at::Half>()),
      reinterpret_cast<cutlass::half_t*>(c.data_ptr<at::Half>()),
      static_cast<int>(size_m), static_cast<int>(size_n),
      static_cast<int>(size_k), static_cast<int>(a.stride(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

template <bool FullTile>
torch::Tensor launch_sm70_marlin_u4b8_gemm_group_size(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, int64_t size_m, int64_t size_n, int64_t size_k,
    int64_t group_size) {
  switch (group_size) {
    case -1:
      return launch_sm70_marlin_u4b8_gemm<-1, FullTile>(
          a, c, b_q_weight, b_scales, size_m, size_n, size_k);
    case 32:
      return launch_sm70_marlin_u4b8_gemm<32, FullTile>(
          a, c, b_q_weight, b_scales, size_m, size_n, size_k);
    case 64:
      return launch_sm70_marlin_u4b8_gemm<64, FullTile>(
          a, c, b_q_weight, b_scales, size_m, size_n, size_k);
    case 128:
      return launch_sm70_marlin_u4b8_gemm<128, FullTile>(
          a, c, b_q_weight, b_scales, size_m, size_n, size_k);
    default:
      TORCH_CHECK(false,
                  "SM70 CUTLASS uint4b8 prototype supports only group_size "
                  "-1, 32, 64, or 128. Got ",
                  group_size);
  }
  return c;
}

torch::Tensor sm70_marlin_u4b8_gemm(torch::Tensor& a, torch::Tensor& c,
                                    torch::Tensor& b_q_weight,
                                    torch::Tensor& b_scales, int64_t size_m,
                                    int64_t size_n, int64_t size_k,
                                    int64_t group_size) {
  c10::cuda::CUDAGuard device_guard(a.device());

  bool const full_tile = (size_k % kCtaK == 0 && size_n % kCtaN == 0);
  if (full_tile) {
    return launch_sm70_marlin_u4b8_gemm_group_size<true>(
        a, c, b_q_weight, b_scales, size_m, size_n, size_k, group_size);
  }
  return launch_sm70_marlin_u4b8_gemm_group_size<false>(
      a, c, b_q_weight, b_scales, size_m, size_n, size_k, group_size);
}

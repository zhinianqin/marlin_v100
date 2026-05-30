#include "core/registration.h"
#include "quantization/marlin/dequant.h"
#include "quantization/marlin/sm70_dense_common.cuh"
#include "quantization/marlin/sm70_dense_gemm.cuh"
#include "quantization/marlin/sm70_dense_iterator_utils.cuh"
#include "quantization/marlin/sm70_dense_splitk.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>
#include <torch/library.h>
#include <cstdlib>
#include <sstream>
#include <string>
#include <type_traits>

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/layout/tensor_op_multiplicand_sm70.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

using marlin::sm70_dense::Sm70DenseCtaGeometry;
using marlin::sm70_dense::Sm70DenseAtomicFp32Epilogue;
using marlin::sm70_dense::Sm70DenseGemmTraits;
using marlin::sm70_dense::Sm70DenseSplitKPartition;
using marlin::sm70_dense::check_sm70_dense_cta_geometry;
using marlin::sm70_dense::check_sm70_dense_n_tile_alignment;
using marlin::sm70_dense::configure_dynamic_smem;
using marlin::sm70_dense::cta_grid;
using marlin::sm70_dense::dispatch_geometry;
using marlin::sm70_dense::kCtaK;
using marlin::sm70_dense::kQuantTileK;
using marlin::sm70_dense::kQuantTileN;
using marlin::sm70_dense::launch_sm70_dense_fp32_to_fp16;
using marlin::sm70_dense::resolve_sm70_dense_cta_geometry;
using marlin::sm70_dense::parse_sm70_dense_split_k;
using marlin::sm70_dense::qword_from_vector;
using marlin::sm70_dense::sm70_dense_active_split_k;
using marlin::sm70_dense::sm70_dense_get_splitk_ctmp;
using marlin::sm70_dense::sm70_dense_splitk_partition;
using marlin::sm70_dense::u4_cta_n_qweight_offset_from_logical;

namespace {

constexpr int kU4ValuesPerWord = 8;
constexpr char const* kSm70MarlinU4B8SplitKEnv = "SM70_MARLIN_U4B8_SPLIT_K";

template <typename Shape_, typename ThreadMap_, int GroupSize_>
class Sm70U4B8IteratorB {
 public:
  using Shape = Shape_;
  using ThreadMap = ThreadMap_;
  static int const kGroupSize = GroupSize_;
  using Element = cutlass::half_t;
  using Fragment = cutlass::Array<
      Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
  static_assert(Shape::kK == kCtaK,
                "The SM70 kU4B8 IteratorB expects CTA_K=32.");
  static_assert(Shape::kN == 64 || Shape::kN == 128 || Shape::kN == 256,
                "The SM70 kU4B8 IteratorB expects CTA_N in {64, 128, 256}.");
  static_assert(ThreadMap::Iterations::kContiguous ==
                    Shape::kN / kQuantTileN,
                "The SM70 kU4B8 IteratorB expects one contiguous iteration "
                "per 64-column quant tile.");
  static_assert(ThreadMap::Delta::kContiguous == kQuantTileN,
                "The SM70 kU4B8 IteratorB expects 64-column deltas.");
  static_assert(ThreadMap::kElementsPerAccess == kU4ValuesPerWord,
                "The SM70 kU4B8 IteratorB expects one packed int4 word per "
                "access.");

  struct Params {
    int size_n;
    int aligned_initial_k_step;

    CUTLASS_HOST_DEVICE
    Params() : size_n(0), aligned_initial_k_step(Shape::kK) {}

    CUTLASS_HOST_DEVICE
    Params(int size_k, int size_n_)
        : size_n(size_n_), aligned_initial_k_step(aligned_k_step(size_k)) {}
  };

 private:
  uint32_t const* qweight_;
  half const* scales_;
  Params params_;
  cutlass::layout::PitchLinearCoord thread_offset_;
  int qweight_base_offset_;
  int k_offset_;
  int n_offset_;
  int aligned_k_step_;
  bool mask_enabled_;
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
        aligned_k_step_(params.aligned_initial_k_step),
        mask_enabled_(true) {
    int const logical_k = threadblock_offset.row() + thread_offset_.strided();
    int const logical_n =
        threadblock_offset.column() + thread_offset_.contiguous();
    qweight_base_offset_ =
        qweight_offset_from_logical(params_, logical_k, logical_n);
    if constexpr (kGroupSize == -1) {
      // GroupSize=-1 uses one scale row for every K tile. Cache it once
      // before the MMA mainloop instead of reloading it from IteratorB::load().
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int const cache_n =
            n_offset_ + thread_offset_.contiguous() +
            c * ThreadMap::Delta::kContiguous;
        half2 const* scale_vec =
            reinterpret_cast<half2 const*>(scales_ + cache_n);
        half2* scale_cache = cached_scales_ + c * 4;
        scale_cache[0] = scale_vec[0];
        scale_cache[1] = scale_vec[1];
        scale_cache[2] = scale_vec[2];
        scale_cache[3] = scale_vec[3];
      }
    }
  }

  CUTLASS_DEVICE
  Sm70U4B8IteratorB& operator++() {
    int const k_advance = aligned_k_step_;
    int const k_advance_qwords =
        (k_advance / kQuantTileK) * (params_.size_n * 2);
    k_offset_ += k_advance;
    aligned_k_step_ = Shape::kK;
    qweight_base_offset_ += k_advance_qwords;
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
  static int aligned_k_step(int size_k) {
    int const k_tail = size_k % Shape::kK;
    return k_tail == 0 ? Shape::kK : k_tail;
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
  static int qweight_offset_from_logical(Params const& params, int logical_k,
                                         int logical_n) {
    return u4_cta_n_qweight_offset_from_logical<Shape::kN>(params.size_n, logical_k,
                                                     logical_n);
  }

  CUTLASS_DEVICE
  int qweight_offset(int s, int c) const {
    if constexpr (ThreadMap::Iterations::kStrided == 1) {
      return qweight_base_offset_ + c;
    } else {
      int const logical_k =
          k_offset_ + thread_offset_.strided() +
          s * ThreadMap::Delta::kStrided;
      int const logical_n =
          n_offset_ + thread_offset_.contiguous() +
          c * ThreadMap::Delta::kContiguous;
      return qweight_offset_from_logical(params_, logical_k, logical_n);
    }
  }

  CUTLASS_DEVICE
  void cache_metadata_lane_vectors(int c, int group, int cache_n) const {
    half2 const* scale_vec = reinterpret_cast<half2 const*>(
        scales_ + group * params_.size_n + cache_n);
    half2* scale_cache = cached_scales_ + c * 4;
    scale_cache[0] = scale_vec[0];
    scale_cache[1] = scale_vec[1];
    scale_cache[2] = scale_vec[2];
    scale_cache[3] = scale_vec[3];
  }

  CUTLASS_DEVICE
  void cache_metadata_vector_words(int c, int group, int cache_n) const {
    uint4 const scale_words = *reinterpret_cast<uint4 const*>(
        scales_ + group * params_.size_n + cache_n);
    half2 const* scale_vec = reinterpret_cast<half2 const*>(&scale_words);
    half2* scale_cache = cached_scales_ + c * 4;
    scale_cache[0] = scale_vec[0];
    scale_cache[1] = scale_vec[1];
    scale_cache[2] = scale_vec[2];
    scale_cache[3] = scale_vec[3];
  }

  CUTLASS_DEVICE
  void cache_current_group_metadata(int group) const {
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
      int const cache_n =
          n_offset_ + thread_offset_.contiguous() +
          c * ThreadMap::Delta::kContiguous;
      if constexpr (Shape::kN == 256) {
        cache_metadata_vector_words(c, group, cache_n);
      } else {
        cache_metadata_lane_vectors(c, group, cache_n);
      }
    }
  }

  CUTLASS_DEVICE
  void load_cta_n_aligned(Fragment& frag) const {
    if constexpr (ThreadMap::Iterations::kStrided == 1) {
      if constexpr (ThreadMap::Iterations::kContiguous == 4) {
        uint4 const qwords =
            *reinterpret_cast<uint4 const*>(qweight_ + qweight_base_offset_);
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
          constexpr int kAccess = ThreadMap::kElementsPerAccess;
          int const frag_base = c * kAccess;
          uint32_t const qword = qword_from_vector(qwords, c);
          half2 const* scale_vec = cached_scales_ + c * 4;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU4B8.id(), false>(
              static_cast<int>(qword), deq);
          frag_vec[0] = __hmul2(deq[0], scale_vec[0]);
          frag_vec[1] = __hmul2(deq[1], scale_vec[1]);
          marlin::dequant<half2, vllm::kU4B8.id(), false>(
              static_cast<int>(qword >> 8), deq);
          frag_vec[2] = __hmul2(deq[0], scale_vec[2]);
          frag_vec[3] = __hmul2(deq[1], scale_vec[3]);
        }
      } else if constexpr (ThreadMap::Iterations::kContiguous == 2) {
        uint2 const qwords =
            *reinterpret_cast<uint2 const*>(qweight_ + qweight_base_offset_);
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
          constexpr int kAccess = ThreadMap::kElementsPerAccess;
          int const frag_base = c * kAccess;
          uint32_t const qword = qword_from_vector(qwords, c);
          half2 const* scale_vec = cached_scales_ + c * 4;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU4B8.id(), false>(
              static_cast<int>(qword), deq);
          frag_vec[0] = __hmul2(deq[0], scale_vec[0]);
          frag_vec[1] = __hmul2(deq[1], scale_vec[1]);
          marlin::dequant<half2, vllm::kU4B8.id(), false>(
              static_cast<int>(qword >> 8), deq);
          frag_vec[2] = __hmul2(deq[0], scale_vec[2]);
          frag_vec[3] = __hmul2(deq[1], scale_vec[3]);
        }
      } else {
        static_assert(ThreadMap::Iterations::kContiguous == 1,
                      "Unsupported SM70 kU4B8 contiguous iteration count.");
        uint32_t const qword = qweight_[qweight_base_offset_];
        half2 const* scale_vec = cached_scales_;

        half2 deq[2];
        half2* frag_vec = reinterpret_cast<half2*>(frag.data());
        marlin::dequant<half2, vllm::kU4B8.id(), false>(
            static_cast<int>(qword), deq);
        frag_vec[0] = __hmul2(deq[0], scale_vec[0]);
        frag_vec[1] = __hmul2(deq[1], scale_vec[1]);
        marlin::dequant<half2, vllm::kU4B8.id(), false>(
            static_cast<int>(qword >> 8), deq);
        frag_vec[2] = __hmul2(deq[0], scale_vec[2]);
        frag_vec[3] = __hmul2(deq[1], scale_vec[3]);
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
        if constexpr (ThreadMap::Iterations::kContiguous == 4) {
          uint4 const qwords =
              *reinterpret_cast<uint4 const*>(qweight_ + qweight_offset(s, 0));
          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
            constexpr int kAccess = ThreadMap::kElementsPerAccess;
            int const frag_base =
                (c + s * ThreadMap::Iterations::kContiguous) * kAccess;
            uint32_t const qword = qword_from_vector(qwords, c);
            half2 const* scale_vec = cached_scales_ + c * 4;

            half2 deq[2];
            half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
            marlin::dequant<half2, vllm::kU4B8.id(), false>(
                static_cast<int>(qword), deq);
            frag_vec[0] = __hmul2(deq[0], scale_vec[0]);
            frag_vec[1] = __hmul2(deq[1], scale_vec[1]);
            marlin::dequant<half2, vllm::kU4B8.id(), false>(
                static_cast<int>(qword >> 8), deq);
            frag_vec[2] = __hmul2(deq[0], scale_vec[2]);
            frag_vec[3] = __hmul2(deq[1], scale_vec[3]);
          }
        } else if constexpr (ThreadMap::Iterations::kContiguous == 2) {
          uint2 const qwords =
              *reinterpret_cast<uint2 const*>(qweight_ + qweight_offset(s, 0));
          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
            constexpr int kAccess = ThreadMap::kElementsPerAccess;
            int const frag_base =
                (c + s * ThreadMap::Iterations::kContiguous) * kAccess;
            uint32_t const qword = qword_from_vector(qwords, c);
            half2 const* scale_vec = cached_scales_ + c * 4;

            half2 deq[2];
            half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
            marlin::dequant<half2, vllm::kU4B8.id(), false>(
                static_cast<int>(qword), deq);
            frag_vec[0] = __hmul2(deq[0], scale_vec[0]);
            frag_vec[1] = __hmul2(deq[1], scale_vec[1]);
            marlin::dequant<half2, vllm::kU4B8.id(), false>(
                static_cast<int>(qword >> 8), deq);
            frag_vec[2] = __hmul2(deq[0], scale_vec[2]);
            frag_vec[3] = __hmul2(deq[1], scale_vec[3]);
          }
        } else {
          static_assert(ThreadMap::Iterations::kContiguous == 1,
                        "Unsupported SM70 kU4B8 contiguous iteration count.");
          uint32_t const qword = qweight_[qweight_offset(s, 0)];
          constexpr int kAccess = ThreadMap::kElementsPerAccess;
          int const frag_base = s * kAccess;
          half2 const* scale_vec = cached_scales_;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU4B8.id(), false>(
              static_cast<int>(qword), deq);
          frag_vec[0] = __hmul2(deq[0], scale_vec[0]);
          frag_vec[1] = __hmul2(deq[1], scale_vec[1]);
          marlin::dequant<half2, vllm::kU4B8.id(), false>(
              static_cast<int>(qword >> 8), deq);
          frag_vec[2] = __hmul2(deq[0], scale_vec[2]);
          frag_vec[3] = __hmul2(deq[1], scale_vec[3]);
        }
      }
    }
  }

  CUTLASS_DEVICE
  void load(Fragment& frag) const {
    if (!mask_enabled_) {
      return;
    }

    if constexpr (kGroupSize != -1) {
      int const first_logical_k = k_offset_ + thread_offset_.strided();
      cache_current_group_metadata(scale_group(first_logical_k));
    }

    load_cta_n_aligned(frag);
  }
};

struct Sm70U4B8GemmSpec {
  template <typename Shape, typename ThreadMap, int GroupSize>
  using IteratorB = Sm70U4B8IteratorB<Shape, ThreadMap, GroupSize>;
};

template <int CtaM, int CtaN, int Warps, int GroupSize>
using Sm70U4B8GemmTraits =
    Sm70DenseGemmTraits<Sm70U4B8GemmSpec, CtaM, CtaN, Warps, GroupSize>;

template <int CtaM, int CtaN, int Warps, int GroupSize>
__global__ __launch_bounds__(Warps * 32, 1)
void sm70_marlin_u4b8_gemm_kernel(
    cutlass::half_t const* __restrict__ a,
    uint32_t const* __restrict__ b_q_weight,
    cutlass::half_t const* __restrict__ b_scales,
    cutlass::half_t* __restrict__ c, int m, int n, int k, int lda) {
  using Traits =
      Sm70U4B8GemmTraits<CtaM, CtaN, Warps, GroupSize>;
  using Mma = typename Traits::Mma;
  using Epilogue = typename Traits::Epilogue;

  extern __shared__ char smem[];
  auto& shared_storage =
      *reinterpret_cast<typename Traits::SharedStorage*>(smem);

  int thread_idx = threadIdx.x;
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_idx = threadIdx.x % 32;

  cutlass::MatrixCoord tb_offset_A{int(blockIdx.x) * CtaM, 0};
  cutlass::MatrixCoord tb_offset_B{0, int(blockIdx.y) * CtaN};
  cutlass::MatrixCoord tb_offset_C{int(blockIdx.x) * CtaM,
                                   int(blockIdx.y) * CtaN};

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

template <int CtaM, int CtaN, int Warps, int GroupSize>
__global__ __launch_bounds__(Warps * 32, 1)
void sm70_marlin_u4b8_gemm_splitk_kernel(
    cutlass::half_t const* __restrict__ a,
    uint32_t const* __restrict__ b_q_weight,
    cutlass::half_t const* __restrict__ b_scales,
    float* __restrict__ c_tmp, int m, int n, int k, int lda, int split_k) {
  using Traits = Sm70U4B8GemmTraits<CtaM, CtaN, Warps, GroupSize>;
  using Mma = typename Traits::Mma;
  using AtomicEpilogue = Sm70DenseAtomicFp32Epilogue<Traits>;

  extern __shared__ char smem[];
  auto& shared_storage =
      *reinterpret_cast<typename Traits::SharedStorage*>(smem);

  int const thread_idx = threadIdx.x;
  int const warp_idx = cutlass::canonical_warp_idx_sync();
  int const lane_idx = threadIdx.x % 32;
  Sm70DenseSplitKPartition const partition =
      sm70_dense_splitk_partition<GroupSize>(k, split_k, int(blockIdx.z));
  if (partition.partition_k == 0) {
    return;
  }

  cutlass::MatrixCoord tb_offset_A{int(blockIdx.x) * CtaM,
                                   partition.k_begin};
  cutlass::MatrixCoord tb_offset_B{partition.k_begin, int(blockIdx.y) * CtaN};
  cutlass::MatrixCoord tb_offset_C{int(blockIdx.x) * CtaM,
                                   int(blockIdx.y) * CtaN};

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

  int const gemm_k_iterations = partition.partition_k / kCtaK;
  mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

  typename AtomicEpilogue::OutputTileIterator iterator_D(
      typename AtomicEpilogue::OutputTileIterator::Params(layout_c),
      reinterpret_cast<cutlass::half_t*>(c_tmp), cutlass::MatrixCoord(m, n),
      thread_idx, tb_offset_C);

  AtomicEpilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx,
                          lane_idx);
  epilogue(iterator_D, accumulators, c_tmp, n);
}

}  // namespace

template <int CtaM, int CtaN, int Warps, int GroupSize>
torch::Tensor launch_sm70_marlin_u4b8_gemm(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, int64_t size_m, int64_t size_n, int64_t size_k,
    int split_k, std::optional<torch::Tensor> const& c_tmp_or_none) {
  auto kernel =
      sm70_marlin_u4b8_gemm_kernel<CtaM, CtaN, Warps, GroupSize>;
  using SharedStorage = typename Sm70U4B8GemmTraits<
      CtaM, CtaN, Warps, GroupSize>::SharedStorage;
  size_t smem_bytes = configure_dynamic_smem<SharedStorage>(kernel);

  dim3 block(Warps * 32);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

  if (split_k == 1) {
    dim3 grid = cta_grid(size_m, size_n, CtaM, CtaN);
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

  TORCH_CHECK(size_k % int64_t(kCtaK) == 0, kSm70MarlinU4B8SplitKEnv,
              " requires K divisible by 32 for split_k > 1. Got K=", size_k,
              ", split_k=", split_k, ".");

  auto split_kernel =
      sm70_marlin_u4b8_gemm_splitk_kernel<CtaM, CtaN, Warps, GroupSize>;
  smem_bytes = configure_dynamic_smem<SharedStorage>(split_kernel);

  int64_t const numel = size_m * size_n;
  auto c_tmp =
      sm70_dense_get_splitk_ctmp(c_tmp_or_none, a.device(), numel);
  C10_CUDA_CHECK(cudaMemsetAsync(
      c_tmp.data_ptr<float>(), 0,
      static_cast<size_t>(numel) * sizeof(float), stream));

  dim3 grid = cta_grid(size_m, size_n, CtaM, CtaN);
  int const active_split_k =
      sm70_dense_active_split_k(static_cast<int>(size_k), split_k);
  grid.z = static_cast<unsigned>(active_split_k);
  split_kernel<<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<cutlass::half_t const*>(a.data_ptr<at::Half>()),
      reinterpret_cast<uint32_t const*>(b_q_weight.data_ptr<int32_t>()),
      reinterpret_cast<cutlass::half_t const*>(b_scales.data_ptr<at::Half>()),
      c_tmp.data_ptr<float>(),
      static_cast<int>(size_m), static_cast<int>(size_n),
      static_cast<int>(size_k), static_cast<int>(a.stride(0)), split_k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  launch_sm70_dense_fp32_to_fp16(
      c_tmp.data_ptr<float>(),
      reinterpret_cast<cutlass::half_t*>(c.data_ptr<at::Half>()), numel,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

struct Sm70U4B8Launcher {
  torch::Tensor& a;
  torch::Tensor& c;
  torch::Tensor& b_q_weight;
  torch::Tensor& b_scales;
  int64_t size_m;
  int64_t size_n;
  int64_t size_k;
  int split_k;
  std::optional<torch::Tensor> const& c_tmp_or_none;

  template <int CtaM, int CtaN, int Warps, int GroupSize>
  torch::Tensor operator()() const {
    return launch_sm70_marlin_u4b8_gemm<CtaM, CtaN, Warps, GroupSize>(
        a, c, b_q_weight, b_scales, size_m, size_n, size_k, split_k,
        c_tmp_or_none);
  }
};

torch::Tensor sm70_marlin_u4b8_gemm(torch::Tensor& a, torch::Tensor& c,
                                    torch::Tensor& b_q_weight,
                                    torch::Tensor& b_scales, int64_t size_m,
                                    int64_t size_n, int64_t size_k,
                                    int64_t group_size,
                                    std::optional<torch::Tensor> const&
                                        c_tmp_or_none) {
  c10::cuda::CUDAGuard device_guard(a.device());

  char const* env_name = "SM70_MARLIN_U4B8_CTA";
  Sm70DenseCtaGeometry const geometry =
      resolve_sm70_dense_cta_geometry(env_name, size_n);
  check_sm70_dense_cta_geometry(env_name, geometry);
  check_sm70_dense_n_tile_alignment(env_name, geometry, size_n);
  int const split_k = parse_sm70_dense_split_k(kSm70MarlinU4B8SplitKEnv);
  Sm70U4B8Launcher const launcher{
      a, c, b_q_weight, b_scales, size_m, size_n, size_k, split_k,
      c_tmp_or_none};
  return dispatch_geometry(launcher, geometry, size_n, size_k, group_size,
                           "uint4b8");
}

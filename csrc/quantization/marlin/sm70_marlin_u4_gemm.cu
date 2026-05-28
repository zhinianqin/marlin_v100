#include "core/registration.h"
#include "quantization/marlin/dequant.h"
#include "quantization/marlin/sm70_dense_common.cuh"
#include "quantization/marlin/sm70_dense_gemm.cuh"
#include "quantization/marlin/sm70_dense_iterator_utils.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
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
using marlin::sm70_dense::Sm70DenseGemmTraits;
using marlin::sm70_dense::check_sm70_dense_cta_geometry;
using marlin::sm70_dense::check_sm70_dense_n_tile_alignment;
using marlin::sm70_dense::configure_dynamic_smem;
using marlin::sm70_dense::cta_grid;
using marlin::sm70_dense::dispatch_geometry;
using marlin::sm70_dense::kCtaK;
using marlin::sm70_dense::kMacroN;
using marlin::sm70_dense::kMacroNTiles;
using marlin::sm70_dense::kQuantTileK;
using marlin::sm70_dense::kQuantTileN;
using marlin::sm70_dense::parse_sm70_dense_cta_geometry;
using marlin::sm70_dense::qword_from_vector;
using marlin::sm70_dense::u4_macro_n_qweight_offset_from_logical;

namespace {

constexpr int kU4ValuesPerWord = 8;
constexpr char const* kSm70MarlinU4SplitKEnv = "SM70_MARLIN_U4_SPLIT_K";

int parse_sm70_marlin_u4_split_k() {
  char const* env = std::getenv(kSm70MarlinU4SplitKEnv);
  if (env == nullptr || env[0] == '\0') {
    return 1;
  }

  std::string value(env);
  if (value == "1" || value == "2" || value == "4" || value == "8") {
    return std::stoi(value);
  }
  TORCH_CHECK(false, kSm70MarlinU4SplitKEnv,
              " supports only 1, 2, 4, or 8. Got: ", env);
  return 1;
}

template <typename Shape_, typename ThreadMap_, int GroupSize_>
class Sm70U4ZpIteratorB {
 public:
  using Shape = Shape_;
  using ThreadMap = ThreadMap_;
  static int const kGroupSize = GroupSize_;
  using Element = cutlass::half_t;
  using Fragment = cutlass::Array<
      Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
  static_assert(Shape::kK == kCtaK,
                "The SM70 kU4 IteratorB expects CTA_K=32.");
  static_assert(Shape::kN == 64 || Shape::kN == 128 || Shape::kN == 256,
                "The SM70 kU4 IteratorB expects CTA_N in {64, 128, 256}.");
  static_assert(ThreadMap::Iterations::kContiguous ==
                    Shape::kN / kQuantTileN,
                "The SM70 kU4 IteratorB expects one contiguous iteration per "
                "64-column quant tile.");
  static_assert(ThreadMap::Delta::kContiguous == kQuantTileN,
                "The SM70 kU4 IteratorB expects 64-column deltas.");
  static_assert(ThreadMap::kElementsPerAccess == kU4ValuesPerWord,
                "The SM70 kU4 IteratorB expects one packed int4 word per "
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
  half const* zp_;
  Params params_;
  cutlass::layout::PitchLinearCoord thread_offset_;
  int qweight_base_offset_;
  int k_offset_;
  int n_offset_;
  int aligned_k_step_;
  bool mask_enabled_;
  mutable half2 cached_scales_[ThreadMap::Iterations::kContiguous * 4];
  mutable half2 cached_zp_[ThreadMap::Iterations::kContiguous * 4];

 public:
  CUTLASS_DEVICE
  Sm70U4ZpIteratorB(Params const& params, uint32_t const* qweight,
                    half const* scales, half const* zp, int thread_id,
                    cutlass::MatrixCoord const& threadblock_offset)
      : qweight_(qweight),
        scales_(scales),
        zp_(zp),
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
      // GroupSize=-1 keeps scale and fp16 zero points stable for
      // every K tile. Cache both planes once before the MMA mainloop.
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

        half2 const* zp_vec = reinterpret_cast<half2 const*>(zp_ + cache_n);
        half2* zp_cache = cached_zp_ + c * 4;
        zp_cache[0] = zp_vec[0];
        zp_cache[1] = zp_vec[1];
        zp_cache[2] = zp_vec[2];
        zp_cache[3] = zp_vec[3];
      }
    }
  }

  CUTLASS_DEVICE
  Sm70U4ZpIteratorB& operator++() {
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
                    "SM70 kU4 prototype only specializes group sizes "
                    "-1, 32, 64, and 128.");
      return logical_k / kGroupSize;
    }
  }

  CUTLASS_DEVICE
  static int qweight_offset_from_logical(Params const& params, int logical_k,
                                         int logical_n) {
    return u4_macro_n_qweight_offset_from_logical(params.size_n, logical_k,
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

    half2 const* zp_vec = reinterpret_cast<half2 const*>(
        zp_ + group * params_.size_n + cache_n);
    half2* zp_cache = cached_zp_ + c * 4;
    zp_cache[0] = zp_vec[0];
    zp_cache[1] = zp_vec[1];
    zp_cache[2] = zp_vec[2];
    zp_cache[3] = zp_vec[3];
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

    uint4 const zp_words = *reinterpret_cast<uint4 const*>(
        zp_ + group * params_.size_n + cache_n);
    half2 const* zp_vec = reinterpret_cast<half2 const*>(&zp_words);
    half2* zp_cache = cached_zp_ + c * 4;
    zp_cache[0] = zp_vec[0];
    zp_cache[1] = zp_vec[1];
    zp_cache[2] = zp_vec[2];
    zp_cache[3] = zp_vec[3];
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
  void load_macro_n_aligned(Fragment& frag) const {
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
          half2 const* zp_vec = cached_zp_ + c * 4;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU4.id(), false>(
              static_cast<int>(qword), deq);
          frag_vec[0] = __hfma2(deq[0], scale_vec[0], __hneg2(zp_vec[0]));
          frag_vec[1] = __hfma2(deq[1], scale_vec[1], __hneg2(zp_vec[1]));
          marlin::dequant<half2, vllm::kU4.id(), false>(
              static_cast<int>(qword >> 8), deq);
          frag_vec[2] = __hfma2(deq[0], scale_vec[2], __hneg2(zp_vec[2]));
          frag_vec[3] = __hfma2(deq[1], scale_vec[3], __hneg2(zp_vec[3]));
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
          half2 const* zp_vec = cached_zp_ + c * 4;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU4.id(), false>(
              static_cast<int>(qword), deq);
          frag_vec[0] = __hfma2(deq[0], scale_vec[0], __hneg2(zp_vec[0]));
          frag_vec[1] = __hfma2(deq[1], scale_vec[1], __hneg2(zp_vec[1]));
          marlin::dequant<half2, vllm::kU4.id(), false>(
              static_cast<int>(qword >> 8), deq);
          frag_vec[2] = __hfma2(deq[0], scale_vec[2], __hneg2(zp_vec[2]));
          frag_vec[3] = __hfma2(deq[1], scale_vec[3], __hneg2(zp_vec[3]));
        }
      } else {
        static_assert(ThreadMap::Iterations::kContiguous == 1,
                      "Unsupported SM70 kU4 contiguous iteration count.");
        uint32_t const qword = qweight_[qweight_base_offset_];
        half2 const* scale_vec = cached_scales_;
        half2 const* zp_vec = cached_zp_;

        half2 deq[2];
        half2* frag_vec = reinterpret_cast<half2*>(frag.data());
        marlin::dequant<half2, vllm::kU4.id(), false>(
            static_cast<int>(qword), deq);
        frag_vec[0] = __hfma2(deq[0], scale_vec[0], __hneg2(zp_vec[0]));
        frag_vec[1] = __hfma2(deq[1], scale_vec[1], __hneg2(zp_vec[1]));
        marlin::dequant<half2, vllm::kU4.id(), false>(
            static_cast<int>(qword >> 8), deq);
        frag_vec[2] = __hfma2(deq[0], scale_vec[2], __hneg2(zp_vec[2]));
        frag_vec[3] = __hfma2(deq[1], scale_vec[3], __hneg2(zp_vec[3]));
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
            half2 const* zp_vec = cached_zp_ + c * 4;

            half2 deq[2];
            half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
            marlin::dequant<half2, vllm::kU4.id(), false>(
                static_cast<int>(qword), deq);
            frag_vec[0] =
                __hfma2(deq[0], scale_vec[0], __hneg2(zp_vec[0]));
            frag_vec[1] =
                __hfma2(deq[1], scale_vec[1], __hneg2(zp_vec[1]));
            marlin::dequant<half2, vllm::kU4.id(), false>(
                static_cast<int>(qword >> 8), deq);
            frag_vec[2] =
                __hfma2(deq[0], scale_vec[2], __hneg2(zp_vec[2]));
            frag_vec[3] =
                __hfma2(deq[1], scale_vec[3], __hneg2(zp_vec[3]));
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
            half2 const* zp_vec = cached_zp_ + c * 4;

            half2 deq[2];
            half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
            marlin::dequant<half2, vllm::kU4.id(), false>(
                static_cast<int>(qword), deq);
            frag_vec[0] =
                __hfma2(deq[0], scale_vec[0], __hneg2(zp_vec[0]));
            frag_vec[1] =
                __hfma2(deq[1], scale_vec[1], __hneg2(zp_vec[1]));
            marlin::dequant<half2, vllm::kU4.id(), false>(
                static_cast<int>(qword >> 8), deq);
            frag_vec[2] =
                __hfma2(deq[0], scale_vec[2], __hneg2(zp_vec[2]));
            frag_vec[3] =
                __hfma2(deq[1], scale_vec[3], __hneg2(zp_vec[3]));
          }
        } else {
          static_assert(ThreadMap::Iterations::kContiguous == 1,
                        "Unsupported SM70 kU4 contiguous iteration count.");
          uint32_t const qword = qweight_[qweight_offset(s, 0)];
          constexpr int kAccess = ThreadMap::kElementsPerAccess;
          int const frag_base = s * kAccess;
          half2 const* scale_vec = cached_scales_;
          half2 const* zp_vec = cached_zp_;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU4.id(), false>(
              static_cast<int>(qword), deq);
          frag_vec[0] = __hfma2(deq[0], scale_vec[0], __hneg2(zp_vec[0]));
          frag_vec[1] = __hfma2(deq[1], scale_vec[1], __hneg2(zp_vec[1]));
          marlin::dequant<half2, vllm::kU4.id(), false>(
              static_cast<int>(qword >> 8), deq);
          frag_vec[2] = __hfma2(deq[0], scale_vec[2], __hneg2(zp_vec[2]));
          frag_vec[3] = __hfma2(deq[1], scale_vec[3], __hneg2(zp_vec[3]));
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

    load_macro_n_aligned(frag);
  }
};

struct Sm70U4ZpGemmSpec {
  template <typename Shape, typename ThreadMap, int GroupSize>
  using IteratorB = Sm70U4ZpIteratorB<Shape, ThreadMap, GroupSize>;
};

template <int CtaM, int CtaN, int Warps, int GroupSize>
using Sm70U4ZpGemmTraits =
    Sm70DenseGemmTraits<Sm70U4ZpGemmSpec, CtaM, CtaN, Warps, GroupSize>;

template <typename Traits>
class Sm70U4AtomicFp32Epilogue {
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

  CUTLASS_DEVICE
  void atomic_store_fragment(OutputTileIterator const& destination_iterator,
                             typename SharedLoadIterator::Fragment const& frag,
                             float* __restrict__ c_tmp, int n) const {
    float const* frag_ptr = reinterpret_cast<float const*>(&frag);
    int const thread_start_row = destination_iterator.thread_start_row();
    int const thread_start_column = destination_iterator.thread_start_column();
    int const extent_row = destination_iterator.extent_row();
    int const extent_column = destination_iterator.extent_column();

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
          bool const row_guard = logical_row < extent_row;

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn;
               ++column) {
            int const logical_column_base =
                thread_start_column + column * ThreadMap::Delta::kColumn;
            int const frag_base =
                (frag_row_idx * ThreadMap::Iterations::kColumn + column) *
                ThreadMap::kElementsPerAccess;

            CUTLASS_PRAGMA_UNROLL
            for (int e = 0; e < ThreadMap::kElementsPerAccess; ++e) {
              int const logical_column = logical_column_base + e;
              if (row_guard && logical_column < extent_column) {
                atomicAdd(c_tmp + int64_t(logical_row) * n + logical_column,
                          frag_ptr[frag_base + e]);
              }
            }
          }
        }
      }
    }
  }

 public:
  CUTLASS_DEVICE
  Sm70U4AtomicFp32Epilogue(SharedStorage& shared_storage, int thread_idx,
                           int warp_idx, int lane_idx)
      : warp_tile_iterator_(shared_storage.reference(), lane_idx),
        shared_load_iterator_(shared_storage.reference(), thread_idx) {
    using WarpCount = typename CutlassEpilogue::WarpCount;
    int const warp_k = warp_idx / (WarpCount::kM * WarpCount::kN);
    int const warp_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
    int const warp_m = warp_mn % WarpCount::kM;
    int const warp_n = warp_mn / WarpCount::kM;

    cutlass::MatrixCoord warp_offset{warp_k * WarpCount::kM + warp_m,
                                     warp_n};
    warp_tile_iterator_.add_tile_offset(warp_offset);
  }

  CUTLASS_DEVICE
  void operator()(OutputTileIterator destination_iterator,
                  AccumulatorTile const& accumulators,
                  float* __restrict__ c_tmp, int n) {
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
      shared_load_iterator_.load(aligned_accum_fragment);

      if (CutlassEpilogue::kPartitionsK > 1) {
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

      atomic_store_fragment(destination_iterator, aligned_accum_fragment, c_tmp,
                            n);
      ++destination_iterator;
    }
  }
};

template <int CtaM, int CtaN, int Warps, int GroupSize>
__global__ __launch_bounds__(Warps * 32, 1)
void sm70_marlin_u4_gemm_kernel(
    cutlass::half_t const* __restrict__ a,
    uint32_t const* __restrict__ b_q_weight,
    cutlass::half_t const* __restrict__ b_scales,
    cutlass::half_t const* __restrict__ b_zeros,
    cutlass::half_t* __restrict__ c, int m, int n, int k, int lda) {
  using Traits = Sm70U4ZpGemmTraits<CtaM, CtaN, Warps, GroupSize>;
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
      reinterpret_cast<half const*>(b_scales),
      reinterpret_cast<half const*>(b_zeros), thread_idx, tb_offset_B);

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
void sm70_marlin_u4_gemm_splitk_kernel(
    cutlass::half_t const* __restrict__ a,
    uint32_t const* __restrict__ b_q_weight,
    cutlass::half_t const* __restrict__ b_scales,
    cutlass::half_t const* __restrict__ b_zeros,
    float* __restrict__ c_tmp, int m, int n, int k, int lda, int split_k) {
  using Traits = Sm70U4ZpGemmTraits<CtaM, CtaN, Warps, GroupSize>;
  using Mma = typename Traits::Mma;
  using AtomicEpilogue = Sm70U4AtomicFp32Epilogue<Traits>;

  extern __shared__ char smem[];
  auto& shared_storage =
      *reinterpret_cast<typename Traits::SharedStorage*>(smem);

  int const thread_idx = threadIdx.x;
  int const warp_idx = cutlass::canonical_warp_idx_sync();
  int const lane_idx = threadIdx.x % 32;
  int const k_partition = k / split_k;
  int const k_begin = int(blockIdx.z) * k_partition;

  cutlass::MatrixCoord tb_offset_A{int(blockIdx.x) * CtaM, k_begin};
  cutlass::MatrixCoord tb_offset_B{k_begin, int(blockIdx.y) * CtaN};
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
      reinterpret_cast<half const*>(b_scales),
      reinterpret_cast<half const*>(b_zeros), thread_idx, tb_offset_B);

  Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
  typename Mma::FragmentC accumulators;
  accumulators.clear();

  int const gemm_k_iterations = k_partition / kCtaK;
  mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

  typename AtomicEpilogue::OutputTileIterator iterator_D(
      typename AtomicEpilogue::OutputTileIterator::Params(layout_c),
      reinterpret_cast<cutlass::half_t*>(c_tmp), cutlass::MatrixCoord(m, n),
      thread_idx, tb_offset_C);

  AtomicEpilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx,
                          lane_idx);
  epilogue(iterator_D, accumulators, c_tmp, n);
}

__global__ void sm70_marlin_u4_fp32_to_fp16_kernel(
    float const* __restrict__ c_tmp, cutlass::half_t* __restrict__ c,
    int64_t numel) {
  int64_t const base =
      (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  half* c_half = reinterpret_cast<half*>(c);

  if (base + 3 < numel) {
    float4 const values = *reinterpret_cast<float4 const*>(c_tmp + base);
    half2* c_half2 = reinterpret_cast<half2*>(c_half + base);
    c_half2[0] = __floats2half2_rn(values.x, values.y);
    c_half2[1] = __floats2half2_rn(values.z, values.w);
    return;
  }

  for (int offset = 0; offset < 4; ++offset) {
    int64_t const idx = base + offset;
    if (idx < numel) {
      c_half[idx] = __float2half_rn(c_tmp[idx]);
    }
  }
}

}  // namespace

template <int CtaM, int CtaN, int Warps, int GroupSize>
torch::Tensor launch_sm70_marlin_u4_gemm(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, torch::Tensor& b_zeros, int64_t size_m,
    int64_t size_n, int64_t size_k, int split_k, bool use_fp32_reduce) {
  auto kernel = sm70_marlin_u4_gemm_kernel<CtaM, CtaN, Warps, GroupSize>;
  using SharedStorage = typename Sm70U4ZpGemmTraits<
      CtaM, CtaN, Warps, GroupSize>::SharedStorage;
  size_t smem_bytes = configure_dynamic_smem<SharedStorage>(kernel);
  dim3 block(Warps * 32);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

  if (split_k == 1) {
    dim3 grid = cta_grid(size_m, size_n, CtaM, CtaN);
    kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<cutlass::half_t const*>(a.data_ptr<at::Half>()),
        reinterpret_cast<uint32_t const*>(b_q_weight.data_ptr<int32_t>()),
        reinterpret_cast<cutlass::half_t const*>(
            b_scales.data_ptr<at::Half>()),
        reinterpret_cast<cutlass::half_t const*>(b_zeros.data_ptr<at::Half>()),
        reinterpret_cast<cutlass::half_t*>(c.data_ptr<at::Half>()),
        static_cast<int>(size_m), static_cast<int>(size_n),
        static_cast<int>(size_k), static_cast<int>(a.stride(0)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
  }

  TORCH_CHECK(use_fp32_reduce, kSm70MarlinU4SplitKEnv,
              " requires use_fp32_reduce=True for split_k > 1.");
  TORCH_CHECK(size_k % (int64_t(kCtaK) * split_k) == 0,
              kSm70MarlinU4SplitKEnv,
              " requires K divisible by 32 * split_k for split_k > 1. Got K=",
              size_k, ", split_k=", split_k, ".");

  auto split_kernel =
      sm70_marlin_u4_gemm_splitk_kernel<CtaM, CtaN, Warps, GroupSize>;
  smem_bytes = configure_dynamic_smem<SharedStorage>(split_kernel);

  auto c_tmp = torch::empty(
      {size_m, size_n},
      torch::TensorOptions().dtype(at::kFloat).device(a.device()));
  C10_CUDA_CHECK(cudaMemsetAsync(
      c_tmp.data_ptr<float>(), 0,
      static_cast<size_t>(c_tmp.numel()) * sizeof(float), stream));

  dim3 grid = cta_grid(size_m, size_n, CtaM, CtaN);
  grid.z = static_cast<unsigned>(split_k);
  split_kernel<<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<cutlass::half_t const*>(a.data_ptr<at::Half>()),
      reinterpret_cast<uint32_t const*>(b_q_weight.data_ptr<int32_t>()),
      reinterpret_cast<cutlass::half_t const*>(b_scales.data_ptr<at::Half>()),
      reinterpret_cast<cutlass::half_t const*>(b_zeros.data_ptr<at::Half>()),
      c_tmp.data_ptr<float>(),
      static_cast<int>(size_m), static_cast<int>(size_n),
      static_cast<int>(size_k), static_cast<int>(a.stride(0)), split_k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int64_t const numel = size_m * size_n;
  dim3 convert_block(256);
  dim3 convert_grid(static_cast<unsigned>(
      (numel + int64_t(convert_block.x) * 4 - 1) /
      (int64_t(convert_block.x) * 4)));
  sm70_marlin_u4_fp32_to_fp16_kernel<<<convert_grid, convert_block, 0,
                                       stream>>>(
      c_tmp.data_ptr<float>(),
      reinterpret_cast<cutlass::half_t*>(c.data_ptr<at::Half>()), numel);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

struct Sm70U4Launcher {
  torch::Tensor& a;
  torch::Tensor& c;
  torch::Tensor& b_q_weight;
  torch::Tensor& b_scales;
  torch::Tensor& b_zeros;
  int64_t size_m;
  int64_t size_n;
  int64_t size_k;
  int split_k;
  bool use_fp32_reduce;

  template <int CtaM, int CtaN, int Warps, int GroupSize>
  torch::Tensor operator()() const {
    return launch_sm70_marlin_u4_gemm<CtaM, CtaN, Warps, GroupSize>(
        a, c, b_q_weight, b_scales, b_zeros, size_m, size_n, size_k, split_k,
        use_fp32_reduce);
  }
};

torch::Tensor sm70_marlin_u4_gemm(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, torch::Tensor& b_zeros, int64_t size_m,
    int64_t size_n, int64_t size_k, int64_t group_size,
    bool use_fp32_reduce) {
  c10::cuda::CUDAGuard device_guard(a.device());

  char const* env_name = "SM70_MARLIN_U4_CTA";
  Sm70DenseCtaGeometry const geometry =
      parse_sm70_dense_cta_geometry(env_name);
  check_sm70_dense_cta_geometry(env_name, geometry);
  check_sm70_dense_n_tile_alignment(env_name, geometry, size_n);
  int const split_k = parse_sm70_marlin_u4_split_k();
  Sm70U4Launcher const launcher{
      a, c, b_q_weight, b_scales, b_zeros, size_m, size_n, size_k, split_k,
      use_fp32_reduce};
  return dispatch_geometry(launcher, geometry, size_n, size_k, group_size,
                           "uint4");
}

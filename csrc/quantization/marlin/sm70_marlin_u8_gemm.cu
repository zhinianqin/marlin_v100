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
using marlin::sm70_dense::Sm70TileMode;
using marlin::sm70_dense::check_sm70_dense_cta_geometry;
using marlin::sm70_dense::configure_dynamic_smem;
using marlin::sm70_dense::cta_grid;
using marlin::sm70_dense::dispatch_geometry;
using marlin::sm70_dense::initial_k_advance;
using marlin::sm70_dense::kCtaK;
using marlin::sm70_dense::kMacroN;
using marlin::sm70_dense::kMacroNTiles;
using marlin::sm70_dense::kQuantTileK;
using marlin::sm70_dense::kQuantTileN;
using marlin::sm70_dense::kSm70TileModeFullTile;
using marlin::sm70_dense::kSm70TileModeResidueK;
using marlin::sm70_dense::kSm70TileModeResidueN;
using marlin::sm70_dense::parse_sm70_dense_cta_geometry;
using marlin::sm70_dense::qword_from_vector;
using marlin::sm70_dense::u8_qweight_offset_from_logical;
using marlin::sm70_dense::u8_qweight_word_stride_from_logical;

namespace {

constexpr int kU8ValuesPerWord = 4;
constexpr int kU8ValuesPerAccess = 8;

template <typename Shape_, typename ThreadMap_, int GroupSize_,
          Sm70TileMode TileMode_>
class Sm70U8ZpBiasIteratorB {
 public:
  using Shape = Shape_;
  using ThreadMap = ThreadMap_;
  static int const kGroupSize = GroupSize_;
  static constexpr Sm70TileMode kTileMode = TileMode_;
  static constexpr bool kFullTile = kSm70TileModeFullTile<kTileMode>;
  static constexpr bool kResidueN = kSm70TileModeResidueN<kTileMode>;
  static constexpr bool kResidueK = kSm70TileModeResidueK<kTileMode>;
  using Element = cutlass::half_t;
  using Fragment = cutlass::Array<
      Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
  static_assert(Shape::kK == kCtaK,
                "The SM70 kU8 IteratorB expects CTA_K=32.");
  static_assert(Shape::kN == 64 || Shape::kN == 128 || Shape::kN == 256,
                "The SM70 kU8 IteratorB expects CTA_N in {64, 128, 256}.");
  static_assert(ThreadMap::Iterations::kContiguous ==
                    Shape::kN / kQuantTileN,
                "The SM70 kU8 IteratorB expects one contiguous iteration per "
                "64-column quant tile.");
  static_assert(ThreadMap::Delta::kContiguous == kQuantTileN,
                "The SM70 kU8 IteratorB expects 64-column deltas.");
  static_assert(ThreadMap::kElementsPerAccess == kU8ValuesPerAccess,
                "The SM70 kU8 IteratorB expects two packed int8 words per "
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
  half const* zp_bias_;
  Params params_;
  cutlass::layout::PitchLinearCoord thread_offset_;
  int qweight_base_offset_;
  int k_offset_;
  int n_offset_;
  int tile_k_end_;
  int next_k_advance_;
  bool mask_enabled_;
  mutable half2 cached_scales_[ThreadMap::Iterations::kContiguous * 4];
  mutable half2 cached_bias_[ThreadMap::Iterations::kContiguous * 4];

 public:
  CUTLASS_DEVICE
  Sm70U8ZpBiasIteratorB(Params const& params, uint32_t const* qweight,
                        half const* scales, half const* zp_bias, int thread_id,
                        cutlass::MatrixCoord const& threadblock_offset)
      : qweight_(qweight),
        scales_(scales),
        zp_bias_(zp_bias),
        params_(params),
        thread_offset_(ThreadMap::initial_offset(thread_id)),
        k_offset_(threadblock_offset.row()),
        n_offset_(threadblock_offset.column()),
        tile_k_end_(threadblock_offset.row() +
                    initial_k_advance(params.size_k)),
        next_k_advance_(initial_k_advance(params.size_k)),
        mask_enabled_(true) {
    int const logical_k = threadblock_offset.row() + thread_offset_.strided();
    int const logical_n =
        threadblock_offset.column() + thread_offset_.contiguous();
    qweight_base_offset_ =
        qweight_offset_from_logical(params_, logical_k, logical_n);
    if constexpr (kGroupSize == -1) {
      // GroupSize=-1 keeps scale and precomputed zero-point bias stable for
      // every K tile. Cache both planes once before the MMA mainloop.
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int const cache_n =
            n_offset_ + thread_offset_.contiguous() +
            c * ThreadMap::Delta::kContiguous;
        if constexpr (kResidueN) {
          if (cache_n + ThreadMap::kElementsPerAccess > params_.size_n) {
            continue;
          }
        }

        half2 const* scale_vec =
            reinterpret_cast<half2 const*>(scales_ + cache_n);
        half2* scale_cache = cached_scales_ + c * 4;
        scale_cache[0] = scale_vec[0];
        scale_cache[1] = scale_vec[1];
        scale_cache[2] = scale_vec[2];
        scale_cache[3] = scale_vec[3];

        half2 const* bias_vec =
            reinterpret_cast<half2 const*>(zp_bias_ + cache_n);
        half2* bias_cache = cached_bias_ + c * 4;
        bias_cache[0] = bias_vec[0];
        bias_cache[1] = bias_vec[1];
        bias_cache[2] = bias_vec[2];
        bias_cache[3] = bias_vec[3];
      }
    }
  }

  CUTLASS_DEVICE
  Sm70U8ZpBiasIteratorB& operator++() {
    int const k_advance = next_k_advance_;
    int const k_advance_qwords =
        (k_advance / kQuantTileK) * (params_.size_n * 4);
    k_offset_ += k_advance;
    int const next_tile_k_end = k_offset_ + Shape::kK;
    tile_k_end_ = next_tile_k_end < params_.size_k ? next_tile_k_end
                                                   : params_.size_k;
    next_k_advance_ = Shape::kK;
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
  static int initial_k_advance(int size_k) {
    return marlin::sm70_dense::initial_k_advance<Shape::kK>(size_k);
  }

  CUTLASS_DEVICE
  int scale_group(int logical_k) const {
    if constexpr (kGroupSize == -1) {
      return 0;
    } else {
      static_assert(kGroupSize == 32 || kGroupSize == 64 ||
                        kGroupSize == 128,
                    "SM70 kU8 prototype only specializes group sizes "
                    "-1, 32, 64, and 128.");
      return logical_k / kGroupSize;
    }
  }

  CUTLASS_DEVICE
  static int qweight_offset_from_logical(Params const& params, int logical_k,
                                         int logical_n) {
    return u8_qweight_offset_from_logical<kResidueN>(params.size_n, logical_k,
                                                     logical_n);
  }

  CUTLASS_DEVICE
  static int qweight_word_stride_from_logical(Params const& params,
                                              int logical_n) {
    return u8_qweight_word_stride_from_logical<kResidueN>(params.size_n,
                                                          logical_n);
  }

  CUTLASS_DEVICE
  int qweight_word_stride(int c) const {
    int const logical_n =
        n_offset_ + thread_offset_.contiguous() +
        c * ThreadMap::Delta::kContiguous;
    return qweight_word_stride_from_logical(params_, logical_n);
  }

  CUTLASS_DEVICE
  int qweight_offset(int s, int c, int word) const {
    if constexpr (ThreadMap::Iterations::kStrided == 1) {
      return qweight_base_offset_ + word * qweight_word_stride(c) + c;
    } else {
      int const logical_k =
          k_offset_ + thread_offset_.strided() +
          s * ThreadMap::Delta::kStrided;
      int const logical_n =
          n_offset_ + thread_offset_.contiguous() +
          c * ThreadMap::Delta::kContiguous;
      return qweight_offset_from_logical(params_, logical_k, logical_n) +
             word * qweight_word_stride_from_logical(params_, logical_n);
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

    half2 const* bias_vec = reinterpret_cast<half2 const*>(
        zp_bias_ + group * params_.size_n + cache_n);
    half2* bias_cache = cached_bias_ + c * 4;
    bias_cache[0] = bias_vec[0];
    bias_cache[1] = bias_vec[1];
    bias_cache[2] = bias_vec[2];
    bias_cache[3] = bias_vec[3];
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

    uint4 const bias_words = *reinterpret_cast<uint4 const*>(
        zp_bias_ + group * params_.size_n + cache_n);
    half2 const* bias_vec = reinterpret_cast<half2 const*>(&bias_words);
    half2* bias_cache = cached_bias_ + c * 4;
    bias_cache[0] = bias_vec[0];
    bias_cache[1] = bias_vec[1];
    bias_cache[2] = bias_vec[2];
    bias_cache[3] = bias_vec[3];
  }

  CUTLASS_DEVICE
  void cache_current_group_metadata(int group) const {
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
      int const cache_n =
          n_offset_ + thread_offset_.contiguous() +
          c * ThreadMap::Delta::kContiguous;
      if constexpr (kResidueN) {
        if (cache_n + ThreadMap::kElementsPerAccess > params_.size_n) {
          continue;
        }
      }

      if constexpr (kTileMode != Sm70TileMode::FullTile || Shape::kN == 256) {
        cache_metadata_vector_words(c, group, cache_n);
      } else {
        cache_metadata_lane_vectors(c, group, cache_n);
      }
    }
  }

  CUTLASS_DEVICE
  void load_full_tile(Fragment& frag) const {
    if constexpr (ThreadMap::Iterations::kStrided == 1) {
      if constexpr (ThreadMap::Iterations::kContiguous == 4) {
        uint4 const qwords0 =
            *reinterpret_cast<uint4 const*>(qweight_ + qweight_base_offset_);
        uint4 const qwords1 =
            *reinterpret_cast<uint4 const*>(
                qweight_ + qweight_base_offset_ + qweight_word_stride(0));
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
          constexpr int kAccess = ThreadMap::kElementsPerAccess;
          int const frag_base = c * kAccess;
          uint32_t const qword0 = qword_from_vector(qwords0, c);
          uint32_t const qword1 = qword_from_vector(qwords1, c);
          half2 const* scale_vec = cached_scales_ + c * 4;
          half2 const* bias_vec = cached_bias_ + c * 4;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU8.id(), false>(
              static_cast<int>(qword0), deq);
          frag_vec[0] = __hfma2(deq[0], scale_vec[0], bias_vec[0]);
          frag_vec[1] = __hfma2(deq[1], scale_vec[1], bias_vec[1]);
          marlin::dequant<half2, vllm::kU8.id(), false>(
              static_cast<int>(qword1), deq);
          frag_vec[2] = __hfma2(deq[0], scale_vec[2], bias_vec[2]);
          frag_vec[3] = __hfma2(deq[1], scale_vec[3], bias_vec[3]);
        }
      } else if constexpr (ThreadMap::Iterations::kContiguous == 2) {
        uint2 const qwords0 =
            *reinterpret_cast<uint2 const*>(qweight_ + qweight_base_offset_);
        uint2 const qwords1 =
            *reinterpret_cast<uint2 const*>(
                qweight_ + qweight_base_offset_ + qweight_word_stride(0));
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
          constexpr int kAccess = ThreadMap::kElementsPerAccess;
          int const frag_base = c * kAccess;
          uint32_t const qword0 = qword_from_vector(qwords0, c);
          uint32_t const qword1 = qword_from_vector(qwords1, c);
          half2 const* scale_vec = cached_scales_ + c * 4;
          half2 const* bias_vec = cached_bias_ + c * 4;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU8.id(), false>(
              static_cast<int>(qword0), deq);
          frag_vec[0] = __hfma2(deq[0], scale_vec[0], bias_vec[0]);
          frag_vec[1] = __hfma2(deq[1], scale_vec[1], bias_vec[1]);
          marlin::dequant<half2, vllm::kU8.id(), false>(
              static_cast<int>(qword1), deq);
          frag_vec[2] = __hfma2(deq[0], scale_vec[2], bias_vec[2]);
          frag_vec[3] = __hfma2(deq[1], scale_vec[3], bias_vec[3]);
        }
      } else {
        static_assert(ThreadMap::Iterations::kContiguous == 1,
                      "Unsupported SM70 kU8 contiguous iteration count.");
        uint32_t const qword0 = qweight_[qweight_base_offset_];
        uint32_t const qword1 =
            qweight_[qweight_base_offset_ + qweight_word_stride(0)];
        half2 const* scale_vec = cached_scales_;
        half2 const* bias_vec = cached_bias_;

        half2 deq[2];
        half2* frag_vec = reinterpret_cast<half2*>(frag.data());
        marlin::dequant<half2, vllm::kU8.id(), false>(
            static_cast<int>(qword0), deq);
        frag_vec[0] = __hfma2(deq[0], scale_vec[0], bias_vec[0]);
        frag_vec[1] = __hfma2(deq[1], scale_vec[1], bias_vec[1]);
        marlin::dequant<half2, vllm::kU8.id(), false>(
            static_cast<int>(qword1), deq);
        frag_vec[2] = __hfma2(deq[0], scale_vec[2], bias_vec[2]);
        frag_vec[3] = __hfma2(deq[1], scale_vec[3], bias_vec[3]);
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
        if constexpr (ThreadMap::Iterations::kContiguous == 4) {
          uint4 const qwords0 =
              *reinterpret_cast<uint4 const*>(
                  qweight_ + qweight_offset(s, 0, 0));
          uint4 const qwords1 =
              *reinterpret_cast<uint4 const*>(
                  qweight_ + qweight_offset(s, 0, 1));
          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
            constexpr int kAccess = ThreadMap::kElementsPerAccess;
            int const frag_base =
                (c + s * ThreadMap::Iterations::kContiguous) * kAccess;
            uint32_t const qword0 = qword_from_vector(qwords0, c);
            uint32_t const qword1 = qword_from_vector(qwords1, c);
            half2 const* scale_vec = cached_scales_ + c * 4;
            half2 const* bias_vec = cached_bias_ + c * 4;

            half2 deq[2];
            half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
            marlin::dequant<half2, vllm::kU8.id(), false>(
                static_cast<int>(qword0), deq);
            frag_vec[0] = __hfma2(deq[0], scale_vec[0], bias_vec[0]);
            frag_vec[1] = __hfma2(deq[1], scale_vec[1], bias_vec[1]);
            marlin::dequant<half2, vllm::kU8.id(), false>(
                static_cast<int>(qword1), deq);
            frag_vec[2] = __hfma2(deq[0], scale_vec[2], bias_vec[2]);
            frag_vec[3] = __hfma2(deq[1], scale_vec[3], bias_vec[3]);
          }
        } else if constexpr (ThreadMap::Iterations::kContiguous == 2) {
          uint2 const qwords0 =
              *reinterpret_cast<uint2 const*>(
                  qweight_ + qweight_offset(s, 0, 0));
          uint2 const qwords1 =
              *reinterpret_cast<uint2 const*>(
                  qweight_ + qweight_offset(s, 0, 1));
          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
            constexpr int kAccess = ThreadMap::kElementsPerAccess;
            int const frag_base =
                (c + s * ThreadMap::Iterations::kContiguous) * kAccess;
            uint32_t const qword0 = qword_from_vector(qwords0, c);
            uint32_t const qword1 = qword_from_vector(qwords1, c);
            half2 const* scale_vec = cached_scales_ + c * 4;
            half2 const* bias_vec = cached_bias_ + c * 4;

            half2 deq[2];
            half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
            marlin::dequant<half2, vllm::kU8.id(), false>(
                static_cast<int>(qword0), deq);
            frag_vec[0] = __hfma2(deq[0], scale_vec[0], bias_vec[0]);
            frag_vec[1] = __hfma2(deq[1], scale_vec[1], bias_vec[1]);
            marlin::dequant<half2, vllm::kU8.id(), false>(
                static_cast<int>(qword1), deq);
            frag_vec[2] = __hfma2(deq[0], scale_vec[2], bias_vec[2]);
            frag_vec[3] = __hfma2(deq[1], scale_vec[3], bias_vec[3]);
          }
        } else {
          static_assert(ThreadMap::Iterations::kContiguous == 1,
                        "Unsupported SM70 kU8 contiguous iteration count.");
          uint32_t const qword0 = qweight_[qweight_offset(s, 0, 0)];
          uint32_t const qword1 = qweight_[qweight_offset(s, 0, 1)];
          constexpr int kAccess = ThreadMap::kElementsPerAccess;
          int const frag_base = s * kAccess;
          half2 const* scale_vec = cached_scales_;
          half2 const* bias_vec = cached_bias_;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU8.id(), false>(
              static_cast<int>(qword0), deq);
          frag_vec[0] = __hfma2(deq[0], scale_vec[0], bias_vec[0]);
          frag_vec[1] = __hfma2(deq[1], scale_vec[1], bias_vec[1]);
          marlin::dequant<half2, vllm::kU8.id(), false>(
              static_cast<int>(qword1), deq);
          frag_vec[2] = __hfma2(deq[0], scale_vec[2], bias_vec[2]);
          frag_vec[3] = __hfma2(deq[1], scale_vec[3], bias_vec[3]);
        }
      }
    }
  }

  CUTLASS_DEVICE
  void load_residue_tile(Fragment& frag) const {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      bool k_valid = true;
      if constexpr (kResidueK) {
        int const logical_k =
            k_offset_ + thread_offset_.strided() +
            s * ThreadMap::Delta::kStrided;
        k_valid = logical_k < tile_k_end_;
      }
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        constexpr int kAccess = ThreadMap::kElementsPerAccess;
        int const frag_base =
            (c + s * ThreadMap::Iterations::kContiguous) * kAccess;

        bool n_valid = true;
        if constexpr (kResidueN) {
          int const logical_n =
              n_offset_ + thread_offset_.contiguous() +
              c * ThreadMap::Delta::kContiguous;
          n_valid = logical_n + kAccess <= params_.size_n;
        }

        bool const valid = k_valid && n_valid;
        if (!valid) {
          half2 const zero = __float2half2_rn(0.0f);
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          frag_vec[0] = zero;
          frag_vec[1] = zero;
          frag_vec[2] = zero;
          frag_vec[3] = zero;
          continue;
        }

        uint32_t const qword0 = qweight_[qweight_offset(s, c, 0)];
        uint32_t const qword1 = qweight_[qweight_offset(s, c, 1)];
        half2 const* scale_vec = cached_scales_ + c * 4;
        half2 const* bias_vec = cached_bias_ + c * 4;

        half2 deq[2];
        half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
        marlin::dequant<half2, vllm::kU8.id(), false>(
            static_cast<int>(qword0), deq);
        frag_vec[0] = __hfma2(deq[0], scale_vec[0], bias_vec[0]);
        frag_vec[1] = __hfma2(deq[1], scale_vec[1], bias_vec[1]);
        marlin::dequant<half2, vllm::kU8.id(), false>(
            static_cast<int>(qword1), deq);
        frag_vec[2] = __hfma2(deq[0], scale_vec[2], bias_vec[2]);
        frag_vec[3] = __hfma2(deq[1], scale_vec[3], bias_vec[3]);
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

    if constexpr (kFullTile) {
      load_full_tile(frag);
    } else {
      load_residue_tile(frag);
    }
  }
};

struct Sm70U8ZpBiasGemmSpec {
  template <typename Shape, typename ThreadMap, int GroupSize,
            Sm70TileMode TileMode>
  using IteratorB =
      Sm70U8ZpBiasIteratorB<Shape, ThreadMap, GroupSize, TileMode>;
};

template <int CtaM, int CtaN, int Warps, int GroupSize,
          Sm70TileMode TileMode>
using Sm70U8ZpBiasGemmTraits =
    Sm70DenseGemmTraits<Sm70U8ZpBiasGemmSpec, CtaM, CtaN, Warps, GroupSize,
                        TileMode>;

template <int CtaM, int CtaN, int Warps, int GroupSize,
          Sm70TileMode TileMode>
__global__ __launch_bounds__(Warps * 32, 1)
void sm70_marlin_u8_gemm_kernel(
    cutlass::half_t const* __restrict__ a,
    uint32_t const* __restrict__ b_q_weight,
    cutlass::half_t const* __restrict__ b_scales,
    cutlass::half_t const* __restrict__ b_zp_bias,
    cutlass::half_t* __restrict__ c, int m, int n, int k, int lda) {
  using Traits =
      Sm70U8ZpBiasGemmTraits<CtaM, CtaN, Warps, GroupSize, TileMode>;
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
      reinterpret_cast<half const*>(b_zp_bias), thread_idx, tb_offset_B);

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

template <int CtaM, int CtaN, int Warps, int GroupSize,
          Sm70TileMode TileMode>
torch::Tensor launch_sm70_marlin_u8_gemm(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, torch::Tensor& b_zp_bias, int64_t size_m,
    int64_t size_n, int64_t size_k) {
  auto kernel =
      sm70_marlin_u8_gemm_kernel<CtaM, CtaN, Warps, GroupSize, TileMode>;
  using SharedStorage = typename Sm70U8ZpBiasGemmTraits<
      CtaM, CtaN, Warps, GroupSize, TileMode>::SharedStorage;
  size_t smem_bytes = configure_dynamic_smem<SharedStorage>(kernel);

  dim3 grid = cta_grid(size_m, size_n, CtaM, CtaN);
  dim3 block(Warps * 32);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

  kernel<<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<cutlass::half_t const*>(a.data_ptr<at::Half>()),
      reinterpret_cast<uint32_t const*>(b_q_weight.data_ptr<int32_t>()),
      reinterpret_cast<cutlass::half_t const*>(b_scales.data_ptr<at::Half>()),
      reinterpret_cast<cutlass::half_t const*>(b_zp_bias.data_ptr<at::Half>()),
      reinterpret_cast<cutlass::half_t*>(c.data_ptr<at::Half>()),
      static_cast<int>(size_m), static_cast<int>(size_n),
      static_cast<int>(size_k), static_cast<int>(a.stride(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

struct Sm70U8Launcher {
  torch::Tensor& a;
  torch::Tensor& c;
  torch::Tensor& b_q_weight;
  torch::Tensor& b_scales;
  torch::Tensor& b_zp_bias;
  int64_t size_m;
  int64_t size_n;
  int64_t size_k;

  template <int CtaM, int CtaN, int Warps, int GroupSize,
            Sm70TileMode TileMode>
  torch::Tensor operator()() const {
    return launch_sm70_marlin_u8_gemm<CtaM, CtaN, Warps, GroupSize, TileMode>(
        a, c, b_q_weight, b_scales, b_zp_bias, size_m, size_n, size_k);
  }
};

torch::Tensor sm70_marlin_u8_gemm(torch::Tensor& a, torch::Tensor& c,
                                  torch::Tensor& b_q_weight,
                                  torch::Tensor& b_scales,
                                  torch::Tensor& b_zp_bias, int64_t size_m,
                                  int64_t size_n, int64_t size_k,
                                  int64_t group_size) {
  c10::cuda::CUDAGuard device_guard(a.device());

  char const* env_name = "SM70_MARLIN_U8_CTA";
  Sm70DenseCtaGeometry const geometry =
      parse_sm70_dense_cta_geometry(env_name);
  check_sm70_dense_cta_geometry(env_name, geometry);
  Sm70U8Launcher const launcher{
      a, c, b_q_weight, b_scales, b_zp_bias, size_m, size_n, size_k};
  return dispatch_geometry(launcher, geometry, size_n, size_k, group_size,
                           "uint8");
}

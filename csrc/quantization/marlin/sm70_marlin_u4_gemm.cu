#include "core/registration.h"
#include "quantization/marlin/dequant.h"

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

namespace {

constexpr int kCtaK = 32;
constexpr int kDefaultCtaM = 128;
constexpr int kDefaultCtaN = 256;
constexpr int kDefaultWarps = 8;
constexpr int kQuantTileK = 16;
constexpr int kQuantTileN = 64;
constexpr int kMacroNTiles = 4;
constexpr int kMacroN = kQuantTileN * kMacroNTiles;
constexpr int kU4ValuesPerWord = 8;
constexpr int kU4WordsPerTile = kQuantTileK * kQuantTileN / kU4ValuesPerWord;

enum class Sm70TileMode {
  FullTile,
  ResidueNOnly,
  ResidueKOnly,
  ResidueKAndN,
};

template <int CtaM, int CtaN, int Warps>
struct Sm70U4WarpShape;

template <>
struct Sm70U4WarpShape<32, 128, 4> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

template <>
struct Sm70U4WarpShape<32, 256, 4> {
  using Type = cutlass::gemm::GemmShape<32, 64, 32>;
};

template <>
struct Sm70U4WarpShape<64, 64, 4> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

template <>
struct Sm70U4WarpShape<64, 128, 4> {
  using Type = cutlass::gemm::GemmShape<32, 64, 32>;
};

template <>
struct Sm70U4WarpShape<64, 128, 8> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

template <>
struct Sm70U4WarpShape<64, 256, 4> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70U4WarpShape<64, 256, 8> {
  using Type = cutlass::gemm::GemmShape<32, 64, 32>;
};

template <>
struct Sm70U4WarpShape<128, 64, 4> {
  using Type = cutlass::gemm::GemmShape<64, 32, 32>;
};

template <>
struct Sm70U4WarpShape<128, 64, 8> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

template <>
struct Sm70U4WarpShape<128, 128, 4> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70U4WarpShape<128, 128, 8> {
  using Type = cutlass::gemm::GemmShape<64, 32, 32>;
};

template <>
struct Sm70U4WarpShape<128, 256, 8> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70U4WarpShape<256, 64, 4> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70U4WarpShape<256, 64, 8> {
  using Type = cutlass::gemm::GemmShape<64, 32, 32>;
};

template <>
struct Sm70U4WarpShape<256, 128, 8> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <typename Shape_, typename ThreadMap_, int GroupSize_,
          Sm70TileMode TileMode_>
class Sm70U4ZpBiasIteratorB {
 public:
  using Shape = Shape_;
  using ThreadMap = ThreadMap_;
  static int const kGroupSize = GroupSize_;
  static constexpr Sm70TileMode kTileMode = TileMode_;
  static constexpr bool kFullTile = kTileMode == Sm70TileMode::FullTile;
  static constexpr bool kResidueN =
      kTileMode == Sm70TileMode::ResidueNOnly ||
      kTileMode == Sm70TileMode::ResidueKAndN;
  static constexpr bool kResidueK =
      kTileMode == Sm70TileMode::ResidueKOnly ||
      kTileMode == Sm70TileMode::ResidueKAndN;
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
  Sm70U4ZpBiasIteratorB(Params const& params, uint32_t const* qweight,
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
  Sm70U4ZpBiasIteratorB& operator++() {
    int const k_advance = next_k_advance_;
    int const k_advance_qwords =
        (k_advance / kQuantTileK) * (params_.size_n * 2);
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
                    "SM70 kU4 prototype only specializes group sizes "
                    "-1, 32, 64, and 128.");
      return logical_k / kGroupSize;
    }
  }

  CUTLASS_DEVICE
  static int qweight_offset_from_logical(Params const& params, int logical_k,
                                         int logical_n) {
    int const k_tile = logical_k / kQuantTileK;
    int const local_k = logical_k - k_tile * kQuantTileK;
    int const n_tile = logical_n / kQuantTileN;
    int const macro_n_tile = n_tile / kMacroNTiles;
    int const macro_first_n_tile = macro_n_tile * kMacroNTiles;
    int const subtile = n_tile - macro_first_n_tile;
    int subtile_count = kMacroNTiles;
    if constexpr (kResidueN) {
      subtile_count = params.size_n / kQuantTileN - macro_first_n_tile;
      subtile_count =
          subtile_count < kMacroNTiles ? subtile_count : kMacroNTiles;
    }
    int const local_n_vec =
        (logical_n - n_tile * kQuantTileN) / ThreadMap::kElementsPerAccess;
    int const local_word = local_k * (kQuantTileN / kU4ValuesPerWord) +
                           local_n_vec;

    return k_tile * (params.size_n * 2) +
           macro_n_tile * kMacroNTiles * kU4WordsPerTile +
           local_word * subtile_count + subtile;
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
  static uint32_t qword_from_vector(uint4 const& words, int c) {
    uint32_t const* words_ptr = reinterpret_cast<uint32_t const*>(&words);
    return words_ptr[c];
  }

  CUTLASS_DEVICE
  static uint32_t qword_from_vector(uint2 const& words, int c) {
    uint32_t const* words_ptr = reinterpret_cast<uint32_t const*>(&words);
    return words_ptr[c];
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
    int const metadata_offset = group * params_.size_n + cache_n;
    uint4 const scale_words =
        *reinterpret_cast<uint4 const*>(scales_ + metadata_offset);
    half2 const* scale_vec = reinterpret_cast<half2 const*>(&scale_words);
    half2* scale_cache = cached_scales_ + c * 4;
    scale_cache[0] = scale_vec[0];
    scale_cache[1] = scale_vec[1];
    scale_cache[2] = scale_vec[2];
    scale_cache[3] = scale_vec[3];

    uint4 const bias_words =
        *reinterpret_cast<uint4 const*>(zp_bias_ + metadata_offset);
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

      if constexpr (kFullTile) {
      cache_metadata_lane_vectors(c, group, cache_n);
      } else {
        cache_metadata_vector_words(c, group, cache_n);
      }
    }
  }

  CUTLASS_DEVICE
  void load_full_tile(Fragment& frag) const {
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
          half2 const* bias_vec = cached_bias_ + c * 4;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU4.id(), false>(
              static_cast<int>(qword), deq);
          frag_vec[0] = __hfma2(deq[0], scale_vec[0], bias_vec[0]);
          frag_vec[1] = __hfma2(deq[1], scale_vec[1], bias_vec[1]);
          marlin::dequant<half2, vllm::kU4.id(), false>(
              static_cast<int>(qword >> 8), deq);
          frag_vec[2] = __hfma2(deq[0], scale_vec[2], bias_vec[2]);
          frag_vec[3] = __hfma2(deq[1], scale_vec[3], bias_vec[3]);
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
          half2 const* bias_vec = cached_bias_ + c * 4;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU4.id(), false>(
              static_cast<int>(qword), deq);
          frag_vec[0] = __hfma2(deq[0], scale_vec[0], bias_vec[0]);
          frag_vec[1] = __hfma2(deq[1], scale_vec[1], bias_vec[1]);
          marlin::dequant<half2, vllm::kU4.id(), false>(
              static_cast<int>(qword >> 8), deq);
          frag_vec[2] = __hfma2(deq[0], scale_vec[2], bias_vec[2]);
          frag_vec[3] = __hfma2(deq[1], scale_vec[3], bias_vec[3]);
        }
      } else {
        static_assert(ThreadMap::Iterations::kContiguous == 1,
                      "Unsupported SM70 kU4 contiguous iteration count.");
        uint32_t const qword = qweight_[qweight_base_offset_];
        half2 const* scale_vec = cached_scales_;
        half2 const* bias_vec = cached_bias_;

        half2 deq[2];
        half2* frag_vec = reinterpret_cast<half2*>(frag.data());
        marlin::dequant<half2, vllm::kU4.id(), false>(
            static_cast<int>(qword), deq);
        frag_vec[0] = __hfma2(deq[0], scale_vec[0], bias_vec[0]);
        frag_vec[1] = __hfma2(deq[1], scale_vec[1], bias_vec[1]);
        marlin::dequant<half2, vllm::kU4.id(), false>(
            static_cast<int>(qword >> 8), deq);
        frag_vec[2] = __hfma2(deq[0], scale_vec[2], bias_vec[2]);
        frag_vec[3] = __hfma2(deq[1], scale_vec[3], bias_vec[3]);
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
            half2 const* bias_vec = cached_bias_ + c * 4;

            half2 deq[2];
            half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
            marlin::dequant<half2, vllm::kU4.id(), false>(
                static_cast<int>(qword), deq);
            frag_vec[0] = __hfma2(deq[0], scale_vec[0], bias_vec[0]);
            frag_vec[1] = __hfma2(deq[1], scale_vec[1], bias_vec[1]);
            marlin::dequant<half2, vllm::kU4.id(), false>(
                static_cast<int>(qword >> 8), deq);
            frag_vec[2] = __hfma2(deq[0], scale_vec[2], bias_vec[2]);
            frag_vec[3] = __hfma2(deq[1], scale_vec[3], bias_vec[3]);
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
            half2 const* bias_vec = cached_bias_ + c * 4;

            half2 deq[2];
            half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
            marlin::dequant<half2, vllm::kU4.id(), false>(
                static_cast<int>(qword), deq);
            frag_vec[0] = __hfma2(deq[0], scale_vec[0], bias_vec[0]);
            frag_vec[1] = __hfma2(deq[1], scale_vec[1], bias_vec[1]);
            marlin::dequant<half2, vllm::kU4.id(), false>(
                static_cast<int>(qword >> 8), deq);
            frag_vec[2] = __hfma2(deq[0], scale_vec[2], bias_vec[2]);
            frag_vec[3] = __hfma2(deq[1], scale_vec[3], bias_vec[3]);
          }
        } else {
          static_assert(ThreadMap::Iterations::kContiguous == 1,
                        "Unsupported SM70 kU4 contiguous iteration count.");
          uint32_t const qword = qweight_[qweight_offset(s, 0)];
          constexpr int kAccess = ThreadMap::kElementsPerAccess;
          int const frag_base = s * kAccess;
          half2 const* scale_vec = cached_scales_;
          half2 const* bias_vec = cached_bias_;

          half2 deq[2];
          half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
          marlin::dequant<half2, vllm::kU4.id(), false>(
              static_cast<int>(qword), deq);
          frag_vec[0] = __hfma2(deq[0], scale_vec[0], bias_vec[0]);
          frag_vec[1] = __hfma2(deq[1], scale_vec[1], bias_vec[1]);
          marlin::dequant<half2, vllm::kU4.id(), false>(
              static_cast<int>(qword >> 8), deq);
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

        int const qword_offset = qweight_offset(s, c);
        uint32_t const qword = qweight_[qword_offset];
        half2 const* scale_vec = cached_scales_ + c * 4;
        half2 const* bias_vec = cached_bias_ + c * 4;

        half2 deq[2];
        half2* frag_vec = reinterpret_cast<half2*>(frag.data() + frag_base);
        marlin::dequant<half2, vllm::kU4.id(), false>(
            static_cast<int>(qword), deq);
        frag_vec[0] = __hfma2(deq[0], scale_vec[0], bias_vec[0]);
        frag_vec[1] = __hfma2(deq[1], scale_vec[1], bias_vec[1]);
        marlin::dequant<half2, vllm::kU4.id(), false>(
            static_cast<int>(qword >> 8), deq);
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

template <int CtaM, int CtaN, int Warps, int GroupSize,
          Sm70TileMode TileMode>
struct Sm70U4ZpBiasGemmTraits {
  static_assert(CtaM == 32 || CtaM == 64 || CtaM == 128 || CtaM == 256,
                "SM70 kU4 supports CTA_M in {32, 64, 128, 256}.");
  static_assert(CtaN == 64 || CtaN == 128 || CtaN == 256,
                "SM70 kU4 supports CTA_N in {64, 128, 256}.");
  static_assert(Warps == 4 || Warps == 8,
                "SM70 kU4 supports 4 or 8 warps.");
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using ThreadblockShape = cutlass::gemm::GemmShape<CtaM, CtaN, kCtaK>;
  using WarpShape = typename Sm70U4WarpShape<CtaM, CtaN, Warps>::Type;
  static_assert(WarpShape::kM <= 64 && WarpShape::kN <= 64,
                "SM70 kU4 keeps per-warp M/N no larger than 64.");
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using MmaCore = cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      cutlass::arch::OpClassTensorOp, 2, cutlass::arch::OpMultiplyAdd>;
  static_assert(MmaCore::kThreads == Warps * 32,
                "SM70 kU4 launch threads must match CUTLASS warp count.");
  using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA,
      128 / cutlass::sizeof_bits<ElementA>::value>;
  using IteratorB = Sm70U4ZpBiasIteratorB<
      ThreadblockShape, typename MmaCore::IteratorThreadMapB, GroupSize,
      TileMode>;
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
                "kU4 B operand must be stored through CUTLASS' predefined "
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

template <int CtaM, int CtaN, int Warps, int GroupSize,
          Sm70TileMode TileMode>
__global__ __launch_bounds__(Warps * 32, 1)
void sm70_marlin_u4_gemm_kernel(
    cutlass::half_t const* __restrict__ a,
    uint32_t const* __restrict__ b_q_weight,
    cutlass::half_t const* __restrict__ b_scales,
    cutlass::half_t const* __restrict__ b_zp_bias,
    cutlass::half_t* __restrict__ c, int m, int n, int k, int lda) {
  using Traits =
      Sm70U4ZpBiasGemmTraits<CtaM, CtaN, Warps, GroupSize, TileMode>;
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

struct Sm70U4CtaGeometry {
  int cta_m;
  int cta_n;
  int warps;
};

Sm70U4CtaGeometry parse_sm70_marlin_u4_cta_geometry() {
  char const* env = std::getenv("SM70_MARLIN_U4_CTA");
  if (env == nullptr || env[0] == '\0') {
    return {kDefaultCtaM, kDefaultCtaN, kDefaultWarps};
  }

  std::string spec(env);
  for (char& ch : spec) {
    if (ch == 'x' || ch == 'X' || ch == '*' || ch == ',') {
      ch = ' ';
    }
  }

  int cta_m = 0;
  int cta_n = 0;
  int warps = 0;
  std::string extra;
  std::istringstream stream(spec);
  TORCH_CHECK(
      (stream >> cta_m >> cta_n >> warps) && !(stream >> extra),
      "SM70_MARLIN_U4_CTA must use format CTA_MxCTA_NxWarps, for example "
      "128x256x8. Got: ",
      env);
  return {cta_m, cta_n, warps};
}

bool sm70_marlin_u4_cta_geometry_supported(Sm70U4CtaGeometry geometry) {
  int const cta_m = geometry.cta_m;
  int const cta_n = geometry.cta_n;
  int const warps = geometry.warps;
  return (cta_m == 32 && cta_n == 128 && warps == 4) ||
         (cta_m == 32 && cta_n == 256 && warps == 4) ||
         (cta_m == 64 && cta_n == 64 && warps == 4) ||
         (cta_m == 64 && cta_n == 128 && warps == 4) ||
         (cta_m == 64 && cta_n == 128 && warps == 8) ||
         (cta_m == 64 && cta_n == 256 && warps == 4) ||
         (cta_m == 64 && cta_n == 256 && warps == 8) ||
         (cta_m == 128 && cta_n == 64 && warps == 4) ||
         (cta_m == 128 && cta_n == 64 && warps == 8) ||
         (cta_m == 128 && cta_n == 128 && warps == 4) ||
         (cta_m == 128 && cta_n == 128 && warps == 8) ||
         (cta_m == 128 && cta_n == 256 && warps == 8) ||
         (cta_m == 256 && cta_n == 64 && warps == 4) ||
         (cta_m == 256 && cta_n == 64 && warps == 8) ||
         (cta_m == 256 && cta_n == 128 && warps == 8);
}

void check_sm70_marlin_u4_cta_geometry(Sm70U4CtaGeometry geometry) {
  TORCH_CHECK(
      sm70_marlin_u4_cta_geometry_supported(geometry),
      "Unsupported SM70_MARLIN_U4_CTA=", geometry.cta_m, "x",
      geometry.cta_n, "x", geometry.warps,
      ". Supported geometries are 32x128x4, 32x256x4, 64x64x4, "
      "64x128x4, 64x128x8, 64x256x4, 64x256x8, 128x64x4, "
      "128x64x8, 128x128x4, 128x128x8, 128x256x8, 256x64x4, "
      "256x64x8, and 256x128x8.");
}

}  // namespace

template <int CtaM, int CtaN, int Warps, int GroupSize,
          Sm70TileMode TileMode>
torch::Tensor launch_sm70_marlin_u4_gemm(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, torch::Tensor& b_zp_bias, int64_t size_m,
    int64_t size_n, int64_t size_k) {
  auto kernel =
      sm70_marlin_u4_gemm_kernel<CtaM, CtaN, Warps, GroupSize, TileMode>;
  size_t smem_bytes =
      sizeof(typename Sm70U4ZpBiasGemmTraits<
             CtaM, CtaN, Warps, GroupSize, TileMode>::SharedStorage);
  if (smem_bytes >= (48u << 10)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
  }

  dim3 grid(static_cast<unsigned>((size_m + CtaM - 1) / CtaM),
            static_cast<unsigned>((size_n + CtaN - 1) / CtaN));
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

template <int CtaM, int CtaN, int Warps, Sm70TileMode TileMode>
torch::Tensor launch_sm70_marlin_u4_gemm_group_size(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, torch::Tensor& b_zp_bias, int64_t size_m,
    int64_t size_n, int64_t size_k, int64_t group_size) {
  switch (group_size) {
    case -1:
      return launch_sm70_marlin_u4_gemm<CtaM, CtaN, Warps, -1, TileMode>(
          a, c, b_q_weight, b_scales, b_zp_bias, size_m, size_n, size_k);
    case 32:
      return launch_sm70_marlin_u4_gemm<CtaM, CtaN, Warps, 32, TileMode>(
          a, c, b_q_weight, b_scales, b_zp_bias, size_m, size_n, size_k);
    case 64:
      return launch_sm70_marlin_u4_gemm<CtaM, CtaN, Warps, 64, TileMode>(
          a, c, b_q_weight, b_scales, b_zp_bias, size_m, size_n, size_k);
    case 128:
      return launch_sm70_marlin_u4_gemm<CtaM, CtaN, Warps, 128, TileMode>(
          a, c, b_q_weight, b_scales, b_zp_bias, size_m, size_n, size_k);
    default:
      TORCH_CHECK(false,
                  "SM70 CUTLASS uint4 prototype supports only group_size "
                  "-1, 32, 64, or 128. Got ",
                  group_size);
  }
  return c;
}

template <int CtaM, int CtaN, int Warps>
torch::Tensor launch_sm70_marlin_u4_gemm_geometry(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, torch::Tensor& b_zp_bias, int64_t size_m,
    int64_t size_n, int64_t size_k, int64_t group_size) {
  bool const residue_k = size_k % kCtaK != 0;
  bool const residue_n = size_n % CtaN != 0 || size_n % kMacroN != 0;
  if (!residue_k && !residue_n) {
    return launch_sm70_marlin_u4_gemm_group_size<
        CtaM, CtaN, Warps, Sm70TileMode::FullTile>(
        a, c, b_q_weight, b_scales, b_zp_bias, size_m, size_n, size_k,
        group_size);
  }
  if (!residue_k) {
    return launch_sm70_marlin_u4_gemm_group_size<
        CtaM, CtaN, Warps, Sm70TileMode::ResidueNOnly>(
        a, c, b_q_weight, b_scales, b_zp_bias, size_m, size_n, size_k,
        group_size);
  }
  if (!residue_n) {
    return launch_sm70_marlin_u4_gemm_group_size<
        CtaM, CtaN, Warps, Sm70TileMode::ResidueKOnly>(
        a, c, b_q_weight, b_scales, b_zp_bias, size_m, size_n, size_k,
        group_size);
  }
  return launch_sm70_marlin_u4_gemm_group_size<
      CtaM, CtaN, Warps, Sm70TileMode::ResidueKAndN>(
      a, c, b_q_weight, b_scales, b_zp_bias, size_m, size_n, size_k,
      group_size);
}

#define DISPATCH_SM70_U4_CTA(CM, CN, W)                                  \
  if (geometry.cta_m == CM && geometry.cta_n == CN &&                     \
      geometry.warps == W) {                                              \
    return launch_sm70_marlin_u4_gemm_geometry<CM, CN, W>(                \
        a, c, b_q_weight, b_scales, b_zp_bias, size_m, size_n, size_k,     \
        group_size);                                                      \
  }

torch::Tensor sm70_marlin_u4_gemm(torch::Tensor& a, torch::Tensor& c,
                                  torch::Tensor& b_q_weight,
                                  torch::Tensor& b_scales,
                                  torch::Tensor& b_zp_bias, int64_t size_m,
                                  int64_t size_n, int64_t size_k,
                                  int64_t group_size) {
  c10::cuda::CUDAGuard device_guard(a.device());

  Sm70U4CtaGeometry const geometry = parse_sm70_marlin_u4_cta_geometry();
  check_sm70_marlin_u4_cta_geometry(geometry);

  DISPATCH_SM70_U4_CTA(32, 128, 4)
  DISPATCH_SM70_U4_CTA(32, 256, 4)
  DISPATCH_SM70_U4_CTA(64, 64, 4)
  DISPATCH_SM70_U4_CTA(64, 128, 4)
  DISPATCH_SM70_U4_CTA(64, 128, 8)
  DISPATCH_SM70_U4_CTA(64, 256, 4)
  DISPATCH_SM70_U4_CTA(64, 256, 8)
  DISPATCH_SM70_U4_CTA(128, 64, 4)
  DISPATCH_SM70_U4_CTA(128, 64, 8)
  DISPATCH_SM70_U4_CTA(128, 128, 4)
  DISPATCH_SM70_U4_CTA(128, 128, 8)
  DISPATCH_SM70_U4_CTA(128, 256, 8)
  DISPATCH_SM70_U4_CTA(256, 64, 4)
  DISPATCH_SM70_U4_CTA(256, 64, 8)
  DISPATCH_SM70_U4_CTA(256, 128, 8)

  TORCH_CHECK(false, "Unreachable SM70 uint4 CTA geometry dispatch.");
}

#undef DISPATCH_SM70_U4_CTA

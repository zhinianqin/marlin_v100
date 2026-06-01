#include "core/registration.h"
#include "quantization/marlin/dequant.h"
#include "quantization/marlin/sm70_marlin_common.cuh"
#include "quantization/marlin/sm70_marlin_gemm.cuh"
#include "quantization/marlin/sm70_marlin_iterator_utils.cuh"
#include "quantization/marlin/sm70_marlin_splitk.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

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

using marlin::sm70::Sm70CtaGeometry;
using marlin::sm70::Sm70SplitKPartition;
using marlin::sm70::Sm70WarpShape;
using marlin::sm70::configure_sm70_dynamic_smem;
using marlin::sm70::kCtaK;
using marlin::sm70::kQuantTileK;
using marlin::sm70::kQuantTileN;
using marlin::sm70::launch_sm70_fp32_to_fp16;
using marlin::sm70::parse_sm70_split_k;
using marlin::sm70::load_qword_vector;
using marlin::sm70::qword_from_vector;
using marlin::sm70::sm70_active_split_k;
using marlin::sm70::sm70_marlin_auto_cta_n;
using marlin::sm70::sm70_get_splitk_ctmp;
using marlin::sm70::sm70_splitk_partition;
using marlin::sm70::u4_cta_n_qweight_offset_from_logical;

namespace marlin_moe_wna16 {
namespace {

constexpr int kU4ValuesPerWord = 8;
constexpr char const* kSm70MarlinMoeU4CtaEnv = "SM70_MARLIN_MOE_U4_CTA";
constexpr char const* kSm70MarlinMoeU4SplitKEnv =
    "SM70_MARLIN_MOE_U4_SPLIT_K";
constexpr char const* kSupportedSm70MoeU4CtaGeometries =
    "32x128x4, 32x256x4, 64x64x4, 64x128x4, 64x128x8, "
    "64x256x4, and 64x256x8";

Sm70CtaGeometry parse_sm70_moe_cta_geometry(char const* env_name) {
  char const* env = std::getenv(env_name);
  TORCH_CHECK(env != nullptr && env[0] != '\0', env_name,
              " must use format CTA_MxCTA_NxWarps when explicitly parsed, "
              "for example 32x128x4.");

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
  TORCH_CHECK((stream >> cta_m >> cta_n >> warps) && !(stream >> extra),
              env_name,
              " must use format CTA_MxCTA_NxWarps, for example 32x128x4. "
              "Got: ",
              env);
  return {cta_m, cta_n, warps};
}

Sm70CtaGeometry sm70_moe_default_cta_geometry(int auto_cta_n) {
  switch (auto_cta_n) {
    case 64:
      return {64, 64, 4};
    case 128:
      return {32, 128, 4};
    case 256:
      return {32, 256, 4};
    default:
      TORCH_CHECK(false, "Unsupported SM70 Marlin MoE uint4 auto CTA_N=",
                  auto_cta_n, ".");
  }
  return {0, 0, 0};
}

Sm70CtaGeometry resolve_sm70_moe_cta_geometry(char const* env_name,
                                                   int64_t size_n) {
  int const auto_cta_n = sm70_marlin_auto_cta_n(size_n);
  char const* env = std::getenv(env_name);
  if (env == nullptr || env[0] == '\0') {
    return sm70_moe_default_cta_geometry(auto_cta_n);
  }

  Sm70CtaGeometry geometry = parse_sm70_moe_cta_geometry(env_name);
  TORCH_CHECK(geometry.cta_n == auto_cta_n, env_name,
              " specifies CTA_N=", geometry.cta_n, " but size_n=", size_n,
              " requires auto CTA_N=", auto_cta_n,
              ". CTA_N is selected from 256, 128, and 64 and is not a free "
              "SM70 MoE uint4 tuning parameter.");
  return geometry;
}

bool sm70_moe_cta_geometry_is_supported(Sm70CtaGeometry geometry) {
  int const cta_m = geometry.cta_m;
  int const cta_n = geometry.cta_n;
  int const warps = geometry.warps;
  return (cta_m == 32 && cta_n == 128 && warps == 4) ||
         (cta_m == 32 && cta_n == 256 && warps == 4) ||
         (cta_m == 64 && cta_n == 64 && warps == 4) ||
         (cta_m == 64 && cta_n == 128 && warps == 4) ||
         (cta_m == 64 && cta_n == 128 && warps == 8) ||
         (cta_m == 64 && cta_n == 256 && warps == 4) ||
         (cta_m == 64 && cta_n == 256 && warps == 8);
}

void check_sm70_moe_cta_geometry(char const* env_name,
                                 Sm70CtaGeometry geometry) {
  TORCH_CHECK(sm70_moe_cta_geometry_is_supported(geometry), "Unsupported ",
              env_name, "=", geometry.cta_m, "x", geometry.cta_n, "x",
              geometry.warps, ". Supported geometries are ",
              kSupportedSm70MoeU4CtaGeometries, ".");
}

void check_sm70_moe_cta_n_alignment(char const* env_name,
                                    Sm70CtaGeometry geometry,
                                    int64_t size_n) {
  TORCH_CHECK(
      size_n % geometry.cta_n == 0 && size_n % kQuantTileN == 0,
      "SM70 Marlin MoE uint4 CUTLASS path requires N alignment for ",
      env_name, "=", geometry.cta_m, "x", geometry.cta_n, "x",
      geometry.warps,
      ". size_n must be divisible by both CTA_N and 64. Got size_n = ",
      size_n, ".");
}

int moe_route_tile_count(int64_t padded_tokens, int64_t moe_block_size,
                         int cta_m) {
  int64_t const moe_blocks =
      (padded_tokens + moe_block_size - 1) / moe_block_size;
  int64_t const m_tiles_per_block = (moe_block_size + cta_m - 1) / cta_m;
  return static_cast<int>(moe_blocks * m_tiles_per_block);
}

int moe_n_tile_count(int64_t size_n, int cta_n) {
  return static_cast<int>(size_n / cta_n);
}

template <int CtaM, int CtaN>
CUTLASS_HOST_DEVICE
void decode_moe_route_tile(int route_tile, int moe_block_size, int& moe_block,
                           int& local_m_offset) {
  int const m_tiles_per_block = (moe_block_size + CtaM - 1) / CtaM;
  moe_block = route_tile / m_tiles_per_block;
  int const local_m_tile = route_tile - moe_block * m_tiles_per_block;
  local_m_offset = local_m_tile * CtaM;
}

template <typename Shape_, typename ThreadMap_>
class Sm70MoeGatherIteratorA {
 public:
  using Shape = Shape_;
  using ThreadMap = ThreadMap_;
  using Element = cutlass::half_t;
  using Layout = cutlass::layout::RowMajor;
  using TensorCoord = cutlass::MatrixCoord;
  using Fragment = cutlass::Array<
      Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
  static_assert(Shape::kK == kCtaK,
                "SM70 MoE U4 IteratorA expects CTA_K=32.");

  struct Params {
    int lda;
    int moe_block_size;
    int top_k;
    int size_m;
    int route_count;
    int padded_tokens;

    CUTLASS_HOST_DEVICE
    Params()
        : lda(0),
          moe_block_size(0),
          top_k(0),
          size_m(0),
          route_count(0),
          padded_tokens(0) {}

    CUTLASS_HOST_DEVICE
    Params(int lda_, int moe_block_size_, int top_k_, int size_m_,
           int route_count_, int padded_tokens_)
        : lda(lda_),
          moe_block_size(moe_block_size_),
          top_k(top_k_),
          size_m(size_m_),
          route_count(route_count_),
          padded_tokens(padded_tokens_) {}
  };

 private:
  cutlass::half_t const* a_;
  int32_t const* sorted_token_ids_;
  Params params_;
  cutlass::layout::PitchLinearCoord thread_offset_;
  int moe_block_;
  int local_m_offset_;
  int k_offset_;
  bool mask_enabled_;

 public:
  CUTLASS_DEVICE
  Sm70MoeGatherIteratorA(Params const& params,
                         cutlass::half_t const* __restrict__ a,
                         int32_t const* __restrict__ sorted_token_ids,
                         int thread_id, int moe_block, int local_m_offset,
                         int k_offset)
      : a_(a),
        sorted_token_ids_(sorted_token_ids),
        params_(params),
        thread_offset_(ThreadMap::initial_offset(thread_id)),
        moe_block_(moe_block),
        local_m_offset_(local_m_offset),
        k_offset_(k_offset),
        mask_enabled_(true) {}

  CUTLASS_DEVICE
  Sm70MoeGatherIteratorA& operator++() {
    k_offset_ += Shape::kK;
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
  void load(Fragment& frag) const {
    cutlass::half_t* frag_ptr = frag.data();
    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < Fragment::kElements; ++idx) {
      frag_ptr[idx] = cutlass::half_t(0);
    }

    if (!mask_enabled_) {
      return;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      int const local_row =
          local_m_offset_ + thread_offset_.strided() +
          s * ThreadMap::Delta::kStrided;
      int const route_row = moe_block_ * params_.moe_block_size + local_row;

      int sorted_id = -1;
      bool valid_row = local_row < params_.moe_block_size &&
                       route_row < params_.padded_tokens;
      if (valid_row) {
        sorted_id = sorted_token_ids_[route_row];
        valid_row = sorted_id >= 0 && sorted_id < params_.route_count;
      }

      int const token_row = valid_row ? (sorted_id / params_.top_k) : 0;

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int const logical_k =
            k_offset_ + thread_offset_.contiguous() +
            c * ThreadMap::Delta::kContiguous;
        int const frag_base =
            (c + s * ThreadMap::Iterations::kContiguous) *
            ThreadMap::kElementsPerAccess;
        bool const valid = valid_row && logical_k < params_.lda;

        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < ThreadMap::kElementsPerAccess; ++e) {
          int const k_element = logical_k + e;
          frag_ptr[frag_base + e] =
              (valid && k_element < params_.lda)
                  ? a_[int64_t(token_row) * params_.lda + k_element]
                  : cutlass::half_t(0);
        }
      }
    }
  }
};

template <typename Shape_, typename ThreadMap_, int GroupSize_>
class Sm70MoeU4ZpIteratorB {
 public:
  using Shape = Shape_;
  using ThreadMap = ThreadMap_;
  static int const kGroupSize = GroupSize_;
  using Element = cutlass::half_t;
  using Fragment = cutlass::Array<
      Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
  static_assert(Shape::kK == kCtaK,
                "SM70 MoE U4 IteratorB expects CTA_K=32.");
  static_assert(Shape::kN == 64 || Shape::kN == 128 || Shape::kN == 256,
                "SM70 MoE U4 IteratorB expects CTA_N in {64, 128, 256}.");
  static_assert(ThreadMap::Iterations::kContiguous ==
                    Shape::kN / kQuantTileN,
                "SM70 MoE U4 IteratorB expects one contiguous iteration per "
                "64-column quant tile.");
  static_assert(ThreadMap::Delta::kContiguous == kQuantTileN,
                "SM70 MoE U4 IteratorB expects 64-column deltas.");
  static_assert(ThreadMap::kElementsPerAccess == kU4ValuesPerWord,
                "SM70 MoE U4 IteratorB expects one packed int4 word per "
                "access.");
  static_assert(ThreadMap::Iterations::kStrided == 1 ||
                    ThreadMap::Iterations::kStrided == 2,
                "SM70 U4-family IteratorB expects one or two strided iterations.");
  static constexpr int kStridedQweightDeltaWords =
      32 * (Shape::kN / kQuantTileN);

  struct Params {
    int size_k;
    int size_n;
    int num_groups;

    CUTLASS_HOST_DEVICE
    Params() : size_k(0), size_n(0), num_groups(0) {}

    CUTLASS_HOST_DEVICE
    Params(int size_k_, int size_n_)
        : size_k(size_k_),
          size_n(size_n_),
          num_groups(GroupSize_ == -1 ? 1 : size_k_ / GroupSize_) {}
  };

 private:
  uint32_t const* qweight_;
  half const* scales_expert_base_;
  half const* zp_expert_base_;
  Params params_;
  cutlass::layout::PitchLinearCoord thread_offset_;
  int qweight_base_offset_;
  int expert_;
  int k_offset_;
  int n_offset_;
  bool mask_enabled_;
  mutable half2 cached_scales_[ThreadMap::Iterations::kContiguous * 4];
  mutable half2 cached_zp_[ThreadMap::Iterations::kContiguous * 4];

 public:
  CUTLASS_DEVICE
  Sm70MoeU4ZpIteratorB(Params const& params, uint32_t const* qweight,
                       half const* scales, half const* zp, int thread_id,
                       int expert, int k_offset, int n_offset)
      : qweight_(qweight),
        scales_expert_base_(scales +
                            (expert >= 0 ? expert : 0) *
                                params.num_groups * params.size_n),
        zp_expert_base_(zp + (expert >= 0 ? expert : 0) *
                                 params.num_groups * params.size_n),
        params_(params),
        thread_offset_(ThreadMap::initial_offset(thread_id)),
        expert_(expert),
        k_offset_(k_offset),
        n_offset_(n_offset),
        mask_enabled_(expert >= 0) {
    int const logical_k = k_offset_ + thread_offset_.strided();
    int const logical_n = n_offset_ + thread_offset_.contiguous();
    qweight_base_offset_ =
        expert_qweight_base_offset() +
        qweight_offset_from_logical(params_, logical_k, logical_n);
    if constexpr (kGroupSize == -1) {
      cache_current_group_metadata(0);
    }
  }

  CUTLASS_DEVICE
  Sm70MoeU4ZpIteratorB& operator++() {
    int const k_advance_qwords =
        (Shape::kK / kQuantTileK) * (params_.size_n * 2);
    k_offset_ += Shape::kK;
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
  void enable_mask() { mask_enabled_ = expert_ >= 0; }

  CUTLASS_DEVICE
  int expert_qweight_base_offset() const {
    return expert_ * (params_.size_k / kQuantTileK) * (params_.size_n * 2);
  }

  CUTLASS_DEVICE
  int metadata_group_offset(int group) const {
    return group * params_.size_n;
  }

  CUTLASS_DEVICE
  int scale_group(int logical_k) const {
    if constexpr (kGroupSize == -1) {
      return 0;
    } else {
      static_assert(kGroupSize == 32 || kGroupSize == 64 ||
                        kGroupSize == 128,
                    "SM70 MoE U4 only specializes group sizes -1, 32, 64, "
                    "and 128.");
      return logical_k / kGroupSize;
    }
  }

  CUTLASS_DEVICE
  static int qweight_offset_from_logical(Params const& params, int logical_k,
                                         int logical_n) {
    return u4_cta_n_qweight_offset_from_logical<Shape::kN>(
        params.size_n, logical_k, logical_n);
  }

  CUTLASS_DEVICE
  void cache_metadata_lane_vectors(int c, int group, int cache_n) const {
    int const metadata_offset = metadata_group_offset(group) + cache_n;
    half2 const* scale_vec = reinterpret_cast<half2 const*>(
        scales_expert_base_ + metadata_offset);
    half2* scale_cache = cached_scales_ + c * 4;
    scale_cache[0] = scale_vec[0];
    scale_cache[1] = scale_vec[1];
    scale_cache[2] = scale_vec[2];
    scale_cache[3] = scale_vec[3];

    half2 const* zp_vec = reinterpret_cast<half2 const*>(
        zp_expert_base_ + metadata_offset);
    half2* zp_cache = cached_zp_ + c * 4;
    zp_cache[0] = zp_vec[0];
    zp_cache[1] = zp_vec[1];
    zp_cache[2] = zp_vec[2];
    zp_cache[3] = zp_vec[3];
  }

  CUTLASS_DEVICE
  void cache_metadata_vector_words(int c, int group, int cache_n) const {
    int const metadata_offset = metadata_group_offset(group) + cache_n;
    uint4 const scale_words = *reinterpret_cast<uint4 const*>(
        scales_expert_base_ + metadata_offset);
    half2 const* scale_vec = reinterpret_cast<half2 const*>(&scale_words);
    half2* scale_cache = cached_scales_ + c * 4;
    scale_cache[0] = scale_vec[0];
    scale_cache[1] = scale_vec[1];
    scale_cache[2] = scale_vec[2];
    scale_cache[3] = scale_vec[3];

    uint4 const zp_words = *reinterpret_cast<uint4 const*>(
        zp_expert_base_ + metadata_offset);
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
  void load_cta_n_aligned(Fragment& frag) const {
    if constexpr (ThreadMap::Iterations::kStrided == 1) {
      if constexpr (ThreadMap::Iterations::kContiguous == 4) {
        auto const qwords =
            load_qword_vector<4>(qweight_ + qweight_base_offset_);
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
        auto const qwords =
            load_qword_vector<2>(qweight_ + qweight_base_offset_);
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
                      "Unsupported SM70 MoE U4 contiguous iteration count.");
        uint32_t const qword = load_qword_vector<1>(qweight_ + qweight_base_offset_);
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
      int const qweight_base = qweight_base_offset_;
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
        int const qweight_base_s =
            qweight_base + s * kStridedQweightDeltaWords;
        if constexpr (ThreadMap::Iterations::kContiguous == 4) {
          auto const qwords =
              load_qword_vector<4>(qweight_ + qweight_base_s);
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
            frag_vec[0] = __hfma2(deq[0], scale_vec[0], __hneg2(zp_vec[0]));
            frag_vec[1] = __hfma2(deq[1], scale_vec[1], __hneg2(zp_vec[1]));
            marlin::dequant<half2, vllm::kU4.id(), false>(
                static_cast<int>(qword >> 8), deq);
            frag_vec[2] = __hfma2(deq[0], scale_vec[2], __hneg2(zp_vec[2]));
            frag_vec[3] = __hfma2(deq[1], scale_vec[3], __hneg2(zp_vec[3]));
          }
        } else if constexpr (ThreadMap::Iterations::kContiguous == 2) {
          auto const qwords =
              load_qword_vector<2>(qweight_ + qweight_base_s);
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
            frag_vec[0] = __hfma2(deq[0], scale_vec[0], __hneg2(zp_vec[0]));
            frag_vec[1] = __hfma2(deq[1], scale_vec[1], __hneg2(zp_vec[1]));
            marlin::dequant<half2, vllm::kU4.id(), false>(
                static_cast<int>(qword >> 8), deq);
            frag_vec[2] = __hfma2(deq[0], scale_vec[2], __hneg2(zp_vec[2]));
            frag_vec[3] = __hfma2(deq[1], scale_vec[3], __hneg2(zp_vec[3]));
          }
        } else {
          static_assert(ThreadMap::Iterations::kContiguous == 1,
                        "Unsupported SM70 MoE U4 contiguous iteration count.");
          uint32_t const qword = load_qword_vector<1>(qweight_ + qweight_base_s);
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

    load_cta_n_aligned(frag);
  }
};

struct Sm70MoeU4ZpGemmSpec {
  template <typename Shape, typename ThreadMap>
  using IteratorA = Sm70MoeGatherIteratorA<Shape, ThreadMap>;

  template <typename Shape, typename ThreadMap, int GroupSize>
  using IteratorB = Sm70MoeU4ZpIteratorB<Shape, ThreadMap, GroupSize>;
};

template <typename Spec, int CtaM, int CtaN, int Warps, int GroupSize>
struct Sm70MarlinMoeGemmTraits {
  static_assert(CtaM == 32 || CtaM == 64 || CtaM == 128 || CtaM == 256,
                "SM70 MoE supports CTA_M in {32, 64, 128, 256}.");
  static_assert(CtaN == 64 || CtaN == 128 || CtaN == 256,
                "SM70 MoE supports CTA_N in {64, 128, 256}.");
  static_assert(Warps == 4 || Warps == 8,
                "SM70 MoE supports 4 or 8 warps.");
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using ThreadblockShape = cutlass::gemm::GemmShape<CtaM, CtaN, kCtaK>;
  using WarpShape = typename Sm70WarpShape<CtaM, CtaN, Warps>::Type;
  static_assert(WarpShape::kM <= 64 && WarpShape::kN <= 64,
                "SM70 MoE keeps per-warp M/N no larger than 64.");
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using MmaCore = cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      cutlass::arch::OpClassTensorOp, 2, cutlass::arch::OpMultiplyAdd>;
  static_assert(MmaCore::kThreads == Warps * 32,
                "SM70 MoE launch threads must match CUTLASS warp count.");
  using IteratorA = typename Spec::template IteratorA<
      ThreadblockShape, typename MmaCore::IteratorThreadMapA>;
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
                "SM70 MoE B operand must use CUTLASS' predefined Volta "
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

template <int CtaM, int CtaN, int Warps, int GroupSize>
using Sm70MoeU4ZpGemmTraits =
    Sm70MarlinMoeGemmTraits<Sm70MoeU4ZpGemmSpec, CtaM, CtaN, Warps,
                            GroupSize>;

template <typename Traits>
class Sm70MoeScatterEpilogue {
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
  void store_fragment(OutputTileIterator const& destination_iterator,
                      typename SharedLoadIterator::Fragment const& frag,
                      int32_t const* __restrict__ sorted_token_ids,
                      float const* __restrict__ topk_weights,
                      cutlass::half_t* __restrict__ c,
                      float* __restrict__ c_tmp, int n, int moe_block,
                      int local_m_offset, int moe_block_size, int route_count,
                      int padded_tokens, bool mul_topk_weights,
                      bool atomic_store) const {
    float const* frag_ptr = reinterpret_cast<float const*>(&frag);
    int const thread_start_row = destination_iterator.thread_start_row();
    int const thread_start_column = destination_iterator.thread_start_column();

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
          int const local_row =
              local_m_offset + thread_start_row + row_offset;
          int const route_row = moe_block * moe_block_size + local_row;
          bool valid_row =
              local_row < moe_block_size && route_row < padded_tokens;
          int sorted_id = -1;
          if (valid_row) {
            sorted_id = sorted_token_ids[route_row];
            valid_row = sorted_id >= 0 && sorted_id < route_count;
          }
          float const route_scale =
              (valid_row && mul_topk_weights) ? topk_weights[sorted_id] : 1.0f;

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn;
               ++column) {
            int const logical_column_base =
                thread_start_column + column * ThreadMap::Delta::kColumn;
            int const frag_base =
                (frag_row_idx * ThreadMap::Iterations::kColumn + column) *
                ThreadMap::kElementsPerAccess;

            if (valid_row) {
              CUTLASS_PRAGMA_UNROLL
              for (int e = 0; e < ThreadMap::kElementsPerAccess; ++e) {
                int64_t const offset =
                    int64_t(sorted_id) * n + logical_column_base + e;
                float const value = frag_ptr[frag_base + e] * route_scale;
                if (atomic_store) {
                  atomicAdd(c_tmp + offset, value);
                } else {
                  c[offset] = cutlass::half_t(value);
                }
              }
            }
          }
        }
      }
    }
  }

 public:
  CUTLASS_DEVICE
  Sm70MoeScatterEpilogue(SharedStorage& shared_storage, int thread_idx,
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
                  int32_t const* __restrict__ sorted_token_ids,
                  float const* __restrict__ topk_weights,
                  cutlass::half_t* __restrict__ c,
                  float* __restrict__ c_tmp, int n, int moe_block,
                  int local_m_offset, int moe_block_size, int route_count,
                  int padded_tokens, bool mul_topk_weights,
                  bool atomic_store) {
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

      store_fragment(destination_iterator, aligned_accum_fragment,
                     sorted_token_ids, topk_weights, c, c_tmp, n, moe_block,
                     local_m_offset, moe_block_size, route_count,
                     padded_tokens, mul_topk_weights, atomic_store);
      ++destination_iterator;
    }
  }
};

template <int CtaM, int CtaN, int Warps, int GroupSize, bool SplitK>
__global__ __launch_bounds__(Warps * 32, 1) void sm70_marlin_u4_gemm_kernel(
    cutlass::half_t const* __restrict__ a,
    uint32_t const* __restrict__ b_q_weight,
    cutlass::half_t const* __restrict__ b_scales,
    cutlass::half_t const* __restrict__ b_zeros,
    cutlass::half_t* __restrict__ c, float* __restrict__ c_tmp,
    int32_t const* __restrict__ sorted_token_ids,
    int32_t const* __restrict__ expert_ids,
    int32_t const* __restrict__ num_tokens_past_padded,
    float const* __restrict__ topk_weights, int moe_block_size, int top_k,
    bool mul_topk_weights, int m, int n, int k, int lda, int split_k) {
  using Traits = Sm70MoeU4ZpGemmTraits<CtaM, CtaN, Warps, GroupSize>;
  using Mma = typename Traits::Mma;
  using Epilogue = Sm70MoeScatterEpilogue<Traits>;

  extern __shared__ char smem[];
  auto& shared_storage =
      *reinterpret_cast<typename Traits::SharedStorage*>(smem);

  int const thread_idx = threadIdx.x;
  int const warp_idx = cutlass::canonical_warp_idx_sync();
  int const lane_idx = threadIdx.x % 32;

  int const padded_tokens = num_tokens_past_padded[0];
  int moe_block = 0;
  int local_m_offset = 0;
  decode_moe_route_tile<CtaM, CtaN>(int(blockIdx.x), moe_block_size,
                                    moe_block, local_m_offset);
  if (moe_block * moe_block_size >= padded_tokens) {
    return;
  }

  int const expert = expert_ids[moe_block];
  if (expert < 0) {
    return;
  }

  int k_begin = 0;
  int partition_k = k;
  if constexpr (SplitK) {
    Sm70SplitKPartition const partition =
        sm70_splitk_partition<GroupSize>(k, split_k, int(blockIdx.z));
    if (partition.partition_k == 0) {
      return;
    }
    k_begin = partition.k_begin;
    partition_k = partition.partition_k;
  }

  int const n_offset = int(blockIdx.y) * CtaN;

  typename Mma::IteratorA iterator_A(
      typename Mma::IteratorA::Params(lda, moe_block_size, top_k, m,
                                      m * top_k, padded_tokens),
      a, sorted_token_ids, thread_idx, moe_block, local_m_offset, k_begin);
  typename Mma::IteratorB iterator_B(
      typename Mma::IteratorB::Params(k, n),
      reinterpret_cast<uint32_t const*>(b_q_weight),
      reinterpret_cast<half const*>(b_scales),
      reinterpret_cast<half const*>(b_zeros), thread_idx, expert, k_begin,
      n_offset);

  Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
  typename Mma::FragmentC accumulators;
  accumulators.clear();

  int const gemm_k_iterations =
      SplitK ? (partition_k / kCtaK) : ((k + kCtaK - 1) / kCtaK);
  mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

  typename Epilogue::OutputTileIterator iterator_D(
      typename Epilogue::OutputTileIterator::Params(
          typename Traits::LayoutC(n)),
      c, cutlass::MatrixCoord(CtaM, n), thread_idx,
      cutlass::MatrixCoord(0, n_offset));

  Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
  epilogue(iterator_D, accumulators, sorted_token_ids, topk_weights, c, c_tmp, n,
           moe_block, local_m_offset, moe_block_size, m * top_k,
           padded_tokens, mul_topk_weights, SplitK);
}

template <int CtaM, int CtaN, int Warps, int GroupSize>
torch::Tensor launch_sm70_marlin_u4_gemm(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, torch::Tensor& b_zeros,
    torch::Tensor& sorted_token_ids, torch::Tensor& expert_ids,
    torch::Tensor& num_tokens_past_padded, torch::Tensor& topk_weights,
    int64_t moe_block_size, int64_t top_k, bool mul_topk_weights,
    int64_t size_m, int64_t size_n, int64_t size_k, int split_k,
    std::optional<torch::Tensor> const& c_tmp_or_none) {
  using Traits = Sm70MoeU4ZpGemmTraits<CtaM, CtaN, Warps, GroupSize>;
  using SharedStorage = typename Traits::SharedStorage;

  auto kernel =
      sm70_marlin_u4_gemm_kernel<CtaM, CtaN, Warps, GroupSize, false>;
  size_t smem_bytes = configure_sm70_dynamic_smem<SharedStorage>(kernel);
  dim3 block(Warps * 32);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

  int const route_tiles =
      moe_route_tile_count(sorted_token_ids.numel(), moe_block_size, CtaM);
  dim3 grid(static_cast<unsigned>(route_tiles),
            static_cast<unsigned>(moe_n_tile_count(size_n, CtaN)));

  if (split_k == 1) {
    kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<cutlass::half_t const*>(a.data_ptr<at::Half>()),
        reinterpret_cast<uint32_t const*>(b_q_weight.data_ptr<int32_t>()),
        reinterpret_cast<cutlass::half_t const*>(
            b_scales.data_ptr<at::Half>()),
        reinterpret_cast<cutlass::half_t const*>(b_zeros.data_ptr<at::Half>()),
        reinterpret_cast<cutlass::half_t*>(c.data_ptr<at::Half>()), nullptr,
        sorted_token_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
        num_tokens_past_padded.data_ptr<int32_t>(),
        topk_weights.data_ptr<float>(), int(moe_block_size), int(top_k),
        mul_topk_weights, int(size_m), int(size_n), int(size_k),
        int(a.stride(0)), split_k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
  }

  TORCH_CHECK(size_k % int64_t(kCtaK) == 0, kSm70MarlinMoeU4SplitKEnv,
              " requires K divisible by 32 for split_k > 1. Got K=", size_k,
              ", split_k=", split_k, ".");

  auto split_kernel =
      sm70_marlin_u4_gemm_kernel<CtaM, CtaN, Warps, GroupSize, true>;
  smem_bytes = configure_sm70_dynamic_smem<SharedStorage>(split_kernel);

  int64_t const numel = size_m * top_k * size_n;
  auto c_tmp = sm70_get_splitk_ctmp(c_tmp_or_none, a.device(), numel);
  C10_CUDA_CHECK(cudaMemsetAsync(
      c_tmp.data_ptr<float>(), 0,
      static_cast<size_t>(numel) * sizeof(float), stream));

  int const active_split_k =
      sm70_active_split_k(static_cast<int>(size_k), split_k);
  grid.z = static_cast<unsigned>(active_split_k);
  split_kernel<<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<cutlass::half_t const*>(a.data_ptr<at::Half>()),
      reinterpret_cast<uint32_t const*>(b_q_weight.data_ptr<int32_t>()),
      reinterpret_cast<cutlass::half_t const*>(b_scales.data_ptr<at::Half>()),
      reinterpret_cast<cutlass::half_t const*>(b_zeros.data_ptr<at::Half>()),
      reinterpret_cast<cutlass::half_t*>(c.data_ptr<at::Half>()),
      c_tmp.data_ptr<float>(), sorted_token_ids.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(), num_tokens_past_padded.data_ptr<int32_t>(),
      topk_weights.data_ptr<float>(), int(moe_block_size), int(top_k),
      mul_topk_weights, int(size_m), int(size_n), int(size_k),
      int(a.stride(0)), split_k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  launch_sm70_fp32_to_fp16(
      c_tmp.data_ptr<float>(),
      reinterpret_cast<cutlass::half_t*>(c.data_ptr<at::Half>()), numel,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

struct Sm70MoeU4Launcher {
  torch::Tensor& a;
  torch::Tensor& c;
  torch::Tensor& b_q_weight;
  torch::Tensor& b_scales;
  torch::Tensor& b_zeros;
  torch::Tensor& sorted_token_ids;
  torch::Tensor& expert_ids;
  torch::Tensor& num_tokens_past_padded;
  torch::Tensor& topk_weights;
  int64_t moe_block_size;
  int64_t top_k;
  bool mul_topk_weights;
  int64_t size_m;
  int64_t size_n;
  int64_t size_k;
  int split_k;
  std::optional<torch::Tensor> const& c_tmp_or_none;

  template <int CtaM, int CtaN, int Warps, int GroupSize>
  torch::Tensor operator()() const {
    return launch_sm70_marlin_u4_gemm<CtaM, CtaN, Warps, GroupSize>(
        a, c, b_q_weight, b_scales, b_zeros, sorted_token_ids, expert_ids,
        num_tokens_past_padded, topk_weights, moe_block_size, top_k,
        mul_topk_weights, size_m, size_n, size_k, split_k, c_tmp_or_none);
  }
};

template <int CtaM, int CtaN, int Warps>
torch::Tensor dispatch_sm70_marlin_u4_group_size(
    Sm70MoeU4Launcher const& launcher, int64_t group_size) {
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
      TORCH_CHECK(false,
                  "SM70 Marlin MoE uint4 path supports only group_size -1, "
                  "32, 64, or 128. Got ",
                  group_size, ".");
  }
  return launcher.c;
}

torch::Tensor dispatch_sm70_marlin_u4_gemm(Sm70MoeU4Launcher const& launcher,
                                           Sm70CtaGeometry geometry,
                                           int64_t group_size) {
#define DISPATCH_SM70_MOE_U4_CTA(CM, CN, W)                              \
  if (geometry.cta_m == CM && geometry.cta_n == CN &&                     \
      geometry.warps == W) {                                              \
    return dispatch_sm70_marlin_u4_group_size<CM, CN, W>(launcher,         \
                                                         group_size);      \
  }

  DISPATCH_SM70_MOE_U4_CTA(32, 128, 4)
  DISPATCH_SM70_MOE_U4_CTA(32, 256, 4)
  DISPATCH_SM70_MOE_U4_CTA(64, 64, 4)
  DISPATCH_SM70_MOE_U4_CTA(64, 128, 4)
  DISPATCH_SM70_MOE_U4_CTA(64, 128, 8)
  DISPATCH_SM70_MOE_U4_CTA(64, 256, 4)
  DISPATCH_SM70_MOE_U4_CTA(64, 256, 8)

#undef DISPATCH_SM70_MOE_U4_CTA

  TORCH_CHECK(false, "Unreachable SM70 Marlin MoE uint4 CTA dispatch.");
}

}  // namespace

torch::Tensor sm70_marlin_u4_gemm(
    torch::Tensor& a, torch::Tensor& c, torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, torch::Tensor& b_zeros,
    torch::Tensor& sorted_token_ids, torch::Tensor& expert_ids,
    torch::Tensor& num_tokens_past_padded, torch::Tensor& topk_weights,
    int64_t moe_block_size, int64_t top_k, bool mul_topk_weights,
    int64_t size_m, int64_t size_n, int64_t size_k, int64_t group_size,
    std::optional<torch::Tensor> const& c_tmp_or_none) {
  c10::cuda::CUDAGuard device_guard(a.device());

  Sm70CtaGeometry const geometry =
      resolve_sm70_moe_cta_geometry(kSm70MarlinMoeU4CtaEnv, size_n);
  check_sm70_moe_cta_geometry(kSm70MarlinMoeU4CtaEnv, geometry);
  check_sm70_moe_cta_n_alignment(kSm70MarlinMoeU4CtaEnv, geometry, size_n);
  TORCH_CHECK(size_k % kCtaK == 0,
              "SM70 Marlin MoE uint4 CUTLASS path requires K divisible by 32.");

  int const split_k = parse_sm70_split_k(kSm70MarlinMoeU4SplitKEnv);
  Sm70MoeU4Launcher const launcher{
      a, c, b_q_weight, b_scales, b_zeros, sorted_token_ids, expert_ids,
      num_tokens_past_padded, topk_weights, moe_block_size, top_k,
      mul_topk_weights, size_m, size_n, size_k, split_k, c_tmp_or_none};
  return dispatch_sm70_marlin_u4_gemm(launcher, geometry, group_size);
}

}  // namespace marlin_moe_wna16

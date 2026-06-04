#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <torch/library.h>
#include <torch/types.h>

#include <cstdint>
#include <optional>

#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "quantization/marlin/sm70_marlin_common.cuh"

namespace marlin::sm70 {

struct Sm70SplitKPartition {
  int k_begin;
  int partition_k;
};

inline int sm70_marlin_auto_requested_split_k_from_tiles(int64_t size_k,
                                        int64_t cta_tiles,
                                        bool k_pressure_path) {
  if (size_k < 4096 || size_k % kCtaK != 0) {
    return 1;
  }

  if (k_pressure_path) {
    if (cta_tiles <= 64) {
      return 8;
    }
    if (cta_tiles <= 128) {
      return 4;
    }
    if (cta_tiles <= 256) {
      return 2;
    }
    return 1;
  }

  if (cta_tiles <= 16) {
    return 8;
  }
  if (cta_tiles <= 32) {
    return 4;
  }
  if (cta_tiles <= 64) {
    return 2;
  }
  return 1;
}

inline int sm70_marlin_dense_auto_requested_split_k(
    int64_t size_m, int64_t size_n, int64_t size_k, Sm70CtaGeometry geometry) {
  int64_t const m_tiles = (size_m + geometry.cta_m - 1) / geometry.cta_m;
  int64_t const n_tiles = size_n / geometry.cta_n;
  int64_t const cta_tiles = m_tiles * (n_tiles > 0 ? n_tiles : 1);
  bool const k_pressure_path = size_k >= 8192 && size_n <= 256;
  return sm70_marlin_auto_requested_split_k_from_tiles(size_k, cta_tiles, k_pressure_path);
}

inline int sm70_marlin_moe_auto_stage_requested_split_k(
    int64_t tokens, int64_t size_n, int64_t size_k, int64_t top_k,
    Sm70CtaGeometry geometry) {
  int64_t const effective_m = tokens * top_k;
  int64_t const m_tiles =
      (effective_m + geometry.cta_m - 1) / geometry.cta_m;
  int64_t const n_tiles = size_n / geometry.cta_n;
  int64_t const cta_tiles = m_tiles * (n_tiles > 0 ? n_tiles : 1);
  return sm70_marlin_auto_requested_split_k_from_tiles(size_k, cta_tiles, false);
}

inline torch::Tensor sm70_get_splitk_ctmp(
    std::optional<torch::Tensor> const& c_tmp_or_none, torch::Device device,
    int64_t required_numel) {
  if (!c_tmp_or_none.has_value()) {
    return torch::empty({required_numel},
                        torch::TensorOptions().dtype(at::kFloat).device(device));
  }

  torch::Tensor c_tmp = c_tmp_or_none.value();
  TORCH_CHECK(c_tmp.device().is_cuda(), "c_tmp must be a CUDA tensor.");
  TORCH_CHECK(c_tmp.device() == device,
              "c_tmp device must match the activation device.");
  TORCH_CHECK(c_tmp.scalar_type() == at::ScalarType::Float,
              "c_tmp must have dtype torch.float32.");
  TORCH_CHECK(c_tmp.is_contiguous(), "c_tmp must be contiguous.");
  TORCH_CHECK(c_tmp.numel() >= required_numel, "c_tmp.numel = ", c_tmp.numel(),
              " is smaller than M*N = ", required_numel, ".");
  return c_tmp;
}

CUTLASS_HOST_DEVICE
int sm70_splitk_ceil_div_int(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

CUTLASS_HOST_DEVICE
int sm70_splitk_min_int(int lhs, int rhs) {
  return lhs < rhs ? lhs : rhs;
}

template <int GroupSize>
CUTLASS_HOST_DEVICE int sm70_splitk_group_tiles() {
  if constexpr (GroupSize > 0) {
    if constexpr (GroupSize >= kCtaK) {
      static_assert(GroupSize % kCtaK == 0,
                    "SM70 Marlin split-K group size must be CTA_K aligned.");
      return GroupSize / kCtaK;
    } else {
      return 1;
    }
  } else {
    return 1;
  }
}

CUTLASS_HOST_DEVICE
int sm70_active_split_k(int k, int requested_split_k) {
  int const total_tiles = k / kCtaK;
  if (total_tiles <= 0) {
    return 0;
  }
  return sm70_splitk_min_int(requested_split_k, total_tiles);
}

CUTLASS_HOST_DEVICE
int sm70_splitk_partition_tile_count(int remaining_tiles,
                                     int remaining_partitions,
                                     int group_tiles) {
  if (remaining_tiles <= 0 || remaining_partitions <= 0) {
    return 0;
  }
  if (remaining_partitions == 1) {
    return remaining_tiles;
  }

  int const target_tiles =
      sm70_splitk_ceil_div_int(remaining_tiles, remaining_partitions);
  int const max_current_tiles = remaining_tiles - (remaining_partitions - 1);
  int partition_tiles = target_tiles;
  if (group_tiles > 1) {
    int const rounded_tiles =
        sm70_splitk_ceil_div_int(partition_tiles, group_tiles) * group_tiles;
    if (rounded_tiles <= max_current_tiles) {
      partition_tiles = rounded_tiles;
    }
  }
  return sm70_splitk_min_int(partition_tiles, max_current_tiles);
}

template <int GroupSize>
CUTLASS_HOST_DEVICE Sm70SplitKPartition sm70_splitk_partition(
    int k, int requested_split_k, int partition_idx) {
  int const active_split_k = sm70_active_split_k(k, requested_split_k);
  if (partition_idx >= active_split_k) {
    return {0, 0};
  }

  int const group_tiles = sm70_splitk_group_tiles<GroupSize>();
  int remaining_tiles = k / kCtaK;
  int start_tiles = 0;
  for (int idx = 0; idx < partition_idx; ++idx) {
    int const partition_tiles = sm70_splitk_partition_tile_count(
        remaining_tiles, active_split_k - idx, group_tiles);
    start_tiles += partition_tiles;
    remaining_tiles -= partition_tiles;
  }

  int const partition_tiles = sm70_splitk_partition_tile_count(
      remaining_tiles, active_split_k - partition_idx, group_tiles);
  return {start_tiles * kCtaK, partition_tiles * kCtaK};
}

template <typename Traits>
class Sm70AtomicFp32Epilogue {
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
                atomicAdd(c_tmp + int64_t(logical_row) * n +
                              logical_column_base + e,
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
  Sm70AtomicFp32Epilogue(SharedStorage& shared_storage, int thread_idx,
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

static __global__ void sm70_fp32_to_fp16_kernel(
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

inline void launch_sm70_fp32_to_fp16(float const* c_tmp,
                                           cutlass::half_t* c,
                                           int64_t numel,
                                           cudaStream_t stream) {
  dim3 convert_block(256);
  dim3 convert_grid(static_cast<unsigned>(
      (numel + int64_t(convert_block.x) * 4 - 1) /
      (int64_t(convert_block.x) * 4)));
  sm70_fp32_to_fp16_kernel<<<convert_grid, convert_block, 0, stream>>>(
      c_tmp, c, numel);
}

}  // namespace marlin::sm70

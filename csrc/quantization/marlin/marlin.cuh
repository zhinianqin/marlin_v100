#pragma once

#ifndef _marlin_cuh
  #define _marlin_cuh
  #include <torch/all.h>

  #include <ATen/cuda/CUDAContext.h>
  #include <c10/cuda/CUDAGuard.h>
  #include <cuda.h>
  #include <cuda_fp16.h>
  #include <cuda_runtime.h>
  #include <iostream>

  #ifndef MARLIN_NAMESPACE_NAME
    #define MARLIN_NAMESPACE_NAME marlin
  #endif

namespace MARLIN_NAMESPACE_NAME {

// Marlin params

static constexpr int min_thread_n = 64;

static constexpr int tile_size = 16;

// Repack params
static constexpr int repack_stages = 8;

static constexpr int repack_threads = 256;

static constexpr int tile_k_size = tile_size;
static constexpr int tile_n_size = tile_k_size * 4;

constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }

__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  reinterpret_cast<int4*>(smem_ptr)[0] =
      reinterpret_cast<const int4*>(glob_ptr)[0];
}

__device__ inline void cp_async_fence() {}

template <int n>
__device__ inline void cp_async_wait() {}

}  // namespace MARLIN_NAMESPACE_NAME

#endif

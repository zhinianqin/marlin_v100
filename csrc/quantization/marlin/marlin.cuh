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

// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
static constexpr int default_threads = 256;

static constexpr int pipe_stages =
    4;  // 4 pipeline stages fit into shared memory

static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;
static constexpr int max_thread_n = 256;

static constexpr int tile_size = 16;
static constexpr int max_par = 16;

// Repack params
static constexpr int repack_stages = 8;

static constexpr int repack_threads = 256;

static constexpr int tile_k_size = tile_size;
static constexpr int tile_n_size = tile_k_size * 4;

// Helpers
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
};

using I4 = Vec<int, 4>;

constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }

__device__ inline void cp_async1_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  if (pred) {
    reinterpret_cast<int32_t*>(smem_ptr)[0] =
        reinterpret_cast<const int32_t*>(glob_ptr)[0];
  }
}

__device__ inline void cp_async2_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  if (pred) {
    reinterpret_cast<int64_t*>(smem_ptr)[0] =
        reinterpret_cast<const int64_t*>(glob_ptr)[0];
  }
}

__device__ __forceinline__ void load_global_cg_u32x4(
    const void* glob_ptr, uint32_t& r0, uint32_t& r1, uint32_t& r2,
    uint32_t& r3) {
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "l"(glob_ptr));
}

__device__ __forceinline__ void store_u32x4(void* smem_ptr, uint32_t r0,
                                            uint32_t r1, uint32_t r2,
                                            uint32_t r3) {
  reinterpret_cast<uint4*>(smem_ptr)[0] = make_uint4(r0, r1, r2, r3);
}

__device__ inline void cp_async4_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  if (pred) {
    uint32_t r0, r1, r2, r3;
    load_global_cg_u32x4(glob_ptr, r0, r1, r2, r3);
    store_u32x4(smem_ptr, r0, r1, r2, r3);
  }
}

__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr,
                                      bool pred = true) {
  if (pred) {
    uint32_t r0, r1, r2, r3;
    load_global_cg_u32x4(glob_ptr, r0, r1, r2, r3);
    store_u32x4(smem_ptr, r0, r1, r2, r3);
  }
}

__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  uint32_t r0, r1, r2, r3;
  load_global_cg_u32x4(glob_ptr, r0, r1, r2, r3);
  store_u32x4(smem_ptr, r0, r1, r2, r3);
}

__device__ inline void cp_async_fence() {}

template <int n>
__device__ inline void cp_async_wait() {}

}  // namespace MARLIN_NAMESPACE_NAME

#endif

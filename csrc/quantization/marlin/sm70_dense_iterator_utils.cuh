#pragma once

#include <cuda_fp16.h>

#include <cstdint>

#include "cutlass/cutlass.h"
#include "quantization/marlin/sm70_dense_common.cuh"

namespace marlin::sm70_dense {

template <int ShapeK>
CUTLASS_DEVICE int initial_k_advance(int size_k) {
  int const residue_k = size_k % ShapeK;
  return residue_k == 0 ? ShapeK : residue_k;
}

CUTLASS_DEVICE uint32_t qword_from_vector(uint4 const& words, int c) {
  uint32_t const* words_ptr = reinterpret_cast<uint32_t const*>(&words);
  return words_ptr[c];
}

CUTLASS_DEVICE uint32_t qword_from_vector(uint2 const& words, int c) {
  uint32_t const* words_ptr = reinterpret_cast<uint32_t const*>(&words);
  return words_ptr[c];
}

template <bool ResidueN>
CUTLASS_DEVICE int u4_qweight_offset_from_logical(int size_n, int logical_k,
                                                  int logical_n) {
  int const k_tile = logical_k / kQuantTileK;
  int const local_k = logical_k - k_tile * kQuantTileK;
  int const n_tile = logical_n / kQuantTileN;
  int const macro_n_tile = n_tile / kMacroNTiles;
  int const macro_first_n_tile = macro_n_tile * kMacroNTiles;
  int const subtile = n_tile - macro_first_n_tile;
  int subtile_count = kMacroNTiles;
  if constexpr (ResidueN) {
    subtile_count = size_n / kQuantTileN - macro_first_n_tile;
    subtile_count =
        subtile_count < kMacroNTiles ? subtile_count : kMacroNTiles;
  }
  int const local_n_vec = (logical_n - n_tile * kQuantTileN) / 8;
  int const local_word = local_k * (kQuantTileN / 8) + local_n_vec;

  return k_tile * (size_n * 2) +
         macro_n_tile * kMacroNTiles * (kQuantTileK * kQuantTileN / 8) +
         local_word * subtile_count + subtile;
}

template <bool ResidueN>
CUTLASS_DEVICE int u8_qweight_offset_from_logical(int size_n, int logical_k,
                                                  int logical_n) {
  int const k_tile = logical_k / kQuantTileK;
  int const local_k = logical_k - k_tile * kQuantTileK;
  int const n_tile = logical_n / kQuantTileN;
  int const macro_n_tile = n_tile / kMacroNTiles;
  int const macro_first_n_tile = macro_n_tile * kMacroNTiles;
  int const subtile = n_tile - macro_first_n_tile;
  int subtile_count = kMacroNTiles;
  if constexpr (ResidueN) {
    subtile_count = size_n / kQuantTileN - macro_first_n_tile;
    subtile_count =
        subtile_count < kMacroNTiles ? subtile_count : kMacroNTiles;
  }
  int const local_n_word = (logical_n - n_tile * kQuantTileN) / 4;
  int const local_word = local_k * (kQuantTileN / 4) + local_n_word;

  return k_tile * (size_n * 4) +
         macro_n_tile * kMacroNTiles * (kQuantTileK * kQuantTileN / 4) +
         local_word * subtile_count + subtile;
}

template <bool ResidueN>
CUTLASS_DEVICE int u8_qweight_word_stride_from_logical(int size_n,
                                                       int logical_n) {
  if constexpr (ResidueN) {
    int const n_tile = logical_n / kQuantTileN;
    int const macro_n_tile = n_tile / kMacroNTiles;
    int const macro_first_n_tile = macro_n_tile * kMacroNTiles;
    int subtile_count = size_n / kQuantTileN - macro_first_n_tile;
    return subtile_count < kMacroNTiles ? subtile_count : kMacroNTiles;
  } else {
    return kMacroNTiles;
  }
}

}  // namespace marlin::sm70_dense

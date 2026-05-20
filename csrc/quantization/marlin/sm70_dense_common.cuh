#pragma once

#include <torch/library.h>

#include <cstdlib>
#include <cstdint>
#include <sstream>
#include <string>

#include "cutlass/gemm/gemm.h"

namespace marlin::sm70_dense {

constexpr int kCtaK = 32;
constexpr int kDefaultCtaM = 128;
constexpr int kDefaultCtaN = 256;
constexpr int kDefaultWarps = 8;
constexpr int kQuantTileK = 16;
constexpr int kQuantTileN = 64;
constexpr int kMacroNTiles = 4;
constexpr int kMacroN = kQuantTileN * kMacroNTiles;

template <int CtaM, int CtaN, int Warps>
struct Sm70DenseWarpShape;

template <>
struct Sm70DenseWarpShape<32, 128, 4> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

template <>
struct Sm70DenseWarpShape<32, 256, 4> {
  using Type = cutlass::gemm::GemmShape<32, 64, 32>;
};

template <>
struct Sm70DenseWarpShape<64, 64, 4> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

template <>
struct Sm70DenseWarpShape<64, 128, 4> {
  using Type = cutlass::gemm::GemmShape<32, 64, 32>;
};

template <>
struct Sm70DenseWarpShape<64, 128, 8> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

template <>
struct Sm70DenseWarpShape<64, 256, 4> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70DenseWarpShape<64, 256, 8> {
  using Type = cutlass::gemm::GemmShape<32, 64, 32>;
};

template <>
struct Sm70DenseWarpShape<128, 64, 4> {
  using Type = cutlass::gemm::GemmShape<64, 32, 32>;
};

template <>
struct Sm70DenseWarpShape<128, 64, 8> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

template <>
struct Sm70DenseWarpShape<128, 128, 4> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70DenseWarpShape<128, 128, 8> {
  using Type = cutlass::gemm::GemmShape<64, 32, 32>;
};

template <>
struct Sm70DenseWarpShape<128, 256, 8> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70DenseWarpShape<256, 64, 4> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70DenseWarpShape<256, 64, 8> {
  using Type = cutlass::gemm::GemmShape<64, 32, 32>;
};

template <>
struct Sm70DenseWarpShape<256, 128, 8> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

struct Sm70DenseCtaGeometry {
  int cta_m;
  int cta_n;
  int warps;
};

inline constexpr char const* kSupportedSm70DenseCtaGeometries =
    "32x128x4, 32x256x4, 64x64x4, 64x128x4, 64x128x8, "
    "64x256x4, 64x256x8, 128x64x4, 128x64x8, 128x128x4, "
    "128x128x8, 128x256x8, 256x64x4, 256x64x8, and 256x128x8";

inline bool sm70_dense_cta_geometry_supported(
    Sm70DenseCtaGeometry geometry) {
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

inline Sm70DenseCtaGeometry parse_sm70_dense_cta_geometry(
    char const* env_name) {
  char const* env = std::getenv(env_name);
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
      (stream >> cta_m >> cta_n >> warps) && !(stream >> extra), env_name,
      " must use format CTA_MxCTA_NxWarps, for example 128x256x8. Got: ",
      env);
  return {cta_m, cta_n, warps};
}

inline void check_sm70_dense_cta_geometry(char const* env_name,
                                          Sm70DenseCtaGeometry geometry) {
  TORCH_CHECK(sm70_dense_cta_geometry_supported(geometry), "Unsupported ",
              env_name, "=", geometry.cta_m, "x", geometry.cta_n, "x",
              geometry.warps, ". Supported geometries are ",
              kSupportedSm70DenseCtaGeometries, ".");
}

inline void check_sm70_dense_full_n_tile(char const* env_name,
                                         Sm70DenseCtaGeometry geometry,
                                         int64_t size_n) {
  TORCH_CHECK(size_n % geometry.cta_n == 0 && size_n % kMacroN == 0,
              "SM70 CUTLASS dense prototype requires full-N tiles for ",
              env_name, "=", geometry.cta_m, "x", geometry.cta_n, "x",
              geometry.warps, ". size_n must be divisible by both CTA_N and ",
              kMacroN, ". Got size_n = ", size_n, ".");
}

}  // namespace marlin::sm70_dense

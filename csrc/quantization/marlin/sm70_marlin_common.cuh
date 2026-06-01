#pragma once

#include <torch/library.h>

#include <cstdlib>
#include <cstdint>
#include <sstream>
#include <string>

#include "cutlass/gemm/gemm.h"

namespace marlin::sm70 {

constexpr int kCtaK = 32;
constexpr int kQuantTileK = 16;
constexpr int kQuantTileN = 64;

template <int CtaM, int CtaN, int Warps>
struct Sm70WarpShape;

template <>
struct Sm70WarpShape<32, 128, 4> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

template <>
struct Sm70WarpShape<32, 256, 4> {
  using Type = cutlass::gemm::GemmShape<32, 64, 32>;
};

template <>
struct Sm70WarpShape<64, 64, 4> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

template <>
struct Sm70WarpShape<64, 128, 4> {
  using Type = cutlass::gemm::GemmShape<32, 64, 32>;
};

template <>
struct Sm70WarpShape<64, 128, 8> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

template <>
struct Sm70WarpShape<64, 256, 4> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70WarpShape<64, 256, 8> {
  using Type = cutlass::gemm::GemmShape<32, 64, 32>;
};

template <>
struct Sm70WarpShape<128, 64, 4> {
  using Type = cutlass::gemm::GemmShape<64, 32, 32>;
};

template <>
struct Sm70WarpShape<128, 64, 8> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

template <>
struct Sm70WarpShape<128, 128, 4> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70WarpShape<128, 128, 8> {
  using Type = cutlass::gemm::GemmShape<64, 32, 32>;
};

template <>
struct Sm70WarpShape<128, 256, 8> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70WarpShape<256, 64, 4> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

template <>
struct Sm70WarpShape<256, 64, 8> {
  using Type = cutlass::gemm::GemmShape<64, 32, 32>;
};

template <>
struct Sm70WarpShape<256, 128, 8> {
  using Type = cutlass::gemm::GemmShape<64, 64, 32>;
};

struct Sm70CtaGeometry {
  int cta_m;
  int cta_n;
  int warps;
};

inline constexpr char const* kSupportedSm70MarlinCtaGeometries =
    "32x128x4, 32x256x4, 64x64x4, 64x128x4, 64x128x8, "
    "64x256x4, 64x256x8, 128x64x4, 128x64x8, 128x128x4, "
    "128x128x8, 128x256x8, 256x64x4, 256x64x8, and 256x128x8";

inline bool sm70_marlin_cta_geometry_supported(
    Sm70CtaGeometry geometry) {
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

inline Sm70CtaGeometry parse_sm70_marlin_cta_geometry(
    char const* env_name) {
  char const* env = std::getenv(env_name);
  TORCH_CHECK(env != nullptr && env[0] != '\0', env_name,
              " must use format CTA_MxCTA_NxWarps when explicitly parsed, "
              "for example 128x256x8.");

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

inline int sm70_marlin_auto_cta_n(int64_t size_n) {
  if (size_n % 256 == 0) {
    return 256;
  }
  if (size_n % 128 == 0) {
    return 128;
  }
  if (size_n % 64 == 0) {
    return 64;
  }
  TORCH_CHECK(false, "SM70 CUTLASS Marlin requires size_n divisible "
                     "by 64. Got size_n = ", size_n, ".");
  return 0;
}

inline int sm70_marlin_auto_cta_m(int64_t size_m, int auto_cta_n) {
  if (auto_cta_n == 64) {
    if (size_m >= 256) {
      return 256;
    }
    if (size_m >= 128) {
      return 128;
    }
    return 64;
  }
  if (auto_cta_n == 128) {
    if (size_m >= 256) {
      return 256;
    }
    if (size_m >= 128) {
      return 128;
    }
    if (size_m >= 64) {
      return 64;
    }
    return 32;
  }
  if (auto_cta_n == 256) {
    if (size_m >= 128) {
      return 128;
    }
    if (size_m >= 64) {
      return 64;
    }
    return 32;
  }
  TORCH_CHECK(false, "SM70 Marlin auto CTA_M requires CTA_N in {64, 128, 256}. "
                     "Got CTA_N = ", auto_cta_n, ".");
  return 0;
}

inline int sm70_marlin_default_warps(int auto_cta_m, int auto_cta_n) {
  if (auto_cta_n == 64) {
    if (auto_cta_m == 64) {
      return 4;
    }
    if (auto_cta_m == 128 || auto_cta_m == 256) {
      return 8;
    }
  } else if (auto_cta_n == 128) {
    if (auto_cta_m == 32) {
      return 4;
    }
    if (auto_cta_m == 64 || auto_cta_m == 128 || auto_cta_m == 256) {
      return 8;
    }
  } else if (auto_cta_n == 256) {
    if (auto_cta_m == 32) {
      return 4;
    }
    if (auto_cta_m == 64 || auto_cta_m == 128) {
      return 8;
    }
  }
  TORCH_CHECK(false, "SM70 Marlin auto CTA default warps has no supported "
                     "geometry for CTA_M=", auto_cta_m,
                     ", CTA_N=", auto_cta_n, ".");
  return 0;
}

inline Sm70CtaGeometry resolve_sm70_marlin_cta_geometry(
    char const* env_name, int64_t size_m, int64_t size_n) {
  int const auto_cta_n = sm70_marlin_auto_cta_n(size_n);
  int const auto_cta_m = sm70_marlin_auto_cta_m(size_m, auto_cta_n);
  char const* env = std::getenv(env_name);
  if (env == nullptr || env[0] == '\0') {
    return {auto_cta_m, auto_cta_n,
            sm70_marlin_default_warps(auto_cta_m, auto_cta_n)};
  }

  Sm70CtaGeometry geometry = parse_sm70_marlin_cta_geometry(env_name);
  TORCH_CHECK(geometry.cta_m == auto_cta_m && geometry.cta_n == auto_cta_n,
              env_name, " specifies CTA_M=", geometry.cta_m, ", CTA_N=",
              geometry.cta_n, " but size_m=", size_m, ", size_n=", size_n,
              " requires auto CTA_M=", auto_cta_m, ", auto CTA_N=",
              auto_cta_n, ". CTA_M and CTA_N are selected automatically and "
              "are not free SM70 Marlin tuning parameters.");
  return geometry;
}

inline void check_sm70_marlin_cta_geometry(char const* env_name,
                                          Sm70CtaGeometry geometry) {
  TORCH_CHECK(sm70_marlin_cta_geometry_supported(geometry), "Unsupported ",
              env_name, "=", geometry.cta_m, "x", geometry.cta_n, "x",
              geometry.warps, ". Supported geometries are ",
              kSupportedSm70MarlinCtaGeometries, ".");
}

inline void check_sm70_marlin_n_tile_alignment(char const* env_name,
                                               Sm70CtaGeometry geometry,
                                               int64_t size_n) {
  TORCH_CHECK(size_n % kQuantTileN == 0,
              "SM70 CUTLASS Marlin requires size_n divisible by ",
              kQuantTileN, ". Got size_n = ", size_n, ".");
  TORCH_CHECK(size_n % geometry.cta_n == 0,
              "SM70 CUTLASS Marlin requires size_n divisible by "
              "CTA_N for ",
              env_name, "=", geometry.cta_m, "x", geometry.cta_n, "x",
              geometry.warps, ". Got size_n = ", size_n, ".");
}

}  // namespace marlin::sm70

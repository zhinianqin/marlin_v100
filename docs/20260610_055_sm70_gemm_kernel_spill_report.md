# SM70 GEMM Kernel Spill Report

Date: 2026-06-10

Branch: `experiment/sm70-hardcoded-geometry-benchmark-20260610`

Commit: `1198212`

## Summary

This report records the `ptxas -v` register and spill status from rebuilding the
workspace with the repository build script.

Command:

```bash
./build.sh 2>&1 | tee /tmp/marlin_v100_build_ptxas.log
```

Build configuration reported by `./build.sh`:

```text
CUDA_HOME: /usr/local/cuda-12.8
TORCH_CUDA_ARCH_LIST: 7.0
CMAKE_ARGS: '-DCMAKE_CUDA_FLAGS=-gencode arch=compute_70,code=sm_70 -Xptxas=-v'
```

Result:

```text
Build status: success
Scanned GEMM kernels: 580
No spill: 541
Spill: 39
Unknown: 0
Max registers/thread: 255
Max stack frame: 232 bytes
Max spill stores: 232 bytes
Max spill loads: 428 bytes
```

Temporary raw artifacts from this run:

```text
/tmp/marlin_v100_build_ptxas.log
/tmp/marlin_v100_gemm_kernel_spill.csv
/tmp/marlin_v100_gemm_kernel_spill_readable.csv
```

The full spill table is copied below so the important result is persisted in
`docs/`.

## Quant Summary

| Path | Quant | Kernels | Spill | No Spill | Max Regs | Max Stack | Max Spill Stores | Max Spill Loads |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | fp8 | 30 | 0 | 30 | 255 | 0 | 0 | 0 |
| dense | mxfp4 | 15 | 0 | 15 | 255 | 0 | 0 | 0 |
| dense | nvfp4 | 15 | 0 | 15 | 255 | 0 | 0 | 0 |
| dense | u4 | 60 | 1 | 59 | 255 | 40 | 36 | 36 |
| dense | u4b8 | 60 | 0 | 60 | 255 | 0 | 0 | 0 |
| dense | u8 | 60 | 1 | 59 | 255 | 24 | 24 | 24 |
| dense | u8b128 | 60 | 0 | 60 | 255 | 0 | 0 | 0 |
| moe | fp8 | 28 | 4 | 24 | 255 | 64 | 64 | 156 |
| moe | mxfp4 | 14 | 0 | 14 | 255 | 0 | 0 | 0 |
| moe | nvfp4 | 14 | 1 | 13 | 255 | 24 | 24 | 24 |
| moe | u4 | 56 | 8 | 48 | 255 | 232 | 232 | 428 |
| moe | u4b8 | 56 | 8 | 48 | 255 | 40 | 52 | 40 |
| moe | u8 | 56 | 8 | 48 | 255 | 176 | 176 | 340 |
| moe | u8b128 | 56 | 8 | 48 | 255 | 64 | 64 | 156 |

## Spill Kernels

All spill cases are `regs=255` and are concentrated on `cta64x256x4`. Kernels
not listed in this table reported zero stack frame, zero spill stores, and zero
spill loads.

| Path | Quant | Kernel | Regs | Stack | Spill Stores | Spill Loads |
|---|---|---|---:|---:|---:|---:|
| moe | u4 | `cta64x256x4_g-1_splitk` | 255 | 232 | 228 | 428 |
| moe | u4 | `cta64x256x4_g-1_main` | 255 | 232 | 232 | 368 |
| moe | u8 | `cta64x256x4_g-1_splitk` | 255 | 176 | 172 | 340 |
| moe | u8 | `cta64x256x4_g-1_main` | 255 | 176 | 176 | 280 |
| moe | fp8 | `cta64x256x4_g-1_splitk` | 255 | 64 | 60 | 156 |
| moe | u8b128 | `cta64x256x4_g-1_splitk` | 255 | 64 | 60 | 156 |
| moe | u8 | `cta64x256x4_g128_splitk` | 255 | 64 | 64 | 136 |
| moe | u8 | `cta64x256x4_g64_splitk` | 255 | 64 | 64 | 136 |
| moe | u8 | `cta64x256x4_g32_splitk` | 255 | 64 | 64 | 136 |
| moe | fp8 | `cta64x256x4_g128_splitk` | 255 | 56 | 56 | 120 |
| moe | fp8 | `cta64x256x4_g-1_main` | 255 | 64 | 64 | 96 |
| moe | u8b128 | `cta64x256x4_g-1_main` | 255 | 64 | 64 | 96 |
| moe | u8 | `cta64x256x4_g128_main` | 255 | 72 | 68 | 76 |
| moe | u8 | `cta64x256x4_g64_main` | 255 | 72 | 68 | 76 |
| moe | u8 | `cta64x256x4_g32_main` | 255 | 72 | 68 | 76 |
| moe | u4 | `cta64x256x4_g128_splitk` | 255 | 40 | 40 | 104 |
| moe | u4 | `cta64x256x4_g64_splitk` | 255 | 40 | 40 | 104 |
| moe | u4 | `cta64x256x4_g32_splitk` | 255 | 40 | 40 | 104 |
| moe | u8b128 | `cta64x256x4_g128_splitk` | 255 | 40 | 40 | 104 |
| moe | u8b128 | `cta64x256x4_g64_splitk` | 255 | 40 | 40 | 104 |
| moe | u8b128 | `cta64x256x4_g32_splitk` | 255 | 40 | 40 | 104 |
| moe | fp8 | `cta64x256x4_g128_main` | 255 | 64 | 60 | 60 |
| moe | u4b8 | `cta64x256x4_g-1_main` | 255 | 40 | 52 | 40 |
| moe | u4 | `cta64x256x4_g128_main` | 255 | 48 | 44 | 44 |
| moe | u4 | `cta64x256x4_g64_main` | 255 | 48 | 44 | 44 |
| moe | u4 | `cta64x256x4_g32_main` | 255 | 48 | 44 | 44 |
| moe | u8b128 | `cta64x256x4_g128_main` | 255 | 48 | 44 | 44 |
| moe | u8b128 | `cta64x256x4_g64_main` | 255 | 48 | 44 | 44 |
| moe | u8b128 | `cta64x256x4_g32_main` | 255 | 48 | 44 | 44 |
| moe | u4b8 | `cta64x256x4_g-1_splitk` | 255 | 40 | 46 | 36 |
| dense | u4 | `cta64x256x4_g-1_main` | 255 | 40 | 36 | 36 |
| moe | nvfp4 | `cta64x256x4_g16_main` | 255 | 24 | 24 | 24 |
| dense | u8 | `cta64x256x4_g-1_main` | 255 | 24 | 24 | 24 |
| moe | u4b8 | `cta64x256x4_g128_splitk` | 255 | 8 | 8 | 40 |
| moe | u4b8 | `cta64x256x4_g64_splitk` | 255 | 8 | 8 | 40 |
| moe | u4b8 | `cta64x256x4_g32_splitk` | 255 | 8 | 8 | 40 |
| moe | u4b8 | `cta64x256x4_g128_main` | 255 | 16 | 12 | 12 |
| moe | u4b8 | `cta64x256x4_g64_main` | 255 | 16 | 12 | 12 |
| moe | u4b8 | `cta64x256x4_g32_main` | 255 | 16 | 12 | 12 |

## Observations

- `cta64x256x4` is the only geometry with spill in this build.
- The worst cases are MoE U4 and MoE U8 with `group_size=-1`.
- Dense spill is limited to two main kernels:
  - U4 `cta64x256x4_g-1_main`: 36B spill stores, 36B spill loads.
  - U8 `cta64x256x4_g-1_main`: 24B spill stores, 24B spill loads.
- Dense FP8, NVFP4, MXFP4, U4B8, and U8B128 have no spill.
- MoE MXFP4 has no spill.

# SM70 Marlin Hardcoded Geometry Benchmark

Date: 2026-06-10

Run ID: `20260610_204258_sm70_hardcoded_geometry_spill_rerun`

Branch: `refactor/cutlass-sm70-marlin-v100`

Commit: `819f59d`

Experiment change branch: `experiment/sm70-hardcoded-geometry-benchmark-20260610`

The complete experimental implementation, including hardcoded geometry compile
guards and benchmark/analyzer scripts, is committed on the experiment change
branch above. The base branch keeps only this Markdown report as the persistent
analysis record.

Diff stat:

```text
csrc/moe/marlin_moe_wna16/sm70_marlin_gemm.cuh  | 19 +++++++++++++
 csrc/quantization/marlin/sm70_marlin_common.cuh | 38 +++++++++++++++++++++++++
 csrc/quantization/marlin/sm70_marlin_splitk.cuh | 17 +++++++++++
 3 files changed, 74 insertions(+)
```

## Scope

This benchmark compares three separate hardcoded geometry builds:

- `64x256x4`
- `64x256x8`
- `32x256x4`

The results are from independent rebuilds, not runtime env CTA selection.
Split-K is fixed to `1` by the same experiment compile-time guard.

MoE rows use a single-stage `moe_wna16_marlin_gemm` benchmark with method-compatible quant layouts:
`awq_moe` covers zero-point U4/U8, while `gptq_moe` and `compressed_tensors_wna16_moe`
cover symmetric U4B8/U8B128. FP8/NVFP4/MXFP4 MoE method rows are recorded as
`SKIP_UNTIMED_METHOD` because this experiment did not add new timed method preparation paths.

Shape matrix:

- `M`: `1, 32, 64, 2048, 4096, 5120`
- `N`: `256, 512, 1024, 2048, 4096`
- `K`: `1024, 2048, 4096`

## Environment

```text
device=Tesla V100-SXM2-32GB
capability=sm70 (7.0)
build_target=SM70 (7.0)
torch=2.10.0+cu128
```

## Commands

```bash
./build.sh  # repeated once per hardcoded geometry with CMAKE_CUDA_FLAGS experiment macros
PYTHONPATH=$PWD ./.venv/bin/python benchmarks/benchmark_sm70_marlin_geometry_experiment.py --paths dense moe --geometry <geometry> --m-values 1 32 64 2048 4096 5120 --n-values 256 512 1024 2048 4096 --k-values 1024 2048 4096 --warmup-iters 1 --iters 5 --check --csv benchmarks/results/<run_id>_<geometry>_geometry_experiment.csv --jsonl benchmarks/results/<run_id>_<geometry>_geometry_experiment.jsonl
PYTHONPATH=$PWD ./.venv/bin/python benchmarks/analyze_sm70_marlin_geometry_experiment.py --run-id <run_id> --csv <three geometry csv files> --build-log <three build logs> --benchmark-log <three benchmark logs>
```

## Artifacts

CSV inputs:

```text
benchmarks/results/20260610_204258_sm70_hardcoded_geometry_spill_rerun_64x256x4_geometry_experiment.csv
benchmarks/results/20260610_204258_sm70_hardcoded_geometry_spill_rerun_64x256x8_geometry_experiment.csv
benchmarks/results/20260610_204258_sm70_hardcoded_geometry_spill_rerun_32x256x4_geometry_experiment.csv
```

Build logs:

```text
benchmarks/results/20260610_204258_sm70_hardcoded_geometry_spill_rerun_64x256x4_build.log
benchmarks/results/20260610_204258_sm70_hardcoded_geometry_spill_rerun_64x256x8_build.log
benchmarks/results/20260610_204258_sm70_hardcoded_geometry_spill_rerun_32x256x4_build.log
```

Benchmark logs:

```text
benchmarks/results/20260610_204258_sm70_hardcoded_geometry_spill_rerun_64x256x4_geometry_experiment.log
benchmarks/results/20260610_204258_sm70_hardcoded_geometry_spill_rerun_64x256x8_geometry_experiment.log
benchmarks/results/20260610_204258_sm70_hardcoded_geometry_spill_rerun_32x256x4_geometry_experiment.log
```

Summary JSON:

```text
benchmarks/results/20260610_204258_sm70_hardcoded_geometry_spill_rerun_sm70_geometry_compare_summary.json
```

## Status

| status | rows |
|---|---|
| OK | 21600 |
| SKIP_UNTIMED_METHOD | 1620 |


```json
{
  "OK": 21600,
  "SKIP_UNTIMED_METHOD": 1620
}
```

## Coverage

```json
{
  "K": {
    "1024": 7740,
    "2048": 7740,
    "4096": 7740
  },
  "M": {
    "1": 3870,
    "2048": 3870,
    "32": 3870,
    "4096": 3870,
    "5120": 3870,
    "64": 3870
  },
  "N": {
    "1024": 4644,
    "2048": 4644,
    "256": 4644,
    "4096": 4644,
    "512": 4644
  },
  "geometry": {
    "32x256x4": 7740,
    "64x256x4": 7740,
    "64x256x8": 7740
  },
  "group_size": {
    "-1": 6210,
    "128": 6210,
    "16": 540,
    "32": 5400,
    "64": 4860
  },
  "method_class": {
    "awq_marlin_linear_method": 2160,
    "awq_moe": 2160,
    "compressed_tensors_w4a16_mxfp4": 270,
    "compressed_tensors_w4a16_nvfp4": 270,
    "compressed_tensors_w4a4_mxfp4_moe": 270,
    "compressed_tensors_w4a4_nvfp4_moe": 270,
    "compressed_tensors_w8a16_fp8": 540,
    "compressed_tensors_w8a8_fp8_moe": 540,
    "compressed_tensors_wna16": 4320,
    "compressed_tensors_wna16_moe": 2160,
    "gptq_marlin_linear_method": 2160,
    "gptq_moe": 2160,
    "marlin_fp8_scaled_mm": 540,
    "marlin_linear_kernel": 4860,
    "quark_w8a8_fp8_moe": 540
  },
  "path": {
    "dense": 15120,
    "moe": 8100
  },
  "quant": {
    "fp8": 2700,
    "mxfp4": 540,
    "nvfp4": 540,
    "uint4": 4320,
    "uint4b8": 5400,
    "uint8": 4320,
    "uint8b128": 5400
  }
}
```

## Overall Best Geometry

| bucket | cases | 64x256x4 wins | 64x256x8 wins | 32x256x4 wins | 64x256x4 win rate | 64x256x8 win rate | 32x256x4 win rate | p10 speedup | p50 speedup | p90 speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | 7200 | 1859 | 1237 | 4104 | 0.258 | 0.172 | 0.570 | 1.0000 | 1.1789 | 1.4454 |


## By Path

| bucket | cases | 64x256x4 wins | 64x256x8 wins | 32x256x4 wins | 64x256x4 win rate | 64x256x8 win rate | 32x256x4 win rate | p10 speedup | p50 speedup | p90 speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 5040 | 1859 | 613 | 2568 | 0.369 | 0.122 | 0.510 | 1.0000 | 1.0990 | 1.3051 |
| moe | 2160 | 0 | 624 | 1536 | 0.000 | 0.289 | 0.711 | 1.1926 | 1.3279 | 1.6345 |


## By Method

| bucket | cases | 64x256x4 wins | 64x256x8 wins | 32x256x4 wins | 64x256x4 win rate | 64x256x8 win rate | 32x256x4 win rate | p10 speedup | p50 speedup | p90 speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| awq_marlin_linear_method | 720 | 264 | 15 | 441 | 0.367 | 0.021 | 0.613 | 1.0000 | 1.1084 | 1.3401 |
| awq_moe | 720 | 0 | 211 | 509 | 0.000 | 0.293 | 0.707 | 1.3117 | 1.4836 | 1.7587 |
| compressed_tensors_w4a16_mxfp4 | 90 | 33 | 12 | 45 | 0.367 | 0.133 | 0.500 | 1.0000 | 1.3982 | 1.6995 |
| compressed_tensors_w4a16_nvfp4 | 90 | 28 | 28 | 34 | 0.311 | 0.311 | 0.378 | 1.0000 | 1.2289 | 1.5368 |
| compressed_tensors_w8a16_fp8 | 180 | 66 | 55 | 59 | 0.367 | 0.306 | 0.328 | 1.0000 | 1.1158 | 1.2535 |
| compressed_tensors_wna16 | 1440 | 538 | 153 | 749 | 0.374 | 0.106 | 0.520 | 1.0000 | 1.0815 | 1.2882 |
| compressed_tensors_wna16_moe | 720 | 0 | 208 | 512 | 0.000 | 0.289 | 0.711 | 1.1800 | 1.2792 | 1.5311 |
| gptq_marlin_linear_method | 720 | 266 | 111 | 343 | 0.369 | 0.154 | 0.476 | 1.0000 | 1.0693 | 1.2108 |
| gptq_moe | 720 | 0 | 205 | 515 | 0.000 | 0.285 | 0.715 | 1.1818 | 1.2798 | 1.5319 |
| marlin_fp8_scaled_mm | 180 | 66 | 60 | 54 | 0.367 | 0.333 | 0.300 | 1.0000 | 1.1068 | 1.2468 |
| marlin_linear_kernel | 1620 | 598 | 179 | 843 | 0.369 | 0.110 | 0.520 | 1.0000 | 1.1238 | 1.3070 |


## By Quant

| bucket | cases | 64x256x4 wins | 64x256x8 wins | 32x256x4 wins | 64x256x4 win rate | 64x256x8 win rate | 32x256x4 win rate | p10 speedup | p50 speedup | p90 speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| fp8 | 540 | 198 | 154 | 188 | 0.367 | 0.285 | 0.348 | 1.0000 | 1.1152 | 1.2569 |
| mxfp4 | 90 | 33 | 12 | 45 | 0.367 | 0.133 | 0.500 | 1.0000 | 1.3982 | 1.6995 |
| nvfp4 | 90 | 28 | 28 | 34 | 0.311 | 0.311 | 0.378 | 1.0000 | 1.2289 | 1.5368 |
| uint4 | 1440 | 396 | 134 | 910 | 0.275 | 0.093 | 0.632 | 1.0000 | 1.2466 | 1.4915 |
| uint4b8 | 1800 | 408 | 335 | 1057 | 0.227 | 0.186 | 0.587 | 1.0000 | 1.1847 | 1.3737 |
| uint8 | 1440 | 399 | 144 | 897 | 0.277 | 0.100 | 0.623 | 1.0000 | 1.2099 | 1.5741 |
| uint8b128 | 1800 | 397 | 430 | 973 | 0.221 | 0.239 | 0.541 | 1.0000 | 1.1654 | 1.3518 |


## By Group Size

| bucket | cases | 64x256x4 wins | 64x256x8 wins | 32x256x4 wins | 64x256x4 win rate | 64x256x8 win rate | 32x256x4 win rate | p10 speedup | p50 speedup | p90 speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| -1 | 1890 | 508 | 305 | 1077 | 0.269 | 0.161 | 0.570 | 1.0000 | 1.1497 | 1.5898 |
| 128 | 1890 | 495 | 362 | 1033 | 0.262 | 0.192 | 0.547 | 1.0000 | 1.1764 | 1.3589 |
| 16 | 90 | 28 | 28 | 34 | 0.311 | 0.311 | 0.378 | 1.0000 | 1.2289 | 1.5368 |
| 32 | 1710 | 432 | 278 | 1000 | 0.253 | 0.163 | 0.585 | 1.0000 | 1.1980 | 1.4383 |
| 64 | 1620 | 396 | 264 | 960 | 0.244 | 0.163 | 0.593 | 1.0000 | 1.1902 | 1.3925 |


## By M Bucket

| bucket | cases | 64x256x4 wins | 64x256x8 wins | 32x256x4 wins | 64x256x4 win rate | 64x256x8 win rate | 32x256x4 win rate | p10 speedup | p50 speedup | p90 speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| M<=64 | 3600 | 11 | 857 | 2732 | 0.003 | 0.238 | 0.759 | 1.0892 | 1.2280 | 1.4326 |
| M>=2048 | 3600 | 1848 | 380 | 1372 | 0.513 | 0.106 | 0.381 | 1.0000 | 1.0000 | 1.4903 |


## By K

| bucket | cases | 64x256x4 wins | 64x256x8 wins | 32x256x4 wins | 64x256x4 win rate | 64x256x8 win rate | 32x256x4 win rate | p10 speedup | p50 speedup | p90 speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 2400 | 626 | 281 | 1493 | 0.261 | 0.117 | 0.622 | 1.0000 | 1.1000 | 1.3724 |
| 2048 | 2400 | 616 | 415 | 1369 | 0.257 | 0.173 | 0.570 | 1.0000 | 1.1917 | 1.4493 |
| 4096 | 2400 | 617 | 541 | 1242 | 0.257 | 0.225 | 0.517 | 1.0000 | 1.2335 | 1.5246 |


## By N

| bucket | cases | 64x256x4 wins | 64x256x8 wins | 32x256x4 wins | 64x256x4 win rate | 64x256x8 win rate | 32x256x4 win rate | p10 speedup | p50 speedup | p90 speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1440 | 506 | 127 | 807 | 0.351 | 0.088 | 0.560 | 1.0000 | 1.1661 | 1.4201 |
| 2048 | 1440 | 505 | 226 | 709 | 0.351 | 0.157 | 0.492 | 1.0000 | 1.1683 | 1.4481 |
| 256 | 1440 | 8 | 493 | 939 | 0.006 | 0.342 | 0.652 | 1.0693 | 1.1930 | 1.4330 |
| 4096 | 1440 | 502 | 172 | 766 | 0.349 | 0.119 | 0.532 | 1.0000 | 1.1671 | 1.4953 |
| 512 | 1440 | 338 | 219 | 883 | 0.235 | 0.152 | 0.613 | 1.0000 | 1.1858 | 1.4263 |


## g-1 vs g128 Sample

| path | method | quant | M | K | N | g-1 best | g128 best | g-1 speedup | g128 speedup |
|---|---|---|---|---|---|---|---|---|---|
| dense | awq_marlin_linear_method | uint4 | 1 | 1024 | 1024 | 32x256x4 | 32x256x4 | 1.039644 | 1.089151 |
| dense | awq_marlin_linear_method | uint4 | 1 | 1024 | 2048 | 32x256x4 | 32x256x4 | 1.039644 | 1.089151 |
| dense | awq_marlin_linear_method | uint4 | 1 | 1024 | 256 | 32x256x4 | 32x256x4 | 1.049507 | 1.089151 |
| dense | awq_marlin_linear_method | uint4 | 1 | 1024 | 4096 | 32x256x4 | 32x256x4 | 1.050000 | 1.099014 |
| dense | awq_marlin_linear_method | uint4 | 1 | 1024 | 512 | 32x256x4 | 32x256x4 | 1.029392 | 1.078411 |
| dense | awq_marlin_linear_method | uint4 | 1 | 2048 | 1024 | 32x256x4 | 32x256x4 | 1.236765 | 1.288173 |
| dense | awq_marlin_linear_method | uint4 | 1 | 2048 | 2048 | 32x256x4 | 32x256x4 | 1.236328 | 1.286940 |
| dense | awq_marlin_linear_method | uint4 | 1 | 2048 | 256 | 32x256x4 | 32x256x4 | 1.226053 | 1.500041 |
| dense | awq_marlin_linear_method | uint4 | 1 | 2048 | 4096 | 32x256x4 | 32x256x4 | 1.245472 | 1.286940 |
| dense | awq_marlin_linear_method | uint4 | 1 | 2048 | 512 | 32x256x4 | 32x256x4 | 1.236765 | 1.296615 |
| dense | awq_marlin_linear_method | uint4 | 1 | 4096 | 1024 | 32x256x4 | 32x256x4 | 1.333376 | 1.409984 |
| dense | awq_marlin_linear_method | uint4 | 1 | 4096 | 2048 | 32x256x4 | 32x256x4 | 1.326693 | 1.411397 |
| dense | awq_marlin_linear_method | uint4 | 1 | 4096 | 256 | 32x256x4 | 32x256x4 | 1.333376 | 1.164113 |
| dense | awq_marlin_linear_method | uint4 | 1 | 4096 | 4096 | 32x256x4 | 32x256x4 | 1.297154 | 1.483184 |
| dense | awq_marlin_linear_method | uint4 | 1 | 4096 | 512 | 32x256x4 | 32x256x4 | 1.339886 | 1.403737 |
| dense | awq_marlin_linear_method | uint4 | 2048 | 1024 | 1024 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 2048 | 1024 | 2048 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 2048 | 1024 | 256 | 32x256x4 | 32x256x4 | 1.079288 | 1.098037 |
| dense | awq_marlin_linear_method | uint4 | 2048 | 1024 | 4096 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 2048 | 1024 | 512 | 32x256x4 | 64x256x8 | 1.094343 | 1.071410 |
| dense | awq_marlin_linear_method | uint4 | 2048 | 2048 | 1024 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 2048 | 2048 | 2048 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 2048 | 2048 | 256 | 32x256x4 | 32x256x4 | 1.238431 | 1.280526 |
| dense | awq_marlin_linear_method | uint4 | 2048 | 2048 | 4096 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 2048 | 2048 | 512 | 32x256x4 | 64x256x8 | 1.139862 | 1.133731 |
| dense | awq_marlin_linear_method | uint4 | 2048 | 4096 | 1024 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 2048 | 4096 | 2048 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 2048 | 4096 | 256 | 32x256x4 | 32x256x4 | 1.326080 | 1.375619 |
| dense | awq_marlin_linear_method | uint4 | 2048 | 4096 | 4096 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 2048 | 4096 | 512 | 32x256x4 | 32x256x4 | 1.183564 | 1.186460 |
| dense | awq_marlin_linear_method | uint4 | 32 | 1024 | 1024 | 32x256x4 | 32x256x4 | 1.049507 | 1.100000 |
| dense | awq_marlin_linear_method | uint4 | 32 | 1024 | 2048 | 32x256x4 | 32x256x4 | 1.049507 | 1.099014 |
| dense | awq_marlin_linear_method | uint4 | 32 | 1024 | 256 | 32x256x4 | 32x256x4 | 1.049507 | 1.158480 |
| dense | awq_marlin_linear_method | uint4 | 32 | 1024 | 4096 | 32x256x4 | 32x256x4 | 1.059961 | 1.099014 |
| dense | awq_marlin_linear_method | uint4 | 32 | 1024 | 512 | 32x256x4 | 32x256x4 | 1.049507 | 1.089151 |
| dense | awq_marlin_linear_method | uint4 | 32 | 2048 | 1024 | 32x256x4 | 32x256x4 | 1.234800 | 1.264487 |
| dense | awq_marlin_linear_method | uint4 | 32 | 2048 | 2048 | 32x256x4 | 32x256x4 | 1.252332 | 1.273516 |
| dense | awq_marlin_linear_method | uint4 | 32 | 2048 | 256 | 32x256x4 | 32x256x4 | 1.224196 | 1.264487 |
| dense | awq_marlin_linear_method | uint4 | 32 | 2048 | 4096 | 32x256x4 | 32x256x4 | 1.241085 | 1.262766 |
| dense | awq_marlin_linear_method | uint4 | 32 | 2048 | 512 | 32x256x4 | 32x256x4 | 1.232783 | 1.264487 |
| dense | awq_marlin_linear_method | uint4 | 32 | 4096 | 1024 | 32x256x4 | 32x256x4 | 1.335496 | 1.371265 |
| dense | awq_marlin_linear_method | uint4 | 32 | 4096 | 2048 | 32x256x4 | 32x256x4 | 1.333376 | 1.371919 |
| dense | awq_marlin_linear_method | uint4 | 32 | 4096 | 256 | 32x256x4 | 32x256x4 | 1.341923 | 1.379574 |
| dense | awq_marlin_linear_method | uint4 | 32 | 4096 | 4096 | 32x256x4 | 32x256x4 | 1.327669 | 1.388605 |
| dense | awq_marlin_linear_method | uint4 | 32 | 4096 | 512 | 32x256x4 | 32x256x4 | 1.324812 | 1.371265 |
| dense | awq_marlin_linear_method | uint4 | 4096 | 1024 | 1024 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 4096 | 1024 | 2048 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 4096 | 1024 | 256 | 32x256x4 | 64x256x8 | 1.099964 | 1.121773 |
| dense | awq_marlin_linear_method | uint4 | 4096 | 1024 | 4096 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 4096 | 1024 | 512 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 4096 | 2048 | 1024 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 4096 | 2048 | 2048 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 4096 | 2048 | 256 | 32x256x4 | 64x256x8 | 1.137931 | 1.151925 |
| dense | awq_marlin_linear_method | uint4 | 4096 | 2048 | 4096 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 4096 | 2048 | 512 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 4096 | 4096 | 1024 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 4096 | 4096 | 2048 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 4096 | 4096 | 256 | 32x256x4 | 64x256x8 | 1.178396 | 1.204993 |
| dense | awq_marlin_linear_method | uint4 | 4096 | 4096 | 4096 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 4096 | 4096 | 512 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 5120 | 1024 | 1024 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 5120 | 1024 | 2048 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 5120 | 1024 | 256 | 32x256x4 | 64x256x8 | 1.078893 | 1.111093 |
| dense | awq_marlin_linear_method | uint4 | 5120 | 1024 | 4096 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 5120 | 1024 | 512 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 5120 | 2048 | 1024 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 5120 | 2048 | 2048 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 5120 | 2048 | 256 | 32x256x4 | 64x256x8 | 1.127474 | 1.151925 |
| dense | awq_marlin_linear_method | uint4 | 5120 | 2048 | 4096 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 5120 | 2048 | 512 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 5120 | 4096 | 1024 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 5120 | 4096 | 2048 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 5120 | 4096 | 256 | 32x256x4 | 64x256x8 | 1.170514 | 1.207344 |
| dense | awq_marlin_linear_method | uint4 | 5120 | 4096 | 4096 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 5120 | 4096 | 512 | 64x256x4 | 64x256x4 | 1.000000 | 1.000000 |
| dense | awq_marlin_linear_method | uint4 | 64 | 1024 | 1024 | 32x256x4 | 32x256x4 | 1.059466 | 1.099014 |
| dense | awq_marlin_linear_method | uint4 | 64 | 1024 | 2048 | 32x256x4 | 32x256x4 | 1.070020 | 1.108973 |
| dense | awq_marlin_linear_method | uint4 | 64 | 1024 | 256 | 32x256x4 | 32x256x4 | 1.069329 | 1.188165 |
| dense | awq_marlin_linear_method | uint4 | 64 | 1024 | 4096 | 32x256x4 | 32x256x4 | 1.069329 | 1.120020 |
| dense | awq_marlin_linear_method | uint4 | 64 | 1024 | 512 | 32x256x4 | 32x256x4 | 1.069329 | 1.108973 |


## Best Geometry Sample

| path | method | quant | group | M | K | N | best | speedup |
|---|---|---|---|---|---|---|---|---|
| dense | marlin_linear_kernel | uint4 | -1 | 1 | 1024 | 256 | 32x256x4 | 1.095238 |
| dense | marlin_linear_kernel | uint4 | -1 | 32 | 1024 | 256 | 32x256x4 | 1.106855 |
| dense | marlin_linear_kernel | uint4 | -1 | 64 | 1024 | 256 | 32x256x4 | 1.158480 |
| dense | marlin_linear_kernel | uint4 | -1 | 2048 | 1024 | 256 | 32x256x4 | 1.135963 |
| dense | marlin_linear_kernel | uint4 | -1 | 4096 | 1024 | 256 | 32x256x4 | 1.104365 |
| dense | marlin_linear_kernel | uint4 | -1 | 5120 | 1024 | 256 | 32x256x4 | 1.101713 |
| dense | marlin_linear_kernel | uint4 | -1 | 1 | 1024 | 512 | 32x256x4 | 1.108973 |
| dense | marlin_linear_kernel | uint4 | -1 | 32 | 1024 | 512 | 32x256x4 | 1.128795 |
| dense | marlin_linear_kernel | uint4 | -1 | 64 | 1024 | 512 | 32x256x4 | 1.148521 |
| dense | marlin_linear_kernel | uint4 | -1 | 2048 | 1024 | 512 | 32x256x4 | 1.097399 |
| dense | marlin_linear_kernel | uint4 | -1 | 4096 | 1024 | 512 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 5120 | 1024 | 512 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 1 | 1024 | 1024 | 32x256x4 | 1.107803 |
| dense | marlin_linear_kernel | uint4 | -1 | 32 | 1024 | 1024 | 32x256x4 | 1.118836 |
| dense | marlin_linear_kernel | uint4 | -1 | 64 | 1024 | 1024 | 32x256x4 | 1.159961 |
| dense | marlin_linear_kernel | uint4 | -1 | 2048 | 1024 | 1024 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 4096 | 1024 | 1024 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 5120 | 1024 | 1024 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 1 | 1024 | 2048 | 32x256x4 | 1.108973 |
| dense | marlin_linear_kernel | uint4 | -1 | 32 | 1024 | 2048 | 32x256x4 | 1.129980 |
| dense | marlin_linear_kernel | uint4 | -1 | 64 | 1024 | 2048 | 32x256x4 | 1.150000 |
| dense | marlin_linear_kernel | uint4 | -1 | 2048 | 1024 | 2048 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 4096 | 1024 | 2048 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 5120 | 1024 | 2048 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 1 | 1024 | 4096 | 32x256x4 | 1.129980 |
| dense | marlin_linear_kernel | uint4 | -1 | 32 | 1024 | 4096 | 32x256x4 | 1.129980 |
| dense | marlin_linear_kernel | uint4 | -1 | 64 | 1024 | 4096 | 32x256x4 | 1.127429 |
| dense | marlin_linear_kernel | uint4 | -1 | 2048 | 1024 | 4096 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 4096 | 1024 | 4096 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 5120 | 1024 | 4096 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 1 | 2048 | 256 | 32x256x4 | 1.254062 |
| dense | marlin_linear_kernel | uint4 | -1 | 32 | 2048 | 256 | 32x256x4 | 1.257993 |
| dense | marlin_linear_kernel | uint4 | -1 | 64 | 2048 | 256 | 32x256x4 | 1.282249 |
| dense | marlin_linear_kernel | uint4 | -1 | 2048 | 2048 | 256 | 32x256x4 | 1.266123 |
| dense | marlin_linear_kernel | uint4 | -1 | 4096 | 2048 | 256 | 32x256x4 | 1.147490 |
| dense | marlin_linear_kernel | uint4 | -1 | 5120 | 2048 | 256 | 32x256x4 | 1.139255 |
| dense | marlin_linear_kernel | uint4 | -1 | 1 | 2048 | 512 | 32x256x4 | 1.262307 |
| dense | marlin_linear_kernel | uint4 | -1 | 32 | 2048 | 512 | 32x256x4 | 1.257993 |
| dense | marlin_linear_kernel | uint4 | -1 | 64 | 2048 | 512 | 32x256x4 | 1.298315 |
| dense | marlin_linear_kernel | uint4 | -1 | 2048 | 2048 | 512 | 32x256x4 | 1.130721 |
| dense | marlin_linear_kernel | uint4 | -1 | 4096 | 2048 | 512 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 5120 | 2048 | 512 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 1 | 2048 | 1024 | 32x256x4 | 1.262307 |
| dense | marlin_linear_kernel | uint4 | -1 | 32 | 2048 | 1024 | 32x256x4 | 1.247969 |
| dense | marlin_linear_kernel | uint4 | -1 | 64 | 2048 | 1024 | 32x256x4 | 1.290282 |
| dense | marlin_linear_kernel | uint4 | -1 | 2048 | 2048 | 1024 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 4096 | 2048 | 1024 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 5120 | 2048 | 1024 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 1 | 2048 | 2048 | 32x256x4 | 1.262766 |
| dense | marlin_linear_kernel | uint4 | -1 | 32 | 2048 | 2048 | 32x256x4 | 1.277285 |
| dense | marlin_linear_kernel | uint4 | -1 | 64 | 2048 | 2048 | 32x256x4 | 1.299967 |
| dense | marlin_linear_kernel | uint4 | -1 | 2048 | 2048 | 2048 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 4096 | 2048 | 2048 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 5120 | 2048 | 2048 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 1 | 2048 | 4096 | 32x256x4 | 1.271207 |
| dense | marlin_linear_kernel | uint4 | -1 | 32 | 2048 | 4096 | 32x256x4 | 1.258301 |
| dense | marlin_linear_kernel | uint4 | -1 | 64 | 2048 | 4096 | 32x256x4 | 1.308350 |
| dense | marlin_linear_kernel | uint4 | -1 | 2048 | 2048 | 4096 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 4096 | 2048 | 4096 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 5120 | 2048 | 4096 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 1 | 4096 | 256 | 32x256x4 | 1.355021 |
| dense | marlin_linear_kernel | uint4 | -1 | 32 | 4096 | 256 | 32x256x4 | 1.362593 |
| dense | marlin_linear_kernel | uint4 | -1 | 64 | 4096 | 256 | 32x256x4 | 1.409366 |
| dense | marlin_linear_kernel | uint4 | -1 | 2048 | 4096 | 256 | 32x256x4 | 1.338354 |
| dense | marlin_linear_kernel | uint4 | -1 | 4096 | 4096 | 256 | 32x256x4 | 1.177977 |
| dense | marlin_linear_kernel | uint4 | -1 | 5120 | 4096 | 256 | 32x256x4 | 1.116280 |
| dense | marlin_linear_kernel | uint4 | -1 | 1 | 4096 | 512 | 32x256x4 | 1.355021 |
| dense | marlin_linear_kernel | uint4 | -1 | 32 | 4096 | 512 | 32x256x4 | 1.360472 |
| dense | marlin_linear_kernel | uint4 | -1 | 64 | 4096 | 512 | 32x256x4 | 1.412763 |
| dense | marlin_linear_kernel | uint4 | -1 | 2048 | 4096 | 512 | 32x256x4 | 1.174670 |
| dense | marlin_linear_kernel | uint4 | -1 | 4096 | 4096 | 512 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 5120 | 4096 | 512 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 1 | 4096 | 1024 | 32x256x4 | 1.369064 |
| dense | marlin_linear_kernel | uint4 | -1 | 32 | 4096 | 1024 | 32x256x4 | 1.356768 |
| dense | marlin_linear_kernel | uint4 | -1 | 64 | 4096 | 1024 | 32x256x4 | 1.409366 |
| dense | marlin_linear_kernel | uint4 | -1 | 2048 | 4096 | 1024 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 4096 | 4096 | 1024 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 5120 | 4096 | 1024 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 1 | 4096 | 2048 | 32x256x4 | 1.365845 |
| dense | marlin_linear_kernel | uint4 | -1 | 32 | 4096 | 2048 | 32x256x4 | 1.371265 |
| dense | marlin_linear_kernel | uint4 | -1 | 64 | 4096 | 2048 | 32x256x4 | 1.416672 |
| dense | marlin_linear_kernel | uint4 | -1 | 2048 | 4096 | 2048 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 4096 | 4096 | 2048 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 5120 | 4096 | 2048 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 1 | 4096 | 4096 | 32x256x4 | 1.335099 |
| dense | marlin_linear_kernel | uint4 | -1 | 32 | 4096 | 4096 | 32x256x4 | 1.347348 |
| dense | marlin_linear_kernel | uint4 | -1 | 64 | 4096 | 4096 | 32x256x4 | 1.423272 |
| dense | marlin_linear_kernel | uint4 | -1 | 2048 | 4096 | 4096 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 4096 | 4096 | 4096 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | -1 | 5120 | 4096 | 4096 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | 32 | 1 | 1024 | 256 | 32x256x4 | 1.188165 |
| dense | marlin_linear_kernel | uint4 | 32 | 32 | 1024 | 256 | 32x256x4 | 1.188165 |
| dense | marlin_linear_kernel | uint4 | 32 | 64 | 1024 | 256 | 32x256x4 | 1.198027 |
| dense | marlin_linear_kernel | uint4 | 32 | 2048 | 1024 | 256 | 32x256x4 | 1.196075 |
| dense | marlin_linear_kernel | uint4 | 32 | 4096 | 1024 | 256 | 64x256x8 | 1.168062 |
| dense | marlin_linear_kernel | uint4 | 32 | 5120 | 1024 | 256 | 64x256x8 | 1.180341 |
| dense | marlin_linear_kernel | uint4 | 32 | 1 | 1024 | 512 | 32x256x4 | 1.207987 |
| dense | marlin_linear_kernel | uint4 | 32 | 32 | 1024 | 512 | 32x256x4 | 1.188165 |
| dense | marlin_linear_kernel | uint4 | 32 | 64 | 1024 | 512 | 32x256x4 | 1.198027 |
| dense | marlin_linear_kernel | uint4 | 32 | 2048 | 1024 | 512 | 64x256x8 | 1.175433 |
| dense | marlin_linear_kernel | uint4 | 32 | 4096 | 1024 | 512 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | 32 | 5120 | 1024 | 512 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | 32 | 1 | 1024 | 1024 | 32x256x4 | 1.186213 |
| dense | marlin_linear_kernel | uint4 | 32 | 32 | 1024 | 1024 | 32x256x4 | 1.188165 |
| dense | marlin_linear_kernel | uint4 | 32 | 64 | 1024 | 1024 | 32x256x4 | 1.198027 |
| dense | marlin_linear_kernel | uint4 | 32 | 2048 | 1024 | 1024 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | 32 | 4096 | 1024 | 1024 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | 32 | 5120 | 1024 | 1024 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | 32 | 1 | 1024 | 2048 | 32x256x4 | 1.178302 |
| dense | marlin_linear_kernel | uint4 | 32 | 32 | 1024 | 2048 | 32x256x4 | 1.188165 |
| dense | marlin_linear_kernel | uint4 | 32 | 64 | 1024 | 2048 | 32x256x4 | 1.198027 |
| dense | marlin_linear_kernel | uint4 | 32 | 2048 | 1024 | 2048 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | 32 | 4096 | 1024 | 2048 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | 32 | 5120 | 1024 | 2048 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | 32 | 1 | 1024 | 4096 | 32x256x4 | 1.186213 |
| dense | marlin_linear_kernel | uint4 | 32 | 32 | 1024 | 4096 | 32x256x4 | 1.198027 |
| dense | marlin_linear_kernel | uint4 | 32 | 64 | 1024 | 4096 | 32x256x4 | 1.217850 |
| dense | marlin_linear_kernel | uint4 | 32 | 2048 | 1024 | 4096 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | 32 | 4096 | 1024 | 4096 | 64x256x4 | 1.000000 |
| dense | marlin_linear_kernel | uint4 | 32 | 5120 | 1024 | 4096 | 64x256x4 | 1.000000 |

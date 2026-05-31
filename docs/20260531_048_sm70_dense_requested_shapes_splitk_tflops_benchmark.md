# SM70 Dense 指定 Shape + 小 M Split-K TFLOPS Benchmark

## 摘要

本轮对 SM70 dense 指定矩阵执行了 serial benchmark，对比对象为：

- baseline：clean `6a6192bdd7df95afc76bb03453c61113cfa89af4`
- current：当前 `sm70-moe-u4-cutlass` dirty worktree

本轮在原指定矩阵中加入了 `M=8`，并对 `M in {1, 8, 16}` 额外覆盖
`split_k in {1, 2, 4, 8}`。其它 M 只跑 `split_k=1`。主指标为
`kernel_like marlin_tflops`，次指标为 `operator marlin_tflops`。

baseline/current benchmark 和 confirm 都严格串行执行；baseline 完整结束后才启动 current，
confirm 也是 baseline confirm 完整结束后才启动 current confirm。本轮全程没有使用
`nvidia-smi` 探测进度，只通过 `ps`、`tail benchmark.log`、`wc -l` 和 CSV 落盘状态观察。

screening 阶段共有 `36 / 864` 个 kernel-like same-config suspect。confirm 使用
`warmup_iters=20`、`iters=100`、`repeats=3` 复跑这些 suspect 后，仍有 `5`
个 confirmed TFLOPS regression。它们全部是 `split_k=1`、`CTA=128x256x8`
的中等 M 行，主要集中在 `M=128`，没有出现在本轮小 M split-K
`split_k=2/4/8` 中，也没有出现在 `M=5120` 大 M 峰值路径中。

## 代码状态

- baseline commit：`6a6192bdd7df95afc76bb03453c61113cfa89af4`
- baseline worktree：`/tmp/marlin_v100_qword_delta_baseline_6a6192b`
- current 分支：`sm70-moe-u4-cutlass`
- current HEAD：`6a6192bdd7df95afc76bb03453c61113cfa89af4`
- current worktree：dirty
- GPU：`Tesla V100-SXM2-32GB`
- CUDA capability：`(7, 0)`

current dirty tracked 文件：

```text
csrc/moe/marlin_moe_wna16/sm70_marlin_u4_gemm.cu
csrc/quantization/marlin/sm70_dense_common.cuh
csrc/quantization/marlin/sm70_dense_iterator_utils.cuh
csrc/quantization/marlin/sm70_marlin_fp8_gemm.cu
csrc/quantization/marlin/sm70_marlin_mxfp4_gemm.cu
csrc/quantization/marlin/sm70_marlin_nvfp4_gemm.cu
csrc/quantization/marlin/sm70_marlin_u4_gemm.cu
csrc/quantization/marlin/sm70_marlin_u4b8_gemm.cu
csrc/quantization/marlin/sm70_marlin_u8_gemm.cu
csrc/quantization/marlin/sm70_marlin_u8b128_gemm.cu
tests/test_marlin_dense.py
```

## 构建与导入

baseline build 成功，import smoke 成功：

```text
import torch, marlin_v100, marlin_v100._C, marlin_v100._moe_C
import ok
Tesla V100-SXM2-32GB
(7, 0)
```

current build 成功，import smoke 成功：

```text
import torch, marlin_v100, marlin_v100._C, marlin_v100._moe_C
import ok
Tesla V100-SXM2-32GB
(7, 0)
```

本轮按用户要求只执行 benchmark 对比，没有额外重跑 correctness pytest。

## Benchmark 矩阵

M：

```text
1, 8, 16, 64, 128, 256, 2048, 4096, 5120
```

K/N：

```text
4096x4096
4096x1024
4096x14336
14336x4096
```

quant / group size：

```text
uint4b8:  -1, 32
uint4:    -1, 32
uint8:    -1, 32
uint8b128:-1, 32
fp8:      -1, 128
nvfp4:    16
mxfp4:    32
```

split-K：

```text
M in {1, 8, 16}: split_k = 1, 2, 4, 8
M in {64, 128, 256, 2048, 4096, 5120}: split_k = 1
```

本轮 `split_k=1` 作为显式维度记录，执行时设置当前 quant 对应的
`SM70_MARLIN_*_SPLIT_K=1`，同时清理其它 dense split-K env。CTA env 全部 unset：

- baseline 记录的 expected CTA 使用 `128xauto_cta_nx8`
- current 记录的 expected CTA 使用当前 Auto CTA_M / Auto CTA_N 逻辑

本轮规格数量：

```text
864 benchmark configs
每个 config 输出 operator + kernel_like 两行
all_results.csv: 1 header + 1728 data rows = 1729 lines
```

timing：

```text
screening: warmup_iters=5, iters=20
confirm:   warmup_iters=20, iters=100, repeats=3
```

## 结果目录

baseline screening：

```text
/root/source/repos/marlin_v100/benchmarks/results/20260531_132022_dense_requested_shapes_splitk_baseline_6a6192b
```

current screening：

```text
/root/source/repos/marlin_v100/benchmarks/results/20260531_132022_dense_requested_shapes_splitk_current_dirty
```

baseline confirm：

```text
/root/source/repos/marlin_v100/benchmarks/results/20260531_132022_dense_requested_shapes_splitk_confirm_baseline_6a6192b
```

current confirm：

```text
/root/source/repos/marlin_v100/benchmarks/results/20260531_132022_dense_requested_shapes_splitk_confirm_current_dirty
```

关键文件：

```text
baseline all results:
/root/source/repos/marlin_v100/benchmarks/results/20260531_132022_dense_requested_shapes_splitk_baseline_6a6192b/all_results.csv

baseline log:
/root/source/repos/marlin_v100/benchmarks/results/20260531_132022_dense_requested_shapes_splitk_baseline_6a6192b/benchmark.log

current all results:
/root/source/repos/marlin_v100/benchmarks/results/20260531_132022_dense_requested_shapes_splitk_current_dirty/all_results.csv

current log:
/root/source/repos/marlin_v100/benchmarks/results/20260531_132022_dense_requested_shapes_splitk_current_dirty/benchmark.log

same-config compare:
/root/source/repos/marlin_v100/benchmarks/results/20260531_132022_dense_requested_shapes_splitk_current_dirty/compare_same_config_tflops.csv

screening summary:
/root/source/repos/marlin_v100/benchmarks/results/20260531_132022_dense_requested_shapes_splitk_current_dirty/compare_summary_tflops.txt

screening suspects:
/root/source/repos/marlin_v100/benchmarks/results/20260531_132022_dense_requested_shapes_splitk_current_dirty/suspect_configs.csv

confirm baseline copy:
/root/source/repos/marlin_v100/benchmarks/results/20260531_132022_dense_requested_shapes_splitk_current_dirty/confirm_baseline.csv

confirm current copy:
/root/source/repos/marlin_v100/benchmarks/results/20260531_132022_dense_requested_shapes_splitk_current_dirty/confirm_current.csv

confirm same-config compare:
/root/source/repos/marlin_v100/benchmarks/results/20260531_132022_dense_requested_shapes_splitk_current_dirty/confirm_same_config_tflops.csv

confirmed suspects:
/root/source/repos/marlin_v100/benchmarks/results/20260531_132022_dense_requested_shapes_splitk_current_dirty/confirmed_tflops_suspects.csv

confirm summary:
/root/source/repos/marlin_v100/benchmarks/results/20260531_132022_dense_requested_shapes_splitk_current_dirty/confirm_summary_tflops.txt
```

baseline/current 都没有生成 `skipped_or_failed_cases.csv`，说明本轮矩阵没有 skip/fail case。

## Baseline 原始结果摘要

baseline screening 完整跑完：

```text
benchmark.log: 2648 lines
all_results.csv: 1729 lines
elapsed_s=1310.140
```

baseline `kernel_like marlin_tflops` overall best：

```text
89.861895 TFLOPS
quant=uint8b128
group_size=-1
split_k=1
M=5120 K=14336 N=4096
CTA=128x256x8
kernel_like_us=6691.328
```

按 quant 汇总：

| quant | rows | median TFLOPS | best TFLOPS | best config |
| --- | ---: | ---: | ---: | --- |
| fp8 | 144 | 3.174754 | 89.125379 | fp8 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| mxfp4 | 72 | 2.651834 | 68.307168 | mxfp4 gs=32 sk=1 M=5120 K=4096 N=14336 CTA=128x256x8 |
| nvfp4 | 72 | 3.236436 | 87.122040 | nvfp4 gs=16 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| uint4 | 144 | 2.959690 | 89.752014 | uint4 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| uint4b8 | 144 | 3.253724 | 88.848930 | uint4b8 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| uint8 | 144 | 3.063871 | 89.044288 | uint8 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| uint8b128 | 144 | 3.181640 | 89.861895 | uint8b128 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |

按 K/N 汇总：

| K | N | rows | median TFLOPS | best TFLOPS | best config |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4096 | 1024 | 216 | 0.604031 | 81.920000 | uint4b8 gs=32 sk=1 M=5120 K=4096 N=1024 CTA=128x256x8 |
| 4096 | 4096 | 216 | 2.084242 | 86.838590 | uint8b128 gs=-1 sk=1 M=5120 K=4096 N=4096 CTA=128x256x8 |
| 4096 | 14336 | 216 | 5.948178 | 88.936395 | uint8 gs=-1 sk=1 M=5120 K=4096 N=14336 CTA=128x256x8 |
| 14336 | 4096 | 216 | 3.608672 | 89.861895 | uint8b128 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |

按 M 汇总：

| M | rows | median TFLOPS | best TFLOPS | best config |
| ---: | ---: | ---: | ---: | --- |
| 1 | 192 | 0.221405 | 0.474899 | uint4b8 gs=32 sk=4 M=1 K=4096 N=14336 CTA=128x256x8 |
| 8 | 192 | 1.750968 | 3.729691 | uint4b8 gs=32 sk=4 M=8 K=4096 N=14336 CTA=128x256x8 |
| 16 | 192 | 3.432513 | 7.126245 | nvfp4 gs=16 sk=4 M=16 K=14336 N=4096 CTA=128x256x8 |
| 64 | 48 | 7.768189 | 27.286363 | uint4b8 gs=32 sk=1 M=64 K=4096 N=14336 CTA=128x256x8 |
| 128 | 48 | 15.311809 | 53.382051 | uint4b8 gs=32 sk=1 M=128 K=4096 N=14336 CTA=128x256x8 |
| 256 | 48 | 30.567808 | 60.102617 | uint4 gs=-1 sk=1 M=256 K=4096 N=14336 CTA=128x256x8 |
| 2048 | 48 | 71.041962 | 85.691730 | uint4 gs=-1 sk=1 M=2048 K=4096 N=14336 CTA=128x256x8 |
| 4096 | 48 | 80.418058 | 87.170544 | uint8 gs=-1 sk=1 M=4096 K=4096 N=14336 CTA=128x256x8 |
| 5120 | 48 | 84.690918 | 89.861895 | uint8b128 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |

按 split-K 汇总：

| split_k | rows | median TFLOPS | best TFLOPS | best config |
| ---: | ---: | ---: | ---: | --- |
| 1 | 432 | 15.311809 | 89.861895 | uint8b128 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| 2 | 144 | 0.886365 | 6.305869 | uint4b8 gs=32 sk=2 M=16 K=4096 N=14336 CTA=128x256x8 |
| 4 | 144 | 1.264649 | 7.126245 | nvfp4 gs=16 sk=4 M=16 K=14336 N=4096 CTA=128x256x8 |
| 8 | 144 | 1.353423 | 6.821591 | uint4b8 gs=32 sk=8 M=16 K=14336 N=4096 CTA=128x256x8 |

## Current 原始结果摘要

current screening 完整跑完：

```text
benchmark.log: 2648 lines
all_results.csv: 1729 lines
elapsed_s=1294.315
```

current `kernel_like marlin_tflops` overall best：

```text
90.478055 TFLOPS
quant=uint4
group_size=-1
split_k=1
M=5120 K=14336 N=4096
CTA=128x256x8
kernel_like_us=6645.760
```

按 quant 汇总：

| quant | rows | median TFLOPS | best TFLOPS | best config |
| --- | ---: | ---: | ---: | --- |
| fp8 | 144 | 4.055390 | 88.520773 | fp8 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| mxfp4 | 72 | 3.391916 | 69.301770 | mxfp4 gs=32 sk=1 M=4096 K=4096 N=14336 CTA=128x256x8 |
| nvfp4 | 72 | 4.235415 | 86.467763 | nvfp4 gs=16 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| uint4 | 144 | 4.427098 | 90.478055 | uint4 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| uint4b8 | 144 | 4.424450 | 88.848930 | uint4b8 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| uint8 | 144 | 4.198424 | 89.423978 | uint8 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| uint8b128 | 144 | 4.061740 | 89.044288 | uint8b128 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |

按 K/N 汇总：

| K | N | rows | median TFLOPS | best TFLOPS | best config |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4096 | 1024 | 216 | 0.833697 | 82.241258 | uint4b8 gs=32 sk=1 M=5120 K=4096 N=1024 CTA=128x256x8 |
| 4096 | 4096 | 216 | 2.899805 | 87.655255 | uint8b128 gs=-1 sk=1 M=5120 K=4096 N=4096 CTA=128x256x8 |
| 4096 | 14336 | 216 | 8.655890 | 88.795188 | uint8b128 gs=-1 sk=1 M=5120 K=4096 N=14336 CTA=128x256x8 |
| 14336 | 4096 | 216 | 5.628859 | 90.478055 | uint4 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |

按 M 汇总：

| M | rows | median TFLOPS | best TFLOPS | best config |
| ---: | ---: | ---: | ---: | --- |
| 1 | 192 | 0.290638 | 0.749595 | uint4 gs=-1 sk=8 M=1 K=4096 N=14336 CTA=32x256x4 |
| 8 | 192 | 2.250204 | 6.036211 | uint4 gs=-1 sk=8 M=8 K=14336 N=4096 CTA=32x256x4 |
| 16 | 192 | 4.262504 | 11.433072 | uint4 gs=-1 sk=8 M=16 K=14336 N=4096 CTA=32x256x4 |
| 64 | 48 | 10.188243 | 36.157793 | uint4b8 gs=32 sk=1 M=64 K=4096 N=14336 CTA=64x256x8 |
| 128 | 48 | 15.309514 | 53.382051 | uint4b8 gs=32 sk=1 M=128 K=4096 N=14336 CTA=128x256x8 |
| 256 | 48 | 30.562126 | 60.164198 | uint4b8 gs=32 sk=1 M=256 K=4096 N=14336 CTA=128x256x8 |
| 2048 | 48 | 70.715425 | 85.942560 | uint8b128 gs=-1 sk=1 M=2048 K=4096 N=14336 CTA=128x256x8 |
| 4096 | 48 | 80.483548 | 87.138205 | uint8b128 gs=-1 sk=1 M=4096 K=4096 N=14336 CTA=128x256x8 |
| 5120 | 48 | 84.937194 | 90.478055 | uint4 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |

按 split-K 汇总：

| split_k | rows | median TFLOPS | best TFLOPS | best config |
| ---: | ---: | ---: | ---: | --- |
| 1 | 432 | 15.309514 | 90.478055 | uint4 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| 2 | 144 | 1.286158 | 11.054265 | uint4 gs=-1 sk=2 M=16 K=4096 N=14336 CTA=32x256x4 |
| 4 | 144 | 1.668335 | 10.309034 | uint4 gs=-1 sk=4 M=16 K=4096 N=14336 CTA=32x256x4 |
| 8 | 144 | 1.793446 | 11.433072 | uint4 gs=-1 sk=8 M=16 K=14336 N=4096 CTA=32x256x4 |

operator 次指标摘要：

| phase | rows | median operator TFLOPS | best operator TFLOPS | best config |
| --- | ---: | ---: | ---: | --- |
| baseline | 864 | 2.988612 | 90.533851 | uint4 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| current | 864 | 4.009866 | 90.596706 | uint8b128 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |

## Current 大 M TFLOPS 汇总

下面汇总 `M in {2048, 4096, 5120}` 的 current `kernel_like marlin_tflops`。
完整 per-case 原始结果见 current `all_results.csv`。

按 M/K/N 汇总：

| M | K | N | rows | median TFLOPS | best TFLOPS | best config |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2048 | 4096 | 1024 | 12 | 59.179075 | 60.787013 | uint8b128 gs=-1 sk=1 M=2048 K=4096 N=1024 CTA=128x256x8 |
| 2048 | 4096 | 4096 | 12 | 69.978265 | 72.471775 | uint4b8 gs=32 sk=1 M=2048 K=4096 N=4096 CTA=128x256x8 |
| 2048 | 4096 | 14336 | 12 | 83.521090 | 85.942560 | uint8b128 gs=-1 sk=1 M=2048 K=4096 N=14336 CTA=128x256x8 |
| 2048 | 14336 | 4096 | 12 | 73.429993 | 76.558350 | uint8b128 gs=-1 sk=1 M=2048 K=14336 N=4096 CTA=128x256x8 |
| 4096 | 4096 | 1024 | 12 | 65.601630 | 67.650065 | uint8 gs=-1 sk=1 M=4096 K=4096 N=1024 CTA=128x256x8 |
| 4096 | 4096 | 4096 | 12 | 80.249877 | 81.640956 | uint8b128 gs=-1 sk=1 M=4096 K=4096 N=4096 CTA=128x256x8 |
| 4096 | 4096 | 14336 | 12 | 84.317093 | 87.138205 | uint8b128 gs=-1 sk=1 M=4096 K=4096 N=14336 CTA=128x256x8 |
| 4096 | 14336 | 4096 | 12 | 83.303957 | 86.068534 | uint8b128 gs=-1 sk=1 M=4096 K=14336 N=4096 CTA=128x256x8 |
| 5120 | 4096 | 1024 | 12 | 80.428353 | 82.241258 | uint4b8 gs=32 sk=1 M=5120 K=4096 N=1024 CTA=128x256x8 |
| 5120 | 4096 | 4096 | 12 | 84.937194 | 87.655255 | uint8b128 gs=-1 sk=1 M=5120 K=4096 N=4096 CTA=128x256x8 |
| 5120 | 4096 | 14336 | 12 | 86.344173 | 88.795188 | uint8b128 gs=-1 sk=1 M=5120 K=4096 N=14336 CTA=128x256x8 |
| 5120 | 14336 | 4096 | 12 | 87.494268 | 90.478055 | uint4 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |

按 quant/group 汇总：

| quant | group_size | rows | median TFLOPS | best TFLOPS | best config |
| --- | ---: | ---: | ---: | ---: | --- |
| fp8 | -1 | 12 | 82.269198 | 88.520773 | fp8 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| fp8 | 128 | 12 | 78.141315 | 84.837472 | fp8 gs=128 sk=1 M=5120 K=4096 N=14336 CTA=128x256x8 |
| mxfp4 | 32 | 12 | 63.522572 | 69.301770 | mxfp4 gs=32 sk=1 M=4096 K=4096 N=14336 CTA=128x256x8 |
| nvfp4 | 16 | 12 | 81.232448 | 86.467763 | nvfp4 gs=16 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| uint4 | -1 | 12 | 83.567955 | 90.478055 | uint4 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| uint4 | 32 | 12 | 80.144334 | 85.299619 | uint4 gs=32 sk=1 M=5120 K=4096 N=14336 CTA=128x256x8 |
| uint4b8 | -1 | 12 | 82.653977 | 88.848930 | uint4b8 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| uint4b8 | 32 | 12 | 83.071160 | 88.634347 | uint4b8 gs=32 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| uint8 | -1 | 12 | 82.664890 | 89.423978 | uint8 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| uint8 | 32 | 12 | 78.402486 | 84.168647 | uint8 gs=32 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| uint8b128 | -1 | 12 | 83.791758 | 89.044288 | uint8b128 gs=-1 sk=1 M=5120 K=14336 N=4096 CTA=128x256x8 |
| uint8b128 | 32 | 12 | 80.399124 | 86.163252 | uint8b128 gs=32 sk=1 M=5120 K=4096 N=14336 CTA=128x256x8 |

大 M 结论：

- `M=5120,K=14336,N=4096` 是本轮 current 峰值形状，最高 `90.478055 TFLOPS`。
- `M=5120,N in {4096,14336}` 的 median 都在 `84+ TFLOPS`。
- `M=2048/4096/5120,N=1024` 明显低于大 N，但仍随 M 增大从约 `59` 提升到约 `80 TFLOPS`。

## Current 小 M Split-K 汇总

下面汇总 `M in {1, 8, 16}` 且 `split_k in {1, 2, 4, 8}` 的 current
`kernel_like marlin_tflops`。完整 per-case 原始结果见 current `all_results.csv`。

按 M/split_k 汇总：

| M | split_k | rows | median TFLOPS | best TFLOPS | best config |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 48 | 0.174342 | 0.588144 | uint4 gs=-1 sk=1 M=1 K=4096 N=14336 CTA=32x256x4 |
| 1 | 2 | 48 | 0.243386 | 0.699317 | uint4b8 gs=-1 sk=2 M=1 K=4096 N=14336 CTA=32x256x4 |
| 1 | 4 | 48 | 0.364282 | 0.692979 | uint4 gs=-1 sk=4 M=1 K=4096 N=14336 CTA=32x256x4 |
| 1 | 8 | 48 | 0.436591 | 0.749595 | uint4 gs=-1 sk=8 M=1 K=4096 N=14336 CTA=32x256x4 |
| 8 | 1 | 48 | 1.398350 | 4.705149 | uint4 gs=-1 sk=1 M=8 K=4096 N=14336 CTA=32x256x4 |
| 8 | 2 | 48 | 2.005581 | 5.494036 | uint4 gs=-1 sk=2 M=8 K=4096 N=14336 CTA=32x256x4 |
| 8 | 4 | 48 | 2.969313 | 5.213091 | uint4b8 gs=-1 sk=4 M=8 K=4096 N=14336 CTA=32x256x4 |
| 8 | 8 | 48 | 3.479136 | 6.036211 | uint4 gs=-1 sk=8 M=8 K=14336 N=4096 CTA=32x256x4 |
| 16 | 1 | 48 | 2.774321 | 9.734790 | uint4b8 gs=-1 sk=1 M=16 K=4096 N=14336 CTA=32x256x4 |
| 16 | 2 | 48 | 4.065545 | 11.054265 | uint4 gs=-1 sk=2 M=16 K=4096 N=14336 CTA=32x256x4 |
| 16 | 4 | 48 | 5.921172 | 10.309034 | uint4 gs=-1 sk=4 M=16 K=4096 N=14336 CTA=32x256x4 |
| 16 | 8 | 48 | 6.452759 | 11.433072 | uint4 gs=-1 sk=8 M=16 K=14336 N=4096 CTA=32x256x4 |

按 quant/split_k 汇总：

| quant | split_k | rows | median TFLOPS | best TFLOPS | best config |
| --- | ---: | ---: | ---: | ---: | --- |
| fp8 | 1 | 24 | 0.933534 | 8.515118 | fp8 gs=-1 sk=1 M=16 K=4096 N=14336 CTA=32x256x4 |
| fp8 | 2 | 24 | 1.277585 | 9.458804 | fp8 gs=128 sk=2 M=16 K=4096 N=14336 CTA=32x256x4 |
| fp8 | 4 | 24 | 1.636036 | 9.039448 | fp8 gs=-1 sk=4 M=16 K=14336 N=4096 CTA=32x256x4 |
| fp8 | 8 | 24 | 1.775606 | 10.138166 | fp8 gs=-1 sk=8 M=16 K=14336 N=4096 CTA=32x256x4 |
| mxfp4 | 1 | 12 | 0.788099 | 6.796326 | mxfp4 gs=32 sk=1 M=16 K=4096 N=14336 CTA=32x256x4 |
| mxfp4 | 2 | 12 | 1.162185 | 8.284461 | mxfp4 gs=32 sk=2 M=16 K=4096 N=14336 CTA=32x256x4 |
| mxfp4 | 4 | 12 | 1.573683 | 7.926600 | mxfp4 gs=32 sk=4 M=16 K=4096 N=14336 CTA=32x256x4 |
| mxfp4 | 8 | 12 | 1.778802 | 9.084198 | mxfp4 gs=32 sk=8 M=16 K=14336 N=4096 CTA=32x256x4 |
| nvfp4 | 1 | 12 | 0.984129 | 8.574804 | nvfp4 gs=16 sk=1 M=16 K=4096 N=14336 CTA=32x256x4 |
| nvfp4 | 2 | 12 | 1.353165 | 9.458804 | nvfp4 gs=16 sk=2 M=16 K=4096 N=14336 CTA=32x256x4 |
| nvfp4 | 4 | 12 | 1.734932 | 9.039448 | nvfp4 gs=16 sk=4 M=16 K=14336 N=4096 CTA=32x256x4 |
| nvfp4 | 8 | 12 | 1.868839 | 10.426182 | nvfp4 gs=16 sk=8 M=16 K=14336 N=4096 CTA=32x256x4 |
| uint4 | 1 | 24 | 1.041038 | 9.458804 | uint4 gs=-1 sk=1 M=16 K=4096 N=14336 CTA=32x256x4 |
| uint4 | 2 | 24 | 1.388153 | 11.054265 | uint4 gs=-1 sk=2 M=16 K=4096 N=14336 CTA=32x256x4 |
| uint4 | 4 | 24 | 1.722421 | 10.309034 | uint4 gs=-1 sk=4 M=16 K=4096 N=14336 CTA=32x256x4 |
| uint4 | 8 | 24 | 1.832266 | 11.433072 | uint4 gs=-1 sk=8 M=16 K=14336 N=4096 CTA=32x256x4 |
| uint4b8 | 1 | 24 | 1.074912 | 9.734790 | uint4b8 gs=-1 sk=1 M=16 K=4096 N=14336 CTA=32x256x4 |
| uint4b8 | 2 | 24 | 1.414138 | 10.988071 | uint4b8 gs=-1 sk=2 M=16 K=4096 N=14336 CTA=32x256x4 |
| uint4b8 | 4 | 24 | 1.756545 | 10.251442 | uint4b8 gs=-1 sk=4 M=16 K=4096 N=14336 CTA=32x256x4 |
| uint4b8 | 8 | 24 | 1.840435 | 11.327210 | uint4b8 gs=-1 sk=8 M=16 K=14336 N=4096 CTA=32x256x4 |
| uint8 | 1 | 24 | 0.982562 | 8.886237 | uint8 gs=-1 sk=1 M=16 K=4096 N=14336 CTA=32x256x4 |
| uint8 | 2 | 24 | 1.332383 | 10.194489 | uint8 gs=-1 sk=2 M=16 K=4096 N=14336 CTA=32x256x4 |
| uint8 | 4 | 24 | 1.681185 | 9.410297 | uint8 gs=-1 sk=4 M=16 K=4096 N=14336 CTA=32x256x4 |
| uint8 | 8 | 24 | 1.785291 | 10.731040 | uint8 gs=-1 sk=8 M=16 K=14336 N=4096 CTA=32x256x4 |
| uint8b128 | 1 | 24 | 0.923306 | 8.779942 | uint8b128 gs=-1 sk=1 M=16 K=4096 N=14336 CTA=32x256x4 |
| uint8b128 | 2 | 24 | 1.271473 | 10.110237 | uint8b128 gs=-1 sk=2 M=16 K=4096 N=14336 CTA=32x256x4 |
| uint8b128 | 4 | 24 | 1.627845 | 9.314761 | uint8b128 gs=-1 sk=4 M=16 K=4096 N=14336 CTA=32x256x4 |
| uint8b128 | 8 | 24 | 1.756455 | 10.794165 | uint8b128 gs=-1 sk=8 M=16 K=14336 N=4096 CTA=32x256x4 |

小 M split-K 结论：

- `M=1/8/16` 的 median TFLOPS 随 split-K 增大整体上升。
- `M=16` 的 current best 为 `11.433072 TFLOPS`，来自 `uint4 gs=-1 split_k=8 K=14336 N=4096`。
- screening 中没有 `split_k=2/4/8` suspect；confirm 后的 confirmed suspects 也全部是 `split_k=1`。

## Screening 对比

screening same-config 对比：

```text
common_kernel_like_rows=864
suspect_rows=36
median_ratio=1.308733
worst_ratio=0.893417
best_ratio=2.169591
```

按 quant 汇总：

| quant | rows | suspects | median ratio | worst ratio | best ratio | median current TFLOPS | best current TFLOPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp8 | 144 | 8 | 1.264103 | 0.916118 | 1.694737 | 4.055390 | 88.520773 |
| mxfp4 | 72 | 0 | 1.342082 | 0.972510 | 1.703349 | 3.391916 | 69.301770 |
| nvfp4 | 72 | 1 | 1.290476 | 0.958478 | 1.630769 | 4.235415 | 86.467763 |
| uint4 | 144 | 9 | 1.378833 | 0.927336 | 2.169591 | 4.427098 | 90.478055 |
| uint4b8 | 144 | 6 | 1.348142 | 0.932476 | 1.969512 | 4.424450 | 88.848930 |
| uint8 | 144 | 6 | 1.302883 | 0.893417 | 1.796610 | 4.198424 | 89.423978 |
| uint8b128 | 144 | 6 | 1.275604 | 0.929825 | 1.810734 | 4.061740 | 89.044288 |

按 quant/group 汇总：

| quant | group_size | rows | suspects | median ratio | worst ratio | best ratio | median current TFLOPS | best current TFLOPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp8 | -1 | 72 | 1 | 1.319802 | 0.916118 | 1.682081 | 4.103024 | 88.520773 |
| fp8 | 128 | 72 | 7 | 1.225108 | 0.931034 | 1.694737 | 3.927959 | 84.837472 |
| mxfp4 | 32 | 72 | 0 | 1.342082 | 0.972510 | 1.703349 | 3.391916 | 69.301770 |
| nvfp4 | 16 | 72 | 1 | 1.290476 | 0.958478 | 1.630769 | 4.235415 | 86.467763 |
| uint4 | -1 | 72 | 6 | 1.440573 | 0.927336 | 1.957317 | 4.541418 | 90.478055 |
| uint4 | 32 | 72 | 3 | 1.341612 | 0.945017 | 2.169591 | 4.336790 | 85.299619 |
| uint4b8 | -1 | 72 | 3 | 1.416995 | 0.932476 | 1.969512 | 4.424450 | 88.848930 |
| uint4b8 | 32 | 72 | 3 | 1.305832 | 0.933566 | 1.831884 | 4.422348 | 88.634347 |
| uint8 | -1 | 72 | 3 | 1.318416 | 0.932343 | 1.796610 | 4.264982 | 89.423978 |
| uint8 | 32 | 72 | 3 | 1.260378 | 0.893417 | 1.745665 | 4.139927 | 84.168647 |
| uint8b128 | -1 | 72 | 4 | 1.309265 | 0.929825 | 1.810734 | 4.219639 | 89.044288 |
| uint8b128 | 32 | 72 | 2 | 1.250000 | 0.935898 | 1.667606 | 3.895914 | 86.163252 |

按 K/N 汇总：

| K | N | rows | suspects | median ratio | worst ratio | best ratio | median current TFLOPS | best current TFLOPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14336 | 4096 | 216 | 0 | 1.388583 | 0.980170 | 1.907975 | 5.628859 | 90.478055 |
| 4096 | 1024 | 216 | 13 | 1.235119 | 0.927336 | 1.908497 | 0.833697 | 82.241258 |
| 4096 | 14336 | 216 | 8 | 1.364831 | 0.955017 | 2.169591 | 8.655890 | 88.795188 |
| 4096 | 4096 | 216 | 15 | 1.256184 | 0.893417 | 1.654054 | 2.899805 | 87.655255 |

按 M 汇总：

| M | rows | suspects | median ratio | worst ratio | best ratio | median current TFLOPS | best current TFLOPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 192 | 0 | 1.419282 | 1.060000 | 2.088572 | 0.290638 | 0.749595 |
| 8 | 192 | 0 | 1.396316 | 1.079208 | 2.169591 | 2.250204 | 6.036211 |
| 16 | 192 | 0 | 1.370509 | 1.076923 | 2.002817 | 4.262504 | 11.433072 |
| 64 | 48 | 1 | 1.318812 | 0.942761 | 1.520833 | 10.188243 | 36.157793 |
| 128 | 48 | 20 | 0.999050 | 0.893417 | 1.097786 | 15.309514 | 53.382051 |
| 256 | 48 | 10 | 1.000930 | 0.898119 | 1.091912 | 30.562126 | 60.164198 |
| 2048 | 48 | 3 | 1.000000 | 0.932432 | 1.059859 | 70.715425 | 85.942560 |
| 4096 | 48 | 2 | 1.000297 | 0.940952 | 1.045279 | 80.483548 | 87.138205 |
| 5120 | 48 | 0 | 1.001675 | 0.972510 | 1.024870 | 84.937194 | 90.478055 |

按 split-K 汇总：

| split_k | rows | suspects | median ratio | worst ratio | best ratio | median current TFLOPS | best current TFLOPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 432 | 36 | 1.017728 | 0.893417 | 1.908497 | 15.309514 | 90.478055 |
| 2 | 144 | 0 | 1.424496 | 1.114458 | 2.169591 | 1.286158 | 11.054265 |
| 4 | 144 | 0 | 1.298504 | 1.076923 | 1.719101 | 1.668335 | 10.309034 |
| 8 | 144 | 0 | 1.447200 | 1.060000 | 2.021605 | 1.793446 | 11.433072 |

## Confirm 对比

confirm 复跑 screening suspects，共 `36` 个 configs。baseline confirm 和 current confirm
都输出 `217` 行：

```text
1 header + 36 configs * 2 metrics * 3 repeats = 217 lines
```

confirm same-config 对比：

```text
common_kernel_like_rows=36
confirmed_suspect_rows=5
median_ratio=1.003237
worst_ratio=0.931148
best_ratio=1.066890
```

按 quant 汇总：

| quant | rows | confirmed suspects | median ratio | worst ratio | best ratio | median current TFLOPS | best current TFLOPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp8 | 8 | 1 | 1.003260 | 0.931148 | 1.006897 | 21.824876 | 63.310247 |
| nvfp4 | 1 | 1 | 0.938710 | 0.938710 | 0.938710 | 47.355045 | 47.355045 |
| uint4 | 9 | 0 | 1.003717 | 1.000000 | 1.027027 | 29.959313 | 65.408249 |
| uint4b8 | 6 | 2 | 1.002641 | 0.939490 | 1.003802 | 21.769745 | 61.119185 |
| uint8 | 6 | 1 | 1.002530 | 0.962963 | 1.056604 | 21.777975 | 50.796071 |
| uint8b128 | 6 | 0 | 1.004094 | 1.000000 | 1.066890 | 22.804573 | 49.097203 |

按 quant + CTA 汇总：

| quant | baseline CTA | current CTA | rows | confirmed suspects | median ratio | worst ratio | best ratio | median current TFLOPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp8 | 128x256x8 | 128x256x8 | 8 | 1 | 1.003260 | 0.931148 | 1.006897 | 21.824876 |
| nvfp4 | 128x256x8 | 128x256x8 | 1 | 1 | 0.938710 | 0.938710 | 0.938710 | 47.355045 |
| uint4 | 128x256x8 | 128x256x8 | 9 | 0 | 1.003717 | 1.000000 | 1.027027 | 29.959313 |
| uint4b8 | 128x256x8 | 128x256x8 | 6 | 2 | 1.002641 | 0.939490 | 1.003802 | 21.769745 |
| uint8 | 128x256x8 | 128x256x8 | 5 | 1 | 1.001761 | 0.962963 | 1.006920 | 29.433714 |
| uint8 | 128x256x8 | 64x256x8 | 1 | 0 | 1.056604 | 1.056604 | 1.056604 | 7.913781 |
| uint8b128 | 128x256x8 | 128x256x8 | 6 | 0 | 1.004094 | 1.000000 | 1.066890 | 22.804573 |

confirmed exact configs：

| quant | group_size | split_k | M | K | N | baseline CTA | current CTA | baseline median TFLOPS | current median TFLOPS | ratio | delta |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| fp8 | 128 | 1 | 128 | 4096 | 1024 | 128x256x8 | 128x256x8 | 3.692169 | 3.437954 | 0.931148 | -0.254215 |
| nvfp4 | 16 | 1 | 128 | 4096 | 14336 | 128x256x8 | 128x256x8 | 50.446955 | 47.355045 | 0.938710 | -3.091910 |
| uint4b8 | -1 | 1 | 128 | 4096 | 14336 | 128x256x8 | 128x256x8 | 49.762928 | 46.751796 | 0.939490 | -3.011132 |
| uint4b8 | -1 | 1 | 256 | 4096 | 4096 | 128x256x8 | 128x256x8 | 30.229218 | 29.026326 | 0.960208 | -1.202891 |
| uint8 | 32 | 1 | 128 | 4096 | 4096 | 128x256x8 | 128x256x8 | 14.665399 | 14.122236 | 0.962963 | -0.543163 |

confirmed 结论：

- 本轮不是 `no confirmed TFLOPS regression for requested dense shape and split-K matrix`。
- confirmed suspect 有 `5` 个，全都在 `split_k=1`。
- confirmed suspect 全都在 `CTA=128x256x8 -> 128x256x8`，因此不是 Auto CTA_M 几何变化导致。
- confirmed suspect 主要集中在 `M=128`，另有一个 `M=256`。
- `M in {1, 8, 16}` 的 split-K `2/4/8` 没有 confirmed regression。
- `M=5120` 没有 screening suspect，也没有 confirmed regression；大 M 峰值路径保持健康。
- `uint4` 和 `uint8b128` 的 screening suspects 在 confirm 后全部消失。

## 解读

本轮对比有两个很清楚的信号：

1. 当前 Auto CTA_M 对小 M split-K 很有帮助。`M=1/8/16` 的 ratio 全部大于 1，
   且 split-K `2/4/8` 没有 suspect。current 的小 M best CTA 变成 `32x256x4`，
   相比 baseline 固定 `128x256x8` 更适合小 M。

2. 大 M 大 N 没有整体回退。`M=5120` 没有 suspect，current overall best 达到
   `90.478055 TFLOPS`，略高于 baseline best `89.861895 TFLOPS`。

剩下的 confirmed regression 是局部 same-CTA 中等 M 问题：

- `fp8 group_size=128, M=128, N=1024`
- `nvfp4 group_size=16, M=128, N=14336`
- `uint4b8 group_size=-1, M=128, N=14336`
- `uint4b8 group_size=-1, M=256, N=4096`
- `uint8 group_size=32, M=128, N=4096`

这些行没有 CTA 几何变化，所以更像是当前 qweight load helper / delta 改动或局部 codegen
在中等 M 上的残余影响，而不是 Auto CTA_M 选择本身。下一步如果要继续收敛，应优先对这
`5` 个 exact config 做 SASS/ptxas 对照，或者针对 helper 抽象做更小粒度 A/B。

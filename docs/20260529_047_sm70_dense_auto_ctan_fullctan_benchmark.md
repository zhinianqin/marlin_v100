# SM70 Dense Auto CTA_N + FullCtaN Compact Benchmark

## 摘要

本轮在 `sm70-moe-u4-cutlass` dirty worktree 上实现并验证了 SM70 dense
Auto CTA_N + FullCtaN compact qweight layout/repack 方案。

功能侧已经通过当前 dense/repack 测试；性能侧与
`fix/sm70-dense-fullmacron-fastpath` 分支做了 serial no-split TFLOPS
benchmark 对比。整体 `kernel_like marlin_tflops` median ratio 为
`1.0009`，但 confirm 后仍存在 `25 / 85` 个局部确认回归，主要集中在
`uint8b128 group_size=32` 的低 N (`N=64/128`) CTA 组合。`mxfp4` 和
`nvfp4` confirm 后没有确认回归。

本轮全程未使用 `nvidia-smi` 探测进度；进度只通过 `ps`、`tail`、`wc`
和 CSV 落盘状态观察。baseline/current benchmark 严格串行执行。

## 代码状态

- Current 分支：`sm70-moe-u4-cutlass`
- Current HEAD：`add9dad`
- Current worktree：dirty
- Baseline 分支/worktree：`/tmp/marlin_v100_fix_fullmacron_fastpath_bench`
- Baseline HEAD：`c0b10e2`
- Baseline 对比目标：`fix/sm70-dense-fullmacron-fastpath`

Benchmark 时的 dirty tracked 文件：

```text
csrc/quantization/marlin/awq_marlin_repack.cu
csrc/quantization/marlin/gptq_marlin_repack.cu
csrc/quantization/marlin/sm70_dense_common.cuh
csrc/quantization/marlin/sm70_dense_iterator_utils.cuh
csrc/quantization/marlin/sm70_marlin_fp8_gemm.cu
csrc/quantization/marlin/sm70_marlin_mxfp4_gemm.cu
csrc/quantization/marlin/sm70_marlin_nvfp4_gemm.cu
csrc/quantization/marlin/sm70_marlin_u4_gemm.cu
csrc/quantization/marlin/sm70_marlin_u4b8_gemm.cu
csrc/quantization/marlin/sm70_marlin_u8_gemm.cu
csrc/quantization/marlin/sm70_marlin_u8b128_gemm.cu
python/marlin_v100/quant_utils.py
tests/test_marlin_dense.py
tests/test_marlin_helpers.py
```

## 实现范围

已实现行为：

- Dense `CTA_N` 按 `256 -> 128 -> 64` 顺序自动选择，取能整除 `size_n`
  的最大值。
- `SM70_MARLIN_*_CTA` 仍接受 `CTA_MxCTA_NxWarps`；如果 env 中的
  `CTA_N` 和自动选择值不一致，kernel 会拒绝 launch。
- Repack 和 dense B iterator 使用 FullCtaN compact layout：
  `GroupTiles = CTA_N / 64`.
- qweight 保持 compact 形状；本方案不把 N padding 到 256。
- 已迁移 dense kernel 的 iterator 热路径使用编译期 CTA_N layout helper，
  不再使用 runtime `subtile_count`。
- 本轮 benchmark 任务没有修改 MoE。

## 验证

Benchmark 前已完成 current worktree 的 build/import 和 correctness 验证：

```text
./build.sh
import torch, marlin_v100, marlin_v100._C, marlin_v100._moe_C
device=Tesla V100-SXM2-32GB
capability=(7, 0)
```

已通过测试：

```text
tests/test_marlin_helpers.py -k "repack or marlin_weight_pack or marlin_unpack"
52 passed, 54 deselected

tests/test_marlin_dense.py -k "partial_n or residue_n or cta_geometry"
142 passed, 235 deselected

tests/test_marlin_dense.py -k "split_k"
88 passed, 289 deselected

tests/test_marlin_dense.py
377 passed

pytest --collect-only -q
642 tests collected

git diff --check
passed
```

## Benchmark 矩阵

Benchmark 模式：

- split-K unset，只测 no-split path。
- 主指标：`kernel_like marlin_tflops`。
- 次指标：`operator marlin_tflops`。
- Screening timing：`warmup_iters=3`, `iters=10`。
- Confirm timing：`warmup_iters=20`, `iters=100`, `repeats=3`。
- Confirm 对比使用 3 次 repeat 的 median TFLOPS。

Shape 和 quant 矩阵：

- Quant 类型：`uint4b8`, `uint4`, `uint8`, `uint8b128`, `fp8`, `nvfp4`, `mxfp4`.
- M: `16`, `128`, `5120`.
- K: `4096`, `16384`.
- N: `64`, `128`, `256`, `4096`, `12288`.
- Group size 覆盖：
  - `uint4b8`, `uint4`, `uint8`, `uint8b128`: `-1`, `32`
  - `fp8`: `-1`, `128`
  - `nvfp4`: `16`
  - `mxfp4`: `32`
- CTA 覆盖：
  - U4/U8 family：覆盖 auto-selected CTA_N 下全部 supported CTA 组合。
  - FP8/NVFP4/MXFP4：覆盖 auto-selected CTA_N 下当前支持的 CTA 子集。

结果目录：

```text
baseline:
/root/source/repos/marlin_v100/benchmarks/results/20260529_234256_dense_auto_ctan_fullctan_baseline_fix_branch_weightgroup

current:
/root/source/repos/marlin_v100/benchmarks/results/20260529_234256_dense_auto_ctan_fullctan_current_weightgroup
```

关键 CSV 文件：

```text
current/all_results.csv
current/compare_same_config_tflops.csv
current/compare_by_quant_tflops.csv
current/compare_by_quant_cta_tflops.csv
current/suspect_kernel_configs.csv
current/confirm_baseline.csv
current/confirm_current.csv
current/confirm_same_config_tflops.csv
current/confirmed_tflops_suspects.csv
current/confirm_by_quant_tflops.csv
current/confirm_by_quant_cta_tflops.csv
current/confirm_summary_tflops.txt
```

如果同一 V100 GPU、build target 和 baseline 分支假设仍成立，且目录没有被清理，
后续 no-split Auto CTA_N / FullCtaN compact 对比可以复用上面的 baseline 目录。

## 当前大 M / 128x256x8 / 正 group-size TFLOPS

本小节只汇总 current 已落盘结果，不重新 benchmark。数据来源：

```text
/root/source/repos/marlin_v100/benchmarks/results/20260529_234256_dense_auto_ctan_fullctan_current_weightgroup/all_results.csv
```

过滤条件：

```text
phase=current
metric=kernel_like
M=5120
CTA=128x256x8
group_size > 0
split-K unset
```

完整 `kernel_like marlin_tflops` 表：

| quant | group_size | K | N | kernel_like TFLOPS | kernel_like us |
|---|---:|---:|---:|---:|---:|
| fp8 | 128 | 4096 | 256 | 31.679033 | 338.944 |
| fp8 | 128 | 16384 | 256 | 37.718562 | 1138.688 |
| fp8 | 128 | 4096 | 4096 | 85.576209 | 2007.552 |
| fp8 | 128 | 16384 | 4096 | 75.803528 | 9065.472 |
| fp8 | 128 | 4096 | 12288 | 83.302957 | 6187.008 |
| fp8 | 128 | 16384 | 12288 | 84.989169 | 24257.024 |
| mxfp4 | 32 | 4096 | 256 | 28.571553 | 375.808 |
| mxfp4 | 32 | 16384 | 256 | 28.787261 | 1491.968 |
| mxfp4 | 32 | 4096 | 4096 | 63.803826 | 2692.608 |
| mxfp4 | 32 | 16384 | 4096 | 63.916246 | 10751.488 |
| mxfp4 | 32 | 4096 | 12288 | 68.029531 | 7576.064 |
| mxfp4 | 32 | 16384 | 12288 | 66.256366 | 31115.264 |
| nvfp4 | 16 | 4096 | 256 | 33.288126 | 322.560 |
| nvfp4 | 16 | 16384 | 256 | 38.800223 | 1106.944 |
| nvfp4 | 16 | 4096 | 4096 | 86.547414 | 1985.024 |
| nvfp4 | 16 | 16384 | 4096 | 86.659175 | 7929.856 |
| nvfp4 | 16 | 4096 | 12288 | 85.300648 | 6042.112 |
| nvfp4 | 16 | 16384 | 12288 | 85.754823 | 24040.447 |
| uint4 | 32 | 4096 | 256 | 35.910136 | 299.008 |
| uint4 | 32 | 16384 | 256 | 43.129089 | 995.840 |
| uint4 | 32 | 4096 | 4096 | 85.860881 | 2000.896 |
| uint4 | 32 | 16384 | 4096 | 84.233544 | 8158.208 |
| uint4 | 32 | 4096 | 12288 | 83.351243 | 6183.424 |
| uint4 | 32 | 16384 | 12288 | 79.848730 | 25818.624 |
| uint4b8 | 32 | 4096 | 256 | 36.535749 | 293.888 |
| uint4b8 | 32 | 16384 | 256 | 44.834889 | 957.952 |
| uint4b8 | 32 | 4096 | 4096 | 87.586613 | 1961.472 |
| uint4b8 | 32 | 16384 | 4096 | 86.861073 | 7911.424 |
| uint4b8 | 32 | 4096 | 12288 | 87.320696 | 5902.336 |
| uint4b8 | 32 | 16384 | 12288 | 78.630913 | 26218.496 |
| uint8 | 32 | 4096 | 256 | 35.787578 | 300.032 |
| uint8 | 32 | 16384 | 256 | 41.363943 | 1038.336 |
| uint8 | 32 | 4096 | 4096 | 83.365050 | 2060.800 |
| uint8 | 32 | 16384 | 4096 | 80.500044 | 8536.576 |
| uint8 | 32 | 4096 | 12288 | 81.733760 | 6305.792 |
| uint8 | 32 | 16384 | 12288 | 81.319436 | 25351.680 |
| uint8b128 | 32 | 4096 | 256 | 36.472208 | 294.400 |
| uint8b128 | 32 | 16384 | 256 | 42.625039 | 1007.616 |
| uint8b128 | 32 | 4096 | 4096 | 84.883464 | 2023.936 |
| uint8b128 | 32 | 16384 | 4096 | 84.744109 | 8109.056 |
| uint8b128 | 32 | 4096 | 12288 | 85.409212 | 6034.432 |
| uint8b128 | 32 | 16384 | 12288 | 84.321742 | 24449.024 |

按 quant 的 best 汇总：

| quant | group_size | best TFLOPS | K | N | kernel_like us |
|---|---:|---:|---:|---:|---:|
| fp8 | 128 | 85.576209 | 4096 | 4096 | 2007.552 |
| mxfp4 | 32 | 68.029531 | 4096 | 12288 | 7576.064 |
| nvfp4 | 16 | 86.659175 | 16384 | 4096 | 7929.856 |
| uint4 | 32 | 85.860881 | 4096 | 4096 | 2000.896 |
| uint4b8 | 32 | 87.586613 | 4096 | 4096 | 1961.472 |
| uint8 | 32 | 83.365050 | 4096 | 4096 | 2060.800 |
| uint8b128 | 32 | 85.409212 | 4096 | 12288 | 6034.432 |

本小节 overall best：

```text
87.586613 TFLOPS
quant=uint4b8
group_size=32
CTA=128x256x8
M=5120 K=4096 N=4096
kernel_like_us=1961.472
```

## Screening 结果

Baseline 和 current screening sweep 均成功完成：

```text
baseline all_results.csv: 2449 lines including header
baseline errors.csv: header only
current all_results.csv: 2449 lines including header
current errors.csv: header only
```

Screening `kernel_like marlin_tflops`:

```text
configs: 1224
median ratio: 1.000891
min ratio: 0.726473
screening suspects: 85
```

按 quant 汇总的 screening 结果：

| quant | metric | count | median ratio | min ratio | suspect count |
|---|---|---:|---:|---:|---:|
| fp8 | kernel_like | 60 | 1.0000 | 0.8905 | 7 |
| mxfp4 | kernel_like | 30 | 1.0000 | 0.8751 | 4 |
| nvfp4 | kernel_like | 30 | 1.0000 | 0.8790 | 2 |
| uint4 | kernel_like | 276 | 1.0020 | 0.9421 | 6 |
| uint4b8 | kernel_like | 276 | 1.0008 | 0.9269 | 13 |
| uint8 | kernel_like | 276 | 1.0020 | 0.8323 | 17 |
| uint8b128 | kernel_like | 276 | 1.0000 | 0.7265 | 36 |

Screening 结果中存在足够多噪声，因此将全部 `85` 个 kernel-like suspect
使用 confirm timing 重新测量。

## Confirm 结果

Confirm 只重跑 `suspect_kernel_configs.csv` 中的配置，并保持 baseline 先跑、
current 后跑。两轮 confirm 均成功完成：

```text
confirm_baseline.csv: 256 lines including header
confirm_current.csv: 256 lines including header
configs: 85
repeats per config: 3
confirmed suspects: 25
median ratio: 0.999615
min ratio: 0.840491
max ratio: 1.060325
```

按 quant 汇总的 confirm 结果：

| quant | configs | median ratio | min ratio | confirmed |
|---|---:|---:|---:|---:|
| fp8 | 7 | 0.9760 | 0.9305 | 3 |
| mxfp4 | 4 | 1.0004 | 0.9992 | 0 |
| nvfp4 | 2 | 1.0368 | 1.0133 | 0 |
| uint4 | 6 | 0.9982 | 0.9389 | 2 |
| uint4b8 | 13 | 1.0003 | 0.9558 | 1 |
| uint8 | 17 | 1.0006 | 0.8582 | 2 |
| uint8b128 | 36 | 0.9840 | 0.8405 | 17 |

已确认回归簇：

| quant | CTA | configs | median ratio | min ratio | confirmed | worst config |
|---|---|---:|---:|---:|---:|---|
| fp8 | 256x64x8 | 2 | 0.9904 | 0.9610 | 1 | g=128 M=16 K=16384 N=64 CTA=256x64x8 ratio=0.9610 |
| fp8 | 64x128x4 | 3 | 0.9518 | 0.9305 | 2 | g=128 M=16 K=16384 N=128 CTA=64x128x4 ratio=0.9305 |
| uint4 | 64x128x4 | 1 | 0.9389 | 0.9389 | 1 | g=32 M=5120 K=16384 N=128 CTA=64x128x4 ratio=0.9389 |
| uint4 | 64x256x4 | 1 | 0.9432 | 0.9432 | 1 | g=32 M=16 K=16384 N=256 CTA=64x256x4 ratio=0.9432 |
| uint4b8 | 128x256x8 | 1 | 0.9558 | 0.9558 | 1 | g=-1 M=128 K=16384 N=4096 CTA=128x256x8 ratio=0.9558 |
| uint8 | 64x128x4 | 4 | 1.0149 | 0.8582 | 1 | g=-1 M=5120 K=16384 N=128 CTA=64x128x4 ratio=0.8582 |
| uint8 | 64x128x8 | 3 | 1.0000 | 0.9353 | 1 | g=-1 M=16 K=16384 N=128 CTA=64x128x8 ratio=0.9353 |
| uint8b128 | 128x128x8 | 1 | 0.9586 | 0.9586 | 1 | g=32 M=16 K=16384 N=128 CTA=128x128x8 ratio=0.9586 |
| uint8b128 | 128x64x8 | 6 | 0.8933 | 0.8405 | 6 | g=32 M=128 K=16384 N=64 CTA=128x64x8 ratio=0.8405 |
| uint8b128 | 256x64x4 | 6 | 0.9752 | 0.8686 | 3 | g=32 M=5120 K=4096 N=64 CTA=256x64x4 ratio=0.8686 |
| uint8b128 | 64x128x8 | 5 | 0.9224 | 0.8507 | 5 | g=32 M=128 K=16384 N=128 CTA=64x128x8 ratio=0.8507 |
| uint8b128 | 64x64x4 | 5 | 0.9844 | 0.9541 | 2 | g=32 M=128 K=16384 N=64 CTA=64x64x4 ratio=0.9541 |

最差的 confirmed configs：

| quant | CTA | group | M | K | N | baseline TFLOPS | current TFLOPS | ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uint8b128 | 128x64x8 | 32 | 128 | 16384 | 64 | 0.6378 | 0.5361 | 0.8405 |
| uint8b128 | 64x128x8 | 32 | 128 | 16384 | 128 | 0.9892 | 0.8416 | 0.8507 |
| uint8 | 64x128x4 | -1 | 5120 | 16384 | 128 | 24.9364 | 21.3995 | 0.8582 |
| uint8b128 | 256x64x4 | 32 | 5120 | 4096 | 64 | 9.6732 | 8.4021 | 0.8686 |
| uint8b128 | 128x64x8 | 32 | 5120 | 16384 | 64 | 13.6445 | 11.9632 | 0.8768 |
| uint8b128 | 128x64x8 | 32 | 128 | 4096 | 64 | 0.4174 | 0.3703 | 0.8870 |
| uint8b128 | 256x64x4 | 32 | 5120 | 16384 | 64 | 10.1165 | 9.0317 | 0.8928 |
| uint8b128 | 128x64x8 | 32 | 5120 | 4096 | 64 | 13.3068 | 11.9700 | 0.8995 |
| uint8b128 | 64x128x8 | -1 | 128 | 16384 | 128 | 1.0161 | 0.9362 | 0.9214 |
| uint8b128 | 64x128x8 | 32 | 16 | 16384 | 128 | 0.1575 | 0.1453 | 0.9224 |
| uint8b128 | 128x64x8 | 32 | 16 | 16384 | 64 | 0.0858 | 0.0793 | 0.9249 |
| fp8 | 64x128x4 | 128 | 16 | 16384 | 128 | 0.1689 | 0.1572 | 0.9305 |

## 结论解读

Auto CTA_N + FullCtaN compact 实现在整体 no-split 吞吐上基本保持稳定：
screening median ratio 为 `1.0009`，suspect 集合 confirm 后的 median ratio
为 `0.9996`。

最明确的正向结果是 MXFP4：4 个 MXFP4 screening suspects 在 confirm 后全部消失，
confirm median ratio 为 `1.0004`，min ratio 为 `0.9992`。这说明在本次对比中，
当前 MXFP4 path 不再复现此前 `128x256x8` 的大幅 TFLOPS collapse。

剩余 confirmed regressions 不是 full-macro-N 大 N 的系统性回归，主要集中在
低 N Auto CTA_N case：

- `uint8b128 group_size=32`，尤其是 `N=64` 搭配 `128x64x8` 和
  `256x64x4`，以及 `N=128` 搭配 `64x128x8`。
- 一个 large-M `uint8 group_size=-1` case，位于 `CTA=64x128x4`，
  `M=5120,K=16384,N=128`.
- 少量 small-M/small-N FP8 和 U4 case，可能更容易受 launch/fixed overhead 影响。

因此，本轮 benchmark 还不能得出 “no confirmed regression” 结论。当前实现已经完成功能验证，
full large-N 行为大体保持，但 low-N FullCtaN compact layout 仍需要后续性能修复，
之后才能认为该分支相对 `fix/sm70-dense-fullmacron-fastpath` 性能干净。

## 后续建议

建议下一步重点：

- 检查 `uint8b128 group_size=32` 在 `CTA_N=64/128` 下的 B iterator/repack
  address pattern；这是最密集的 confirmed cluster。
- 对比 baseline 分支和 current 分支在 `128x64x8`、`64x128x8`、`256x64x4`
  上的 SASS/register pressure。
- 检查 `GroupTiles=1/2` compact layout 是否相对 full-macro-N fast path 改变了
  coalescing 或 instruction scheduling。
- 修复后优先使用现有 `confirmed_tflops_suspects.csv` configs 做 reconfirm；
  只有当该 cluster 消失后，再 rerun full screening matrix。

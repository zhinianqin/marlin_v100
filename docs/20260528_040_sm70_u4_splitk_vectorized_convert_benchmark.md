# 2026-05-28 SM70 U4 split-K vectorized convert benchmark

## 背景

本轮实验对象是 dense `uint4` zero-point 的 SM70 split-K atomic fp32 reduce 路径：

```text
csrc/quantization/marlin/sm70_marlin_u4_gemm.cu
```

改动点只在 split-K 后处理的 `fp32 C_tmp -> fp16 C` convert kernel：

- baseline：每个线程 scalar 处理 1 个 fp32，并用 `__float2half_rn` 写 1 个 half。
- after：每个线程优先处理 4 个连续 fp32，使用 `float4` load，再用 `__floats2half2_rn` 写两个连续 `half2`；保留 scalar tail fallback。
- no-split 快路径不经过该 convert kernel，理论上不受影响。

固定 benchmark 条件：

| 项 | 值 |
|---|---|
| branch | `sm70-u4-splitk-atomic-reduce` |
| base commit | `0f55f51` |
| GPU | `Tesla V100-SXM2-32GB` |
| capability | `sm70 (7.0)` |
| quant | `uint4` zero-point |
| group_size | `128` |
| CTA | 15 个 supported U4 CTA 全扫 |
| M | `1, 2, 4, 8, 16, 32, 64, 128` |
| N | `4096` |
| K | `4096, 8192, 16384` |
| split_k | unset, `2`, `4`, `8` |
| act_order | `off` |
| is_k_full | `true` |
| use_fp32_reduce | `true` |
| warmup / iters | `20 / 100` |

CTA 列表：

```text
32x128x4, 32x256x4, 64x64x4, 64x128x4, 64x128x8,
64x256x4, 64x256x8, 128x64x4, 128x64x8, 128x128x4,
128x128x8, 128x256x8, 256x64x4, 256x64x8, 256x128x8
```

结果文件：

```text
baseline_dir=benchmarks/results/20260528_102712_sm70_u4_splitk_allcta_convert_baseline_fast
after_dir=benchmarks/results/20260528_104245_sm70_u4_splitk_allcta_convert_float4

baseline_all=benchmarks/results/20260528_102712_sm70_u4_splitk_allcta_convert_baseline_fast/all_results.csv
baseline_best=benchmarks/results/20260528_102712_sm70_u4_splitk_allcta_convert_baseline_fast/best_by_shape.csv

after_all=benchmarks/results/20260528_104245_sm70_u4_splitk_allcta_convert_float4/all_results.csv
after_best=benchmarks/results/20260528_104245_sm70_u4_splitk_allcta_convert_float4/best_by_shape.csv
compare_best=benchmarks/results/20260528_104245_sm70_u4_splitk_allcta_convert_float4/compare_best_by_shape.csv
compare_same_config=benchmarks/results/20260528_104245_sm70_u4_splitk_allcta_convert_float4/compare_same_config.csv
compare_summary=benchmarks/results/20260528_104245_sm70_u4_splitk_allcta_convert_float4/compare_summary.txt
```

## 验证状态

已完成：

```text
./build.sh
PYTHONPATH=$PWD/python ./.venv/bin/python -c "import torch, marlin_v100, marlin_v100._C, marlin_v100._moe_C; ..."
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q tests/test_marlin_dense.py -k "uint4_zp_split_k"
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q tests/test_marlin_dense.py -k "uint4_zp"
PYTHONPATH=$PWD/python ./.venv/bin/pytest --collect-only -q
git diff --check
```

结果：

| 项 | 结果 |
|---|---|
| build | pass |
| import | pass |
| `uint4_zp_split_k` | `19 passed, 246 deselected` |
| `uint4_zp` | `55 passed, 210 deselected` |
| collect-only | `451 tests collected` |
| `git diff --check` | pass |

## 结论

- `kernel_like_us` best-by-shape 没有任何超过 2% 的回退；24 个 `(M,K)` shape 中最大 after/baseline latency ratio 是 `1.0083`。
- `operator_us` best-by-shape 也没有超过 2% 的回退；24 个 shape 中最大 ratio 是 `1.0000`。
- vectorized convert 对 best 配置的收益很小但整体偏正向：`kernel_like_us` best 平均 ratio 是 `0.9954`，`operator_us` best 平均 ratio 是 `0.9935`。
- 重点区间 `M=64/128, K=8192/16384, split_k>1` 的 same-config `kernel_like_us` 平均 ratio 是 `0.9948`；180 行里 23 行快于 1%，151 行在 +/-1% 内，6 行慢于 1%。
- split-K best CTA/split 基本不变：`kernel_like_us` 只有 4/24 个 shape 的 best CTA/split 发生变化，且变化都在 1% 左右的 benchmark 噪声范围内。
- `M<=16` 仍应按 small-M/fixed-overhead 主导来解读，即使部分大 K case 的 FLOPs 超过脚本里的 `1e9` launch-dominated 阈值；这些 TFLOPs 不代表大矩阵吞吐上限。
- 建议保留 `float4 -> half2 + half2` convert：它没有 correctness 风险和 best latency 回退，代码路径更短；可测收益不大，主要被 atomic reduce 主 kernel、memset 和额外 launch 掩盖。

## kernel_like_us best-by-shape

ratio 是 `after_us / baseline_us`，小于 `1.0` 表示 vectorized convert 更快。

| M | K | baseline best | baseline us | after best | after us | ratio | torch us | launch_dominated |
|---:|---:|---|---:|---|---:|---:|---:|---|
| 1 | 4096 | 32x128x4/8 | 123.904 | 32x128x4/8 | 122.880 | 0.9917 | 58.368 | yes |
| 2 | 4096 | 32x256x4/8 | 122.880 | 32x128x4/8 | 123.904 | 1.0083 | 79.872 | yes |
| 4 | 4096 | 32x128x4/8 | 123.904 | 32x128x4/8 | 123.904 | 1.0000 | 79.872 | yes |
| 8 | 4096 | 32x128x4/8 | 125.952 | 32x128x4/8 | 124.928 | 0.9919 | 79.872 | yes |
| 16 | 4096 | 32x128x4/4 | 132.096 | 32x128x4/8 | 131.072 | 0.9922 | 79.872 | yes |
| 32 | 4096 | 32x128x4/4 | 140.288 | 32x128x4/4 | 140.288 | 1.0000 | 68.608 | no |
| 64 | 4096 | 64x64x4/4 | 163.840 | 64x64x4/2 | 162.816 | 0.9938 | 71.680 | no |
| 128 | 4096 | 64x64x4/unset | 179.200 | 64x64x4/unset | 177.152 | 0.9886 | 108.544 | no |
| 1 | 8192 | 32x256x4/8 | 143.360 | 32x128x4/8 | 142.336 | 0.9929 | 111.616 | yes |
| 2 | 8192 | 32x256x4/8 | 143.360 | 32x256x4/8 | 142.336 | 0.9929 | 114.688 | yes |
| 4 | 8192 | 32x128x4/8 | 144.384 | 32x128x4/8 | 143.360 | 0.9929 | 114.688 | yes |
| 8 | 8192 | 32x128x4/8 | 146.432 | 32x128x4/8 | 145.408 | 0.9930 | 113.664 | yes |
| 16 | 8192 | 32x128x4/8 | 151.552 | 32x128x4/8 | 151.552 | 1.0000 | 117.760 | no |
| 32 | 8192 | 32x128x4/8 | 165.376 | 32x128x4/8 | 164.864 | 0.9969 | 110.592 | no |
| 64 | 8192 | 64x64x4/4 | 203.776 | 64x64x4/4 | 202.752 | 0.9950 | 113.664 | no |
| 128 | 8192 | 128x128x8/2 | 263.168 | 128x128x8/2 | 259.072 | 0.9844 | 152.576 | no |
| 1 | 16384 | 32x256x4/8 | 183.296 | 32x256x4/8 | 182.272 | 0.9944 | 193.536 | yes |
| 2 | 16384 | 32x256x4/8 | 182.272 | 32x256x4/8 | 182.272 | 1.0000 | 195.584 | yes |
| 4 | 16384 | 32x256x4/8 | 183.296 | 32x256x4/8 | 183.296 | 1.0000 | 195.584 | yes |
| 8 | 16384 | 32x256x4/8 | 185.344 | 32x256x4/8 | 185.344 | 1.0000 | 193.536 | no |
| 16 | 16384 | 32x256x4/8 | 194.560 | 32x256x4/8 | 193.536 | 0.9947 | 195.584 | no |
| 32 | 16384 | 32x128x4/8 | 208.896 | 32x128x4/8 | 207.872 | 0.9951 | 191.488 | no |
| 64 | 16384 | 64x256x4/8 | 278.528 | 64x256x4/8 | 276.480 | 0.9926 | 196.608 | no |
| 128 | 16384 | 64x256x4/4 | 394.240 | 64x256x4/4 | 393.216 | 0.9974 | 248.320 | no |

## operator_us 摘要

`operator_us` 包含完整 Python/operator 调用路径。结论与 `kernel_like_us` 一致，并且 best-by-shape 没有回退：

| 指标 | 值 |
|---|---:|
| best-by-shape ratio min | `0.9857` |
| best-by-shape ratio max | `1.0000` |
| best-by-shape ratio avg | `0.9935` |
| slower than 2% count | `0 / 24` |
| best CTA/split changed | `4 / 24` |

代表性大 K / 大 M 行：

| M | K | baseline best | baseline us | after best | after us | ratio |
|---:|---:|---|---:|---|---:|---:|
| 64 | 8192 | 64x64x4/4 | 206.848 | 64x64x4/4 | 204.800 | 0.9901 |
| 128 | 8192 | 128x128x8/2 | 264.192 | 128x128x8/2 | 261.120 | 0.9884 |
| 64 | 16384 | 64x256x4/8 | 280.576 | 64x256x4/8 | 278.528 | 0.9927 |
| 128 | 16384 | 64x256x4/4 | 397.312 | 64x256x4/4 | 395.264 | 0.9948 |

## same-config 趋势

`compare_same_config.csv` 按 `(CTA, split_k, M, K, N, metric)` 对齐 baseline 和 after。整体统计如下：

| 范围 | count | avg ratio | min ratio | max ratio | faster >1% | neutral +/-1% | slower >1% |
|---|---:|---:|---:|---:|---:|---:|---:|
| kernel_like all | 1440 | 0.9963 | 0.9495 | 1.0764 | 117 | 1293 | 30 |
| kernel_like split_k>1 | 1080 | 0.9959 | 0.9495 | 1.0395 | 82 | 980 | 18 |
| kernel_like focus M=64/128,K=8192/16384,split_k>1 | 180 | 0.9948 | 0.9748 | 1.0388 | 23 | 151 | 6 |
| operator split_k>1 | 1080 | 0.9945 | 0.9498 | 1.0533 | 160 | 906 | 14 |

focus 范围里最快的一些 same-config 改善：

| CTA | split_k | M | K | baseline us | after us | ratio |
|---|---:|---:|---:|---:|---:|---:|
| 128x128x8 | 4 | 128 | 16384 | 446.464 | 435.200 | 0.9748 |
| 64x128x8 | 2 | 64 | 16384 | 355.328 | 347.136 | 0.9769 |
| 64x64x4 | 2 | 64 | 16384 | 325.120 | 318.464 | 0.9795 |
| 256x128x8 | 4 | 128 | 16384 | 624.640 | 614.400 | 0.9836 |
| 128x128x8 | 2 | 128 | 8192 | 263.168 | 259.072 | 0.9844 |

这些改善幅度不大，但方向符合预期：更大的 `C_tmp` convert 工作量下，vectorized convert 更可能被测出来。与此同时，split-K atomic 主 kernel 仍然是主耗时来源，所以 convert 优化不会像 split-K 本身那样带来大幅收益。

## 保留判断

保留本次 vectorized convert 修改。

理由：

- correctness 已通过现有 U4 split-K 与 U4 zero-point 测试。
- best-by-shape 没有超过 2% 的 regression。
- 重点 `M=64/128,K=8192/16384` 区间同配置统计略偏正向。
- no-split 快路径不经过 convert kernel，不改变默认 no-split 行为。
- `N=4096` 下 `numel = M * N` 天然 4 元素对齐，float4 load 与 half2 store 适合当前 benchmark；tail fallback 保证未来不规则 numel 仍正确。

后续如果继续优化 split-K，优先级应放在 atomic reduce 主 kernel、减少/融合 memset 和 convert launch，以及自动 split factor/CTA 选择上；当前 convert kernel 已不是主要瓶颈。

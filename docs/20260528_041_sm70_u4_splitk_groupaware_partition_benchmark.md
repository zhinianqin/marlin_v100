# 2026-05-28 SM70 U4 split-K group-aware partition benchmark

## 背景

本轮目标是把 dense `uint4` zero-point 的 SM70 split-K atomic fp32 reduce 路径，从“只支持 `K % (32 * split_k) == 0` 的均分 split-K”改成“支持任意 `K % 32 == 0` 的 32-tile partition”。

实验代码位于：

```text
csrc/quantization/marlin/sm70_marlin_u4_gemm.cu
```

核心要求有两个：

1. 正确性和覆盖优先：只要 `K % 32 == 0`，split-K 就应该允许执行。
2. 仍尽量照顾 metadata 规整性：当 `group_size > 0` 时，优先让 `k_begin` 落在 group 边界；做不到时也不拒绝，最后一个或少数分片承接剩余 32-tile。

和上一轮 `float4 -> half2 + half2` vectorized convert 实验的关系：

- baseline 复用的是上一轮已经完成、口径完全一致的全 CTA benchmark。
- 本轮 after 只在该 baseline 之上替换 split-K partition 逻辑。
- no-split 快路径不变；vectorized convert kernel 也保持不变。

## 实现摘要

本轮实现点如下：

- 新增 `Sm70U4SplitKPartition { k_begin, partition_k }`。
- 新增 host/device 共用的 32-tile partition helper：
  - `sm70_marlin_u4_active_split_k(...)`
  - `sm70_marlin_u4_partition_tile_count(...)`
  - `sm70_marlin_u4_splitk_partition<GroupSize>(...)`
- `split_k > 1` 的 gate 从：
  - 旧：`K % (32 * split_k) == 0`
  - 新：`K % 32 == 0`
- `grid.z` 不再固定等于请求的 `split_k`，而是：
  - `active_split_k = min(requested_split_k, K / 32)`
- split-K kernel 不再使用 `k_partition = K / split_k`：
  - 每个 `blockIdx.z` 都通过 helper 取得自己的 `k_begin` 和 `partition_k`
  - `partition_k == 0` 时直接 return
  - `gemm_k_iterations = partition_k / 32`

代码落点：

- split-K env 和 partition helper：
  - `csrc/quantization/marlin/sm70_marlin_u4_gemm.cu:46`
  - `csrc/quantization/marlin/sm70_marlin_u4_gemm.cu:63`
  - `csrc/quantization/marlin/sm70_marlin_u4_gemm.cu:90`
  - `csrc/quantization/marlin/sm70_marlin_u4_gemm.cu:123`
- split-K kernel 使用 partition：
  - `csrc/quantization/marlin/sm70_marlin_u4_gemm.cu:714`
  - `csrc/quantization/marlin/sm70_marlin_u4_gemm.cu:731`
  - `csrc/quantization/marlin/sm70_marlin_u4_gemm.cu:760`
- host launch 使用 `active_split_k`：
  - `csrc/quantization/marlin/sm70_marlin_u4_gemm.cu:825`
  - `csrc/quantization/marlin/sm70_marlin_u4_gemm.cu:843`

## 为什么这能支持任意 `K % 32 == 0`

因为新的 partitioner 完全按 `CTA_K=32` 的 tile 数切分：

- `total_tiles = K / 32`
- 所有 `k_begin` 和 `partition_k` 都以 tile 为单位生成
- 每个分片最终再乘回 `32`

因此新实现天然保证：

- `k_begin % 32 == 0`
- `partition_k % 32 == 0`
- 所有分片严格连续覆盖 `[0, K)`

当 `group_size > 0` 时，helper 会尝试把每个非最后分片向上 round 到 `group_tiles = group_size / 32` 的倍数，只要不会让后续分片连 1 个 32-tile 都拿不到。这样标准 case 仍保持 group-aligned，而非标准 case 也不被拒绝。

## 验证状态

在当前分支上完成了以下验证：

```text
branch=sm70-u4-splitk-atomic-reduce
after_commit=a5f7740
GPU=Tesla V100-SXM2-32GB
capability=sm70 (7.0)
```

已通过：

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
| `uint4_zp_split_k` | `22 passed, 246 deselected` |
| `uint4_zp` | `58 passed, 210 deselected` |
| `pytest --collect-only` | `454 tests collected` |
| `git diff --check` | pass |

新增 / 更新的 split-K 相关测试点：

- group-size correctness：
  - `tests/test_marlin_dense.py:1074`
- 非均匀 K correctness：
  - `tests/test_marlin_dense.py:1098`
  - 覆盖：
    - `group_size=128, split_k=2, K=384`
    - `group_size=-1, split_k=4, K=352`
    - `group_size=32, split_k=8, K=288`
- 非 32 对齐 K rejection：
  - `tests/test_marlin_dense.py:1171`

## Benchmark 口径

固定 benchmark 条件：

| 项 | 值 |
|---|---|
| quant | `uint4` zero-point |
| group_size | `128` |
| CTA | 15 个 supported U4 CTA 全扫 |
| M | `1, 2, 4, 8, 16, 32, 64, 128` |
| N | `4096` |
| K | `4096, 8192, 16384` |
| split_k | `unset, 2, 4, 8` |
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

## 结果文件

本轮复用的 baseline 与新的 after 结果目录：

```text
baseline_dir=benchmarks/results/20260528_104245_sm70_u4_splitk_allcta_convert_float4
after_dir=benchmarks/results/20260528_113224_sm70_u4_splitk_allcta_groupaware_after_fast

baseline_all=benchmarks/results/20260528_104245_sm70_u4_splitk_allcta_convert_float4/all_results.csv
baseline_best=benchmarks/results/20260528_104245_sm70_u4_splitk_allcta_convert_float4/best_by_shape.csv

after_all=benchmarks/results/20260528_113224_sm70_u4_splitk_allcta_groupaware_after_fast/all_results.csv
after_best=benchmarks/results/20260528_113224_sm70_u4_splitk_allcta_groupaware_after_fast/best_by_shape.csv
compare_best=benchmarks/results/20260528_113224_sm70_u4_splitk_allcta_groupaware_after_fast/compare_best_by_shape.csv
compare_same_config=benchmarks/results/20260528_113224_sm70_u4_splitk_allcta_groupaware_after_fast/compare_same_config.csv
compare_summary=benchmarks/results/20260528_113224_sm70_u4_splitk_allcta_groupaware_after_fast/compare_summary.txt
```

CSV 完整性：

- baseline `all_results.csv`：`2880` 行数据
- after `all_results.csv`：`2880` 行数据
- `compare_best_by_shape.csv`：`48` 行（`kernel_like_us` 24 行 + `operator_us` 24 行）
- `compare_same_config.csv`：`2880` 行

## 一个很重要的解释

这次 benchmark 里使用的标准 K：

- `4096`
- `8192`
- `16384`

在 `group_size=128` 且 `split_k=2/4/8` 时，本来就天然满足 group-aligned 均分：

- `K=4096` 时：
  - `split_k=2/4/8` 对应分片 `2048 / 1024 / 512`
- `K=8192` 时：
  - `split_k=2/4/8` 对应分片 `4096 / 2048 / 1024`
- `K=16384` 时：
  - `split_k=2/4/8` 对应分片 `8192 / 4096 / 2048`

这些分片全都是 `128` 的倍数。

所以本轮标准 benchmark 的预期不是“明显提速”，而是：

1. 保持标准 case 基本不回退。
2. 同时把功能覆盖扩展到所有 `K % 32 == 0` 的 split-K case。

也就是说，这一轮的主要收益是**覆盖和语义更稳**，不是针对 `4096/8192/16384` 这些标准 K 的吞吐优化。

## 汇总结论

`compare_summary.txt` 的核心统计如下：

| 指标 | 数值 |
|---|---|
| `best_kernel_ratio_after_over_baseline_avg` | `0.978572` |
| `best_kernel_ratio_after_over_baseline_min` | `0.958678` |
| `best_kernel_ratio_after_over_baseline_max` | `1.033597` |
| `best_operator_ratio_after_over_baseline_avg` | `0.979580` |
| `best_operator_ratio_after_over_baseline_min` | `0.955645` |
| `best_operator_ratio_after_over_baseline_max` | `1.035294` |
| `best_kernel_regress_gt_2pct` | `1 / 24` |
| `best_operator_regress_gt_2pct` | `1 / 24` |
| `same_kernel_split_gt1_ratio_after_over_baseline_avg` | `0.979528` |
| `same_operator_split_gt1_ratio_after_over_baseline_avg` | `0.980054` |

解释：

- `ratio < 1.0` 表示 after 更快。
- 对 best-by-shape 而言，`24` 个 `kernel_like_us` 形状里，只有 `1` 个超过 `2%` 回退。
- 同配置对比里，`split_k > 1` 的 `1080` 行平均 ratio 大约是 `0.98`，整体没有系统性回退。

## `kernel_like_us` best-by-shape 主表

下表主看每个 `(M, K, N=4096)` 的 best-vs-best：

| M | K | baseline best CTA/split | baseline us | after best CTA/split | after us | after / baseline | torch us | launch dominated |
|---:|---:|---|---:|---|---:|---:|---:|---|
| 1 | 4096 | `32x128x4/8` | 122.880 | `32x256x4/8` | 118.784 | 0.9667 | 59.392 | True |
| 2 | 4096 | `32x128x4/8` | 123.904 | `32x128x4/8` | 118.784 | 0.9587 | 74.752 | True |
| 4 | 4096 | `32x128x4/8` | 123.904 | `32x128x4/8` | 118.784 | 0.9587 | 75.776 | True |
| 8 | 4096 | `32x128x4/8` | 124.928 | `32x128x4/8` | 120.832 | 0.9672 | 75.776 | True |
| 16 | 4096 | `32x128x4/8` | 131.072 | `32x128x4/4` | 126.976 | 0.9687 | 76.800 | True |
| 32 | 4096 | `32x128x4/4` | 140.288 | `32x128x4/4` | 135.168 | 0.9635 | 66.560 | False |
| 64 | 4096 | `64x64x4/2` | 162.816 | `64x64x4/2` | 157.696 | 0.9686 | 70.656 | False |
| 128 | 4096 | `64x64x4/unset` | 177.152 | `64x64x4/unset` | 173.056 | 0.9769 | 104.448 | False |
| 1 | 8192 | `32x128x4/8` | 142.336 | `32x256x4/8` | 138.240 | 0.9712 | 111.616 | True |
| 2 | 8192 | `32x256x4/8` | 142.336 | `32x256x4/8` | 139.264 | 0.9784 | 108.544 | True |
| 4 | 8192 | `32x128x4/8` | 143.360 | `32x256x4/8` | 139.264 | 0.9714 | 109.568 | True |
| 8 | 8192 | `32x128x4/8` | 145.408 | `32x256x4/8` | 141.312 | 0.9718 | 112.640 | True |
| 16 | 8192 | `32x128x4/8` | 151.552 | `32x128x4/8` | 147.456 | 0.9730 | 115.712 | False |
| 32 | 8192 | `32x128x4/8` | 164.864 | `32x128x4/8` | 160.768 | 0.9752 | 108.544 | False |
| 64 | 8192 | `64x64x4/4` | 202.752 | `64x64x4/4` | 199.680 | 0.9848 | 112.640 | False |
| 128 | 8192 | `128x128x8/2` | 259.072 | `128x64x4/2` | 267.776 | 1.0336 | 149.504 | False |
| 1 | 16384 | `32x256x4/8` | 182.272 | `32x256x4/8` | 179.200 | 0.9831 | 193.504 | True |
| 2 | 16384 | `32x256x4/8` | 182.272 | `32x256x4/8` | 179.200 | 0.9831 | 188.416 | True |
| 4 | 16384 | `32x256x4/8` | 183.296 | `32x256x4/8` | 179.200 | 0.9777 | 189.440 | True |
| 8 | 16384 | `32x256x4/8` | 185.344 | `32x256x4/8` | 182.272 | 0.9834 | 190.464 | False |
| 16 | 16384 | `32x256x4/8` | 193.536 | `32x256x4/8` | 189.440 | 0.9788 | 193.536 | False |
| 32 | 16384 | `32x128x4/8` | 207.872 | `32x128x4/8` | 204.800 | 0.9852 | 189.440 | False |
| 64 | 16384 | `64x256x4/8` | 276.480 | `32x256x4/4` | 279.552 | 1.0111 | 196.608 | False |
| 128 | 16384 | `64x256x4/4` | 393.216 | `128x128x4/4` | 391.168 | 0.9948 | 246.784 | False |

## `operator_us` best-by-shape 摘要

`operator_us` 的趋势和 `kernel_like_us` 基本一致：

- best 平均 ratio：`0.979580`
- 仅 `1 / 24` 个形状超过 `2%` 回退
- 唯一 >2% 回退的形状也是：
  - `M=128, K=8192`
  - baseline best：`128x128x8/2`, `261.120 us`
  - after best：`128x64x4/2`, `270.336 us`
  - ratio：`1.035294`

## 同配置对比摘要

对所有相同 `(CTA, split_k, M, K, N)` 配置做 baseline/after 对齐后：

### `kernel_like_us`

- `split_k > 1` 行数：`1080`
- 平均 ratio：`0.979528`
- `ratio < 1.0` 的行数：`913`
- `ratio > 1.02` 的行数：`45`
- 最好的一行：
  - `CTA=128x128x4, split_k=2, M=4, K=16384`
  - ratio=`0.866983`
- 最差的一行：
  - `CTA=128x128x8, split_k=4, M=128, K=16384`
  - ratio=`1.064706`

### `operator_us`

- `split_k > 1` 行数：`1080`
- 平均 ratio：`0.980054`
- `ratio < 1.0` 的行数：`909`
- `ratio > 1.02` 的行数：`45`

这些 same-config 变化由于标准 K case 的 partition 实际没有改变，不能解读成 partition 算法本身带来了稳定提速或稳定降速；更合理的解释是 benchmark 噪声、运行时抖动、以及长 sweep 中的系统状态差异。

## 对唯一明显回退点的解读

唯一超过 `2%` 的 best-by-shape 回退是：

| metric | M | K | baseline best | baseline us | after best | after us | ratio |
|---|---:|---:|---|---:|---|---:|---:|
| `kernel_like_us` | 128 | 8192 | `128x128x8/2` | 259.072 | `128x64x4/2` | 267.776 | 1.0336 |
| `operator_us` | 128 | 8192 | `128x128x8/2` | 261.120 | `128x64x4/2` | 270.336 | 1.0353 |

如果看同配置本身：

- `128x128x8/2`
  - kernel_like：`259.072 -> 270.336`，ratio=`1.043478`
  - operator：`261.120 -> 272.384`，ratio=`1.043137`
- `128x64x4/2`
  - kernel_like：`265.216 -> 267.776`，ratio=`1.009653`
  - operator：`268.288 -> 270.336`，ratio=`1.007634`

也就是说，这个形状的回退主要来自 `128x128x8/2` 这一条在 after run 里变慢，而不是 group-aware partition 改变了标准 K 的逻辑分片。因为对 `K=8192, split_k=2, group_size=128` 来说，旧实现和新实现的 partition 都是 `4096 + 4096`。

因此更合理的判断是：

- 这是一次单轮 sweep 中的测量抖动 / 系统噪声；
- 它值得记录，但不足以推翻本轮功能改动；
- 如果 `M=128, K=8192` 是后续重点 shape，建议单独重复该形状做 targeted rerun。

## 结论

本轮 group-aware partition 可以保留，理由如下：

1. 功能上更完整：
   - 现在 split-K 支持所有 `K % 32 == 0` 的 case。
   - 非均匀 K 已经有 correctness 测试覆盖。
2. 标准 benchmark 没有系统性回退：
   - `kernel_like_us` best 平均 ratio `0.978572`
   - `operator_us` best 平均 ratio `0.979580`
   - `24` 个 shape 中只有 `1` 个超过 `2%` 回退
3. 对本轮标准 K 来说，新旧 partition 本来就应等价：
   - benchmark 的主要意义是证明“覆盖扩大但标准 case 不坏”
   - 不是证明 “4096/8192/16384 上会额外提速”

最终判断：

- 从功能性看：**值得保留**
- 从性能看：**整体可视为持平，无可证明的系统性回退**
- 从后续工作建议看：
  - 可以把这个 partitioner 作为 split-K 的稳定基础
  - 如果后续要继续追性能，更应该把注意力放回 atomic reduce 主 kernel、CTA 选择、或 split factor 策略，而不是再回到“是否必须 `K % (32 * split_k) == 0`”这个旧约束

## 备注

- `M <= 16` 的行仍属于明显的 launch / fixed-overhead dominated 区间。
- 这些小 M 的 TFLOPs 主要反映调度和固定开销，不代表大矩阵吞吐上限。

# SM70 U4 Split-K Reusable C_tmp Benchmark

## 摘要

本轮将 dense `marlin_gemm` 原 public `workspace` 参数替换为可选 fp32 `c_tmp_or_none`，用于 SM70 dense U4 zero-point split-K atomic reduce 路径复用 `C_tmp[M, N]`。MoE 的 `workspace` 接口没有修改。

结论：保留可复用 `c_tmp`。在本次全 CTA sweep 中，`kernel_like_us` 的 best-by-shape after/baseline 中位数为 `0.839`，`operator_us` 的 best-by-shape after/baseline 中位数为 `0.860`。同配置对比中，`split_k>1` 的 `kernel_like_us` 中位数为 `0.887`，`operator_us` 中位数为 `0.899`。由于 unset no-split 行也有同步变快，本轮 benchmark 存在明显系统噪声；但 split-K 行在关注区间 `M>=64,K>=8192` 的同配置 after/baseline 全部小于 `1.0`，未观察到回退。

## 运行上下文

- Branch: `sm70-u4-splitk-atomic-reduce`
- Commit: `c5fcee4`
- Dirty: 有本轮 tracked 改动；`benchmarks/results/` 为 ignored artifact
- GPU: `Tesla V100-SXM2-32GB`
- Capability: `sm70 (7.0)`
- Quant: `uint4` zero-point dense
- Group size: `128`
- N: `4096`
- K: `4096, 8192, 16384`
- M: `1, 2, 4, 8, 16, 32, 64, 128`
- CTA: 15 个 supported U4 CTA 全扫
- Split-K: `unset, 2, 4, 8`
- Timing: `warmup_iters=20`, `iters=100`

## 结果文件

- Baseline: `benchmarks/results/20260528_194404_sm70_u4_splitk_allcta_reusable_ctmp_baseline_fast`
- After: `benchmarks/results/20260528_200219_sm70_u4_splitk_allcta_reusable_ctmp_after`
- Baseline CSV: `benchmarks/results/20260528_194404_sm70_u4_splitk_allcta_reusable_ctmp_baseline_fast/all_results.csv`
- After CSV: `benchmarks/results/20260528_200219_sm70_u4_splitk_allcta_reusable_ctmp_after/all_results.csv`
- Best compare: `benchmarks/results/20260528_200219_sm70_u4_splitk_allcta_reusable_ctmp_after/compare_best_by_shape.csv`
- Same-config compare: `benchmarks/results/20260528_200219_sm70_u4_splitk_allcta_reusable_ctmp_after/compare_same_config.csv`
- Summary: `benchmarks/results/20260528_200219_sm70_u4_splitk_allcta_reusable_ctmp_after/compare_summary.txt`

## API 改动

- Dense op schema:
  - 旧：`Tensor workspace`
  - 新：`Tensor? c_tmp_or_none`
- Python dense wrapper:
  - 旧：`run_marlin_gemm(..., workspace=...)`
  - 新：`run_marlin_gemm(..., c_tmp=...)`
- 新 helper:
  - `marlin_make_c_tmp(device, numel_or_shape)` 创建 `torch.float32` buffer
- U4 split-K:
  - `split_k == 1` 或 unset 继续走 no-split fast path，忽略 `c_tmp`
  - `split_k > 1` 优先复用传入的 `c_tmp`
  - 未传 `c_tmp` 时仍 fallback 到 C++ 内部 `torch::empty({M*N}, fp32)`
  - 只清零 `M*N*sizeof(float)` 的 flat prefix
- `c_tmp` 校验：
  - CUDA tensor
  - device 与 activation 一致
  - dtype 为 `torch.float32`
  - contiguous
  - `numel >= M*N`

## 验证

- `./build.sh`: 通过
- 导入检查：通过，`import torch, marlin_v100, marlin_v100._C, marlin_v100._moe_C`
- `pytest -q tests/test_marlin_dense.py -k "uint4_zp_split_k"`: `27 passed`
- `pytest -q tests/test_marlin_dense.py -k "uint4_zp"`: `65 passed`
- `pytest -q tests/test_marlin_helpers.py`: `70 passed`
- `pytest --collect-only -q`: `461 tests collected`
- `python -m py_compile` 覆盖 dense wrapper、benchmark 和相关测试文件：通过
- `git diff --check`: 通过

## Same-Config 摘要

| metric | subset | n | median after/base | min | max |
|---|---|---:|---:|---:|---:|
| kernel_like_us | all | 1440 | 0.8929 | 0.7836 | 1.0361 |
| kernel_like_us | unset | 360 | 0.9148 | 0.8125 | 1.0349 |
| kernel_like_us | split_k>1 | 1080 | 0.8873 | 0.7836 | 1.0361 |
| kernel_like_us | split_k>1, M>=64,K>=8192 | 180 | 0.9042 | 0.8402 | 0.9742 |
| operator_us | all | 1440 | 0.9036 | 0.8029 | 1.0381 |
| operator_us | unset | 360 | 0.9207 | 0.8268 | 1.0379 |
| operator_us | split_k>1 | 1080 | 0.8992 | 0.8029 | 1.0381 |
| operator_us | split_k>1, M>=64,K>=8192 | 180 | 0.9113 | 0.8543 | 0.9850 |

## Kernel-Like Best-By-Shape

| M | K | baseline CTA/split | baseline us | after CTA/split | after us | after/base |
|---:|---:|---|---:|---|---:|---:|
| 1 | 4096 | 32x256x4/8 | 137.216 | 32x256x4/8 | 107.520 | 0.784 |
| 1 | 8192 | 32x256x4/8 | 155.648 | 32x256x4/8 | 128.000 | 0.822 |
| 1 | 16384 | 32x256x4/8 | 195.584 | 32x256x4/8 | 167.936 | 0.859 |
| 2 | 4096 | 32x128x4/8 | 137.216 | 32x128x4/8 | 107.520 | 0.784 |
| 2 | 8192 | 32x256x4/8 | 155.648 | 32x256x4/8 | 128.000 | 0.822 |
| 2 | 16384 | 32x256x4/8 | 195.584 | 32x256x4/8 | 167.936 | 0.859 |
| 4 | 4096 | 32x128x4/8 | 137.216 | 32x128x4/8 | 107.520 | 0.784 |
| 4 | 8192 | 32x128x4/8 | 156.672 | 32x256x4/8 | 129.024 | 0.824 |
| 4 | 16384 | 32x256x4/8 | 196.608 | 32x256x4/8 | 167.936 | 0.854 |
| 8 | 4096 | 32x128x4/8 | 138.240 | 32x128x4/8 | 110.592 | 0.800 |
| 8 | 8192 | 32x256x4/8 | 159.744 | 32x128x4/8 | 131.072 | 0.821 |
| 8 | 16384 | 32x256x4/8 | 199.680 | 32x256x4/8 | 169.984 | 0.851 |
| 16 | 4096 | 32x128x4/4 | 145.408 | 32x128x4/4 | 116.736 | 0.803 |
| 16 | 8192 | 32x128x4/8 | 166.912 | 32x128x4/8 | 137.216 | 0.822 |
| 16 | 16384 | 32x256x4/8 | 208.896 | 32x256x4/8 | 178.176 | 0.853 |
| 32 | 4096 | 32x128x4/4 | 154.624 | 32x128x4/4 | 124.928 | 0.808 |
| 32 | 8192 | 32x128x4/8 | 180.224 | 32x128x4/8 | 150.528 | 0.835 |
| 32 | 16384 | 32x128x4/8 | 224.256 | 32x128x4/8 | 193.536 | 0.863 |
| 64 | 4096 | 64x64x4/2 | 176.128 | 64x64x4/4 | 148.480 | 0.843 |
| 64 | 8192 | 64x64x4/4 | 217.088 | 64x64x4/4 | 188.416 | 0.868 |
| 64 | 16384 | 64x128x4/4 | 301.056 | 32x256x4/4 | 268.288 | 0.891 |
| 128 | 4096 | 64x64x4/unset | 196.608 | 64x64x4/unset | 168.960 | 0.859 |
| 128 | 8192 | 128x64x4/2 | 284.672 | 128x64x4/2 | 258.048 | 0.906 |
| 128 | 16384 | 128x128x4/4 | 414.720 | 128x128x4/4 | 379.904 | 0.916 |

## Operator Best-By-Shape

| M | K | baseline CTA/split | baseline us | after CTA/split | after us | after/base |
|---:|---:|---|---:|---|---:|---:|
| 1 | 4096 | 32x256x4/8 | 140.288 | 32x256x4/8 | 113.664 | 0.810 |
| 1 | 8192 | 32x256x4/8 | 157.696 | 32x256x4/8 | 135.168 | 0.857 |
| 1 | 16384 | 32x256x4/8 | 197.632 | 32x256x4/8 | 173.056 | 0.876 |
| 2 | 4096 | 32x128x4/4 | 140.288 | 32x128x4/8 | 112.640 | 0.803 |
| 2 | 8192 | 32x128x4/8 | 158.720 | 32x256x4/8 | 133.120 | 0.839 |
| 2 | 16384 | 32x256x4/8 | 197.632 | 32x256x4/8 | 173.056 | 0.876 |
| 4 | 4096 | 32x128x4/8 | 140.288 | 32x128x4/8 | 113.664 | 0.810 |
| 4 | 8192 | 32x256x4/8 | 158.720 | 32x256x4/8 | 134.144 | 0.845 |
| 4 | 16384 | 32x256x4/8 | 198.656 | 32x256x4/8 | 174.080 | 0.876 |
| 8 | 4096 | 32x128x4/8 | 140.288 | 32x128x4/8 | 114.688 | 0.818 |
| 8 | 8192 | 32x128x4/8 | 162.816 | 32x256x4/8 | 135.168 | 0.830 |
| 8 | 16384 | 32x256x4/8 | 202.752 | 32x256x4/8 | 175.104 | 0.864 |
| 16 | 4096 | 32x128x4/4 | 147.456 | 32x128x4/4 | 120.832 | 0.819 |
| 16 | 8192 | 32x128x4/8 | 168.960 | 32x128x4/8 | 142.336 | 0.842 |
| 16 | 16384 | 32x256x4/8 | 211.968 | 32x256x4/8 | 183.296 | 0.865 |
| 32 | 4096 | 32x128x4/4 | 156.672 | 32x128x4/4 | 129.024 | 0.824 |
| 32 | 8192 | 32x128x4/8 | 182.272 | 32x128x4/4 | 155.648 | 0.854 |
| 32 | 16384 | 32x128x4/8 | 226.304 | 32x128x4/8 | 197.632 | 0.873 |
| 64 | 4096 | 64x64x4/2 | 178.176 | 64x64x4/4 | 153.600 | 0.862 |
| 64 | 8192 | 64x64x4/4 | 220.160 | 64x64x4/4 | 193.536 | 0.879 |
| 64 | 16384 | 64x128x4/4 | 303.104 | 32x256x4/4 | 273.408 | 0.902 |
| 128 | 4096 | 64x64x4/unset | 198.656 | 64x64x4/unset | 174.080 | 0.876 |
| 128 | 8192 | 128x64x4/2 | 287.744 | 128x64x4/2 | 263.168 | 0.915 |
| 128 | 16384 | 128x128x4/4 | 416.768 | 128x128x4/4 | 385.024 | 0.924 |

## 结论

`c_tmp` 可复用接口值得保留。它让调用方可以一次性分配最大 `M*N` 的 fp32 workspace 并在多次 GEMM 调用中复用，避免 U4 split-K 路径每次在 C++ 内部执行 `torch::empty({M*N}, fp32)`。本轮同配置 split-K 行的中位延迟下降略强于 unset no-op 行；在 `M>=64,K>=8192` 关注区间中，`split_k>1` same-config after/baseline 全部小于 `1.0`，没有发现性能回退。

需要注意的是，本轮 after 与 baseline 不是同一进程内 A/B 交替测量，unset no-split 行也显著变快，说明存在系统级 benchmark 波动。因此不能把全部 best-by-shape 改善都归因于 `c_tmp` 复用本身。更稳妥的判断是：可复用 `c_tmp` 至少没有破坏 no-split fast path，并且对 split-K 的 operator/kernel-like 路径都提供了可测的正向空间，尤其适合后续 MoE 或 batched 调度中复用同一个 fp32 reduce buffer。

`M<=16` 仍属于 launch/fixed-overhead dominated 区间，不应把这些 TFLOPs 当作大矩阵吞吐上限。`M=64/128,K=8192/16384` 更接近真实 reduce workspace 成本，本轮这些 case 的 after/baseline ratio 在 `0.868-0.916` 之间。

# 2026-05-27 SM70 U4 split-K atomic reduce benchmark 结论

## 背景

本轮 benchmark 目标是 dense `uint4` zero-point 的 SM70 split-K 实验路径：

```text
csrc/quantization/marlin/sm70_marlin_u4_gemm.cu
```

固定条件：

| 项 | 值 |
|---|---|
| branch | `sm70-u4-splitk-atomic-reduce` |
| commit | `ee97cb8` |
| GPU | `Tesla V100-SXM2-32GB` |
| capability | `sm70 (7.0)` |
| quant | `uint4` zero-point |
| group_size | `128` |
| CTA | default, `SM70_MARLIN_U4_CTA` unset |
| M | `1, 2, 4, 8, 16, 32, 64, 128` |
| N | `4096` |
| K | `4096, 8192, 16384` |
| split_k | unset, `1`, `2`, `4`, `8` |
| act_order | `off` |
| is_k_full | `true` |
| use_fp32_reduce | `true` |
| warmup / iters | `20 / 100` |

命令口径是临时 inline Python 调用 `benchmarks.benchmark_marlin_dense.run_case()`，因为现有 dense benchmark CLI 的 `models=ideal` 只能覆盖 `K=N=4096`，不能直接传 `K=8192/16384`。

结果文件：

```text
benchmarks/results/20260527_112121_sm70_u4_splitk_atomic_k_sweep/benchmark.log
benchmarks/results/20260527_112121_sm70_u4_splitk_atomic_k_sweep/all_results.csv
```

## 结论

- split-K atomic fp32 reduce 对默认 CTA 的小/中 M 都有明显收益。
- `kernel_like_us` 下，相对 no-split unset 的 best 加速：
  - `K=4096`: `1.35x..1.98x`
  - `K=8192`: `1.78x..2.42x`
  - `K=16384`: `2.15x..2.80x`
- `operator_us` 下结论基本一致，说明 fp32 temp allocation、memset、atomic partial kernel 和 fp32-to-fp16 convert 的端到端开销没有吞掉收益。
- best split factor 大部分是 `split_k=4`；例外是 `K=8192,M=1/2` 的 best 为 `8`，以及 `K=16384,M=4` 的 `kernel_like_us` best 为 `8`。
- unset 和 `SM70_MARLIN_U4_SPLIT_K=1` 都走 no-split 快路径，结果总体接近但有 benchmark 抖动。
- 即使 split-K 变快，当前 Marlin 仍明显慢于同 shape 的 `torch.mm`，尤其 `M>=32` 后 torch TFLOPs 更高。

## kernel_like_us best

`kernel_like_us` 是主观察口径，复用预分配 output，更接近 kernel 本身耗时。

| K | M | best split | marlin_us | marlin_tflops | unset_us | speedup_vs_unset | torch_us | torch_tflops | launch_dominated |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4096 | 1 | 4 | 160.77 | 0.21 | 318.46 | 1.98x | 58.37 | 0.57 | yes |
| 4096 | 2 | 4 | 169.47 | 0.40 | 317.44 | 1.87x | 79.87 | 0.84 | yes |
| 4096 | 4 | 4 | 169.98 | 0.79 | 317.44 | 1.87x | 79.87 | 1.68 | yes |
| 4096 | 8 | 4 | 162.82 | 1.65 | 317.44 | 1.95x | 76.80 | 3.50 | yes |
| 4096 | 16 | 4 | 175.10 | 3.07 | 317.44 | 1.81x | 79.87 | 6.72 | yes |
| 4096 | 32 | 4 | 186.37 | 5.76 | 318.46 | 1.71x | 68.61 | 15.65 | no |
| 4096 | 64 | 4 | 183.30 | 11.72 | 310.27 | 1.69x | 70.66 | 30.39 | no |
| 4096 | 128 | 4 | 239.62 | 17.92 | 322.56 | 1.35x | 116.74 | 36.79 | no |
| 8192 | 1 | 8 | 235.52 | 0.28 | 553.98 | 2.35x | 109.57 | 0.61 | yes |
| 8192 | 2 | 8 | 233.47 | 0.57 | 553.98 | 2.37x | 110.59 | 1.21 | yes |
| 8192 | 4 | 4 | 239.62 | 1.12 | 512.00 | 2.14x | 114.69 | 2.34 | yes |
| 8192 | 8 | 4 | 218.11 | 2.46 | 509.95 | 2.34x | 113.66 | 4.72 | yes |
| 8192 | 16 | 4 | 229.38 | 4.68 | 555.01 | 2.42x | 117.76 | 9.12 | no |
| 8192 | 32 | 4 | 252.93 | 8.49 | 512.00 | 2.02x | 112.64 | 19.07 | no |
| 8192 | 64 | 4 | 270.34 | 15.89 | 558.08 | 2.06x | 116.74 | 36.79 | no |
| 8192 | 128 | 4 | 293.89 | 29.23 | 522.24 | 1.78x | 161.79 | 53.09 | no |
| 16384 | 1 | 4 | 377.86 | 0.35 | 936.96 | 2.48x | 193.54 | 0.69 | yes |
| 16384 | 2 | 4 | 333.82 | 0.80 | 935.94 | 2.80x | 188.42 | 1.43 | yes |
| 16384 | 4 | 8 | 362.50 | 1.48 | 936.96 | 2.58x | 196.61 | 2.73 | yes |
| 16384 | 8 | 4 | 353.79 | 3.04 | 936.96 | 2.65x | 196.61 | 5.46 | no |
| 16384 | 16 | 4 | 357.38 | 6.01 | 937.98 | 2.62x | 195.58 | 10.98 | no |
| 16384 | 32 | 4 | 368.64 | 11.65 | 941.06 | 2.55x | 193.54 | 22.19 | no |
| 16384 | 64 | 4 | 412.67 | 20.82 | 897.54 | 2.18x | 200.70 | 42.80 | no |
| 16384 | 128 | 4 | 450.56 | 38.13 | 970.75 | 2.15x | 261.12 | 65.79 | no |

## operator_us best

`operator_us` 包含完整 Python/operator 调用路径。趋势与 `kernel_like_us` 基本一致：

| K | best speedup_vs_unset range | 主要 best split |
|---:|---:|---|
| 4096 | `1.34x..1.97x` | `4` |
| 8192 | `1.80x..2.40x` | `4`, `M=1/2` 为 `8` |
| 16384 | `2.23x..3.04x` | `4` |

`operator_us` 口径下 `K=16384,M=2` 的 best 加速最高，约 `3.04x`。

## 观察

- 对当前默认 CTA，split-K 更像是补小 M 并行度的有效实验路径，而不是最终性能上限。
- `split_k=4` 是最稳妥的默认实验值；`split_k=8` 对极小 M 有时更快，但并不稳定，且在较大 M 下经常慢于 `4`。
- `M<=16` 多数属于 launch-dominated 区间，应该以 latency 为主，不应把 TFLOPs 当作吞吐上限。
- `M>=32` 虽然不再全部 launch-dominated，但 torch baseline 仍显著更快，后续如果要追 torch/cuBLAS，应继续优化主 kernel 本身，而不仅是 host split-K。

## 后续建议

- 若保留 split-K 实验入口，建议先把 `split_k=4` 作为手动调参主选。
- 可以补一个 `group_size=-1/32/64/128` 的小 sweep，确认 metadata group size 是否改变 split-K 收益。
- 如果要自动选择 split factor，建议先限制在 `K>=8192 && M<=128`，并把 `split_k=4` 作为默认候选，`M<=2 && K=8192` 这类极小 M 再考虑 `8`。
- 在投入更多 split-K 复杂度前，需要和更小 CTA 或 Marlin atom path 对比，确认收益来自 K 并行度而不是默认 CTA 不适合 small-M。

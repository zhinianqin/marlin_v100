# SM70 Marlin 自动策略字段命名说明

日期：2026-06-04

## 结论

Dense/MoE benchmark 与矩阵 summary 中，自动策略观测字段统一命名为：

```text
auto_cta_geometry
auto_split_k
```

这两个字段只描述自动策略在 launch 前按 shape 推导出的期望配置，不表示从 CUDA kernel runtime 反查到的状态。

## 字段语义

| 字段 | 含义 |
| --- | --- |
| `auto_cta_geometry` | 自动策略选择的 CTA geometry，格式为 `CTA_MxCTA_NxWarps`。 |
| `auto_split_k` | 自动策略选择的 split-K 值。 |

Dense 是单段 GEMM，因此字段值是单值：

```text
auto_cta_geometry=64x256x4
auto_split_k=2
```

MoE 包含 stage1 和 stage2 两段 GEMM。若两段策略相同，字段仍写单值；若两段不同，字段写 stage pair：

```text
auto_cta_geometry=stage1=32x256x4;stage2=32x128x4
auto_split_k=stage1=2;stage2=1
```

## 命名原因

- `auto_` 表示该字段来自自动策略，不是 env/manual tuning，也不是 benchmark 输入维度。
- `cta_geometry` 明确包含 `CTA_M`、`CTA_N` 和 `Warps`，不是单独的 CTA_N 或 CTA_M。
- Dense/MoE 共享同一 CSV schema；MoE 的两段 GEMM 差异通过字段值中的 `stage1`/`stage2` 表示。

## Artifact 约定

新的 reduced full benchmark 使用 20260604 文件名，避免覆盖历史 20260603 结果：

```text
benchmarks/results/20260604_dense_auto_cta_geometry_splitk_iters1.csv
benchmarks/results/20260604_dense_auto_cta_geometry_splitk_iters1.log
benchmarks/results/20260604_moe_auto_cta_geometry_splitk_iters1.csv
benchmarks/results/20260604_moe_auto_cta_geometry_splitk_iters1.log
```

历史 benchmark CSV/log 和历史验证文档保持原样，不作为本次字段命名的最新 schema 依据。

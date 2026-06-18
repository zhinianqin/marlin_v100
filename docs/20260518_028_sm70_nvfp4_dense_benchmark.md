# SM70 NVFP4 dense benchmark 接入记录

## 背景

本轮把 SM70 dense-only NVFP4 weight path 接入现有 benchmark 框架：

- kernel 路径：`csrc/quantization/marlin/sm70_marlin_nvfp4_gemm.cu`
- benchmark 入口：`benchmark.sh dense`
- Python 脚本：`benchmarks/benchmark_marlin_dense.py`

这不是新增独立 C++ benchmark，也不改变 `marlin_gemm` Torch op ABI。benchmark 继续复用现有 CUDA event timing、dense 结果表和 `dense.run_marlin_gemm(...)` 调用路径。

NVFP4 benchmark 只覆盖当前已经实现的 dense weight-only 能力：

| 项 | 支持范围 |
|---|---|
| weight type | `nvfp4` / `float4_e2m1f` / `vllm::kFE2M1f` |
| activation | `float16` |
| output | `float16` |
| scales | preconverted `float16` |
| global scale | single-element `float32` |
| group_size | `16` |
| act_order | 不支持 |
| is_k_full | 只支持 `true` |
| MoE / MXFP4 / raw FP8 scale / BF16 / FP4 activation | 不支持 |

## 实现

`benchmarks/benchmark_marlin_dense.py` 中 dense quant candidates 新增：

```python
"nvfp4": scalar_types.float4_e2m1f
```

benchmark 层在 runtime dense group sizes 基础上额外暴露 `group_size=16`，但这个值只允许 NVFP4 case 使用。这样不会把 `16` 误表达成 uint4、uint8 或 FP8 的通用 dense group size。

统一 case filter 现在同时覆盖 FP8 和 NVFP4：

- `group_size != -1` 时要求 `size_k % group_size == 0`。
- 非 NVFP4 quant 自动跳过 `group_size=16`。
- NVFP4 只允许 `group_size == 16`。
- NVFP4 跳过 `act_order=True`。
- NVFP4 跳过 `is_k_full=False`。
- NVFP4 要求 `size_k % 32 == 0`。
- NVFP4 要求 `size_n % 64 == 0`。

NVFP4 数据流复用本地 helper：

```python
weight_ref, q_weight, scales, global_scale, g_idx, sort_indices, _ = (
    marlin_quantize_nvfp4(weight, 16)
)
```

其中：

- `q_weight` 是 SM70 native Marlin int32 packed FP4 layout。
- `scales` 是预转换后的 FP16 block scales，即 `fp8_scale.to(float16) * 128`。
- `global_scale` 是预转换后的 FP32 single-element tensor，即 `global_scale * 128`。
- benchmark 传给 kernel 的 `b_q_type` 是 `scalar_types.float4_e2m1f.id`。
- torch baseline 使用 `torch.matmul(a, weight_ref)`，仅作为同形状 FP16 GEMM latency 对照。

operator timing 和 kernel-like timing 都会把 `global_scale` 传入 `dense.run_marlin_gemm(...)`。这条 benchmark 路径不测试 raw FP8 scale ABI。

## 使用方式

只跑 NVFP4 smoke：

```bash
DENSE_ARGS="--models smoke --batch-sizes 16 --quant-types nvfp4 --group-sizes 16 --act-order off --is-k-full true --warmup-iters 1 --iters 2" \
  BENCH_PRESET=smoke ./benchmark.sh dense
```

典型 quick benchmark：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types nvfp4 --group-sizes 16 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

默认 dense smoke/full benchmark 现在也会包含 `nvfp4`：

```bash
DENSE_ARGS="--models smoke --batch-sizes 16 --warmup-iters 1 --iters 1" \
  BENCH_PRESET=smoke ./benchmark.sh dense
```

此时全局 `--group-sizes` 默认包含：

```text
-1 32 64 128 16
```

但 `16` 只会生成 NVFP4 case；其它 quant type 会自动跳过 `group_size=16`。

NVFP4 kernel 支持独立 CTA geometry env override：

```bash
SM70_MARLIN_NVFP4_CTA=128x256x8 \
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types nvfp4 --group-sizes 16 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

本轮不新增 benchmark CLI sweep 参数；需要比较不同 CTA 时，通过多次设置 `SM70_MARLIN_NVFP4_CTA` 外部重复运行。

## 验证

建议先做静态和导入检查：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python -m py_compile benchmarks/benchmark_marlin_dense.py
PYTHONPATH=$PWD/python ./.venv/bin/python -c "import benchmarks.benchmark_marlin_dense"
```

再做现有回归：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest tests/test_calibration.py tests/test_marlin_helpers.py -q
PYTHONPATH=$PWD/python ./.venv/bin/pytest tests/test_marlin_dense.py -k "fp8 or nvfp4" -q
```

benchmark smoke：

```bash
DENSE_ARGS="--models smoke --batch-sizes 16 --quant-types nvfp4 --group-sizes 16 --act-order off --is-k-full true --warmup-iters 1 --iters 2" \
  BENCH_PRESET=smoke ./benchmark.sh dense
```

默认 dense sanity：

```bash
DENSE_ARGS="--models smoke --batch-sizes 16 --warmup-iters 1 --iters 1" \
  BENCH_PRESET=smoke ./benchmark.sh dense
```

当前 SM70 机器上 benchmark 结果只作为入口和 case filtering 的开发信号；最终数值与性能验收仍放到支持 Marlin 运行的 SM75 环境。

## 本次 smoke 结果

NVFP4-only smoke 日志：

```text
benchmarks/results/20260518_205524_dense_smoke.log
```

关键检查点：

- `quant_types=['nvfp4']`
- `group_sizes=[16]`
- `total_cases=2`
- case 只包含 `group_size=16`
- 没有 `act_order=True`
- 没有 `is_k_full=False`

默认 dense smoke 日志：

```text
benchmarks/results/20260518_205550_dense_smoke.log
```

关键检查点：

- 默认 quant list 包含 `nvfp4`
- 默认 group list 包含 `16`
- `total_cases=38`
- `group_size=16` 只生成 NVFP4 case
- FP8 仍只生成 `group_size=-1` 和 `128`

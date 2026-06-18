# SM70 MXFP4 dense benchmark 接入记录

## 背景

本轮把 SM70 dense-only MXFP4 weight path 接入现有 benchmark 框架：

- kernel 路径：`csrc/quantization/marlin/sm70_marlin_mxfp4_gemm.cu`
- benchmark 入口：`benchmark.sh dense`
- Python 脚本：`benchmarks/benchmark_marlin_dense.py`

这不是新增独立 C++ benchmark，也不改变 `marlin_gemm` Torch op ABI。benchmark 继续复用现有 CUDA event timing、dense 结果表和 `dense.run_marlin_gemm(...)` 调用路径。

MXFP4 benchmark 只覆盖当前已经实现的 dense weight-only 能力：

| 项 | 支持范围 |
|---|---|
| weight type | `mxfp4` / `float4_e2m1f` / `vllm::kFE2M1f` |
| activation | `float16` |
| output | `float16` |
| scales | preconverted `float16` E8M0 scale values |
| global scale | 不使用 |
| group_size | `32` |
| act_order | 不支持 |
| is_k_full | 只支持 `true` |
| MoE / FP4 activation / BF16 / bias / zero-point / raw E8M0 decode | 不支持 |

## 实现

`benchmarks/benchmark_marlin_dense.py` 中 dense quant candidates 新增：

```python
"mxfp4": scalar_types.float4_e2m1f
```

MXFP4 与 NVFP4 的 weight type 都是 `float4_e2m1f / kFE2M1f`，本地 benchmark 路由通过 scale/global-scale 组合区分：

- NVFP4 使用 `group_size=16`，并传入 FP32 single-element `global_scale`。
- MXFP4 使用 `group_size=32`，不传 `global_scale`。

统一 case filter 现在同时覆盖 FP8、NVFP4 和 MXFP4：

- `group_size != -1` 时要求 `size_k % group_size == 0`。
- `group_size=16` 只允许 NVFP4 case 使用。
- MXFP4 只允许 `group_size == 32`。
- MXFP4 跳过 `act_order=True`。
- MXFP4 跳过 `is_k_full=False`。
- MXFP4 要求 `size_k % 32 == 0`。
- MXFP4 要求 `size_n % 64 == 0`。

MXFP4 数据流复用本地 helper：

```python
weight_ref, q_weight, scales, g_idx, sort_indices, _ = (
    marlin_quantize_mxfp4(weight, 32)
)
```

其中：

- `q_weight` 是 SM70 native Marlin int32 packed FP4 layout。
- `scales` 是预转换后的 FP16 E8M0 block scale 数值。
- 本地 SM70 direct ABI 不消费 raw `torch.float8_e8m0fnu` scale tensor。
- benchmark 传给 kernel 的 `b_q_type` 是 `scalar_types.float4_e2m1f.id`。
- torch baseline 使用 `torch.matmul(a, weight_ref)`，仅作为同形状 FP16 GEMM latency 对照。

operator timing 和 kernel-like timing 都不传 `global_scale`。这条 benchmark 路径不测试 raw E8M0 scale ABI。

## 使用方式

只跑 MXFP4 smoke：

```bash
DENSE_ARGS="--models smoke --batch-sizes 16 --quant-types mxfp4 --group-sizes 32 --act-order off --is-k-full true --warmup-iters 1 --iters 2" \
  BENCH_PRESET=smoke ./benchmark.sh dense
```

典型 quick benchmark：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types mxfp4 --group-sizes 32 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

默认 dense smoke/full benchmark 现在也会包含 `mxfp4`：

```bash
DENSE_ARGS="--models smoke --batch-sizes 16 --warmup-iters 1 --iters 1" \
  BENCH_PRESET=smoke ./benchmark.sh dense
```

此时全局 `--group-sizes` 默认包含：

```text
-1 32 64 128 16
```

但 `mxfp4` 只会生成 `group_size=32` case；`nvfp4` 仍只会生成 `group_size=16` case；FP8 仍只生成 `group_size=-1` 和 `128` case。

MXFP4 kernel 支持独立 CTA geometry env override：

```bash
SM70_MARLIN_MXFP4_CTA=128x256x8 \
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types mxfp4 --group-sizes 32 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

本轮不新增 benchmark CLI sweep 参数；需要比较不同 CTA 时，通过多次设置 `SM70_MARLIN_MXFP4_CTA` 外部重复运行。

## 验证

建议先做静态和导入检查：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python -m py_compile benchmarks/benchmark_marlin_dense.py
PYTHONPATH=$PWD/python ./.venv/bin/python -c "import benchmarks.benchmark_marlin_dense"
```

再做现有回归：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest tests/test_calibration.py tests/test_marlin_helpers.py -q
PYTHONPATH=$PWD/python ./.venv/bin/pytest tests/test_marlin_dense.py -k "mxfp4 or nvfp4 or fp4" -q
```

benchmark smoke：

```bash
DENSE_ARGS="--models smoke --batch-sizes 16 --quant-types mxfp4 --group-sizes 32 --act-order off --is-k-full true --warmup-iters 1 --iters 2" \
  BENCH_PRESET=smoke ./benchmark.sh dense
```

默认 dense sanity：

```bash
DENSE_ARGS="--models smoke --batch-sizes 16 --warmup-iters 1 --iters 1" \
  BENCH_PRESET=smoke ./benchmark.sh dense
```

当前 SM70 机器上 benchmark 结果只作为入口和 case filtering 的开发信号；最终数值与性能验收仍放到支持 Marlin 运行的 SM75 环境。

## 本次 smoke 结果

MXFP4-only smoke 日志：

```text
benchmarks/results/20260518_230111_dense_smoke.log
```

关键检查点：

- `quant_types=['mxfp4']`
- `group_sizes=[32]`
- `total_cases=2`
- case 只包含 `group_size=32`
- 没有 `act_order=True`
- 没有 `is_k_full=False`

默认 dense smoke 日志：

```text
benchmarks/results/20260518_230132_dense_smoke.log
```

关键检查点：

- 默认 quant list 包含 `mxfp4`
- 默认 group list 包含 `-1, 32, 64, 128, 16`
- `total_cases=40`
- `mxfp4` 只生成 `group_size=32` case
- `nvfp4` 仍只生成 `group_size=16` case
- FP8 仍只生成 `group_size=-1` 和 `128` case

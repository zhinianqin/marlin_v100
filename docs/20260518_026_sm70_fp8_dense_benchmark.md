# SM70 FP8 dense benchmark 接入记录

## 背景

本轮把 dense SM70 FP8 weight-only 路径接入现有 benchmark 框架：

- kernel 路径：`csrc/quantization/marlin/sm70_marlin_fp8_gemm.cu`
- benchmark 入口：`benchmark.sh dense`
- Python 脚本：`benchmarks/benchmark_marlin_dense.py`

这不是新增独立 C++ benchmark，也不改变 `marlin_gemm` Torch op ABI。benchmark 继续复用现有 CUDA event timing、dense 结果表、`marlin_quantize(...)` helper 和 `dense.run_marlin_gemm(...)` 调用路径。

FP8 benchmark 只覆盖当前已经实现的 dense weight-only 能力：

| 项 | 支持范围 |
|---|---|
| weight type | `fp8` / `float8_e4m3fn` / `vllm::kFE4M3fn` |
| activation | `float16` |
| output | `float16` |
| scales | fused `float16` |
| group_size | `-1`、`128` |
| act_order | 不支持 |
| is_k_full | 只支持 `true` |
| MoE | 不支持 |
| FP8 activation / NVFP4 / MXFP4 / BF16 | 不支持 |

## 实现

`benchmarks/benchmark_marlin_dense.py` 中 dense quant candidates 新增：

```python
"fp8": scalar_types.float8_e4m3fn
```

因此：

- `--quant-types fp8` 可以直接使用。
- 默认 `--quant-types` 会包含当前 support matrix 返回的 `fp8`。
- 默认 dense smoke/full benchmark 不需要额外手动传 `fp8`。

脚本新增统一 case filter：

```python
_is_supported_dense_benchmark_case(...)
```

该 filter 同时用于 case 生成和 `run_case(...)` 入口，避免 benchmark 列出无效 case 后在 kernel runtime 才报错。

通用过滤保持不变：

- `group_size != -1` 时要求 `size_k % group_size == 0`。
- `act_order=True, is_k_full=True, group_size=-1` 继续跳过。

FP8 额外过滤：

- 只允许 `group_size in (-1, 128)`。
- 跳过 `act_order=True`。
- 跳过 `is_k_full=False`。
- 要求 `size_k % 32 == 0`。
- 要求 `size_n % 64 == 0`。

FP8 数据流复用现有 helper：

```python
weight_ref, q_weight, scales, g_idx, sort_indices, _ = marlin_quantize(
    weight, scalar_types.float8_e4m3fn, group_size, False
)
```

其中：

- `q_weight` 是 SM70 native Marlin int32 packed layout。
- `scales` 是已经融合 FP8 exponent bias 的 FP16 scale，即 helper 中的 `scale * 256`。
- benchmark 传给 kernel 的 `b_q_type` 是 `scalar_types.float8_e4m3fn.id`。
- torch baseline 继续使用 `torch.matmul(a, weight_ref)`，仅作为同形状 FP16 GEMM latency 对照。

## 使用方式

只跑 FP8 smoke：

```bash
DENSE_ARGS="--models smoke --batch-sizes 16 --quant-types fp8 --group-sizes -1 128 --act-order off --is-k-full true --warmup-iters 1 --iters 2" \
  BENCH_PRESET=smoke ./benchmark.sh dense
```

典型 quick benchmark：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types fp8 --group-sizes -1 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

默认 dense smoke/full benchmark 现在也会包含 `fp8`：

```bash
DENSE_ARGS="--models smoke --batch-sizes 16 --warmup-iters 1 --iters 1" \
  BENCH_PRESET=smoke ./benchmark.sh dense
```

此时全局 `--group-sizes` 默认仍是：

```text
-1 32 64 128
```

但 FP8 只会生成 `-1` 和 `128`，不会生成 unsupported 的 `32/64` case。

FP8 kernel 已经支持独立 CTA geometry env override：

```bash
SM70_MARLIN_FP8_CTA=128x256x8 \
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types fp8 --group-sizes 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

本轮不新增 benchmark CLI sweep 参数；需要比较不同 CTA 时，通过多次设置 `SM70_MARLIN_FP8_CTA` 外部重复运行。

## 验证

静态检查：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python -m py_compile benchmarks/benchmark_marlin_dense.py
PYTHONPATH=$PWD/python ./.venv/bin/python -c "import benchmarks.benchmark_marlin_dense"
```

结果：通过。

现有 helper 和 calibration 回归：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest tests/test_calibration.py tests/test_marlin_helpers.py -q
```

结果：

```text
64 passed in 4.75s
```

FP8 dense CUDA 回归：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest tests/test_marlin_dense.py -k "fp8" -q
```

结果：

```text
12 passed, 171 deselected in 2.01s
```

## Benchmark smoke 结果

FP8-only smoke 命令：

```bash
DENSE_ARGS="--models smoke --batch-sizes 16 --quant-types fp8 --group-sizes -1 128 --act-order off --is-k-full true --warmup-iters 1 --iters 2" \
  BENCH_PRESET=smoke ./benchmark.sh dense
```

日志：

```text
benchmarks/results/20260518_185731_dense_smoke.log
```

关键检查点：

- `quant_types=['fp8']`
- `group_sizes=[-1, 128]`
- `total_cases=4`
- case 只包含：
  - `16x256x256, group_size=-1`
  - `16x256x256, group_size=128`
  - `16x512x512, group_size=-1`
  - `16x512x512, group_size=128`
- 没有 `group_size=32/64`
- 没有 `act_order=True`
- 没有 `is_k_full=False`

默认 dense smoke 命令：

```bash
DENSE_ARGS="--models smoke --batch-sizes 16 --warmup-iters 1 --iters 1" \
  BENCH_PRESET=smoke ./benchmark.sh dense
```

日志：

```text
benchmarks/results/20260518_185748_dense_smoke.log
```

关键检查点：

- 默认 quant list 已包含 FP8：

```text
quant_types=['uint4', 'uint4b8', 'uint8', 'uint8b128', 'fp8']
```

- 默认 group list 仍是：

```text
group_sizes=[-1, 32, 64, 128]
```

- `total_cases=36`，说明：
  - 4 个 integer quant type 各自保留 `-1/32/64/128`。
  - FP8 只生成 `-1/128`。
  - 2 个 smoke shapes 下总数为 `2 * (4 * 4 + 2) = 36`。

这证明默认 dense benchmark 已包含 FP8，同时不会因为 FP8 不支持 `group_size=32/64` 而失败。

## 注意事项

- smoke benchmark 的 `M=16` 属于 launch-dominated 小形状，结果主要用于确认入口、case filtering 和 kernel 可运行，不用于判断 FP8 kernel 最终吞吐。
- 性能比较应使用较大的 `M/K/N`，例如 `ideal + batch_size=5120`，并开启 `--report-tflops`。
- FP8 的 torch baseline 是 `a @ weight_ref`，其中 `weight_ref` 来自 helper 返回的 FP16 reference weight；它不是 FP8 kernel 的逐指令等价路径，只作为同形状 GEMM latency 参照。

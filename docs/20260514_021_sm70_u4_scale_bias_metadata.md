# SM70 u4 Scale+Bias Metadata 重构

## 摘要

本次改动把 dense SM70 `uint4` 路径从运行时解包 packed zero-point，改为离线预计算 zero-point bias：

- `b_scales`：fp16、连续、shape 为 `(num_groups, size_n)`
- `b_zp_bias`：fp16、连续、shape 为 `(num_groups, size_n)`
- 运行时数学：`dequantized = q * scale + zp_bias`

`zp_bias` 在 Python helper 中预先计算，公式是 `-zero_point * scale`。CUDA kernel 热路径不再读取 packed zero-point word，不再解包 zero 值，也不再现场计算 `-zero * scale`。

MoE 本轮有意不改：MoE 仍然使用 packed zero-point metadata，因此继续保留 `b_zeros` / zero-point 命名。`marlin.cu` 中旧 generic Marlin selector 代码也仍然保留 `is_zp_float` 命名，但当前启用的 SM70 CUTLASS dense 路径会在进入 legacy fallback 前直接 return。

## 接口变化

Dense Torch op schema：

- `b_zeros_or_none` -> `b_zp_bias_or_none`
- `is_zp_float` -> `use_zp_bias`

Dense Python wrapper：

- `b_zeros` -> `b_zp_bias`
- `is_zp_float` -> `use_zp_bias`
- 当调用方不显式传 `use_zp_bias` 时，`run_marlin_gemm(...)` 会用 `b_zp_bias is not None` 自动推导。

Dense helper：

- 新增 `marlin_quantize_uint4_zp_bias(...)`
- 返回 `weight, q_weight, scales, zp_bias, dequantized`
- 新增 `marlin_dequantize_uint4_zp_bias(...)`

packed zero-point helper / reference 仍然保留，用于测试和 MoE，但不再作为 dense SM70 `uint4` kernel 输入。

## CUDA 路径

`csrc/quantization/marlin/sm70_marlin_u4_gemm.cu` 现在使用 `Sm70U4ZpBiasIteratorB`。

`GroupSize=-1` 时，iterator constructor 一次性缓存两个 metadata 平面：

- 每个 contiguous access 缓存 4 个 `half2` scale
- 每个 contiguous access 缓存 4 个 `half2` bias

`GroupSize=32/64/128` 时，`load()` 在进入 `load_full_tile()` 或 `load_residue_tile()` 前调用 `cache_current_group_metadata(...)`。full/residue 热路径只消费 `cached_scales_` 和 `cached_bias_`。

full/residue dequant 顺序保持流式：

1. load packed qword
2. dequant low half
3. 立刻用 `__hfma2(q, scale, bias)` 写 low half
4. dequant high half
5. 立刻用 `__hfma2(q, scale, bias)` 写 high half

## 验证

构建：

```bash
./build.sh
```

结果：通过。

导入：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python -c \
  "import marlin_v100, marlin_v100._C, marlin_v100._moe_C; print('imports ok')"
```

结果：通过。

helper 测试：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_marlin_helpers.py::test_uint4_repack_layout_matches_reference \
  tests/test_marlin_helpers.py::test_marlin_dequantize_uint4_zp_matches_quantize_helper_output \
  tests/test_marlin_helpers.py::test_marlin_quantize_uint4_zp_bias_round_trip_matches_original_weight \
  tests/test_marlin_helpers.py::test_marlin_dequantize_uint4_zp_bias_matches_quantize_helper_output
```

结果：`8 passed`。

dense u4 正确性：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_8_row_bucket_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_size_m_24_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_small_tile_matches_reference
```

结果：`14 passed`。

接口错误覆盖：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_requires_bias \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_rejects_packed_zero_points \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_rejects_bias_without_flag \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_rejects_zp_bias_metadata
```

结果：`4 passed`。

u4b8 smoke：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_marlin_helpers.py::test_repack_layout_matches_reference_for_supported_dense_quant_types \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_sm70_scale_zp_math_consistency_matches_reference
```

结果：`14 passed`。

MoE smoke：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_marlin_moe.py::test_fused_marlin_moe_uint4_zp_accuracy \
  tests/test_marlin_moe.py::test_moe_wna16_uint4_zp_stage1_accuracy
```

结果：失败，原因是当前 SM70 MoE build 已有能力门控：

```text
SM70 MoE build currently does not enable uint4 zero-point kernels;
use uint4b8 or uint8b128 weights on the GPU Marlin path.
```

因此这两个 `uint4_zp` MoE 测试在当前 SM70 build 上不能作为 passing smoke 使用。本轮 scale+bias 改动没有修改 MoE schema 或 MoE kernel。

## 资源

命令：

```bash
/usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage \
  python/marlin_v100/_C.abi3.so | c++filt | \
  rg -A8 -B2 "sm70_marlin_u4_gemm_kernel|sm70_marlin_u4b8_gemm_kernel"
```

Dense `u4`：

| Kernel | REG | STACK | LOCAL | Spill |
|---|---:|---:|---:|---:|
| `u4<128,false>` | 254 | 0 | 0 | 0 |
| `u4<64,false>` | 254 | 0 | 0 | 0 |
| `u4<32,false>` | 254 | 0 | 0 | 0 |
| `u4<-1,false>` | 252 | 0 | 0 | 0 |
| `u4<128,true>` | 250 | 0 | 0 | 0 |
| `u4<64,true>` | 250 | 0 | 0 | 0 |
| `u4<32,true>` | 250 | 0 | 0 | 0 |
| `u4<-1,true>` | 244 | 0 | 0 | 0 |

Dense `u4b8` smoke 对照：

| Kernel | REG | STACK | LOCAL | Spill |
|---|---:|---:|---:|---:|
| `u4b8<128,false>` | 244 | 0 | 0 | 0 |
| `u4b8<64,false>` | 244 | 0 | 0 | 0 |
| `u4b8<32,false>` | 244 | 0 | 0 | 0 |
| `u4b8<-1,false>` | 238 | 0 | 0 | 0 |
| `u4b8<128,true>` | 250 | 0 | 0 | 0 |
| `u4b8<64,true>` | 250 | 0 | 0 | 0 |
| `u4b8<32,true>` | 250 | 0 | 0 | 0 |
| `u4b8<-1,true>` | 238 | 0 | 0 | 0 |

## Benchmark

命令：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4 --group-sizes -1 32 64 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

日志：`benchmarks/results/20260514_121023_dense_quick.log`

形状：`5120x4096x4096`

| group_size | operator TFLOPs | kernel_like TFLOPs |
|---:|---:|---:|
| `-1` | 76.68 | 78.25 |
| `32` | 77.74 | 76.75 |
| `64` | 78.14 | 76.89 |
| `128` | 78.69 | 77.03 |

和 `docs/20260513_018_sm70_u4_paired_metadata_cache.md` 中记录的 paired packed-zero metadata 路径相比，full-tile `group_size=-1` 接近原结果，positive group 有小幅回退。主要原因是现在需要额外读取 fp16 bias 平面。收益是热路径明显更简单：没有 packed qzero 地址计算，没有 zero unpack，也没有运行时 `bias = -zero * scale` 计算。

## Metadata 空间代价

旧 dense u4 metadata：

- `scale`：每个 `(group, n)` 一个 fp16
- `zero_point`：每个 `(group, n)` 一个 packed 4-bit 值

新 dense u4 metadata：

- `scale`：每个 `(group, n)` 一个 fp16
- `zp_bias`：每个 `(group, n)` 一个 fp16

因此 metadata 占用增加，但运行时路径变成两个规则连续的 fp16 平面，并且所有 group 统一使用 `q * scale + bias` 数学。

## 后续方向

- 如果 positive group TFLOPs 再次成为优先目标，需要检查两个 fp16 metadata 平面加载在 SASS 中的形态，并考虑更激进地向量化 scale+bias prefetch。
- 当前名为 `uint4_zp` 的 MoE 测试在 SM70 上仍被 capability gate 拦截。后续如果要迁移 MoE，需要单独决定 MoE 是继续使用 packed zero-point metadata，还是引入 MoE 专用 scale+bias 布局。

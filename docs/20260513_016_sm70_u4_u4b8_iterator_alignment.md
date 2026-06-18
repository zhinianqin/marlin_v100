# 2026-05-13 SM70 u4/u4b8 IteratorB 对齐结果

## 目标

本轮以 `csrc/quantization/marlin/sm70_marlin_u4b8_gemm.cu` 的实现形态为
标准，整理 dense `uint4` zero-point 路径，降低 `u4` 和 `u4b8` 两个
SM70 int4 GEMM 文件之间的心智负担。

保留的明确差异：

- `u4b8`: `128x256x32 / 8 warp`
- `u4`: `128x128x32 / 4 warp`
- `u4` 仍需要读取 `b_zeros` 并缓存 `scale` 与 `bias = -zero * scale`
- `u4b8` 仍只做 bias-8 dequant 后乘 scale

公共 Torch op schema、Python wrapper 参数、qweight repack layout、scale/zero
metadata shape、MoE 路径均未修改。

## 代码改动

### u4b8

`csrc/quantization/marlin/sm70_marlin_u4b8_gemm.cu`:

- 删除 `kCacheScales`。
- 删除独立 `cache_single_group_scales()`。
- 在 `load()` 内直接展开 `GroupSize_ == -1` 的 scale cache。
- `GroupSize_ == 32/64/128` 继续 direct scale load，不引入 positive group cache。
- `load_full_tile()` / `load_residue_tile()` 只用 `if constexpr (kGroupSize != -1)`
  区分 direct scale 与 cached scale。

### u4

`csrc/quantization/marlin/sm70_marlin_u4_gemm.cu`:

- 增加 `kFullTile = FullTile_`，full/residue 分裂写法对齐 `u4b8`。
- 删除 `cached_group_` 和 `cached_group_ == group` runtime early-return。
- `GroupSize_ == -1` 在编译期固定 group 0，positive group 才计算
  `logical_k / GroupSize_`。
- qweight 状态从 `qweight_offsets_[kCount]` 收缩为：
  - `qweight_base_offset_`
  - `qweight_strided_offsets_[kStrided]`
- `operator++()` 只更新 `qweight_base_offset_`。
- 访问公式变为：

```text
qweight_offset(s, c) = qweight_base_offset_
                     + qweight_strided_offsets_[s]
                     + c
```

说明：`u4` 的 128N/4-warp CUTLASS thread map 不是 `u4b8` 的单 K-strided
iteration，因此不能做到纯 `base + c`；保留 per-`s` 相对 offset 是这个几何下
必要的最小差异。

- full-tile 路径用 `uint2` 一次读取两个 64-column qword。
- residue invalid fragment 显式写 0，避免留下未定义 B fragment。
- zero-point 功能保持不变：metadata cache 仍生成 `cached_scales_` 与
  `cached_bias_`，热路径继续使用 `__hfma2(q, scale, bias)`。

`u4` 中原先仅注释掉的 `cached_group_ == group` 分支已经正式移除。这个 runtime
if 在热路径中会让调度变差；本轮采用无条件刷新 metadata 的形态。

## 构建与资源

构建命令：

```bash
./build.sh
```

结果：通过。

资源复核命令：

```bash
/usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage \
  python/marlin_v100/_C.abi3.so | c++filt | \
  rg -A8 -B2 "sm70_marlin_u4b8_gemm_kernel|sm70_marlin_u4_gemm_kernel"
```

`u4b8` 资源：

| specialization | REG | STACK | LOCAL | spill |
| --- | ---: | ---: | ---: | ---: |
| `<128, false>` | 238 | 0 | 0 | 0 |
| `<64, false>` | 238 | 0 | 0 | 0 |
| `<32, false>` | 238 | 0 | 0 | 0 |
| `<-1, false>` | 242 | 0 | 0 | 0 |
| `<128, true>` | 252 | 0 | 0 | 0 |
| `<64, true>` | 252 | 0 | 0 | 0 |
| `<32, true>` | 252 | 0 | 0 | 0 |
| `<-1, true>` | 238 | 0 | 0 | 0 |

`u4` 资源：

| specialization | REG | STACK | LOCAL | spill |
| --- | ---: | ---: | ---: | ---: |
| `<128, false>` | 255 | 0 | 0 | 0 |
| `<64, false>` | 255 | 0 | 0 | 0 |
| `<32, false>` | 255 | 0 | 0 | 0 |
| `<-1, false>` | 255 | 0 | 0 | 0 |
| `<128, true>` | 254 | 0 | 0 | 0 |
| `<64, true>` | 254 | 0 | 0 | 0 |
| `<32, true>` | 254 | 0 | 0 | 0 |
| `<-1, true>` | 248 | 0 | 0 | 0 |

结论：

- 两个 dense SM70 int4 kernel family 均保持 `STACK=0`、`LOCAL=0`、无 spill。
- `u4` 在 4-warp 形态下仍接近寄存器上限，但没有回到 spill。
- `u4b8` 的 positive group 仍保持 direct scale load 的无 spill 形态。

## 验证

导入：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python -c \
  "import marlin_v100, marlin_v100._C, marlin_v100._moe_C; print('imports ok')"
```

结果：

```text
imports ok
```

pytest 收集：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest --collect-only -q
```

结果：

```text
226 tests collected in 1.45s
```

定向测试：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_calibration.py \
  tests/test_marlin_helpers.py::test_repack_layout_matches_reference_for_supported_dense_quant_types \
  tests/test_marlin_helpers.py::test_uint4b8_act_order_repack_layout_matches_reference \
  tests/test_marlin_helpers.py::test_uint4_repack_layout_matches_reference \
  tests/test_marlin_helpers.py::test_marlin_dequantize_uint4_zp_matches_quantize_helper_output \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_sm70_scale_zp_math_consistency_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_residue_n_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_residue_k_single_group_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_residue_k_rejects_multi_group_metadata \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_8_row_bucket_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_size_m_24_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_small_tile_matches_reference
```

结果：

```text
46 passed in 3.80s
```

## Benchmark

设备：

```text
Tesla V100-SXM2-32GB, sm70
```

### uint4b8

命令：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4b8 --group-sizes -1 32 64 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

日志：

```text
benchmarks/results/20260513_181638_dense_quick.log
```

结果：

| group_size | operator us | operator TFLOPs | kernel_like us | kernel_like TFLOPs |
| ---: | ---: | ---: | ---: | ---: |
| `-1` | 2123.26 | 80.91 | 2061.31 | 83.34 |
| `32` | 2097.15 | 81.92 | 2124.29 | 80.87 |
| `64` | 2198.53 | 78.14 | 2140.16 | 80.27 |
| `128` | 2093.57 | 82.06 | 2088.45 | 82.26 |

### uint4

命令：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4 --group-sizes -1 32 64 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

日志：

```text
benchmarks/results/20260513_181722_dense_quick.log
```

结果：

| group_size | operator us | operator TFLOPs | kernel_like us | kernel_like TFLOPs |
| ---: | ---: | ---: | ---: | ---: |
| `-1` | 2188.80 | 78.49 | 2171.90 | 79.10 |
| `32` | 2187.78 | 78.53 | 2189.31 | 78.47 |
| `64` | 2174.46 | 79.01 | 2181.12 | 78.77 |
| `128` | 2121.22 | 80.99 | 2152.45 | 79.82 |

## 结论

- 本轮结构对齐可以保留：正确性通过，两个 kernel family 均无 stack/local/spill。
- `u4b8` 删除 `kCacheScales` 后，`GroupSize_ == -1` 的 cache 从 constructor
  移入 `load()`，full-tile `-1` 仍保持 `REG=238`，benchmark 没有回退。
- `u4` 删除 runtime `cached_group_` if 后，代码结构更接近 `u4b8`，并且当前
  quick benchmark 的 `uint4` positive groups 明显高于
  `docs/20260512_014_sm70_int4_qweight_iterator_restructure.md` 中记录的
  `68-70 TFLOPs` 区间。
- `u4` 的 128N/4-warp thread map 仍需要 `qweight_strided_offsets_[s]`；这是
  与 `u4b8` 的 256N/8-warp 几何差异，不建议强行消掉。
- 后续如果继续降低心智负担，优先抽出共享的 qweight macro-N offset helper；
  不建议把 positive group scale cache 重新引入热路径。
